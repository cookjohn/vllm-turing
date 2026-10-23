"""Prefill all-reduce / compute overlap for the tensor-parallel MLP (Qwen2MoeMLP: gate_up -> SiLU*mul -> down -> TP
all-reduce; used by Qwen3.5 / Qwen3-Next). On GPUs without peer access the prefill all-reduce is a large share of the
prefill time and runs serialized with the GEMMs. For batches of at least VLLM_TURING_OVERLAP_MIN tokens (default 1024)
the MLP runs on VLLM_TURING_OVERLAP_K token chunks (default 4): chunk i's partial down projection is all-reduced on a
side stream (through vLLM's TP all-reduce, so the compressed / host paths still apply) while chunk i+1's GEMMs run; only
the last chunk's all-reduce stays exposed. The MLP is per-token, so the result is the same as the unsplit MLP (same
GEMMs per row). Smaller batches (decode, CUDA-graph sizes) take the original path. The forward goes through the opaque
custom op vllm_turing::mlp_overlap so torch.compile does not trace the stream logic. VLLM_TURING_OVERLAP=1 enables."""
import os

import torch

ENABLED = os.environ.get("VLLM_TURING_OVERLAP", "0") == "1"
MIN_TOKENS = int(os.environ.get("VLLM_TURING_OVERLAP_MIN", "1024"))
K = int(os.environ.get("VLLM_TURING_OVERLAP_K", "4"))
_REG = []
_STREAMS = {}
_orig_forward = None

_frag = torch.library.Library("vllm_turing", "FRAGMENT")
_frag.define("mlp_overlap(Tensor x, int idx) -> Tensor")


def _side_stream(dev):
    s = _STREAMS.get(dev)
    if s is None:
        s = _STREAMS[dev] = torch.cuda.Stream(device=dev)
    return s


@torch.library.impl(_frag, "mlp_overlap", "CUDA")
def _mlp_overlap(x, idx):
    mlp = _REG[idx]
    n = x.shape[0]
    if n < MIN_TOKENS:
        return _orig_forward(mlp, x)
    from vllm.distributed import tensor_model_parallel_all_reduce
    cur = torch.cuda.current_stream()
    comm = _side_stream(x.device)
    k = K if n >= K * (MIN_TOKENS // 4) else 2
    rows = [(i * n) // k for i in range(k + 1)]
    reduced = []
    for i in range(k):
        xi = x[rows[i]:rows[i + 1]]
        gate_up, _ = mlp.gate_up_proj(xi)
        h = mlp.act_fn(gate_up)
        y = mlp.down_proj.quant_method.apply(mlp.down_proj, h, None)   # partial sum, no all-reduce
        ev = torch.cuda.Event()
        ev.record(cur)
        comm.wait_event(ev)
        y.record_stream(comm)
        with torch.cuda.stream(comm):
            r = tensor_model_parallel_all_reduce(y)
        r.record_stream(cur)
        reduced.append(r)
    cur.wait_stream(comm)
    return torch.cat(reduced, dim=0)


@torch.library.register_fake("vllm_turing::mlp_overlap")
def _mlp_overlap_fake(x, idx):
    return torch.empty_like(x)


def install():
    global _orig_forward
    if not ENABLED:
        return
    from vllm.model_executor.models import qwen2_moe
    M = qwen2_moe.Qwen2MoeMLP
    if getattr(M, "_vllm_turing_overlap", False):
        return
    orig_init, _orig_forward = M.__init__, M.forward

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self._turing_idx = len(_REG)
        _REG.append(self)

    def forward(self, x):
        dp = self.down_proj
        if self.expert_gate is not None or not getattr(dp, "reduce_results", False) or getattr(dp, "tp_size", 1) == 1:
            return _orig_forward(self, x)
        return torch.ops.vllm_turing.mlp_overlap(x, self._turing_idx)

    M.__init__, M.forward, M._vllm_turing_overlap = __init__, forward, True

    # The traced MLP forward differs from upstream's: give the AOT compile cache its own key (vLLM hashes the config
    # and re-checks only the source files of the stored trace, so a cached non-overlap graph would be reused silently).
    from vllm.compilation import caching
    if not getattr(caching, "_vllm_turing_overlap", False):
        orig_factors = caching.aot_compile_hash_factors

        def aot_compile_hash_factors(vllm_config):
            return [*orig_factors(vllm_config), f"vllm_turing_overlap k={K} min={MIN_TOKENS}"]

        caching.aot_compile_hash_factors = aot_compile_hash_factors
        caching._vllm_turing_overlap = True
