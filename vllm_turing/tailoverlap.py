"""Prefill all-reduce / compute overlap for the tensor-parallel decoder layers of Qwen3.5 / Qwen3-Next (vLLM 0.31).
Each layer's two all-reduces (attention / GDN out_proj, MLP down_proj) otherwise run serialized with the GEMMs: 13.5%
of a 32K-token prefill over NVLink with NCCL (26% over PCIe with the host all-reduce). Everything after the attention
core is per-token (out_proj -> all-reduce -> residual add + norm ->
gate_up -> SiLU*mul -> down_proj -> all-reduce), so for batches of at least MIN tokens that tail runs on K token chunks:
chunk i's all-reduces go on a side stream while chunk i + 1's GEMMs run on the compute stream, and only the last
chunk's MLP all-reduce stays exposed (~85% of the wait hidden). The arithmetic per row is that of the unsplit layer.
The decoder layer forward is replaced: the input norm and the attention core stay in the compiled graph (the out_proj
of the attention / GDN module becomes an identity, its GEMM moves into the tail); the tail is the opaque custom op
vllm_turing::layer_tail (eager, so torch.compile does not trace the streams). Smaller batches (decode, CUDA graphs) take
the unchunked eager tail. The post-attention norm + residual add + int8 quantization of the MLP input run as one fused
kernel (rms_norm_dynamic_per_token_quant, the Gemma-style 1 + w norm weight precomputed) when the MLP's int8 GEMM takes
dynamic symmetric activations (W8A8); otherwise the model's norm (a torch.compile'd function returning a new residual
tensor) and the GEMM's own quantization.
The chunk all-reduces take the copy-engine all-reduce (cear.py, GPUs with peer access) straight into the output slice:
NCCL's kernels hold SMs for the whole transfer, which slowed the concurrent GEMMs 12% and doubled NCCL's own time
(the tail was 12% slower than no overlap over NVLink). The MLP activation runs as one SiLU * up + int8 quantization
kernel (csrc/silu_quant.cu) for the down projection when it is a W8A8 int8 GEMM (the eager tail misses inductor's
fusion of the two).
Chunk bounds at the fractions VLLM_TURING_TAIL_SPLIT of the batch (default 1/6, 1/2, 5/6: a 6720-token step in
1120 / 2240 / 2240 / 1120 rows): small first and last chunks expose little of the first all-reduce and the last one,
and these row counts run the tail GEMMs at the full-batch rate (wave quantization of the 256-row tiles: per token 1120
and 2240 rows are as fast as 6720, 1680 and 3360 2.5-5% slower); batches under 4096 tokens: two halves.
VLLM_TURING_TAIL=1 enables; VLLM_TURING_TAIL_SPLIT fractions, VLLM_TURING_TAIL_MIN tokens (default 2048),
VLLM_TURING_TAIL_FUSED=0 keeps the separate norm / activation and quantization kernels."""
import ctypes
import logging
import os
import types

import torch

from . import _build

logger = logging.getLogger("vllm.turing")
ENABLED = os.environ.get("VLLM_TURING_TAIL", "0") == "1"
SPLIT = sorted(float(f) for f in os.environ.get("VLLM_TURING_TAIL_SPLIT", "0.16667,0.5,0.83333").split(",") if f)
SPLIT = tuple(f for f in SPLIT if 0.0 < f < 1.0) or (0.5,)     # a tuple: logger.info_once caches its arguments
MAX_PART = max(b - a for a, b in zip((0.0, *SPLIT), (*SPLIT, 1.0)))   # the largest chunk as a fraction of the batch
MIN_TOKENS = int(os.environ.get("VLLM_TURING_TAIL_MIN", "2048"))
FUSED = os.environ.get("VLLM_TURING_TAIL_FUSED", "1") != "0"
_REG = []
_STREAMS = {}
_CE = []          # [the TP group's copy-engine all-reduce or None], looked up at the first chunked call
_actq = None


def _actq_lib():
    global _actq
    if _actq is None:
        _actq = _build.load("silu_quant")
        _actq.silu_mul_quant.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 2 + [ctypes.c_void_p]
        _actq.silu_mul_quant.restype = ctypes.c_int
    return _actq


def silu_mul_quant(gate_up: torch.Tensor):
    """(M, 2I) fp16 -> int8 (M, I) of SiLU(gate) * up and its per-row scales (M, 1) float32."""
    M, I2 = gate_up.shape
    q = torch.empty(M, I2 // 2, dtype=torch.int8, device=gate_up.device)
    sc = torch.empty(M, 1, dtype=torch.float32, device=gate_up.device)
    rc = _actq_lib().silu_mul_quant(gate_up.data_ptr(), q.data_ptr(), sc.data_ptr(), M, I2 // 2,
                                    torch.cuda.current_stream().cuda_stream)
    if rc != 0:
        raise RuntimeError(f"silu_mul_quant failed: {rc}")
    return q, sc


def _ce():
    if not _CE:
        ce = None
        try:
            from vllm.distributed import get_tp_group
            ce = getattr(get_tp_group().device_communicator, "turing_ce", None)
        except Exception:   # noqa: BLE001
            ce = None
        _CE.append(ce)
    return _CE[0]

_frag = torch.library.Library("vllm_turing", "FRAGMENT")
_frag.define("layer_tail(Tensor attn, Tensor residual, int idx) -> (Tensor, Tensor)")


def _side_stream(dev):
    s = _STREAMS.get(dev)
    if s is None:
        # high priority: the block scheduler hands SMs freed by the GEMM's blocks to the all-reduce kernel first
        s = _STREAMS[dev] = torch.cuda.Stream(device=dev, priority=-1)
    return s


def _proj(linear, x):
    """The GEMM of a RowParallelLinear without its all-reduce (this rank's partial sum)."""
    bias = None if (linear.tp_rank > 0 or linear.skip_bias_add) else linear.bias
    return linear.quant_method.apply(linear, x, bias)


class _Tail:
    """The per-token tail of one decoder layer."""

    def __init__(self, layer):
        attn = layer.linear_attn if layer.layer_type == "linear_attention" else layer.self_attn
        self.out_proj = attn.out_proj if layer.layer_type == "linear_attention" else attn.o_proj
        self.norm = layer.post_attention_layernorm
        self.mlp = layer.mlp
        self.fused = None               # decided at the first call: the weights are loaded and processed by then
        self.actq = False               # SiLU * up + int8 quantization in one kernel for the down projection

    def _decide(self):
        from vllm import _custom_ops as ops
        gu = self.mlp.gate_up_proj
        kernel = getattr(gu.quant_method, "kernel", None) or getattr(getattr(gu, "scheme", None), "kernel", None)
        self.fused = False
        norm_type = type(self.norm).__name__
        if FUSED and kernel is not None and hasattr(kernel, "_get_layer_params") and \
                norm_type in ("GemmaRMSNorm", "RMSNorm") and getattr(self.norm, "variance_size_override", None) is None \
                and hasattr(ops, "rms_norm_dynamic_per_token_quant"):
            try:
                w_q, w_s, i_s, i_zp, azp_adj = kernel._get_layer_params(gu)
                if i_s is None and i_zp is None and azp_adj is None and gu.bias is None and w_q.dtype == torch.int8:
                    w = self.norm.weight.data
                    # GemmaRMSNorm scales by 1 + w (in float32); the fused kernel takes the weight as it is
                    self.w1 = (1.0 + w.float()).to(w.dtype) if norm_type == "GemmaRMSNorm" else w
                    self.w_q, self.w_s = w_q, w_s
                    self.fused = True
                    dp = self.mlp.down_proj
                    dk = getattr(dp.quant_method, "kernel", None) or getattr(getattr(dp, "scheme", None), "kernel", None)
                    if type(self.mlp.act_fn).__name__ == "SiluAndMul" and dk is not None and dp.bias is None:
                        wd_q, wd_s, di_s, di_zp, dazp = dk._get_layer_params(dp)
                        if di_s is None and di_zp is None and dazp is None and wd_q.dtype == torch.int8:
                            self.wd_q, self.wd_s, self.actq = wd_q, wd_s, True
            except Exception as e:   # noqa: BLE001
                logger.info("vllm_turing tail: no fused norm + quant (%s)", e)
        logger.info_once("vllm_turing: layer tail overlap on (chunk bounds %s, min %d tokens, fused norm+quant: %s, "
                         "fused activation+quant: %s, copy-engine all-reduce: %s)", SPLIT, MIN_TOKENS, self.fused,
                         self.actq, _ce() is not None)

    def mlp_partial(self, h, res):
        """res (a slice of the layer's residual) += h; MLP of the normed sum; -> this rank's partial down projection."""
        if self.fused is None:
            self._decide()
        mlp = self.mlp
        if self.fused:
            from vllm import _custom_ops as ops
            x_q, x_s = ops.rms_norm_dynamic_per_token_quant(h, self.w1, self.norm.variance_epsilon, torch.int8, None,
                                                            res)      # res += h in place; x_q = int8 norm(res)
            gate_up = ops.cutlass_scaled_mm(x_q, self.w_q, scale_a=x_s, scale_b=self.w_s, out_dtype=h.dtype, bias=None)
            if self.actq:
                a_q, a_s = silu_mul_quant(gate_up)
                return ops.cutlass_scaled_mm(a_q, self.wd_q, scale_a=a_s, scale_b=self.wd_s, out_dtype=h.dtype,
                                             bias=None)
        else:
            x, r = self.norm(h, res)
            if r.data_ptr() != res.data_ptr():
                res.copy_(r)
            gate_up, _ = mlp.gate_up_proj(x)
        return _proj(mlp.down_proj, mlp.act_fn(gate_up))

    def plain(self, attn, residual):
        from vllm.distributed import tensor_model_parallel_all_reduce as ar
        res = residual.clone()
        h = ar(_proj(self.out_proj, attn))
        return ar(self.mlp_partial(h, res)), res

    def chunked(self, attn, residual):
        from vllm.distributed import tensor_model_parallel_all_reduce as ar
        n = attn.shape[0]
        fr = SPLIT if n >= 4096 else (0.5,)
        bounds = [0] + [int(f * n) // 16 * 16 for f in fr] + [n]
        cur = torch.cuda.current_stream(attn.device)
        comm = _side_stream(attn.device)
        # Every buffer is allocated on the compute stream, and no record_stream: a block freed while the side stream
        # still uses it counts as free in the allocator's statistics (vLLM's memory profile) but stays unusable until
        # the side stream passes it, so the steady state needed more memory than the profile reserved (OOM retries,
        # each freeing the allocator cache behind a device sync: 14% of a prefill idle). The tensors the side stream
        # reads or writes are held until the compute stream has waited for it; freed after that, they are safe.
        res = residual.clone()
        mid = torch.empty_like(residual)        # the all-reduced attention output
        out = torch.empty_like(residual)        # the all-reduced MLP output
        held = []
        ce = _ce()

        def reduce_async(y, dst):
            """y (computed on the compute stream) all-reduced on the side stream into dst; -> event after it."""
            ev = torch.cuda.Event()
            ev.record(cur)
            comm.wait_event(ev)
            held.append(y)
            with torch.cuda.stream(comm):
                if ce is not None and ce.usable(y):
                    ce.all_reduce(y, dst)            # no allocation on the side stream's pool
                else:
                    dst.copy_(ar(y))
                done = torch.cuda.Event()
                done.record(comm)
            return done

        first = [reduce_async(_proj(self.out_proj, attn[a:b]), mid[a:b]) for a, b in zip(bounds, bounds[1:])]
        for (a, b), done in zip(zip(bounds, bounds[1:]), first):
            cur.wait_event(done)
            reduce_async(self.mlp_partial(mid[a:b], res[a:b]), out[a:b])
        cur.wait_stream(comm)
        held.clear()                            # after the wait: later compute-stream allocations follow the side stream
        return out, res


@torch.library.impl(_frag, "layer_tail", "CUDA")
def _layer_tail(attn, residual, idx):
    t = _REG[idx]
    if attn.shape[0] < MIN_TOKENS or torch.cuda.is_current_stream_capturing():
        return t.plain(attn, residual)
    return t.chunked(attn, residual)


@torch.library.register_fake("vllm_turing::layer_tail")
def _layer_tail_fake(attn, residual, idx):
    return torch.empty_like(residual), torch.empty_like(residual)


def _register(self):
    """After a decoder layer's __init__: its tail object, if the layer qualifies."""
    self._turing_tail = -1
    mlp = getattr(self, "mlp", None)
    if (getattr(self, "use_attn_reduce_scatter_for_moe", False) or getattr(self, "layer_scale", False)
            or mlp is None or not hasattr(mlp, "gate_up_proj") or getattr(mlp, "expert_gate", None) is not None
            or self.layer_type not in ("linear_attention", "full_attention")):
        return
    t = _Tail(self)
    if t.out_proj.tp_size > 1 and t.out_proj.reduce_results and mlp.down_proj.reduce_results:
        self._turing_tail = len(_REG)
        _REG.append(t)
        t.out_proj.forward = types.MethodType(lambda m, x: (x, None), t.out_proj)   # its GEMM runs in the tail


def _patch_layer_class(L):
    """Wrap the class's own __init__ (a subclass may define its own without calling the base one) and replace its own
    forward (a subclass without one inherits the patched base forward)."""
    if "_vllm_turing_tail" in L.__dict__:        # not getattr: a subclass inherits the base class's mark
        return
    L._vllm_turing_tail = True
    if "__init__" in L.__dict__:
        orig_init = L.__init__

        def __init__(self, *args, **kwargs):
            orig_init(self, *args, **kwargs)
            _register(self)

        L.__init__ = __init__
    if "forward" not in L.__dict__:
        return
    orig_fwd = L.forward

    def forward(self, hidden_states, residual, positions, **kwargs):
        if getattr(self, "_turing_tail", -1) < 0:
            return orig_fwd(self, hidden_states, residual, positions, **kwargs)
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        if self.layer_type == "linear_attention":
            attn = self.linear_attn(hidden_states=hidden_states)
        else:
            attn = self.self_attn(hidden_states=hidden_states, positions=positions)
        return torch.ops.vllm_turing.layer_tail(attn, residual, self._turing_tail)

    L.forward = forward


def install():
    if not ENABLED:
        return
    from vllm.model_executor.models import qwen3_5, qwen3_next
    _patch_layer_class(qwen3_next.Qwen3NextDecoderLayer)
    _patch_layer_class(qwen3_5.Qwen3_5DecoderLayer)         # its own __init__ (does not call the base one)
    # The traced layer forward differs from upstream's: give the AOT compile cache its own key (vLLM hashes the config
    # and re-checks only the source files of the stored trace: a cached graph without the tail op would be reused).
    from vllm.compilation import caching
    if not getattr(caching, "_vllm_turing_tail", False):
        orig_factors = caching.aot_compile_hash_factors

        def aot_compile_hash_factors(vllm_config):
            return [*orig_factors(vllm_config), "vllm_turing_tail"]      # the knobs act inside the eager op

        caching.aot_compile_hash_factors = aot_compile_hash_factors
        caching._vllm_turing_tail = True
