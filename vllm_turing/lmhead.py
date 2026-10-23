"""INT8 LM head: after loading, vLLM's ParallelLMHead weight (fp16, 1.27 GB per GPU at TP2 for Qwen3.8's 248k vocab)
is replaced by int8 rows with an fp32 scale per row (half the memory, more KV cache); logits come from the W8A16 GEMV
(csrc/w8a16_gemv.cu, fp16 activations) for up to 4 rows, the tensor-core W8A16 GEMM (csrc/w8a16_mma.cu) for 5..64
rows (speculative verify of one or several requests), the GEMV in 16-row passes up to GEMV_MAX_M and chunked
dequantization above. VLLM_TURING_LMHEAD=0 disables (VLLM_TURING_LMHEAD_MMA=0: GEMV only)."""
import ctypes
import logging
import os

import torch

from . import _build

logger = logging.getLogger("vllm.turing")   # under "vllm": uses vLLM's log handler
GEMV_MAX_M = 256   # rows through the GEMV in 16-row passes; more (prompt logprobs, warm-up) dequantize chunks instead
MMA = os.environ.get("VLLM_TURING_LMHEAD_MMA", "1") != "0"   # tensor-core W8A16 for 5..64 rows (_mma_cfg)
_lib = None


def lib():
    global _lib
    if _lib is None:
        _lib = _build.load("w8a16_gemv")
        _lib.w8a16_gemv.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
        _lib.w8a16_gemv.restype = ctypes.c_int
    return _lib


_mlib = None


def mma_lib():
    global _mlib
    if _mlib is None:
        _mlib = _build.load("w8a16_mma")
        _mlib.w8a16_mma.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
        _mlib.w8a16_mma.restype = ctypes.c_int
    return _mlib


def _mma_cfg(M, N, K):
    """Tensor-core W8A16 (csrc/w8a16_mma.cu) tiling for M rows, or None (GEMV). Full 124160 x 5120 head, 10-23:
    M = 5 / 16 / 30 rows 1.22 / 1.25 / 2.0 ms vs the GEMV's 1.31 / 2.79 / 5.47 (16-row passes); M <= 4: GEMV."""
    if not MMA or N % 2 != 0:
        return None
    if 5 <= M <= 32 and K % 512 == 0:
        return 2          # 1 n-tile per warp, 512-byte runs per weight row
    if 33 <= M <= 64 and K % 256 == 0:
        return 1
    return None


@torch.no_grad()
def quantize_rows(w: torch.Tensor, chunk: int = 16384):
    """[N, K] float -> int8 [N, K] and fp32 per-row scales [N] (symmetric, amax / 127)."""
    q = torch.empty(w.shape, dtype=torch.int8, device=w.device)
    s = torch.empty(w.shape[0], dtype=torch.float32, device=w.device)
    for i in range(0, w.shape[0], chunk):
        wf = w[i:i + chunk].float()
        sc = wf.abs().amax(dim=1).clamp(min=1e-12) / 127.0
        q[i:i + chunk] = torch.round(wf / sc[:, None]).clamp(-128, 127).to(torch.int8)
        s[i:i + chunk] = sc
    return q, s


_frag = torch.library.Library("vllm_turing", "FRAGMENT")
_frag.define("w8a16_mm(Tensor x, Tensor w, Tensor s, Tensor? bias) -> Tensor")


@torch.library.impl(_frag, "w8a16_mm", "CUDA")
def _w8a16_mm(x, w, s, bias):
    M, K = x.shape
    N = w.shape[0]
    out = torch.empty(M, N, dtype=x.dtype, device=x.device)
    cfg = _mma_cfg(M, N, K) if x.dtype == torch.float16 and x.is_contiguous() else None
    if cfg is not None:
        rc = mma_lib().w8a16_mma(x.data_ptr(), w.data_ptr(), s.data_ptr(), bias.data_ptr() if bias is not None else None,
                                 out.data_ptr(), M, N, K, cfg, torch.cuda.current_stream().cuda_stream)
        if rc != 0:
            raise RuntimeError(f"w8a16_mma launch failed: {rc}")
        return out
    if 1 <= M <= GEMV_MAX_M and K % 16 == 0 and x.is_contiguous() and x.dtype in (torch.float16, torch.bfloat16):
        # the GEMV takes up to 16 rows per pass, each pass reads the head once (1.1 ms for 124160 x 5120): with 6
        # concurrent requests the MTP verify has 30 rows, where the dequantizing fallback below took ~27 ms per step
        for r0 in range(0, M, 16):
            r1 = min(M, r0 + 16)
            rc = lib().w8a16_gemv(x[r0:r1].data_ptr(), w.data_ptr(), s.data_ptr(),
                                  bias.data_ptr() if bias is not None else None, out[r0:r1].data_ptr(), r1 - r0, N, K,
                                  0 if x.dtype == torch.float16 else 1, torch.cuda.current_stream().cuda_stream)
            if rc != 0:
                raise RuntimeError(f"w8a16_gemv launch failed: {rc}")
        return out
    c = 4096   # larger M (prompt logprobs, startup warm-up): dequantize 4096 rows (80 MB fp32 temp) at a time
    for j in range(0, N, c):
        wd = (w[j:j + c].float() * s[j:j + c, None]).to(x.dtype)
        out[:, j:j + c] = x @ wd.t()
    if bias is not None:
        out += bias
    return out


@torch.library.register_fake("vllm_turing::w8a16_mm")
def _w8a16_mm_fake(x, w, s, bias):
    return x.new_empty(x.shape[0], w.shape[0])


def install():
    from vllm.platforms import current_platform
    cap = current_platform.get_device_capability()
    if cap is None or cap.to_int() != 75 or os.environ.get("VLLM_TURING_LMHEAD", "1") == "0":
        return
    if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "LMHEAD_OFF")):  # off switch
        return
    from vllm.model_executor.layers import vocab_parallel_embedding as vpe
    M = vpe.UnquantizedEmbeddingMethod
    if getattr(M, "_vllm_turing_lmhead", False):
        return
    orig_pw, orig_apply = M.process_weights_after_loading, M.apply

    def process_weights_after_loading(self, layer):
        orig_pw(self, layer)
        if isinstance(layer, vpe.ParallelLMHead) and not hasattr(layer, "turing_w8") and layer.weight.dim() == 2 \
                and layer.weight.is_cuda and layer.weight.shape[1] % 16 == 0:
            w = layer.weight.data
            layer.turing_w8, layer.turing_s = quantize_rows(w)
            K = w.shape[1]
            layer.weight.data = torch.empty(0, K, dtype=w.dtype, device=w.device)   # free the fp16 head
            del w
            torch.cuda.empty_cache()
            logger.info("vllm_turing: int8 LM head, %d x %d per rank", layer.turing_w8.shape[0], K)

    def apply(self, layer, x, bias=None):
        if hasattr(layer, "turing_w8"):
            shp = x.shape
            y = torch.ops.vllm_turing.w8a16_mm(x.reshape(-1, shp[-1]).contiguous(), layer.turing_w8, layer.turing_s, bias)
            return y.view(*shp[:-1], y.shape[-1])
        return orig_apply(self, layer, x, bias)

    M.process_weights_after_loading = process_weights_after_loading
    M.apply = apply
    M._vllm_turing_lmhead = True
