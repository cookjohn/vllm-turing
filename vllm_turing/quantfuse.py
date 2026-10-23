"""Dynamic per-token int8 quantization of W8A8 activations fused into the preceding compiled kernel (vLLM 0.31).
vLLM's int8 linear layers quantize their input with ops.scaled_int8_quant, an opaque CUDA op that torch.compile cannot
fuse: every int8 GEMM was preceded by a standalone kernel re-reading the fp16 activations the previous fused kernel
(norm + residual, SiLU * mul, ...) had just written (prefill: 1632 kernels, 2.5% of a 32K prefill; decode: 274 kernels
per step, 1.8%). While torch.compile traces (the compiled pieces of the model and the CUDA-graph decode pieces), the
dynamic symmetric case is computed with plain torch ops instead, the formula of vLLM's kernel (float32 absmax per row,
scale = absmax / 127, round-half-even of x * (127 / absmax), clamped), which inductor folds into the producer's Triton
kernel; eager calls keep the CUDA op. Inductor drops the producer's fp16 rounding before the quantization (0.2-0.85%
of the int8 values off by one vs the CUDA op): PPL zh / code / en 6.5566 / 1.4882 / 2.4631 (6.5614 / 1.4880 / 2.4638
without); 32K prefill 2573 -> 2779 tok/s with the big-M GEMM (w8a8.py), decode +1-2%. VLLM_TURING_QUANTFUSE=0 disables."""
import os

import torch

ENABLED = os.environ.get("VLLM_TURING_QUANTFUSE", "1") != "0"


def quant_torch(x: torch.Tensor):
    """Per-row dynamic symmetric int8 quantization: (int8 like x, float32 scales [rows, 1])."""
    shape = x.shape
    xf = x.reshape(-1, shape[-1]).float()
    absmax = xf.abs().amax(dim=-1, keepdim=True)
    q = torch.round(xf * (127.0 / absmax)).clamp(-128.0, 127.0).to(torch.int8)
    return q.reshape(shape), absmax / 127.0


def install():
    if not ENABLED:
        return
    from vllm import _custom_ops as ops
    if getattr(ops.scaled_int8_quant, "_vllm_turing", False):
        return
    orig = ops.scaled_int8_quant

    def scaled_int8_quant(input, scale=None, azp=None, symmetric=True):
        if scale is None and azp is None and symmetric and torch.compiler.is_compiling():
            q, s = quant_torch(input)
            return q, s, None
        return orig(input, scale, azp, symmetric)

    scaled_int8_quant._vllm_turing = True
    scaled_int8_quant._orig = orig
    ops.scaled_int8_quant = scaled_int8_quant

    # the traced graphs differ from upstream's: their own AOT compile cache key (vLLM re-checks only the source files
    # of the stored trace, a cached graph with the opaque op would be reused silently)
    from vllm.compilation import caching
    if not getattr(caching, "_vllm_turing_quantfuse", False):
        orig_factors = caching.aot_compile_hash_factors

        def aot_compile_hash_factors(vllm_config):
            return [*orig_factors(vllm_config), "vllm_turing_quantfuse"]

        caching.aot_compile_hash_factors = aot_compile_hash_factors
        caching._vllm_turing_quantfuse = True
