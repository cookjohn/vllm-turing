"""Small-M INT8 W8A8 GEMM (csrc/w8a8_gemv.cu) behind vLLM's cutlass_scaled_mm.

vLLM's W8A8 int8 linear layers call vllm._custom_ops.cutlass_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias), which
runs torch.ops._C.cutlass_scaled_mm. The wrapper is replaced by one that runs the custom op vllm_turing::scaled_mm_
instead: same arguments, dispatching at run time (also inside torch.compile / CUDA graph capture, where the batch size
is concrete) to the GEMV kernel for M <= VLLM_TURING_GEMV_MAX_M (default 12) rows, to the int8 tensor-core kernel up to
VLLM_TURING_MMA_MAX_M (default and maximum 32: MTP4 verify of up to 6 concurrent requests; CUTLASS pads them to 128-row
tiles, ~1.35x slower), and larger M (prefill) to a CUTLASS kernel at the tile configuration that measured fastest on the
2080 Ti (csrc/w8a8_big.cu, 256 x 128 x 64 raster-swizzled: ~10% over vLLM's 128 x 128 x 64; VLLM_TURING_BIG=0 keeps
vLLM's), else to the CUTLASS op.
The kernel is built with the CUDA 13 nvcc of the env (nvidia/cu13, matches torch) into a plain shared library loaded
with ctypes (no torch C++ ABI involved)."""
import ctypes
import os

import torch

from . import _build

MAX_M = int(os.environ.get("VLLM_TURING_GEMV_MAX_M", "12"))     # dp4a GEMV up to here (faster than the MMA kernel)
MMA_MAX_M = min(int(os.environ.get("VLLM_TURING_MMA_MAX_M", "32")), 32)
BIG = os.environ.get("VLLM_TURING_BIG", "1") != "0"
_big = None
_lib = None


def lib():
    global _lib
    if _lib is None:
        _lib = _build.load("w8a8_gemv")
        _lib.w8a8_gemv.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 6 + [ctypes.c_void_p]
        _lib.w8a8_gemv.restype = ctypes.c_int
        _lib.w8a8_mma.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 6 + [ctypes.c_void_p]
        _lib.w8a8_mma.restype = ctypes.c_int
    return _lib


def gemv_ok(out, a, b, sa, sb, bias, max_m=None):
    """The GEMV path's layout requirements (else CUTLASS); the tensor-core path has the same ones."""
    M, K = a.shape
    return (1 <= M <= (MAX_M if max_m is None else max_m) and K % 16 == 0 and a.dtype == torch.int8 and a.is_contiguous()
            and b.dtype == torch.int8 and b.stride(0) == 1 and b.stride(1) == K
            and out.is_contiguous() and out.dtype in (torch.float16, torch.bfloat16)
            and sa.dtype == torch.float32 and sb.dtype == torch.float32
            and sa.numel() in (1, M) and sb.numel() in (1, b.shape[1])
            and a.data_ptr() % 16 == 0 and b.data_ptr() % 16 == 0)


def biglib():
    global _big
    if _big is None:
        _big = _build.load("w8a8_big", cutlass=True)
        _big.w8a8_big.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
        _big.w8a8_big.restype = ctypes.c_int
    return _big


def big_ok(out, a, b, sa, sb, bias):
    """The prefill kernel's requirements: per-row activation and per-column weight scales, no bias, 16-aligned shapes."""
    M, K = a.shape
    N = b.shape[1]
    return (BIG and M > MMA_MAX_M and bias is None and K % 16 == 0 and N % 16 == 0 and a.dtype == torch.int8
            and a.is_contiguous() and b.dtype == torch.int8 and b.stride(0) == 1 and b.stride(1) == K
            and out.is_contiguous() and out.dtype in (torch.float16, torch.bfloat16)
            and sa.dtype == torch.float32 and sb.dtype == torch.float32 and sa.numel() == M and sb.numel() == N
            and sa.is_contiguous() and sb.is_contiguous() and a.data_ptr() % 16 == 0 and b.data_ptr() % 16 == 0)


def big(out, a, b, sa, sb):
    M, K = a.shape
    N = b.shape[1]
    rc = biglib().w8a8_big(a.data_ptr(), b.data_ptr(), sa.data_ptr(), sb.data_ptr(), out.data_ptr(), M, N, K,
                           0 if out.dtype == torch.float16 else 1, torch.cuda.current_stream().cuda_stream)
    if rc != 0:
        raise RuntimeError(f"w8a8_big failed: {rc}")


def mma_ok(out, a, b, sa, sb, bias):
    return a.shape[0] > MAX_M and gemv_ok(out, a, b, sa, sb, bias, MMA_MAX_M)


def mma(out, a, b, sa, sb, bias):
    M, K = a.shape
    N = b.shape[1]
    rc = lib().w8a8_mma(a.data_ptr(), b.data_ptr(), sa.data_ptr(), sb.data_ptr(),
                        bias.data_ptr() if bias is not None else None, out.data_ptr(),
                        M, N, K, int(sa.numel() > 1), int(sb.numel() > 1),
                        0 if out.dtype == torch.float16 else 1, torch.cuda.current_stream().cuda_stream)
    if rc != 0:
        raise RuntimeError(f"w8a8_mma launch failed: {rc}")


def gemv(out, a, b, sa, sb, bias):
    M, K = a.shape
    N = b.shape[1]
    rc = lib().w8a8_gemv(a.data_ptr(), b.data_ptr(), sa.data_ptr(), sb.data_ptr(),
                         bias.data_ptr() if bias is not None else None, out.data_ptr(),
                         M, N, K, int(sa.numel() > 1), int(sb.numel() > 1),
                         0 if out.dtype == torch.float16 else 1, torch.cuda.current_stream().cuda_stream)
    if rc != 0:
        raise RuntimeError(f"w8a8_gemv launch failed: {rc}")


_lib_def = torch.library.Library("vllm_turing", "DEF")
_lib_def.define("scaled_mm_(Tensor(a!) out, Tensor a, Tensor b, Tensor a_scales, Tensor b_scales, Tensor? bias) -> ()")


@torch.library.impl(_lib_def, "scaled_mm_", "CUDA")
def _scaled_mm_cuda(out, a, b, a_scales, b_scales, bias):
    if gemv_ok(out, a, b, a_scales, b_scales, bias):
        gemv(out, a, b, a_scales, b_scales, bias)
    elif mma_ok(out, a, b, a_scales, b_scales, bias):
        mma(out, a, b, a_scales, b_scales, bias)
    elif big_ok(out, a, b, a_scales, b_scales, bias):
        big(out, a, b, a_scales, b_scales)
    else:
        torch.ops._C.cutlass_scaled_mm(out, a, b, a_scales, b_scales, bias)


@torch.library.register_fake("vllm_turing::scaled_mm_")
def _scaled_mm_fake(out, a, b, a_scales, b_scales, bias):
    return None


def install():
    """Replace vllm._custom_ops.cutlass_scaled_mm (same behaviour, CUTLASS call -> vllm_turing::scaled_mm_)."""
    from vllm import _custom_ops as ops
    from vllm.platforms import current_platform

    if getattr(ops.cutlass_scaled_mm, "_vllm_turing", False):
        return
    cap = current_platform.get_device_capability()
    if cap is None or cap.to_int() != 75:
        return

    def cutlass_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias=None):
        assert out_dtype is torch.bfloat16 or out_dtype is torch.float16
        assert bias is None or bias.numel() == b.shape[1] and bias.dtype == out_dtype
        target_shape = (*a.shape[:-1], b.shape[1])
        a = a.view(-1, a.shape[-1])
        if not (b.shape[0] % 16 == 0 and b.shape[1] % 16 == 0):
            from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import triton_scaled_mm
            out = triton_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)
        else:
            out = torch.empty((a.shape[0], b.shape[1]), dtype=out_dtype, device=a.device)
            torch.ops.vllm_turing.scaled_mm_(out, a, b, scale_a, scale_b, bias)
        return out.view(*target_shape)

    cutlass_scaled_mm._vllm_turing = True
    ops.cutlass_scaled_mm = cutlass_scaled_mm
