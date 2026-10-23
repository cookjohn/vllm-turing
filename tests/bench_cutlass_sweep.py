#!/usr/bin/env python3
"""Builds cutlass_sweep.cu against the CUTLASS headers shipped in vllm/third_party and times every tile config on the
TP2 prefill GEMM shapes (M rows of a prefill chunk x per-rank N x K), against vLLM's cutlass_scaled_mm (config 0 +
scaling epilogue) and cuBLAS int8 (torch._int_mm, int32 output). Checks each config's output against cutlass_scaled_mm
with unit scales. usage: bench_cutlass_sweep.py [device] [M]   (vLLM env python)"""
import ctypes
import os
import subprocess
import sys

import torch
import vllm
import vllm._custom_ops  # noqa: F401

torch.cuda.set_device(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
M = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
HERE = os.path.dirname(os.path.abspath(__file__))
INC = os.path.join(os.path.dirname(vllm.__file__), "third_party", "fmha_sm100", "cutlass", "include")
SRC, SO = os.path.join(HERE, "cutlass_sweep.cu"), "/tmp/libcutlass_sweep.so"
# the system toolkit's nvcc: the pip nvcc's bundled CCCL headers reject CUTLASS's <cuda/std/...> includes
NVCC = os.environ.get("NVCC", "/usr/local/cuda/bin/nvcc")
if not os.path.exists(SO) or os.path.getmtime(SO) < os.path.getmtime(SRC):
    subprocess.run([NVCC, "-O3", "-std=c++17", "-gencode=arch=compute_75,code=sm_75", "-shared", "-Xcompiler",
                    "-fPIC", "--expt-relaxed-constexpr", f"-I{INC}", "-o", SO, SRC], check=True)
lib = ctypes.CDLL(SO)
lib.cutlass_gemm_cfg.argtypes = [ctypes.c_int] + [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
lib.cutlass_gemm_cfg.restype = ctypes.c_int
NAMES = {0: "128x128x64 w64x64 (vLLM)", 1: "256x128x64 w64x64", 2: "128x256x64 w64x64", 3: "256x128x32 w64x64",
         4: "128x128x128 w64x64", 5: "64x128x64 w32x64", 6: "128x64x64 w64x32", 7: "256x128x64 sw4",
         8: "256x128x64 sw8", 9: "128x128x64 sw8", 10: "256x128x64 w128x64", 11: "128x128x64 w32x64",
         12: "256x64x64 w64x64"}
SHAPES = {"mlp gate_up": (8704, 5120), "mlp down": (5120, 8704), "gdn in_proj": (8192, 5120),
          "gdn out_proj": (5120, 3072), "attn qkv": (3584, 5120)}
PEAK = 68 * 2048 * 1845e6


def bench(fn, it=10, rep=5):
    fn()
    best = 1e9
    for _ in range(rep):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(it):
            fn()
        e1.record()
        torch.cuda.synchronize()
        best = min(best, e0.elapsed_time(e1) / it)
    return best


tot = {}
for name, (N, K) in SHAPES.items():
    a = torch.randint(-128, 128, (M, K), dtype=torch.int8, device="cuda")
    w = torch.randint(-128, 128, (N, K), dtype=torch.int8, device="cuda")
    one_a, one_w = torch.ones(M, 1, device="cuda"), torch.ones(N, 1, device="cuda")
    ref = torch.empty(M, N, dtype=torch.float16, device="cuda")
    out = torch.empty_like(ref)
    torch.ops._C.cutlass_scaled_mm(ref, a, w.t(), one_a, one_w, None)
    ops = 2.0 * M * N * K
    rows = []
    tv = bench(lambda: torch.ops._C.cutlass_scaled_mm(ref, a, w.t(), one_a, one_w, None))
    tot.setdefault("vllm cutlass_scaled_mm", 0)
    tot["vllm cutlass_scaled_mm"] += tv
    rows.append(f"vllm cutlass_scaled_mm {tv:6.2f} ms {100 * ops / (tv * 1e-3) / PEAK:4.0f}%")
    ti = bench(lambda: torch._int_mm(a, w.t()))
    tot.setdefault("cuBLAS _int_mm (int32 out)", 0)
    tot["cuBLAS _int_mm (int32 out)"] += ti
    rows.append(f"cuBLAS _int_mm (int32 out) {ti:6.2f} ms {100 * ops / (ti * 1e-3) / PEAK:4.0f}%")
    st = torch.cuda.current_stream().cuda_stream
    for cfg, cname in NAMES.items():
        rc = lib.cutlass_gemm_cfg(cfg, a.data_ptr(), w.data_ptr(), out.data_ptr(), M, N, K, st)
        torch.cuda.synchronize()
        if rc != 0:
            rows.append(f"cfg {cfg:2d} {cname:26s} not runnable (rc {rc})")
            continue
        exact = torch.equal(out, ref)
        t = bench(lambda: lib.cutlass_gemm_cfg(cfg, a.data_ptr(), w.data_ptr(), out.data_ptr(), M, N, K, st))
        tot.setdefault(cname, 0)
        tot[cname] += t
        rows.append(f"cfg {cfg:2d} {cname:26s} {t:6.2f} ms {100 * ops / (t * 1e-3) / PEAK:4.0f}%"
                    + ("" if exact else "  OUTPUT DIFFERS"))
    print(f"{name} M={M} N={N} K={K}\n    " + "\n    ".join(rows), flush=True)
    del a, w, ref, out
print(f"sum over the shapes (one of each, M={M}):")
for k, v in sorted(tot.items(), key=lambda x: x[1]):
    print(f"  {k:28s} {v:7.2f} ms")
