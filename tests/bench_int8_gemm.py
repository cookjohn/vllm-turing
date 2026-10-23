#!/usr/bin/env python3
"""Prefill-sized INT8 W8A8 GEMMs per TP2 rank of Qwen3.8-27B (M = one aligned prefill chunk): vLLM's
cutlass_scaled_mm (per-token activation scale x per-channel weight scale -> fp16, the production path) vs
torch._int_mm (cuBLASLt IMMA, int32 out) + a separate scaling pass; achieved TOPS against the 2080 Ti's dense int8
peak (~215 TOPS at the boost clock).
usage: bench_int8_gemm.py [M]   (vLLM env python, one GPU)"""
import sys
import time

import torch

from vllm import _custom_ops as ops

M = int(sys.argv[1]) if len(sys.argv) > 1 else 6720
dev = "cuda"
SHAPES = {   # name: (K, N) per rank
    "mlp gate_up": (5120, 17408), "mlp down": (8704, 5120),
    "gdn in_proj_qkvz": (5120, 8192), "gdn out_proj": (3072, 5120),
    "attn qkv": (5120, 7168), "attn o_proj": (3072, 5120),
}


def timed(fn, it=10):
    fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(it):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / it * 1e3


torch.manual_seed(0)
for name, (K, N) in SHAPES.items():
    a = torch.randint(-127, 128, (M, K), dtype=torch.int8, device=dev)
    w = torch.randint(-127, 128, (N, K), dtype=torch.int8, device=dev)
    sa = torch.rand(M, 1, device=dev) * 0.01
    sw = torch.rand(N, 1, device=dev) * 0.01
    b = w.t()                                    # [K, N] column-major view, as vLLM stores the weight
    ms_c = timed(lambda: ops.cutlass_scaled_mm(a, b, sa, sw, torch.float16))
    try:
        ms_i = timed(lambda: torch._int_mm(a, b))
        ms_s = timed(lambda: (torch._int_mm(a, b).float() * sa * sw.view(1, -1)).half())
    except Exception as e:   # noqa: BLE001
        ms_i = ms_s = float("nan")
    ops_ = 2 * M * K * N
    print(f"{name:18s} K={K:5d} N={N:5d}: cutlass_scaled_mm {ms_c:7.3f} ms ({ops_ / ms_c / 1e9:5.1f} TOPS) | "
          f"_int_mm {ms_i:7.3f} ms ({ops_ / ms_i / 1e9:5.1f} TOPS), +scale {ms_s:7.3f} ms", flush=True)
