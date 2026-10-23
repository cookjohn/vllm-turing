#!/usr/bin/env python3
"""vllm_turing small-M W8A8 GEMV vs vLLM's CUTLASS op at the Qwen3.8-27B per-GPU shapes (TP2): exactness and speed.
usage: test_w8a8.py   (run with the vLLM env python, one GPU)"""
import time

import torch
import vllm._custom_ops  # noqa: F401  (loads torch.ops._C)

from vllm_turing import w8a8

SHAPES = {"gdn in_proj_qkvz": (8192, 5120), "gdn out_proj": (5120, 3072), "attn qkv": (7168, 5120),
          "attn o_proj": (5120, 3072), "mlp gate_up": (17408, 5120), "mlp down": (5120, 8704)}


def bench(fn, it=50):
    fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(it):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / it


tot_c, tot_g = {}, {}
for name, (N, K) in SHAPES.items():
    w = torch.randint(-128, 128, (N, K), dtype=torch.int8, device="cuda")
    b = w.t()                                       # vLLM's layout: transposed view of the row-major [N, K] weight
    sb = torch.rand(N, 1, device="cuda") * 1e-3
    for M in (1, 3, 4, 5, 6, 8, 10, 14, 16):
        a = torch.randint(-128, 128, (M, K), dtype=torch.int8, device="cuda")
        sa = torch.rand(M, 1, device="cuda") * 1e-2
        oc = torch.empty(M, N, dtype=torch.float16, device="cuda")
        og = torch.empty_like(oc)
        torch.ops._C.cutlass_scaled_mm(oc, a, b, sa, sb, None)
        assert w8a8.gemv_ok(og, a, b, sa, sb, None, 16), name     # the kernel takes up to 16 rows
        w8a8.gemv(og, a, b, sa, sb, None)
        diff = (oc.float() - og.float()).abs().max().item()
        rel = diff / oc.float().abs().max().item()
        tc = bench(lambda: torch.ops._C.cutlass_scaled_mm(oc, a, b, sa, sb, None))
        tg = bench(lambda: w8a8.gemv(og, a, b, sa, sb, None))
        tot_c[M] = tot_c.get(M, 0) + tc
        tot_g[M] = tot_g.get(M, 0) + tg
        print(f"{name:17s} M={M}  cutlass {tc * 1e6:7.1f} us ({N * K / tc / 1e9:5.0f} GB/s)  "
              f"gemv {tg * 1e6:7.1f} us ({N * K / tg / 1e9:5.0f} GB/s)  max rel diff {rel:.1e}")
for M in tot_c:
    print(f"one layer of each shape, M={M}: cutlass {tot_c[M] * 1e6:.0f} us, gemv {tot_g[M] * 1e6:.0f} us "
          f"({tot_c[M] / tot_g[M]:.2f}x)")
