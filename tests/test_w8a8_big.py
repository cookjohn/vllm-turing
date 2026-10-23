#!/usr/bin/env python3
"""vllm_turing prefill int8 GEMM (csrc/w8a8_big.cu: CUTLASS 256x128x64 with vLLM's scaling epilogue) vs vLLM's
cutlass_scaled_mm at the Qwen3.8-27B TP2 prefill shapes: bitwise agreement (same int32 sums, same float epilogue order)
and speed at M = 1680 / 4096 / 6720 (prefill chunk sizes). Timings are the minimum over repeats.
usage: test_w8a8_big.py [device]   (vLLM env python)"""
import sys

import torch
import vllm._custom_ops  # noqa: F401

from vllm_turing import w8a8

torch.cuda.set_device(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
SHAPES = {"mlp gate_up": (8704, 5120), "mlp down": (5120, 8704), "gdn in_proj": (8192, 5120),
          "gdn out_proj": (5120, 3072), "attn qkv": (3584, 5120), "attn o_proj": (5120, 3072)}
PEAK = 68 * 2048 * 1845e6
fails = 0


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


for M in (1680, 4096, 6720):
    tv = tb = 0.0
    for name, (N, K) in SHAPES.items():
        a = torch.randint(-128, 128, (M, K), dtype=torch.int8, device="cuda")
        w = torch.randint(-128, 128, (N, K), dtype=torch.int8, device="cuda")
        b = w.t()
        sa = torch.rand(M, 1, device="cuda") * 1e-2
        sb = torch.rand(N, 1, device="cuda") * 1e-3
        for dtype in (torch.float16, torch.bfloat16):
            ref = torch.empty(M, N, dtype=dtype, device="cuda")
            out = torch.empty_like(ref)
            torch.ops._C.cutlass_scaled_mm(ref, a, b, sa, sb, None)
            assert w8a8.big_ok(out, a, b, sa, sb, None), name
            w8a8.big(out, a, b, sa, sb)
            torch.cuda.synchronize()
            same = torch.equal(out, ref)
            if not same:
                rel = ((out.float() - ref.float()).abs().max() / ref.float().abs().max()).item()
                ok = rel < 1e-3
                fails += not ok
                print(f"{'PASS' if ok else 'FAIL'}  {name} M={M} {dtype}: not bitwise equal, max rel diff {rel:.1e}")
        o2 = torch.empty(M, N, dtype=torch.float16, device="cuda")
        t1 = bench(lambda: torch.ops._C.cutlass_scaled_mm(o2, a, b, sa, sb, None))
        t2 = bench(lambda: w8a8.big(o2, a, b, sa, sb))
        tv += t1
        tb += t2
        ops = 2.0 * M * N * K
        print(f"{name:13s} M={M}: vLLM {t1:6.2f} ms ({100 * ops / (t1 * 1e-3) / PEAK:3.0f}%)  big {t2:6.2f} ms "
              f"({100 * ops / (t2 * 1e-3) / PEAK:3.0f}%)  {t1 / t2:.2f}x")
        del a, w, b, ref, out, o2
    print(f"M={M}: one layer of each shape vLLM {tv:.2f} ms, big {tb:.2f} ms ({tv / tb:.2f}x)")
print("PASS  bitwise equal to cutlass_scaled_mm on all shapes / dtypes" if not fails else f"{fails} FAILED")
