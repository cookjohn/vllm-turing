#!/usr/bin/env python3
"""vllm_turing int8 tensor-core kernel (M = 13..32, MTP4 verify of 3..6 concurrent requests) vs vLLM's CUTLASS op at the
Qwen3.8-27B per-GPU shapes (TP2): exactness (fp16 / bf16, bias, M not a multiple of 8, K not a multiple of 64, N not a
multiple of 16) and speed, with the GEMV kernel for M <= 16 as reference. Timings are the minimum over repeats (the
GPUs may be shared with a running server).
usage: test_w8a8_mma.py [device]   (run with the vLLM env python)"""
import sys

import torch
import vllm._custom_ops  # noqa: F401  (loads torch.ops._C)

from vllm_turing import w8a8

torch.cuda.set_device(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
SHAPES = {"gdn in_proj_qkvz": (8192, 5120), "gdn out_proj": (5120, 3072), "attn qkv": (7168, 5120),
          "attn o_proj": (5120, 3072), "mlp gate_up": (17408, 5120), "mlp down": (5120, 8704)}
fails = 0


def bench(fn, it=30, rep=5):
    fn()
    best = 1e9
    for _ in range(rep):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(it):
            fn()
        e1.record()
        torch.cuda.synchronize()
        best = min(best, e0.elapsed_time(e1) / it / 1e3)
    return best


def check(N, K, M, dtype=torch.float16, with_bias=False, per_row=True):
    global fails
    w = torch.randint(-128, 128, (N, K), dtype=torch.int8, device="cuda")
    b = w.t()
    sb = torch.rand(N, 1, device="cuda") * 1e-3
    a = torch.randint(-128, 128, (M, K), dtype=torch.int8, device="cuda")
    sa = torch.rand(M, 1, device="cuda") * 1e-2 if per_row else torch.rand(1, device="cuda") * 1e-2
    bias = torch.randn(N, dtype=dtype, device="cuda") if with_bias else None
    # float64 reference on the CPU (the GPU may be nearly full with a running server); exact for int8 sums
    ref = (a.cpu().double() @ w.cpu().double().t()) * sa.cpu().double().view(-1, 1) * sb.cpu().double().view(1, -1)
    if bias is not None:
        ref = ref + bias.cpu().double()
    out = torch.empty(M, N, dtype=dtype, device="cuda")
    w8a8.mma(out, a, b, sa, sb, bias)
    torch.cuda.synchronize()
    err = ((out.cpu().double() - ref).abs() / (ref.abs() + 1e-3)).max().item()
    del w, b
    tol = 4e-3 if dtype == torch.float16 else 2e-2
    ok = err < tol
    fails += not ok
    if not ok:
        print(f"FAIL  exactness N={N} K={K} M={M} {dtype} bias={with_bias} per_row={per_row}: max rel err {err:.2e}")
    return ok


# ---- exactness ------------------------------------------------------------------------------------------------------
n_ok = 0
for M in (13, 15, 17, 20, 23, 25, 29, 30, 32):
    n_ok += check(5120, 8704, M)
n_ok += check(8192, 5120, 31, torch.bfloat16, with_bias=True)
n_ok += check(5120, 3072, 21, per_row=False)
n_ok += check(1000, 5136, 27)        # K % 128 = 16, N % 16 = 8
n_ok += check(300, 5168, 32, torch.bfloat16)
n_ok += check(7168, 5120, 1)          # the kernel also handles small M
n_ok += check(17408, 5120, 20)
print(f"{'PASS' if not fails else 'FAIL'}  exactness vs float64 reference: {n_ok} cases")

# ---- the dispatch inside vLLM's op ----------------------------------------------------------------------------------
w = torch.randint(-128, 128, (5120, 8704), dtype=torch.int8, device="cuda")
a = torch.randint(-128, 128, (30, 8704), dtype=torch.int8, device="cuda")
sa, sb = torch.rand(30, 1, device="cuda") * 1e-2, torch.rand(5120, 1, device="cuda") * 1e-3
o1, o2 = torch.empty(30, 5120, dtype=torch.float16, device="cuda"), torch.empty(30, 5120, dtype=torch.float16,
                                                                                device="cuda")
torch.ops.vllm_turing.scaled_mm_(o1, a, w.t(), sa, sb, None)
torch.ops._C.cutlass_scaled_mm(o2, a, w.t(), sa, sb, None)
rel = ((o1.float() - o2.float()).abs().max() / o2.float().abs().max()).item()
ok = w8a8.mma_ok(o1, a, w.t(), sa, sb, None) and rel < 1e-3
fails += not ok
print(f"{'PASS' if ok else 'FAIL'}  scaled_mm_ dispatch at M=30 uses the tensor-core kernel, vs CUTLASS max rel {rel:.1e}")

# ---- speed ----------------------------------------------------------------------------------------------------------
tot = {}
for name, (N, K) in SHAPES.items():
    w = torch.randint(-128, 128, (N, K), dtype=torch.int8, device="cuda")
    b = w.t()
    sb = torch.rand(N, 1, device="cuda") * 1e-3
    row = []
    for M in (5, 10, 13, 16, 20, 25, 30, 32):
        a = torch.randint(-128, 128, (M, K), dtype=torch.int8, device="cuda")
        sa = torch.rand(M, 1, device="cuda") * 1e-2
        o = torch.empty(M, N, dtype=torch.float16, device="cuda")
        tc = bench(lambda: torch.ops._C.cutlass_scaled_mm(o, a, b, sa, sb, None))
        tm = bench(lambda: w8a8.mma(o, a, b, sa, sb, None))
        tg = bench(lambda: w8a8.gemv(o, a, b, sa, sb, None)) if M <= 16 else None
        t = tot.setdefault(M, [0, 0, 0])
        t[0] += tc
        t[1] += tm
        t[2] += tg or 0
        row.append(f"M={M}: cutlass {tc * 1e6:.0f} / mma {tm * 1e6:.0f} ({N * K / tm / 1e9:.0f} GB/s)"
                   + (f" / gemv {tg * 1e6:.0f}" if tg else ""))
    print(f"{name} N={N} K={K}\n    " + "\n    ".join(row))
print("one layer of each shape (us):")
for M, (tc, tm, tg) in tot.items():
    print(f"  M={M:2d}: cutlass {tc * 1e6:5.0f}  mma {tm * 1e6:5.0f}" + (f"  gemv {tg * 1e6:5.0f}" if tg else "")
          + f"  -> best {min(x for x in (tc, tm, tg) if x) * 1e6:.0f}")
print("ALL PASS" if not fails else f"{fails} FAILED")
