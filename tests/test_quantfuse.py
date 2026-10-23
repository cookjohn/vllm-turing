#!/usr/bin/env python3
"""vllm_turing quantfuse: the torch formula of the dynamic per-token int8 quantization, eager and compiled by inductor
with a producer (RMSNorm-like and SiLU * mul), against vLLM's CUDA op (ops.scaled_int8_quant): scales and int8 values,
and the time of producer + quant as one compiled kernel vs compiled producer + the CUDA op. The eager formula matches the
CUDA op; the compiled fusion quantizes the producer's float32 values without the intermediate fp16 rounding (inductor
drops the round trip), so scales differ at fp16 rounding level (< 1e-3 relative) and ~1% of the int8 values by one.
usage: test_quantfuse.py [device]   (vLLM env python)"""
import sys

import torch
import vllm._custom_ops as ops

from vllm_turing import quantfuse

torch.cuda.set_device(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
cuda_quant = ops.scaled_int8_quant
fails = 0


def bench(fn, it=20, rep=5):
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


def compare(name, q, s, qr, sr, fused=False):
    global fails
    ds = ((s.view(-1) - sr.view(-1)).abs() / sr.view(-1).abs().clamp_min(1e-12)).max().item()
    off = (q.int() - qr.int()).abs()
    n1, nbig = (off == 1).sum().item(), (off > 1).sum().item()
    ok = nbig == 0 and (ds <= 1e-3 and n1 <= 0.02 * q.numel() if fused else ds <= 1e-6 and n1 <= 1e-4 * q.numel())
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {name}: scale max rel diff {ds:.1e}, int8 off by 1: {n1} of {q.numel()} "
          f"({100 * n1 / q.numel():.4f}%), off by more: {nbig}")


def rmsnorm(x, r, w):
    h = x + r
    v = h.float().pow(2).mean(-1, keepdim=True)
    return (h.float() * torch.rsqrt(v + 1e-6) * (1.0 + w.float())).to(x.dtype)


def silu_mul(g):
    a, b = g.chunk(2, dim=-1)
    return torch.nn.functional.silu(a) * b


for M in (5, 30, 6720):
    x = torch.randn(M, 5120, dtype=torch.float16, device="cuda") * 3
    r = torch.randn_like(x)
    w = torch.randn(5120, dtype=torch.float16, device="cuda") * 0.1
    g = torch.randn(M, 17408, dtype=torch.float16, device="cuda") * 2
    for pname, prod, args in (("norm", rmsnorm, (x, r, w)), ("silu*mul", silu_mul, (g,))):
        ref_in = prod(*args)
        qr, sr, _ = cuda_quant(ref_in.contiguous())
        q, s = quantfuse.quant_torch(ref_in)
        compare(f"eager torch formula, {pname}, M={M}", q, s, qr, sr)
        fused = torch.compile(lambda *a: quantfuse.quant_torch(prod(*a)), dynamic=False)
        qc, sc = fused(*args)
        compare(f"compiled {pname} + quant, M={M}", qc, sc, qr, sr, fused=True)
        sep = torch.compile(prod, dynamic=False)
        t_sep = bench(lambda: cuda_quant(sep(*args).contiguous()))
        t_fused = bench(lambda: fused(*args))
        print(f"      {pname} M={M}: compiled producer + CUDA quant {t_sep * 1e3:7.1f} us, fused {t_fused * 1e3:7.1f} us "
              f"({t_sep / t_fused:.2f}x)")
print("ALL PASS" if not fails else f"{fails} FAILED")
