#!/usr/bin/env python3
"""csrc/silu_quant.cu (SiLU(gate) * up + per-row dynamic int8 quantization in one kernel) vs vLLM's silu_and_mul followed
by scaled_int8_quant: int8 values and scales bitwise equal, at the TP2 MLP width (I = 8704) and a few others; speed of
both at a 1680-row chunk.
usage: test_silu_quant.py [device]   (vLLM env python)"""
import sys

import torch
from vllm import _custom_ops as ops

from vllm_turing.tailoverlap import silu_mul_quant

torch.cuda.set_device(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
fails = 0


def ref(gu):
    I = gu.shape[1] // 2
    act = torch.empty(gu.shape[0], I, dtype=gu.dtype, device=gu.device)
    torch.ops._C.silu_and_mul(act, gu)
    q, s, _ = ops.scaled_int8_quant(act, None, None, symmetric=True)
    return q, s


for M, I in ((1, 8704), (7, 8704), (1680, 8704), (300, 4096), (5, 136)):
    gu = (torch.randn(M, 2 * I, device="cuda") * 3).half()
    gu[0, :I] = 0                  # a zero activation row
    q, s = silu_mul_quant(gu)
    rq, rs = ref(gu)
    ok = torch.equal(q, rq) and torch.equal(s.view(-1), rs.view(-1).float())
    nd = (q != rq).sum().item()
    fails += not ok
    print(f"M={M:5d} I={I:5d}: {'bitwise equal' if ok else f'DIFFER ({nd} int8 values)'}")


def bench(fn, it=50):
    fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(it):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / it


gu = torch.randn(1680, 2 * 8704, device="cuda").half()
print(f"1680 x 8704: fused {bench(lambda: silu_mul_quant(gu)):.3f} ms, silu_and_mul + scaled_int8_quant "
      f"{bench(lambda: ref(gu)):.3f} ms")
print("FAIL" if fails else "PASS")
