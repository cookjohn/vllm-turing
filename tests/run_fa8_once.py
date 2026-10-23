#!/usr/bin/env python3
"""One fiattn8 causal prefill (QLEN query tokens at DEPTH context, the production shape) through the hooked FlashInfer
wrapper, for profilers (Nsight Compute: --kernel-name fa8_kernel). Warm-up run first, then the profiled run.
usage: run_fa8_once.py [depth=32768] [qlen=6720]   (vLLM env python, one GPU)"""
import sys

import flashinfer
import torch

from vllm_turing import fiattn8

HQ, HKV, D, PAGE = 12, 2, 256, 1680
depth = int(sys.argv[1]) if len(sys.argv) > 1 else 32768
qlen = int(sys.argv[2]) if len(sys.argv) > 2 else 6720
dev = "cuda"
torch.manual_seed(0)
fiattn8.ENABLED = True
fiattn8.install()
W = flashinfer.BatchPrefillWithPagedKVCacheWrapper
kv = depth + qlen
npg = (kv + PAGE - 1) // PAGE
raw = (torch.randn(npg, PAGE, HKV, 2 * D, device=dev, dtype=torch.float16) * 0.6).to(torch.float8_e4m3fn)
k, v = raw.split(D, dim=-1)
q = torch.randn(qlen, HQ, D, device=dev, dtype=torch.float16)
w = W(torch.empty(256 << 20, dtype=torch.uint8, device=dev), "NHD")
w.plan(torch.tensor([0, qlen], dtype=torch.int32), torch.tensor([0, npg], dtype=torch.int32),
       torch.arange(npg, dtype=torch.int32, device=dev), torch.tensor([kv - (npg - 1) * PAGE], dtype=torch.int32),
       HQ, HKV, D, PAGE, causal=True, q_data_type=torch.float16, kv_data_type=torch.float8_e4m3fn)
for _ in range(2):
    o = w.run(q, (k, v), k_scale=1.0, v_scale=1.0)
torch.cuda.synchronize()
print("ok", o.shape, torch.isfinite(o).all().item())
