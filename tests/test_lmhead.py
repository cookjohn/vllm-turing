#!/usr/bin/env python3
"""INT8 LM head (vllm_turing.lmhead) on the real Qwen3.8-27B lm_head (one TP2 shard: 124160 rows x 5120): logit error
and top-1 / top-5 agreement vs the fp16 head on hidden states drawn like real ones (RMS-normalized, scaled by the
final norm weight), and GEMV speed vs the fp16 cuBLAS GEMV.
usage: test_lmhead.py [checkpoint dir]   (vLLM env python, one GPU)"""
import json
import os
import sys
import time

import torch
from safetensors import safe_open

from vllm_turing import lmhead

ck = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/models/Qwen3.8-27B-W8A8")
wm = json.load(open(os.path.join(ck, "model.safetensors.index.json")))["weight_map"]
with safe_open(os.path.join(ck, wm["lm_head.weight"]), "pt") as f:
    W = f.get_tensor("lm_head.weight")[:124160].to("cuda", torch.float16)
with safe_open(os.path.join(ck, wm["model.language_model.norm.weight"]), "pt") as f:
    g = f.get_tensor("model.language_model.norm.weight").to("cuda", torch.float32)
torch.manual_seed(0)
q, s = lmhead.quantize_rows(W)
for M in (1, 3, 5, 8, 10, 14, 16):
    h = torch.randn(512 * M, W.shape[1], device="cuda")
    h = (h / h.pow(2).mean(-1, keepdim=True).sqrt() * (1 + g)).half()   # Qwen RMSNorm output (1 + weight) scaling
    ref = (h.float() @ W.float().t())
    out = torch.cat([torch.ops.vllm_turing.w8a16_mm(h[i:i + M].contiguous(), q, s, None) for i in range(0, h.shape[0], M)])
    rel = ((out.float() - ref).norm() / ref.norm()).item()
    top1 = (out.float().argmax(-1) == ref.argmax(-1)).float().mean().item()
    top5 = (out.float().topk(5).indices.sort().values == ref.topk(5).indices.sort().values).all(-1).float().mean().item()
    print(f"M={M}: rel logit err {rel:.2e}, top-1 agreement {top1 * 100:.2f}%, top-5 set agreement {top5 * 100:.2f}%")


def bench(fn, it=50):
    fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(it):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / it * 1e3


for M in (1, 3, 5, 8, 10, 14, 16):
    x = torch.randn(M, W.shape[1], device="cuda", dtype=torch.float16)
    t16 = bench(lambda: x @ W.t())
    t8 = bench(lambda: torch.ops.vllm_turing.w8a16_mm(x, q, s, None))
    print(f"M={M}: fp16 head {t16:.3f} ms, int8 head {t8:.3f} ms ({W.numel() / t8 / 1e6:.0f} GB/s)")
