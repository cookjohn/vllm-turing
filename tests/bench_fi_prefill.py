#!/usr/bin/env python3
"""FlashInfer FA2 paged prefill on sm75 for this model's attention shape per TP2 rank (12 Q heads, 2 KV heads,
head_dim 256, FP8 e4m3 KV, page 1680, NHD): one causal 6720-token chunk (an align-mode prefill chunk) at a given
context depth, use_fp16_qk_reduction False vs True: time, achieved TFLOPS, output error vs an fp32 reference on a
sample of query rows.
usage: bench_fi_prefill.py [depth ...]   (vLLM env python, one GPU)"""
import sys
import time

import flashinfer
import torch

dev = "cuda"
HQ, HKV, D, PAGE, QLEN = 12, 2, 256, 1680, 6720
depths = [int(a) for a in sys.argv[1:]] or [32768, 131072]
torch.manual_seed(0)
ws = torch.empty(256 << 20, dtype=torch.uint8, device=dev)


def ref_rows(q, k, v, rows, total):
    """fp32 attention for query rows `rows` (causal, aligned to the end of the kv sequence)."""
    out = []
    g = HQ // HKV
    for r in rows:
        kv_end = total - QLEN + r + 1
        qr = q[r].float()                                   # [HQ, D]
        kk = k[:kv_end].float()                             # [L, HKV, D]
        vv = v[:kv_end].float()
        s = torch.einsum("hd,lhd->hl", qr, kk.repeat_interleave(g, dim=1)) * D ** -0.5
        p = torch.softmax(s, dim=-1)
        out.append(torch.einsum("hl,lhd->hd", p, vv.repeat_interleave(g, dim=1)))
    return torch.stack(out)


for depth in depths:
    total = depth + QLEN
    n_pages = (total + PAGE - 1) // PAGE
    kv = (torch.randn(n_pages, 2, PAGE, HKV, D, device=dev, dtype=torch.float16) * 0.5).to(torch.float8_e4m3fn)
    # QK-normed model: unit-RMS q / k rows
    q = torch.randn(QLEN, HQ, D, device=dev, dtype=torch.float16)
    k_flat = kv[:, 0].reshape(-1, HKV, D)[:total]
    v_flat = kv[:, 1].reshape(-1, HKV, D)[:total]
    rows = torch.linspace(0, QLEN - 1, 24).long().tolist()
    ref = ref_rows(q, k_flat, v_flat, rows, total)
    last = total - (n_pages - 1) * PAGE
    flops = 4 * HQ * D * QLEN * (depth + QLEN / 2)
    for fp16qk in (False, True):
        w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD", backend="fa2")
        w.plan(torch.tensor([0, QLEN], dtype=torch.int32), torch.tensor([0, n_pages], dtype=torch.int32),
               torch.arange(n_pages, dtype=torch.int32, device=dev), torch.tensor([last], dtype=torch.int32),
               HQ, HKV, D, PAGE, causal=True, q_data_type=torch.float16, kv_data_type=torch.float8_e4m3fn,
               use_fp16_qk_reduction=fp16qk)
        o = w.run(q, kv)
        torch.cuda.synchronize()
        it = 5
        t = time.time()
        for _ in range(it):
            o = w.run(q, kv)
        torch.cuda.synchronize()
        ms = (time.time() - t) / it * 1e3
        err = (o[rows].float() - ref).abs()
        print(f"depth {depth:6d}: fp16_qk_reduction={fp16qk!s:5}: {ms:8.2f} ms = {flops / ms / 1e9:5.1f} TFLOPS | "
              f"err vs fp32 max {err.max().item():.2e} mean {err.mean().item():.2e}", flush=True)
    del kv
    torch.cuda.empty_cache()
