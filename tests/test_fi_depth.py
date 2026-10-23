#!/usr/bin/env python3
"""FlashInfer decode cost vs context depth at the Qwen3.8-27B TP2 shapes (12 query / 2 KV heads x 256, page 1680):
a k-token speculative verify flattened into k single-token decodes (what patches/flashinfer_spec_decode.py does) vs one
decode with the k tokens folded into the query heads (GQA group 6 -> 6k: every KV page is read and dequantized once),
for FP8 and FP16 KV caches, tensor-core decode as vLLM uses it.
usage: test_fi_depth.py   (vLLM env python, one GPU; ninja / nvcc on PATH for FlashInfer's JIT)"""
import time

import torch
import flashinfer

dev = "cuda"
hq, hk, d, page = 12, 2, 256, 1680


def bench(fn, it=30):
    fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(it):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / it * 1e6


for kv_dtype in (torch.float8_e4m3fn, torch.float16):
    for ctx in (4096, 32768, 131072):
        npages = (ctx + page - 1) // page
        kv = (torch.randn(npages, 2, page, hk, d, device=dev, dtype=torch.float16) * 0.5).to(kv_dtype)
        pages = torch.arange(npages, dtype=torch.int32, device=dev)
        res = []
        for k in (1, 5, 8):
            # flattened: k entries over the same pages, lengths ctx - (k - 1 - j)
            sl = torch.tensor([ctx - (k - 1 - j) for j in range(k)], dtype=torch.int32)
            nb = (sl + page - 1) // page
            indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(nb, 0).to(torch.int32)]).to(dev)
            idx = torch.cat([pages[:n] for n in nb.tolist()])
            last = torch.where(sl % page == 0, page, sl % page).to(torch.int32).to(dev)
            w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(torch.empty(256 << 20, dtype=torch.uint8, device=dev),
                                                             "NHD", use_tensor_cores=True)
            w.plan(indptr, idx, last, hq, hk, d, page, q_data_type=torch.float16, kv_data_type=kv_dtype)
            q = torch.randn(k, hq, d, device=dev, dtype=torch.float16)
            t_flat = bench(lambda: w.run(q, kv))
            # folded: one entry, k * hq query heads (token-major within each KV head's group)
            wf = flashinfer.BatchDecodeWithPagedKVCacheWrapper(torch.empty(256 << 20, dtype=torch.uint8, device=dev),
                                                              "NHD", use_tensor_cores=True)
            ip1 = torch.tensor([0, npages], dtype=torch.int32, device=dev)
            l1 = torch.tensor([ctx % page or page], dtype=torch.int32, device=dev)
            wf.plan(ip1, pages, l1, hq * k, hk, d, page, q_data_type=torch.float16, kv_data_type=kv_dtype)
            qf = torch.randn(1, hq * k, d, device=dev, dtype=torch.float16)
            t_fold = bench(lambda: wf.run(qf, kv))
            res.append(f"k={k}: flattened {t_flat:7.0f} us, folded {t_fold:6.0f} us")
        mb = npages * page * hk * d * 2 * kv.element_size() / 1e6
        print(f"{str(kv_dtype).split('.')[-1]:14s} ctx {ctx:6d} ({mb:5.0f} MB KV): " + " | ".join(res))
