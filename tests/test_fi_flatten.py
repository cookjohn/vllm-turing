#!/usr/bin/env python3
"""Exactness of the sm75 spec-as-decode flattening (patches/flashinfer_spec_decode.py): on one paged KV cache, the
causal FlashInfer prefill over k verify tokens per request (reference) vs k single-token FlashInfer decodes per
request whose KV length is seq_len - (k - 1 - j) - the expansion the patch builds (same numpy code).
Shapes: 12 query heads, 2 KV heads, head_dim 256, fp16 (one GPU of TP2).
usage: test_fi_flatten.py   (vLLM env python, one GPU)"""
import numpy as np
import torch
import flashinfer

torch.manual_seed(0)
dev = "cuda"
hq, hk, d, k = 12, 2, 256, 3
for page in (16, 784):
    seq_lens = np.array([1000, 37, 4101], dtype=np.int32)          # include the k new tokens of each request
    nreq = len(seq_lens)
    pages_per = (seq_lens + page - 1) // page
    total_pages = int(pages_per.sum()) + 5
    kv = (torch.randn(total_pages, 2, page, hk, d, device=dev, dtype=torch.float16) * 0.5)
    perm = torch.randperm(total_pages, device=dev).to(torch.int32)   # scattered pages, like a real block table
    max_pages = int(pages_per.max())
    block_table = torch.zeros(nreq, max_pages, dtype=torch.int32, device=dev)
    o = 0
    for i, n in enumerate(pages_per):
        block_table[i, :n] = perm[o:o + n]
        o += n
    q = torch.randn(nreq * k, hq, d, device=dev, dtype=torch.float16)

    def paged(sl, bt):
        nb = (sl + page - 1) // page
        indptr = np.concatenate([[0], np.cumsum(nb)]).astype(np.int32)
        idx = torch.cat([bt[i, :nb[i]] for i in range(len(sl))])
        last = np.where((sl % page == 0) & (sl != 0), page, sl % page).astype(np.int32)
        return torch.tensor(indptr, device=dev), idx, torch.tensor(last, device=dev)

    ws = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
    # reference: causal prefill, k query tokens per request (the last k positions of each sequence)
    pre = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD")
    indptr, idx, last = paged(seq_lens, block_table)
    qo = torch.arange(0, nreq * k + 1, k, dtype=torch.int32, device=dev)
    pre.plan(qo, indptr, idx, last, hq, hk, d, page, causal=True, q_data_type=torch.float16)
    ref = pre.run(q, kv)
    # flattened: the patch's expansion
    offs = np.tile(np.arange(k - 1, -1, -1, dtype=seq_lens.dtype), nreq)
    sl_f = np.maximum(np.repeat(seq_lens, k) - offs, 0)
    bt_f = block_table.repeat_interleave(k, dim=0)
    indptr_f, idx_f, last_f = paged(sl_f, bt_f)
    dec = flashinfer.BatchDecodeWithPagedKVCacheWrapper(torch.empty(256 << 20, dtype=torch.uint8, device=dev), "NHD",
                                                        use_tensor_cores=True)
    dec.plan(indptr_f, idx_f, last_f, hq, hk, d, page, q_data_type=torch.float16)
    out = dec.run(q, kv)
    err = ((out.float() - ref.float()).abs().max() / ref.float().abs().max()).item()
    # negative control: every token sees the whole sequence (what an off-by-k bug would do)
    indptr_b, idx_b, last_b = paged(np.repeat(seq_lens, k), bt_f)
    dec.plan(indptr_b, idx_b, last_b, hq, hk, d, page, q_data_type=torch.float16)
    bad = ((dec.run(q, kv).float() - ref.float()).abs().max() / ref.float().abs().max()).item()
    print(f"page {page:4d}: flattened decode vs causal prefill max rel err {err:.1e} (non-causal control {bad:.1e})")

# non-causal blocks (DFlash drafts: k = 8 queries, 16 query / 4 KV heads x 128 per TP2 rank, sliding window 2048):
# FlashInfer's non-causal prefill vs k single-token decodes that all see the whole sequence (the patch's expansion
# for causal=False). Without a window the two are the same attention; with the window the decodes measure it from the
# last key for every query (the prefill from each query's own position), so a few keys at the window's left edge
# differ for the first queries of the block (expected small, nonzero).
hq, hk, d, k = 16, 4, 128, 8
for page, wl in ((64, -1), (1664, -1), (64, 2047)):
    seq_lens = np.array([3000, 12, 9001], dtype=np.int32)
    nreq = len(seq_lens)
    pages_per = (seq_lens + page - 1) // page
    total_pages = int(pages_per.sum()) + 5
    kv = (torch.randn(total_pages, 2, page, hk, d, device=dev, dtype=torch.float16) * 0.5)
    perm = torch.randperm(total_pages, device=dev).to(torch.int32)
    block_table = torch.zeros(nreq, int(pages_per.max()), dtype=torch.int32, device=dev)
    o = 0
    for i, n in enumerate(pages_per):
        block_table[i, :n] = perm[o:o + n]
        o += n
    q = torch.randn(nreq * k, hq, d, device=dev, dtype=torch.float16)

    def paged(sl, bt):
        nb = (sl + page - 1) // page
        indptr = np.concatenate([[0], np.cumsum(nb)]).astype(np.int32)
        idx = torch.cat([bt[i, :nb[i]] for i in range(len(sl))])
        last = np.where((sl % page == 0) & (sl != 0), page, sl % page).astype(np.int32)
        return torch.tensor(indptr, device=dev), idx, torch.tensor(last, device=dev)

    pre = flashinfer.BatchPrefillWithPagedKVCacheWrapper(torch.empty(256 << 20, dtype=torch.uint8, device=dev), "NHD")
    indptr, idx, last = paged(seq_lens, block_table)
    qo = torch.arange(0, nreq * k + 1, k, dtype=torch.int32, device=dev)
    pre.plan(qo, indptr, idx, last, hq, hk, d, page, causal=False, window_left=wl, q_data_type=torch.float16)
    ref = pre.run(q, kv)
    bt_f = block_table.repeat_interleave(k, dim=0)
    indptr_f, idx_f, last_f = paged(np.repeat(seq_lens, k), bt_f)
    dec = flashinfer.BatchDecodeWithPagedKVCacheWrapper(torch.empty(256 << 20, dtype=torch.uint8, device=dev), "NHD",
                                                        use_tensor_cores=True)
    dec.plan(indptr_f, idx_f, last_f, hq, hk, d, page, window_left=wl, q_data_type=torch.float16)
    out = dec.run(q, kv)
    err = ((out.float() - ref.float()).abs().max() / ref.float().abs().max()).item()
    print(f"non-causal k={k} page {page:4d} window_left {wl:5d}: flattened decode vs non-causal prefill "
          f"max rel err {err:.1e}")
