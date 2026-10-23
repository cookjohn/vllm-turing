#!/usr/bin/env python3
"""Speculative verify as one causal FA2 prefill per request (k query tokens) on a CUDA-graph FlashInfer prefill
wrapper with fixed planning buffers (the approach of weicj/vLLM-2080Ti-Definitive) vs the k flattened single-token
decodes of patches/flashinfer_spec_decode.py: graph safety (capture once, re-plan with other lengths / pages, replay,
compare with an uncaptured causal prefill) and replay time by context depth. Shapes: one TP2 rank of Qwen3.8-27B
(12 query / 2 KV heads x 256), page 1680, FP8 KV with scales, batch 1 and 2.
usage: test_fi_specprefill.py   (vLLM env python, one GPU; ninja / nvcc on PATH)"""
import torch
import flashinfer

dev = "cuda"
hq, hk, d, page = 12, 2, 256, 1680
kvd = torch.float8_e4m3fn
ks, vs = 0.05, 0.04


def layout(seq_lens, pages_per):
    nb = [(s + page - 1) // page for s in seq_lens]
    indptr = torch.tensor([0] + list(torch.tensor(nb).cumsum(0).tolist()), dtype=torch.int32)
    idx = torch.cat([pages_per[i][:n] for i, n in enumerate(nb)]).to(torch.int32)
    last = torch.tensor([s % page or page for s in seq_lens], dtype=torch.int32)
    return indptr, idx, last


def graph_wrapper(batch, max_pages):
    ib = lambda n: torch.empty(n, dtype=torch.int32, device=dev)   # noqa: E731
    return flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(256 << 20, dtype=torch.uint8, device=dev), "NHD", backend="fa2", use_cuda_graph=True,
        qo_indptr_buf=ib(batch + 1), paged_kv_indptr_buf=ib(batch + 1), paged_kv_indices_buf=ib(max_pages),
        paged_kv_last_page_len_buf=ib(batch))


def plan(w, k, seq_lens, pages_per):
    indptr, idx, last = layout(seq_lens, pages_per)
    qo = torch.arange(0, len(seq_lens) * k + 1, k, dtype=torch.int32)
    w.plan(qo, indptr, idx, last, hq, hk, d, page, causal=True, q_data_type=torch.float16, kv_data_type=kvd)


def ref(k, seq_lens, pages_per, q, kv):
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(torch.empty(256 << 20, dtype=torch.uint8, device=dev), "NHD",
                                                       backend="fa2")
    plan(w, k, seq_lens, pages_per)
    return w.run(q, kv, k_scale=ks, v_scale=vs)


def timed_graph(fn, it=50):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(it):
        g.replay()
    e1.record()
    torch.cuda.synchronize()
    return g, e0.elapsed_time(e1) / it * 1e3


torch.manual_seed(0)
max_ctx = 131072 + 64
npages_total = 2 * ((max_ctx + page - 1) // page) + 8
kv = (torch.randn(npages_total, 2, page, hk, d, device=dev, dtype=torch.float16) * 2).to(kvd)
perm = torch.randperm(npages_total, device=dev).to(torch.int32)
per_req = [perm[: npages_total // 2], perm[npages_total // 2:]]

# 1) graph safety: capture at one set of lengths, re-plan at others, replay, compare with an uncaptured prefill
for batch, k in ((1, 5), (2, 5), (2, 8)):
    w = graph_wrapper(batch, batch * ((max_ctx + page - 1) // page))
    q = torch.randn(batch * k, hq, d, device=dev, dtype=torch.float16)
    out = torch.empty(batch * k, hq, d, device=dev, dtype=torch.float16)
    lens0 = [3000 + 977 * i for i in range(batch)]
    plan(w, k, lens0, per_req)
    g, _ = timed_graph(lambda: w.run(q, kv, k_scale=ks, v_scale=vs, out=out), it=1)
    worst = 0.0
    for lens in ([3000 + 977 * i + k for i in range(batch)], [40001 + 10 * i for i in range(batch)],
                 [1681] * batch, [130000 - 7 * i for i in range(batch)]):
        plan(w, k, lens, per_req)
        q.copy_(torch.randn_like(q))
        g.replay()
        r = ref(k, lens, per_req, q, kv)
        worst = max(worst, ((out.float() - r.float()).abs().max() / r.float().abs().max()).item())
    print(f"graph safety batch {batch} k {k}: max rel err after re-plans {worst:.1e}")

# 2) replay time: graph prefill (k queries per request) vs k flattened tensor-core decodes per request
for batch, k in ((1, 5), (1, 8), (2, 5), (2, 8)):
    for ctx in (4096, 32768, 131072):
        lens = [ctx] * batch
        w = graph_wrapper(batch, batch * ((max_ctx + page - 1) // page))
        plan(w, k, lens, per_req)
        q = torch.randn(batch * k, hq, d, device=dev, dtype=torch.float16)
        out = torch.empty_like(q)
        _, t_pre = timed_graph(lambda: w.run(q, kv, k_scale=ks, v_scale=vs, out=out))
        fl = [lens[i] - (k - 1 - j) for i in range(batch) for j in range(k)]
        pf = [per_req[i] for i in range(batch) for _ in range(k)]
        indptr, idx, last = layout(fl, pf)
        ib = lambda n: torch.empty(n, dtype=torch.int32, device=dev)   # noqa: E731
        wd = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            torch.empty(256 << 20, dtype=torch.uint8, device=dev), "NHD", use_cuda_graph=True, use_tensor_cores=True,
            paged_kv_indptr_buffer=ib(batch * k + 1), paged_kv_indices_buffer=ib(idx.numel()),
            paged_kv_last_page_len_buffer=ib(batch * k))
        wd.plan(indptr, idx, last, hq, hk, d, page, q_data_type=torch.float16, kv_data_type=kvd)
        outd = torch.empty_like(q)
        _, t_dec = timed_graph(lambda: wd.run(q, kv, k_scale=ks, v_scale=vs, out=outd))
        print(f"batch {batch} k {k} ctx {ctx:6d}: graph prefill {t_pre:6.0f} us, flattened decodes {t_dec:6.0f} us")
