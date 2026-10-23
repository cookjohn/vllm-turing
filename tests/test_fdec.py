#!/usr/bin/env python3
"""vllm_turing fdec (csrc/fdec.cu) vs FlashInfer's FA2 prefill wrapper: speculative verify (q_len query tokens per
request, causal or not) and decode attention over FP8 paged KV with 16-token pages, K|V packed in the last dim of one
cache as vLLM allocates it (HND and NHD views), random page order, k / v scales; plans whose last-page length is <= 0
(nosync.py's upper-bound plans); then CUDA-graph replay time vs FlashInfer by context depth. One TP2 rank of
Qwen3.8-27B: 12 Q / 2 KV heads x 256.
usage: test_fdec.py [--time-only | --check-only]   (vLLM env python, one GPU)"""
import os
import sys

import torch
import flashinfer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vllm_turing import fdec   # noqa: E402

dev = "cuda"
hq, hk, d, page = 12, 2, 256, 16
kvd = torch.float8_e4m3fn
ks, vs = 0.05, 0.04
scale = d ** -0.5
torch.manual_seed(0)
fi_ws = torch.empty(128 << 20, dtype=torch.uint8, device=dev)


def make_cache(npages, layout):
    raw = torch.empty(npages, hk, page, 2 * d, dtype=torch.uint8, device=dev) if layout == "HND" else \
        torch.empty(npages, page, hk, 2 * d, dtype=torch.uint8, device=dev)
    for i in range(0, npages, 1024):   # chunks: next to a running server
        raw[i:i + 1024].view(kvd).copy_((torch.randn(raw[i:i + 1024].shape, device=dev) * 3).to(kvd))
    kv = raw.view(kvd)
    return kv[..., :d], kv[..., d:]


def meta(lens, perm, extra_page=None):
    """indptr / indices / last for lengths; extra_page[i]: plan one more page than needed (last <= 0)"""
    nb, last = [], []
    for i, n in enumerate(lens):
        p = (n + page - 1) // page
        if extra_page and extra_page[i]:
            nb.append(p + 1)
            last.append(n - p * page)
        else:
            nb.append(p)
            last.append(n - (p - 1) * page)
    indptr = [0]
    for n in nb:
        indptr.append(indptr[-1] + n)
    # request i's pages from its own slice of perm: the same pages with or without an extra planned page
    per = len(perm) // len(lens)
    idx = torch.cat([perm[i * per: i * per + n] for i, n in enumerate(nb)])
    return (torch.tensor(indptr, dtype=torch.int32), idx, torch.tensor(last, dtype=torch.int32))


def fi_ref(q, K, V, layout, q_len, causal, lens, perm):
    indptr, idx, last = meta(lens, perm)
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(fi_ws, layout, backend="fa2")
    qo = torch.arange(0, len(lens) * q_len + 1, q_len, dtype=torch.int32)
    w.plan(qo, indptr, idx, last, hq, hk, d, page, causal=causal, q_data_type=torch.float16, kv_data_type=kvd,
           sm_scale=scale)
    return w.run(q, (K, V), k_scale=ks, v_scale=vs)


def exact_ref(q, K, V, layout, q_len, causal, lens, perm):
    """float64 on the GPU for small cases"""
    indptr, idx, last = meta(lens, perm)
    out = torch.zeros(q.shape, dtype=torch.float64, device=dev)
    for b, n in enumerate(lens):
        pg = idx[indptr[b]:indptr[b + 1]].long()
        if layout == "HND":
            kk = K[pg].permute(0, 2, 1, 3).reshape(-1, hk, d)[:n].double() * ks
            vv = V[pg].permute(0, 2, 1, 3).reshape(-1, hk, d)[:n].double() * vs
        else:
            kk = K[pg].reshape(-1, hk, d)[:n].double() * ks
            vv = V[pg].reshape(-1, hk, d)[:n].double() * vs
        for j in range(q_len):
            lim = n - (q_len - 1 - j) if causal else n
            if lim <= 0:
                continue
            for h in range(hq):
                s = (kk[:lim, h // 6] @ q[b * q_len + j, h].double()) * scale
                out[b * q_len + j, h] = torch.softmax(s, 0) @ vv[:lim, h // 6]
    return out


def ours(q, K, V, layout, q_len, causal, lens, perm, extra_page=None):
    indptr, idx, last = meta(lens, perm, extra_page)
    out = torch.full_like(q, float("nan"))
    ok = fdec.attention(q, K, V, layout, indptr.to(dev), idx.to(dev), last.to(dev), q_len, causal, scale, ks, vs, out)
    assert ok
    return out


def rel(a, b):
    return ((a.double() - b.double()).abs().max() / b.double().abs().max().clamp_min(1e-9)).item()


check = "--time-only" not in sys.argv
timing = "--check-only" not in sys.argv
max_ctx = 131072
npages = 2 * max_ctx // page + 64

if check:
    worst = 0.0
    for layout in ("HND", "NHD"):
        K, V = make_cache(npages, layout)
        perm = torch.randperm(npages, device=dev).to(torch.int32)
        # exact reference, small
        for q_len, causal, lens in ((5, True, [1, 4, 17, 300]), (1, True, [16, 33]), (8, False, [9, 1000]),
                                    (5, True, [3])):
            q = torch.randn(len(lens) * q_len, hq, d, device=dev, dtype=torch.float16)
            e = exact_ref(q, K, V, layout, q_len, causal, lens, perm)
            o = ours(q, K, V, layout, q_len, causal, lens, perm)
            f = fi_ref(q, K, V, layout, q_len, causal, lens, perm) if min(lens) >= q_len or not causal else None
            r_o = rel(o, e)
            r_f = rel(f, e) if f is not None else float("nan")
            print(f"{layout} q_len {q_len} causal {causal} lens {lens}: vs float64 ours {r_o:.1e}, FlashInfer {r_f:.1e}")
            worst = max(worst, r_o)
        # FlashInfer reference, long
        for q_len, causal, lens in ((5, True, [131000]), (1, True, [120000]), (5, True, [65536, 40001]),
                                    (8, True, [70000, 3]), (5, False, [50000]),
                                    (5, True, [4096, 777, 16384, 100, 9000, 20000]),
                                    (1, True, [1, 2, 4097, 30000, 15, 64])):
            if causal and min(lens) < q_len:
                lens = [max(n, q_len) for n in lens]
            q = torch.randn(len(lens) * q_len, hq, d, device=dev, dtype=torch.float16)
            f = fi_ref(q, K, V, layout, q_len, causal, lens, perm)
            o = ours(q, K, V, layout, q_len, causal, lens, perm)
            r = rel(o, f)
            o2 = ours(q, K, V, layout, q_len, causal, lens, perm, extra_page=[i % 2 == 0 for i in range(len(lens))])
            r2 = rel(o2, f)
            print(f"{layout} q_len {q_len} causal {causal} lens {lens}: vs FlashInfer {r:.1e} "
                  f"(upper-bound plan {r2:.1e})")
            worst = max(worst, r, r2)
        del K, V
    print(f"worst relative error {worst:.1e}", "OK" if worst < 1e-2 else "FAIL")


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
    best = 1e9
    for _ in range(3):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(it):
            g.replay()
        e1.record()
        torch.cuda.synchronize()
        best = min(best, e0.elapsed_time(e1) / it * 1e3)
    return best


if timing:
    layout = "HND"
    K, V = make_cache(npages, layout)
    perm = torch.randperm(npages, device=dev).to(torch.int32)
    ib = lambda n: torch.empty(n, dtype=torch.int32, device=dev)   # noqa: E731
    for batch, q_len in ((1, 5), (1, 1), (1, 8), (2, 5), (6, 5)):
        for ctx in (4096, 32768, 120000):
            if batch * ctx > max_ctx * 2:
                continue
            lens = [ctx] * batch
            indptr, idx, last = meta(lens, perm)
            w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                fi_ws, layout, backend="fa2", use_cuda_graph=True,
                qo_indptr_buf=ib(batch + 1), paged_kv_indptr_buf=ib(batch + 1), paged_kv_indices_buf=ib(idx.numel()),
                paged_kv_last_page_len_buf=ib(batch))
            qo = torch.arange(0, batch * q_len + 1, q_len, dtype=torch.int32)
            w.plan(qo, indptr, idx, last, hq, hk, d, page, causal=True, q_data_type=torch.float16, kv_data_type=kvd,
                   sm_scale=scale)
            q = torch.randn(batch * q_len, hq, d, device=dev, dtype=torch.float16)
            out_f, out_o = torch.empty_like(q), torch.empty_like(q)
            t_f = timed_graph(lambda: w.run(q, (K, V), k_scale=ks, v_scale=vs, out=out_f))
            di, dx, dl = indptr.to(dev), idx.to(dev), last.to(dev)
            t_o = timed_graph(lambda: fdec.attention(q, K, V, layout, di, dx, dl, q_len, True, scale, ks, vs, out_o))
            gbs = batch * ctx * hk * 2 * d / (t_o * 1e-6) / 1e9
            print(f"batch {batch} q_len {q_len} ctx {ctx:6d}: FlashInfer {t_f:6.0f} us, fdec {t_o:6.0f} us "
                  f"({t_f / t_o:.2f}x, {gbs:.0f} GB/s), max rel diff {rel(out_o, out_f):.1e}")
