#!/usr/bin/env python3
"""vllm_turing fiattn8 (csrc/fiattn8.cu through the hooked FlashInfer prefill wrapper) vs FlashInfer FA2 and an fp32
reference: causal prefill on a vLLM-style FP8 paged cache (pages of 1680 tokens holding K|V packed in the last dim,
shuffled page ids), 12 Q / 2 KV heads, D 256; single requests at several context depths (one and several KV chunks,
query sub-batches), two requests in one batch, NHD and HND views; then timing of one 6720-token chunk at depth.
usage: test_fiattn8.py [timing depths ...]   (vLLM env python, one GPU)"""
import sys
import time

import flashinfer
import torch

from vllm_turing import fiattn8

HQ, HKV, D, PAGE = 12, 2, 256, 1680
dev = "cuda"
torch.manual_seed(0)
ws = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
fiattn8.ENABLED = True
fiattn8.install()
W = flashinfer.BatchPrefillWithPagedKVCacheWrapper
assert getattr(W, "_vllm_turing_fa8", False)


def make_cache(n_pages, layout):
    """vLLM-like packed cache [pages, page, heads, 2D] (K | V in the last dim) -> K / V views in `layout`."""
    raw = (torch.randn(n_pages, PAGE, HKV, 2 * D, device=dev, dtype=torch.float16) * 0.6).to(torch.float8_e4m3fn)
    k, v = raw.split(D, dim=-1)                      # NHD views [pages, page, heads, D]
    if layout == "HND":
        k, v = k.transpose(1, 2), v.transpose(1, 2)  # [pages, heads, page, D] views
    return raw, k, v


def reference(q, k_nhd, v_nhd, pages, kv_len, rows, k_scale, v_scale):
    """fp32 causal attention for query rows (positions kv_len - ntok + row)."""
    ntok = q.shape[0]
    kk = k_nhd[pages].reshape(-1, HKV, D)[:kv_len].float() * k_scale
    vv = v_nhd[pages].reshape(-1, HKV, D)[:kv_len].float() * v_scale
    kk, vv = kk.repeat_interleave(HQ // HKV, 1), vv.repeat_interleave(HQ // HKV, 1)
    out = []
    for r in rows:
        end = kv_len - ntok + r + 1
        s = torch.einsum("hd,lhd->hl", q[r].float(), kk[:end]) * D ** -0.5
        out.append(torch.einsum("hl,lhd->hd", torch.softmax(s, -1), vv[:end]))
    return torch.stack(out)


def plan_run(wrapper, qo, kv_indptr, indices, last, q, kv, k_scale, v_scale, use_hook=True):
    fiattn8.bypass = not use_hook
    try:
        wrapper.plan(torch.tensor(qo, dtype=torch.int32), torch.tensor(kv_indptr, dtype=torch.int32), indices,
                     torch.tensor(last, dtype=torch.int32), HQ, HKV, D, PAGE, causal=True, q_data_type=torch.float16,
                     kv_data_type=torch.float8_e4m3fn)
    finally:
        fiattn8.bypass = False
    return wrapper.run(q, kv, k_scale=k_scale, v_scale=v_scale)


def check(name, reqs, layout="NHD", k_scale=1.0, v_scale=1.0):
    """reqs: [(ntok, kv_len)]"""
    n_pages = sum((kv + PAGE - 1) // PAGE for _, kv in reqs) + 3
    raw, k, v = make_cache(n_pages, layout)
    k_nhd, v_nhd = raw.split(D, dim=-1)
    perm = torch.randperm(n_pages, device=dev).to(torch.int32)
    qo, kvi, last, idx, p = [0], [0], [], [], 0
    for ntok, kv in reqs:
        npg = (kv + PAGE - 1) // PAGE
        idx.append(perm[p:p + npg]); p += npg
        qo.append(qo[-1] + ntok); kvi.append(kvi[-1] + npg); last.append(kv - (npg - 1) * PAGE)
    indices = torch.cat(idx)
    q = torch.randn(qo[-1], HQ, D, device=dev, dtype=torch.float16)
    w = W(ws, layout)
    ours = plan_run(w, qo, kvi, indices, last, q, (k, v), k_scale, v_scale).clone()
    fi = plan_run(w, qo, kvi, indices, last, q, (k, v), k_scale, v_scale, use_hook=False)
    e_fi, e_ours, worst = 0.0, 0.0, 0.0
    for r, (ntok, kv) in enumerate(reqs):
        rows = sorted(set(torch.linspace(0, ntok - 1, 12).long().tolist() + [0, ntok - 1]))
        ref = reference(q[qo[r]:qo[r + 1]], k_nhd, v_nhd, indices[kvi[r]:kvi[r + 1]].long(), kv, rows, k_scale,
                        v_scale)
        a = ours[qo[r]:qo[r + 1]][rows].float()
        b = fi[qo[r]:qo[r + 1]][rows].float()
        e_ours = max(e_ours, (a - ref).abs().max().item())
        e_fi = max(e_fi, (b - ref).abs().max().item())
        worst = max(worst, (ours[qo[r]:qo[r + 1]].float() - fi[qo[r]:qo[r + 1]].float()).abs().max().item())
    ok = e_ours < 3 * e_fi + 2e-2 and torch.isfinite(ours).all().item()
    print(f"{name:44s} max err vs fp32: int8 {e_ours:.2e}  FlashInfer {e_fi:.2e} | int8 vs FlashInfer all rows "
          f"{worst:.2e}  {'ok' if ok else 'BAD'}", flush=True)
    return ok


ok = True
ok &= check("1 req, 6720 tokens from scratch", [(6720, 6720)])
ok &= check("1 req, 300 tokens from scratch", [(300, 300)])
ok &= check("1 req, 6720 at depth 13440 (2 chunks)", [(6720, 20160)])
ok &= check("1 req, 2000 at depth 70000 (5 chunks)", [(2000, 72000)])
ok &= check("1 req, 5000 at depth 16000 (QB split)", [(5000, 21000)])
ok &= check("2 reqs, 3360 at 0 + 1500 at 30000", [(3360, 3360), (1500, 31500)])
ok &= check("HND views, 1 req 4000 at depth 9000", [(4000, 13000)], layout="HND")
ok &= check("k/v scales 0.5 / 2.0", [(1000, 5000)], k_scale=0.5, v_scale=2.0)
print("correctness", "ok" if ok else "FAILED", flush=True)

# FlashInfer keeps the small prefills (< MIN_TOKENS rows): its split-KV temp at depth must fit the shared workspace
small_ws = torch.empty(192 << 20, dtype=torch.uint8, device=dev)
for ntok in (8, 64, fiattn8.MIN_TOKENS - 1):
    kv = 250000 + ntok
    npg = (kv + PAGE - 1) // PAGE
    raw, k, v = make_cache(npg, "NHD")
    q = torch.randn(ntok, HQ, D, device=dev, dtype=torch.float16)
    o = plan_run(W(small_ws, "NHD"), [0, ntok], [0, npg], torch.arange(npg, dtype=torch.int32, device=dev),
                 [kv - (npg - 1) * PAGE], q, (k, v), 1.0, 1.0, use_hook=False)
    torch.cuda.synchronize()
    print(f"FlashInfer {ntok:3d}-token prefill at depth 250000 with a 192 MiB workspace: ok "
          f"(finite {torch.isfinite(o).all().item()})", flush=True)
    del raw, k, v
    torch.cuda.empty_cache()

QLEN = 6720
for depth in [int(a) for a in sys.argv[1:]] or [0, 32768, 131072]:
    kv = depth + QLEN
    npg = (kv + PAGE - 1) // PAGE
    raw, k, v = make_cache(npg, "NHD")
    indices = torch.arange(npg, dtype=torch.int32, device=dev)
    q = torch.randn(QLEN, HQ, D, device=dev, dtype=torch.float16)
    flops = 4 * HQ * D * QLEN * (depth + QLEN / 2)
    res = []
    for hook in (False, True):
        w = W(ws, "NHD")
        plan_run(w, [0, QLEN], [0, npg], indices, [kv - (npg - 1) * PAGE], q, (k, v), 1.0, 1.0, use_hook=hook)
        torch.cuda.synchronize()
        it = 3 if depth > 60000 else 6
        t = time.time()
        for _ in range(it):
            w.run(q, (k, v), k_scale=1.0, v_scale=1.0)
        torch.cuda.synchronize()
        res.append((time.time() - t) / it * 1e3)
    print(f"6720-token chunk at depth {depth:6d}: FlashInfer {res[0]:8.2f} ms ({flops / res[0] / 1e9:5.1f} TFLOPS) | "
          f"int8 {res[1]:8.2f} ms ({flops / res[1] / 1e9:5.1f} TFLOPS) | x{res[0] / res[1]:.2f}", flush=True)
    del raw, k, v
    torch.cuda.empty_cache()
