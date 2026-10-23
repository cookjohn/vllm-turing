#!/usr/bin/env python3
"""4-bit pruned draft head (vllm_turing.draftvocab q4_pack + csrc/dhead_q4.cu) on real Qwen3.8-27B lm_head rows (one
rank's half of the 65536 ranked draft ids): the kernel's argmax vs the argmax of the same dequantized 4-bit matrix
(exactness of the kernel), vs the fp16 head and vs the int8 head (what 4 bits cost in draft agreement), on hidden
states drawn like real ones; and kernel speed vs the int8 row-subset GEMV.
usage: test_dhead.py [checkpoint dir] [ranked id file]   (vLLM env python, one GPU)"""
import json
import os
import sys
import time

import torch
from safetensors import safe_open

from vllm_turing import draftvocab, lmhead

ck = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/models/Qwen3.8-27B-SQ06")
rk = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/models/draft_vocab_ranked.txt")
wm = json.load(open(os.path.join(ck, "model.safetensors.index.json")))["weight_map"]
ids = sorted(set(int(t) for t in open(rk).read().split()[:65536]))
ids = torch.tensor(ids[len(ids) // 2:])          # the upper half: what rank 1 drafts with
with safe_open(os.path.join(ck, wm["lm_head.weight"]), "pt") as f:
    W = f.get_tensor("lm_head.weight")[ids].to("cuda", torch.float16)
with safe_open(os.path.join(ck, wm["model.language_model.norm.weight"]), "pt") as f:
    g = f.get_tensor("model.language_model.norm.weight").to("cuda", torch.float32)
N, K = W.shape
print(f"{N} rows x {K}")
q8, s8 = lmhead.quantize_rows(W)
W8 = q8.float() * s8[:, None]
wq, d = draftvocab.q4_pack(W8)

# reference dequant of the packed layout
words = wq.view(torch.int32).view(N, K // 32, 4)
sh = (4 * torch.arange(4, device="cuda", dtype=torch.int32)).view(1, 1, 1, 4)
ev = (words.unsqueeze(-1) >> sh) & 15
od = (words.unsqueeze(-1) >> (sh + 16)) & 15
qq = torch.stack([ev, od], dim=-1).view(N, K // 32, 32).float() - 8
W4 = (qq * d.float().unsqueeze(-1)).view(N, K)
print(f"4-bit weight rel err vs fp16: {((W4 - W.float()).norm() / W.float().norm()).item():.4f}, "
      f"int8: {((W8 - W.float()).norm() / W.float().norm()).item():.4f}")

gids = ids.to("cuda", torch.int64)
lib = draftvocab.qlib()
nb = max(lib.dhead_blocks(m, N) for m in (1, 3))
pval = torch.empty(8 * nb, dtype=torch.float32, device="cuda")
pidx = torch.empty(8 * nb, dtype=torch.int32, device="cuda")


def kern(x):
    M = x.shape[0]
    ex = torch.empty(M, 1, 4, dtype=torch.float16, device="cuda")
    rc = lib.dhead_argmax(x.data_ptr(), wq.data_ptr(), d.data_ptr(), gids.data_ptr(), pval.data_ptr(), pidx.data_ptr(),
                          ex.data_ptr(), M, N, K, 0, 1, torch.cuda.current_stream().cuda_stream)
    assert rc == 0, rc
    return ex[:, 0, 1].long() * 2048 + ex[:, 0, 2].long()


torch.manual_seed(0)
for M in (1, 2, 4, 8):
    h = torch.randn(256 * M, K, device="cuda")
    h = (h / h.pow(2).mean(-1, keepdim=True).sqrt() * (1 + g)).half()
    got = torch.cat([kern(h[i:i + M].contiguous()) for i in range(0, h.shape[0], M)])
    a4 = gids[(h.float() @ W4.t()).argmax(-1)]
    a8 = gids[(h.float() @ W8.t()).argmax(-1)]
    a16 = gids[(h.float() @ W.float().t()).argmax(-1)]
    print(f"M={M}: kernel == 4-bit ref {(got == a4).float().mean().item() * 100:.2f}%, "
          f"4-bit vs fp16 {(a4 == a16).float().mean().item() * 100:.2f}%, int8 vs fp16 "
          f"{(a8 == a16).float().mean().item() * 100:.2f}%, kernel vs fp16 {(got == a16).float().mean().item() * 100:.2f}%")


def bench(fn, it=200):
    fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(it):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / it * 1e3


import ctypes   # noqa: E402

lmhead.lib().w8a16_gemv_rows.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 4 + [ctypes.c_void_p, ctypes.c_void_p]
rows = torch.arange(N, dtype=torch.int32, device="cuda")
nbytes = wq.numel() + 2 * d.numel()
for M in (1, 2, 4):
    x = torch.randn(M, K, device="cuda", dtype=torch.float16)
    out = torch.empty(M, N, dtype=torch.float16, device="cuda")
    t4 = bench(lambda: kern(x))
    t8 = bench(lambda: lmhead.lib().w8a16_gemv_rows(x.data_ptr(), q8.data_ptr(), s8.data_ptr(), None, out.data_ptr(),
                                                     M, N, K, 0, rows.data_ptr(), torch.cuda.current_stream().cuda_stream))
    print(f"M={M}: 4-bit argmax {t4 * 1e3:.0f} us ({nbytes / t4 / 1e6:.0f} GB/s), int8 rows GEMV only "
          f"{t8 * 1e3:.0f} us ({q8.numel() / t8 / 1e6:.0f} GB/s)")

# 4-bit logits kernel (dhead_logits, DFlash2 candidate head): exactness vs the dequantized matrix, top-16 overlap with the
# fp16 head, and speed vs the int8 head over the full 124k-row shard (what the candidate head read before)
lib.dhead_logits.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
lib.dhead_logits.restype = ctypes.c_int


def klog(x):
    out = torch.empty(x.shape[0], N, dtype=torch.float16, device="cuda")
    assert lib.dhead_logits(x.data_ptr(), wq.data_ptr(), d.data_ptr(), out.data_ptr(), x.shape[0], N, K,
                            torch.cuda.current_stream().cuda_stream) == 0
    return out


for M in (1, 7, 14, 16):
    h = torch.randn(M, K, device="cuda")
    h = (h / h.pow(2).mean(-1, keepdim=True).sqrt() * (1 + g)).half()
    got = klog(h).float()
    ref = h.float() @ W4.t()
    r16 = h.float() @ W.float().t()
    rel = ((got - ref).norm() / ref.norm()).item()
    ov = sum(len(set(a.tolist()) & set(b.tolist())) for a, b in zip(got.topk(16).indices, r16.topk(16).indices)) / (16 * M)
    print(f"logits M={M}: rel err vs 4-bit ref {rel:.1e}, top-16 overlap with fp16 head {ov * 100:.1f}%")
full = torch.randint(-127, 127, (124160, K), dtype=torch.int8, device="cuda")
fs = torch.rand(124160, device="cuda") * 1e-3
for M in (7, 14):
    x = torch.randn(M, K, device="cuda", dtype=torch.float16)
    tl = bench(lambda: klog(x))
    tf = bench(lambda: torch.ops.vllm_turing.w8a16_mm(x, full, fs, None))
    print(f"M={M}: 4-bit pruned logits {tl * 1e3:.0f} us ({nbytes / tl / 1e6:.0f} GB/s), int8 full shard "
          f"{tf * 1e3:.0f} us ({full.numel() / tf / 1e6:.0f} GB/s)")
