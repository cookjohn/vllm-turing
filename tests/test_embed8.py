#!/usr/bin/env python3
"""vllm_turing embed8 (csrc/embed8.cu via torch.ops.vllm_turing.embed8): one-kernel int8 vocab-parallel lookup vs the
torch reference (vLLM's get_masked_input_and_mask + dequantized rows + zeroed foreign rows), int32 / int64 ids, a
shard with an added-token range, out-of-shard ids; and launch time for decode-sized batches.
usage: test_embed8.py   (vLLM env python, one GPU)"""
import time

import torch

from vllm_turing import embed8  # noqa: F401  (registers the op)

dev = "cuda"
torch.manual_seed(0)
V, D = 124160, 5120
org_start, org_end, pad, added_start, added_end = 124160, 248320, 0, 248320, 248400   # rank 1 of 2 (+80 added)
rows = (org_end - org_start) + pad + (added_end - added_start)
w = torch.randn(rows, D, device=dev) * 0.02
sc = w.abs().amax(1).clamp(min=1e-12) / 127
q = torch.round(w / sc[:, None]).clamp(-127, 127).to(torch.int8)
deq = (q.float() * sc[:, None]).half()
added_offset = added_start - (org_end - org_start) - pad


def ref(ids):
    om = (ids >= org_start) & (ids < org_end)
    am = (ids >= added_start) & (ids < added_end)
    off = org_start * om + (added_start - (org_end - org_start) - pad) * am
    m = om | am
    local = m * (ids - off)
    out = deq[local.long()]
    out[~m] = 0
    return out


worst = 0.0
for dt in (torch.int32, torch.int64):
    ids = torch.cat([torch.randint(0, 248400, (300,)), torch.tensor([0, 124159, 124160, 248319, 248320, 248399])]).to(dev, dt)
    got = torch.ops.vllm_turing.embed8(ids, q, sc, org_start, org_end, added_start, added_end, added_offset)
    r = ref(ids.long())
    worst = max(worst, (got.float() - r.float()).abs().max().item())
    ids2 = ids.view(2, -1)
    got2 = torch.ops.vllm_turing.embed8(ids2, q, sc, org_start, org_end, added_start, added_end, added_offset)
    assert got2.shape == (2, ids.numel() // 2, D)
print(f"embed8 vs reference: max abs diff {worst:.2e} (fp16 rounding only expected)")

ids = torch.randint(124160, 248320, (5,), device=dev)
for _ in range(10):
    torch.ops.vllm_turing.embed8(ids, q, sc, org_start, org_end, added_start, added_end, added_offset)
torch.cuda.synchronize()
t = time.time()
for _ in range(1000):
    torch.ops.vllm_turing.embed8(ids, q, sc, org_start, org_end, added_start, added_end, added_offset)
torch.cuda.synchronize()
print(f"5-token lookup: {(time.time() - t) * 1e3:.1f} us per call through the custom op")
t = time.time()
for _ in range(1000):
    embed8._embed8_call(ids, q, sc, org_start, org_end, added_start, added_end, added_offset)
torch.cuda.synchronize()
print(f"5-token lookup: {(time.time() - t) * 1e3:.1f} us per call direct (eager path)")
