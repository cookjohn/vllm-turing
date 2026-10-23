#!/usr/bin/env python3
"""Per-tensor int8 per-channel quantization error of a BF16 checkpoint (same quantizer as w8a8_convert.py): lists the
tensors above a relative-error threshold, with the outlier ratio (max |w| / rms) of their worst rows.
usage: w8a8_errs.py <src dir> [threshold, default 0.03] [cuda device]"""
import json
import os
import re
import sys

import torch
from safetensors import safe_open

src = sys.argv[1]
thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.03
dev = sys.argv[3] if len(sys.argv) > 3 else "cuda:0"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
QUANT = re.compile(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "w8a8_convert.py")).read()
                   .split('QUANT = re.compile(r"')[1].split('")\n')[0].replace('"\n                   r"', ""))
index = json.load(open(os.path.join(src, "model.safetensors.index.json")))
rows = []
for shard in sorted(set(index["weight_map"].values())):
    with safe_open(os.path.join(src, shard), framework="pt") as f:
        for name in f.keys():
            if not QUANT.match(name):
                continue
            w = f.get_tensor(name).to(dev, torch.float32)
            scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 127.0
            dq = torch.round(w / scale).clamp(-128, 127) * scale
            e = ((dq - w).norm() / w.norm()).item()
            ratio = (w.abs().amax(dim=1) / w.pow(2).mean(dim=1).sqrt().clamp(min=1e-12)).max().item()
            rows.append((e, name, tuple(w.shape), ratio))
rows.sort(reverse=True)
for e, name, shape, ratio in rows:
    if e >= thr:
        print(f"{e:.4f}  {name}  {shape}  max/rms {ratio:.0f}")
print(f"{sum(1 for r in rows if r[0] >= thr)} of {len(rows)} tensors >= {thr}")
