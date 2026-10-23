#!/usr/bin/env python3
"""Summarize a PyTorch allocator snapshot dumped by vllm_turing.memdbg (VLLM_TURING_MEMDBG=2) on CUDA OOM: live
(active_allocated) blocks grouped by allocation site (innermost frame in vllm / vllm_turing / flashinfer / fla /
model code, or by size when no frames were recorded), largest first.
usage: oom_snapshot_summary.py /tmp/oom_snapshot_<pid>.pickle [top]"""
import pickle
import sys

snap = pickle.load(open(sys.argv[1], "rb"))
top = int(sys.argv[2]) if len(sys.argv) > 2 else 30
groups, total = {}, 0
for seg in snap.get("segments", []):
    for blk in seg.get("blocks", []):
        if blk.get("state") != "active_allocated":
            continue
        size = blk.get("size", 0)
        total += size
        frames = blk.get("frames") or []
        site = None
        for f in frames:
            fn = f.get("filename", "")
            if any(s in fn for s in ("vllm_turing", "/vllm/", "flashinfer", "flash_linear", "inductor_cache")):
                site = f"{fn.split('site-packages/')[-1]}:{f.get('line')} {f.get('name')}"
                break
        if site is None:
            site = f"<no frames> {size / 2**20:.1f} MiB" if not frames else f"{frames[0].get('filename', '?')}:" \
                   f"{frames[0].get('line')} {frames[0].get('name')}"
        g = groups.setdefault(site, [0, 0])
        g[0] += size
        g[1] += 1
print(f"live: {total / 2**30:.3f} GiB in {sum(g[1] for g in groups.values())} blocks")
for site, (size, n) in sorted(groups.items(), key=lambda kv: -kv[1][0])[:top]:
    print(f"{size / 2**20:9.1f} MiB {n:5d}  {site}")
