#!/usr/bin/env python3
"""Compare the greedy texts two vllm_bench.py decode runs saved with BENCH_TEXT (speculative decoding must not change
greedy output): per (round, prompt) identical or the first differing character with context.
usage: bench_text_diff.py a.jsonl b.jsonl"""
import json
import sys


def load(p):
    return {(d["round"], d["prompt"]): d["text"] for d in map(json.loads, open(p, encoding="utf-8"))}


a, b = load(sys.argv[1]), load(sys.argv[2])
same = 0
for key in sorted(a):
    x, y = a[key], b.get(key)
    if x == y:
        same += 1
        continue
    if y is None:
        print(key, "missing in", sys.argv[2])
        continue
    i = next((j for j, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
    print(key, f"DIFFER at char {i} of {len(x)}/{len(y)}: {x[max(0, i - 30):i + 30]!r}  vs  {y[max(0, i - 30):i + 30]!r}")
print(f"{same} of {len(a)} identical")
