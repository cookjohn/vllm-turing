#!/usr/bin/env python3
"""MTP draft acceptance on a fixed workload: N sequential greedy requests (code, Chinese, English prompts, natural end or
MAX tokens), acceptance from the server's spec-decode counters (/metrics deltas): mean accepted drafts per verify step,
mean tokens per step, decode tok/s. For comparing serving modes on the same prompts.
usage: spec_accept.py <base url> [max_tokens=700]   (vLLM env python)"""
import os
import json
import re
import sys
import time
import urllib.request

KEY = open(os.path.expanduser("~/models/api-key.txt")).read().strip()
BASE = sys.argv[1].rstrip("/")
MAX = int(sys.argv[2]) if len(sys.argv) > 2 else 700
PROMPTS = [
    "Write a Python module implementing an interval tree with insert, delete and overlap queries, with tests.",
    "用中文详细介绍 Raft 共识算法的领导者选举、日志复制和安全性保证。",
    "Explain how a modern CPU's out-of-order execution engine works, step by step.",
    "Write a Rust function that parses a CSV line with quoted fields, plus unit tests.",
    "写一个 JavaScript 的防抖和节流函数，并解释它们的区别和使用场景。",
    "Describe the main causes and consequences of the 2008 financial crisis.",
]


def metrics():
    req = urllib.request.Request(BASE + "/metrics", headers={"Authorization": "Bearer " + KEY})
    text = urllib.request.urlopen(req, timeout=30).read().decode()
    out = {}
    for name in ("spec_decode_num_drafts", "spec_decode_num_draft_tokens", "spec_decode_num_accepted_tokens",
                 "generation_tokens"):
        m = re.findall(r"^vllm:%s_total\{[^}]*\} ([0-9.e+]+)$" % name, text, re.M)
        out[name] = sum(float(x) for x in m)
    return out


model = json.load(urllib.request.urlopen(urllib.request.Request(BASE + "/v1/models", headers={
    "Authorization": "Bearer " + KEY}), timeout=60))["data"][0]["id"]
m0, t0, gen = metrics(), time.time(), 0
for p in PROMPTS:
    body = {"model": model, "messages": [{"role": "user", "content": p}], "max_tokens": MAX, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    r = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body, ensure_ascii=False).encode(),
                               {"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    gen += json.load(urllib.request.urlopen(r, timeout=600))["usage"]["completion_tokens"]
wall = time.time() - t0
m1 = metrics()
d = {k: m1[k] - m0[k] for k in m0}
steps = d["spec_decode_num_drafts"] or 1
print(f"{len(PROMPTS)} requests, {gen} tokens in {wall:.1f} s ({gen / wall:.1f} tok/s incl. prefill); verify steps "
      f"{steps:.0f}, accepted drafts per step {d['spec_decode_num_accepted_tokens'] / steps:.3f} of "
      f"{d['spec_decode_num_draft_tokens'] / steps:.1f}, tokens per step {gen / steps:.3f}")
