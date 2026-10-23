#!/usr/bin/env python3
"""Calibration traffic for vllm_turing.calib: token chunks of the zh / code / en corpora (and a few chat prompts) sent as
prompts with a short generation, so every quantized linear layer (target and MTP draft) sees prefill and decode
activations. usage: calib_feed.py <base url> [chunk tokens, default 4096] [chunks per file, default 4]"""
import os
import json
import sys
import urllib.request

KEY = open(os.path.expanduser("~/models/api-key.txt")).read().strip()
base = sys.argv[1].rstrip("/")
ctx = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
per = int(sys.argv[3]) if len(sys.argv) > 3 else 4


def post(path, body):
    req = urllib.request.Request(base + path, json.dumps(body).encode(),
                                 {"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=3600))


model = json.load(urllib.request.urlopen(urllib.request.Request(base + "/v1/models",
                                                                headers={"Authorization": "Bearer " + KEY})))["data"][0]["id"]
n = 0
for f in ("zh", "code", "en"):
    toks = post("/tokenize", {"model": model, "prompt": open(f"/tmp/ppl/{f}.txt", encoding="utf-8").read(),
                              "add_special_tokens": False})["tokens"]
    step = max(ctx, (len(toks) - ctx) // max(per - 1, 1))
    for i in range(per):
        chunk = toks[i * step:i * step + ctx]
        if len(chunk) < ctx // 2:
            break
        post("/v1/completions", {"model": model, "prompt": chunk, "max_tokens": 32, "temperature": 0})
        n += len(chunk)
for p in ("用中文解释一下量子纠缠，并举一个生活中的类比。", "Write a Rust function that parses an ISO-8601 date string.",
          "Summarize the causes of the French Revolution in five bullet points."):
    post("/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": p}], "max_tokens": 128,
                                  "temperature": 0})
print(f"calibration traffic sent: {n} prompt tokens + chat prompts")
