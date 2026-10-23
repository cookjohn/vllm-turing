#!/usr/bin/env python3
"""Concurrency benchmark for the vLLM server (OpenAI chat API, streaming).
  streams N [N ...]       N generations at once (code / Chinese / mixed prompts, each with a unique first line): per
                          stream decode tok/s, aggregate output tok/s, longest time to first token
  agents N TURNS CTX_K    N agent-like sessions at once, each with its own ~CTX_K thousand-token code context: every
                          turn appends the previous answer and a new question to the session's history and asks for up to
                          GEN tokens, then pauses 1-3 s (tool time). Per turn: wait for the first token, prompt tokens
                          cached (prefix cache hits; a drop = the session's cache was evicted), decode tok/s; overall
                          aggregate output tok/s and wall time
usage: vllm_conc_bench.py <base url> streams 1 2 4 6 | agents 4 6 30   (env GEN = tokens per turn, default 200;
       THINK=1 thinking on)"""
import glob
import json
import os
import random
import statistics
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

KEY = open(os.path.expanduser("~/models/api-key.txt")).read().strip()
BASE = sys.argv[1].rstrip("/")
URL = BASE + "/v1/chat/completions"
_req = urllib.request.Request(BASE + "/v1/models", headers={"Authorization": "Bearer " + KEY})
MODEL = json.load(urllib.request.urlopen(_req, timeout=60))["data"][0]["id"]
GEN = int(os.environ.get("GEN", "200"))
THINK = os.environ.get("THINK") == "1"
PROMPTS = [
    ("code", "Write a Python module implementing an LRU cache class with type hints, docstrings and thread safety, then "
             "8 pytest unit tests for it. Output only code.", 500),
    ("zh", "用中文详细解释 TCP 拥塞控制的慢启动、拥塞避免、快速重传和快速恢复，每个部分写一段，并举一个具体的数值例子。", 500),
    ("mixed", "先用两段中文解释二叉堆的原理和复杂度，然后给出带类型注解的 Python 实现（push/pop/heapify）和 4 个 pytest "
              "测试。", 500),
]


def chat(messages, max_tokens):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0.6, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": THINK}}
    req = urllib.request.Request(URL, json.dumps(body, ensure_ascii=False).encode(),
                                 {"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    t0, t_first, t_last, text, usage = time.time(), None, None, [], {}
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices", []):
                delta = ch.get("delta") or {}
                piece = (delta.get("content") or "") + (delta.get("reasoning_content") or delta.get("reasoning") or "")
                if piece:
                    t_last = time.time()
                    t_first = t_first or t_last
                    text.append(piece)
    n = usage.get("completion_tokens", 0)
    return {"prompt": usage.get("prompt_tokens", 0),
            "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0, "gen": n,
            "ttft": (t_first or time.time()) - t0,
            "tps": (n - 1) / (t_last - t_first) if t_first and t_last > t_first and n > 1 else 0.0,
            "wall": time.time() - t0, "text": "".join(text)}


def streams(n):
    def one(i):
        name, p, mt = PROMPTS[i % len(PROMPTS)]
        return chat([{"role": "user", "content": f"[stream {i} {time.time_ns()}]\n{p}"}], mt)
    t0 = time.time()
    with ThreadPoolExecutor(n) as ex:
        rs = list(ex.map(one, range(n)))
    wall = time.time() - t0
    gen = sum(r["gen"] for r in rs)
    print(f"streams {n}: per stream {statistics.mean(r['tps'] for r in rs):6.1f} tok/s "
          f"(min {min(r['tps'] for r in rs):.1f}), aggregate {gen / wall:6.1f} tok/s, "
          f"first token max {max(r['ttft'] for r in rs):.2f} s, {gen} tok in {wall:.1f} s", flush=True)


def corpus():
    src = sorted(glob.glob(os.path.expanduser("~/llama.cpp/src/*.cpp"))) + \
        sorted(glob.glob(os.path.expanduser("~/llama.cpp/src/models/*.cpp")))
    return "\n".join(f"// ===== file: {p.split('/')[-1]} =====\n" + open(p, errors="ignore").read() for p in src)


def agents(n, turns, ctx_k):
    text = corpus()
    chars = int(ctx_k * 1000 * 3.4)
    lock = threading.Lock()
    rows = []

    def session(i):
        rnd = random.Random(i)
        off = (i * 997_331) % max(1, len(text) - chars)
        doc = text[off:off + chars]
        msgs = [{"role": "system", "content": f"You are a code reviewer. Session {i} {time.time_ns()}."},
                {"role": "user", "content": f"<code>\n{doc}\n</code>\n\nList the three most important functions "
                                            f"in this code and say in one sentence what each does."}]
        for t in range(turns):
            r = chat(msgs, GEN)
            with lock:
                rows.append((i, t, r))
                print(f"  session {i} turn {t}: prompt {r['prompt']:6d} cached {r['cached']:6d} | first token "
                      f"{r['ttft']:6.2f} s | {r['gen']:4d} tok @ {r['tps']:5.1f} tok/s", flush=True)
            msgs = msgs + [{"role": "assistant", "content": r["text"]},
                           {"role": "user", "content": f"Now look at another part of the code (#{t + 1}): pick one "
                                                       f"more function and explain its error handling in 3 sentences."}]
            time.sleep(rnd.uniform(1, 3))

    t0 = time.time()
    with ThreadPoolExecutor(n) as ex:
        list(ex.map(session, range(n)))
    wall = time.time() - t0
    later = [r for _, t, r in rows if t > 0]
    gen = sum(r["gen"] for _, _, r in rows)
    miss = [r for r in later if r["cached"] < 0.8 * r["prompt"]]
    print(f"agents {n} x {turns} turns, ~{ctx_k}k context: first turn first token mean "
          f"{statistics.mean(r['ttft'] for _, t, r in rows if t == 0):.1f} s; later turns first token median "
          f"{statistics.median(r['ttft'] for r in later):.2f} s, max {max(r['ttft'] for r in later):.2f} s, "
          f"cache misses {len(miss)}/{len(later)}; decode per stream {statistics.mean(r['tps'] for _, _, r in rows):.1f} "
          f"tok/s; aggregate {gen / wall:.1f} tok/s over {wall:.0f} s", flush=True)


mode = sys.argv[2]
if mode == "streams":
    for k in sys.argv[3:]:
        streams(int(k))
elif mode == "agents":
    agents(int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]))
else:
    sys.exit(__doc__)
