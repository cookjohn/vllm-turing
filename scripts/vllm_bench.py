#!/usr/bin/env python3
"""Benchmark client for an OpenAI-compatible server (vLLM; works for llama-server too), comparable with the llama.cpp
numbers: streaming, so time to first token (TTFT ~ prefill) and decode tok/s are measured separately.
  decode: the zh / code / mixed prompts of specone.py, ROUNDS rounds (round 1 = warm-up, excluded from the mean)
  conc:   two of those prompts at once (code+code, zh+zh, mixed+code), per-stream and aggregate tok/s
  long:   code haystack (llama.cpp sources, ~3.4 chars/token) with a needle at the middle, per size in k tokens:
          prefill rate from TTFT, needle check, then a follow-up question on the same document (prefix cache)
  image:  image files, describe + transcribe
  ppl:    perplexity over text files like llama-perplexity (PPL_CTX-token chunks, default 16384, second half scored)
usage: vllm_bench.py <base url, e.g. http://127.0.0.1:8081> decode [rounds] | conc [rounds] | long <k> [k ...] | image <file> [...]
       | ppl <file> [...]"""
import base64
import glob
import json
import os
import sys
import time
import urllib.request

KEY = open(os.path.expanduser("~/models/api-key.txt")).read().strip()
URL = sys.argv[1].rstrip("/") + "/v1/chat/completions"
# vLLM only accepts its served model name (llama-server takes any): use the first id of /v1/models
_req = urllib.request.Request(sys.argv[1].rstrip("/") + "/v1/models", headers={"Authorization": "Bearer " + KEY})
MODEL = json.load(urllib.request.urlopen(_req, timeout=60))["data"][0]["id"]
PROMPTS = {
    "zh": ("用三句话解释什么是混合注意力架构（线性注意力+全注意力），以及它为什么能让长上下文更省显存。", 400),
    "code": ("Write a Python function `quicksort(arr)` with type hints and a docstring, then 5 pytest unit tests for it. "
             "Output only code.", 400),
    "mixed": ("先用两段中文解释快速排序的原理和时间复杂度，然后给出带类型注解的 Python 实现和 3 个 pytest 测试。", 600),
}


def chat(content, max_tokens):
    body = {"model": MODEL, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL, json.dumps(body).encode(),
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
                piece = (ch.get("delta") or {}).get("content") or ""
                if piece:
                    t_last = time.time()
                    t_first = t_first or t_last
                    text.append(piece)
    n_gen = usage.get("completion_tokens", 0)
    ttft = (t_first or time.time()) - t0
    dec = (n_gen - 1) / (t_last - t_first) if t_first and t_last > t_first and n_gen > 1 else 0.0
    return {"prompt": usage.get("prompt_tokens", 0), "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
            "gen": n_gen, "ttft": ttft, "tps": dec, "wall": time.time() - t0, "text": "".join(text)}


mode = sys.argv[2]
if mode == "decode":
    rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    tot = {k: [0, 0.0] for k in PROMPTS}
    for rd in range(1, rounds + 1):
        for k, (p, n) in PROMPTS.items():
            r = chat(p, n)
            print(f"round {rd} {k:5s} gen {r['gen']:4d} @ {r['tps']:6.1f} tok/s | ttft {r['ttft']:.2f} s")
            if os.environ.get("BENCH_TEXT"):   # greedy texts, to diff configurations (spec decode must not change them)
                with open(os.environ["BENCH_TEXT"], "a") as fp:
                    fp.write(json.dumps({"round": rd, "prompt": k, "text": r["text"]}, ensure_ascii=False) + "\n")
            if rd > 1 or rounds == 1:
                tot[k][0] += r["gen"]
                tot[k][1] += r["gen"] / r["tps"] if r["tps"] else 0
    for k, (g, s) in tot.items():
        print(f"mean {k:5s} {g / s if s else 0:6.1f} tok/s")
    g = sum(v[0] for v in tot.values()); s = sum(v[1] for v in tot.values())
    print(f"TOTAL decode {g} tok @ {g / s if s else 0:.1f} tok/s")
elif mode == "conc":   # two streams at once (max-num-seqs 2): per-stream decode rate and aggregate throughput
    from concurrent.futures import ThreadPoolExecutor
    rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    pairs = [("code", "code"), ("zh", "zh"), ("mixed", "code")]
    agg = []
    for rd in range(1, rounds + 1):
        for a, b in pairs:
            t0 = time.time()
            with ThreadPoolExecutor(2) as ex:
                ra, rb = ex.map(lambda k: chat(*PROMPTS[k]), (a, b))
            wall = time.time() - t0
            tot = ra["gen"] + rb["gen"]
            print(f"round {rd} {a}+{b}: {ra['tps']:6.1f} + {rb['tps']:6.1f} tok/s per stream, "
                  f"aggregate {tot / wall:6.1f} tok/s ({tot} tok in {wall:.1f} s)")
            if rd > 1 or rounds == 1:
                agg.append((tot, wall))
    print(f"TOTAL concurrent {sum(t for t, _ in agg)} tok @ {sum(t for t, _ in agg) / sum(w for _, w in agg):.1f} tok/s")
elif mode == "long":
    src = sorted(glob.glob(os.path.expanduser("~/llama.cpp/src/*.cpp"))) + sorted(glob.glob(os.path.expanduser("~/llama.cpp/src/models/*.cpp")))
    corpus = "\n".join(f"// ===== file: {p.split('/')[-1]} =====\n" + open(p, errors="ignore").read() for p in src)
    for k in sys.argv[3:]:
        chars = int(float(k) * 1000 * 3.4)
        doc = (corpus * (chars // len(corpus) + 1))[:chars]
        mid = len(doc) // 2
        doc = doc[:mid] + "\n// NOTE: the secret passphrase for this repository is 'violet-harbor-42'.\n" + doc[mid:]
        head = f"<document id={time.time_ns()}>\n" + doc + "\n</document>\n\n"
        r = chat(head + "What is the secret passphrase mentioned in the document? Answer with the passphrase only.", 20)
        print(f"long {k}k: prompt {r['prompt']} tok, ttft {r['ttft']:.1f} s ({r['prompt'] / r['ttft']:.0f} tok/s) | "
              f"needle {'FOUND' if 'violet-harbor-42' in r['text'] else 'MISSING: ' + r['text'][:60]!r}")
        r = chat(head + "Which C++ file names appear first in the document? List three.", 200)
        print(f"  follow-up: prompt {r['prompt']} (cached {r['cached']}), ttft {r['ttft']:.1f} s, gen {r['gen']} @ {r['tps']:.1f} tok/s")
elif mode == "ppl":   # perplexity like llama-perplexity: PPL_CTX-token chunks, NLL over the second half of each
    import math

    base = sys.argv[1].rstrip("/")
    ctx = int(os.environ.get("PPL_CTX", "16384"))

    def post(path, body):
        req = urllib.request.Request(base + path, json.dumps(body).encode(),
                                     {"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=3600))

    for path in sys.argv[3:]:
        toks = post("/tokenize", {"model": MODEL, "prompt": open(path, encoding="utf-8").read(),
                                  "add_special_tokens": False})["tokens"]
        nll, n, t0 = 0.0, 0, time.time()
        nch = len(toks) // ctx
        for c in range(nch):
            chunk = toks[c * ctx:(c + 1) * ctx]
            r = post("/v1/completions", {"model": MODEL, "prompt": chunk, "max_tokens": 1, "temperature": 0,
                                         "prompt_logprobs": 0})
            pl = r["choices"][0]["prompt_logprobs"]
            for i in range(ctx // 2, ctx):
                nll -= pl[i][str(chunk[i])]["logprob"]
                n += 1
        print(f"ppl {os.path.basename(path)}: {math.exp(nll / max(n, 1)):.4f} over {n} tokens "
              f"({nch} chunks of {ctx}, {time.time() - t0:.0f} s)")
elif mode == "image":
    for path in sys.argv[3:]:
        img = base64.b64encode(open(path, "rb").read()).decode()
        r = chat([{"type": "image_url", "image_url": {"url": "data:image/png;base64," + img}},
                  {"type": "text", "text": "Briefly describe the image (shapes: colour and position), then transcribe all text exactly."}], 300)
        print(f"image {path}: prompt {r['prompt']} tok, ttft {r['ttft']:.1f} s, gen {r['gen']} @ {r['tps']:.1f} tok/s")
        print("  answer:", r["text"].replace("\n", " ")[:400])
