#!/usr/bin/env python3
"""Functional acceptance tests of the production vLLM server (OpenAI API, production flags: qwen3 reasoning parser,
qwen3_coder tool calls) after the optimizations. Each test prints PASS / FAIL with its numbers:
  models, arithmetic, thinking (reasoning split), tool call + tool-result round trip, streaming inter-token latency,
  greedy determinism, multi-turn conversation over a 30K-token document (prefix cache: second-turn time to first
  token), a long prefill while another request streams (the streaming request's token gaps during the prefill),
  4096-token generation, 24 mixed requests on 2 workers, a needle at the end of a ~250K-token prompt (near the 262144
  maximum).
usage: vllm_acceptance.py <base url> [--skip-max] [--only name,name]   (names as in the run list below)"""
import os
import glob
import json
import sys
import threading
import time
import urllib.request

BASE = sys.argv[1].rstrip("/")
SKIP_MAX = "--skip-max" in sys.argv
ONLY = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else None
KEY = open(os.path.expanduser("~/models/api-key.txt")).read().strip()
H = {"Authorization": "Bearer " + KEY, "Content-Type": "application/json"}
MODEL = "qwen3.8-27b"
results = []

src = sorted(glob.glob(os.path.expanduser("~/llama.cpp/src/*.cpp")))
CORPUS = "\n".join(f"// ===== file: {p.split('/')[-1]} =====\n" + open(p, errors="ignore").read() for p in src)


def doc(ktok, fact="", at=0.5):
    chars = int(ktok * 1000 * 3.4)
    d = (CORPUS * (chars // len(CORPUS) + 1))[:chars]
    if fact:
        i = int(len(d) * at)
        d = d[:i] + f"\n// NOTE: {fact}\n" + d[i:]
    return f"<document id={time.time_ns()}>\n{d}\n</document>\n\n"


def chat(messages, max_tokens, thinking=False, tools=None, stream=False, extra=None):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": thinking}}
    if tools:
        body["tools"], body["tool_choice"] = tools, "auto"
    if extra:
        body.update(extra)
    if stream:
        body["stream"], body["stream_options"] = True, {"include_usage": True}
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), H)
    t0 = time.time()
    if not stream:
        r = json.load(urllib.request.urlopen(req, timeout=3600))
        m = r["choices"][0]["message"]
        return {"text": m.get("content") or "", "reasoning": m.get("reasoning_content") or m.get("reasoning") or "",
                "tool_calls": m.get("tool_calls") or [], "finish": r["choices"][0].get("finish_reason"),
                "usage": r.get("usage") or {}, "wall": time.time() - t0}
    times, text, usage = [], [], {}
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for line in resp:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices", []):
                piece = (ch.get("delta") or {}).get("content") or ""
                if piece:
                    times.append(time.time())
                    text.append(piece)
    return {"text": "".join(text), "times": times, "t0": t0, "usage": usage,
            "ttft": (times[0] - t0) if times else float("nan"), "wall": time.time() - t0}


def check(name, ok, detail):
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)
    results.append((name, ok))


def run(name, fn):
    try:
        fn()
    except Exception as e:   # noqa: BLE001
        check(name, False, f"exception {type(e).__name__}: {str(e)[:200]}")


def t_models():
    r = json.load(urllib.request.urlopen(urllib.request.Request(BASE + "/v1/models", headers=H), timeout=60))
    ids = [m["id"] for m in r["data"]]
    check("models", all(x in ids for x in ("qwen3.8-27b", "qwen", "q")), f"served names {ids}")


def t_arith():
    r = chat([{"role": "user", "content": "What is 17 * 23? Answer with the number only."}], 20)
    check("arithmetic", "391" in r["text"], repr(r["text"][:40]))


def t_thinking():
    r = chat([{"role": "user", "content": "How many prime numbers are there between 1 and 30? Reply with the count."}],
             2000, thinking=True)
    check("thinking (reasoning split)", len(r["reasoning"]) > 20 and "10" in r["text"] and "<think>" not in r["text"],
          f"reasoning {len(r['reasoning'])} chars, answer {r['text'].strip()[:60]!r}")


TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather of a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"},
                                                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
                   "required": ["city"]}}}]


def t_tools():
    msgs = [{"role": "user", "content": "What's the weather in Paris right now? Use celsius."}]
    r = chat(msgs, 300, tools=TOOLS)
    calls = r["tool_calls"]
    ok = bool(calls) and calls[0]["function"]["name"] == "get_weather"
    args = json.loads(calls[0]["function"]["arguments"]) if ok else {}
    ok = ok and "paris" in str(args.get("city", "")).lower()
    check("tool call", ok, f"finish {r['finish']}, call {calls[0]['function'] if calls else None}")
    if not ok:
        return
    msgs += [{"role": "assistant", "content": r["text"] or None, "tool_calls": calls},
             {"role": "tool", "tool_call_id": calls[0]["id"],
              "content": json.dumps({"temperature": 18, "unit": "celsius", "condition": "sunny"})}]
    r2 = chat(msgs, 200, tools=TOOLS)
    check("tool result round trip", "18" in r2["text"], repr(r2["text"][:100]))


def t_stream():
    r = chat([{"role": "user", "content": "Write a Python function that parses an ISO 8601 duration string into "
                                          "seconds, with docstring and 5 tests. Output only code."}], 400, stream=True)
    gaps = sorted(b - a for a, b in zip(r["times"], r["times"][1:]))
    n = r["usage"].get("completion_tokens", len(r["times"]))
    rate = (n - 1) / (r["times"][-1] - r["times"][0]) if len(r["times"]) > 1 else 0
    check("streaming", len(r["times"]) > 50,
          f"ttft {r['ttft']:.2f} s, {rate:.1f} tok/s, chunk gap p50 {1e3 * gaps[len(gaps) // 2]:.0f} ms / p99 "
          f"{1e3 * gaps[int(len(gaps) * 0.99)]:.0f} ms / max {1e3 * gaps[-1]:.0f} ms ({len(r['times'])} chunks)")


def t_determinism():
    m = [{"role": "user", "content": "Explain in about 150 words why the sky is blue."}]
    a, b = chat(m, 200)["text"], chat(m, 200)["text"]
    check("greedy determinism", a == b and len(a) > 200, f"identical {a == b}, {len(a)} chars")


def t_multiturn():
    d = doc(30, "the release codename of this build is 'amber-falcon-7'.", at=0.3)
    m = [{"role": "user", "content": d + "Summarize what this code base does in three sentences."}]
    r1 = chat(m, 200, stream=True)
    m += [{"role": "assistant", "content": r1["text"]},
          {"role": "user", "content": "What is the release codename mentioned in the document? Answer only with it."}]
    r2 = chat(m, 30, stream=True)
    p1, p2 = r1["usage"].get("prompt_tokens", 0), r2["usage"].get("prompt_tokens", 0)
    check("multi-turn over a 30K document", "amber-falcon-7" in r2["text"] and r2["ttft"] < r1["ttft"] / 3,
          f"turn 1 {p1} tok ttft {r1['ttft']:.1f} s; turn 2 {p2} tok ttft {r2['ttft']:.2f} s (prefix cache), "
          f"answer {r2['text'].strip()[:40]!r}")


def t_mixed_load():
    """A 64K-token prefill starts while another request is streaming: the streamer's token gaps during the prefill."""
    out = {}

    def streamer():
        out["s"] = chat([{"role": "user", "content": "Write a detailed Python tutorial on asyncio with many code "
                                                     "examples."}], 700, stream=True)

    def prefill():
        out["p"] = chat([{"role": "user", "content": doc(64, "the vault code is 'cobalt-river-19'.") +
                          "What is the vault code mentioned in the document? Answer only with it."}], 20, stream=True)

    ts = threading.Thread(target=streamer)
    ts.start()
    time.sleep(3)
    t_p0 = time.time()
    tp = threading.Thread(target=prefill)
    tp.start()
    tp.join()
    ts.join()
    s, p = out["s"], out["p"]
    t_p1 = t_p0 + p["ttft"]
    during = [t for t in s["times"] if t_p0 <= t <= t_p1]
    gaps = [b - a for a, b in zip(s["times"], s["times"][1:]) if t_p0 <= a <= t_p1]
    before = [b - a for a, b in zip(s["times"], s["times"][1:]) if a < t_p0]
    check("long prefill while another request streams", "cobalt-river-19" in p["text"] and len(s["times"]) > 100,
          f"64K prefill ttft {p['ttft']:.1f} s ({p['usage'].get('prompt_tokens', 0)} tok); streamer before: "
          f"{len(before) / max(sum(before), 1e-9):.0f} chunks/s, during the prefill: {len(during)} chunks in "
          f"{p['ttft']:.1f} s, longest gap {1e3 * max(gaps or [0]):.0f} ms")


def t_long_gen():
    r = chat([{"role": "user", "content": "Write a long, detailed technical article about database indexing."}],
             4096, stream=True, extra={"ignore_eos": True})
    n = r["usage"].get("completion_tokens", 0)
    rate = (n - 1) / (r["times"][-1] - r["times"][0]) if len(r["times"]) > 1 else 0
    check("4096-token generation", n == 4096, f"{n} tokens at {rate:.1f} tok/s")


def t_stability():
    prompts = [("用两句话介绍一下长城。", 120), ("Write a bash one-liner that counts lines in all .py files.", 120),
               ("Explain TCP slow start briefly.", 150), ("把下面这句话翻译成英文：今天天气很好，我们去公园散步吧。", 80)]
    errs, done = [], []
    lock = threading.Lock()

    def worker(w):
        for i in range(12):
            p, n = prompts[(i + w) % len(prompts)]
            try:
                r = chat([{"role": "user", "content": p}], n)
                with lock:
                    (done if r["text"].strip() else errs).append(i)
            except Exception as e:   # noqa: BLE001
                with lock:
                    errs.append(str(e)[:80])

    t0 = time.time()
    th = [threading.Thread(target=worker, args=(w,)) for w in range(2)]
    [t.start() for t in th]
    [t.join() for t in th]
    check("24 mixed requests on 2 workers", not errs and len(done) == 24,
          f"{len(done)} ok, {len(errs)} failed {errs[:2]}, {time.time() - t0:.1f} s")


def t_max_context():
    d = doc(250, "the final answer to the audit is 'saffron-glacier-88'.", at=0.97)
    r = chat([{"role": "user", "content": d + "What is the final answer to the audit mentioned in the document? "
                                              "Answer only with it."}], 30, stream=True)
    p = r["usage"].get("prompt_tokens", 0)
    check("needle near the end of a ~250K-token prompt", "saffron-glacier-88" in r["text"] and p > 240000,
          f"{p} prompt tokens, ttft {r['ttft']:.1f} s ({p / r['ttft']:.0f} tok/s), answer {r['text'].strip()[:40]!r}")


for name, fn in (("models", t_models), ("arithmetic", t_arith), ("thinking", t_thinking), ("tool call", t_tools),
                 ("streaming", t_stream), ("determinism", t_determinism), ("multi-turn", t_multiturn),
                 ("mixed load", t_mixed_load), ("long generation", t_long_gen), ("stability", t_stability),
                 ("max context", t_max_context)):
    if (ONLY is None and not (SKIP_MAX and name == "max context")) or (ONLY is not None and name in ONLY):
        run(name, fn)
print(f"=== acceptance: {sum(ok for _, ok in results)} / {len(results)} passed", flush=True)
