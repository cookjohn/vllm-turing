#!/usr/bin/env python3
"""Where the time of agent sessions through the monitoring proxy goes (request logs of vllm_monitor.py): per request
the prefill wait (first token) vs the generation, prefix-cache hits, reasoning vs visible output; per session the point
where consecutive requests first differ (the client rewriting earlier context costs a re-prefill from there).
usage: harness_stats.py <requests-*.jsonl.gz> ...   (python3; a file still being appended is read up to its end)"""
import collections
import gzip
import hashlib
import json
import sys
import zlib


def records(paths):
    for p in paths:
        try:
            with gzip.open(p, "rt") as fh:
                for line in fh:
                    try:
                        yield json.loads(line)
                    except ValueError:
                        continue
        except (EOFError, zlib.error, OSError):
            continue


def _strip_cc(x):
    """Drop Anthropic cache_control markers (the client moves its breakpoint to the newest message every turn; the text
    vLLM sees does not change)."""
    if isinstance(x, dict):
        return {k: _strip_cc(v) for k, v in x.items() if k != "cache_control"}
    if isinstance(x, list):
        return [_strip_cc(v) for v in x]
    return x


def msg_keys(body):
    """One string per message (role + content), for the first differing message between consecutive requests."""
    out = []
    for m in (body or {}).get("messages", []):
        out.append(json.dumps(_strip_cc(m), ensure_ascii=False, sort_keys=True))
    return out


def session_key(r):
    body = r.get("body") or {}
    msgs = body.get("messages", [])
    first = json.dumps(msgs[0], ensure_ascii=False, sort_keys=True)[:4000] if msgs else ""
    sysp = json.dumps(body.get("system", ""), ensure_ascii=False)[:4000]
    return (r.get("client"), hashlib.sha1((sysp + first).encode()).hexdigest()[:12])


tot = collections.Counter()
rows = []
prev = {}
div = collections.Counter()          # where the prefix broke: "append only" / "message i of n changed" buckets
div_examples = []
for r in records(sys.argv[1:]):
    if r.get("status") != 200 or not r.get("t_first") or not r.get("t_end"):
        continue
    pt, ct = r.get("prompt_tokens") or 0, r.get("cached_tokens") or 0
    out = r.get("completion_tokens") or 0
    ttft = r["t_first"] - r["t0"]
    gen = r["t_end"] - r["t_first"]
    rows.append((pt, ct, out, ttft, gen, r.get("out_tok_r") or 0, r.get("finish"), len(r.get("tool_calls") or []),
                 r.get("n_tools") or 0, (r.get("params") or {}).get("ctx_edit")))
    sk = session_key(r)
    keys = msg_keys(r.get("body"))
    if sk in prev:
        pk = prev[sk]
        i = 0
        while i < min(len(pk), len(keys)) and pk[i] == keys[i]:
            i += 1
        if i == len(pk):
            div["append only"] += 1
        else:
            frac = i / max(len(pk), 1)
            b = "changed in the last 10% of messages" if frac >= 0.9 else (
                "changed at 50-90%" if frac >= 0.5 else "changed before 50%")
            div[b] += 1
            if len(div_examples) < 12:
                a = pk[i][:220] if i < len(pk) else "(none)"
                c = keys[i][:220] if i < len(keys) else "(none)"
                div_examples.append((i, len(pk), len(keys), pt - ct, a, c))
    prev[sk] = keys

n = len(rows)
if not n:
    sys.exit("no records")
P = sum(x[0] for x in rows)
C = sum(x[1] for x in rows)
O = sum(x[2] for x in rows)
R = sum(x[5] for x in rows)
TT = sum(x[3] for x in rows)
G = sum(x[4] for x in rows)
print(f"requests {n}, prompt tokens {P / 1e6:.1f} M ({C / max(P, 1) * 100:.1f}% from the prefix cache), "
      f"output {O / 1e3:.0f} k tokens ({R / max(O, 1) * 100:.0f}% reasoning by the proxy's count)")
print(f"wall time in requests {(TT + G) / 3600:.2f} h: waiting for the first token {TT / 3600:.2f} h "
      f"({TT / (TT + G) * 100:.0f}%), generating {G / 3600:.2f} h; mean output {O / n:.0f} tokens, "
      f"generation {O / max(G, 1e-9):.1f} tok/s")
unc = sorted(x[0] - x[1] for x in rows)
print("uncached prompt tokens per request: median %d, p90 %d, p99 %d, max %d" %
      (unc[n // 2], unc[int(n * .9)], unc[int(n * .99)], unc[-1]))
big = [x for x in rows if x[0] - x[1] > 20000]
print(f"requests with > 20k uncached tokens: {len(big)}, their first-token wait {sum(x[3] for x in big) / 3600:.2f} h")
pl = sorted(x[0] for x in rows)
print("prompt tokens per request: median %d, p90 %d, max %d; tools per request median %d" %
      (pl[n // 2], pl[int(n * .9)], pl[-1], sorted(x[8] for x in rows)[n // 2]))
fin = collections.Counter(x[6] for x in rows)
print("finish:", dict(fin.most_common(5)), "| tool calls per request:",
      dict(collections.Counter(min(x[7], 4) for x in rows).most_common()))
ob = collections.Counter(("<64" if x[2] < 64 else "<256" if x[2] < 256 else "<1k" if x[2] < 1024 else "<4k" if x[2] < 4096
                          else ">=4k") for x in rows)
print("output tokens per request:", dict(sorted(ob.items())))
print("ctx_edit applied:", sum(1 for x in rows if x[9]), "requests")
print("consecutive requests of a session:", dict(div))
for i, a, b, u, pa, pb in div_examples:
    print(f"  first change at message {i} of {a} -> {b} (uncached {u}):\n    before: {pa}\n    after:  {pb}")
