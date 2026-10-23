#!/usr/bin/env python3
"""Replay of a captured agent session (monitor logs) through ctxedit.py with the model's tokenizer: per request the
original and the forwarded prompt size, and the prefix-cache cost = tokens from the first message that differs from
the previous request's forwarded messages to the end (what vLLM has to prefill again), with and without the edits.
usage: test_ctxedit_replay.py [--client IP] [--day YYYY-MM-DD] [TRIGGER KEEP STEP ...]   (vLLM env python, on the
server; default parameter set = the ctxedit defaults)"""
import argparse
import glob
import gzip
import json
import os
import sys
import time

from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ctxedit   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--client", default=None, help="only requests from this client address (default: all)")
ap.add_argument("--day", default=time.strftime("%Y-%m-%d"))
ap.add_argument("--log-dir", default=os.path.expanduser("~/monitor/logs"))
ap.add_argument("--min-msgs", type=int, default=50)
ap.add_argument("params", nargs="*", type=int, help="TRIGGER KEEP STEP triples")
a = ap.parse_args()
tok = Tokenizer.from_file(os.path.expanduser("~/models/Qwen3.8-27B-SQ06/tokenizer.json"))
count = lambda s: len(tok.encode(s, add_special_tokens=False).ids)   # noqa: E731

bodies = []
for f in sorted(glob.glob(os.path.join(a.log_dir, f"requests-{a.day}.jsonl.gz"))):
    with gzip.open(f, "rt") as fh:
        for line in fh:
            if a.client and a.client not in line[:400]:
                continue
            r = json.loads(line)
            b = r.get("body") or {}
            if r.get("kind") == "anthropic" and a.client in (None, r.get("client")) and len(b.get("messages") or []) >= a.min_msgs:
                bodies.append((r["t0"], r["id"], r.get("status"), b))
bodies.sort(key=lambda x: x[0])
first = json.dumps(bodies[-1][3]["messages"][0], sort_keys=True)
bodies = [x for x in bodies if json.dumps(x[3]["messages"][0], sort_keys=True) == first]   # one conversation
print(f"{len(bodies)} requests of one conversation, {len(bodies[0][3]['messages'])}..{len(bodies[-1][3]['messages'])} "
      f"messages")

sets = [tuple(a.params[i:i + 3]) for i in range(0, len(a.params), 3)] or [(ctxedit.TRIGGER, ctxedit.KEEP, ctxedit.STEP)]
ed = ctxedit.ContextEditor(count)
for trig, keep, step in [(None, None, None)] + sets:
    if trig is not None:
        ctxedit.TRIGGER, ctxedit.KEEP, ctxedit.STEP = trig, keep, step
    prev, resent, moves, mx, rows = None, 0, 0, 0, []
    for t0, rid, status, b in bodies:
        if trig is None:
            out, info = None, None
        else:
            out, info = ed.edit("anthropic", b)
        msgs = (out or b)["messages"]
        sizes = [ed.msg_tokens("anthropic", m) for m in msgs]
        sent = sum(sizes) + ed.tokens(ctxedit._dump(b.get("system"))) + ed.tokens(ctxedit._dump(b.get("tools")))
        mx = max(mx, sent)
        if prev is not None:
            n = min(len(prev), len(msgs))
            d = next((i for i in range(n) if msgs[i] != prev[i]), n)
            again = sum(sizes[d:])
            resent += again
            if d < len(prev) - 2:
                moves += 1
                rows.append(f"    #{rid}: prefix differs from message {d} of {len(prev)}: {again} tokens prefilled "
                            f"again; sent {sent} of {info['orig_tokens'] if info else sent}")
        prev = msgs
    name = "no edits" if trig is None else f"TRIGGER {trig} KEEP {keep} STEP {step}"
    print(f"{name}: max prompt sent {mx}, last {sent}; prefix-cache misses {moves}, tokens prefilled again "
          f"{resent} (over {len(bodies)} requests)")
    for r in rows[:12]:
        print(r)
