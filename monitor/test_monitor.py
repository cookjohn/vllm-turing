#!/usr/bin/env python3
"""Test of vllm_monitor.py against a running vLLM: requests through the monitor (MON) and directly (DIRECT) -
streamed chat output identical, tool call, thinking, non-stream, completions, Anthropic /v1/messages, a client that
disconnects mid-stream (the monitor must record it as aborted and vLLM must stop generating), then the dashboard API
(list, detail, live events) and the proxy's latency overhead.
usage: test_monitor.py <monitor url> <direct url>"""
import os
import json
import socket
import sys
import threading
import time
import urllib.request
from urllib.parse import urlparse

MON, DIRECT = sys.argv[1].rstrip("/"), sys.argv[2].rstrip("/")
KEY = open(os.path.expanduser("~/models/api-key.txt")).read().strip()
H = {"Authorization": "Bearer " + KEY, "Content-Type": "application/json"}
ok_all = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)
    ok_all.append(ok)


def post(base, path, body, stream=False):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), H)
    t0 = time.time()
    r = urllib.request.urlopen(req, timeout=600)
    if not stream:
        return json.load(r), time.time() - t0, None
    first, text, raw = None, [], b""
    for line in r:
        raw += line
        s = line.decode().strip()
        if s.startswith("data: ") and s != "data: [DONE]":
            d = json.loads(s[6:])
            for ch in d.get("choices", []):
                piece = (ch.get("delta") or {}).get("content") or ch.get("text") or ""
                if piece:
                    first = first or time.time()
                    text.append(piece)
    return "".join(text), time.time() - t0, (first - t0 if first else None)


def chat(msgs, n=200, **kw):
    return {"model": "qwen3.8-27b", "messages": msgs, "max_tokens": n, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}, **kw}


# 1. identical streamed output through the monitor and directly
body = chat([{"role": "user", "content": "List the planets of the solar system with one fact each."}], 300,
            stream=True)
a, ta, fa = post(MON, "/v1/chat/completions", body, True)
b, tb, fb = post(DIRECT, "/v1/chat/completions", body, True)
check("stream passthrough identical", a == b and len(a) > 200,
      f"{len(a)} chars; ttft monitor {fa:.3f} s / direct {fb:.3f} s; total {ta:.2f} / {tb:.2f} s")

# 2. tool call (non-stream) and 3. thinking (stream)
tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {
    "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
d, _, _ = post(MON, "/v1/chat/completions", chat([{"role": "user", "content": "Weather in Rome?"}], 200, tools=tools))
tc = d["choices"][0]["message"].get("tool_calls") or []
check("tool call passthrough", bool(tc) and tc[0]["function"]["name"] == "get_weather", str(tc[:1])[:120])
th = chat([{"role": "user", "content": "Is 221 prime? Answer yes or no."}], 1500, stream=True)
th["chat_template_kwargs"] = {"enable_thinking": True}
txt, _, _ = post(MON, "/v1/chat/completions", th, True)
check("thinking stream passthrough", "no" in txt.lower(), repr(txt.strip()[:60]))

# 4. completions
c, _, _ = post(MON, "/v1/completions", {"model": "qwen3.8-27b", "prompt": "def fibonacci(n):", "max_tokens": 64,
                                       "temperature": 0, "stream": True}, True)
check("completions stream", len(c) > 20, repr(c[:50]))

# 5. Anthropic messages API
try:
    req = urllib.request.Request(MON + "/v1/messages", json.dumps({
        "model": "qwen3.8-27b", "max_tokens": 64, "messages": [{"role": "user", "content": "Say hello in French."}]
    }).encode(), {**H, "x-api-key": KEY, "anthropic-version": "2023-06-01"})
    d = json.load(urllib.request.urlopen(req, timeout=120))
    check("anthropic /v1/messages", d.get("type") == "message", repr(str(d.get("content"))[:80]))
except Exception as e:   # noqa: BLE001
    check("anthropic /v1/messages", False, f"{e}")

# 6. client disconnect mid-stream: raw socket, read a little, close
u = urlparse(MON)
body = json.dumps(chat([{"role": "user", "content": "Write a 3000-word essay about rivers."}], 3000, stream=True))
s = socket.create_connection((u.hostname, u.port))
s.sendall((f"POST /v1/chat/completions HTTP/1.1\r\nHost: {u.hostname}\r\nAuthorization: Bearer {KEY}\r\n"
           f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n{body}").encode())
got = b""
t0 = time.time()
while len(got) < 3000 and time.time() - t0 < 60:
    got += s.recv(4096)
s.close()
time.sleep(3)
m = urllib.request.urlopen(urllib.request.Request(DIRECT + "/metrics"), timeout=10).read().decode()
running = sum(float(l.split()[-1]) for l in m.splitlines() if l.startswith("vllm:num_requests_running"))
check("client disconnect aborts the vLLM request", running == 0, f"vLLM running requests 3 s after the disconnect: {running:.0f}")

# 7. dashboard API
time.sleep(1)
lst = json.load(urllib.request.urlopen(urllib.request.Request(MON + "/monitor/api/list", headers=H), timeout=10))
recs = lst["recs"]
states = [r["state"] for r in recs]
check("dashboard list", len(recs) >= 6 and "aborted" in states and states.count("done") >= 5,
      f"{len(recs)} records, states {states}")
last_done = [r for r in recs if r["state"] == "done" and r["kind"] == "chat"][0]
det = json.load(urllib.request.urlopen(urllib.request.Request(MON + f"/monitor/api/req/{last_done['id']}",
                                                              headers=H), timeout=10))
check("dashboard detail", bool(det.get("body", {}).get("messages")) and bool(det.get("content") or det.get("tools")),
      f"#{det['id']} est_prompt {det.get('est_prompt')} out {len(det.get('content') or '')} chars tools "
      f"{[t['name'] for t in det.get('tools') or []]} wait {(det['t_first'] or det['t_end']) - det['t0']:.2f} s")
# live events while a request streams
events = []


def listen():
    """raw socket: urllib's line iteration buffers a chunked event stream"""
    u = urlparse(MON)
    s = socket.create_connection((u.hostname, u.port))
    s.sendall(f"GET /monitor/api/events HTTP/1.1\r\nHost: {u.hostname}\r\nAuthorization: Bearer {KEY}\r\n\r\n".encode())
    s.settimeout(1)
    buf, t0 = b"", time.time()
    while time.time() - t0 < 8:
        try:
            buf += s.recv(65536)
        except socket.timeout:
            continue
    s.close()
    for line in buf.split(b"\n"):
        i = line.find(b"data: {")
        if i >= 0:
            try:
                events.append(json.loads(line[i + 6:]))
            except Exception:   # noqa: BLE001
                pass


th = threading.Thread(target=listen)
th.start()
time.sleep(0.5)
post(MON, "/v1/chat/completions", chat([{"role": "user", "content": "Count from 1 to 60 in words."}], 300,
                                      stream=True), True)
th.join()
kinds = [e["t"] for e in events]
deltas = sum(len(e.get("c", "")) for e in events if e["t"] == "p")
check("live events", "s" in kinds and kinds.count("p") >= 3 and "srv" in kinds,
      f"{len(events)} events ({kinds.count('p')} progress, {kinds.count('srv')} server), {deltas} streamed chars")
print(f"=== monitor test: {sum(ok_all)} / {len(ok_all)} passed")
