#!/usr/bin/env python3
"""Test of vllm_monitor.py's max_tokens fitting: a fake upstream with vLLM's context check (1000-token context, one
character = one token; the prompt is truncated at the input limit + 1, so the error says "at least") and its
/v1/messages/count_tokens and /tokenize endpoints. Through the monitor a request whose prompt + max_tokens exceeds the
context must be resent with max_tokens = the room left (Anthropic, streamed or not, and OpenAI chat), a prompt that leaves
less than VMON_MIN_OUTPUT tokens must get the Anthropic API's "prompt is too long" error, and requests that fit or fail
for other reasons must pass unchanged. usage: test_fit.py   (vLLM env python; starts the monitor from this directory)"""
import asyncio
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request

from aiohttp import web

HERE = os.path.dirname(os.path.abspath(__file__))
UP, MON = 8086, 8087
TOTAL, MIN_OUT = 1000, 100
KEY = open(os.path.expanduser("~/models/api-key.txt")).read().strip()
calls = []


def plen(b):
    """Prompt size: characters of string contents; an agent conversation (block lists): JSON characters / 20."""
    return sum(len(m["content"]) if isinstance(m["content"], str) else len(json.dumps(m["content"])) // 20
               for m in b.get("messages", []))


def notes(b):
    """Tool results the monitor cleared / thinking blocks left in an agent conversation."""
    blocks = [x for m in b.get("messages", []) if isinstance(m["content"], list) for x in m["content"]]
    return (sum(x.get("type") == "tool_result" and str(x.get("content")).startswith("[Output of ") for x in blocks),
            sum(x.get("type") == "thinking" for x in blocks))


def too_long(b, key):
    n, mt = plen(b), b.get(key) or 0
    lim = TOTAL - mt
    if n <= lim:
        return None
    shown = min(n, lim + 1)
    q = "at least " if shown == lim + 1 else ""
    return (f"This model's maximum context length is {TOTAL} tokens. However, you requested {mt} output tokens and "
            f"your prompt contains {q}{shown} input tokens, for a total of {q}{shown + mt} tokens. Please reduce the "
            f"length of the input prompt or the number of requested output tokens.")


async def messages(request):
    b = await request.json()
    calls.append(("messages", b.get("max_tokens"), notes(b)))
    if b.get("bad"):
        return web.json_response({"type": "error", "error": {"type": "BadRequestError", "message": "other"}},
                                 status=400)
    msg = too_long(b, "max_tokens")
    if msg:
        return web.json_response({"type": "error", "error": {"type": "BadRequestError", "message": msg}}, status=400)
    text = f"max_tokens={b['max_tokens']}"
    if any(isinstance(m["content"], list) for m in b["messages"]):
        text += " notes=%d thinking=%d" % notes(b)
    if b.get("stream"):
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        await resp.write(b'event: message_start\ndata: {"type":"message_start","message":{"id":"m1","usage":{}}}\n\n'
                         b'event: content_block_delta\ndata: ' + json.dumps(
                             {"type": "content_block_delta", "index": 0,
                              "delta": {"type": "text_delta", "text": text}}).encode() +
                         b'\n\nevent: message_stop\ndata: {"type":"message_stop"}\n\n')
        return resp
    return web.json_response({"id": "m1", "type": "message", "content": [{"type": "text", "text": text}],
                              "stop_reason": "end_turn", "usage": {"input_tokens": plen(b), "output_tokens": 3}})


async def chat(request):
    b = await request.json()
    calls.append(("chat", b.get("max_tokens")))
    msg = too_long(b, "max_tokens")
    if msg:
        return web.json_response({"error": {"message": msg, "type": "BadRequestError", "code": 400}}, status=400)
    return web.json_response({"id": "c1", "choices": [{"message": {"content": f"max_tokens={b['max_tokens']}"},
                                                       "finish_reason": "stop"}],
                              "usage": {"prompt_tokens": plen(b), "completion_tokens": 3}})


async def count_tokens(request):
    b = await request.json()
    calls.append(("count_tokens", sorted(b)))
    return web.json_response({"input_tokens": plen(b)})


async def tokenize(request):
    b = await request.json()
    calls.append(("tokenize", sorted(b)))
    return web.json_response({"count": plen(b), "max_model_len": TOTAL, "tokens": []})


def post(path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{MON}{path}", json.dumps(body).encode(),
                                 {"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def agent(steps, mt):
    """A Claude Code style conversation: thinking + tool call, tool result of ~300 words, per step."""
    msgs = [{"role": "user", "content": "Summarize the logs."}]
    for k in range(steps):
        msgs.append({"role": "assistant", "content": [
            {"type": "thinking", "thinking": " ".join(f"idea{k}x{i}" for i in range(80)), "signature": "s"},
            {"type": "tool_use", "id": f"t{k}", "name": "Read", "input": {"file_path": f"/var/log/app{k}.log"}}]})
        msgs.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{k}",
                                                  "content": " ".join(f"line{k} event{i} ok" for i in range(300))}]})
    msgs.append({"role": "user", "content": "Now the summary please."})
    return {"model": "q", "max_tokens": mt, "messages": msgs}


def anth(n, mt, **kw):
    return dict({"model": "q", "max_tokens": mt, "system": "s", "tools": [],
                 "messages": [{"role": "user", "content": "x" * n}]}, **kw)


CASES = [
    # name, path, body, expected status, text the response must contain, expected upstream calls
    ("anthropic: resent with the room left", "/v1/messages", anth(700, 800), 200,
     f"max_tokens={TOTAL - 700 - 64}", ["messages", "count_tokens", "messages"]),
    ("anthropic stream: resent", "/v1/messages", anth(700, 500, stream=True), 200,
     f"max_tokens={TOTAL - 700 - 64}", ["messages", "count_tokens", "messages"]),
    ("anthropic: no room -> prompt is too long", "/v1/messages", anth(900, 800), 400,
     f"prompt is too long: 900 tokens > {TOTAL - MIN_OUT} maximum", ["messages", "count_tokens"]),
    ("anthropic: fits, unchanged", "/v1/messages", anth(300, 600), 200, "max_tokens=600", ["messages"]),
    ("anthropic: other 400, unchanged", "/v1/messages", anth(300, 800, bad=True), 400, '"other"', ["messages"]),
    # context edit (VMON_CTX_* below): the forwarded conversation has its older tool results cleared, also in the
    # max_tokens retry (fitted to the edited prompt)
    ("anthropic: long agent conversation edited", "/v1/messages", agent(10, 50), 200, "notes=",
     ["messages"]),
    ("anthropic: edited conversation, max_tokens retry keeps the edit", "/v1/messages", agent(10, 800), 200, "notes=",
     ["messages", "count_tokens", "messages"]),
    ("chat: resent (tokenize count)", "/v1/chat/completions",
     {"model": "q", "max_tokens": 600, "messages": [{"role": "user", "content": "y" * 600}]}, 200,
     f"max_tokens={TOTAL - 600 - 64}", ["chat", "tokenize", "chat"]),
    ("chat: no room, vLLM's error", "/v1/chat/completions",
     {"model": "q", "max_tokens": 600, "messages": [{"role": "user", "content": "y" * 990}]}, 400,
     "maximum context length is 1000", ["chat", "tokenize"]),
]


async def main():
    app = web.Application(client_max_size=1 << 26)
    app.router.add_post("/v1/messages", messages)
    app.router.add_post("/v1/messages/count_tokens", count_tokens)
    app.router.add_post("/v1/chat/completions", chat)
    app.router.add_post("/tokenize", tokenize)
    app.router.add_get("/health", lambda r: web.Response(text="ok"))
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", UP).start()
    mon = subprocess.Popen([sys.executable, os.path.join(HERE, "vllm_monitor.py"), "--listen", f"127.0.0.1:{MON}",
                            "--upstream", f"http://127.0.0.1:{UP}", "--log-dir", "/tmp/vmon-fit-logs"],
                           env=dict(os.environ, VMON_MIN_OUTPUT=str(MIN_OUT), VMON_CTX_TRIGGER="6000",
                                    VMON_CTX_KEEP="1500", VMON_CTX_STEP="1500"),
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    fails = 0
    try:
        await asyncio.sleep(4)
        for name, path, body, st, want, want_calls in CASES:
            calls.clear()
            status, text = await asyncio.to_thread(post, path, body)
            got = [c[0] for c in calls]
            ok = status == st and want in text and got == want_calls
            if "notes=" in want:          # every forwarded copy edited: notes present, older thinking dropped
                ok = ok and all(c[2][0] > 0 and c[2][1] < 10 for c in calls if c[0] == "messages")
            if "count_tokens" in got:     # the count request carries the prompt fields only
                ok = ok and next(c for c in calls if c[0] == "count_tokens")[1] == sorted(
                    k for k in ("messages", "model", "system", "tools") if k in body)
            fails += not ok
            print(f"{'PASS' if ok else 'FAIL'}  {name}: HTTP {status}, calls {got}"
                  + ("" if ok else f"\n      response {text[:300]}\n      calls {calls}"))
        recs = json.loads(await asyncio.to_thread(lambda: urllib.request.urlopen(urllib.request.Request(
            f"http://127.0.0.1:{MON}/monitor/api/list", headers={"Authorization": "Bearer " + KEY})).read()))["recs"]
        sent = [r["params"].get("max_tokens_sent") for r in recs]
        ok = sent.count(TOTAL - 700 - 64) == 2 and sent.count(TOTAL - 600 - 64) == 1
        edits = [r["params"]["ctx_edit"] for r in recs if "ctx_edit" in r["params"]]
        ok = ok and len(edits) == 2 and all(e["results_cleared"] > 0 for e in edits)
        print(f"      context edits on the dashboard: {edits}")
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'}  dashboard records max_tokens_sent: {sent}")
    finally:
        mon.terminate()
        err = mon.communicate()[1].decode()
        await runner.cleanup()
    print("\n".join(l for l in err.splitlines() if "resent" in l or "leaves" in l or "Error" in l)[-2000:])
    print("ALL PASS" if not fails else f"{fails} FAILED")


asyncio.run(main())
