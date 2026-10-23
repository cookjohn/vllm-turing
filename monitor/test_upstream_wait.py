#!/usr/bin/env python3
"""vllm_monitor.py while vLLM restarts: with nothing listening upstream, a chat request is held (not failed) and
completes once a (fake) upstream comes up 5 s later; a non-generation request (/health) fails at once with 502.
usage: test_upstream_wait.py   (vLLM env python; starts the monitor from this directory)"""
import asyncio
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

from aiohttp import web

HERE = os.path.dirname(os.path.abspath(__file__))
UP, MON = 8096, 8097
KEY = open(os.path.expanduser("~/models/api-key.txt")).read().strip()


async def chat(request):
    return web.json_response({"id": "c1", "choices": [{"message": {"content": "back"}, "finish_reason": "stop"}],
                              "usage": {"prompt_tokens": 3, "completion_tokens": 1}})


def get(path, body=None):
    req = urllib.request.Request(f"http://127.0.0.1:{MON}{path}", json.dumps(body).encode() if body else None,
                                 {"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, r.read().decode(), time.time() - t0
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(), time.time() - t0


async def main():
    mon = subprocess.Popen([sys.executable, os.path.join(HERE, "vllm_monitor.py"), "--listen", f"127.0.0.1:{MON}",
                            "--upstream", f"http://127.0.0.1:{UP}", "--log-dir", "/tmp/vmon-wait-logs"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    fails = 0
    try:
        await asyncio.sleep(4)
        st, _, dt = await asyncio.to_thread(get, "/health")
        ok = st == 502 and dt < 3
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'}  /health with vLLM down: HTTP {st} after {dt:.1f} s")
        task = asyncio.create_task(asyncio.to_thread(get, "/v1/chat/completions",
                                                     {"model": "q", "messages": [{"role": "user", "content": "hi"}]}))
        await asyncio.sleep(5)
        app = web.Application()
        app.router.add_post("/v1/chat/completions", chat)
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", UP).start()
        st, body, dt = await task
        ok = st == 200 and "back" in body and 5 <= dt < 12
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'}  chat request held while vLLM was down: HTTP {st} after {dt:.1f} s")
        await runner.cleanup()
    finally:
        mon.terminate()
    print("ALL PASS" if not fails else f"{fails} FAILED")


asyncio.run(main())
