#!/usr/bin/env python3
"""Keep-alive test of vllm_monitor.py: a fake upstream that is silent for 23 s before its SSE events (like vLLM during a
long prompt); through the monitor the client must receive ': keep-alive' comment lines about every 10 s, then the
events unchanged. usage: test_keepalive.py   (vLLM env python; starts the monitor from this directory)"""
import asyncio
import os
import subprocess
import sys
import time
import urllib.request

from aiohttp import web

HERE = os.path.dirname(os.path.abspath(__file__))
UP, MON = 8084, 8085
KEY = open(os.path.expanduser("~/models/api-key.txt")).read().strip()
EVENTS = (b'event: message_start\ndata: {"type":"message_start","message":{"id":"m1","usage":{}}}\n\n'
          b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
          b'"delta":{"type":"text_delta","text":"hi"}}\n\nevent: message_stop\ndata: {"type":"message_stop"}\n\n')


async def messages(request):
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)
    await asyncio.sleep(23)
    await resp.write(EVENTS)
    return resp


def client(out):
    req = urllib.request.Request(f"http://127.0.0.1:{MON}/v1/messages", b'{"model":"q","messages":[],"stream":true}',
                                 {"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    t0 = time.time()
    r = urllib.request.urlopen(req, timeout=60)
    body = b""
    while True:
        b = r.read1(65536)
        if not b:
            break
        out.append((time.time() - t0, b))
        body += b
    out.append(("body", body))


async def main():
    app = web.Application()
    app.router.add_post("/v1/messages", messages)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", UP).start()
    mon = subprocess.Popen([sys.executable, os.path.join(HERE, "vllm_monitor.py"), "--listen", f"127.0.0.1:{MON}",
                            "--upstream", f"http://127.0.0.1:{UP}", "--log-dir", "/tmp/vmon-ka-logs"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        await asyncio.sleep(4)
        out = []
        await asyncio.to_thread(client, out)
        body = out[-1][1]
        kas = [round(t, 1) for t, b in out[:-1] if b.startswith(b": keep-alive")]
        ok = len(kas) == 2 and body.endswith(EVENTS) and body.replace(b": keep-alive\n\n", b"") == EVENTS
        print(f"{'PASS' if ok else 'FAIL'}  keep-alive comments at {kas} s, events intact after them: "
              f"{body.replace(b': keep-alive' + bytes([10, 10]), b'') == EVENTS}")
    finally:
        mon.terminate()
        await runner.cleanup()


asyncio.run(main())
