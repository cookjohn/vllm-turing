#!/usr/bin/env python3
"""Offline test of the monitor's live prompt-progress display (no GPU): a fake vLLM upstream (8 s simulated prefill,
then a short SSE stream; /metrics with KV usage growing by 4 of 193 blocks every 2.75 s like 6720-token chunks) and
a simulated engine progress feed (vllm_turing/progress.py format) around a monitor instance on spare ports.
Phase 1 (feed): the waiting request must show the engine view (computed / prompt / cached) and a prompt rate near
2440 tok/s, the server a live prompt rate. Phase 2 (no feed): the server must show the KV-growth estimate.
usage: test_progress_sim.py   (vLLM env python; starts the monitor from this directory)"""
import asyncio
import json
import os
import subprocess
import sys
import time
import urllib.request

from aiohttp import web

HERE = os.path.dirname(os.path.abspath(__file__))
UP, MON = 8086, 8087
FEED = "/dev/shm/vmon_sim_progress.json"
KEY = open(os.path.expanduser("~/models/api-key.txt")).read().strip()
H = {"Authorization": "Bearer " + KEY}
T0 = time.time()
STATE = {"kv_blocks": 0}


async def chat(request):
    await asyncio.sleep(8)                                  # simulated prefill
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)
    for w in ("Hello", " from", " the", " fake", " upstream"):
        await resp.write(f'data: {{"id":"x","choices":[{{"index":0,"delta":{{"content":"{w}"}}}}]}}\n\n'.encode())
        await asyncio.sleep(0.1)
    await resp.write(b"data: [DONE]\n\n")
    return resp


async def metrics(request):
    kv = STATE["kv_blocks"] / 193
    return web.Response(text=f'vllm:cache_config_info{{block_size="1680",cache_dtype="fp8"}} 1.0\n'
                             f'vllm:kv_cache_usage_perc{{engine="0"}} {kv}\nvllm:num_requests_running{{engine="0"}} 1.0\n')


async def health(request):
    return web.Response(text="ok")


async def kv_grower(stop):
    """KV usage: +40 blocks when scheduled (cache hit + state blocks), then +4 blocks every 2.75 s."""
    await asyncio.sleep(1.0)
    STATE["kv_blocks"] = 40
    while not stop.is_set():
        await asyncio.sleep(2.75)
        STATE["kv_blocks"] += 4


async def feed_writer(stop, rid):
    computed, total, t_last = 20000 + 6720, 0, time.time()
    while not stop.is_set():
        now = time.time()
        if now - t_last >= 2.75:
            computed, total, t_last = min(computed + 6720, 60000), total + 6720, now
        data = {"t": now, "prefill_total": total, "reqs": [{"id": f"chatcmpl-{rid}-0", "prompt": 60000,
                                                            "computed": computed, "cached": 20000, "out": 0,
                                                            "status": "RUNNING", "arrival": T0}]}
        with open(FEED + ".tmp", "w") as f:
            json.dump(data, f)
        os.replace(FEED + ".tmp", FEED)
        await asyncio.sleep(0.2)


def get(path):
    return json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{MON}{path}", headers=H),
                                            timeout=10))


def client(rid, out):
    req = urllib.request.Request(f"http://127.0.0.1:{MON}/v1/chat/completions", json.dumps({
        "model": "q", "messages": [{"role": "user", "content": "hi " * 2000}], "stream": True}).encode(),
        {**H, "Content-Type": "application/json", "X-Request-Id": rid})
    out["text"] = urllib.request.urlopen(req, timeout=60).read().decode()


async def phase(name, rid, with_feed):
    import threading
    stop = asyncio.Event()
    tasks = [asyncio.create_task(kv_grower(stop))]
    if with_feed:
        tasks.append(asyncio.create_task(feed_writer(stop, rid)))
    out = {}
    th = threading.Thread(target=client, args=(rid, out))
    th.start()
    await asyncio.sleep(6.5)
    lst = await asyncio.to_thread(get, "/monitor/api/list")
    rec = [r for r in lst["recs"] if r.get("vreq") == rid][0]
    srv = lst["srv"]
    stop.set()
    for t in tasks:
        t.cancel()
    await asyncio.to_thread(th.join)
    ok = True
    if with_feed:
        ok = rec.get("v") is not None and rec["v"]["prompt"] == 60000 and rec.get("v_rate") and \
            1500 < rec["v_rate"] < 3500 and srv.get("progress_feed") and (srv.get("prompt_live") or 0) > 500
        detail = (f"engine view {rec.get('v')}, request rate {rec.get('v_rate')}, server live "
                  f"{srv.get('prompt_live')} (feed {srv.get('progress_feed')})")
    else:
        ok = not srv.get("progress_feed") and 1000 < (srv.get("prompt_live_kv") or 0) < 4000
        detail = f"KV estimate {srv.get('prompt_live_kv')} tok/s, blocks {srv.get('kv_blocks')}"
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)
    return ok


async def main():
    app = web.Application()
    app.router.add_post("/v1/chat/completions", chat)
    app.router.add_get("/metrics", metrics)
    app.router.add_get("/health", health)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", UP).start()
    env = dict(os.environ, VLLM_TURING_PROGRESS_FILE=FEED)
    mon = subprocess.Popen([sys.executable, os.path.join(HERE, "vllm_monitor.py"), "--listen", f"127.0.0.1:{MON}",
                            "--upstream", f"http://127.0.0.1:{UP}", "--log-dir", "/tmp/vmon-sim-logs"], env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        await asyncio.sleep(4)
        ok1 = await phase("exact progress from the engine feed", "simtest-feed", True)
        os.remove(FEED)
        STATE["kv_blocks"] = 0
        await asyncio.sleep(7)                                    # feed goes stale (5 s)
        ok2 = await phase("KV-growth estimate without the feed", "simtest-kv", False)
        print(f"=== progress simulation: {int(ok1) + int(ok2)} / 2 passed")
    finally:
        mon.terminate()
        await runner.cleanup()
        if os.path.exists(FEED):
            os.remove(FEED)


asyncio.run(main())
