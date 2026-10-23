#!/bin/bash
# Unprofiled timing of the requests vllm_util_probe.sh profiles (same production configuration): the 512-token code
# decode (streamed: first-token and per-token times) three times, and one fresh 32K-token prefill; with the profiled
# GPU busy time per step this gives the true GPU idle time (the torch profiler adds CPU time per kernel launch).
# usage: vllm_decode_timing.sh   (env MODEL, LEN 262144)
V=$HOME/vllm-env/bin
export PATH=$V:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
MODEL=${MODEL:-$HOME/models/Qwen3.8-27B-SQ06}
KEY=$(cat $HOME/models/api-key.txt)
log=/tmp/vllm-timing.log
VLLM_TURING_AR8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 setsid $V/vllm serve $MODEL \
    --served-model-name q --host 127.0.0.1 --port 8081 --api-key "$KEY" --tensor-parallel-size 2 --dtype float16 \
    --max-model-len ${LEN:-262144} --gpu-memory-utilization 0.95 --max-num-seqs 2 --max-num-batched-tokens 6730 \
    --attention-backend FLASHINFER --kv-cache-dtype fp8 --block-size 1680 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":4}' > $log 2>&1 &
pid=$!
for i in $(seq 1 600); do
    sleep 2; kill -0 $pid 2>/dev/null || break
    [ "$(curl -s -o /dev/null -w '%{http_code}' localhost:8081/health)" = 200 ] && break
done
[ "$(curl -s -o /dev/null -w '%{http_code}' localhost:8081/health)" = 200 ] || { echo "start failed"; tail -3 $log; kill -9 -$pid; exit 1; }
$V/python - "$MODEL" <<'EOF'
import json, os, random, sys, time, urllib.request
from transformers import AutoTokenizer
model = sys.argv[1]
key = open(os.path.expanduser("~/models/api-key.txt")).read().strip()
H = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
CODE = ("Write a Python module implementing a thread-safe LRU cache with per-entry TTL, size limits, statistics and an "
        "async API, with docstrings and pytest unit tests.\n\n```python\n")


def stream(body):
    body = dict(body, stream=True)
    r = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8081/v1/completions",
                                                      json.dumps(body).encode(), H), timeout=3600)
    t0 = time.time(); first = None; n = 0
    for line in r:
        line = line.decode().strip()
        if not line.startswith("data:") or line.endswith("[DONE]"):
            continue
        d = json.loads(line[5:])
        if d["choices"] and d["choices"][0].get("text"):
            n += 1
            if first is None:
                first = time.time()
    return first - t0, time.time() - first


for i in range(3):
    ttft, dec = stream({"model": "q", "prompt": CODE, "max_tokens": 512, "temperature": 0, "ignore_eos": True})
    print(f"decode run {i + 1}: ttft {ttft:.3f} s, 511 tokens after the first in {dec:.3f} s = {511 / dec:.1f} tok/s, "
          f"{dec / (136 - 1) * 1e3:.1f} ms per verify step (136 steps as profiled)", flush=True)
tk = AutoTokenizer.from_pretrained(model)
src = open(os.path.expanduser("~/vllm-env/lib/python3.12/site-packages/vllm/v1/worker/gpu/model_runner.py")).read()
ids = tk.encode(src, add_special_tokens=False)
ids = (ids * (32768 // len(ids) + 1))[:32768]
ttft, _ = stream({"model": "q", "prompt": f"timing probe {random.random()}\n" + tk.decode(ids), "max_tokens": 1,
                  "temperature": 0})
print(f"prefill 32K: ttft {ttft:.2f} s = {32791 / ttft:.0f} tok/s", flush=True)
EOF
grep -h "SpecDecoding metrics" $log | tail -2 | sed -E 's/.*\] //' | cut -c1-200
kill -INT $pid; for i in $(seq 1 30); do kill -0 $pid 2>/dev/null || break; sleep 2; done; kill -9 -$pid 2>/dev/null
