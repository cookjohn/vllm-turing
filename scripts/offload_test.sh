#!/bin/bash
# KV offloading to host RAM vs eviction: a test server (:8081, production's configuration) gets a long "main session"
# prompt (LONG tokens, default 120000), then FLOOD distinct 30k-token "sub-agent" prompts (default 9, 3 at a time) that
# need more KV than is free, then the main session's prompt again: its cached tokens, time to the first token and
# whether the greedy continuation is unchanged. EXTRA_ARGS: e.g. "--kv-offloading-size 24". Run under probe_safe.sh:
#   PW=... probe_safe.sh 1800 bash -c 'TAG=base offload_test.sh; TAG=off EXTRA_ARGS="--kv-offloading-size 24" offload_test.sh'
V=$HOME/vllm-env/bin
export PATH=$V:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
MODEL=${MODEL:-$HOME/models/Qwen3.8-27B-SQ06}
KEY=$(cat $HOME/models/api-key.txt)
log=/tmp/offload${TAG:+-$TAG}.log
P2PV=NCCL_P2P_DISABLE=1
nvidia-smi nvlink --status 2>/dev/null | grep -q "GB/s" && P2PV=NCCL_P2P_DISABLE=0
env VLLM_TURING_AR8=1 VLLM_TURING_TAIL=1 VLLM_TURING_ASYNC_SPEC=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $P2PV \
    setsid $V/vllm serve $MODEL --served-model-name q --host 127.0.0.1 --port 8081 --api-key "$KEY" \
    --tensor-parallel-size 2 --dtype float16 --max-model-len 262144 --gpu-memory-utilization 0.95 --max-num-seqs 6 \
    --max-num-batched-tokens 7070 --attention-backend FLASHINFER --kv-cache-dtype fp8 --mamba-ssm-cache-dtype float16 \
    --block-size 880 --long-prefill-token-threshold 1760 --enable-prompt-tokens-details \
    --prefix-match-unit 80 --enable-mamba-shared-prefix-checkpoint \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":4}' $EXTRA_ARGS > $log 2>&1 &
pid=$!
for i in $(seq 1 300); do
    sleep 2; kill -0 $pid 2>/dev/null || break
    [ "$(curl -s -o /dev/null -w '%{http_code}' localhost:8081/health)" = 200 ] && break
done
if [ "$(curl -s -o /dev/null -w '%{http_code}' localhost:8081/health)" != 200 ]; then
    echo "[$TAG] start failed"; grep -E "Error|error" $log | tail -8; kill -9 -$pid 2>/dev/null; exit 1
fi
echo "[$TAG] extra args: ${EXTRA_ARGS:-none}; $(grep -oE "GPU KV cache size: [0-9,]+ tokens" $log | head -1)"
LONG=${LONG:-120000} FLOOD=${FLOOD:-9} $V/python - "$MODEL" <<'EOF'
import json, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer
key = open(os.path.expanduser("~/models/api-key.txt")).read().strip()
tk = AutoTokenizer.from_pretrained(sys.argv[1])
src = open(os.path.expanduser("~/vllm-env/lib/python3.12/site-packages/vllm/v1/worker/gpu/model_runner.py")).read()
ids = tk.encode(src, add_special_tokens=False)
def text(n, head):
    return head + tk.decode((ids * (n // len(ids) + 1))[:n])
def req(prompt, max_tokens):
    body = {"model": "q", "prompt": prompt, "max_tokens": max_tokens, "temperature": 0}
    r = urllib.request.Request("http://127.0.0.1:8081/v1/completions", json.dumps(body).encode(),
                               {"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    t = time.time()
    d = json.load(urllib.request.urlopen(r, timeout=1800))
    u = d["usage"]
    return time.time() - t, u["prompt_tokens"], (u.get("prompt_tokens_details") or {}).get("cached_tokens"), d["choices"][0]["text"]
main = text(int(os.environ["LONG"]), "main session context\n") + "\n\nSummarize the code above in three sentences.\n"
dt, p, c, t1 = req(main, 24)
print(f"  main session, first: prompt {p}, cached {c}, {dt:.1f} s")
dt, p, c, _ = req(main, 1)
print(f"  main session, at once again: cached {c}, {dt:.2f} s")
n = int(os.environ["FLOOD"])
t0 = time.time()
with ThreadPoolExecutor(3) as ex:
    res = list(ex.map(lambda i: req(text(30000, f"sub-agent {i} {time.time()}\n"), 1), range(n)))
print(f"  {n} sub-agent prompts of {res[0][1]} tokens: {time.time() - t0:.1f} s")
dt, p, c, t2 = req(main, 24)
print(f"  main session after them: prompt {p}, cached {c}, {dt:.2f} s, continuation {'identical' if t1 == t2 else 'DIFFERENT'}")
if t1 != t2:
    print("    before:", repr(t1[:120])); print("    after: ", repr(t2[:120]))
EOF
grep -iE "offload" $log | grep -viE "warning|non-default" | tail -3 | cut -c1-200
kill -INT $pid; for i in $(seq 1 30); do kill -0 $pid 2>/dev/null || break; sleep 2; done; kill -9 -$pid 2>/dev/null
exit 0
