#!/bin/bash
# Cross-session prefix reuse on the hybrid (GDN + attention) model: a test server (:8081) with production's cache
# settings (block 880, fp16 GDN state, MTP4, prefix caching in align mode) gets three "new sub-agent" requests in a row
# that share Claude Code's 47 tool definitions + system prompt (from a logged request, /tmp/cc_subagent.json) and differ
# in the task message; prints each request's cached prompt tokens and time to the first token. Variants: EXTRA_ARGS
# (e.g. "--prefix-match-unit 80 --enable-mamba-shared-prefix-checkpoint"). Run under probe_safe.sh (stops production):
#   PW=... probe_safe.sh 1500 bash -c 'TAG=base prefix_share_test.sh; TAG=junc EXTRA_ARGS="..." prefix_share_test.sh'
V=$HOME/vllm-env/bin
export PATH=$V:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
MODEL=${MODEL:-$HOME/models/Qwen3.8-27B-SQ06}
KEY=$(cat $HOME/models/api-key.txt)
log=/tmp/prefix-share${TAG:+-$TAG}.log
P2PV=NCCL_P2P_DISABLE=1
nvidia-smi nvlink --status 2>/dev/null | grep -q "GB/s" && P2PV=NCCL_P2P_DISABLE=0
env VLLM_TURING_AR8=1 VLLM_TURING_TAIL=1 VLLM_TURING_ASYNC_SPEC=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $P2PV \
    setsid $V/vllm serve $MODEL --served-model-name q --host 127.0.0.1 --port 8081 --api-key "$KEY" \
    --tensor-parallel-size 2 --dtype float16 --max-model-len 262144 --gpu-memory-utilization 0.95 --max-num-seqs 6 \
    --max-num-batched-tokens 7070 --attention-backend FLASHINFER --kv-cache-dtype fp8 --mamba-ssm-cache-dtype float16 \
    --block-size 880 --long-prefill-token-threshold 1760 --enable-prompt-tokens-details \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
    --chat-template $HOME/models/qwen3.8-chat-template-inline-system.jinja \
    --speculative-config '{"method":"mtp","num_speculative_tokens":4}' $EXTRA_ARGS > $log 2>&1 &
pid=$!
for i in $(seq 1 300); do
    sleep 2; kill -0 $pid 2>/dev/null || break
    [ "$(curl -s -o /dev/null -w '%{http_code}' localhost:8081/health)" = 200 ] && break
done
if [ "$(curl -s -o /dev/null -w '%{http_code}' localhost:8081/health)" != 200 ]; then
    echo "[$TAG] start failed"; grep -E "Error|error" $log | tail -5; kill -9 -$pid 2>/dev/null; exit 1
fi
echo "[$TAG] extra args: ${EXTRA_ARGS:-none}; $(grep -oE "GPU KV cache size: [0-9,]+ tokens" $log | head -1)"
grep -iE "fine-grained|shared.prefix|partial" $log | grep -v "^$" | tail -3 | cut -c1-200
$V/python - <<'EOF'
import json, os, random, time, urllib.request
key = open(os.path.expanduser("~/models/api-key.txt")).read().strip()
d = json.load(open("/tmp/cc_subagent.json"))
tasks = ["Find every place where the retry timeout is configured and summarize the defaults.",
         "Search the repository for uses of the deprecated config loader and list the files.",
         "Look for the tests that cover the HTTP client and report which ones are flaky.",
         "Check how logging is initialized in the CLI entry point and describe the flow."]
for i, task in enumerate(tasks):
    body = {"model": "q", "max_tokens": 1, "temperature": 0, "tools": d["tools"],
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "system", "content": d["system"]},
                         {"role": "user", "content": f"<task id={random.randint(0, 1 << 30)}>\n{task}"}]}
    r = urllib.request.Request("http://127.0.0.1:8081/v1/chat/completions", json.dumps(body).encode(),
                               {"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    t = time.time()
    u = json.load(urllib.request.urlopen(r, timeout=900))["usage"]
    dt = time.time() - t
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    print(f"  sub-agent {i + 1}: prompt {u['prompt_tokens']} tokens, cached {cached}, first token after {dt:.2f} s")
EOF
kill -INT $pid; for i in $(seq 1 30); do kill -0 $pid 2>/dev/null || break; sleep 2; done; kill -9 -$pid 2>/dev/null
exit 0
