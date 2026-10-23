#!/bin/bash
# Compute utilization and waiting time of the production configuration (MTP4, 262144, FP8 KV, all vllm_turing parts):
# one server (:8081) with the torch profiler, two profiled phases - (1) one fresh 32K-token prefill (max_tokens 1),
# (2) a 512-token decode of a code request at short context - with nvidia-smi samples (SM clock, power, "GPU
# utilization") during each, then tests/trace_util.py on both ranks' traces.
# usage: vllm_util_probe.sh   (env MODEL, LEN 262144, NTOK 32768, GEN 512, EXTRA_ENV "VAR=value ..." for the server,
#        TAG: output under /tmp/vutil-TAG instead of /tmp/vutil, UTIL GPU memory fraction, PHASES "prefill decode"
#        (decode6: 6 concurrent decode requests; decodelong: decode after a CTX-token prompt (default 120000) that
#        was prefilled unprofiled first; MAXSEQ sets --max-num-seqs, default 2, 6 with decode6 / kvload; kvload: KVN
#        (default 6) concurrent requests with distinct CTX2-token prompts (default 30000) generating GEN2 tokens
#        (default 256), unprofiled, with vLLM's KV usage / running / waiting sampled from /metrics),
#        BLOCK (--block-size, default 1680), SSM_DTYPE (GDN state dtype, default float16 as in production), PREFILL_CAP
#        (--long-prefill-token-threshold, default the smallest multiple of BLOCK >= 1680)
TOOLS=$(cd "$(dirname "$0")" && pwd); REPO=$(dirname "$TOOLS")   # this directory, the repository root
V=$HOME/vllm-env/bin
export PATH=$V:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
MODEL=${MODEL:-$HOME/models/Qwen3.8-27B-SQ06}
KEY=$(cat $HOME/models/api-key.txt)
OUT=/tmp/vutil${TAG:+-$TAG}; rm -rf $OUT; mkdir -p $OUT
log=/tmp/vllm-util${TAG:+-$TAG}.log
# P2P as in prod_start.sh: NVLink up -> NCCL over the link plus vLLM's custom all-reduce (CUSTOM_AR=0: NCCL only);
# else no NCCL P2P (the plugin's host all-reduce)
B=${BLOCK:-1680}
MS=${MAXSEQ:-$(case "$PHASES" in *decode6*|*kvload*) echo 6;; *) echo 2;; esac)}
CAP=${PREFILL_CAP:-$(( (1680 + B - 1) / B * B ))}
# prefill chunks are whole blocks in align mode: the block multiple nearest above 6720, + the verify tokens
BT=$(( (6720 + B - 1) / B * B + MS * 5 ))
P2PV=NCCL_P2P_DISABLE=1; CARF=; ALLOCV=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if nvidia-smi nvlink --status 2>/dev/null | grep -q "GB/s"; then
    P2PV=NCCL_P2P_DISABLE=0
    [ "${CUSTOM_AR:-1}" = 1 ] || CARF=--disable-custom-all-reduce
fi
env $EXTRA_ENV VLLM_TURING_AR8=1 $ALLOCV $P2PV setsid $V/vllm serve $MODEL $CARF \
    --served-model-name q --host 127.0.0.1 --port 8081 --api-key "$KEY" --tensor-parallel-size 2 --dtype float16 \
    --max-model-len ${LEN:-262144} --gpu-memory-utilization ${UTIL:-0.95} --max-num-seqs $MS --max-num-batched-tokens $BT \
    --attention-backend FLASHINFER --kv-cache-dtype fp8 --block-size $B --mamba-ssm-cache-dtype ${SSM_DTYPE:-float16} \
    --long-prefill-token-threshold $CAP \
    --speculative-config '{"method":"mtp","num_speculative_tokens":4}' \
    --profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$OUT\",\"torch_profiler_with_stack\":false,\"active_iterations\":3000}" \
    > $log 2>&1 &
pid=$!
for i in $(seq 1 600); do
    sleep 2; kill -0 $pid 2>/dev/null || break
    [ "$(curl -s -o /dev/null -w '%{http_code}' localhost:8081/health)" = 200 ] && break
done
[ "$(curl -s -o /dev/null -w '%{http_code}' localhost:8081/health)" = 200 ] || { echo "start failed"; tail -3 $log; kill -9 -$pid; exit 1; }
req() {  # $1 = mode (prefill|decode|warm|warmlong|decodelong|kvload), $2 = tokens
    $V/python - "$1" "$2" "$MODEL" "${CTX:-120000}" "${CTX2:-30000}" <<'EOF'
import json, os, random, sys, time, urllib.request
from transformers import AutoTokenizer
mode, n, model = sys.argv[1], int(sys.argv[2]), sys.argv[3]
key = open(os.path.expanduser("~/models/api-key.txt")).read().strip()
if mode in ("prefill", "warm", "warmlong", "decodelong", "kvload"):
    tk = AutoTokenizer.from_pretrained(model)
    src = open(os.path.expanduser("~/vllm-env/lib/python3.12/site-packages/vllm/v1/worker/gpu/model_runner.py")).read()
    ids = tk.encode(src, add_special_tokens=False)
    nc = int(sys.argv[4]) if mode == "decodelong" else int(sys.argv[5]) if mode == "kvload" else n
    ids = (ids * (nc // len(ids) + 1))[:nc]
    # warmlong / decodelong: one fixed prompt (the second request hits the prefix cache), then a generation there
    longp = mode in ("warmlong", "decodelong")
    head = "long-context decode probe\n" if longp else f"utilization probe {random.random()}\n"
    gen = mode in ("decodelong", "kvload")
    body = {"model": "q", "prompt": head + tk.decode(ids) + ("\n\nExplain the code above in detail.\n" if gen else ""),
            "max_tokens": n if gen else 1, "temperature": 0, "ignore_eos": gen}
else:
    body = {"model": "q", "max_tokens": n, "temperature": 0, "ignore_eos": True,
            "prompt": "Write a Python module implementing a thread-safe LRU cache with per-entry TTL, size limits, "
                      "statistics and an async API, with docstrings and pytest unit tests.\n\n```python\n"}
r = urllib.request.Request("http://127.0.0.1:8081/v1/completions", json.dumps(body).encode(),
                           {"Authorization": "Bearer " + key, "Content-Type": "application/json"})
t = time.time(); u = json.load(urllib.request.urlopen(r, timeout=3600))["usage"]; dt = time.time() - t
print(f"{mode}: prompt {u['prompt_tokens']} tokens, generated {u['completion_tokens']}, {dt:.2f} s")
import os
if mode == "kvload":
    print(f"kvload done at +{time.time() - float(os.environ['REQT0']):.1f} s")
print(f"TOKENS {u['prompt_tokens'] if not mode.startswith('decode') else u['completion_tokens']} CTX {u['prompt_tokens'] + u['completion_tokens'] // 2}")
EOF
}
phase() {  # $1 = name, $2 = mode, $3 = tokens, $4 = concurrent requests (default 1)
    nvidia-smi --query-gpu=index,clocks.sm,power.draw,utilization.gpu --format=csv,noheader,nounits -lms 100 \
        > $OUT/smi-$1.csv &
    smi=$!
    curl -s -X POST -H "Authorization: Bearer $KEY" localhost:8081/start_profile > /dev/null
    if [ "${4:-1}" -gt 1 ]; then
        rm -f /tmp/vutil-req-$1-*.txt
        # wait for these PIDs only: a bare wait also waits for the nvidia-smi sampler above, which never exits
        # (10-22: hung the probe for 6 hours with production stopped)
        pids=
        for i in $(seq 1 $4); do req $2 $3 > /tmp/vutil-req-$1-$i.txt & pids="$pids $!"; done
        wait $pids
        cat /tmp/vutil-req-$1-*.txt | awk '/^TOKENS/ {t += $2; c = $4; next} {print} END {print "TOKENS", t, "CTX", c}'             > /tmp/vutil-req-$1.txt
    else
        req $2 $3 > /tmp/vutil-req-$1.txt
    fi
    curl -s -X POST -H "Authorization: Bearer $KEY" localhost:8081/stop_profile > /dev/null
    kill $smi
    sleep 30
    mkdir -p $OUT/$1; mv $OUT/*.pt.trace.json.gz $OUT/$1/ 2>/dev/null; rm -f $OUT/profiler_out_*.txt 2>/dev/null
    grep -v TOKENS /tmp/vutil-req-$1.txt
}
kvload() {   # KVN concurrent distinct prompts; KV usage / running / waiting from /metrics every second
    echo "=== kvload: ${KVN:-6} x ${CTX2:-30000}-token prompts, ${GEN2:-256} tokens each, block $B"
    ( while true; do
          curl -s -H "Authorization: Bearer $KEY" localhost:8081/metrics | awk -v t=$(date +%s) \
              '/^vllm:kv_cache_usage_perc/ {k = $NF} /^vllm:num_requests_running/ {r = $NF}
               /^vllm:num_requests_waiting/ {w = $NF} END {print t, k, r, w}'
          sleep 1
      done ) > $OUT/kvload-metrics.txt &
    local sam=$! pids=
    export REQT0=$(date +%s.%N)
    rm -f /tmp/vutil-kv-*.txt
    for i in $(seq 1 ${KVN:-6}); do
        req kvload ${GEN2:-256} > /tmp/vutil-kv-$i.txt & pids="$pids $!"
    done
    wait $pids   # these PIDs only (the sampler never exits)
    kill $sam
    cat /tmp/vutil-kv-*.txt | grep -v TOKENS
    awk '$2 != "" {if ($2 > k) k = $2; if ($4 > w) w = $4; if ($3 > r) r = $3; n++}
         END {printf "KV usage max %.1f%%, running max %d, waiting max %d (%d samples)\n", k * 100, r, w, n}' \
        $OUT/kvload-metrics.txt
}
req warm 2048 > /dev/null
req decode 64 > /dev/null
PHASES=${PHASES:-prefill decode}
for p in $PHASES; do
    case $p in
        prefill) phase prefill prefill ${NTOK:-32768} ;;
        decode)  phase decode decode ${GEN:-512} ;;
        decode6) phase decode6 decode ${GEN:-512} 6 ;;
        decodelong) req warmlong ${CTX:-120000} > /dev/null; phase decodelong decodelong ${GEN:-512} ;;
        kvload) kvload ;;
    esac
done
echo "allocator mapping failures: $(grep -c "memory mapping failed" $log)"
kill -INT $pid; for i in $(seq 1 30); do kill -0 $pid 2>/dev/null || break; sleep 2; done; kill -9 -$pid 2>/dev/null
for p in $PHASES; do
    [ $p = kvload ] && continue   # reported above, no trace
    read -r _ tok _ ctx <<< "$(grep TOKENS /tmp/vutil-req-$p.txt)"
    echo "=== $p"
    awk -F', ' '$4 > 20 {n[$1]++; c[$1] += $2; w[$1] += $3; u[$1] += $4} END {for (g in n) printf "GPU%s: SM clock %.0f MHz, power %.0f W, nvidia-smi utilization %.0f%% (%d samples)\n", g, c[g]/n[g], w[g]/n[g], u[g]/n[g], n[g]}' $OUT/smi-$p.csv | sort
    for r in 0 1; do
        mhz=$(awk -F', ' -v g=$r '$1 == g && $4 > 20 {n++; c += $2} END {print (n ? c / n : 1545)}' $OUT/smi-$p.csv)
        f=$(ls $OUT/$p/rank$r.*.pt.trace.json.gz 2>/dev/null | head -1)
        [ -n "$f" ] || { echo "rank $r: no trace"; continue; }
        echo "rank $r:"
        $V/python $REPO/tests/trace_util.py $f $(case $p in decode*) echo decode;; *) echo $p;; esac) $tok $mhz $ctx
    done
done
