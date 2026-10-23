#!/bin/bash
# Trial of vllm_turing features on the production configuration: a test server on :8081 per run "name|VAR=value ...|util",
# then (1) greedy decode texts vs the production baseline saved in BASE (bench_text_diff.py), decode tok/s, (2) prefill
# rate at LONG k tokens, (3) perplexity on /tmp/ppl/{zh,code,en}.txt (production 6.5614 / 1.4880 / 2.4638), (4) a
# 2-stream and a mixed-load check, (5) allocator mapping failures in the log. Production is stopped first and restarted
# with prod_start.sh at the end (PROD=0: left down; the monitor holds client requests up to 15 minutes each).
# usage: PW=<sudo password> RUNS="name|VAR=value,VAR2=value|0.95 ..." vllm_feature_trial.sh   (env BASE, LONG "32 128",
#        PPL=0 skips the perplexity, LEN context (200000: a first compile of a new graph needs KV headroom),
#        STREAMS "1 2 4 6" concurrent stream counts (default 2), SPEC=1 adds spec_accept.py, EXTRA_ARGS more
#        vllm serve arguments for every run). With production up, run it under probe_safe.sh with PROD=0.
TOOLS=$(cd "$(dirname "$0")" && pwd); REPO=$(dirname "$TOOLS")   # this directory, the repository root
V=$HOME/vllm-env/bin
MD=$HOME/models
export PATH=$V:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
BASE=${BASE:-/tmp/conc/base_texts.jsonl}
LONG=${LONG:-32 128}
NSPEC=4; MAXSEQ=6; BATCH=$((6720 + MAXSEQ * (NSPEC + 1)))
K=$(cat $MD/api-key.txt)
echo "$PW" | sudo -S -p "" systemctl stop vllm-server 2>/dev/null; sleep 3
run_cfg() {   # $1 = name, $2 = env assignments, $3 = gpu memory utilization (default 0.95), $4 = "car": custom all-reduce
    name=$1; envs=$2; util=${3:-0.95}; car=$4
    log=/tmp/vllm-trial-$name.log
    echo "=== $name ($envs) $car"
    t0=$(date +%s)
    # GPU peer access as in prod_start.sh: NVLink up -> NCCL over the link (custom all-reduce only for "car")
    P2PV=NCCL_P2P_DISABLE=1; CARF=
    if nvidia-smi nvlink --status 2>/dev/null | grep -q "GB/s"; then
        P2PV=NCCL_P2P_DISABLE=0
        [ "$car" = car ] || CARF=--disable-custom-all-reduce
    fi
    env ${envs//,/ } VLLM_TURING_AR8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $P2PV \
      setsid $V/vllm serve $MD/Qwen3.8-27B-SQ06 $CARF --served-model-name qwen3.8-27b --host 127.0.0.1 --port 8081 \
      --api-key "$K" --tensor-parallel-size 2 --dtype float16 --enable-prompt-tokens-details \
      --max-model-len ${LEN:-200000} --gpu-memory-utilization $util --max-num-seqs $MAXSEQ --max-num-batched-tokens $BATCH \
      --attention-backend FLASHINFER --kv-cache-dtype fp8 --block-size 1680 --long-prefill-token-threshold 1680 \
      --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$NSPEC}" \
      --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
      --chat-template $MD/qwen3.8-chat-template-inline-system.jinja $EXTRA_ARGS > $log 2>&1 &
    pid=$!
    for i in $(seq 1 600); do
        sleep 2; kill -0 $pid 2>/dev/null || break
        [ "$(curl -s -o /dev/null -w '%{http_code}' localhost:8081/health)" = 200 ] && break
    done
    if [ "$(curl -s -o /dev/null -w '%{http_code}' localhost:8081/health)" != 200 ]; then
        echo "start failed after $(( $(date +%s) - t0 )) s"; grep -E "Error|error:|Traceback" $log | tail -5
        kill -9 -$pid 2>/dev/null; return 1
    fi
    echo "up after $(( $(date +%s) - t0 )) s; $(grep -hE 'GPU KV cache size' $log | sed -E 's/.*\] //' | tail -1)"
    grep -hoE "Actual usage is .*CUDAGraph memory" $log | tail -1
    grep -h "vllm_turing" $log | sed -E 's/.*\] //' | sort -u | head -12
    grep -h "all-reduce backends" $log | sed -E 's/.*\] //' | head -1 | cut -c1-80
    txt=/tmp/conc/trial_texts_$name.jsonl; rm -f $txt
    BENCH_TEXT=$txt $V/python $TOOLS/vllm_bench.py http://127.0.0.1:8081 decode 2 2>&1 | grep -E "^mean|TOTAL"
    $V/python $TOOLS/bench_text_diff.py $BASE $txt 2>&1 | tail -4
    $V/python $TOOLS/vllm_bench.py http://127.0.0.1:8081 long $LONG 2>&1 | grep -E "^long|follow"
    if [ "${PPL:-1}" = 1 ]; then
        $V/python $TOOLS/vllm_bench.py http://127.0.0.1:8081 ppl /tmp/ppl/zh.txt /tmp/ppl/code.txt /tmp/ppl/en.txt 2>&1 | grep "^ppl"
    fi
    $V/python $TOOLS/vllm_conc_bench.py http://127.0.0.1:8081 streams ${STREAMS:-2} 2>&1 | grep "^streams"
    [ -n "$SPEC" ] && $V/python $TOOLS/spec_accept.py http://127.0.0.1:8081 2>&1 | tail -1
    $V/python $TOOLS/vllm_acceptance.py http://127.0.0.1:8081 --only "mixed load" 2>&1 | grep -E "PASS|FAIL" | head -3
    curl -s -o /dev/null -w "health %{http_code}\n" localhost:8081/health
    grep -ciE "error|traceback" $log | sed 's/^/log errors: /'
    grep -c "memory mapping failed" $log | sed 's/^/allocator mapping failures: /'
    kill -INT $pid; for i in $(seq 1 30); do kill -0 $pid 2>/dev/null || break; sleep 2; done; kill -9 -$pid 2>/dev/null
    sleep 3
}
for r in $RUNS; do IFS="|" read -r n e u c <<< "$r"; run_cfg "$n" "$e" "$u" "$c"; done
[ "${PROD:-1}" = 1 ] && PW="$PW" $TOOLS/prod_start.sh
