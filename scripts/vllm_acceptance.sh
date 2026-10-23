#!/bin/bash
# Post-optimization acceptance test: a server with exactly the production arguments of prod_start.sh (MTP 4 or
# MTP_ADAPT=1, 262144 context, tool / reasoning parsers, served-name aliases) on :8081 instead of :8080, then
#   1. vllm_acceptance.py: functional tests (tools, thinking, multi-turn prefix cache, mixed load, max context, ...)
#   2. vllm_bench.py decode / conc / long / image and the reference lane (ref_lane.sh): performance
#   3. the server log: errors, OOM warnings; VRAM
# Production is not touched. usage: vllm_acceptance.sh   (env MODEL, LEN, MTP_ADAPT, SKIP_MAX=1 skips the 250K test)
TOOLS=$(cd "$(dirname "$0")" && pwd); REPO=$(dirname "$TOOLS")   # this directory, the repository root
V=$HOME/vllm-env/bin
MD=$HOME/models
export PATH=$V:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
MODEL=${MODEL:-$MD/Qwen3.8-27B-SQ06}
LEN=${LEN:-262144}
NSPEC=4; DEPTHS=
if [ "${MTP_ADAPT:-0}" = 1 ]; then NSPEC=7; DEPTHS="VLLM_TURING_MTP_DEPTHS=2,3,4,5,6,7"; fi
BATCH=$((6720 + 2 * (NSPEC + 1)))
log=/tmp/vllm-acceptance.log
echo "governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor), MTP $NSPEC ${DEPTHS:+(adaptive)}, context $LEN"
env $DEPTHS VLLM_TURING_AR8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 setsid $V/vllm serve $MODEL \
    --served-model-name qwen3.8-27b qwen q \
    --host 127.0.0.1 --port 8081 --api-key "$(cat $MD/api-key.txt)" --tensor-parallel-size 2 --dtype float16 \
    --max-model-len $LEN --gpu-memory-utilization 0.95 --max-num-seqs 2 --max-num-batched-tokens $BATCH \
    --attention-backend FLASHINFER --kv-cache-dtype fp8 --block-size 1680 \
    --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$NSPEC}" \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 > $log 2>&1 &
pid=$!
t0=$(date +%s)
for i in $(seq 1 750); do
    sleep 2; kill -0 $pid 2>/dev/null || break
    [ "$(curl -s -o /dev/null -w '%{http_code}' localhost:8081/health)" = 200 ] && break
done
[ "$(curl -s -o /dev/null -w '%{http_code}' localhost:8081/health)" = 200 ] || { echo "start failed"; grep Error $log | tail -3; kill -9 -$pid; exit 1; }
echo "ready after $(( $(date +%s) - t0 )) s; $(grep -hE 'GPU KV cache size' $log | sed -E 's/.*\] //' | head -1)"
echo "=== 1. functional"
$V/python $TOOLS/vllm_acceptance.py http://127.0.0.1:8081 ${SKIP_MAX:+--skip-max}
echo "=== 2. performance"
$V/python $TOOLS/vllm_bench.py http://127.0.0.1:8081 decode 3 | grep -E "mean|TOTAL"
$V/python $TOOLS/vllm_bench.py http://127.0.0.1:8081 conc 2 | grep -E "TOTAL|round 2"
$V/python $TOOLS/vllm_bench.py http://127.0.0.1:8081 long 8 32 128
$V/python $TOOLS/vllm_bench.py http://127.0.0.1:8081 image /tmp/vision_test.png /tmp/vision_ocr.png /tmp/vision_test_16m.png | cut -c1-180
$TOOLS/ref_lane.sh acceptance $MODEL 2>&1 | grep "ref lane"
echo "=== 3. server log"
echo "VRAM $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')"
echo "errors: $(grep -cE 'Traceback|ERROR|out of memory|OutOfMemory' $log), OOM-mapping warnings: $(grep -c 'memory mapping failed' $log), health $(curl -s -o /dev/null -w '%{http_code}' localhost:8081/health)"
grep -hE "SpecDecoding metrics" $log | tail -1 | sed -E 's/.*\] //' | cut -c1-160
kill -INT $pid; for i in $(seq 1 30); do kill -0 $pid 2>/dev/null || break; sleep 2; done; kill -9 -$pid 2>/dev/null
