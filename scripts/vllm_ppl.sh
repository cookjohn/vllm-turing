#!/bin/bash
# Perplexity of the vLLM server like llama-perplexity (16384-token chunks, second half scored) on the zh / code / en
# corpora (/tmp/ppl, made by ppl_corpus.py; llama.cpp Q4_K_M single GPU: 6.5789 / 1.4866 / 2.4700), for each config
# "name|extra vllm args|VAR=value ...". Prompt logprobs need memory for full-vocabulary logits: utilization 0.80,
# 1024-token batches, 17k context. Production must be down.
# usage: vllm_ppl.sh "fp16head||VLLM_TURING_LMHEAD=0" "int8head||"
TOOLS=$(cd "$(dirname "$0")" && pwd); REPO=$(dirname "$TOOLS")   # this directory, the repository root
V=$HOME/vllm-env/bin
export PATH=$V:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
MODEL=${MODEL:-$HOME/models/Qwen3.8-27B-W8A8}
[ -s /tmp/ppl/zh.txt ] || python3 $TOOLS/ppl_corpus.py > /dev/null
for cfg in "$@"; do
    IFS="|" read -r name extra envs <<< "$cfg"
    log=/tmp/vllm-ppl-$name.log
    env $envs NCCL_P2P_DISABLE=1 setsid $V/vllm serve $MODEL --served-model-name q --host 127.0.0.1 --port 8081 \
        --api-key "$(cat $HOME/models/api-key.txt)" --tensor-parallel-size 2 --dtype float16 \
        --max-model-len 17000 --gpu-memory-utilization ${UTIL:-0.80} --max-num-seqs 1 --max-num-batched-tokens 1024 \
        --attention-backend FLASHINFER $extra > $log 2>&1 &
    pid=$!
    for i in $(seq 1 450); do
        sleep 2; kill -0 $pid 2>/dev/null || break
        [ "$(curl -s -o /dev/null -w '%{http_code}' localhost:8081/health)" = 200 ] && break
    done
    echo "=== $name ($envs $extra)"
    $V/python $TOOLS/vllm_bench.py http://127.0.0.1:8081 ppl /tmp/ppl/zh.txt /tmp/ppl/code.txt /tmp/ppl/en.txt 2>&1 | tail -4
    grep -h "int8 LM head" $log | head -1 | sed -E 's/^.*\] //'
    kill -INT $pid; for i in $(seq 1 30); do kill -0 $pid 2>/dev/null || break; sleep 2; done; kill -9 -$pid 2>/dev/null; sleep 5   # this server's process group only
done
