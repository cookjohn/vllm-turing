#!/bin/bash
# SmoothQuant calibration run: the W8A16 model served eagerly (forward pre-hooks of vllm_turing.calib record the
# per-input-channel activation max of every quantized linear layer), MTP 1 so the draft layer is calibrated too, fed
# with corpus chunks + chat prompts (calib_feed.py). Stats -> $OUT/rank<r>[_m1].pt. Production must be down.
# usage: calib_run.sh   (env MODEL, OUT)
TOOLS=$(cd "$(dirname "$0")" && pwd); REPO=$(dirname "$TOOLS")   # this directory, the repository root
V=$HOME/vllm-env/bin
export PATH=$V:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
MODEL=${MODEL:-$HOME/models/Qwen3.8-27B-W8A16-g128}
OUT=${OUT:-$HOME/models/calib-w8a16}
rm -rf $OUT
log=/tmp/vllm-calib.log
VLLM_TURING_CALIB=$OUT NCCL_P2P_DISABLE=1 setsid $V/vllm serve $MODEL --served-model-name q --host 127.0.0.1 --port 8081 \
    --api-key "$(cat $HOME/models/api-key.txt)" --tensor-parallel-size 2 --dtype float16 \
    --max-model-len 17000 --gpu-memory-utilization 0.92 --max-num-seqs 1 --max-num-batched-tokens 4096 \
    --attention-backend FLASHINFER --enforce-eager --compilation-config '{"mode": 0}' \
    --speculative-config '{"method":"mtp","num_speculative_tokens":1}' > $log 2>&1 &
pid=$!
for i in $(seq 1 450); do
    sleep 2; kill -0 $pid 2>/dev/null || break
    [ "$(curl -s -o /dev/null -w '%{http_code}' localhost:8081/health)" = 200 ] && break
done
grep -E "vllm.turing|turing" $log | sed -E 's/^.*\] //' | sort -u | head -8
python3 $TOOLS/calib_feed.py http://127.0.0.1:8081 4096 4
kill -INT $pid; for i in $(seq 1 30); do kill -0 $pid 2>/dev/null || break; sleep 2; done; kill -9 -$pid 2>/dev/null   # this server's process group only
ls -la $OUT
$V/python - "$OUT" <<'EOF'
import glob, sys, torch
for f in sorted(glob.glob(sys.argv[1] + "/*.pt")):
    d = torch.load(f)
    ks = sorted(d)
    print(f, len(ks), "modules; e.g.", ks[0], tuple(d[ks[0]].shape), "max", round(float(d[ks[0]].max()), 2))
    big = sorted(((float(v.max() / v.median().clamp(min=1e-6)), k) for k, v in d.items()), reverse=True)[:4]
    print("  largest max/median:", [(round(r, 1), k.split(".", 3)[-1]) for r, k in big])
EOF
