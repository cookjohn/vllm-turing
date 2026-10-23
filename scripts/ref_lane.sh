#!/bin/bash
# The reference performance lane of weicj/vLLM-2080Ti-Definitive (their launcher's numbers in profiles/*/README.md),
# against our server on :8081: completions, exact 4096 prompt tokens of " the" filler + 128 generated (ignore_eos, output
# restricted to " the": every speculative draft is accepted, i.e. the speculative ceiling), warm-up excluded, median of
# three; then one 32768 / 512 run. Each request gets a unique prompt variant (no prefix-cache hits).
# usage: ref_lane.sh <name> [model dir]   (called by vllm_tp_trial.sh when LANE is set)
TOOLS=$(cd "$(dirname "$0")" && pwd); REPO=$(dirname "$TOOLS")   # this directory, the repository root
V=$HOME/vllm-env/bin
name=$1
model=${2:-$HOME/models/Qwen3.8-27B-SQ06}
out=/tmp/lane-$name.jsonl
rm -f $out
served=$(curl -s -H "Authorization: Bearer $(cat $HOME/models/api-key.txt)" localhost:8081/v1/models \
    | $V/python -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])")
run() {   # label, prompt tokens, gen tokens
    $V/python $TOOLS/ref_profile_request.py --model-dir "$model" --served-name "$served" \
        --base-url http://127.0.0.1:8081/v1 --endpoint completions --prompt-tokens $2 --gen-tokens $3 --label $1 \
        --prompt-variant "lane-$name-$1-$RANDOM$RANDOM" --out $out --read-timeout 1800 --ignore-eos --pure-filler \
        --allowed-token-text " the" > /dev/null 2>&1
}
run warmup 4096 128
for i in 1 2 3; do run lane4k-$i 4096 128; done
run lane32k 32768 512
$V/python - "$out" <<'EOF'
import json, statistics, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
four = [r for r in rows if r["label"].startswith("lane4k")]
long = [r for r in rows if r["label"] == "lane32k"]
ok = all(r.get("allowed_token_only") for r in four + long)
med = lambda k, rs: statistics.median(r[k] for r in rs if r.get(k)) if any(r.get(k) for r in rs) else 0
print(f"ref lane 4K/128: prefill {med('prefill_tok_s', four):.1f} / decode {med('decode_tok_s', four):.2f} tok/s | "
      f"32K/512: prefill {med('prefill_tok_s', long):.1f} / decode {med('decode_tok_s', long):.2f} tok/s"
      f"{'' if ok else '  (output not restricted to the filler token!)'}")
EOF
