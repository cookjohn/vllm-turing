#!/bin/bash
# A/B of the KV block size with production settings (fp16 GDN state, tail overlap, async decode steps, fdec): 1680
# (GDN state pages padded 100 %) vs 880 (5 %): 6 concurrent 30k-token requests (KV usage / waiting), 32K prefill,
# 120k decode, short decode. Run under probe_safe.sh (stops production): PW=... probe_safe.sh 3000 ~/tools/block_ab.sh
TOOLS=$(cd "$(dirname "$0")" && pwd); REPO=$(dirname "$TOOLS")   # this directory, the repository root
for b in 1680 880; do
    EXTRA_ENV="VLLM_TURING_TAIL=1 VLLM_TURING_ASYNC_SPEC=1" BLOCK=$b MAXSEQ=6 PHASES="kvload prefill decodelong decode" \
        TAG=blk$b $TOOLS/vllm_util_probe.sh > /tmp/vutil-blk$b.out 2>&1
done
