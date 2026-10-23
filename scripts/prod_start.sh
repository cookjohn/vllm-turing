#!/bin/bash
# Production (vLLM, since 10-19): vLLM 0.31 (~/vllm-env) serving Qwen3.8-27B as SmoothQuant INT8 W8A8
# (~/models/Qwen3.8-27B-SQ06: made from the BF16 original by w8a8_convert.py --smooth ~/models/calib-w8a16 --alpha 0.6;
# PPL zh/code/en 6.55/1.488/2.461, better than a llama.cpp Q4_K_M GGUF) with tensor parallelism over both 2080 Ti
# (NVLink bridge since 10-22, before: PCIe without P2P; see P2P below), FP8 KV cache, FlashInfer attention, MTP
# speculative decoding (4 tokens; block size 880, a multiple of the 5-token verify step, keeps the
# hybrid-model prefix cache working), on :8080 (OpenAI and Anthropic API, key in ~/models/api-key.txt).
# Turing speedups come from the vllm_turing plugin (this repository, pip install -e; loaded automatically): int8 GEMV for decode, tensor-core
# GDN prefill, int8-QK prefill attention, host-memory all-reduce, int8-compressed prefill all-reduce (VLLM_TURING_AR8=1),
# int8 LM head and input embedding, pruned draft vocabulary with a balanced 4-bit draft head, sync-free MTP draft steps,
# a low-memory vision encoder and one FlashInfer workspace counted in the memory profile; and
# patches/flashinfer_spec_decode.py applied to the installed vLLM (FULL CUDA graphs for MTP verify). Expandable
# allocator segments: ~1 GiB more KV cache than the default allocator (whose segments the prefill temporaries fragment).
# Context: the model's native 262144 tokens. The first start of a new configuration compiles in-process and has ~0.8 GiB
# less KV room: at 262144 it fails once and the unit restarts (Restart=on-failure) with the compile cache warm.
# MTP_ADAPT=1: adaptive draft depth 2..7 per step (vllm_turing/mtpdepth.py; ~3% faster decode, less KV headroom).
# Tool calls (qwen3_coder XML format) and thinking (reasoning_content) are parsed. vLLM only accepts its served model
# names: qwen3.8-27b, qwen, q.
# Runs as the transient unit vllm-server (systemd-run: a stop removes it, it does not survive a reboot; rerun this
# script). Startup takes 2-7 minutes (torch.compile / CUDA graphs / FlashInfer JIT, cached after the first run).
# usage: PW=<sudo password> prod_start.sh   (env MODEL, LEN = max context, UTIL = GPU memory fraction, NSPEC = MTP tokens,
#        MTP_ADAPT=1 adaptive depth up to 7, MAXSEQ = concurrent requests, OFFLOAD_GB = host RAM for evicted KV,
#        TAIL=1 prefill all-reduce overlap, ASYNC=0 synced decode steps)
TOOLS=$(cd "$(dirname "$0")" && pwd); REPO=$(dirname "$TOOLS")   # this directory, the repository root
MD=$HOME/models
V=$HOME/vllm-env/bin
# optimization periods: `touch scripts/NO_PROD` keeps production down (scripts that end with prod_start.sh leave the
# GPUs free); remove the file to start production again
if [ -e $TOOLS/NO_PROD ]; then
    echo "production start skipped: $TOOLS/NO_PROD exists"
    exit 0
fi
MODEL=${MODEL:-$MD/Qwen3.8-27B-SQ06}
LEN=${LEN:-262144}   # the model's native maximum (KV pool ~290k tokens with MTP 4 at util 0.95)
UTIL=${UTIL:-0.95}
NSPEC=${NSPEC:-4}
DEPTHS=
if [ "${MTP_ADAPT:-0}" = 1 ]; then
    NSPEC=7
    DEPTHS="--setenv=VLLM_TURING_MTP_DEPTHS=2,3,4,5,6,7"
fi
# TAIL (default auto = on with NVLink, see P2P below): prefill all-reduce / compute overlap (vllm_turing/tailoverlap.py:
# each decoder layer's per-token tail on token chunks, the all-reduces on a side stream through the copy-engine
# all-reduce over P2P, cear.py). 10-22: 32K prefill 3385 vs 3190 tok/s, 128K 2454 vs 2350, decode and perplexity
# unchanged, KV -1%. Without peer access (NCCL / host all-reduce on SMs) it was slower. The first start with it compiles
# a new graph: at 262144 that start can fail once for lack of KV headroom (the compile holds GPU memory during the
# memory profile) and the unit restarts warm
TAILV=${TAIL:-auto}
# ASYNC=1 (default): decode steps without waiting for the GPU (vllm_turing/nosync.py, VLLM_TURING_ASYNC_SPEC: FlashInfer
# planned from the CPU's length upper bound, exact lengths fixed on the GPU; the CPU prepares the next step while the
# GPU runs this one): GPU idle in decode 18% -> 4%, ~35 instead of ~38-42 ms per step, decode +8-13%. Greedy output at
# temperature 0 is no longer bit-identical run to run (FlashInfer may split the KV differently when the bound adds a
# page: the same attention summed in another order). ASYNC=0: the synced steps
ASYNCSPEC=
[ "${ASYNC:-1}" = 1 ] && ASYNCSPEC="--setenv=VLLM_TURING_ASYNC_SPEC=1"
# SSM_DTYPE (default float16): GDN recurrent state cache dtype. The checkpoint asks for float32 (SSM_DTYPE=auto); the
# MTP verify writes the full state after every draft token (rollback), 1.57 MB per head group per token, which made the
# GDN update the third-largest kernel at 6 streams. fp16 halves that traffic (all kernels still compute in fp32).
# 10-23: perplexity 6.5506 / 1.4883 / 2.4618 vs 6.5516 / 1.4877 / 2.4648, needles found, 6 streams 368 vs ~345 tok/s;
# simulated decode drift with the model's decay rates levels off at 0.08 % median / 0.33 % max output error
SSM_DTYPE=${SSM_DTYPE:-float16}
# BLOCK: KV block size; multiple of 16 (FlashInfer) and of NSPEC + 1 (prefix cache in align mode), at least the GDN state
# page, which vLLM pads every GDN block up to (one pool for both): 1680 matched the fp32 state; with fp16 state that
# padded each GDN block by 100 %. 880 (fp16 default): 5 % padding. 10-23 A/B (fp16 state): KV pool 281,617 -> 295,326
# tokens; 6 concurrent 30k requests 96 % -> 86 % KV, all 6 running instead of 5 + 1 queued; a continued conversation
# re-prefills at most one block (880 instead of 1680 tokens; 120k decode window 45.6 -> 42.6 ms per step); 32K prefill
# and short decode unchanged. (Before that, real agent traffic with 6 running requests kept the KV pool at 85-100 %
# and up to 15 requests waiting.)
BLOCK=${BLOCK:-$([ "$SSM_DTYPE" = float16 ] && echo 880 || echo 1680)}
# concurrent requests (10-21: 2 -> 6; the 3rd request used to queue behind two long generations). MTP4 verify of 3..6
# requests = 15..30 rows per GEMM: vllm_turing's int8 tensor-core kernel (1.3-1.4x CUTLASS). Each running request also
# holds 3 x (1 + NSPEC) (+1 while a block boundary is passed) blocks of GDN state (~12 measured at MTP4) while it runs
MAXSEQ=${MAXSEQ:-6}
# prefill chunks are whole blocks in align mode (4 x 1680 = 6720, 8 x 880 = 7040): a larger token budget only inflates
# the memory profile's dummy batch; + the verify tokens of the decoding requests
BATCH=$(( (6720 + BLOCK - 1) / BLOCK * BLOCK + MAXSEQ * (NSPEC + 1) ))
# OFFLOAD_GB (default 24, 0 = off): prefix-cache blocks evicted from the GPU are kept in that much pinned host RAM (both
# GPUs together; vLLM native offloading, the blocks carry their GDN state groups too) and loaded back over PCIe instead of
# recomputed. 10-23: concurrent sub-agents evicted the main agent session (~160k tokens, half the KV pool) between its
# turns, every turn re-prefilled it (70-120 s). Test (offload_test.sh: 120k session, 9 x 30k sub-agents, session again):
# cached 0 / 47.9 s without offload, 118800 of 120016 / 1.65 s with it, same greedy continuation. Offloading needs the
# cumem allocator with expandable segments (KV pool 295,326 -> 280,394 tokens)
OFFLOAD_GB=${OFFLOAD_GB:-24}
OFFLOAD=
[ "$OFFLOAD_GB" != 0 ] && OFFLOAD="--kv-offloading-size $OFFLOAD_GB --enable-cumem-allocator"
# while another request is running, a long prompt is prefilled in chunks of at most PREFILL_CAP tokens (vLLM
# --long-prefill-token-threshold; not applied to a request that runs alone): a 64K prefill next to a streaming request
# takes 30.6 instead of 28.1 s, the stream gets 45 instead of 16 steps meanwhile (longest pause 0.94 instead of 3.3 s).
# 0 disables. Default: the smallest whole number of blocks >= 1680 tokens (1760 at block 880)
PREFILL_CAP=${PREFILL_CAP:-$(( (1680 + BLOCK - 1) / BLOCK * BLOCK ))}
# cross-session prefix reuse (SHARE=1 default): a prefix-cache hit needs the GDN state at the hit point, and vLLM keeps
# one per prompt (near its end). --enable-mamba-shared-prefix-checkpoint also stores it where a new request's prefix
# leaves an earlier one's (sub-agents sharing tools + system prompt), --prefix-match-unit 80 lets hits land on 80-token
# boundaries inside a block. 10-23 test (4 sub-agents, 47 tools, 18k tokens): cached 16720 -> 17920 of 18036, first
# token 0.49 -> 0.21 s (the first one 4.9 s); with the per-session billing header in the prompt (the proxy now drops
# it) only the junction checkpoint can share the tool block. SHARE=0: off
SHAREF=
[ "${SHARE:-1}" = 1 ] && SHAREF="--prefix-match-unit 80 --enable-mamba-shared-prefix-checkpoint"
# MONITOR=1 (default): clients reach :8080 through the monitoring proxy (~/monitor/vllm_monitor.py, unit vllm-monitor:
# live inputs / outputs / timings at http://<host>:8080/monitor?key=<API key>, logs ~/monitor/logs), vLLM listens on
# 127.0.0.1:8090 only. MONITOR=0: vLLM on 0.0.0.0:8080 directly
MONITOR=${MONITOR:-1}
# chat template = the model's, but non-leading system messages render in place (vLLM otherwise merges Claude Code's
# per-turn reminders into the leading system prompt: only ~21% of each turn's prompt was reusable from the prefix
# cache, a full re-prefill of ~100K tokens per turn) and reasoning effort high / max = xhigh (Claude Code sends high,
# the model's template rejected it with HTTP 400); made and checked by scripts/python/vllm_monitor/check_inline_system.py
CHAT_TEMPLATE=${CHAT_TEMPLATE:-$MD/qwen3.8-chat-template-inline-system.jinja}
if [ "$MONITOR" = 1 ]; then VHOST=127.0.0.1; VPORT=8090; else VHOST=0.0.0.0; VPORT=8080; fi
# a running monitor stays up: it holds the clients' requests while vLLM restarts (VMON_UPSTREAM_WAIT_S) and sends them
# when vLLM is back (new monitor code: vmon_upgrade.sh)
KEEPMON=0
[ "$MONITOR" = 1 ] && systemctl is-active --quiet vllm-monitor && KEEPMON=1
STOP="vllm-server vllm-monitor-next"
[ "$KEEPMON" = 1 ] || STOP="$STOP vllm-monitor"
echo "$PW" | sudo -S -p "" systemctl stop $STOP 2>/dev/null
echo "$PW" | sudo -S -p "" systemctl reset-failed $STOP 2>/dev/null
# the interim zero-downtime layout (vLLM on :8080, monitor on :8088 / :8089, iptables redirect of incoming :8080 to it)
# would now point at nothing: remove the redirects
for p in 8088 8089; do
    while echo "$PW" | sudo -S -p "" iptables -t nat -D PREROUTING -p tcp --dport 8080 -j REDIRECT --to-ports $p 2>/dev/null; do :; done
done
if [ "$MONITOR" = 1 ] && [ "$KEEPMON" = 0 ]; then
    echo "$PW" | sudo -S -p "" systemd-run --unit=vllm-monitor -p Restart=always -p RestartSec=2 --uid=$(id -un) --gid=$(id -gn) \
      -p WorkingDirectory=$HOME --setenv=HOME=$HOME \
      $V/python $REPO/monitor/vllm_monitor.py --listen 0.0.0.0:8080 --upstream http://127.0.0.1:$VPORT \
      --log-dir $HOME/monitor/logs
fi
# decode is partly CPU bound (Python step preparation): the performance governor is worth ~4.5% (not persistent)
# (the password goes only to sudo: with cached sudo credentials it is not read, so it must not share stdin with data)
echo "$PW" | sudo -S -p "" sh -c 'for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > "$g"; done' || true
sleep 3
# GPU peer access (P2P=auto: on when NVLink links are up, 10-22 bridge NV2 ~47 GB/s per direction): NCCL uses the link
# for prefill and vLLM's custom all-reduce takes the decode / verify all-reduces (CUSTOM_AR=0: NCCL for all). The
# custom all-reduce copies through its staging buffer in CUDA graphs too (vllm_turing customar.py), so it works with
# expandable allocator segments; vs NCCL 10-22: decode 111.9 vs 108.7 tok/s, 2 streams 145 vs 136, +16k KV tokens,
# same texts and perplexity. Without NVLink (GeForce has no PCIe P2P) NCCL must not try P2P and the plugin's
# host-memory all-reduce takes over (vllm_turing hostar.py installs only without peer access)
P2P=${P2P:-auto}
if [ "$P2P" = auto ]; then
    nvidia-smi nvlink --status 2>/dev/null | grep -q "GB/s" && P2P=1 || P2P=0
fi
ALLOC="--setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
P2PENV="--setenv=NCCL_P2P_DISABLE=1"
CARFLAG=
if [ "$P2P" = 1 ]; then
    P2PENV=
    [ "${CUSTOM_AR:-1}" = 1 ] || CARFLAG=--disable-custom-all-reduce
fi
[ "$TAILV" = auto ] && TAILV=$P2P
TAIL=
[ "$TAILV" = 1 ] && TAIL="--setenv=VLLM_TURING_TAIL=1"
echo "P2P=$P2P custom all-reduce $([ -z "$CARFLAG" ] && [ "$P2P" = 1 ] && echo on || echo off), prefill tail overlap $([ "$TAILV" = 1 ] && echo on || echo off)"
echo "$PW" | sudo -S -p "" systemd-run --unit=vllm-server -p Restart=on-failure -p RestartSec=30 --uid=$(id -un) --gid=$(id -gn) \
  -p WorkingDirectory=$HOME $P2PENV --setenv=VLLM_API_KEY="$(cat $MD/api-key.txt)" \
  --setenv=HOME=$HOME --setenv=PATH=$V:/usr/local/cuda/bin:/usr/bin:/bin --setenv=CUDA_HOME=/usr/local/cuda \
  --setenv=VLLM_TURING_AR8=1 $ALLOC $DEPTHS $TAIL $ASYNCSPEC \
  $V/vllm serve $MODEL $CARFLAG \
  --served-model-name qwen3.8-27b qwen q \
  --host $VHOST --port $VPORT --tensor-parallel-size 2 --dtype float16 --enable-prompt-tokens-details \
  --max-model-len $LEN --gpu-memory-utilization $UTIL --max-num-seqs $MAXSEQ --max-num-batched-tokens $BATCH $OFFLOAD \
  --attention-backend FLASHINFER --kv-cache-dtype fp8 --mamba-ssm-cache-dtype $SSM_DTYPE --block-size $BLOCK \
  --long-prefill-token-threshold $PREFILL_CAP $SHAREF \
  --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$NSPEC}" \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 --chat-template $CHAT_TEMPLATE
for i in $(seq 1 750); do   # a cold first start at 262144 fails once and restarts warm: up to ~12 minutes
    sleep 2
    [ "$(curl -s -o /dev/null -w '%{http_code}' localhost:8080/health)" = 200 ] && break
done
# warm-up: the first sampled batches of several requests JIT-compile vLLM's top-p sampler kernels (Triton, ~3 s; the
# first 2-stream run after a restart measured ~110 instead of ~150 tok/s). A few short sampled requests, 1 / 2 / 4 / 6
# at a time, straight to vLLM (not through the monitor: not logged as client requests)
if [ "$(curl -s -o /dev/null -w '%{http_code}' localhost:8080/health)" = 200 ]; then
    VLLM_API_KEY="$(cat $MD/api-key.txt)" $V/python - "http://$VHOST:$VPORT" <<'EOF' || true
import json, os, sys, threading, urllib.request
url, key = sys.argv[1] + "/v1/chat/completions", os.environ["VLLM_API_KEY"]
def one(i):
    body = {"model": "qwen3.8-27b", "max_tokens": 24, "temperature": 0.6,
            "messages": [{"role": "user", "content": f"warm-up {i}: name three prime numbers"}]}
    req = urllib.request.Request(url, json.dumps(body).encode(),
                                 {"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=120).read()
    except Exception:
        pass
for n in (1, 2, 4, 6):
    ts = [threading.Thread(target=one, args=(i,)) for i in range(n)]
    [t.start() for t in ts]
    [t.join() for t in ts]
EOF
fi
echo "production (vLLM) health $(curl -s -o /dev/null -w '%{http_code}' localhost:8080/health) after $((i*2)) s," \
     "VRAM $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')" \
     "$([ "$MONITOR" = 1 ] && echo "| monitor: http://<host>:8080/monitor?key=<API key>")"
