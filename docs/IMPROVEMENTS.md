# Improvements

What was changed to serve Qwen3.8-27B on two RTX 2080 Ti 22 GB (Turing, sm75, TP2 over NVLink) with vLLM 0.31.0,
and what each change measured. Numbers are from this machine; "before" is the configuration right before the change.
The code for each item names its switch (`VLLM_TURING_<NAME>=0` turns a plugin component off).

Model facts that shape most of the work: 64 decoder layers, of which 16 are full attention (2 KV heads × 256 dims
per rank at TP2, 6 query heads per KV head) and 48 are Gated DeltaNet (GDN) linear-attention layers with a recurrent
state; one MTP layer for speculative decoding; a 248k-token vocabulary; a vision tower.

## Model and quantization

| Change | Effect |
|---|---|
| SmoothQuant INT8 W8A8 checkpoint (`scripts/w8a8_convert.py --smooth`, alpha 0.6; activation statistics from `vllm_turing/calib.py` on a W8A16 copy, `scripts/calib_run.sh`). Turing has int8 tensor cores but no FP8. | Perplexity zh / code / en 6.55 / 1.488 / 2.461, better than a llama.cpp Q4_K_M GGUF of the same model |
| INT8 LM head with per-row scales (`lmhead.py`): W8A16 GEMV for up to 4 rows, tensor-core W8A16 GEMM (`csrc/w8a16_mma.cu`) for 5–64 rows | Head memory and read time halved (1.27 GB per GPU before) |
| INT8 input embedding table (`embed8.py`, one lookup kernel) | 0.59 GiB per GPU freed, ~33k more KV tokens; perplexity unchanged |
| INT8 weight-only fc for DFlash drafters (`dflashfc.py`) | 250 → 125 MB read per draft step |

## Matrix multiplication

| Change | Effect |
|---|---|
| Small-M W8A8 GEMV (`csrc/w8a8_gemv.cu`) for up to 12 rows, behind vLLM's `cutlass_scaled_mm` | Decode and single-request verify without CUTLASS's 128-row tile padding |
| INT8 tensor-core kernel for 13–32 rows (MTP verify of 3–6 concurrent requests), split-K table tuned per shape, activation prefetch | 1.3–1.4× CUTLASS at these sizes |
| Prefill GEMM at the fastest CUTLASS tile for the 2080 Ti (`csrc/w8a8_big.cu`, 256 × 128 × 64, raster-swizzled) | ~10% over vLLM's 128 × 128 × 64; 32K prefill 2,573 → 2,779 tok/s |
| Per-token activation quantization fused into the producing compiled kernel (`quantfuse.py`) and SiLU·mul + quantization in one kernel (`csrc/silu_quant.cu`) | Removes a standalone kernel before every int8 GEMM (2.5% of a 32K prefill, 1.8% of decode) |

## Attention

| Change | Effect |
|---|---|
| FlashInfer backend patch (`patches/flashinfer_spec_decode.py`): sm75 support, FULL CUDA graphs for the MTP verify, the verify as one causal prefill per request on a CUDA-graph wrapper | Before, the verify ran as k flattened decodes that re-read the whole KV cache per verify token (MTP4 decode fell from 122 to 98.5 tok/s between 4K and 32K context) |
| INT8-QK prefill attention for FP8 paged KV (`fiattn8.py`, `csrc/fiattn8.cu`, warp-specialized, int8 tensor cores for QK) | ~2.8× FlashInfer's FA2 kernel on sm75 (which runs QK as f16 MMA with f32 accumulation at half rate, ~28 TFLOPS) |
| Long-context verify / draft attention (`fdec.py`, `csrc/fdec.cu`): FP8 KV streamed straight into MMA fragments through a 2-instruction FP8 → f16 bit conversion, 16-row tiles instead of FlashInfer's 64 (30 rows used by a 5-token verify), split-KV with a merge kernel, CUDA-graph safe | At 120k context: 5-token verify 546 → 233 µs (528 GB/s, 86% of peak bandwidth), 1-token draft 528 → 230 µs; end to end, 120k decode 59.4 → 70.4 tok/s |
| One FlashInfer workspace per device, allocated inside the memory profile (`fiws.py`) | The per-builder workspaces were allocated after the profile; the process ran ~0.8 GiB over budget |

## Gated DeltaNet (linear attention)

| Change | Effect |
|---|---|
| Tensor-core chunked prefill for GDN (`gdn.py`, `csrc/gdn_chunk.cu`) in place of FLA's Triton kernels: state recurrence, output, and the WY preparation fused into one kernel that keeps A and its inverse in shared memory | GDN prefill on tensor cores; three Triton kernels → one |
| fp16 GDN recurrent state cache (`--mamba-ssm-cache-dtype float16`; kernels still compute in fp32) | The MTP verify writes the whole state after every draft token: 6 streams 345 → 368 tok/s; perplexity and needle tests unchanged |

## Communication (tensor parallelism)

| Change | Effect |
|---|---|
| vLLM's custom all-reduce working with expandable allocator segments (`customar.py`: the staging-buffer copy inside CUDA graphs) | Custom all-reduce over NVLink with the allocator that gives ~1 GiB more KV cache |
| Copy-engine all-reduce over P2P (`cear.py`, `csrc/ce_ar.cu`) | 17.2 MB in 0.50 ms vs NCCL 0.61 ms, without occupying SMs |
| Prefill layer-tail overlap (`tailoverlap.py`): the per-token tail of each layer (out_proj, all-reduce, norm, MLP, all-reduce) on token chunks, all-reduces on a side stream | 32K prefill 3,190 → 3,385 tok/s, 128K 2,350 → 2,454 |
| Low-latency P2P all-reduce for small messages, and for GPUs without peer access: host-memory all-reduce (`hostar.py`), int8-compressed prefill all-reduce, copy-engine all-gather through NUMA-local pinned memory (`dmaag.py`, 11.3 GB/s per direction vs NCCL's 5.7) | Decode all-reduces without NCCL's latency; the PCIe paths stay for machines without NVLink |

## Speculative decoding

| Change | Effect |
|---|---|
| Sync-free MTP draft steps (`nosync.py`): exact draft lengths computed on the CPU from one early non-blocking copy | Removed a GPU drain before every draft step (~0.6 ms each, about half of a draft step) |
| Asynchronous decode steps (`VLLM_TURING_ASYNC_SPEC=1`): FlashInfer planned from the CPU's length upper bound, exact lengths fixed on the GPU, staging rings for the plans | GPU idle in decode 18% → 4%, decode +8–13% |
| Pruned draft vocabulary with a balanced 4-bit draft head (`draftvocab.py`, `csrc/dhead_q4.cu`) | The draft head reads 65k of 248k rows at 4 bits; the target still verifies with its full head (output exact) |
| Adaptive MTP depth per step (`mtpdepth.py`), MTP + DFlash2 combined drafting (`combo.py`), KV grouping for sliding-window drafters (`kvgroups.py`) | Optional modes: per-text choice of draft depth / drafter |

## Memory and KV cache

| Change | Effect |
|---|---|
| KV block size 880 instead of 1680 with the fp16 GDN state: vLLM pads every GDN state block to the attention block size, and 1680 had been chosen for the fp32 state (100% padding with fp16) | KV pool 281,617 → 295,326 tokens; 6 concurrent 30k requests peak at 86% instead of 96% KV, all 6 run instead of 5 + 1 queued; a continued conversation re-prefills at most 880 tokens |
| KV offloading to 24 GB of pinned host RAM (`--kv-offloading-size 24 --enable-cumem-allocator`) | Long agent sessions are no longer evicted by concurrent sub-agents: a 120k session after 9 × 30k other requests: 0 cached / 47.9 s to the first token before, 118,800 cached / 1.65 s after (identical output) |
| Low-memory vision encoder forward (`vitlowmem.py`) and 256-token prompt-logprob chunks (`promptlp.py`) | Peaks the startup memory profile did not reserve no longer take the engine down |

## Serving agent workloads (`monitor/`)

The monitoring proxy sits in front of vLLM on the public port: live dashboard, per-request logs, engine progress
(`progress.py`), and fixes for long agent sessions sent over the Anthropic API.

| Change | Effect |
|---|---|
| Context editing (`ctxedit.py`): past a token threshold, older tool traffic is cleared from the forwarded copy of the conversation (user messages and the recent part are kept), so sessions continue past the 262k window | Long sessions keep running ("logical 1M context") |
| `max_tokens` fit: a request whose prompt + `max_tokens` exceeds the window is resent with the room that is left | No context-length errors for clients that always ask for 32k output tokens |
| No extended thinking for Anthropic requests without `thinking` (the API's meaning; the chat template thinks by default and maps effort "high" to its maximum) | Helper calls (page summaries and the like): ~5,000 → 121 output tokens, ~170 → 6 s each; a queue of 10–15 waiting requests cleared |
| Cross-session prefix reuse: the proxy drops a per-session client header that sat right after the tool definitions; vLLM's shared-prefix GDN checkpoint (`--enable-mamba-shared-prefix-checkpoint --prefix-match-unit 80`) | New sub-agent sessions reuse the shared tools + system prompt: 3 sessions of 37k tokens: 0 / 0 / 36,000 cached, first token 10.9 → 0.9 s from the third session on |
| Operations reports (`ops_report.py`, `harness_stats.py`) | Concurrency, waits, cache hits and generation speed per time window from the request logs |

## Tooling

| Script | |
|---|---|
| `scripts/prod_start.sh` | Production start: every option with the measurement behind its default |
| `scripts/probe_safe.sh` | Runs a GPU test with production stopped, under a hard time limit, and always restarts production with a clean environment |
| `scripts/vmon_upgrade.sh` | Zero-downtime upgrade of the monitoring proxy (second instance, iptables redirect, drain) |
| `scripts/vllm_util_probe.sh` | Profiled prefill / decode / long-context decode / concurrent KV-load phases, compute utilization from the traces (`tests/trace_util.py`) |
| `scripts/block_ab.sh`, `offload_test.sh`, `prefix_share_test.sh`, `vllm_feature_trial.sh` | A/B harnesses for the KV block size, KV offloading, cross-session prefix reuse and plugin features |
| `scripts/vllm_acceptance.sh`, `vllm_bench.py`, `vllm_conc_bench.py`, `vllm_ppl.sh` | Acceptance checks, decode / concurrency / long-context benchmarks, perplexity |
