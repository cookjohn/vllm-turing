# vllm-turing

Turing (sm75) speedups for vLLM, plus the serving setup built around them: Qwen3.8-27B (hybrid GDN + attention,
262k context, MTP head, vision) served as an INT8 W8A8 checkpoint with tensor parallelism over two RTX 2080 Ti 22 GB
cards (NVLink bridge), for interactive use and for agent clients over the Anthropic and OpenAI APIs.

vLLM's fast paths target Ampere and newer. On Turing, the stock vLLM falls back to slow or missing kernels: no FP8
tensor cores, no FlashAttention 2, f16 MMA with f32 accumulation at half rate on GeForce cards, and FlashInfer without
its trtllm kernels. This repository replaces those paths with hand-written CUDA kernels and fixes the parts of the
serving stack that break or slow down at long context on this hardware.

## Results

Measured on 2× RTX 2080 Ti 22 GB with NVLink (TP2), vLLM 0.31.0, MTP with 4 draft tokens, FP8 KV cache, context
262,144 tokens:

| | |
|---|---|
| Decode, one stream, short context | ~110 tok/s |
| Decode, aggregate over 1 / 2 / 4 / 6 streams | 114 / 150 / 272 / 374 tok/s |
| Decode, one stream at 120k context | 70–80 tok/s (59 before the long-context verify kernel) |
| Prefill, 32K / 128K prompt | ~3,550 / ~2,550 tok/s |
| KV cache pool | ~280k tokens on the GPUs + 24 GB host RAM offload |
| Perplexity zh / code / en, W8A8 SmoothQuant | 6.55 / 1.488 / 2.461 |

[docs/IMPROVEMENTS.md](docs/IMPROVEMENTS.md) lists every change with its measured effect.

## Contents

| Path | |
|---|---|
| `vllm_turing/` | The vLLM plugin (loaded automatically through the `vllm.general_plugins` entry point): Python hooks and CUDA kernels (`csrc/`, built on first use with nvcc and loaded with ctypes) |
| `patches/flashinfer_spec_decode.py` | Patch for vLLM's FlashInfer backend: sm75 support and FULL CUDA graphs for speculative verify |
| `monitor/` | Monitoring reverse proxy in front of vLLM (dashboard, request logs, context editing for long agent sessions, Anthropic-API fixes) and log analysis tools |
| `scripts/` | Production start script, model conversion (W8A8 / SmoothQuant calibration), benchmarks, A/B test harnesses |
| `templates/` | The chat template used in production (tools, reasoning effort, inline system messages) |
| `tests/` | Kernel tests against reference implementations, micro-benchmarks, trace analysis |

## Requirements

- 2× NVIDIA Turing GPUs (developed on RTX 2080 Ti 22 GB; NVLink optional, the PCIe paths are kept), driver 580
- vLLM 0.31.0 with torch 2.13.0+cu130 and FlashInfer 0.7.0.post1 (`scripts/vllm-install.sh` sets up such an
  environment with uv)
- nvcc: the CUDA 13 compiler from the `nvidia-cuda-nvcc` wheel in the same environment (matches torch), or
  `/usr/local/cuda` for the CUTLASS-based kernel

Several kernels are specialized for this model's shapes (head dim 256, 6 query heads per KV head per rank at TP2,
GDN key dim 128). Calls outside the supported cases go to the original vLLM / FlashInfer code paths.

## Setup

```bash
scripts/vllm-install.sh                              # vLLM environment in ~/vllm-env
~/vllm-env/bin/pip install -e .                      # the plugin (vllm.general_plugins entry point)
~/vllm-env/bin/python patches/flashinfer_spec_decode.py   # patch the installed vLLM's FlashInfer backend (idempotent)

# model: BF16 original -> SmoothQuant INT8 W8A8 (calibrate activations on a W8A16 copy, then convert)
~/vllm-env/bin/python scripts/w8a8_convert.py ~/models/Qwen3.8-27B ~/models/Qwen3.8-27B-W8A16-g128 cuda:0 --w8a16
scripts/calib_run.sh                                 # activation statistics -> ~/models/calib-w8a16
~/vllm-env/bin/python scripts/w8a8_convert.py ~/models/Qwen3.8-27B ~/models/Qwen3.8-27B-SQ06 cuda:0 \
    --smooth ~/models/calib-w8a16 --alpha 0.6

cp templates/qwen3.8-chat-template-inline-system.jinja ~/models/
echo "<your API key>" > ~/models/api-key.txt
PW=<sudo password> scripts/prod_start.sh             # vLLM on 127.0.0.1:8090, monitoring proxy on :8080
```

`prod_start.sh` runs vLLM and the proxy as transient systemd units and documents every option it sets, with the
measurements behind each default. Tests that need the GPUs while production runs go through `scripts/probe_safe.sh`
(stops production, hard time limit, always restarts it).

Every plugin component can be switched off on its own (`VLLM_TURING_<NAME>=0`, see each module's docstring);
`VLLM_TURING_DISABLE=1` turns the plugin off entirely.

## Status

Built for one machine and one model, and kept in step with vLLM 0.31.0. Other vLLM versions will need the hooks
and the patch adjusted. The kernels are tested against reference implementations (`tests/`); the end-to-end
numbers come from the scripts in `scripts/`.
