"""Prompt-logprob chunk size (vLLM 0.31, model runner v2). Prompt logprobs are computed over chunks of CHUNK_SIZE prompt
tokens; each chunk materializes its full-vocabulary logits (1024 x 248320: 486 MiB fp16, then fp32 copies for the
log-softmax), memory the startup profile does not reserve. With the KV cache sized at 0.95 of the GPU a request with
prompt_logprobs (perplexity measurements) ran the worker out of memory and took the engine down. 256-token chunks need
a quarter of that. VLLM_TURING_PROMPT_LP_CHUNK sets the size (0 keeps vLLM's)."""
import logging
import os

logger = logging.getLogger("vllm.turing")
CHUNK = int(os.environ.get("VLLM_TURING_PROMPT_LP_CHUNK", "256"))


def install():
    if CHUNK <= 0:
        return
    try:
        from vllm.v1.worker.gpu.sample import prompt_logprob
    except ImportError:
        return
    if getattr(prompt_logprob, "CHUNK_SIZE", None) is not None and prompt_logprob.CHUNK_SIZE > CHUNK:
        prompt_logprob.CHUNK_SIZE = CHUNK
