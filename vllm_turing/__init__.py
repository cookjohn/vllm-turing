"""Turing (sm75) speedups for vLLM, loaded in every vLLM process (API server, engine core, workers) through the
vllm.general_plugins entry point. VLLM_TURING_DISABLE=1 turns everything off.
  - w8a8: small-M INT8 W8A8 GEMM (decode / speculative verify) behind cutlass_scaled_mm
  - gdn: tensor-core GDN chunked prefill (state recurrence + output) in place of FLA's Triton kernels
  - hostar: two-rank small-message all-reduce through pinned host memory (GPUs without peer access)
  - embed8: int8 input embedding table (lookups only: 1.18 -> 0.59 GiB per rank)
  - lmhead: int8 LM head (per-row scales, fp16 activations, W8A16 GEMV): half the head memory and read time
  - draftvocab: greedy speculative drafts over a pruned vocabulary (ranked token list), balanced 4-bit draft head
  - kvgroups: KV cache grouping that does not let a sliding-window drafter (DFlash2) shrink the group size
  - overlap: prefill MLP all-reduce / GEMM overlap on token chunks (VLLM_TURING_OVERLAP=1; slower on 2x 2080 Ti)
  - tailoverlap: the whole per-token tail of a decoder layer (out_proj, both all-reduces, norm, MLP) on token chunks
    with the all-reduces on a side stream, fused norm + int8 quant (VLLM_TURING_TAIL=1)
  - quantfuse: W8A8 activation quantization traced as torch ops under torch.compile, fused into the producer kernel
  - promptlp: 256-token prompt-logprob chunks (vLLM's 1024 materialized ~1.5 GiB of logits outside the memory profile)
  - customar: vLLM's custom all-reduce (NVLink P2P) through its staging buffer in CUDA graphs, so it works with
    expandable allocator segments
  - dflashfc: int8 weight-only DFlash fc (25600 -> 5120, replicated: 250 -> 125 MB read per draft step)
  - nosync: MTP draft steps without the per-step device sync of FlashInfer's metadata build (exact lengths)
  - mtpdepth: adaptive MTP draft depth per step (VLLM_TURING_MTP_DEPTHS=3,5,7, num_speculative_tokens = max)
  - combo: MTP + DFlash2 combined speculator, adaptive per-step drafter choice (VLLM_TURING_COMBO=1)
  - debug_dflash: DFlash drafter activation / draft diagnostics (VLLM_TURING_DRAFT_DEBUG=1, with --enforce-eager)
  - memdbg: startup memory-profile breakdown, encoder vs decoder peak (VLLM_TURING_MEMDBG=1)
  - vitlowmem: vision encoder forward with a bounded working set (the profiled max-size image's activation peak is
    reserved out of the KV cache)
  - fiattn8: int8-QK causal prefill attention for FP8 paged KV (port of the llama.cpp fork's kernel, ~2.8x FlashInfer)
  - fdec: speculative verify / draft decode attention over FP8 KV (~2.3x FlashInfer's FA2 kernel at 120k context)
  - fiws: one FlashInfer workspace per device for all attention builders, allocated inside the memory profile
  - progress: live per-request prefill progress for the monitoring proxy (/dev/shm/vllm_progress.json)
  - patches/flashinfer_spec_decode.py (applied once to the installed vLLM): FlashInfer on sm75 with FULL CUDA graphs
    for speculative verify and non-causal (DFlash) draft blocks"""
import logging
import os

logger = logging.getLogger("vllm.turing")   # under "vllm": uses vLLM's log handler
_registered = False


def register():
    global _registered
    if _registered or os.environ.get("VLLM_TURING_DISABLE") == "1":
        return
    _registered = True
    from . import (calib, combo, debug_dflash, dflashfc, draftvocab, embed8, fdec, fiattn8, fiws, gdn, hostar,
                   kvgroups, customar, lmhead, memdbg, mtpdepth, nosync, overlap, progress, promptlp, quantfuse,
                   tailoverlap, vitlowmem, w8a8)

    debug_dflash.install()
    memdbg.install()
    vitlowmem.install()
    fiattn8.install()
    fiws.install()      # after fiattn8: sizes the shared workspace by whether fiattn8 is active
    progress.install()
    combo.install()
    mtpdepth.install()
    nosync.install()
    fdec.install()      # after fiattn8 (wraps its prefill run) and nosync
    dflashfc.install()
    kvgroups.install()
    overlap.install()
    tailoverlap.install()
    quantfuse.install()
    promptlp.install()
    customar.install()
    calib.install()
    w8a8.install()
    gdn.install()
    hostar.install()
    lmhead.install()
    embed8.install()   # after lmhead: wraps its process_weights_after_loading
    draftvocab.install()
    logger.info("vllm_turing: small-M W8A8 GEMV active for M <= %d, GDN prefill kernels %s", w8a8.MAX_M,
                "active" if gdn._orig else "off")
