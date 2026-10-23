"""INT8 input embedding table (vLLM 0.31): the target's embed_tokens (VocabParallelEmbedding, 248k x 5120, fp16: 1.18 GiB
per TP2 rank) is only ever read row-wise (token lookups), so it is stored as int8 with a per-row fp32 scale: 0.59 GiB per
rank freed for the KV cache (~33k more tokens at FP8 KV; PPL zh/code/en 6.5562/1.4875/2.4646 vs 6.5545/1.4880/2.4654).
The lookup is one kernel (csrc/embed8.cu: shard masking + gather + dequant + zero rows of other ranks' tokens, as the
fused vocab_parallel_embedding op), registered as the custom op vllm_turing::embed8 (opaque to torch.compile), then the
usual TP all-reduce. The MTP / DFlash drafters share the target's module. Not the LM head (lmhead.py).
VLLM_TURING_EMBED8=0 disables."""
import ctypes
import logging
import os
import types

import torch

from . import _build

logger = logging.getLogger("vllm.turing")
ENABLED = os.environ.get("VLLM_TURING_EMBED8", "1") != "0"
_lib = None


def lib():
    global _lib
    if _lib is None:
        _lib = _build.load("embed8")
        _lib.embed8_lookup.argtypes = ([ctypes.c_void_p, ctypes.c_int] + [ctypes.c_void_p] * 3 + [ctypes.c_int] * 2
                                       + [ctypes.c_longlong] * 5 + [ctypes.c_void_p])
        _lib.embed8_lookup.restype = ctypes.c_int
    return _lib


_frag = torch.library.Library("vllm_turing", "FRAGMENT")
_frag.define("embed8(Tensor ids, Tensor q, Tensor s, int org_start, int org_end, int added_start, int added_end, "
             "int added_offset) -> Tensor")


def _embed8_impl(ids, q, s, org_start, org_end, added_start, added_end, added_offset):
    flat = ids.reshape(-1)
    if flat.dtype not in (torch.int32, torch.int64):
        flat = flat.long()
    flat = flat.contiguous()
    D = q.shape[1]
    out = torch.empty(flat.numel(), D, dtype=torch.float16, device=q.device)
    rc = lib().embed8_lookup(flat.data_ptr(), int(flat.dtype == torch.int64), q.data_ptr(), s.data_ptr(),
                             out.data_ptr(), flat.numel(), D, org_start, org_end, added_start, added_end, added_offset,
                             torch.cuda.current_stream().cuda_stream)
    if rc != 0:
        raise RuntimeError(f"embed8_lookup failed: {rc}")
    return out.view(*ids.shape, D)


@torch.library.impl(_frag, "embed8", "CUDA")
def _embed8(ids, q, s, org_start, org_end, added_start, added_end, added_offset):
    return _embed8_impl(ids, q, s, org_start, org_end, added_start, added_end, added_offset)


def _embed8_call(*args):
    """The dispatcher round trip of a Python custom op costs tens of us on this host's CPUs: call the implementation
    directly in eager mode, through the registered op only while torch.compile traces."""
    if torch.compiler.is_compiling():
        return torch.ops.vllm_turing.embed8(*args)
    return _embed8_impl(*args)


@torch.library.register_fake("vllm_turing::embed8")
def _embed8_fake(ids, q, s, org_start, org_end, added_start, added_end, added_offset):
    return q.new_empty(*ids.shape, q.shape[1], dtype=torch.float16)


def _forward(self, input_):
    si = self.shard_indices
    added_offset = (si.added_vocab_start_index - (si.org_vocab_end_index - si.org_vocab_start_index)
                    - si.num_org_vocab_padding)
    out = _embed8_call(input_, self.turing_e8, self.turing_e8s, si.org_vocab_start_index,
                                       si.org_vocab_end_index, si.added_vocab_start_index, si.added_vocab_end_index,
                                       added_offset)
    if self.tp_size > 1:
        if self.parallel_group is not None:
            out = self.parallel_group.all_reduce(out)
        else:
            from vllm.distributed import tensor_model_parallel_all_reduce
            out = tensor_model_parallel_all_reduce(out)
    return out


def install():
    if not ENABLED:
        return
    from vllm.model_executor.layers import vocab_parallel_embedding as vpe
    M = vpe.UnquantizedEmbeddingMethod
    if getattr(M, "_vllm_turing_embed8", False):
        return
    orig_pw = M.process_weights_after_loading

    def process_weights_after_loading(self, layer):
        orig_pw(self, layer)
        w = getattr(layer, "weight", None)
        if (isinstance(layer, vpe.ParallelLMHead) or not isinstance(layer, vpe.VocabParallelEmbedding) or w is None
                or w.dim() != 2 or w.dtype != torch.float16 or w.numel() == 0 or w.shape[1] % 8):
            return
        q = torch.empty(w.shape, dtype=torch.int8, device=w.device)
        s = torch.empty(w.shape[0], dtype=torch.float32, device=w.device)
        for i in range(0, w.shape[0], 8192):
            wf = w.data[i:i + 8192].float()
            sc = wf.abs().amax(dim=1).clamp(min=1e-12) / 127.0
            q[i:i + 8192] = torch.round(wf / sc[:, None]).clamp(-127, 127).to(torch.int8)
            s[i:i + 8192] = sc
        layer.turing_e8, layer.turing_e8s = q, s
        layer.weight.data = torch.empty(0, dtype=w.dtype, device=w.device)   # free the fp16 table
        layer.forward = types.MethodType(_forward, layer)
        torch.cuda.empty_cache()
        logger.info("vllm_turing: int8 input embedding %s (%.2f -> %.2f GiB)", tuple(q.shape),
                    q.numel() * 2 / 2**30, (q.numel() + 4 * s.numel()) / 2**30)

    orig_emb = M.embedding

    def embedding(self, layer, input_):   # direct quant_method.embedding callers (shard-local ids, no masking needed)
        q = getattr(layer, "turing_e8", None)
        if q is None:
            return orig_emb(self, layer, input_)
        return _embed8_call(input_, q, layer.turing_e8s, 0, q.shape[0], 0, 0, 0)

    M.process_weights_after_loading, M.embedding, M._vllm_turing_embed8 = (process_weights_after_loading, embedding,
                                                                           True)

    # the lookup may be traced inside compiled graphs: own AOT compile cache key (a cached graph from before would
    # still read the freed fp16 table)
    from vllm.compilation import caching
    if not getattr(caching, "_vllm_turing_embed8", False):
        orig_f = caching.aot_compile_hash_factors

        def aot_compile_hash_factors(vllm_config):
            return [*orig_f(vllm_config), "vllm_turing_embed8 v2"]

        caching.aot_compile_hash_factors = aot_compile_hash_factors
        caching._vllm_turing_embed8 = True
