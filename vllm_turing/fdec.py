"""Speculative verify / draft decode attention over long FP8 contexts on Turing (csrc/fdec.cu). At 120k context the
decode step spent ~10 ms of ~45 in FlashInfer's FA2 kernel (20 calls of ~535 us: the 5-token verify of 16 layers, the
drafter's 5-token first step and its 1-token steps) at ~230 GB/s; fd_run streams the FP8 KV straight into MMA
fragments with no wasted Q rows. Hooks the wrappers vLLM's FlashInfer backend plans for uniform verify batches
(patches/flashinfer_spec_decode.py _turing_plan_spec_prefill: k tokens per request, causal or not) and for decodes
(_get_decode_wrapper): run() of a qualifying call (f16 queries, D 256, 6 Q heads per KV head, FP8 e4m3 KV with
16-token pages, no window / soft cap / sinks / lse output) calls fd_run with the wrapper's device page metadata, so
CUDA graphs capture it like the FlashInfer kernel (lengths and pages change between replays, the grid does not).
Anything else goes to FlashInfer. VLLM_TURING_FDEC=0 disables."""
import ctypes
import logging
import math
import os

import torch

from . import _build

logger = logging.getLogger("vllm.turing")
ENABLED = os.environ.get("VLLM_TURING_FDEC", "1") != "0"
MAXB = int(os.environ.get("VLLM_TURING_FDEC_MAXB", "8"))   # requests the shared workspace is first sized for
D, G, PAGE = 256, 6, 16
_lib = None
_ws = {}          # device -> (ws_o int32, ws_lse f32); replaced only outside capture, old ones kept alive
_keep = []
_nsm = {}
_stats = [0, 0]
bypass = False    # tests: FlashInfer for everything


def lib():
    global _lib
    if _lib is None:
        _lib = _build.load("fdec")
        _lib.fd_run.argtypes = ([ctypes.c_void_p, ctypes.c_longlong, ctypes.c_longlong, ctypes.c_void_p]
                                + [ctypes.c_longlong] * 4 + [ctypes.c_void_p] * 3 + [ctypes.c_int] * 5
                                + [ctypes.c_float] * 2 + [ctypes.c_void_p] * 3 + [ctypes.c_longlong] * 2
                                + [ctypes.c_void_p])
        _lib.fd_run.restype = ctypes.c_int
    return _lib


def splits(dev, h, mt):
    """KV splits per (request, KV head, row tile): one request fills at most one wave of two blocks per SM (rounded
    down: 138 blocks of q_len 8 on 136 slots ran a second wave, 414 vs 280 us at 120k)."""
    n = _nsm.get(dev)
    if n is None:
        n = _nsm[dev] = torch.cuda.get_device_properties(dev).multi_processor_count
    return max(1, 2 * n // (h * mt))


def _workspace(dev, partials):
    """Split partials (f16 O, f32 lse): the head of fiws's shared FlashInfer workspace when there is one (allocated in
    the memory profile; FlashInfer's temporaries there live within one run() call as ours do, and all calls run one
    after another on the compute stream), else an own buffer (not inside a capture)."""
    key = dev.index if dev.index is not None else torch.cuda.current_device()
    w = _ws.get(key)
    if w is not None and w[1].numel() >= partials * 16:
        return w
    nb = partials * (2048 * 4 + 16 * 4)
    from . import fiws
    sh = fiws._shared.get(key)
    if sh is not None and sh.numel() >= nb:
        raw = sh[:nb]
    elif torch.cuda.is_current_stream_capturing():
        return None
    else:
        raw = torch.empty(nb, dtype=torch.uint8, device=dev)
    w = _ws[key] = (raw[: partials * 8192].view(torch.int32), raw[partials * 8192:].view(torch.float32))
    _keep.append(raw)   # graphs captured with an older workspace keep using it
    return w


def _as_float(x):
    if x is None:
        return 1.0
    if isinstance(x, torch.Tensor):
        return float(x.item())
    return float(x)


def attention(q, k_cache, v_cache, layout, indptr, indices, last, q_len, causal, scale, k_scale, v_scale, out):
    """q / out [B * q_len, H * 6, 256] f16; k_cache / v_cache: FP8 views [pages, 16, H, 256] (NHD) or [pages, H, 16,
    256] (HND) of one cache; indptr / indices / last: int32 device paged metadata. False if the workspace cannot grow
    (inside a capture)."""
    B = q.shape[0] // q_len
    H = k_cache.shape[2] if layout == "NHD" else k_cache.shape[1]
    if layout == "NHD":
        s_page, s_tok, s_head = k_cache.stride(0), k_cache.stride(1), k_cache.stride(2)
    else:
        s_page, s_head, s_tok = k_cache.stride(0), k_cache.stride(1), k_cache.stride(2)
    dev = q.device
    mt = -(-q_len * G // 16)
    S = splits(dev, H, mt)
    w = _workspace(dev, max(B, MAXB) * H * mt * S)
    if w is None:
        return False
    v_off = v_cache.data_ptr() - k_cache.data_ptr()
    rc = lib().fd_run(q.data_ptr(), q.stride(0), q.stride(1), k_cache.data_ptr(), s_page, s_tok, s_head, v_off,
                      indptr.data_ptr(), indices.data_ptr(), last.data_ptr(), B, q_len, int(bool(causal)), H, S,
                      scale * k_scale * 256.0 * math.log2(math.e), v_scale * 256.0, w[0].data_ptr(), w[1].data_ptr(),
                      out.data_ptr(), out.stride(0), out.stride(1), torch.cuda.current_stream(dev).cuda_stream)
    if rc != 0:
        raise RuntimeError(f"fd_run failed: {rc}")
    return True


def _qualifies(self, q, kv, args, kw, q_len):
    if bypass or args or kw.get("return_lse") or kw.get("lse") is not None or kw.get("sinks") is not None:
        return None
    if kw.get("kv_cache_sf") is not None or not isinstance(kv, (tuple, list)) or len(kv) != 2:
        return None
    k_cache, v_cache = kv
    out = kw.get("out")
    layout = getattr(self, "_kv_layout", None)
    if (out is None or layout not in ("NHD", "HND") or q.dtype != torch.float16 or q.dim() != 3
            or q.shape[2] != D or q.stride(2) != 1 or q.shape[0] % q_len != 0 or q_len * G > 96
            or k_cache.dtype != torch.float8_e4m3fn or v_cache.dtype != torch.float8_e4m3fn or k_cache.dim() != 4
            or k_cache.shape[3] != D or k_cache.stride(3) != 1 or v_cache.stride() != k_cache.stride()
            or out.dtype != torch.float16 or out.shape != q.shape or out.stride(2) != 1):
        return None
    H = k_cache.shape[2] if layout == "NHD" else k_cache.shape[1]
    page = k_cache.shape[1] if layout == "NHD" else k_cache.shape[2]
    if page != PAGE or q.shape[1] != G * H or k_cache.data_ptr() % 16 or v_cache.data_ptr() % 16:
        return None
    if any(st % 16 for st in k_cache.stride()[:3]) or q.stride(0) % 8 or q.stride(1) % 8 or q.data_ptr() % 16:
        return None
    if (getattr(self, "_window_left", -1) not in (-1, None) or getattr(self, "_logits_soft_cap", 0.0)
            or getattr(self, "_pos_encoding_mode", "NONE") != "NONE"):
        return None
    if kw.get("window_left") not in (None, -1) or kw.get("logits_soft_cap"):
        return None
    bufs = (getattr(self, "_paged_kv_indptr_buf", None), getattr(self, "_paged_kv_indices_buf", None),
            getattr(self, "_paged_kv_last_page_len_buf", None))
    B = q.shape[0] // q_len
    if (any(x is None or not x.is_cuda or x.dtype != torch.int32 for x in bufs) or bufs[0].numel() < B + 1
            or bufs[2].numel() < B):
        return None
    return k_cache, v_cache, layout, out, bufs


def install():
    if not ENABLED:
        return
    from vllm.platforms import current_platform
    cap = current_platform.get_device_capability()   # NVML: no CUDA context in processes that do not use the GPU
    if cap is None or cap.to_int() != 75:
        return
    try:
        import flashinfer
        from vllm.v1.attention.backends import flashinfer as fi
    except Exception:   # noqa: BLE001
        return
    Bld = fi.FlashInferMetadataBuilder
    if getattr(Bld, "_vllm_turing_fdec", False) or not hasattr(Bld, "_turing_plan_spec_prefill"):
        return
    orig_tpsp, orig_gdw = Bld._turing_plan_spec_prefill, Bld._get_decode_wrapper

    def _turing_plan_spec_prefill(self, num_reqs, k, causal, paged_kv_indices):
        w = orig_tpsp(self, num_reqs, k, causal, paged_kv_indices)
        w._turing_fdec = (k, bool(causal))
        return w

    def _get_decode_wrapper(self, batch_size, use_cudagraph=False):
        w = orig_gdw(self, batch_size, use_cudagraph)
        w._turing_fdec = (1, True)
        return w

    Bld._turing_plan_spec_prefill, Bld._get_decode_wrapper = _turing_plan_spec_prefill, _get_decode_wrapper

    def hook(W):
        orig_run = W.run

        def run(self, q, paged_kv_cache, *args, **kw):
            st = self.__dict__.get("_turing_fdec")
            if st is not None:
                ok = _qualifies(self, q, paged_kv_cache, args, kw, st[0])
                if ok is not None:
                    k_cache, v_cache, layout, out, (indptr, indices, last) = ok
                    sm = getattr(self, "_sm_scale", None)
                    scale = (D ** -0.5 if sm is None else sm) * _as_float(kw.get("q_scale"))
                    if attention(q, k_cache, v_cache, layout, indptr, indices, last, st[0], st[1], scale,
                                 _as_float(kw.get("k_scale")), _as_float(kw.get("v_scale")), out):
                        _stats[0] += 1
                        if _stats[0] % 20000 == 1:
                            logger.info("vllm_turing fdec: verify / decode attention calls %d, FlashInfer %d",
                                        *_stats)
                        return out
                _stats[1] += 1
            return orig_run(self, q, paged_kv_cache, *args, **kw)

        W.run = run

    hook(flashinfer.BatchPrefillWithPagedKVCacheWrapper)
    hook(flashinfer.BatchDecodeWithPagedKVCacheWrapper)
    Bld._vllm_turing_fdec = True
    logger.info("vllm_turing: FP8 verify / decode attention kernel (fdec)")
