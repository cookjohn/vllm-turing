"""INT8-QK causal prefill attention for FP8 paged KV on Turing (csrc/fiattn8.cu, port of our llama.cpp fork's
fattn-int8): FlashInfer's FA2 prefill on sm75 runs the QK product as f16 MMA with f32 accumulation (half rate on
GeForce Turing) and dequantizes FP8 in the inner loop, ~28 TFLOPS; at long context attention is most of the prefill
time. Hooks flashinfer.BatchPrefillWithPagedKVCacheWrapper: plan() records whether a plan qualifies (causal, D 256,
6 Q heads per KV head, f16 queries, FP8 e4m3 KV, no window / soft cap / custom mask, not a CUDA-graph wrapper, at least
VLLM_TURING_FA8_MIN_TOKENS query tokens) and its per-request layout; run() of a qualifying plan calls the int8 kernel
per request (query sub-batches of VLLM_TURING_FA8_QB tokens bound the f32 merge workspace), anything else goes to
FlashInfer. VLLM_TURING_FA8=0 disables."""
import ctypes
import logging
import os

import torch

from . import _build

logger = logging.getLogger("vllm.turing")
ENABLED = os.environ.get("VLLM_TURING_FA8", "1") != "0"
MIN_TOKENS = int(os.environ.get("VLLM_TURING_FA8_MIN_TOKENS", "256"))
QB = max(16, int(os.environ.get("VLLM_TURING_FA8_QB", "4096")) // 16 * 16)
CHUNK = 16384           # KV positions converted per kernel launch (K8 + V16: 24 MiB for 2 KV heads)
D, G = 256, 6
_lib = None
_ws = {}                # device -> dict of workspace tensors
_stats = [0, 0]
_installed = False
bypass = False          # tests: plan / run with FlashInfer


def active():
    """Large causal prefill chunks run here, not in FlashInfer (fiws sizes the FlashInfer workspace for the rest)."""
    return _installed


def lib():
    global _lib
    if _lib is None:
        _lib = _build.load("fiattn8")
        _lib.fa8_prefill.argtypes = ([ctypes.c_void_p, ctypes.c_longlong, ctypes.c_longlong, ctypes.c_void_p,
                                      ctypes.c_void_p] + [ctypes.c_longlong] * 3 + [ctypes.c_int, ctypes.c_void_p,
                                      ctypes.c_longlong] + [ctypes.c_int] * 3 + [ctypes.c_float] * 3
                                     + [ctypes.c_void_p, ctypes.c_longlong] + [ctypes.c_void_p] * 5
                                     + [ctypes.c_int, ctypes.c_void_p])
        _lib.fa8_prefill.restype = ctypes.c_int
    return _lib


def _workspace(dev, nkv, nh, qb):
    w = _ws.get(dev)
    if w is None or w["nkv"] < nkv or w["rows"] < qb * nh:
        w = {"nkv": nkv, "rows": qb * nh,
             "K8": torch.empty(nkv * CHUNK * D, dtype=torch.int8, device=dev),
             "KS": torch.empty(nkv * CHUNK, dtype=torch.float32, device=dev),
             "V16": torch.empty(nkv * CHUNK * D, dtype=torch.float16, device=dev),
             "A": torch.empty(qb * nh * D, dtype=torch.float32, device=dev),
             "MS": torch.empty(qb * nh * 2, dtype=torch.float32, device=dev)}
        _ws[dev] = w
    return w


def _as_float(x):
    if x is None:
        return 1.0
    if isinstance(x, torch.Tensor):
        return float(x.item())
    return float(x)


def run_prefill(q, k_cache, v_cache, layout, page_size, reqs, indices, scale, k_scale, v_scale, out):
    """reqs: [(q_start, ntok, kv_len, page_start)]; indices: int32 page ids (device); out [T, NH, D] f16."""
    nh, nkv = q.shape[1], k_cache.shape[2] if layout == "NHD" else k_cache.shape[1]
    if layout == "NHD":    # [pages, page_size, heads, D]
        s_page, s_tok, s_head = k_cache.stride(0), k_cache.stride(1), k_cache.stride(2)
    else:                  # HND: [pages, heads, page_size, D]
        s_page, s_head, s_tok = k_cache.stride(0), k_cache.stride(1), k_cache.stride(2)
    assert (v_cache.stride(0), v_cache.stride(1), v_cache.stride(2)) == (k_cache.stride(0), k_cache.stride(1),
                                                                         k_cache.stride(2))
    dev = q.device
    w = _workspace(dev, nkv, nh, QB)
    stream = torch.cuda.current_stream(dev).cuda_stream
    esz_q, esz_o = q.element_size(), out.element_size()
    for q_start, ntok, kv_len, page_start in reqs:
        pages = indices[page_start:]
        for s0 in range(0, ntok, QB):
            n = min(QB, ntok - s0)
            rc = lib().fa8_prefill(q.data_ptr() + (q_start + s0) * q.stride(0) * esz_q, q.stride(0), q.stride(1),
                                   k_cache.data_ptr(), v_cache.data_ptr(), s_page, s_tok, s_head, page_size,
                                   pages.data_ptr(), kv_len - ntok + s0, n, nh, nkv, scale, k_scale, v_scale,
                                   out.data_ptr() + (q_start + s0) * out.stride(0) * esz_o, out.stride(0),
                                   w["K8"].data_ptr(), w["KS"].data_ptr(), w["V16"].data_ptr(), w["A"].data_ptr(),
                                   w["MS"].data_ptr(), CHUNK, stream)
            if rc != 0:
                raise RuntimeError(f"fa8_prefill failed: {rc}")
    return out


def install():
    if not ENABLED:
        return
    from vllm.platforms import current_platform
    cap = current_platform.get_device_capability()   # NVML: no CUDA context in processes that do not use the GPU
    if cap is None or cap.to_int() != 75:
        return
    try:
        import flashinfer
    except Exception:   # noqa: BLE001
        return
    W = flashinfer.BatchPrefillWithPagedKVCacheWrapper
    if getattr(W, "_vllm_turing_fa8", False):
        return
    orig_plan, orig_run = W.plan, W.run

    def plan(self, qo_indptr, paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len, num_qo_heads, num_kv_heads,
             head_dim_qk, page_size, *args, **kw):
        self._turing_fa8 = None
        try:
            ok = (not bypass and not args and not getattr(self, "_use_cuda_graph", False) and kw.get("causal", False)
                  and kw.get("custom_mask") is None and kw.get("packed_custom_mask") is None
                  and kw.get("window_left", -1) < 0 and not kw.get("logits_soft_cap")
                  and kw.get("pos_encoding_mode", "NONE") == "NONE" and not kw.get("use_fp16_qk_reduction", False)
                  and head_dim_qk == D and kw.get("head_dim_vo") in (None, D) and num_qo_heads == G * num_kv_heads
                  and kw.get("q_data_type", "float16") in ("float16", torch.float16)
                  and kw.get("kv_data_type") in ("float8_e4m3fn", torch.float8_e4m3fn)
                  and getattr(self, "_kv_layout", "NHD") in ("NHD", "HND")
                  and qo_indptr.device.type == "cpu" and paged_kv_indptr.device.type == "cpu"
                  and paged_kv_last_page_len.device.type == "cpu" and paged_kv_indices.is_cuda
                  and paged_kv_indices.dtype == torch.int32)
            if ok:
                qo = qo_indptr.tolist()
                if qo[-1] - qo[0] >= MIN_TOKENS:
                    kvi, last = paged_kv_indptr.tolist(), paged_kv_last_page_len.tolist()
                    reqs = []
                    for r in range(len(qo) - 1):
                        npg = kvi[r + 1] - kvi[r]
                        kv_len = (npg - 1) * page_size + last[r] if npg > 0 else 0
                        ntok = qo[r + 1] - qo[r]
                        if ntok > kv_len:
                            reqs = None
                            break
                        reqs.append((qo[r], ntok, kv_len, kvi[r]))
                    if reqs is not None:
                        sm = kw.get("sm_scale")
                        self._turing_fa8 = (reqs, paged_kv_indices, page_size, D ** -0.5 if sm is None else sm,
                                            self._kv_layout)
        except Exception:   # noqa: BLE001
            self._turing_fa8 = None
        if self._turing_fa8 is not None:
            # run() goes to the int8 kernel: no FlashInfer plan (its temp buffers for a large chunk would not fit the
            # small shared workspace); the attributes vLLM's forward asserts on, as FlashInfer's plan sets them
            self._causal, self._window_left = True, kw.get("window_left", -1)
            self._logits_soft_cap = kw.get("logits_soft_cap") or 0.0
            self._sm_scale = kw.get("sm_scale")
            return None
        return orig_plan(self, qo_indptr, paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len, num_qo_heads,
                         num_kv_heads, head_dim_qk, page_size, *args, **kw)

    def run(self, q, paged_kv_cache, *args, **kw):
        st = getattr(self, "_turing_fa8", None)
        if st is None:
            _stats[1] += 1
            return orig_run(self, q, paged_kv_cache, *args, **kw)
        if isinstance(paged_kv_cache, (tuple, list)):
            k_cache, v_cache = paged_kv_cache
        else:
            k_cache, v_cache = paged_kv_cache.unbind(1)
        out = kw.get("out")
        if out is None:
            out = torch.empty_like(q)
        if (args or kw.get("return_lse") or kw.get("lse") is not None or kw.get("sinks") is not None
                or kw.get("kv_cache_sf") is not None or kw.get("window_left") not in (None, -1)
                or q.dtype != torch.float16 or q.dim() != 3 or q.stride(2) != 1
                or k_cache.dtype != torch.float8_e4m3fn or k_cache.stride(-1) != 1 or out.dtype != torch.float16
                or out.stride(2) != 1 or out.stride(1) != D):
            raise RuntimeError("vllm_turing fa8: a prefill planned for the int8 kernel was run with unsupported "
                               "arguments (VLLM_TURING_FA8=0 disables the int8 path)")
        reqs, indices, page_size, scale, layout = st
        scale *= _as_float(kw.get("q_scale"))
        run_prefill(q, k_cache, v_cache, layout, page_size, reqs, indices, scale, _as_float(kw.get("k_scale")),
                    _as_float(kw.get("v_scale")), out)
        _stats[0] += 1
        if _stats[0] % 5000 == 1:
            logger.info("vllm_turing fa8: int8 prefill attention calls %d, FlashInfer %d", *_stats)
        return out

    W.plan, W.run, W._vllm_turing_fa8 = plan, run, True
    global _installed
    _installed = True
    logger.info("vllm_turing: int8-QK prefill attention for FP8 KV (>= %d query tokens)", MIN_TOKENS)
