"""Tensor-core GDN chunked prefill (csrc/gdn_chunk.cu) in place of FLA's Triton chunk_gated_delta_rule_fwd_h and
chunk_fwd_o (vllm.third_party.flash_linear_attention.ops.chunk calls them by module-level name). Same signatures and
results; anything outside the supported case (fp16, K = 128, chunk 64, natural-exp scalar gating, no gk) goes to the
original Triton function. VLLM_TURING_GDN=0 disables.
chunk_gated_delta_rule_fwd itself is replaced too: the WY preparation (FLA's chunk_scaled_dot_kkt_fwd, solve_tril and
recompute_w_u_fwd, three Triton kernels) runs as one kernel (gdn_wu) that keeps A and its inverse in shared memory;
A is returned as None (the autograd forward does not use it). VLLM_TURING_GDN_WU=0 keeps FLA's three kernels."""
import ctypes
import os

import torch

from . import _build

_lib = None
_orig = {}
WU = os.environ.get("VLLM_TURING_GDN_WU", "1") != "0"


def lib():
    global _lib
    if _lib is None:
        _lib = _build.load("gdn_chunk")
        _lib.gdn_h_fp16.argtypes = [ctypes.c_void_p] * 10 + [ctypes.c_int] * 5 + [ctypes.c_void_p]
        _lib.gdn_h_fp16.restype = ctypes.c_int
        _lib.gdn_o_fp16.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_float] + [ctypes.c_int] * 5 + [ctypes.c_void_p]
        _lib.gdn_o_fp16.restype = ctypes.c_int
        _lib.gdn_wu_fp16.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
        _lib.gdn_wu_fp16.restype = ctypes.c_int
    return _lib


def _ptr(t):
    return t.data_ptr() if t is not None else None


def _i32(t):
    return t if t is None or t.dtype == torch.int32 else t.to(torch.int32)


def chunk_gated_delta_rule_fwd_h(k, w, u, g=None, gk=None, initial_state=None, output_final_state=False,
                                 chunk_size=64, save_new_value=True, cu_seqlens=None, chunk_indices=None,
                                 chunk_offsets=None, use_exp2=False):
    B, T, Hg, K = k.shape
    H, V = u.shape[-2], u.shape[-1]
    if not (k.dtype == torch.float16 and w.dtype == k.dtype and u.dtype == k.dtype and g is not None
            and g.dtype == torch.float32 and gk is None and not use_exp2 and K == 128 and V % 64 == 0
            and chunk_size == 64 and save_new_value and k.is_contiguous() and w.is_contiguous()
            and u.is_contiguous() and g.is_contiguous() and H % Hg == 0):
        return _orig["h"](k, w, u, g=g, gk=gk, initial_state=initial_state, output_final_state=output_final_state,
                          chunk_size=chunk_size, save_new_value=save_new_value, cu_seqlens=cu_seqlens,
                          chunk_indices=chunk_indices, chunk_offsets=chunk_offsets, use_exp2=use_exp2)
    from vllm.third_party.flash_linear_attention.ops.index import prepare_chunk_indices, prepare_chunk_offsets
    if cu_seqlens is None:
        N, NT = B, (T + 63) // 64
    else:
        N = len(cu_seqlens) - 1
        if chunk_indices is None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, 64)
        NT = len(chunk_indices)
        if chunk_offsets is None:
            chunk_offsets = prepare_chunk_offsets(cu_seqlens, 64)
    h = k.new_empty(B, NT, H, V, K)
    final_state = k.new_empty(N, H, V, K, dtype=torch.float32) if output_final_state else None
    v_new = torch.empty_like(u)
    h0 = initial_state
    if h0 is not None and (h0.dtype != torch.float32 or not h0.is_contiguous()):
        h0 = h0.float().contiguous()
    rc = lib().gdn_h_fp16(_ptr(k), _ptr(w), _ptr(u), _ptr(g), _ptr(h0), _ptr(h), _ptr(v_new), _ptr(final_state),
                          _ptr(_i32(cu_seqlens)), _ptr(_i32(chunk_offsets)), N, T, H, Hg, V,
                          torch.cuda.current_stream().cuda_stream)
    if rc != 0:
        raise RuntimeError(f"gdn_h launch failed: {rc}")
    return h, v_new, final_state


def chunk_fwd_o(q, k, v, h, g=None, scale=None, cu_seqlens=None, chunk_indices=None, chunk_size=64,
                core_attn_out=None):
    B, T, Hg, K = q.shape
    H, V = v.shape[-2], v.shape[-1]
    if not (q.dtype == torch.float16 and k.dtype == q.dtype and v.dtype == q.dtype and h.dtype == q.dtype
            and g is not None and g.dtype == torch.float32 and K == 128 and V % 64 == 0 and chunk_size == 64
            and q.is_contiguous() and k.is_contiguous() and v.is_contiguous() and h.is_contiguous()
            and g.is_contiguous() and H % Hg == 0):
        return _orig["o"](q, k, v, h, g=g, scale=scale, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
                          chunk_size=chunk_size, core_attn_out=core_attn_out)
    from vllm.third_party.flash_linear_attention.ops.index import prepare_chunk_indices
    if cu_seqlens is not None and chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, 64)
    NT = (T + 63) // 64 * B if cu_seqlens is None else len(chunk_indices)
    if scale is None:
        scale = K ** -0.5
    if core_attn_out is not None:
        assert core_attn_out.numel() >= v.numel()
        o = core_attn_out[: v.numel()].view(*v.shape)
    else:
        o = torch.empty_like(v)
    ci = None if chunk_indices is None else _i32(chunk_indices).contiguous()
    rc = lib().gdn_o_fp16(_ptr(q), _ptr(k), _ptr(v), _ptr(h), _ptr(g), _ptr(o), _ptr(_i32(cu_seqlens)), _ptr(ci),
                          float(scale), NT, T, H, Hg, V, torch.cuda.current_stream().cuda_stream)
    if rc != 0:
        raise RuntimeError(f"gdn_o launch failed: {rc}")
    return o


def gdn_wu(k, v, beta, g, cu_seqlens=None, chunk_indices=None):
    """w, u of FLA's WY representation (= recompute_w_u_fwd(k, v, beta, solve_tril(chunk_scaled_dot_kkt_fwd(k, beta,
    g)), g)) in one kernel; k [B, T, Hg, 128], v [B, T, H, 128] fp16, beta [B, T, H] fp32, g the chunk-local cumsum."""
    B, T, Hg, K = k.shape
    H = v.shape[-2]
    if cu_seqlens is not None:
        if chunk_indices is None:
            from vllm.third_party.flash_linear_attention.ops.index import prepare_chunk_indices
            chunk_indices = prepare_chunk_indices(cu_seqlens, 64)
        NT = len(chunk_indices)
    else:
        NT = B * ((T + 63) // 64)
    w = k.new_empty(B, T, H, K)
    u = torch.empty_like(v)
    rc = lib().gdn_wu_fp16(_ptr(k), _ptr(v), _ptr(beta), _ptr(g), _ptr(w), _ptr(u), _ptr(_i32(cu_seqlens)),
                           _ptr(_i32(chunk_indices)), NT, T, H, Hg, torch.cuda.current_stream().cuda_stream)
    if rc != 0:
        raise RuntimeError(f"gdn_wu launch failed: {rc}")
    return w, u


def chunk_gated_delta_rule_fwd(q, k, v, g, beta, scale, initial_state, output_final_state, cu_seqlens=None,
                               chunk_indices=None, chunk_offsets=None, core_attn_out=None):
    from vllm.third_party.flash_linear_attention.ops import chunk as fla_chunk
    B, T, Hg, K = k.shape
    H, V = v.shape[-2], v.shape[-1]
    if not (k.dtype == torch.float16 and v.dtype == k.dtype and beta.dtype == torch.float32 and K == 128
            and V == 128 and H % Hg == 0 and tuple(beta.shape) == (B, T, H) and k.is_contiguous()
            and v.is_contiguous() and beta.is_contiguous() and (cu_seqlens is None or B == 1)):
        return _orig["fwd"](q, k, v, g, beta, scale, initial_state, output_final_state, cu_seqlens=cu_seqlens,
                            chunk_indices=chunk_indices, chunk_offsets=chunk_offsets, core_attn_out=core_attn_out)
    g = fla_chunk.chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices)
    if not g.is_contiguous():
        g = g.contiguous()
    w, u = gdn_wu(k, v, beta, g, cu_seqlens, chunk_indices)
    h, v_new, final_state = fla_chunk.chunk_gated_delta_rule_fwd_h(
        k=k, w=w, u=u, g=g, initial_state=initial_state, output_final_state=output_final_state,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, chunk_offsets=chunk_offsets)
    o = fla_chunk.chunk_fwd_o(q=q, k=k, v=v_new, h=h, g=g, scale=scale, cu_seqlens=cu_seqlens,
                              chunk_indices=chunk_indices, core_attn_out=core_attn_out)
    if getattr(fla_chunk, "SUPPRESS_LEVEL", 0) < 3:
        return g, o, None, final_state, None, None, None
    return g, o, None, final_state, w, h, v_new


def install():
    from vllm.platforms import current_platform
    cap = current_platform.get_device_capability()
    if cap is None or cap.to_int() != 75 or os.environ.get("VLLM_TURING_GDN", "1") == "0":
        return
    if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "GDN_OFF")):  # off switch file
        return
    from vllm.third_party.flash_linear_attention.ops import chunk as fla_chunk
    if "h" in _orig:
        return
    _orig["h"] = fla_chunk.chunk_gated_delta_rule_fwd_h
    _orig["o"] = fla_chunk.chunk_fwd_o
    fla_chunk.chunk_gated_delta_rule_fwd_h = chunk_gated_delta_rule_fwd_h
    fla_chunk.chunk_fwd_o = chunk_fwd_o
    if WU:
        _orig["fwd"] = fla_chunk.chunk_gated_delta_rule_fwd
        fla_chunk.chunk_gated_delta_rule_fwd = chunk_gated_delta_rule_fwd
