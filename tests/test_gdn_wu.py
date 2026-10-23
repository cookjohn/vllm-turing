#!/usr/bin/env python3
"""vllm_turing gdn_wu (csrc/gdn_chunk.cu: kkt + (I + A)^-1 + w, u in one kernel) vs FLA's three Triton kernels
(chunk_scaled_dot_kkt_fwd, solve_tril, recompute_w_u_fwd), both against a float64 reference built from the same fp16
inputs; packed variable-length batch with a short sequence and partial chunks, per-GPU TP2 shapes (H = 24 value heads,
Hg = 8 key heads, K = V = 128), fp32 beta as in vLLM. Also the end-to-end chunk_gated_delta_rule_fwd (o, final state)
with FLA's WY preparation vs ours (the h / o kernels are FLA's in both), and the times.
usage: test_gdn_wu.py [total tokens, default 6720]   (vLLM env python, one GPU)"""
import sys
import time

import torch
import torch.nn.functional as F

from vllm.third_party.flash_linear_attention.ops import chunk as fla_chunk
from vllm.third_party.flash_linear_attention.ops.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from vllm.third_party.flash_linear_attention.ops.cumsum import chunk_local_cumsum
from vllm.third_party.flash_linear_attention.ops.index import prepare_chunk_indices, prepare_chunk_offsets
from vllm.third_party.flash_linear_attention.ops.solve_tril import solve_tril
from vllm.third_party.flash_linear_attention.ops.wy_fast import recompute_w_u_fwd

from vllm_turing import gdn

torch.manual_seed(0)
dev, dt = "cuda", torch.float16
Ttot = int(sys.argv[1]) if len(sys.argv) > 1 else 6720
H, Hg, K, V = 24, 8, 128, 128
cu = torch.tensor([0, 1000, 1037, Ttot], dtype=torch.int32, device=dev)   # partial chunks, a 37-token sequence
N = len(cu) - 1
q = F.normalize(torch.randn(1, Ttot, Hg, K, device=dev), dim=-1).to(dt)
k = F.normalize(torch.randn(1, Ttot, Hg, K, device=dev), dim=-1).to(dt)
v = (torch.randn(1, Ttot, H, V, device=dev) * 0.5).to(dt)
g = -F.softplus(torch.randn(1, Ttot, H, device=dev)) * 0.2
beta = torch.sigmoid(torch.randn(1, Ttot, H, device=dev))               # fp32, as vLLM passes it
h0 = torch.randn(N, H, V, K, device=dev) * 0.05
ci = prepare_chunk_indices(cu, 64)
co = prepare_chunk_offsets(cu, 64)
gc = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu, chunk_indices=ci)


def fla_wu():
    A = chunk_scaled_dot_kkt_fwd(k=k, beta=beta, g=gc, cu_seqlens=cu, chunk_indices=ci, output_dtype=torch.float32)
    A = solve_tril(A=A, cu_seqlens=cu, chunk_indices=ci, output_dtype=k.dtype)
    return recompute_w_u_fwd(k=k, v=v, beta=beta, A=A, g_cumsum=gc, cu_seqlens=cu, chunk_indices=ci)


def exact_wu():
    """float64 w, u per sequence and 64-token chunk (rows past the end padded with zeros), on the CPU (the GPU may
    be serving)."""
    dev = "cpu"
    w = torch.zeros(1, Ttot, H, K, dtype=torch.float64, device=dev)
    u = torch.zeros(1, Ttot, H, V, dtype=torch.float64, device=dev)
    rep = H // Hg
    for n in range(N):
        b, e = cu[n].item(), cu[n + 1].item()
        L = e - b
        P = (L + 63) // 64 * 64
        kk = torch.zeros(P, H, K, dtype=torch.float64, device=dev)
        vv = torch.zeros(P, H, V, dtype=torch.float64, device=dev)
        bb = torch.zeros(P, H, dtype=torch.float64, device=dev)
        gg = torch.zeros(P, H, dtype=torch.float64, device=dev)
        kk[:L] = k[0, b:e].cpu().double().repeat_interleave(rep, dim=1)
        vv[:L] = v[0, b:e].cpu().double()
        bb[:L] = beta[0, b:e].cpu().double()
        gg[:L] = gc[0, b:e].cpu().double()
        C = P // 64
        kc = kk.view(C, 64, H, K).permute(0, 2, 1, 3)               # [C, H, 64, K]
        vc = vv.view(C, 64, H, V).permute(0, 2, 1, 3)
        bc = bb.view(C, 64, H).permute(0, 2, 1)                     # [C, H, 64]
        gch = gg.view(C, 64, H).permute(0, 2, 1)
        A = (bc[..., :, None] * (kc @ kc.transpose(-1, -2)) * torch.exp(gch[..., :, None] - gch[..., None, :]))
        A = torch.tril(A, diagonal=-1)
        Tm = torch.linalg.inv(torch.eye(64, dtype=torch.float64, device=dev) + A)
        uc = Tm @ (vc * bc[..., None])
        wc = Tm @ (kc * (bc * torch.exp(gch))[..., None])
        w[0, b:e] = wc.permute(0, 2, 1, 3).reshape(P, H, K)[:L]
        u[0, b:e] = uc.permute(0, 2, 1, 3).reshape(P, H, V)[:L]
    return w, u


def rel(a, b):
    a, b = a.double().cpu(), b.double().cpu()
    return ((a - b).norm() / b.norm().clamp(min=1e-12)).item()


w_r, u_r = fla_wu()
w_m, u_m = gdn.gdn_wu(k, v, beta, gc, cu, ci)
w_x, u_x = exact_wu()
fails = 0
for name, m, r, x in (("w", w_m, w_r, w_x), ("u", u_m, u_r, u_x)):
    e_mr, e_mx, e_rx = rel(m, r), rel(m, x), rel(r, x)
    mx = (m.double().cpu() - r.double().cpu()).abs().max().item()
    print(f"{name}: ours vs FLA {e_mr:.2e} (max abs {mx:.2e}) | vs float64: ours {e_mx:.2e}, FLA {e_rx:.2e}")
    fails += not (e_mx < 2 * e_rx + 1e-4 and torch.isfinite(m).all().item())

# end to end: FLA's forward with its WY kernels vs ours (FLA's h / o kernels in both)
gdn._orig["fwd"] = fla_chunk.chunk_gated_delta_rule_fwd
args = dict(scale=K ** -0.5, initial_state=h0, output_final_state=True, cu_seqlens=cu, chunk_indices=ci,
            chunk_offsets=co)
_, o_r, _, ht_r, *_ = fla_chunk.chunk_gated_delta_rule_fwd(q, k, v, g, beta, **args)
_, o_m, _, ht_m, *_ = gdn.chunk_gated_delta_rule_fwd(q, k, v, g, beta, **args)
e_o, e_h = rel(o_m, o_r), rel(ht_m, ht_r)
print(f"end to end: o {e_o:.2e}, final state {e_h:.2e}")
fails += not (e_o < 5e-3 and e_h < 5e-3)


def bench(fn, it=20):
    fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(it):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / it * 1e3


t_r, t_m = bench(fla_wu), bench(lambda: gdn.gdn_wu(k, v, beta, gc, cu, ci))
print(f"WY preparation, {Ttot} tokens: FLA Triton (kkt + solve_tril + w_u) {t_r:.3f} ms, gdn_wu {t_m:.3f} ms "
      f"({t_r / t_m:.1f}x)")
print("FAIL" if fails else "PASS")
