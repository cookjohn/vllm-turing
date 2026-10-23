#!/usr/bin/env python3
"""vllm_turing GDN chunk kernels vs FLA's Triton originals: inputs built with the FLA pipeline itself (chunk-local
cumsum, kkt, solve_tril, recompute_w_u), variable-length batch with an initial state, per-GPU TP2 shapes
(H = 24 value heads, Hg = 8 key heads, K = V = 128). Compares h, v_new, the final state and o; times both.
usage: test_gdn.py [total tokens, default 4096]   (vLLM env python, one GPU)"""
import sys
import time

import torch
import torch.nn.functional as F

from vllm.third_party.flash_linear_attention.ops import chunk as fla_chunk
from vllm.third_party.flash_linear_attention.ops.chunk_delta_h import chunk_gated_delta_rule_fwd_h as orig_h
from vllm.third_party.flash_linear_attention.ops.chunk_o import chunk_fwd_o as orig_o
from vllm.third_party.flash_linear_attention.ops.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from vllm.third_party.flash_linear_attention.ops.cumsum import chunk_local_cumsum
from vllm.third_party.flash_linear_attention.ops.index import prepare_chunk_indices, prepare_chunk_offsets
from vllm.third_party.flash_linear_attention.ops.solve_tril import solve_tril
from vllm.third_party.flash_linear_attention.ops.wy_fast import recompute_w_u_fwd

from vllm_turing import gdn

torch.manual_seed(0)
dev, dt = "cuda", torch.float16
Ttot = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
H, Hg, K, V = 24, 8, 128, 128
cu = torch.tensor([0, 1000, Ttot], dtype=torch.int32, device=dev)   # two sequences, the first ends mid-chunk
N = len(cu) - 1
q = F.normalize(torch.randn(1, Ttot, Hg, K, device=dev), dim=-1).to(dt)
k = F.normalize(torch.randn(1, Ttot, Hg, K, device=dev), dim=-1).to(dt)
v = (torch.randn(1, Ttot, H, V, device=dev) * 0.5).to(dt)
g = -F.softplus(torch.randn(1, Ttot, H, device=dev)) * 0.2         # log decay per token
beta = torch.sigmoid(torch.randn(1, Ttot, H, device=dev)).to(dt)
h0 = torch.randn(N, H, V, K, device=dev) * 0.05
ci = prepare_chunk_indices(cu, 64)
co = prepare_chunk_offsets(cu, 64)

gc = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu, chunk_indices=ci)
A = chunk_scaled_dot_kkt_fwd(k=k, beta=beta, g=gc, cu_seqlens=cu, chunk_indices=ci, output_dtype=torch.float32)
A = solve_tril(A=A, cu_seqlens=cu, chunk_indices=ci, output_dtype=k.dtype)
w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A, g_cumsum=gc, cu_seqlens=cu, chunk_indices=ci)

gdn._orig["h"], gdn._orig["o"] = orig_h, orig_o
args = dict(g=gc, initial_state=h0, output_final_state=True, cu_seqlens=cu, chunk_indices=ci, chunk_offsets=co)
h_r, vn_r, ht_r = orig_h(k, w, u, **args)
h_m, vn_m, ht_m = gdn.chunk_gated_delta_rule_fwd_h(k, w, u, **args)
o_r = orig_o(q, k, vn_r, h_r, g=gc, scale=K ** -0.5, cu_seqlens=cu, chunk_indices=ci)
o_m = gdn.chunk_fwd_o(q, k, vn_m, h_m, g=gc, scale=K ** -0.5, cu_seqlens=cu, chunk_indices=ci)
o_x = gdn.chunk_fwd_o(q, k, vn_r, h_r, g=gc, scale=K ** -0.5, cu_seqlens=cu, chunk_indices=ci)   # o kernel alone


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-12)).item()


print(f"T={Ttot}: rel err  h {rel(h_m, h_r):.2e}  v_new {rel(vn_m, vn_r):.2e}  final {rel(ht_m, ht_r):.2e}  "
      f"o {rel(o_m, o_r):.2e}  (o kernel on the same inputs {rel(o_x, o_r):.2e})")


def bench(fn, it=20):
    fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(it):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / it * 1e3


th_r = bench(lambda: orig_h(k, w, u, **args))
th_m = bench(lambda: gdn.chunk_gated_delta_rule_fwd_h(k, w, u, **args))
to_r = bench(lambda: orig_o(q, k, vn_r, h_r, g=gc, scale=K ** -0.5, cu_seqlens=cu, chunk_indices=ci))
to_m = bench(lambda: gdn.chunk_fwd_o(q, k, vn_r, h_r, g=gc, scale=K ** -0.5, cu_seqlens=cu, chunk_indices=ci))
print(f"chunk_delta_h: triton {th_r:.2f} ms, cuda {th_m:.2f} ms ({th_r / th_m:.1f}x) | "
      f"chunk_o: triton {to_r:.2f} ms, cuda {to_m:.2f} ms ({to_r / to_m:.1f}x)")
