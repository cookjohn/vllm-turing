#!/usr/bin/env python3
"""Copy-engine all-reduce (csrc/ce_ar.cu) in one process on both GPUs: bitwise check against x0 + x1 (sizes and a
stress loop with changing data), time per call at the prefill sizes (one decoder-layer all-reduce of T tokens x 5120
fp16), and the overlap with the int8 prefill GEMM: 1680-token chunks of the MLP gate_up GEMM on the compute streams
while the chunk all-reduces run on high-priority side streams, vs each alone.
usage: bench_ce_ar.py [blocks of the add kernel, default 16]   (vLLM env python, 2 GPUs with peer access)"""
import ctypes
import sys
import time

import torch
import vllm._custom_ops  # noqa: F401

from vllm_turing import _build, w8a8

BLOCKS = int(sys.argv[1]) if len(sys.argv) > 1 else 16
lib = _build.load("ce_ar")
lib.ce_alloc.argtypes = [ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
lib.ce_enable_peer.argtypes = [ctypes.c_int]
lib.ce_all_reduce.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long, ctypes.c_int, ctypes.c_void_p,
                              ctypes.c_void_p, ctypes.c_long, ctypes.c_uint, ctypes.c_int, ctypes.c_void_p]
assert lib.ce_driver_ok() == 0, "no stream memory operations in the driver"
CAP = 18 << 20       # per receive buffer: half of a 3456 x 5120 fp16 tensor
H = 5120
bases = []
for d in (0, 1):
    torch.cuda.set_device(d)
    p, h = ctypes.c_void_p(), ctypes.create_string_buffer(64)
    assert lib.ce_alloc(256 + 2 * CAP, ctypes.byref(p), h) == 0
    assert lib.ce_enable_peer(1 - d) == 0
    bases.append(p.value)
side = [torch.cuda.Stream(device=d, priority=-1) for d in (0, 1)]
seq = [0]


def ar(xs, outs):
    seq[0] += 1
    for r in (0, 1):
        with torch.cuda.device(r):
            rc = lib.ce_all_reduce(xs[r].data_ptr(), outs[r].data_ptr(), xs[r].numel(), r, bases[r], bases[1 - r], CAP,
                                   seq[0], BLOCKS, side[r].cuda_stream)
            assert rc == 0, rc


def sync():
    for d in (0, 1):
        torch.cuda.synchronize(d)


def pair(T, scale=1.0):
    return [torch.randn(T, H, device=f"cuda:{d}").mul_(scale).half() for d in (0, 1)]


def follow():   # the side streams read inputs written on the default streams
    for d in (0, 1):
        side[d].wait_stream(torch.cuda.default_stream(d))


bad = 0
for T in (16, 64, 1680, 1751, 3456):
    xs, outs = pair(T, 4.0), [torch.empty(T, H, dtype=torch.half, device=f"cuda:{d}") for d in (0, 1)]
    follow()
    ar(xs, outs)
    sync()
    ref = (xs[0].cpu() + xs[1].cpu())
    ok = all(torch.equal(o.cpu(), ref) for o in outs)
    bad += not ok
    print(f"T={T:5d}: {'bitwise equal on both ranks' if ok else 'MISMATCH'}")
# stress: back-to-back calls with fresh data, no host sync in between (buffer reuse ordering)
torch.manual_seed(0)
pend = []
for i in range(300):
    T = (1680, 1751, 64, 3456)[i % 4]
    xs, outs = pair(T), [torch.empty(T, H, dtype=torch.half, device=f"cuda:{d}") for d in (0, 1)]
    follow()
    ar(xs, outs)
    for d in (0, 1):
        torch.cuda.default_stream(d).wait_stream(side[d])
    if i % 25 == 0:
        pend.append((xs, outs))
sync()
st = sum(not all(torch.equal(o.cpu(), xs[0].cpu() + xs[1].cpu()) for o in outs) for xs, outs in pend)
bad += st
print(f"stress: 300 calls, {len(pend) - st} of {len(pend)} checked results exact")


def timed(fn, it):
    fn()
    sync()
    t = time.perf_counter()
    for _ in range(it):
        fn()
    sync()
    return (time.perf_counter() - t) / it * 1e3


for T in (1680, 3456):
    xs, outs = pair(T), [torch.empty(T, H, dtype=torch.half, device=f"cuda:{d}") for d in (0, 1)]
    follow()
    ms = timed(lambda: ar(xs, outs), 50)
    print(f"all-reduce {T} x {H} fp16 ({T * H * 2 / 1e6:.1f} MB): {ms:.3f} ms, {T * H * 2 / ms / 1e6:.1f} GB/s")

# overlap: per device NCH chunks of gate_up (M = 1680, K = 5120, N = 17408), each followed by its all-reduce
M, K, N, NCH = 1680, 5120, 17408, 8
g = []
for d in (0, 1):
    with torch.cuda.device(d):
        a = torch.randint(-128, 128, (M, K), dtype=torch.int8, device="cuda")
        b = torch.randint(-128, 128, (N, K), dtype=torch.int8, device="cuda").t()
        sa, sb = torch.rand(M, 1, device="cuda") * 1e-3, torch.rand(N, 1, device="cuda") * 1e-3
        o = torch.empty(M, N, dtype=torch.half, device="cuda")
        assert w8a8.big_ok(o, a, b, sa, sb, None)
        g.append((a, b, sa, sb, o))
xs, outs = pair(M), [torch.empty(M, H, dtype=torch.half, device=f"cuda:{d}") for d in (0, 1)]


def gemms():
    for _ in range(NCH):
        for d in (0, 1):
            with torch.cuda.device(d):
                w8a8.big(g[d][4], *g[d][:4])


def ars():
    follow()
    for _ in range(NCH):
        ar(xs, outs)


def both():
    for _ in range(NCH):
        for d in (0, 1):
            with torch.cuda.device(d):
                w8a8.big(g[d][4], *g[d][:4])
        follow()                     # each all-reduce after the chunk's GEMM (as in the layer tail)
        ar(xs, outs)


t0 = time.perf_counter()
while time.perf_counter() - t0 < 2.0:      # clocks up from idle
    gemms()
    sync()
tg = ta = tb = 1e9
for _ in range(5):                         # interleaved, best of 5 (clock and thermal drift)
    tg, ta, tb = min(tg, timed(gemms, 5)), min(ta, timed(ars, 5)), min(tb, timed(both, 5))
print(f"{NCH} chunks: GEMMs alone {tg:.2f} ms, all-reduces alone {ta:.2f} ms, both overlapped {tb:.2f} ms "
      f"(serial {tg + ta:.2f}; hidden {(tg + ta - tb) / ta * 100:.0f}% of the all-reduce time, GEMM-only bound {tg:.2f})")
print("FAIL" if bad else "PASS")
