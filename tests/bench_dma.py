#!/usr/bin/env python3
"""Copy-engine bandwidth between the two GPUs' memory and pinned host memory (no peer access on this host): D2H / H2D
per GPU, both GPUs at once, host buffers first-touched on the GPU's own NUMA node or the other one, and a pipelined
two-GPU exchange (D2H of piece i on each GPU, then the peer's H2D of piece i: an all-gather done by DMA only, which
leaves the SMs free). One process, events order the streams across devices.
usage: python bench_dma.py [MiB]   (vLLM env python, both GPUs; run under numactl for host placement experiments)"""
import sys
import time

import torch

MIB = float(sys.argv[1]) if len(sys.argv) > 1 else 32.8
n = int(MIB * 2**20)
print("async engines:", [torch.cuda.get_device_properties(d).async_engine_count
                         if hasattr(torch.cuda.get_device_properties(d), "async_engine_count") else "?" for d in (0, 1)])
dev = [torch.device("cuda", d) for d in (0, 1)]
g = [torch.full((n,), d + 1, dtype=torch.int8, device=dev[d]) for d in (0, 1)]
recv = [torch.empty(n, dtype=torch.int8, device=dev[d]) for d in (0, 1)]
h = [torch.empty(n, dtype=torch.int8, pin_memory=True) for _ in (0, 1)]
s_d2h = [torch.cuda.Stream(device=d) for d in dev]
s_h2d = [torch.cuda.Stream(device=d) for d in dev]


def timed(fn, it=20):
    fn()
    for d in dev:
        torch.cuda.synchronize(d)
    t = time.time()
    for _ in range(it):
        fn()
    for d in dev:
        torch.cuda.synchronize(d)
    return (time.time() - t) / it * 1e3


def gbs(ms, nbytes=n):
    return f"{ms:6.3f} ms = {nbytes / ms / 1e6:5.2f} GB/s"


for d in (0, 1):
    with torch.cuda.stream(s_d2h[d]):
        print(f"GPU{d} D2H: {gbs(timed(lambda: h[d].copy_(g[d], non_blocking=True)))}", end="  |  ")
    with torch.cuda.stream(s_h2d[d]):
        print(f"H2D: {gbs(timed(lambda: recv[d].copy_(h[d], non_blocking=True)))}")


def both_d2h():
    for d in (0, 1):
        with torch.cuda.stream(s_d2h[d]):
            h[d].copy_(g[d], non_blocking=True)


def both_h2d():
    for d in (0, 1):
        with torch.cuda.stream(s_h2d[d]):
            recv[d].copy_(h[1 - d], non_blocking=True)


def duplex():
    for d in (0, 1):
        with torch.cuda.stream(s_d2h[d]):
            h[d].copy_(g[d], non_blocking=True)
        with torch.cuda.stream(s_h2d[d]):
            recv[d].copy_(h[1 - d], non_blocking=True)


print(f"both GPUs D2H at once: {gbs(timed(both_d2h))} per GPU")
print(f"both GPUs H2D at once: {gbs(timed(both_h2d))} per GPU")
print(f"both GPUs D2H + H2D at once (full duplex): {gbs(timed(duplex))} per GPU per direction")

for P in (4, 8, 16):
    piece = n // P
    ev = [[torch.cuda.Event() for _ in range(P)] for _ in (0, 1)]

    def exchange():
        for p in range(P):
            sl = slice(p * piece, (p + 1) * piece if p < P - 1 else n)
            for d in (0, 1):
                with torch.cuda.stream(s_d2h[d]):
                    h[d][sl].copy_(g[d][sl], non_blocking=True)
                    ev[d][p].record()
            for d in (0, 1):
                with torch.cuda.stream(s_h2d[d]):
                    s_h2d[d].wait_event(ev[1 - d][p])
                    recv[d][sl].copy_(h[1 - d][sl], non_blocking=True)

    ms = timed(exchange)
    ok = all(recv[d].eq(2 - d).all().item() for d in (0, 1))
    print(f"pipelined DMA all-gather, {P:2d} pieces: {gbs(ms)} per GPU per direction (NCCL ~6.0 ms) "
          f"{'ok' if ok else 'WRONG'}")
