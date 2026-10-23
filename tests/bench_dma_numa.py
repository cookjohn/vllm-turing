#!/usr/bin/env python3
"""Copy-engine exchange between the two GPUs through pinned host memory with each GPU's staging buffer on its own NUMA
node (GPU0 at 02:00.0 on node 0 = CPUs 0-15,32-47, GPU1 at 83:00.0 on node 1 = CPUs 16-31,48-63): pinned pages are
placed by first touch, so each buffer is allocated from a thread bound to that node's CPUs. Measures D2H + H2D at once
(each GPU writes its own buffer and reads the peer's) for local / remote placements, and the pipelined DMA all-gather.
usage: python bench_dma_numa.py [MiB]   (vLLM env python, both GPUs)"""
import os
import sys
import threading
import time

import torch

MIB = float(sys.argv[1]) if len(sys.argv) > 1 else 32.8
n = int(MIB * 2**20)
NODE_CPUS = {0: set(range(0, 16)) | set(range(32, 48)), 1: set(range(16, 32)) | set(range(48, 64))}


def pinned_on(node):
    out = {}

    def alloc():
        os.sched_setaffinity(0, NODE_CPUS[node])
        t = torch.empty(n, dtype=torch.int8, pin_memory=True)
        t.fill_(0)                          # first touch from this node (cudaHostAlloc may already have touched)
        out["t"] = t

    th = threading.Thread(target=alloc)
    th.start()
    th.join()
    return out["t"]


dev = [torch.device("cuda", d) for d in (0, 1)]
g = [torch.full((n,), d + 1, dtype=torch.int8, device=dev[d]) for d in (0, 1)]
recv = [torch.empty(n, dtype=torch.int8, device=dev[d]) for d in (0, 1)]
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


def run(name, h):
    def duplex():
        for d in (0, 1):
            with torch.cuda.stream(s_d2h[d]):
                h[d].copy_(g[d], non_blocking=True)
            with torch.cuda.stream(s_h2d[d]):
                recv[d].copy_(h[1 - d], non_blocking=True)

    def d2h_only():
        for d in (0, 1):
            with torch.cuda.stream(s_d2h[d]):
                h[d].copy_(g[d], non_blocking=True)

    def h2d_only():
        for d in (0, 1):
            with torch.cuda.stream(s_h2d[d]):
                recv[d].copy_(h[1 - d], non_blocking=True)

    P = 8
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

    gbs = lambda ms: f"{ms:6.3f} ms = {n / ms / 1e6:5.2f} GB/s"
    print(f"{name}: D2H both {gbs(timed(d2h_only))} | H2D both {gbs(timed(h2d_only))} | full duplex "
          f"{gbs(timed(duplex))} | pipelined all-gather {gbs(timed(exchange))}", flush=True)
    assert all(recv[d].eq(2 - d).all().item() for d in (0, 1))


run("buffers local to each writer GPU (h0 node0, h1 node1)", [pinned_on(0), pinned_on(1)])
run("buffers local to each reader GPU (h0 node1, h1 node0)", [pinned_on(1), pinned_on(0)])
run("both buffers on node 0", [pinned_on(0), pinned_on(0)])
run("both buffers on node 1", [pinned_on(1), pinned_on(1)])
