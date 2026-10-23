#!/usr/bin/env python3
"""vllm_turing host-memory all-reduce, two processes on GPUs 0 and 1: exact sums (fp16 / bf16, decode sizes and odd
lengths), CUDA-graph replay (a captured sequence of calls must stay correct over replays), latency vs NCCL.
usage: test_hostar.py   (vLLM env python, both GPUs free)"""
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def run(rank):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29533")
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", rank=rank, world_size=2)
    cpu = dist.new_group(backend="gloo")
    nccl = dist.new_group(backend="nccl")
    from vllm_turing.hostar import HostAllReduce
    har = HostAllReduce(cpu, torch.device("cuda", rank))

    torch.manual_seed(1234 + rank)
    bad = 0
    for dt in (torch.float16, torch.bfloat16):
        for n in (5120, 2 * 5120, 3 * 5120, 6 * 5120, 2, 4098, 65536 // 2 * 2 - 2):
            x = torch.randn(n, device="cuda", dtype=dt)
            ref = x.clone()
            dist.all_reduce(ref, group=nccl)
            y = har.all_reduce(x)
            if not torch.equal(y, ref):
                bad += 1
                print(f"rank {rank}: MISMATCH dtype={dt} n={n} max diff {(y.float() - ref.float()).abs().max().item()}")
    # CUDA graph: 64 calls of mixed sizes captured once, replayed 50 times with fresh inputs
    xs = [torch.randn(n, device="cuda", dtype=torch.float16) for n in (5120, 15360, 5120, 30720) * 16]
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for x in xs:
            har.all_reduce(x)   # warm-up outside the graph
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        ys = [har.all_reduce(x) for x in xs]
    for it in range(50):
        for x in xs:
            x.normal_()
        g.replay()
        for x, y in zip(xs, ys):
            ref = x.clone()
            dist.all_reduce(ref, group=nccl)
            if not torch.equal(y, ref):
                bad += 1
                print(f"rank {rank}: graph replay {it} mismatch")
                break
    torch.cuda.synchronize()

    def lat(fn, x, it=2000):
        for _ in range(50):
            fn(x)
        torch.cuda.synchronize()
        dist.barrier(group=cpu)
        t = time.time()
        for _ in range(it):
            fn(x)
        torch.cuda.synchronize()
        return (time.time() - t) / it * 1e6

    for n in (5120, 3 * 5120, 6 * 5120, 65536):
        x = torch.randn(n, device="cuda", dtype=torch.float16)
        t_h = lat(har.all_reduce, x)
        t_n = lat(lambda a: dist.all_reduce(a, group=nccl), x.clone())
        if rank == 0:
            print(f"{n * 2 // 1024:4d} KB: host all-reduce {t_h:6.1f} us, NCCL {t_n:6.1f} us")
    # per-call cost inside a CUDA graph (how vLLM runs decode): 64 calls captured, replayed
    def graph_lat(fn, n, calls=64, reps=50):
        xs = [torch.randn(n, device="cuda", dtype=torch.float16) for _ in range(calls)]
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for x in xs:
                fn(x)
        torch.cuda.current_stream().wait_stream(st)
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            for x in xs:
                fn(x)
        gr.replay()
        torch.cuda.synchronize()
        dist.barrier(group=cpu)
        t = time.time()
        for _ in range(reps):
            gr.replay()
        torch.cuda.synchronize()
        return (time.time() - t) / reps / calls * 1e6

    for n in (5120, 3 * 5120, 6 * 5120):
        t_h = graph_lat(har.all_reduce, n)
        t_n = graph_lat(lambda a: dist.all_reduce(a, group=nccl), n)
        if rank == 0:
            print(f"in CUDA graph, {n * 2 // 1024:3d} KB: host all-reduce {t_h:6.1f} us/call, NCCL {t_n:6.1f} us/call")
    if rank == 0:
        print("correctness:", "all exact" if bad == 0 else f"{bad} mismatches")
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(run, nprocs=2)
