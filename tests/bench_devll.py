#!/usr/bin/env python3
"""P2P LL all-reduce (hostar.DevLLAllReduce: the host_ar.cu LL kernel on IPC-shared device memory, push mode) on two
GPUs, one process each as in vLLM: bitwise check against x0 + x1 at decode sizes (1-6 requests x 5 MTP verify tokens
x 5120) and the time per all-reduce, eager and in a CUDA graph of 128 back-to-back all-reduces (a decode step has
~146), next to vLLM's custom all-reduce (copy path, as customar.py runs it) where it can be built standalone.
usage: bench_devll.py   (vLLM env python, 2 GPUs with peer access)"""
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

H = 5120


def make(T, rank, seed):
    g = torch.Generator(device="cuda").manual_seed(seed * 2 + rank)
    return (torch.randn(T, H, device="cuda", generator=g) * 4).half()


def run(rank):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29541")
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", rank=rank, world_size=2)
    from vllm_turing.hostar import DevLLAllReduce
    ll = DevLLAllReduce(dist.group.WORLD, torch.device("cuda", rank))
    bad = 0
    for i, T in enumerate((1, 5, 10, 15, 30, 7)):
        x = make(T, rank, i)
        ref = make(T, 0, i).float() + make(T, 1, i).float()
        y = ll.all_reduce(x)
        torch.cuda.synchronize()
        ok = torch.equal(y, ref.half())
        bad += not ok
        if rank == 0:
            print(f"T={T:3d} ({T * H * 2 // 1024} KB): {'bitwise equal' if ok else 'MISMATCH'}", flush=True)
    # stress: many calls without host syncs
    xs = [make(5, rank, 100 + i) for i in range(64)]
    ys = [ll.all_reduce(x) for x in xs]
    torch.cuda.synchronize()
    st = sum(not torch.equal(y, (make(5, 0, 100 + i).float() + make(5, 1, 100 + i).float()).half())
             for i, y in enumerate(ys))
    bad += st
    if rank == 0:
        print(f"stress: 64 back-to-back calls, {64 - st} exact", flush=True)

    ca = None
    try:
        from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
        from vllm_turing import customar
        customar.install()
        ca = CustomAllreduce(group=dist.group.WORLD, device=torch.device("cuda", rank))
        if ca.disabled:
            ca = None
    except Exception as e:   # noqa: BLE001
        if rank == 0:
            print(f"vLLM custom all-reduce not available standalone: {e}", flush=True)
        ca = None

    def timed(fn, n_ar, it=20):
        fn()
        torch.cuda.synchronize()
        dist.barrier()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(it):
            fn()
        e1.record()
        torch.cuda.synchronize()
        return e0.elapsed_time(e1) / it / n_ar * 1e3

    for T in tuple(int(x) for x in os.environ.get("BENCH_T", "5,30").split(",")):
        x = make(T, rank, 7)
        res = {}
        for name, ar in (("LL P2P", ll.all_reduce), ("vLLM custom", (lambda t: ca.custom_all_reduce(t)) if ca else None)):
            if ar is None:
                continue
            eager = timed(lambda: [ar(x) for _ in range(128)], 128)
            g = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            dist.barrier()
            if name.startswith("vLLM"):
                ctx = ca.capture()
                ctx.__enter__()
            with torch.cuda.stream(s):
                for _ in range(2):
                    y = x
                    for _ in range(128):
                        y = ar(x)
                torch.cuda.synchronize()
                dist.barrier()
                with torch.cuda.graph(g, stream=s):
                    for _ in range(128):
                        y = ar(x)
            if name.startswith("vLLM"):
                ctx.__exit__(None, None, None)
            torch.cuda.current_stream().wait_stream(s)
            graph = timed(lambda: g.replay(), 128)
            res[name] = (eager, graph)
        if rank == 0:
            print(f"T={T} ({T * H * 2 // 1024} KB): " + ", ".join(f"{k} eager {e:.1f} us / graph {gr:.1f} us"
                                                                 for k, (e, gr) in res.items()), flush=True)
    if rank == 0:
        print("FAIL" if bad else "PASS", flush=True)
    dist.barrier()


if __name__ == "__main__":
    mp.spawn(run, nprocs=2)
