#!/usr/bin/env python3
"""Compressed (int8, per-128 fp16 scale) large all-reduce vs NCCL fp16 all-reduce, two processes on GPUs 0 and 1:
relative error against the exact fp32 sum, identical result on both ranks, latency at prefill sizes.
usage: test_ar8.py   (vLLM env python, both GPUs free)"""
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def run(rank):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29534")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=2)
    from vllm_turing import hostar
    torch.manual_seed(7 + rank)
    for tokens in (512, 2048, 4096):
        x = torch.randn(tokens, 5120, device="cuda", dtype=torch.float16)
        x[:, 17] *= 40   # an outlier channel, like the residual stream's massive activations
        exact = x.float().clone()
        dist.all_reduce(exact)
        y = hostar.ar8_all_reduce(x, None, None)
        rel = ((y.float() - exact).norm() / exact.norm()).item()
        other = y.clone()
        dist.broadcast(other, src=0)
        same = torch.equal(other, y)

        def lat(fn, it=20):
            fn()
            torch.cuda.synchronize()
            t = time.time()
            for _ in range(it):
                fn()
            torch.cuda.synchronize()
            return (time.time() - t) / it * 1e3

        t8 = lat(lambda: hostar.ar8_all_reduce(x, None, None))
        tn = lat(lambda: dist.all_reduce(x.clone()))
        if rank == 0:
            print(f"{tokens:5d} tokens ({tokens * 5120 * 2 / 2**20:.0f} MB): ar8 {t8:.2f} ms vs NCCL fp16 {tn:.2f} ms, "
                  f"rel err {rel:.2e}, identical on both ranks: {same}")
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(run, nprocs=2)
