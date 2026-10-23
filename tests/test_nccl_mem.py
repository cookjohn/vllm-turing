#!/usr/bin/env python3
"""Device memory NCCL allocates outside the torch allocator on the two 2080 Ti (no P2P: NCCL_P2P_DISABLE=1), per first
use of all_reduce / all_gather at prefill sizes, through vLLM's PyNcclCommunicator (the communicator vLLM and the AR8
path use). Prints cudaMemGetInfo-used minus torch-reserved after each step on rank 0.
usage: NCCL_P2P_DISABLE=1 test_nccl_mem.py   (vLLM env python, two GPUs)"""
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def nontorch(dev):
    free, total = torch.cuda.mem_get_info(dev)
    return (total - free - torch.cuda.memory_reserved(dev)) / 2**20


def run(rank):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29533")
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("gloo", rank=rank, world_size=2)
    torch.zeros(1, device=dev)
    log = []
    log.append(("context", nontorch(dev)))
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    comm = PyNcclCommunicator(group=dist.group.WORLD, device=dev)
    torch.cuda.synchronize()
    log.append(("pynccl init", nontorch(dev)))
    x = torch.randn(8192 * 5120, device=dev, dtype=torch.float16)
    for i in range(3):
        comm.all_reduce(x)
    torch.cuda.synchronize()
    log.append(("all_reduce 80 MB", nontorch(dev)))
    p = torch.empty(8192 * 5120 + 8192 * 80, dtype=torch.uint8, device=dev)
    g = torch.empty(2 * p.numel(), dtype=torch.uint8, device=dev)
    for i in range(3):
        comm.all_gather(g, p)
    torch.cuda.synchronize()
    log.append(("all_gather 42 MB", nontorch(dev)))
    s = torch.empty(4096 * 5120 + 4096 * 80, dtype=torch.uint8, device=dev)
    gs = torch.empty(2 * s.numel(), dtype=torch.uint8, device=dev)
    comm.all_gather(gs, s)
    torch.cuda.synchronize()
    log.append(("all_gather 21 MB", nontorch(dev)))
    if rank == 0:
        for k, v in log:
            print(f"{k:20s} non-torch device memory {v:8.1f} MiB")
    dist.barrier()


if __name__ == "__main__":
    mp.spawn(run, nprocs=2)
