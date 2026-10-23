#!/usr/bin/env python3
"""Two-GPU all_gather_into_tensor bandwidth for prefill-sized AR8 payloads (int8, 6720 x 5120 per rank = one aligned
prefill chunk) under the current NCCL environment (NCCL_PROTO, NCCL_SHM_USE_CUDA_MEMCPY, NCCL_MIN_NCHANNELS, ...).
usage: NCCL_P2P_DISABLE=1 [NCCL_...] python bench_allgather.py [rows ...]   (vLLM env python, both GPUs)"""
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank, sizes):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29581")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=2, device_id=torch.device("cuda", rank))
    for rows in sizes:
        n = rows * 5120
        src = torch.full((n,), rank + 1, dtype=torch.int8, device="cuda")
        dst = torch.empty(2 * n, dtype=torch.int8, device="cuda")
        for _ in range(5):
            dist.all_gather_into_tensor(dst, src)
        torch.cuda.synchronize()
        dist.barrier()
        it = 20
        t = time.time()
        for _ in range(it):
            dist.all_gather_into_tensor(dst, src)
        torch.cuda.synchronize()
        ms = (time.time() - t) / it * 1e3
        assert dst[:n].eq(1).all() and dst[n:].eq(2).all()
        if rank == 0:
            print(f"  {rows:6d} rows x 5120 int8 ({n / 2**20:6.1f} MiB/rank): {ms:7.3f} ms = {n / ms / 1e6:5.2f} GB/s "
                  f"per direction", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    sizes = [int(a) for a in sys.argv[1:]] or [6720, 3360, 1680, 512]
    keys = ("NCCL_PROTO", "NCCL_ALGO", "NCCL_SHM_USE_CUDA_MEMCPY", "NCCL_MIN_NCHANNELS", "NCCL_MAX_NCHANNELS",
            "NCCL_BUFFSIZE", "NCCL_NTHREADS", "NCCL_SHM_DISABLE", "NCCL_P2P_DISABLE")
    print("env:", {k: os.environ[k] for k in keys if k in os.environ}, flush=True)
    mp.spawn(worker, args=(sizes,), nprocs=2)
