#!/usr/bin/env python3
"""vllm_turing dmaag (copy-engine all-gather through NUMA-local pinned host memory, csrc/dma_ag.cu), two processes like
vLLM's TP workers (gloo CPU group for the setup): exact gathered halves for several payload sizes; 40 back-to-back calls
without host syncs (each rank writes a new pattern on the compute stream, the gathered peer half is copied out on the
same stream) to exercise the call-parity double buffers; time per call vs NCCL all_gather_into_tensor on the same
buffers. usage: test_dmaag.py   (vLLM env python, both GPUs, NCCL_P2P_DISABLE=1 as in vLLM)"""
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29591")
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("gloo", rank=rank, world_size=2)
    nccl = dist.new_group(backend="nccl", device_id=dev)
    from vllm_turing import dmaag
    dag = dmaag.DmaAllGather(dist.group.WORLD, dev)
    ok = True
    for m in (1 << 20, 8 << 20, 35 * 10**6):
        g = torch.zeros(2 * m, dtype=torch.uint8, device=dev)
        mine, theirs = g[rank * m:(rank + 1) * m], g[(1 - rank) * m:(2 - rank) * m]
        K = 40
        res = torch.empty(K, 4096, dtype=torch.uint8, device=dev)
        for i in range(K):
            mine.fill_((7 * i + 3 * rank + 1) % 251)            # pattern of (rank, call) on the compute stream
            mine[-1] = (i * 13 + rank) % 256
            dag.all_gather(g, m)
            res[i, :4095].copy_(theirs[:4095])
            res[i, 4095] = theirs[-1]
        torch.cuda.synchronize()
        q = 1 - rank
        for i in range(K):
            want = (7 * i + 3 * q + 1) % 251
            if not (res[i, :4095].eq(want).all().item() and res[i, 4095].item() == (i * 13 + q) % 256):
                ok = False
                print(f"rank {rank}: m={m} call {i}: wrong peer data", flush=True)
                break
        dist.barrier()
        it = 20
        torch.cuda.synchronize(); dist.barrier()
        t = time.time()
        for _ in range(it):
            dag.all_gather(g, m)
        torch.cuda.synchronize()
        t_dag = (time.time() - t) / it * 1e3
        src = mine.clone()
        torch.cuda.synchronize(); dist.barrier()
        dist.all_gather_into_tensor(g, src, group=nccl)
        torch.cuda.synchronize(); dist.barrier()
        t = time.time()
        for _ in range(it):
            dist.all_gather_into_tensor(g, src, group=nccl)
        torch.cuda.synchronize()
        t_nccl = (time.time() - t) / it * 1e3
        if rank == 0:
            print(f"payload {m / 2**20:6.1f} MiB/rank: DMA all-gather {t_dag:7.3f} ms ({m / t_dag / 1e6:5.2f} GB/s) | NCCL "
                  f"{t_nccl:7.3f} ms ({m / t_nccl / 1e6:5.2f} GB/s) | x{t_nccl / t_dag:.2f} | back-to-back "
                  f"{'ok' if ok else 'FAILED'}", flush=True)
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    mp.spawn(worker, nprocs=2)
