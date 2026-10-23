"""Two-rank all-gather by copy engines through NUMA-local pinned host memory (csrc/dma_ag.cu) for the AR8 prefill
all-reduce (hostar.py) on GPUs without peer access. NCCL's SHM all-gather moves ~5.7 GB/s per direction here and runs
on SMs; the copy engines move 11.3 GB/s per direction with both directions at once when each GPU writes into memory of
its own CPU socket and reads the peer's across the socket link (tests/bench_dma_numa.py; both buffers on one node:
~7.3 GB/s). The segment holds per rank two call-parity buffers of VLLM_TURING_DAG_MB MiB plus flags; each rank's pages
are first touched by a thread bound to its GPU's NUMA node. Used outside CUDA graph capture for payloads of at least
VLLM_TURING_DAG_MIN_KB KiB (NCCL otherwise). VLLM_TURING_DAG=0 disables."""
import ctypes
import logging
import os
import threading

import torch
import torch.distributed as dist

from . import _build

logger = logging.getLogger("vllm.turing")
ENABLED = os.environ.get("VLLM_TURING_DAG", "1") != "0"
CAP = int(os.environ.get("VLLM_TURING_DAG_MB", "40")) << 20      # bytes per buffer (AR8 payload of 6736 x 5120: 35 MB)
MIN_BYTES = int(os.environ.get("VLLM_TURING_DAG_MIN_KB", "2048")) << 10   # 1 MiB: NCCL 0.20 vs 0.22 ms
P = int(os.environ.get("VLLM_TURING_DAG_PIECES", "8"))
FLAG_BYTES = 4096                                                   # per (rank, buffer): P int32 flags
_lib = None


def lib():
    global _lib
    if _lib is None:
        _lib = _build.load("dma_ag")
        _lib.dag_register.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        _lib.dag_register.restype = ctypes.c_void_p
        _lib.dag_all_gather.argtypes = ([ctypes.c_void_p, ctypes.c_longlong, ctypes.c_int, ctypes.c_void_p,
                                         ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
                                        + [ctypes.c_void_p] * 6)
        _lib.dag_all_gather.restype = ctypes.c_int
    return _lib


def _gpu_node_cpus(device):
    """CPUs of the NUMA node of the GPU's PCI slot (None if unknown)."""
    try:
        p = torch.cuda.get_device_properties(device)
        addr = f"{p.pci_domain_id:04x}:{p.pci_bus_id:02x}:{p.pci_device_id:02x}.0"
        node = int(open(f"/sys/bus/pci/devices/{addr}/numa_node").read())
        if node < 0:
            return None, None
        cpus = set()
        for part in open(f"/sys/devices/system/node/node{node}/cpulist").read().strip().split(","):
            a, _, b = part.partition("-")
            cpus.update(range(int(a), int(b or a) + 1))
        return node, cpus
    except Exception:   # noqa: BLE001
        return None, None


class DmaAllGather:
    def __init__(self, cpu_group, device):
        from multiprocessing import shared_memory

        self.rank = dist.get_rank(cpu_group)
        ranks = dist.get_process_group_ranks(cpu_group)
        self.device = device
        region = 2 * CAP + 2 * FLAG_BYTES                       # per rank: two buffers + their flags
        self.size = 2 * region
        name = [None]
        if self.rank == 0:
            self.shm = shared_memory.SharedMemory(create=True, size=self.size)   # pages placed by first touch
            name[0] = self.shm.name
        dist.broadcast_object_list(name, src=ranks[0], group=cpu_group)
        if self.rank != 0:
            self.shm = shared_memory.SharedMemory(name=name[0])
        base = ctypes.addressof(ctypes.c_char.from_buffer(self.shm.buf))
        node, cpus = _gpu_node_cpus(device)
        mine = base + self.rank * region

        def touch():
            if cpus:
                os.sched_setaffinity(0, cpus)
            ctypes.memset(mine, 0, region)                      # this rank's buffers + flags on its GPU's node

        th = threading.Thread(target=touch)
        th.start()
        th.join()
        dist.barrier(group=cpu_group)
        dev_base = lib().dag_register(base, self.size)
        oks = [None, None]
        dist.all_gather_object(oks, bool(dev_base), group=cpu_group)
        if self.rank == 0:
            self.shm.unlink()
        if not all(oks):
            raise RuntimeError("cudaHostRegister of the DMA all-gather segment failed")
        self.host = [base + r * region for r in (0, 1)]
        self.devp = [dev_base + r * region for r in (0, 1)]
        self.seq = 0
        self.d2h = torch.cuda.Stream(device=device)
        self.h2d = torch.cuda.Stream(device=device)
        self.ev = [torch.cuda.Event() for _ in range(3)]
        for e in self.ev:
            e.record()                                          # materialize the CUDA events (lazy in torch)
        logger.info("vllm_turing: DMA all-gather (copy engines, %d MiB x 2 buffers per rank, staging on NUMA node %s)",
                    CAP >> 20, node)

    def usable(self, m):
        return MIN_BYTES <= m <= CAP and not torch.cuda.is_current_stream_capturing()

    def all_gather(self, gathered, m):
        """gathered: device uint8 [2m]; this rank's payload already in half `rank` (written on the current stream)."""
        self.seq += 1
        b = self.seq & 1
        r, q = self.rank, 1 - self.rank
        stage = lambda rank: self.host[rank] + b * CAP
        flags = lambda rank: self.devp[rank] + 2 * CAP + b * FLAG_BYTES
        rc = lib().dag_all_gather(gathered.data_ptr(), m, r, stage(r), stage(q), flags(r), flags(q), self.seq, P,
                                  torch.cuda.current_stream().cuda_stream, self.d2h.cuda_stream,
                                  self.h2d.cuda_stream, self.ev[0].cuda_event, self.ev[1].cuda_event,
                                  self.ev[2].cuda_event)
        if rc != 0:
            raise RuntimeError("DMA all-gather launch failed")
