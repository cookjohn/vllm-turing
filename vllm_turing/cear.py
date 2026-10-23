"""Two-rank fp16 all-reduce on the copy engines over P2P (csrc/ce_ar.cu) for the tensor-parallel group when the GPUs
have peer access (2x 2080 Ti NVLink): peer DMA pushes for the reduce-scatter and the all-gather, the cross-GPU ordering
as stream memory operations, one small add kernel. 17.2 MB (a 1680-token layer all-reduce) in 0.50 ms vs NCCL 0.61 ms,
and it holds no SMs while the data moves: next to the int8 prefill GEMMs it hides 90% of its time at 2.4% GEMM
slowdown (NCCL's kernels: 12%). Each rank owns a buffer (256 bytes of flags, then two receive buffers of CAP bytes)
shared with the peer by CUDA IPC. Used by the prefill layer-tail overlap (tailoverlap.py) and, in vLLM's all-reduce
dispatch, for messages above the custom all-reduce's range that fit (hostar.py). VLLM_TURING_CE_AR=0 disables;
VLLM_TURING_CE_AR_MB: receive buffer size (default: half of a tail chunk of the largest batch)."""
import ctypes
import logging
import os

import torch
import torch.distributed as dist

from . import _build

logger = logging.getLogger("vllm.turing")
ENABLED = os.environ.get("VLLM_TURING_CE_AR", "1") != "0"
BLOCKS = int(os.environ.get("VLLM_TURING_CE_AR_BLOCKS", "8"))   # add-kernel blocks (each takes an SM from the GEMMs)
_lib = None


def lib():
    global _lib
    if _lib is None:
        _lib = _build.load("ce_ar")
        _lib.ce_alloc.argtypes = [ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        _lib.ce_open.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        _lib.ce_all_reduce.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long, ctypes.c_int, ctypes.c_void_p,
                                       ctypes.c_void_p, ctypes.c_long, ctypes.c_uint, ctypes.c_int, ctypes.c_void_p]
        _lib.ce_all_reduce.restype = ctypes.c_int
    return _lib


def default_cap(chunks: float) -> int:
    """Bytes per receive buffer: half of the largest layer-tail chunk (max batch / chunks + 64 rows)."""
    mb = os.environ.get("VLLM_TURING_CE_AR_MB")
    if mb:
        return int(float(mb) * (1 << 20))
    try:
        from vllm.config import get_current_vllm_config
        cfg = get_current_vllm_config()
        tokens = cfg.scheduler_config.max_num_batched_tokens
        hidden = cfg.model_config.get_hidden_size()
    except Exception:   # noqa: BLE001
        tokens, hidden = 6750, 5120
    rows = int(tokens / max(chunks, 1.0)) + 64
    return (rows * hidden * 2 // 2 + 255) // 256 * 256


class CeAllReduce:
    def __init__(self, cpu_group, device, cap: int, blocks: int = BLOCKS):
        self.rank = dist.get_rank(cpu_group)
        self.cap, self.blocks, self.seq = cap, blocks, 0
        L = lib()
        with torch.cuda.device(device):
            p, h = ctypes.c_void_p(), ctypes.create_string_buffer(64)
            rc = L.ce_alloc(256 + 2 * cap, ctypes.byref(p), h)
            if rc != 0:
                raise RuntimeError(f"ce_alloc failed: {rc}")
            self.my = p.value
            handles = [None, None]
            dist.all_gather_object(handles, h.raw, group=cpu_group)
            q = ctypes.c_void_p()
            ok = L.ce_open(ctypes.create_string_buffer(handles[1 - self.rank], 64), ctypes.byref(q)) == 0
            oks = [None, None]
            dist.all_gather_object(oks, ok, group=cpu_group)
            if not all(oks):
                raise RuntimeError("cudaIpcOpenMemHandle of the peer's buffer failed")
            self.peer = q.value

    def usable(self, t: torch.Tensor) -> bool:
        n = t.numel()
        return (t.dtype == torch.float16 and t.is_contiguous() and n % 16 == 0 and t.data_ptr() % 16 == 0
                and (n - n // 2) * 2 <= self.cap)

    def all_reduce(self, t: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        """out (a new tensor if None) = the sum over both ranks, on the current stream; both ranks call in the same order."""
        if out is None:
            out = torch.empty_like(t)
        self.seq += 1
        rc = lib().ce_all_reduce(t.data_ptr(), out.data_ptr(), t.numel(), self.rank, self.my, self.peer, self.cap,
                                 self.seq & 0xFFFFFFFF, self.blocks, torch.cuda.current_stream().cuda_stream)
        if rc != 0:
            raise RuntimeError(f"copy-engine all-reduce failed: {rc}")
        return out
