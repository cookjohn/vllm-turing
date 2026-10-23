"""Two-rank small-message all-reduce through pinned host shared memory (csrc/host_ar.cu) for the tensor-parallel group
when the GPUs have no peer access (NCCL: ~28 us per decode all-reduce on 2x 2080 Ti across sockets). Hooked into
vLLM's CudaCommunicator: messages up to SLOT_BYTES take this path, larger ones (prefill) stay on NCCL.
VLLM_TURING_HAR=0 disables."""
import ctypes
import logging
import os

import torch
import torch.distributed as dist

from . import _build

logger = logging.getLogger("vllm.turing")   # under "vllm": uses vLLM's log handler
SLOT_BYTES = 128 * 1024              # largest all-reduce on this path (data bytes)
SLOT_WORDS = SLOT_BYTES // 4         # 32-bit data words per slot
REGION = 2 * SLOT_WORDS * 8          # per rank: double buffer of 8-byte LL words
_lib = None


def lib():
    global _lib
    if _lib is None:
        _lib = _build.load("host_ar")
        _lib.har_register.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        _lib.har_register.restype = ctypes.c_void_p
        _lib.har_blocks.argtypes = [ctypes.c_int]
        _lib.har_blocks.restype = ctypes.c_int
        _lib.har_all_reduce.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                                        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        _lib.har_all_reduce.restype = ctypes.c_int
        _lib.ar8_quant.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        _lib.ar8_quant.restype = ctypes.c_int
        _lib.ar8_dequant_sum.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        _lib.ar8_dequant_sum.restype = ctypes.c_int
        _lib.har_preload.restype = ctypes.c_int
    return _lib


# Large (prefill) all-reduces: int8 + fp16 scale per 128 values over an NCCL all-gather (about half the PCIe bytes of
# fp16, the prefill all-reduce bottleneck on GPUs without peer access); both ranks sum the same dequantized operands.
# Lossy: env VLLM_TURING_AR8=1 enables (validate with perplexity).
AR8 = os.environ.get("VLLM_TURING_AR8", "0") == "1"
PERSIST = os.environ.get("VLLM_TURING_AR8_PERSIST", "1") != "0"   # persistent AR8 scratch (_ar8_scratch)
# with peer access: AR8 over NCCL for all-reduces of at least this many bytes (0, the default: off; its scratch costs
# ~0.1 GiB of KV cache, which a 262144-token context at utilization 0.95 does not have)
AR8_P2P_MIN = int(os.environ.get("VLLM_TURING_AR8_P2P_MIN_KB", "0")) << 10
# with peer access: the copy-engine all-reduce (cear.py) for eager messages from this size up to its buffers (smaller
# ones: vLLM's custom all-reduce, max 8 MB); created with the layer-tail overlap (or VLLM_TURING_CE_AR_MB set)
CE_MIN = 8 << 20


def ar8_ok(t: torch.Tensor) -> bool:
    return t.dtype == torch.float16 and t.is_contiguous() and t.numel() % 128 == 0 and t.numel() * 2 > SLOT_BYTES


_ar8_buf = {}   # device -> persistent uint8 scratch: packed (n + n / 64) then gathered (2x that)


def _ar8_scratch(n: int, dev) -> torch.Tensor | None:
    """One persistent scratch per device, sized by the first (largest: the profile run's max batch) call and grown only
    outside CUDA graph capture (captured graphs keep its address). Fresh per-call buffers of these odd sizes fragmented
    the caching allocator: ~0.8 GiB less KV cache at 8192-token batches."""
    need = 3 * (n + n // 64)
    b = _ar8_buf.get(dev)
    if b is None or b.numel() < need:
        if torch.cuda.is_current_stream_capturing():
            return None
        _ar8_buf[dev] = b = torch.empty(need, dtype=torch.uint8, device=dev)
    return b


def ar8_all_reduce(t: torch.Tensor, pynccl, device_group, dag=None) -> torch.Tensor:
    n = t.numel()
    st = torch.cuda.current_stream().cuda_stream
    m = n + n // 64
    sc = _ar8_scratch(n, t.device) if PERSIST else None
    gathered = sc[m:3 * m] if sc is not None else torch.empty(2 * m, dtype=torch.uint8, device=t.device)
    if dag is not None and dag.usable(m):
        # copy-engine all-gather (dmaag.py): quantize straight into this rank's half
        if lib().ar8_quant(t.data_ptr(), gathered.data_ptr() + dag.rank * m, n, st) != 0:
            raise RuntimeError("ar8_quant failed")
        dag.all_gather(gathered, m)
        out = torch.empty_like(t)
        if lib().ar8_dequant_sum(gathered.data_ptr(), out.data_ptr(), n, st) != 0:
            raise RuntimeError("ar8_dequant_sum failed")
        return out
    packed = sc[:m] if sc is not None else torch.empty(m, dtype=torch.uint8, device=t.device)
    if lib().ar8_quant(t.data_ptr(), packed.data_ptr(), n, st) != 0:
        raise RuntimeError("ar8_quant failed")
    if pynccl is not None and not pynccl.disabled:
        pynccl.all_gather(gathered, packed)
    else:
        dist.all_gather_into_tensor(gathered, packed, group=device_group)
    out = torch.empty_like(t)
    if lib().ar8_dequant_sum(gathered.data_ptr(), out.data_ptr(), n, st) != 0:
        raise RuntimeError("ar8_dequant_sum failed")
    return out


class HostAllReduce:
    def __init__(self, cpu_group, device):
        from multiprocessing import shared_memory

        self.rank = dist.get_rank(cpu_group)
        ranks = dist.get_process_group_ranks(cpu_group)
        name = [None]
        if self.rank == 0:
            self.shm = shared_memory.SharedMemory(create=True, size=2 * REGION)
            self.shm.buf[:] = bytes(2 * REGION)   # all flags 0: the first sequence number is 1
            name[0] = self.shm.name
        dist.broadcast_object_list(name, src=ranks[0], group=cpu_group)
        if self.rank != 0:
            self.shm = shared_memory.SharedMemory(name=name[0])
        self._addr = ctypes.addressof(ctypes.c_char.from_buffer(self.shm.buf))
        dev = lib().har_register(self._addr, 2 * REGION)
        ok = [dev is not None and dev != 0]
        oks = [None, None]
        dist.all_gather_object(oks, ok[0], group=cpu_group)
        if self.rank == 0:
            self.shm.unlink()                     # both ranks are attached: the name is no longer needed
        if not all(oks):
            raise RuntimeError("cudaHostRegister of the shared segment failed")
        self.my = dev + self.rank * REGION
        self.peer = dev + (1 - self.rank) * REGION
        self.counters = torch.zeros(lib().har_blocks(SLOT_WORDS), dtype=torch.int32, device=device)

    def should_use(self, t: torch.Tensor) -> bool:
        return (t.dtype in (torch.float16, torch.bfloat16) and t.is_contiguous() and t.numel() % 2 == 0
                and 0 < t.numel() * 2 <= SLOT_BYTES)

    def all_reduce(self, t: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(t)
        rc = lib().har_all_reduce(t.data_ptr(), out.data_ptr(), t.numel() // 2, 0 if t.dtype == torch.float16 else 1,
                                  self.my, self.peer, self.counters.data_ptr(), SLOT_WORDS,
                                  torch.cuda.current_stream().cuda_stream)
        if rc != 0:
            raise RuntimeError(f"host all-reduce launch failed: {rc}")
        return out


# With peer access (NVLink) the same LL kernel runs on device memory in push mode: each rank writes its LL words into
# the peer's receive region (P2P stores over the link, IPC-shared cudaMalloc buffer) and polls its own, local region.
# Replaces vLLM's custom all-reduce for messages up to LL_BYTES. In CUDA graphs (tests/bench_devll.py, 10-22): 50 KB
# (1 request x 5 MTP verify tokens x 5120 fp16) 5.5 vs 7.9 us, 100 KB 7.5 vs 8.9, 150 KB 9.0 vs 8.7, 300 KB 16.1 vs
# 13.2 (the LL words carry 4 data bytes in 8: twice the link bytes, so larger messages stay with the custom
# all-reduce). VLLM_TURING_LL=0 disables, VLLM_TURING_LL_KB sets the limit.
LL_ENABLED = os.environ.get("VLLM_TURING_LL", "1") != "0"
LL_BYTES = int(os.environ.get("VLLM_TURING_LL_KB", "128")) << 10


class DevLLAllReduce:
    def __init__(self, cpu_group, device):
        from . import cear
        self.rank = dist.get_rank(cpu_group)
        self.slot_words = LL_BYTES // 4
        region = 2 * self.slot_words * 8
        L = cear.lib()
        with torch.cuda.device(device):
            if lib().har_preload() != 0:
                raise RuntimeError("har_kernel preload failed")
            p, h = ctypes.c_void_p(), ctypes.create_string_buffer(64)
            if L.ce_alloc(region, ctypes.byref(p), h) != 0:
                raise RuntimeError("LL region allocation failed")
            self.local = p.value                 # polled: the peer writes here
            handles = [None, None]
            dist.all_gather_object(handles, h.raw, group=cpu_group)
            q = ctypes.c_void_p()
            ok = L.ce_open(ctypes.create_string_buffer(handles[1 - self.rank], 64), ctypes.byref(q)) == 0
            oks = [None, None]
            dist.all_gather_object(oks, ok, group=cpu_group)
            if not all(oks):
                raise RuntimeError("cudaIpcOpenMemHandle of the peer's LL region failed")
            self.remote = q.value                # written: the peer's region
            self.counters = torch.zeros(lib().har_blocks(self.slot_words), dtype=torch.int32, device=device)

    def should_use(self, t: torch.Tensor) -> bool:
        return (t.dtype in (torch.float16, torch.bfloat16) and t.is_contiguous() and t.numel() % 2 == 0
                and 0 < t.numel() * 2 <= LL_BYTES)

    def all_reduce(self, t: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(t)
        rc = lib().har_all_reduce(t.data_ptr(), out.data_ptr(), t.numel() // 2, 0 if t.dtype == torch.float16 else 1,
                                  self.remote, self.local, self.counters.data_ptr(), self.slot_words,
                                  torch.cuda.current_stream().cuda_stream)
        if rc != 0:
            raise RuntimeError(f"LL all-reduce launch failed: {rc}")
        return out


def install():
    from vllm.platforms import current_platform
    cap = current_platform.get_device_capability()
    if cap is None or cap.to_int() != 75 or os.environ.get("VLLM_TURING_HAR", "1") == "0":
        return
    if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "HAR_OFF")):  # off switch file
        return
    from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
    if getattr(CudaCommunicator, "_vllm_turing_har", False):
        return
    orig_init, orig_ar = CudaCommunicator.__init__, CudaCommunicator.all_reduce

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self.turing_har = None
        try:
            name = getattr(self, "unique_name", "") or ""
            n = torch.cuda.device_count()
            peer_ok = n > 1 and all(torch.cuda.can_device_access_peer(i, j) for i in range(n) for j in range(n) if i != j)
            if self.world_size == 2 and name.startswith("tp") and not peer_ok and self.device.type == "cuda":
                self.turing_har = HostAllReduce(self.cpu_group, self.device)
                logger.info("vllm_turing: host-memory all-reduce for %s (<= %d KB)", name, SLOT_BYTES // 1024)
        except Exception as e:   # noqa: BLE001
            logger.warning("vllm_turing: host all-reduce unavailable (%s), NCCL is used", e)
            self.turing_har = None
        self.turing_dag = None
        # with peer access (NVLink) the int8-compressed all-reduce still halves the bytes of the large prefill
        # all-reduces: quantize + NCCL all-gather over the link + dequantize-sum, 1.68 vs 2.24 ms for a 6720-token
        # chunk (NCCL fp16); smaller messages (decode, verify) keep vLLM's custom all-reduce / NCCL
        self.turing_ar8_p2p = False
        try:
            if (self.turing_har is None and AR8 and AR8_P2P_MIN > 0 and self.world_size == 2
                    and name.startswith("tp") and peer_ok):
                self.turing_ar8_p2p = True
                logger.info("vllm_turing: int8-compressed all-reduce over NCCL for %s (>= %d KB)", name,
                            AR8_P2P_MIN >> 10)
        except Exception:   # noqa: BLE001
            self.turing_ar8_p2p = False
        self.turing_ce = None
        try:
            from . import cear, tailoverlap
            if (self.turing_har is None and cear.ENABLED and self.world_size == 2 and name.startswith("tp")
                    and peer_ok and self.device.type == "cuda"
                    and (tailoverlap.ENABLED or os.environ.get("VLLM_TURING_CE_AR_MB"))):
                self.turing_ce = cear.CeAllReduce(self.cpu_group, self.device, cear.default_cap(1.0 / tailoverlap.MAX_PART))
                logger.info("vllm_turing: copy-engine all-reduce for %s (2 x %.1f MB receive buffers)", name,
                            self.turing_ce.cap / 2**20)
        except Exception as e:   # noqa: BLE001
            logger.warning("vllm_turing: copy-engine all-reduce unavailable (%s)", e)
            self.turing_ce = None
        self.turing_ll = None
        try:
            if (self.turing_har is None and LL_ENABLED and self.world_size == 2 and name.startswith("tp") and peer_ok
                    and self.device.type == "cuda"):
                self.turing_ll = DevLLAllReduce(self.cpu_group, self.device)
                logger.info("vllm_turing: LL all-reduce over P2P device memory for %s (<= %d KB)", name,
                            LL_BYTES >> 10)
        except Exception as e:   # noqa: BLE001
            logger.warning("vllm_turing: P2P LL all-reduce unavailable (%s)", e)
            self.turing_ll = None
        if self.turing_har is not None and AR8:
            try:
                from . import dmaag
                if dmaag.ENABLED:
                    self.turing_dag = dmaag.DmaAllGather(self.cpu_group, self.device)
            except Exception as e:   # noqa: BLE001
                logger.warning("vllm_turing: DMA all-gather unavailable (%s), AR8 uses NCCL", e)
                self.turing_dag = None

    def all_reduce(self, input_):
        har = getattr(self, "turing_har", None)
        if har is not None:
            if har.should_use(input_):
                return har.all_reduce(input_)
            if AR8 and ar8_ok(input_):
                return ar8_all_reduce(input_, getattr(self, "pynccl_comm", None), self.device_group,
                                      getattr(self, "turing_dag", None))
        elif getattr(self, "turing_ll", None) is not None and self.turing_ll.should_use(input_):
            return self.turing_ll.all_reduce(input_)
        elif (getattr(self, "turing_ar8_p2p", False) and input_.numel() * 2 >= AR8_P2P_MIN and ar8_ok(input_)
              and not torch.cuda.is_current_stream_capturing()):
            return ar8_all_reduce(input_, getattr(self, "pynccl_comm", None), self.device_group, None)
        else:
            ce = getattr(self, "turing_ce", None)
            if (ce is not None and input_.numel() * 2 >= CE_MIN and ce.usable(input_)
                    and not torch.cuda.is_current_stream_capturing()):
                return ce.all_reduce(input_)
        return orig_ar(self, input_)

    CudaCommunicator.__init__ = __init__
    CudaCommunicator.all_reduce = all_reduce
    CudaCommunicator._vllm_turing_har = True
