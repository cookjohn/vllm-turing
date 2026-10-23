"""vLLM's custom all-reduce with expandable allocator segments (vLLM 0.31; 2x 2080 Ti over NVLink). The custom
all-reduce reads the peer's buffers over P2P. Inside CUDA graphs it reads the captured graph tensors directly and, after
capture, exchanges their CUDA IPC handles (register_graph_buffers); memory of PyTorch's expandable segments (cuMemMap)
has no legacy IPC handles: "custom_all_reduce.cuh: invalid argument" at startup. Outside graphs it already copies the
input into its own pre-registered staging buffer (cudaMalloc, IPC-shared at init) and reduces from there. This takes the
copy path in graphs too (the device-to-device copy is captured like the kernel; at most max_size = 8 MB, the decode /
verify all-reduces are 50-300 KB), so every captured address is the staging buffer: vLLM's register_graph_buffers still
runs (a captured all-reduce reads its peer pointers from a slot that only it fills) and exchanges the staging buffer's
legacy handle. The custom all-reduce then runs with the allocator setting that gives ~1 GiB more KV cache.
VLLM_TURING_CUSTOM_AR_COPY=0 disables."""
import logging
import os

import torch

logger = logging.getLogger("vllm.turing")
ENABLED = os.environ.get("VLLM_TURING_CUSTOM_AR_COPY", "1") != "0"


def install():
    if not ENABLED:
        return
    try:
        from vllm.distributed.device_communicators import custom_all_reduce as car
    except ImportError:
        return
    C = car.CustomAllreduce
    if getattr(C, "_vllm_turing_copy", False):
        return

    def custom_all_reduce(self, input):
        if self.disabled or not self.should_custom_ar(input):
            return None
        if self._IS_CAPTURING and not torch.cuda.is_current_stream_capturing():
            return torch.empty_like(input)      # warm-up inside the capture context: allocation pattern only
        return self.all_reduce(input, registered=False)

    C.custom_all_reduce, C._vllm_turing_copy = custom_all_reduce, True
    logger.info("vllm_turing: custom all-reduce through its staging buffer in CUDA graphs")
