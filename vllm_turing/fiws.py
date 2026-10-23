"""One FlashInfer workspace per device, allocated before the memory profile (vLLM 0.31, FlashInfer backend). vLLM gives
every set of attention groups (the target model, and separately the MTP / DFlash drafter) its own FlashInfer workspace
(394 MiB default, at least max_num_batched_tokens x Q heads x head dim x 16 bytes), allocated when the groups are
built, after the memory profile: the KV cache was sized without them and the process ran ~0.8 GiB over its budget
(a max-size image prompt then OOMed). Here every FlashInfer builder on a device gets the same buffer (their kernels run
one after another on the compute stream, as vLLM's micro-batch builders already share theirs), allocated at the start
of the profile run so that the KV cache is sized with it. When fiattn8 takes the large prefill chunks, FlashInfer only
runs small prefills (split-KV temp of < 256 query rows at any depth fits) and the decode / verify paths, so the buffer
is VLLM_TURING_FI_WS_MB (default 192) MiB instead of the size formula. VLLM_TURING_FI_WS=0 disables."""
import logging
import os

import torch

logger = logging.getLogger("vllm.turing")
ENABLED = os.environ.get("VLLM_TURING_FI_WS", "1") != "0"
SMALL_MB = int(os.environ.get("VLLM_TURING_FI_WS_MB", "192"))
_shared = {}


def _formula_size(vllm_config):
    from vllm import envs
    from vllm.v1.attention.backends import flashinfer as fib
    size = envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE
    mc, pc = vllm_config.model_config, vllm_config.parallel_config
    est = (vllm_config.scheduler_config.max_num_batched_tokens * mc.get_num_attention_heads(pc) * mc.get_head_size()
           * fib.FLASHINFER_PREFILL_WORKSPACE_BYTES_PER_ELEM)
    return max(size, est)


def workspace(device, vllm_config=None):
    dev = torch.device(device)
    key = dev.index if dev.index is not None else torch.cuda.current_device()
    buf = _shared.get(key)
    if buf is None:
        from . import fiattn8
        if fiattn8.active():
            size = SMALL_MB << 20
        elif vllm_config is not None:
            size = _formula_size(vllm_config)
        else:
            from vllm.config import get_current_vllm_config
            size = _formula_size(get_current_vllm_config())
        buf = _shared[key] = torch.zeros(size, dtype=torch.uint8, device=torch.device("cuda", key))
        logger.info("vllm_turing: shared FlashInfer workspace %.0f MiB on cuda:%d", size / 2**20, key)
    return buf


def install():
    if not ENABLED:
        return
    try:
        from vllm.v1.attention.backends import flashinfer as fib
        from vllm.v1.worker.gpu import model_runner as mr
    except Exception:   # noqa: BLE001
        return
    B = fib.FlashInferMetadataBuilder
    if getattr(B, "_vllm_turing_fiws", False):
        return

    def _get_workspace_buffer(self):
        if self._workspace_buffer is None:
            self._workspace_buffer = workspace(self.device, getattr(self, "vllm_config", None))
        return self._workspace_buffer

    B._get_workspace_buffer, B._vllm_turing_fiws = _get_workspace_buffer, True

    R = mr.GPUModelRunner
    orig_profile = R.profile_run

    def profile_run(self, *args, **kwargs):
        try:
            backend = getattr(getattr(self.vllm_config, "attention_config", None), "backend", None)
            if backend is not None and "FLASHINFER" in str(backend).upper():
                workspace(self.device, self.vllm_config)   # sized into the profile (retained), shared afterwards
        except Exception as e:   # noqa: BLE001
            logger.warning("vllm_turing: FlashInfer workspace preallocation failed: %s", e)
        return orig_profile(self, *args, **kwargs)

    R.profile_run = profile_run
