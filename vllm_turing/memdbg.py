"""Memory debugging. VLLM_TURING_MEMDBG=1: startup memory-profile breakdown (the peak tensor memory of the multimodal
encoder profile and of the decoder's profile dummy run separately; vLLM only reports their combined peak as "peak
activation") and the peak of every real step with at least 1024 scheduled tokens (the profile's dummy run skips
attention, so the GDN chunk kernels' and FlashInfer's prefill memory is not in it). VLLM_TURING_MEMDBG=2 also records
the allocator history (slow) and on a CUDA OOM dumps it to /tmp/oom_snapshot_<pid>.pickle with a summary of the largest
live blocks by allocation site."""
import logging
import os

import torch

logger = logging.getLogger("vllm.turing")
LEVEL = int(os.environ.get("VLLM_TURING_MEMDBG", "0") or 0)
ENABLED = LEVEL >= 1


def _measured(name, fn):
    def wrapper(*args, **kwargs):
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        try:
            return fn(*args, **kwargs)
        finally:
            torch.cuda.synchronize()
            logger.info("vllm_turing memdbg: %s peak +%.3f GiB, retained +%.3f GiB (base %.3f GiB)", name,
                        (torch.cuda.max_memory_allocated() - before) / 2**30,
                        (torch.cuda.memory_allocated() - before) / 2**30, before / 2**30)
    return wrapper


def _summarize(snapshot, top=25):
    """Live blocks grouped by their innermost vLLM / vllm_turing / model frame."""
    groups = {}
    for seg in snapshot.get("segments", []):
        for blk in seg.get("blocks", []):
            if blk.get("state") != "active_allocated":
                continue
            frames = blk.get("frames") or []
            site = "?"
            for f in frames:
                fn = f.get("filename", "")
                if "vllm" in fn or "flashinfer" in fn or "flash_linear" in fn:
                    site = f"{fn.split('site-packages/')[-1]}:{f.get('line')} {f.get('name')}"
                    break
            g = groups.setdefault(site, [0, 0])
            g[0] += blk.get("size", 0)
            g[1] += 1
    for site, (size, n) in sorted(groups.items(), key=lambda kv: -kv[1][0])[:top]:
        logger.warning("vllm_turing memdbg OOM: %8.1f MiB in %4d blocks  %s", size / 2**20, n, site)


def _install_history():
    torch.cuda.memory._record_memory_history(max_entries=400000)

    def on_oom(device, alloc, device_alloc, device_free):
        try:
            snap = torch.cuda.memory._snapshot()
            path = f"/tmp/oom_snapshot_{os.getpid()}.pickle"
            import pickle
            with open(path, "wb") as f:
                pickle.dump(snap, f)
            logger.warning("vllm_turing memdbg OOM on device %s (alloc %.1f MiB, free %.1f MiB): snapshot %s", device,
                           alloc / 2**20, device_free / 2**20, path)
            _summarize(snap)
        except Exception as e:   # noqa: BLE001
            logger.warning("vllm_turing memdbg OOM snapshot failed: %s", e)

    torch._C._cuda_attach_out_of_memory_observer(on_oom)


def install():
    if not ENABLED:
        return
    from vllm.v1.worker.gpu import model_runner as mr
    from vllm.v1.worker.gpu.mm import encoder_runner as er
    R = mr.GPUModelRunner
    if getattr(R, "_vllm_turing_memdbg", False):
        return
    er.EncoderRunner.profile_encoder_cache = _measured("encoder profile", er.EncoderRunner.profile_encoder_cache)
    er.EncoderRunner.execute_mm_encoder = _measured("encoder run", er.EncoderRunner.execute_mm_encoder)
    orig_dummy = R._dummy_run
    measured_dummy = _measured("decoder profile dummy run", orig_dummy)

    def _dummy_run(self, *args, **kwargs):
        if kwargs.get("is_profile", False):
            return measured_dummy(self, *args, **kwargs)
        return orig_dummy(self, *args, **kwargs)

    orig_profile = R.profile_run
    measured_profile = _measured("whole profile_run", orig_profile)

    def profile_run(self, *args, **kwargs):
        if LEVEL >= 2:
            _install_history()
        return measured_profile(self, *args, **kwargs)

    orig_exec = R.execute_model

    def execute_model(self, scheduler_output, *args, **kwargs):
        n = getattr(scheduler_output, "total_num_scheduled_tokens", 0) or 0
        if n < 1024:
            return orig_exec(self, scheduler_output, *args, **kwargs)
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        out = orig_exec(self, scheduler_output, *args, **kwargs)
        torch.cuda.synchronize()
        logger.info("vllm_turing memdbg: step with %d tokens peak +%.3f GiB over %.3f GiB allocated (max %.3f GiB, "
                    "reserved %.3f GiB)", n, (torch.cuda.max_memory_allocated() - before) / 2**30, before / 2**30,
                    torch.cuda.max_memory_allocated() / 2**30, torch.cuda.memory_reserved() / 2**30)
        return out

    R._dummy_run, R.profile_run, R.execute_model, R._vllm_turing_memdbg = (_dummy_run, profile_run, execute_model,
                                                                           True)
    logger.info("vllm_turing: memory debugging level %d", LEVEL)
