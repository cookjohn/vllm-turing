"""Activation calibration for SmoothQuant: with env VLLM_TURING_CALIB=<dir>, every loaded model gets forward pre-hooks on
its language-model linear layers that keep the per-input-channel max |x| (fp32), saved as <dir>/rank<r>[_draft].pt
({module name: tensor[K local]}) every 32 recorded forwards of the first hooked module and at exit. Row-parallel layers
see their K shard (rank order = shard order). Run the server eagerly (--enforce-eager, compilation mode 0) or the hooks
are compiled away."""
import atexit
import logging
import os

import torch

logger = logging.getLogger("vllm.turing")   # under "vllm": uses vLLM's log handler
TARGETS = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj", "in_proj_qkvz", "in_proj_ba", "out_proj")


def _attach(model, out_dir, tag):
    from vllm.distributed import get_tensor_model_parallel_rank
    from vllm.model_executor.layers.linear import LinearBase

    rank = get_tensor_model_parallel_rank()
    path = os.path.join(out_dir, f"rank{rank}{tag}.pt")
    stats, calls = {}, [0]
    names = [n for n, m in model.named_modules() if isinstance(m, LinearBase) and n.split(".")[-1] in TARGETS
             and "visual" not in n]
    first = names[0] if names else None

    def save():
        torch.save({k: v.cpu() for k, v in stats.items()}, path)

    def make(name):
        def hook(mod, args):
            x = args[0]
            if not torch.is_tensor(x) or x.numel() == 0:
                return
            a = x.reshape(-1, x.shape[-1]).abs().amax(dim=0).float()
            prev = stats.get(name)
            stats[name] = a if prev is None else torch.maximum(prev, a)
            if name == first:
                calls[0] += 1
                if calls[0] % 32 == 0:
                    save()
        return hook

    mods = dict(model.named_modules())
    for n in names:
        mods[n].register_forward_pre_hook(make(n))
    atexit.register(save)
    logger.info("vllm_turing: calibration hooks on %d linear layers -> %s", len(names), path)


def install():
    out_dir = os.environ.get("VLLM_TURING_CALIB")
    if not out_dir:
        return
    os.makedirs(out_dir, exist_ok=True)
    from vllm.model_executor.model_loader import base_loader
    L = base_loader.BaseModelLoader
    if getattr(L, "_vllm_turing_calib", False):
        return
    orig = L.load_model
    seen = [0]

    def load_model(self, *args, **kwargs):
        model = orig(self, *args, **kwargs)
        try:
            _attach(model, out_dir, "" if seen[0] == 0 else f"_m{seen[0]}")   # second load = the MTP draft model
            seen[0] += 1
        except Exception as e:   # noqa: BLE001
            logger.warning("vllm_turing: calibration hooks failed (%s)", e)
        return model

    L.load_model = load_model
    L._vllm_turing_calib = True
