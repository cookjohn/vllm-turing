"""DFlash / DFlash2 drafter diagnostics (env VLLM_TURING_DRAFT_DEBUG=1, run with --enforce-eager so the hooks fire on
every step): per call, the absmax and non-finite count of the aux target hidden states (per target layer), the fc /
hidden_norm output, each draft decoder layer, the final norm; and the proposed draft tokens next to the tokens the
target then sampled. Logs the first VLLM_TURING_DRAFT_DEBUG_N calls (default 12) on TP rank 0."""
import logging
import os

import torch

logger = logging.getLogger("vllm.turing")
N = int(os.environ.get("VLLM_TURING_DRAFT_DEBUG_N", "12"))
_calls = {"n": 0}


def _st(t):
    if not isinstance(t, torch.Tensor):
        ts = [x for x in t if isinstance(x, torch.Tensor)] if isinstance(t, (tuple, list)) else []
        return " | ".join(_st(x) for x in ts) if ts else "-"
    f = t.float()
    bad = (~torch.isfinite(f)).sum().item()
    fin = f[torch.isfinite(f)]
    am = fin.abs().max().item() if fin.numel() else float("nan")
    return f"{tuple(t.shape)} absmax {am:.3g} nonfinite {bad}"


def _rank0():
    try:
        from vllm.distributed import get_tensor_model_parallel_rank
        return get_tensor_model_parallel_rank() == 0
    except Exception:   # noqa: BLE001
        return True


def install():
    if os.environ.get("VLLM_TURING_DRAFT_DEBUG") != "1":
        return
    from vllm.model_executor.models import qwen3_dflash as qd
    M = qd.DFlashQwen3Model
    orig_init = M.__init__

    def __init__(self, *a, **k):
        orig_init(self, *a, **k)

        def hook(name):
            def f(mod, inp, out):
                if _calls["n"] < N and _rank0():
                    logger.info("dflash-debug call %d %s in %s out %s", _calls["n"], name, _st(inp), _st(out))
            return f

        for i, layer in enumerate(self.layers):
            layer.register_forward_hook(hook(f"layer{i}"))
        self.norm.register_forward_hook(hook("norm"))
        self.hidden_norm.register_forward_hook(hook("hidden_norm"))
        if hasattr(self, "fc"):
            self.fc.register_forward_hook(hook("fc"))

    M.__init__ = __init__

    from vllm.v1.worker.gpu.spec_decode.dflash import speculator as ds
    S = ds.DFlashSpeculator
    orig_propose = S.propose

    def propose(self, input_batch, attn_metadata, slot_mappings, last_hidden_states, aux_hidden_states, num_sampled,
                num_rejected, last_sampled, *a, **k):
        dummy = k.get("dummy_run", False)
        if not dummy and _calls["n"] < N and _rank0() and aux_hidden_states:
            for j, h in enumerate(aux_hidden_states):
                logger.info("dflash-debug call %d aux[%d] %s", _calls["n"], j, _st(h))
            logger.info("dflash-debug call %d last_sampled %s num_sampled %s num_rejected %s", _calls["n"],
                        last_sampled[:input_batch.num_reqs].tolist(), num_sampled[:input_batch.num_reqs].tolist(),
                        num_rejected[:input_batch.num_reqs].tolist())
        out = orig_propose(self, input_batch, attn_metadata, slot_mappings, last_hidden_states, aux_hidden_states,
                           num_sampled, num_rejected, last_sampled, *a, **k)
        if not dummy and _rank0():
            if _calls["n"] < N:
                logger.info("dflash-debug call %d drafts %s", _calls["n"], out.tolist())
            _calls["n"] += 1
        return out

    S.propose = propose
    logger.info("vllm_turing: DFlash drafter debug hooks installed")
