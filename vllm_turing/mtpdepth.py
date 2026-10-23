"""Adaptive MTP draft depth per step (vLLM 0.31, model runner v2): env VLLM_TURING_MTP_DEPTHS="3,5,7" with the MTP
speculative config at num_speculative_tokens = the largest depth. The scheduler always schedules num_speculative_tokens
drafts (async scheduling, fixed verify length), but the drafting cost is one MTP forward per depth step (~1.2 ms on 2x
2080 Ti) while the acceptance per extra position depends on the text: Chinese prose saturates near 2.3 tokens per step
(MTP4 57.6 tok/s, MTP7 50.1), code keeps accepting (MTP4 118, MTP7 146). Each step drafts only the chosen depth; the
later positions keep stale drafts from earlier steps (accepted only where they equal the target's choice: exact).

Choice (_Chooser): expected tokens per step from the per-position acceptance chain over the measured step time per
depth (an exploration-based per-option score, like combo.py's, never re-tried a depth that once scored low and could
not follow acceptance changes inside one answer). Env VLLM_TURING_MTP_DEPTH_BETA (EMA weight, default 0.3),
VLLM_TURING_MTP_DEPTH_REFRESH (full depth every N steps, default 12). Rank 0 logs every 500 steps."""
import logging
import os
from collections import deque

import numpy as np
import torch

logger = logging.getLogger("vllm.turing")

DEPTHS = tuple(int(x) for x in os.environ.get("VLLM_TURING_MTP_DEPTHS", "").split(",") if x.strip())
LAG = 2
SYNC = 256


class _Chooser:
    """Depth from the acceptance chain: q[k] = P(draft k accepted | drafts 1..k-1 accepted), fast EMA over the verified
    steps that drafted position k (censored above the drafted depth); expected tokens per step at depth d =
    1 + sum_k<=d prod_j<=k q[j], over the step time T(d) (measured per depth with CUDA events on rank 0 and broadcast every
    SYNC steps; depths not measured yet from a line fitted to the measured ones). Every REFRESH steps the full depth is
    drafted so the deep positions' estimates follow the text. Deterministic on every TP rank (identical accepted counts,
    LAG-step read-back, synced times)."""

    def __init__(self, options, max_reqs: int, cjk: np.ndarray, verify_len: int):
        from vllm.distributed import get_tp_group
        g = get_tp_group()
        self.cpu_group, self.src, self.tp, self.rank0 = g.cpu_group, g.ranks[0], g.world_size, g.rank_in_group == 0
        self.opts = sorted(options)
        self.verify_len = verify_len
        self.D = self.opts[-1]
        self.q = np.full(self.D + 1, 0.75)
        self.beta = float(os.environ.get("VLLM_TURING_MTP_DEPTH_BETA", "0.3"))
        self.refresh = int(os.environ.get("VLLM_TURING_MTP_DEPTH_REFRESH", "12"))
        self.t_ema = {d: None for d in self.opts}
        self.T = {d: 38.0 + 1.2 * d for d in self.opts}
        self.ring = 16
        self.buf = torch.zeros(self.ring, max_reqs, dtype=torch.int64, pin_memory=True)
        self.ev_data = [torch.cuda.Event() for _ in range(self.ring)]
        self.ev = [torch.cuda.Event(enable_timing=True) for _ in range(self.ring)]
        self.meta = {}
        self.pending = deque()
        self.step = 0
        self.counts = {d: 0 for d in self.opts}
        self.tok = [0.0, 0]

    def record(self, num_sampled, last_tokens, mask: np.ndarray) -> None:
        n = num_sampled.shape[0]
        slot = self.step % self.ring
        self.ev[slot].record()   # propose start of this step
        self.buf[slot, :n].copy_(num_sampled.to(torch.int64), non_blocking=True)
        self.ev_data[slot].record()
        self.pending.append((self.step, mask.copy()))
        while self.pending and self.pending[0][0] <= self.step - LAG:
            t, m = self.pending.popleft()
            s = t % self.ring
            self.ev_data[s].synchronize()
            prev = self.meta.pop(t - 1, None)   # drafts verified at step t were proposed at step t - 1
            if prev is None or not m.any():
                continue
            d, dec = prev
            acc = np.minimum(self.buf[s, :len(m)].numpy()[m] - 1, d)   # accepted drafts among the d drafted
            self.tok[0] += float(acc.mean()) + 1
            self.tok[1] += 1
            for k in range(1, d + 1):
                elig = acc >= k - 1
                if not elig.any():
                    break
                self.q[k] = (1 - self.beta) * self.q[k] + self.beta * float((acc[elig] >= k).mean())
            if dec and m.all():
                ms = self.ev[(t - 1) % self.ring].elapsed_time(self.ev[s])
                e = self.t_ema[d]
                self.t_ema[d] = ms if e is None else 0.9 * e + 0.1 * min(ms, 2.5 * e)

    def _times(self):
        meas = [(d, t) for d, t in self.T.items() if self.t_ema_synced.get(d)] if hasattr(self, "t_ema_synced") else []
        if len(meas) >= 2:
            x = np.array([d for d, _ in meas], dtype=float)
            y = np.array([t for _, t in meas], dtype=float)
            bb, aa = np.polyfit(x, y, 1)
            return {d: (self.T[d] if self.t_ema_synced.get(d) else aa + bb * d) for d in self.opts}
        return self.T

    def expected(self, d: int) -> float:
        return 1.0 + float(np.cumprod(self.q[1:d + 1]).sum())

    def choose(self, decode_only: bool) -> int:
        if self.tp > 1 and self.step % SYNC == SYNC - 1:   # same step on every rank
            import torch.distributed as dist
            t = torch.tensor([self.t_ema[d] if self.t_ema[d] is not None else -1.0 for d in self.opts],
                             dtype=torch.float64)
            dist.broadcast(t, src=self.src, group=self.cpu_group)
            self.t_ema_synced = {}
            for i, d in enumerate(self.opts):
                if t[i] > 0:
                    self.T[d] = float(t[i])
                    self.t_ema_synced[d] = True
        if self.refresh and self.step % self.refresh == self.refresh - 1:
            d = self.D
        else:
            T = self._times()
            d = max(self.opts, key=lambda x: self.expected(x) / T[x])
        self.meta[self.step] = (d, decode_only)
        for k in [k for k in self.meta if k < self.step - self.ring]:
            del self.meta[k]
        self.counts[d] += 1
        self.step += 1
        if self.rank0 and self.step % 500 == 0:
            T = self._times()
            logger.info("vllm_turing mtp depth: step %d, q %s, expected tokens %s, step ms %s, depth steps %s, "
                        "mean tokens/step %.2f", self.step, " ".join("%.2f" % x for x in self.q[1:]),
                        " ".join("%d:%.2f" % (x, self.expected(x)) for x in self.opts),
                        " ".join("%d:%.1f" % (x, T[x]) for x in self.opts), self.counts,
                        self.tok[0] / max(self.tok[1], 1))
        return d


def _make_class():
    from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator

    from .combo import _cjk_table

    class DepthMTPSpeculator(MTPSpeculator):
        def load_model(self, target_model) -> None:
            super().load_model(target_model)
            cjk = _cjk_table(self.vllm_config.model_config.tokenizer, self.vocab_size)
            opts = [d for d in DEPTHS if 1 <= d <= self.num_speculative_steps] or [self.num_speculative_steps]
            self.chooser = _Chooser(opts, self.max_num_reqs, cjk, self.num_speculative_steps + 1)
            logger.info("vllm_turing: adaptive MTP draft depth %s of %d", opts, self.num_speculative_steps)

        def _configure_fused_multi_step_decode(self) -> None:
            super()._configure_fused_multi_step_decode()
            self.use_fused_multi_step_decode = False   # one graph per draft step: the loop can stop early

        @torch.inference_mode()
        def propose(self, input_batch, attn_metadata, slot_mappings, last_hidden_states, aux_hidden_states,
                    num_sampled, num_rejected, last_sampled, next_prefill_tokens, temperature, seeds, dp_sync=None,
                    dummy_run=False, skip_attn_for_dummy_run=False, mm_inputs=None, is_profile=False):
            args = (input_batch, attn_metadata, slot_mappings, last_hidden_states, aux_hidden_states, num_sampled,
                    num_rejected, last_sampled, next_prefill_tokens, temperature, seeds)
            kw = dict(dp_sync=dp_sync, dummy_run=dummy_run, skip_attn_for_dummy_run=skip_attn_for_dummy_run,
                      mm_inputs=mm_inputs, is_profile=is_profile)
            ch = getattr(self, "chooser", None)
            if dummy_run or ch is None:
                return super().propose(*args, **kw)
            num_reqs = input_batch.num_reqs
            mask = np.asarray(input_batch.num_scheduled_tokens[:num_reqs]) == ch.verify_len
            ch.record(num_sampled[:num_reqs], last_sampled.reshape(-1)[input_batch.idx_mapping[:num_reqs]], mask)
            depth = ch.choose(bool(mask.all()))
            n = self.num_speculative_steps
            self.num_speculative_steps = depth
            try:
                super().propose(*args, **kw)
            finally:
                self.num_speculative_steps = n
            return self.draft_tokens[:num_reqs]

    return DepthMTPSpeculator


def install():
    if not DEPTHS:
        return
    import sys

    from vllm.v1.worker.gpu import spec_decode
    if getattr(spec_decode, "_vllm_turing_mtpdepth", False):
        return
    orig = spec_decode.init_speculator

    def init_speculator(vllm_config, device):
        sc = vllm_config.speculative_config
        if sc is not None and sc.method == "mtp" and not sc.use_multi_module_mtp() and not sc.use_gemma4_mtp():
            return _make_class()(vllm_config, device)
        return orig(vllm_config, device)

    spec_decode.init_speculator = init_speculator
    spec_decode._vllm_turing_mtpdepth = True
    mr = sys.modules.get("vllm.v1.worker.gpu.model_runner")
    if mr is not None:
        mr.init_speculator = init_speculator
    logger.info("vllm_turing: adaptive MTP draft depth enabled (%s)", ",".join(map(str, DEPTHS)))
