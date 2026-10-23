"""MTP + DFlash2 combined speculative decoding with per-step adaptive drafter choice (vLLM 0.31, model runner v2).

Launch with the DFlash2 drafter as the speculative config (method "dflash", num_speculative_tokens = 7, i.e. the
8-token verify of the DFlash block) and env VLLM_TURING_COMBO=1: the speculator becomes ComboSpeculator, the DFlash2
speculator with the target's MTP head attached as a second drafter (MTPSpeculator, VLLM_TURING_COMBO_MTP_N draft
tokens, default 4). The scheduler always schedules num_speculative_tokens drafts per request (async scheduling), so
every step verifies 8 tokens: MTP steps pad their drafts by repeating the last one (a padded token is accepted only
where it equals the target's own choice, so greedy and sampled outputs stay exact).

Each step one drafter drafts; the other only updates its context so it never falls behind: on DFlash steps the MTP
head runs its step 0 over the verified tokens (fills its KV cache, one MTP forward), on MTP steps DFlash writes the
context K/V of the new target features (fc + per-layer K/V, no draft forward).

Choice: like the llama.cpp version (LLAMA_SPEC_ADAPTIVE), throughput = EMA(tokens per step) / step time per drafter,
scored separately for CJK-dominated and other text (Chinese prose favours MTP, code DFlash), with exploration of the
trailing drafter every 16 steps (if within 75 % of the leader) or every 64. Every TP rank runs the same controller on
the same inputs (accepted counts and sampled tokens are identical on all ranks), read back LAG steps late through
pinned buffers so the CPU never waits on the current step; step times are measured with CUDA events (period from one
propose to the next, decode-only steps) and rank 0's values are broadcast every SYNC steps, starting from
VLLM_TURING_COMBO_T_MTP / _T_DFL (ms). VLLM_TURING_COMBO_FORCE=mtp|dflash pins one drafter, =alt alternates them (testing). Every 500 steps
rank 0 logs scores, step times and the per-path split (other drafter's update / drafting)."""
import copy
import logging
import os
from collections import deque

import numpy as np
import torch

logger = logging.getLogger("vllm.turing")

ENABLED = os.environ.get("VLLM_TURING_COMBO", "0") == "1"
N_MTP = int(os.environ.get("VLLM_TURING_COMBO_MTP_N", "4"))
T_MTP = float(os.environ.get("VLLM_TURING_COMBO_T_MTP", "35"))
T_DFL = float(os.environ.get("VLLM_TURING_COMBO_T_DFL", "42"))
FORCE = os.environ.get("VLLM_TURING_COMBO_FORCE", "")
LAG = 2
SYNC = 256
CTX_CHUNK = 2048
MTP, DFL = 0, 1
NAMES = ("mtp", "dflash")


def _cjk_table(tokenizer_path: str, vocab: int) -> np.ndarray:
    """Per token id: 1 = contains CJK, 0 = other letters, -1 = neutral (spaces, digits, punctuation, special)."""
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    texts = tk.batch_decode([[i] for i in range(min(vocab, len(tk)))])
    t = np.full(vocab, -1, dtype=np.int8)
    for i, s in enumerate(texts):
        cjk = any("㐀" <= c <= "鿿" or "　" <= c <= "〿" or "＀" <= c <= "￯" for c in s)
        if cjk:
            t[i] = 1
        elif any(c.isalpha() for c in s):
            t[i] = 0
    return t


class _Controller:
    """Drafter choice per step. Identical on every TP rank: the inputs (accepted counts, sampled tokens) are identical
    and are read back LAG steps late, the step times are rank 0's CUDA-event measurements broadcast to all ranks every
    SYNC steps (live per-rank timings would differ and split the collectives)."""

    def __init__(self, max_reqs: int, cjk: np.ndarray, verify_len: int):
        from vllm.distributed import get_tp_group
        g = get_tp_group()
        self.cpu_group, self.src, self.tp, self.rank0 = g.cpu_group, g.ranks[0], g.world_size, g.rank_in_group == 0
        self.T = [T_MTP, T_DFL]          # ms per step of each path (verify + the drafter's work), synced
        self.t_ema = [None, None]        # this rank's measurement
        # prior tokens per step (accepted + bonus) per bucket (0 other, 1 CJK) and drafter, from the single-drafter runs
        self.score = {(0, MTP): 3.9, (0, DFL): 4.6, (1, MTP): 3.0, (1, DFL): 2.3}
        self.seen = {k: 0 for k in self.score}
        self.beta = 0.25
        self.cjk_ema = 0.0
        self.cjk = cjk
        self.verify_len = verify_len
        self.ring = 16
        self.buf = torch.zeros(self.ring, 2, max_reqs, dtype=torch.int64, pin_memory=True)
        self.ev_data = [torch.cuda.Event() for _ in range(self.ring)]
        self.ev = [[torch.cuda.Event(enable_timing=True) for _ in range(3)] for _ in range(self.ring)]
        self.meta = {}           # step -> (choice, bucket, decode-only)
        self.pending = deque()   # (step, decode mask) whose verification data is in the ring
        self.step = 0
        self.counts = [0, 0]
        self.diag = [[0.0, 0.0, 0] for _ in range(2)]   # per choice: other drafter's update ms, drafting ms, n

    def bucket(self) -> int:
        return 1 if self.cjk_ema > 0.5 else 0

    def mark(self, i: int) -> None:
        """Timing points of the current step: 0 propose start, 1 after the other drafter's update, 2 after drafting
        (the slot is fixed at mark 0: choose() advances self.step in between)."""
        if i == 0:
            self.cur = self.step % self.ring
        self.ev[self.cur][i].record()

    def record(self, num_sampled: torch.Tensor, last_tokens: torch.Tensor, mask: np.ndarray) -> None:
        """This step's verification of the previous step's drafts (async copy, read LAG steps later)."""
        n = num_sampled.shape[0]
        slot = self.step % self.ring
        # dtype conversion on the GPU first: a non-blocking device-to-host copy_ that also converts (int32 -> int64)
        # goes through a CPU temporary read before the transfer lands (stale counts, scrambled scores)
        self.buf[slot, 0, :n].copy_(num_sampled.to(torch.int64), non_blocking=True)
        self.buf[slot, 1, :n].copy_(last_tokens.to(torch.int64), non_blocking=True)
        self.ev_data[slot].record()
        self.pending.append((self.step, mask.copy()))
        while self.pending and self.pending[0][0] <= self.step - LAG:
            t, m = self.pending.popleft()
            s = t % self.ring
            self.ev_data[s].synchronize()
            ns = self.buf[s, 0, :len(m)].numpy()
            toks = self.buf[s, 1, :len(m)].numpy()
            for tok in (toks[m] if m.any() else toks):
                c = self.cjk[tok] if 0 <= tok < len(self.cjk) else -1
                if c >= 0:
                    self.cjk_ema = 0.8 * self.cjk_ema + 0.2 * float(c)
            prev = self.meta.pop(t - 1, None)   # drafts verified at step t were proposed at step t - 1
            if prev is None or not m.any():
                continue
            k = (prev[1], prev[0])
            v = float(ns[m].mean())
            self.score[k] = v if self.seen[k] == 0 else (1 - self.beta) * self.score[k] + self.beta * v
            self.seen[k] += 1
            # step period t-1 -> t (drafting at t-1 plus the verify at t) belongs to the drafter chosen at t-1
            if prev[2] and m.all():
                e0, e1 = self.ev[(t - 1) % self.ring], self.ev[s]
                e1[0].synchronize()
                ms = e0[0].elapsed_time(e1[0])
                d = prev[0]
                if self.t_ema[d] is None:
                    self.t_ema[d] = ms
                else:
                    self.t_ema[d] = 0.9 * self.t_ema[d] + 0.1 * min(ms, 2.5 * self.t_ema[d])
                dg = self.diag[d]
                dg[0] += e0[0].elapsed_time(e0[1])
                dg[1] += e0[1].elapsed_time(e0[2])
                dg[2] += 1

    def choose(self, decode_only: bool) -> int:
        if self.tp > 1 and self.step % SYNC == SYNC - 1:   # same step on every rank
            import torch.distributed as dist
            t = torch.tensor([x if x is not None else -1.0 for x in self.t_ema], dtype=torch.float64)
            dist.broadcast(t, src=self.src, group=self.cpu_group)
            for d in (MTP, DFL):
                if t[d] > 0:
                    self.T[d] = float(t[d])
        if FORCE in NAMES:
            c = NAMES.index(FORCE)
        elif FORCE == "alt":   # test: drafters alternate (each one's context from the other's steps every 2nd step)
            c = self.step % 2
        else:
            b = self.bucket()
            rate = [self.score[(b, d)] / self.T[d] for d in (MTP, DFL)]
            lead = int(rate[DFL] > rate[MTP])
            period = 16 if rate[1 - lead] >= 0.75 * rate[lead] else 64
            c = 1 - lead if self.step % period == period - 1 else lead
        self.meta[self.step] = (c, self.bucket(), decode_only)
        for k in [k for k in self.meta if k < self.step - self.ring]:
            del self.meta[k]
        self.counts[c] += 1
        self.step += 1
        if self.rank0 and self.step % 500 == 0:
            dg = ["%.2f/%.2f ms" % (g[0] / g[2], g[1] / g[2]) if g[2] else "-" for g in self.diag]
            logger.info("vllm_turing combo: step %d, cjk %.2f, tokens/step mtp %.2f/%.2f dflash %.2f/%.2f "
                        "(other/cjk), step ms mtp %.1f dflash %.1f (synced %.1f/%.1f), update/draft mtp %s "
                        "dflash %s, steps mtp %d dflash %d", self.step, self.cjk_ema, self.score[(0, MTP)],
                        self.score[(1, MTP)], self.score[(0, DFL)], self.score[(1, DFL)], self.t_ema[0] or 0,
                        self.t_ema[1] or 0, self.T[0], self.T[1], dg[0], dg[1], *self.counts)
        return c


def _make_classes():
    from vllm.config import SpeculativeConfig
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import prepare_dflash_inputs
    from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator
    from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator

    class ComboMTP(MTPSpeculator):
        """MTP drafter drafting N_MTP tokens whose step 0 (draft prefill over the target's verify batch) is captured
        for the engine's verify length (1 + 7), not its own 1 + N_MTP."""

        def __init__(self, vllm_config, device, verify_len: int):
            super().__init__(vllm_config, device)
            self.verify_len = verify_len

        def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
            n = self.num_speculative_steps
            self.num_speculative_steps = self.verify_len - 1   # prefill manager: uniform query len verify_len
            try:
                super().init_cudagraph_manager(cudagraph_mode)
            finally:
                self.num_speculative_steps = n

        def context_only(self, *args, **kwargs) -> None:
            """Step 0 only: fills the MTP KV cache for the verified tokens (its draft is discarded)."""
            n = self.num_speculative_steps
            self.num_speculative_steps = 1
            try:
                self.propose(*args, **kwargs)
            finally:
                self.num_speculative_steps = n

    class ComboSpeculator(DFlash2Speculator):
        def __init__(self, vllm_config, device):
            super().__init__(vllm_config, device)
            sc = vllm_config.speculative_config
            mtp_sc = SpeculativeConfig(method="mtp", num_speculative_tokens=N_MTP,
                                       target_model_config=sc.target_model_config,
                                       target_parallel_config=sc.target_parallel_config)
            mcfg = copy.copy(vllm_config)
            mcfg.speculative_config = mtp_sc
            self.mtp = ComboMTP(mcfg, device, verify_len=self.num_speculative_steps + 1)
            self.verify_len = self.num_speculative_steps + 1
            self.ctl = None
            logger.info("vllm_turing combo: DFlash2 (%d drafts) + MTP (%d drafts), verify %d tokens",
                        self.num_speculative_steps, N_MTP, self.verify_len)

        # --- lifecycle: the runner sees one speculator; both drafters get every call ---------------------------
        def load_model(self, target_model) -> None:
            super().load_model(target_model)
            self._dfl_layers = set(self.draft_attn_layer_names)
            # one pruned / 4-bit draft head for both: DFlash2 takes its candidates from it, MTP its greedy drafts
            self.mtp._turing_dv_reuse = getattr(self, "_turing_dv", None)
            self.mtp.load_model(target_model)
            self.draft_attn_layer_names = self._dfl_layers | self.mtp.draft_attn_layer_names
            tok = self.vllm_config.model_config.tokenizer
            cjk = _cjk_table(tok, self.vocab_size)
            self.ctl = _Controller(self.max_num_reqs, cjk, self.verify_len)
            logger.info("vllm_turing combo: draft layers dflash %d, mtp %d; CJK tokens %d", len(self._dfl_layers),
                        len(self.mtp.draft_attn_layer_names), int((cjk == 1).sum()))

        def set_attn(self, model_state, kv_cache_config, block_tables, target_input_buffers, target_attn_groups):
            union = self.draft_attn_layer_names
            self.draft_attn_layer_names = self._dfl_layers
            try:
                super().set_attn(model_state, kv_cache_config, block_tables, target_input_buffers, target_attn_groups)
            finally:
                self.draft_attn_layer_names = union
            self.mtp.set_attn(model_state, kv_cache_config, block_tables, target_input_buffers, target_attn_groups)

        def init_cudagraph_manager(self, cudagraph_mode) -> None:
            super().init_cudagraph_manager(cudagraph_mode)
            self.mtp.init_cudagraph_manager(cudagraph_mode)

        def capture(self) -> None:
            super().capture()
            self.mtp.capture()

        def observe_verification(self, idx_mapping, num_sampled, num_rejected) -> None:
            super().observe_verification(idx_mapping, num_sampled, num_rejected)
            self.mtp.observe_verification(idx_mapping, num_sampled, num_rejected)

        def set_eplb_state(self, eplb_state) -> None:
            super().set_eplb_state(eplb_state)
            self.mtp.set_eplb_state(eplb_state)

        # --- DFlash context-only update (MTP steps) -------------------------------------------------------------
        def _dflash_context(self, input_batch, aux_hidden_states, num_sampled, num_rejected, last_sampled,
                            next_prefill_tokens, temperature, seeds) -> None:
            num_target_tokens = input_batch.num_tokens
            # chunks: a prefill chunk's concatenated target features (8192 x 5 x 5120) would add ~0.5 GB of
            # temporaries next to the MTP drafter's, beyond what the memory profiler measured
            for a in range(0, num_target_tokens, CTX_CHUNK):
                b = min(a + CTX_CHUNK, num_target_tokens)
                self.hidden_states[a:b].copy_(
                    self.model.combine_hidden_states(torch.cat([h[a:b] for h in aux_hidden_states], dim=-1)))
            self.prepare_context_anchor(input_batch, num_rejected)
            for i, gid in enumerate(self.draft_kv_cache_group_ids):
                prepare_dflash_inputs(
                    self.input_buffers, self.block_tables.slot_mappings[gid], self.context_positions,
                    self._context_slot_mappings[i], self.sample_indices, self.sample_pos, self.sample_idx_mapping,
                    self.temperature, self.seeds, input_batch, num_sampled, num_rejected, last_sampled,
                    next_prefill_tokens, temperature, seeds, self.block_tables.input_block_tables[gid],
                    self.block_tables.kernel_block_sizes[gid], self.block_tables.cp_rank, self.dcp_size,
                    self.block_tables.cp_interleave, self.parallel_drafting_token_id, self.num_query_per_req,
                    self.num_speculative_steps, self.max_num_reqs, self.max_num_tokens, self.max_model_len,
                    self.sample_from_anchor)
            for a in range(0, num_target_tokens, CTX_CHUNK):
                self._precompute_context_kv(a, min(a + CTX_CHUNK, num_target_tokens))

        @torch.inference_mode()
        def propose(self, input_batch, attn_metadata, slot_mappings, last_hidden_states, aux_hidden_states,
                    num_sampled, num_rejected, last_sampled, next_prefill_tokens, temperature, seeds, dp_sync=None,
                    dummy_run=False, skip_attn_for_dummy_run=False, mm_inputs=None, is_profile=False):
            mtp_args = (input_batch, attn_metadata, slot_mappings, last_hidden_states, None, num_sampled,
                        num_rejected, last_sampled, next_prefill_tokens, temperature, seeds)
            kw = dict(dp_sync=dp_sync, dummy_run=dummy_run, skip_attn_for_dummy_run=skip_attn_for_dummy_run,
                      is_profile=is_profile)
            dfl_args = (input_batch, attn_metadata, slot_mappings, last_hidden_states, aux_hidden_states,
                        num_sampled, num_rejected, last_sampled, next_prefill_tokens, temperature, seeds)
            if dummy_run or self.ctl is None:   # profiling / warmup / capture: exercise both drafters
                self.mtp.propose(*mtp_args, **kw)
                return super().propose(*dfl_args, **kw)
            num_reqs = input_batch.num_reqs
            ctl = self.ctl
            ctl.mark(0)
            mask = np.asarray(input_batch.num_scheduled_tokens[:num_reqs]) == self.verify_len
            ctl.record(num_sampled[:num_reqs], last_sampled.reshape(-1)[input_batch.idx_mapping[:num_reqs]], mask)
            choice = ctl.choose(bool(mask.all()))
            if choice == DFL:
                self.mtp.context_only(*mtp_args, **kw)
                ctl.mark(1)
                out = super().propose(*dfl_args, **kw)
                ctl.mark(2)
                return out
            self._dflash_context(input_batch, aux_hidden_states, num_sampled, num_rejected, last_sampled,
                                 next_prefill_tokens, temperature, seeds)
            ctl.mark(1)
            d = self.mtp.propose(*mtp_args, **kw)
            out = torch.empty(num_reqs, self.num_speculative_steps, dtype=d.dtype, device=d.device)
            out[:, :N_MTP] = d[:num_reqs]
            out[:, N_MTP:] = d[:num_reqs, -1:]
            ctl.mark(2)
            return out

    return ComboSpeculator


def install():
    if not ENABLED:
        return
    import sys

    from vllm.v1.worker.gpu import spec_decode
    if getattr(spec_decode, "_vllm_turing_combo", False):
        return
    orig = spec_decode.init_speculator

    def init_speculator(vllm_config, device):
        sc = vllm_config.speculative_config
        if (sc is not None and sc.method == "dflash"
                and "DFlash2DraftModel" in sc.draft_model_config.architectures):
            return _make_classes()(vllm_config, device)
        return orig(vllm_config, device)

    spec_decode.init_speculator = init_speculator
    spec_decode._vllm_turing_combo = True
    mr = sys.modules.get("vllm.v1.worker.gpu.model_runner")
    if mr is not None:
        mr.init_speculator = init_speculator
    logger.info("vllm_turing: MTP + DFlash2 combined speculator enabled (MTP %d drafts)", N_MTP)
