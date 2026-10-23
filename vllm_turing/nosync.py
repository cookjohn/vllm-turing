"""Sync-free MTP draft steps (vLLM 0.31, model runner v2, FlashInfer). With speculative decoding the FlashInfer metadata
builder reads exact sequence lengths with seq_lens.cpu() (a device sync) for every build, and the autoregressive
drafter builds metadata before each draft step: every draft step drained the GPU, which then idled while the CPU
planned and launched the next step (MTP4: 3 drains per decode step, adaptive MTP7: 6; ~0.6 ms each, about half of a
draft step's wall time). The draft step lengths are deterministic: step i sees target_seq_len - num_rejected + i
(clamped to max_model_len, as _prepare_decode_inputs_kernel / _update_draft_inputs_kernel compute them on the GPU). At
the start of propose() that base is copied to pinned host memory without blocking; before the first draft step the CPU
waits for that copy only (the GPU meanwhile runs the draft prefill), and every draft step's metadata gets the exact
lengths with patches/flashinfer_spec_decode.py's _TURING_EXACT_SEQ_LENS set, so FlashInfer plans without a sync.
VLLM_TURING_NOSYNC=0 disables; VLLM_TURING_NOSYNC_CHECK=1 also syncs and asserts equality (testing)."""
import logging
import os

import torch

logger = logging.getLogger("vllm.turing")
ENABLED = os.environ.get("VLLM_TURING_NOSYNC", "1") != "0"
CHECK = os.environ.get("VLLM_TURING_NOSYNC_CHECK", "0") == "1"
# draft steps 2.. re-plan the step-1 FlashInfer decode plan with lengths + (step - 1) (flashinfer_spec_decode.py
# _turing_replan_decode) instead of rebuilding all attention metadata: ~1.3 ms less Python per draft step
FAST = os.environ.get("VLLM_TURING_DRAFT_FASTPLAN", "1") != "0"


# --- asynchronous mode: no wait for the GPU in a decode step --------------------------------------------------------
# Even sync-free, a decode step waited once for the GPU: propose() needs the exact draft base before planning the draft
# steps, i.e. the end of the target verify forward (~28 of ~34 ms), and the next step's verify build read seq_lens.cpu().
# Everything the CPU did after that wait (draft planning and launches, sampling bookkeeping, the next step's inputs and
# attention metadata, ~7 ms) ran while the GPU idled. FlashInfer's FA2 planner sees KV lengths only as page counts
# (PrefillPlanImpl takes the page indptr; splits are whole pages; vLLM's FlashInfer page is 16 tokens), and its paged
# kernel takes each request's length from the device indptr and last-page lengths ((pages - 1) * page + last, signed)
# and clamps split starts to that length (a split past it is empty: output 0, lse -inf, weight 0 in the merge). So a
# plan made from the CPU's upper bound computes the exact attention once the GPU writes last = exact - (planned pages
# - 1) * page (<= 0 when the exact length ends a page earlier) after the plan's copy. With VLLM_TURING_ASYNC_SPEC=1:
#  - draft steps: base = the verify step's CPU upper bound (the exact base is at most 2 x num_speculative_tokens below:
#    the last step's and this step's rejections); planned and launched without waiting.
#  - verify steps of continuing decode-only batches: planned from the CPU upper bound (at most num_speculative_tokens
#    above: the previous step's rejections), no seq_lens.cpu().
#  Where the bound adds a page, FlashInfer may split the KV differently than for the exact length: the same attention
#  summed in another order (greedy texts can differ in late tokens, as with any change of batch composition).
#  - each FlashInfer plan stages its info in a ring of pinned buffers (the single buffer forced a wait for the previous
#    plan's copy, queued behind the target forward once the CPU runs ahead).
#  - the device fix-up counts rows whose exact length exceeds the planned pages (a violated upper bound; the length is
#    clamped to the plan, never an out-of-bounds read), read back without blocking: any is logged as an error and turns
#    the mode off (drafts only lose acceptance, a verify step would be wrong).
#  - armed after Worker.compile_or_warm_up_model: vLLM's kernel warm-up runs a synthetic decode step whose CPU and
#    device lengths are unrelated.
ASYNC = os.environ.get("VLLM_TURING_ASYNC_SPEC", "0") == "1"
DEBUG = os.environ.get("VLLM_TURING_ASYNC_DEBUG", "0") == "1"
RING = 4
_astats = {"draft": 0, "draft_wait": 0, "verify": 0, "verify_sync": 0}
_viol = {}            # device -> int64 counter tensor of rows whose exact length the plan cannot hold
_viol_host = {}       # device -> (pinned int64 [1], event) of the last non-blocking readback
_disabled = [True]     # until the worker's warm-up has finished (its synthetic steps break the length bounds)


def _ring_plan(wrapper, plan):
    """Run plan() with the wrapper's pinned staging buffer taken from a ring (waits only if the slot's copy from RING
    plans ago has not run yet)."""
    r = wrapper.__dict__.get("_turing_ring")
    if r is None:
        ref = wrapper._pin_memory_int_workspace_buffer
        r = wrapper._turing_ring = {"bufs": [ref] + [torch.empty(ref.shape, dtype=ref.dtype, pin_memory=True)
                                                     for _ in range(RING - 1)], "evs": [None] * RING, "i": 0}
    i = r["i"]
    r["i"] = (i + 1) % RING
    ev = r["evs"][i]
    if ev is not None and not ev.query():
        ev.synchronize()
    wrapper._pin_memory_int_workspace_buffer = r["bufs"][i]
    out = plan()
    if ev is None:
        ev = r["evs"][i] = torch.cuda.Event()
    ev.record()
    return out


def _pages(x, page):
    return (x + (page - 1)) // page


def _fix_last_page(wrapper, exact, n, page):
    """Make the wrapper's planned pages describe the exact device lengths `exact` (int [n]): last-page lengths [:n] :=
    exact - (planned pages - 1) * page (<= 0 when the exact length ends before the last planned page); counts rows the
    plan cannot hold (device counter, no sync) and clamps them to the plan."""
    buf = getattr(wrapper, "_paged_kv_last_page_len_buf", None)
    ind = getattr(wrapper, "_paged_kv_indptr_buf", None)
    if buf is None or ind is None or not buf.is_cuda or not ind.is_cuda:
        return False
    ex = exact.to(torch.int32)
    planned = ind[1:n + 1] - ind[:n]
    bad = (ex > planned * page) | (ex < 1)
    buf[:n].copy_(torch.minimum(ex, planned * page) - (planned - 1) * page)
    v = _viol.get(buf.device)
    if v is None:
        v = _viol[buf.device] = torch.zeros(1, dtype=torch.int64, device=buf.device)
    v += bad.sum()
    if CHECK:
        torch.cuda.synchronize()
        want = exact.cpu().to(torch.int64)
        pl = planned.cpu().to(torch.int64)
        got = buf[:n].cpu().to(torch.int64)
        assert torch.equal((pl - 1) * page + got, want) and bool((want <= pl * page).all()), ("length", pl, got, want)
    return True


def _check_viol():
    """Non-blocking readback of the mismatch counters; logs and disables the mode on any mismatch."""
    for dev, v in _viol.items():
        h = _viol_host.get(dev)
        if h is not None and h[1].query() and h[0][0].item() > 0 and not _disabled[0]:
            _disabled[0] = True
            logger.error("vllm_turing async spec: %d rows longer than their plan (violated upper bound); "
                         "asynchronous mode turned off", h[0][0].item())
        if h is None:
            h = _viol_host[dev] = (torch.zeros(1, dtype=torch.int64, pin_memory=True), torch.cuda.Event())
        if h[1].query():
            h[0].copy_(v, non_blocking=True)
            h[1].record()


def _log_stats():
    s = sum(_astats.values())
    if s % 4000 == 3999:
        _check_viol()
        logger.info("vllm_turing async spec: draft loops without wait %d (waited %d), verify builds without sync %d "
                    "(synced %d)", _astats["draft"], _astats["draft_wait"], _astats["verify"], _astats["verify_sync"])
    elif s % 500 == 499:
        _check_viol()


def _draft_loop(self, num_reqs, batch_desc, num_tokens_across_dp, base, fix=None, page=None):
    """AutoRegressiveSpeculator._multi_step_decode (non-fused, positions advance), with the full metadata build only
    for the first draft step (and after a page-boundary crossing). fix: exact draft base on the device (asynchronous
    mode: `base` is an upper bound, the wrappers' last-page lengths are fixed on the GPU after each plan)."""
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer

    positions = self.input_buffers.positions[:num_reqs]
    query_start_loc = self.input_buffers.query_start_loc[: num_reqs + 1]
    idx_mapping = self.idx_mapping[:num_reqs]
    full = batch_desc.cg_mode == CUDAGraphMode.FULL
    builders = [g.get_metadata_builder() for groups in self.attn_groups for g in groups]
    can_fast = bool(builders) and all(hasattr(b, "_turing_replan_decode") for b in builders)
    attn_metadata = None
    slot_mappings_by_layer = None
    built_at = 0   # draft step of the last full build: re-plans are relative to its lengths
    ev = getattr(self, "_turing_plan_ev", None)
    if ev is None:
        ev = self._turing_plan_ev = torch.cuda.Event()
    for step in range(1, self.num_speculative_steps):
        slot_mappings = self.block_tables.compute_slot_mappings(idx_mapping, query_start_loc, positions,
                                                                batch_desc.num_tokens)
        if not full or attn_metadata is None:
            slot_mappings_by_layer = build_slot_mappings_by_layer(slot_mappings, self.kv_cache_config)
        if step > 1 and fix is None:
            # a plan stages its info in the wrapper's pinned host buffer and copies it asynchronously: the previous
            # step's copy must have run before this plan overwrites the buffer (without the per-step sync the CPU can
            # be ahead of the GPU here); waits for that copy only, not for the previous step's graph (asynchronous
            # mode: staging ring, no wait)
            ev.synchronize()
        if not (attn_metadata is not None and can_fast
                and all(b._turing_replan_decode(step - built_at) for b in builders)):
            attn_metadata = self._build_uniform_attn_metadata(num_reqs=num_reqs, batch_desc=batch_desc,
                                                              num_query_per_req=1, seq_lens_cpu_upper_bound=base,
                                                              step=step)
            built_at = step
        if fix is not None:
            exact = torch.clamp(fix + step, max=self.max_model_len)
            if CHECK:   # the drafter's own lengths for this step (prepare_decode_inputs / the previous draft graph)
                torch.cuda.synchronize()
                assert torch.equal(self.input_buffers.seq_lens[:num_reqs].cpu().to(torch.int64),
                                   exact.cpu().to(torch.int64)), ("draft length", step,
                                                                  self.input_buffers.seq_lens[:num_reqs].cpu(),
                                                                  exact.cpu())
            for b in builders:
                d = getattr(b, "_turing_dec", None)
                if d is not None:
                    _fix_last_page(d[0], exact, num_reqs, page)
        ev.record()
        self.current_draft_step.fill_(step)
        if full:
            self.decode_cudagraph_manager.run_fullgraph(batch_desc)
        else:
            self._generate_draft(num_reqs, batch_desc.num_tokens, attn_metadata, slot_mappings_by_layer,
                                 num_tokens_across_dp=num_tokens_across_dp, cudagraph_runtime_mode=batch_desc.cg_mode)


# --- target verify build without the sync -------------------------------------------------------------------------
# The verify step's FlashInfer build synced too (seq_lens.cpu()), at the start of every step, after which the CPU still
# planned and launched while the GPU idled. A continuing request's exact verify length is its post-acceptance length
# from the previous step (the base copied in propose) + this step's scheduled tokens (the GPU computes seq_lens as
# num_computed + query length, the query lengths from these same scheduled counts). Guard: the runner's optimistic
# upper bound minus the exact value must be within [0, num_speculative_tokens] (the last step's rejections) for every
# row, else (new / resumed / stale requests) the normal synced build runs.
_prev = {}          # req_id -> exact num_computed after the request's last step
_pending = None     # (req_ids, pinned base buffer, event, n) of the last propose
_exact_batch = None   # id() of the InputBatch whose seq_lens_cpu_upper_bound was made exact
_stats = [0, 0]       # exact target builds, fallbacks
TARGET = os.environ.get("VLLM_TURING_NOSYNC_TARGET", "0") == "1"   # measured: no gain (CPU prep is the long pole), off


def _install_target(fi):
    import numpy as np
    from vllm.v1.worker.gpu import model_runner as mr
    R = mr.GPUModelRunner
    if getattr(R, "_vllm_turing_nosync", False):
        return
    orig_prepare = R.prepare_inputs

    def prepare_inputs(self, scheduler_output, *args, **kwargs):
        global _pending, _exact_batch
        _exact_batch = None
        if _pending is not None:
            ids, buf, ev, n = _pending
            ev.synchronize()
            _prev.update(zip(ids, buf[:n].tolist()))
            _pending = None
        for r in scheduler_output.scheduled_new_reqs:
            _prev.pop(r.req_id, None)
        for rid in scheduler_output.scheduled_cached_reqs.resumed_req_ids:
            _prev.pop(rid, None)
        for rid in scheduler_output.finished_req_ids:
            _prev.pop(rid, None)
        for rid in (scheduler_output.preempted_req_ids or ()):
            _prev.pop(rid, None)
        ib = orig_prepare(self, scheduler_output, *args, **kwargs)
        n = ib.num_reqs
        nspec = self.num_speculative_steps
        if n == 0 or nspec == 0 or self.pcp_manager is not None or self.dcp_size > 1:
            return ib
        rids = ib.req_ids[:n]
        if (_stats[0] + _stats[1]) % 2000 == 1999:
            logger.info("vllm_turing nosync: target verify builds without sync %d, with sync %d", *_stats)
        if not all(r in _prev for r in rids):
            _stats[1] += 1
            return ib
        exact = np.fromiter((_prev[r] for r in rids), dtype=np.int64, count=n) + ib.num_scheduled_tokens[:n]
        ub = ib.seq_lens_cpu_upper_bound
        gap = ub[:n].numpy().astype(np.int64) - exact
        if gap.min() < 0 or gap.max() > nspec:
            _stats[1] += 1
            return ib
        if CHECK:
            want = ib.seq_lens[:n].cpu().numpy()
            assert np.array_equal(want, exact), (want, exact, rids)
        ub[:n] = torch.from_numpy(exact.astype(np.int32))
        _exact_batch = id(ib)
        _stats[0] += 1
        ms = self.model_state
        if not getattr(ms, "_turing_nosync", False):
            orig_pa = ms.prepare_attn

            def prepare_attn(input_batch, *a, **k):
                if id(input_batch) != _exact_batch:
                    return orig_pa(input_batch, *a, **k)
                fi._TURING_EXACT_SEQ_LENS = True
                try:
                    return orig_pa(input_batch, *a, **k)
                finally:
                    fi._TURING_EXACT_SEQ_LENS = False

            ms.prepare_attn = prepare_attn
            ms._turing_nosync = True
        return ib

    R.prepare_inputs, R._vllm_turing_nosync = prepare_inputs, True


def _page_size(obj):
    """The FlashInfer page size of the attention groups of a speculator / model runner (None if not uniform)."""
    p = obj.__dict__.get("_turing_page")
    if p is None:
        sizes = set()
        for groups in getattr(obj, "attn_groups", None) or []:
            for g in groups:
                b = g.get_metadata_builder()
                if hasattr(b, "_turing_plan_spec_prefill"):      # FlashInfer builders only (not the GDN ones)
                    sizes.add(getattr(b, "page_size", None))
        p = obj._turing_page = sizes.pop() if len(sizes) == 1 and None not in sizes else 0
    return p or None


_seen = set()           # requests with a finished scheduled step since they were (re)scheduled
_async_batch = [None]   # (id(InputBatch), num_reqs, page size) of a verify batch built without the sync


def _install_async(fi):
    """Asynchronous verify builds + staging rings (see the asynchronous mode notes above)."""
    import numpy as np
    from vllm.v1.worker import gpu_worker
    from vllm.v1.worker.gpu import model_runner as mr

    Wk = gpu_worker.Worker
    orig_warm = Wk.compile_or_warm_up_model

    def compile_or_warm_up_model(self, *a, **k):
        out = orig_warm(self, *a, **k)
        torch.cuda.synchronize()
        for v in _viol.values():
            v.zero_()
        _viol_host.clear()
        _seen.clear()
        _disabled[0] = False
        logger.info("vllm_turing async spec: armed after the warm-up")
        return out

    Wk.compile_or_warm_up_model = compile_or_warm_up_model

    # staging rings for every decode plan (draft steps) and every speculative-verify plan
    orig_fpd = fi.fast_plan_decode

    def fast_plan_decode(wrapper, *a, **k):
        return _ring_plan(wrapper, lambda: orig_fpd(wrapper, *a, **k))

    fi.fast_plan_decode = fast_plan_decode
    B = fi.FlashInferMetadataBuilder
    orig_tpsp = B._turing_plan_spec_prefill

    def _turing_plan_spec_prefill(self, num_reqs, k, causal, paged_kv_indices):
        w = self.__dict__.get("_turing_spec_pws", {}).get((num_reqs, k, causal))
        if w is None:
            return orig_tpsp(self, num_reqs, k, causal, paged_kv_indices)
        return _ring_plan(w, lambda: orig_tpsp(self, num_reqs, k, causal, paged_kv_indices))

    B._turing_plan_spec_prefill = _turing_plan_spec_prefill

    R = mr.GPUModelRunner
    orig_prepare = R.prepare_inputs

    def prepare_inputs(self, scheduler_output, *args, **kwargs):
        _async_batch[0] = None
        for r in scheduler_output.scheduled_new_reqs:
            _seen.discard(r.req_id)
        for rid in scheduler_output.scheduled_cached_reqs.resumed_req_ids:
            _seen.discard(rid)
        for rid in scheduler_output.finished_req_ids:
            _seen.discard(rid)
        for rid in (scheduler_output.preempted_req_ids or ()):
            _seen.discard(rid)
        ib = orig_prepare(self, scheduler_output, *args, **kwargs)
        n = ib.num_reqs
        nspec = self.num_speculative_steps
        if n == 0 or nspec == 0:
            return ib
        rids = ib.req_ids[:n]
        ok = (not _disabled[0] and self.pcp_manager is None and self.dcp_size <= 1
              and all(r in _seen for r in rids)
              and bool(np.all(np.asarray(ib.num_scheduled_tokens[:n]) == nspec + 1)))
        _seen.update(rids)
        page = _page_size(self) if ok else None
        if ok and page:
            ok = int(ib.seq_lens_cpu_upper_bound[:n].min()) > nspec
        if ok and page:
            _async_batch[0] = (id(ib), n, page)
        else:
            _astats["verify_sync"] += 1
            if DEBUG and _astats["verify_sync"] <= 20:
                logger.info("vllm_turing async spec debug: verify sync: disabled %s page %s seen %s sched %s ub %s",
                            _disabled[0], page, [r in _seen for r in rids][:6],
                            list(np.asarray(ib.num_scheduled_tokens[:n]))[:6],
                            ib.seq_lens_cpu_upper_bound[:n].tolist()[:6])
        ms = self.model_state
        if not getattr(ms, "_turing_async", False):
            orig_pa = ms.prepare_attn

            def prepare_attn(input_batch, *a, **k):
                st = _async_batch[0]
                if st is None or st[0] != id(input_batch):
                    return orig_pa(input_batch, *a, **k)
                fi._TURING_EXACT_SEQ_LENS = True
                try:
                    md = orig_pa(input_batch, *a, **k)
                finally:
                    fi._TURING_EXACT_SEQ_LENS = False
                done = set()
                for m in (md or {}).values():
                    dec = getattr(m, "decode", None)
                    w = getattr(dec, "wrapper", None) if dec is not None else None
                    if w is not None and id(w) not in done:
                        done.add(id(w))
                        _fix_last_page(w, input_batch.seq_lens[:st[1]], st[1], st[2])
                _astats["verify"] += 1
                _log_stats()
                return md

            ms.prepare_attn = prepare_attn
            ms._turing_async = True
        return ib

    R.prepare_inputs, R._vllm_turing_async = prepare_inputs, True


def install():
    if not ENABLED:
        return
    try:
        import vllm.v1.attention.backends.flashinfer as fi
        from vllm.v1.worker.gpu.spec_decode.autoregressive import speculator as ar
    except Exception:   # noqa: BLE001
        return
    if not hasattr(fi, "_TURING_EXACT_SEQ_LENS"):
        logger.warning("vllm_turing: nosync needs patches/flashinfer_spec_decode.py (_TURING_EXACT_SEQ_LENS); off")
        return
    S = ar.AutoRegressiveSpeculator
    if getattr(S, "_vllm_turing_nosync", False):
        return
    orig_propose, orig_msd = S.propose, S._multi_step_decode

    def propose(self, input_batch, attn_metadata, slot_mappings, last_hidden_states, aux_hidden_states, num_sampled,
                num_rejected, *args, **kwargs):
        self._turing_base = None
        if not kwargs.get("dummy_run", False) and self.advance_draft_positions:
            n = input_batch.num_reqs
            buf = getattr(self, "_turing_base_buf", None)
            if buf is None:
                buf = self._turing_base_buf = torch.zeros(self.max_num_reqs, dtype=torch.int32, pin_memory=True)
                self._turing_base_ev = torch.cuda.Event()
            base_dev = (input_batch.seq_lens[:n] - num_rejected[:n]).to(torch.int32)
            buf[:n].copy_(base_dev, non_blocking=True)
            self._turing_base_ev.record()
            self._turing_base = n
            self._turing_base_dev = base_dev
            if TARGET:
                global _pending
                _pending = (list(input_batch.req_ids[:n]), buf, self._turing_base_ev, n)
        return orig_propose(self, input_batch, attn_metadata, slot_mappings, last_hidden_states, aux_hidden_states,
                            num_sampled, num_rejected, *args, **kwargs)

    def _multi_step_decode(self, num_reqs, skip_attn, batch_desc, num_tokens_across_dp, seq_lens_cpu_upper_bound):
        n = getattr(self, "_turing_base", None)
        if n is None or n != num_reqs or skip_attn:
            return orig_msd(self, num_reqs, skip_attn, batch_desc, num_tokens_across_dp, seq_lens_cpu_upper_bound)
        if ASYNC and FAST and not _disabled[0] and seq_lens_cpu_upper_bound is not None:
            page = _page_size(self)
            ub = seq_lens_cpu_upper_bound[:num_reqs].to(torch.int64)
            nsteps = self.num_speculative_steps
            gap = 2 * nsteps        # last step's and this step's rejections
            ok = page is not None and int(ub.min()) > gap and int(ub.max()) + nsteps <= self.max_model_len
            if ok:
                _astats["draft"] += 1
                _log_stats()
                fi._TURING_EXACT_SEQ_LENS = True
                try:
                    return _draft_loop(self, num_reqs, batch_desc, num_tokens_across_dp,
                                       ub.to(torch.int32), fix=self._turing_base_dev, page=page)
                finally:
                    fi._TURING_EXACT_SEQ_LENS = False
            _astats["draft_wait"] += 1
            if DEBUG and _astats["draft_wait"] <= 20:
                logger.info("vllm_turing async spec debug: draft wait: page %s ub %s nsteps %d max_len %d",
                            page, ub.tolist()[:8], nsteps, self.max_model_len)
        self._turing_base_ev.synchronize()   # the copy only, not the draft prefill queued after it
        base = self._turing_base_buf[:num_reqs].clone()
        if CHECK:
            torch.cuda.synchronize()
            want = self.input_buffers.seq_lens[:num_reqs].cpu() - 1   # step 1 seq_lens minus the step
            assert torch.equal(base.clamp(max=self.max_model_len - 1), want.clamp(max=self.max_model_len - 1)), (
                base, want)
        fi._TURING_EXACT_SEQ_LENS = True
        try:
            if not FAST:
                # _build_attn_metadata adds the step (1, 2, ...) to these and clamps at max_model_len
                return orig_msd(self, num_reqs, skip_attn, batch_desc, num_tokens_across_dp, base)
            return _draft_loop(self, num_reqs, batch_desc, num_tokens_across_dp, base)
        finally:
            fi._TURING_EXACT_SEQ_LENS = False

    S.propose, S._multi_step_decode, S._vllm_turing_nosync = propose, _multi_step_decode, True
    if ASYNC:
        _install_async(fi)
    elif TARGET:
        _install_target(fi)
    logger.info("vllm_turing: sync-free MTP draft steps%s%s", " + target verify build" if TARGET and not ASYNC else
                " + asynchronous decode steps (no GPU wait)" if ASYNC else "", " (checked)" if CHECK else "")
