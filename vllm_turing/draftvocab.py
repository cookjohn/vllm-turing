"""Pruned draft vocabulary for greedy speculative drafting (MTP / EAGLE): the draft model's LM head is evaluated only on
the N most frequent tokens (a ranked id list, one id per whitespace-separated entry, plus every added / special token)
instead of the 248k-row head, each draft step. Every TP rank runs the int8 head GEMV (lmhead.py) over the kept rows of
its vocabulary shard (row-index subset: no weight copy), takes a local argmax, and the ranks exchange (value, id).
Only affects which token is drafted; the target model verifies with its full head, so output stays exact.
Needs the int8 LM head. env VLLM_TURING_DRAFT_VOCAB (ranked id file, default ~/models/draft_vocab_ranked.txt if present,
"0" disables), VLLM_TURING_DRAFT_VOCAB_N (default 65536), VLLM_TURING_DRAFT_Q4 (default 1: draft with a balanced
4-bit copy of the kept rows, _Q4Head; 0: the int8 row subset of each rank's own shard)."""
import ctypes
import logging
import os

import torch

from . import lmhead

logger = logging.getLogger("vllm.turing")   # under "vllm": uses vLLM's log handler
DEFAULT_FILE = os.path.expanduser("~/models/draft_vocab_ranked.txt")
Q4 = os.environ.get("VLLM_TURING_DRAFT_Q4", "1") != "0"   # balanced 4-bit copy of the pruned head (_Q4Head)
DFLASH_Q4 = os.environ.get("VLLM_TURING_DFLASH_Q4", "1") != "0"   # DFlash2 candidates from the 4-bit pruned head


class _PrunedHead:
    def __init__(self, lm_head, ids: torch.Tensor):
        si = lm_head.shard_indices
        start, end = si.org_vocab_start_index, si.org_vocab_end_index
        mine = ids[(ids >= start) & (ids < end)].sort().values
        dev = lm_head.turing_w8.device
        self.rows = (mine - start).to(torch.int32).to(dev)          # rows of this rank's head shard
        self.gids = mine.to(torch.int64).to(dev)                     # their global token ids
        self.w8, self.s = lm_head.turing_w8, lm_head.turing_s
        self.tp = lm_head.tp_size
        self.q4 = None

    def sample(self, h: torch.Tensor) -> torch.Tensor:
        if self.q4 is not None:
            ids = self.q4.sample(h)
            if ids is not None:
                return ids
        x = h.reshape(-1, h.shape[-1]).contiguous()
        M, K, N = x.shape[0], x.shape[1], self.rows.shape[0]
        out = torch.empty(M, N, dtype=x.dtype, device=x.device)
        if M <= 16:
            rc = lmhead.lib().w8a16_gemv_rows(x.data_ptr(), self.w8.data_ptr(), self.s.data_ptr(), None, out.data_ptr(),
                                              M, N, K, 0 if x.dtype == torch.float16 else 1, self.rows.data_ptr(),
                                              torch.cuda.current_stream().cuda_stream)
            if rc != 0:
                raise RuntimeError(f"w8a16_gemv_rows failed: {rc}")
        else:
            rows = self.rows.long()
            out = x @ (self.w8[rows].float() * self.s[rows, None]).to(x.dtype).t()
        val, arg = out.float().max(dim=-1)
        ids = self.gids[arg]
        if self.tp > 1:
            from vllm.distributed import tensor_model_parallel_all_gather
            vals = tensor_model_parallel_all_gather(val.unsqueeze(-1), dim=-1)                    # [M, tp]
            idss = tensor_model_parallel_all_gather(ids.to(torch.float64).unsqueeze(-1), dim=-1)   # exact for < 2^53
            ids = idss.gather(-1, vals.argmax(dim=-1, keepdim=True)).squeeze(-1).to(torch.int64)
        return ids


_qlib = None


def qlib():
    global _qlib
    if _qlib is None:
        from . import _build
        _qlib = _build.load("dhead_q4")
        _qlib.dhead_blocks.argtypes = [ctypes.c_int, ctypes.c_int]
        _qlib.dhead_blocks.restype = ctypes.c_int
        _qlib.dhead_argmax.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int] * 5 + [ctypes.c_void_p]
        _qlib.dhead_argmax.restype = ctypes.c_int
        _qlib.dhead_logits.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
        _qlib.dhead_logits.restype = ctypes.c_int
    return _qlib


def q4_pack(w: torch.Tensor):
    """fp32 [N, K] -> (packed uint8 [N, K / 2], fp16 scales [N, K / 32]) in the dhead_q4.cu layout (Q4_0 blocks of 32:
    d = signed max / -8, q = round(w / d) + 8 clamped to 0..15)."""
    N, K = w.shape
    g = w.view(N, K // 32, 32)
    idx = g.abs().argmax(dim=-1, keepdim=True)
    d = (g.gather(-1, idx) / -8.0).squeeze(-1)                                     # [N, K/32]
    dh = d.half()
    inv = torch.where(dh.float() != 0, 1.0 / dh.float(), torch.zeros_like(d))
    q = (g * inv.unsqueeze(-1) + 8.0).round().clamp(0, 15).to(torch.int32)        # [N, K/32, 32]
    q = q.view(N, K // 32, 4, 4, 2)                                                 # word j, pair i, (even, odd)
    sh = (4 * torch.arange(4, device=w.device, dtype=torch.int32)).view(1, 1, 1, 4)
    words = ((q[..., 0] << sh) | (q[..., 1] << (sh + 16))).sum(dim=-1, dtype=torch.int32)   # [N, K/32, 4]
    return words.contiguous().view(torch.uint8).view(N, K // 2), dh.contiguous()


class _Q4Head:
    """Balanced 4-bit copy of the pruned draft head: the kept ids are split evenly over the TP ranks (rank r takes the
    r-th slice of the sorted id list, rows fetched from whichever rank holds them), so every rank reads ~1/tp of
    N * K / 2 bytes per draft token instead of its own (uneven) share of the int8 rows. Fused argmax per rank, one
    host all-reduce of the [M, tp, 4] fp16 candidate slots, then the best candidate."""

    def __init__(self, lm_head, ids: torch.Tensor):
        import torch.distributed as dist
        from vllm.distributed import get_tp_group
        si = lm_head.shard_indices
        dev = lm_head.turing_w8.device
        w8, s8 = lm_head.turing_w8, lm_head.turing_s
        K = w8.shape[1]
        tp, rank = lm_head.tp_size, lm_head.tp_rank
        n = len(ids)
        bounds = [(r * n) // tp for r in range(tp + 1)]
        mine = ids[bounds[rank]:bounds[rank + 1]]
        my_lo, my_hi = si.org_vocab_start_index, si.org_vocab_end_index
        wf =torch.empty(len(mine), K, dtype=torch.float16, device=dev)
        if tp == 1:
            loc = (mine - my_lo).to(dev)
            for a in range(0, len(mine), 4096):
                r = loc[a:a + 4096]
                wf[a:a + 4096] = (w8[r].float() * s8[r, None]).half()
        else:
            g = get_tp_group()
            shards = [None] * tp
            dist.all_gather_object(shards, (my_lo, my_hi), group=g.cpu_group)
            for src in range(tp):
                lo, hi = shards[src]
                # rows of src's shard wanted by the other ranks, in id order (every rank computes the same list)
                want = torch.cat([ids[bounds[t]:bounds[t + 1]] for t in range(tp) if t != src])
                want = want[(want >= lo) & (want < hi)].sort().values
                if src == rank:   # own rows for own slice: no transfer
                    own = mine[(mine >= lo) & (mine < hi)]
                    pos = torch.searchsorted(mine, own).to(dev)
                    loc = (own - lo).to(dev)
                    for a in range(0, len(own), 4096):
                        r = loc[a:a + 4096]
                        wf[pos[a:a + 4096]] = (w8[r].float() * s8[r, None]).half()
                if len(want) == 0:
                    continue
                buf = torch.empty(len(want), K, dtype=torch.int8, device=dev)
                sbuf = torch.empty(len(want), dtype=torch.float32, device=dev)
                if src == rank:
                    loc = (want - lo).to(dev)
                    buf.copy_(w8[loc])
                    sbuf.copy_(s8[loc])
                dist.broadcast(buf, src=g.ranks[src], group=g.device_group)
                dist.broadcast(sbuf, src=g.ranks[src], group=g.device_group)
                if src != rank:
                    sel = (mine >= lo) & (mine < hi)
                    take = torch.searchsorted(want, mine[sel]).to(dev)
                    pos = sel.nonzero().squeeze(-1).to(dev)
                    for a in range(0, len(take), 4096):
                        t = take[a:a + 4096]
                        wf[pos[a:a + 4096]] = (buf[t].float() * sbuf[t, None]).half()
                del buf, sbuf
        self.wq = torch.empty(len(mine), K // 2, dtype=torch.uint8, device=dev)
        self.d = torch.empty(len(mine), K // 32, dtype=torch.float16, device=dev)
        for a in range(0, len(mine), 4096):
            self.wq[a:a + 4096], self.d[a:a + 4096] = q4_pack(wf[a:a + 4096].float())
        del wf
        torch.cuda.empty_cache()
        self.gids = mine.to(torch.int64).to(dev)
        self.N, self.K, self.tp, self.rank = len(mine), K, tp, rank
        nb = max(qlib().dhead_blocks(m, self.N) for m in (1, 3))
        self.pval = torch.empty(8 * nb, dtype=torch.float32, device=dev)
        self.pidx = torch.empty(8 * nb, dtype=torch.int32, device=dev)

    def sample(self, h: torch.Tensor):
        x = h.reshape(-1, h.shape[-1])
        M = x.shape[0]
        if M > 8 or x.dtype != torch.float16:
            return None
        x = x.contiguous()
        ex = torch.empty(M, self.tp, 4, dtype=torch.float16, device=x.device)
        rc = qlib().dhead_argmax(x.data_ptr(), self.wq.data_ptr(), self.d.data_ptr(), self.gids.data_ptr(),
                                 self.pval.data_ptr(), self.pidx.data_ptr(), ex.data_ptr(), M, self.N, self.K,
                                 self.rank, self.tp, torch.cuda.current_stream().cuda_stream)
        if rc != 0:
            raise RuntimeError(f"dhead_argmax failed: {rc}")
        if self.tp > 1:
            from vllm.distributed import tensor_model_parallel_all_reduce
            ex = tensor_model_parallel_all_reduce(ex)
        best = ex[..., 0].float().argmax(dim=-1, keepdim=True)                          # [M, 1]
        hi = ex[..., 1].gather(-1, best).squeeze(-1).to(torch.int64)
        lo = ex[..., 2].gather(-1, best).squeeze(-1).to(torch.int64)
        return hi * 2048 + lo

    def topk(self, h: torch.Tensor, k: int):
        """Global top-k (ids int64 [M, k], logits fp32 [M, k], descending) over the kept vocabulary: per-rank 4-bit
        logits and top-k, one host all-reduce of the [M, tp, k, 4] fp16 candidate slots, top-k of the union."""
        x = h.reshape(-1, h.shape[-1])
        M = x.shape[0]
        if M > 16 or x.dtype != torch.float16 or k > self.N:
            return None
        x = x.contiguous()
        logits = torch.empty(M, self.N, dtype=torch.float16, device=x.device)
        rc = qlib().dhead_logits(x.data_ptr(), self.wq.data_ptr(), self.d.data_ptr(), logits.data_ptr(), M, self.N,
                                 self.K, torch.cuda.current_stream().cuda_stream)
        if rc != 0:
            raise RuntimeError(f"dhead_logits failed: {rc}")
        v, i = logits.topk(k, dim=-1)
        g = self.gids[i]
        if self.tp == 1:
            return g, v.float()
        from vllm.distributed import tensor_model_parallel_all_reduce
        ex = torch.zeros(M, self.tp, k, 4, dtype=torch.float16, device=x.device)
        ex[:, self.rank, :, 0] = v
        ex[:, self.rank, :, 1] = (g >> 11).to(torch.float16)
        ex[:, self.rank, :, 2] = (g & 2047).to(torch.float16)
        ex = tensor_model_parallel_all_reduce(ex)
        vals = ex[..., 0].reshape(M, self.tp * k)
        ids = (ex[..., 1].to(torch.int64) * 2048 + ex[..., 2].to(torch.int64)).reshape(M, self.tp * k)
        v2, sel = vals.topk(k, dim=-1)
        return ids.gather(-1, sel), v2.float()


def _load_ids(vllm_config):
    path = os.environ.get("VLLM_TURING_DRAFT_VOCAB", DEFAULT_FILE)
    if path == "0" or not os.path.exists(path):
        return None
    n = int(os.environ.get("VLLM_TURING_DRAFT_VOCAB_N", "65536"))
    ids = []
    with open(path) as f:
        for tok in f.read().split():
            ids.append(int(tok))
            if len(ids) >= n:
                break
    try:   # added / special tokens (chat markers, think tags, tool-call tags) must stay draftable
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained(vllm_config.model_config.tokenizer, trust_remote_code=True)
        ids += list(tk.added_tokens_decoder.keys())
    except Exception as e:   # noqa: BLE001
        logger.warning("vllm_turing: draft vocab without added tokens (%s)", e)
    return torch.tensor(sorted(set(i for i in ids if i >= 0)), dtype=torch.int64)


def _attach(obj):
    """After the draft model is loaded: build the pruned head when the draft shares the int8 target head."""
    obj._turing_dv = None
    reuse = getattr(obj, "_turing_dv_reuse", None)   # combo: the MTP drafter shares the DFlash drafter's head
    if reuse is not None:
        obj._turing_dv = reuse
        return
    try:
        lm = getattr(obj.model, "lm_head", None)
        if lm is None or not hasattr(lm, "turing_w8") or getattr(obj, "use_heterogeneous_vocab", False):
            return
        ids = _load_ids(obj.vllm_config)
        if ids is not None:
            obj._turing_dv = _PrunedHead(lm, ids)
            logger.info("vllm_turing: pruned draft vocabulary, %d tokens (%d on this rank)", len(ids),
                        obj._turing_dv.rows.shape[0])
            if Q4:
                obj._turing_dv.q4 = _Q4Head(lm, ids)
                logger.info("vllm_turing: 4-bit balanced draft head, %d rows on this rank (%.1f MB)",
                            obj._turing_dv.q4.N, (obj._turing_dv.q4.wq.numel() + 2 * obj._turing_dv.q4.d.numel()) / 1e6)
                if DFLASH_Q4 and hasattr(obj.model, "compute_candidates"):   # DFlash2 candidate head
                    obj.model._turing_q4 = obj._turing_dv.q4
    except Exception as e:   # noqa: BLE001
        logger.warning("vllm_turing: pruned draft vocabulary unavailable (%s)", e)
        obj._turing_dv = None


def install():
    from vllm.platforms import current_platform
    cap = current_platform.get_device_capability()
    if cap is None or cap.to_int() != 75 or os.environ.get("VLLM_TURING_DRAFT_VOCAB", "") == "0":
        return
    if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DRAFTVOCAB_OFF")):
        return
    lib = lmhead.lib()
    lib.w8a16_gemv_rows.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 4 + [ctypes.c_void_p, ctypes.c_void_p]
    lib.w8a16_gemv_rows.restype = ctypes.c_int

    # model runner v1: SpecDecodeBaseProposer._greedy_sample
    from vllm.v1.spec_decode import llm_base_proposer as lbp
    P = lbp.SpecDecodeBaseProposer
    if not getattr(P, "_vllm_turing_dv", False):
        orig_load, orig_greedy = P.load_model, P._greedy_sample

        def load_model(self, target_model):
            orig_load(self, target_model)
            _attach(self)

        def _greedy_sample(self, hidden_states):
            dv = getattr(self, "_turing_dv", None)
            return dv.sample(hidden_states) if dv is not None else orig_greedy(self, hidden_states)

        P.load_model, P._greedy_sample, P._vllm_turing_dv = load_model, _greedy_sample, True

    # model runner v2 (vllm/v1/worker/gpu): DraftModelSpeculator.sample_draft, greedy case (no draft_logits cache)
    try:
        from vllm.v1.worker.gpu.spec_decode import speculator as sp
    except Exception:   # noqa: BLE001
        return
    S = sp.DraftModelSpeculator
    if getattr(S, "_vllm_turing_dv", False):
        return
    s_load, s_sample = S.load_model, S.sample_draft

    def s_load_model(self, target_model):
        s_load(self, target_model)
        _attach(self)

    def sample_draft(self, hidden_states, sample_src_positions, idx_mapping, temperature, seeds, draft_step,
                     draft_logits):
        dv = getattr(self, "_turing_dv", None)
        if dv is not None and draft_logits is None:
            return dv.sample(hidden_states)
        return s_sample(self, hidden_states, sample_src_positions, idx_mapping, temperature, seeds, draft_step,
                        draft_logits)

    S.load_model, S.sample_draft, S._vllm_turing_dv = s_load_model, sample_draft, True

    # DFlash2: top-k candidates per draft position from the 4-bit pruned head instead of the full int8 head (7 query
    # rows per request read the whole 124k-row shard otherwise); scale / soft cap as the candidate logits processor
    if DFLASH_Q4:
        try:
            from vllm.model_executor.models import qwen3_dflash2 as qd2
        except Exception:   # noqa: BLE001
            return
        D = qd2.DFlash2Qwen3ForCausalLM
        orig_cc = D.compute_candidates

        def compute_candidates(self, hidden_states):
            q4 = getattr(self, "_turing_q4", None)
            if q4 is not None:
                r = q4.topk(hidden_states, self.model.candidate_selector.top_k)
                if r is not None:
                    ids, vals = r
                    lp = self.candidate_logits_processor
                    if lp.scale != 1.0:
                        vals = vals * lp.scale
                    if lp.soft_cap is not None:
                        vals = torch.tanh(vals / lp.soft_cap) * lp.soft_cap
                    return ids, vals
            return orig_cc(self, hidden_states)

        D.compute_candidates = compute_candidates
