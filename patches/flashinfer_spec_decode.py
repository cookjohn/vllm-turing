#!/usr/bin/env python3
"""Patch vLLM's FlashInfer attention backend (vllm/v1/attention/backends/flashinfer.py) for speculative decoding on
Turing (sm75), where FlashInfer has no trtllm kernels: uniform speculative-verify decode batches (k = 1 + num_spec
query tokens per request) run as k single-token FlashInfer decodes per request over the same pages, token j seeing
seq_len - (k - 1 - j) keys (the native decode wrappers are already planned per query token, so only the paged-KV
metadata is expanded). With that the backend reports UNIFORM_BATCH CUDA-graph support, so vLLM keeps FULL graphs for
the verify step (otherwise it drops to PIECEWISE graphs and the MTP step is CPU-bound). Verify batches next to
prefills (or with cascade attention) go to the prefill path, which takes any query length.
Also lowers the backend's capability floor to sm75 (FlashInfer's FA2 kernels are correct at this model's shapes).
Non-causal uniform blocks (DFlash drafts) are flattened the same way with every query seeing the whole sequence.
Idempotent (step by step); env VLLM_TURING_FI_SPEC=0 disables the flattening at run time.
usage: flashinfer_spec_decode.py [path to flashinfer.py, default: the installed vllm's]"""
import os
import sys

if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    import vllm
    path = os.path.join(os.path.dirname(vllm.__file__), "v1", "attention", "backends", "flashinfer.py")
s = open(path, encoding="utf-8").read()
s0 = s


def sub(old, new, cnt=1):
    """Replace old by new unless new is already there (each step idempotent: re-running adds missing steps)."""
    global s
    if new in s:
        return
    assert s.count(old) == cnt, (s.count(old), old[:100])
    s = s.replace(old, new)


# helper (module level, above the decorated FIDecode dataclass)
if "def _turing_spec_as_decode" not in s:
  sub("\n@dataclass\nclass FIDecode:\n", '''

def _turing_spec_as_decode() -> bool:
    """sm75: run uniform speculative-verify decodes as flattened single-token FlashInfer decodes (vllm_turing)."""
    import os as _os

    return current_platform.is_device_capability(75) and _os.environ.get("VLLM_TURING_FI_SPEC", "1") != "0"


@dataclass
class FIDecode:
''')

# capability floor 8.0 -> 7.5 (if not already lowered by hand)
if "return capability >= DeviceCapability(8, 0) and capability <= DeviceCapability(" in s:
    sub("return capability >= DeviceCapability(8, 0) and capability <= DeviceCapability(",
        "return capability >= DeviceCapability(7, 5) and capability <= DeviceCapability(")

# persistent paged-KV buffers: one entry per verify token (1 + num_spec per request) instead of per request
sub('''        # Preparing persistent buffers
        self.paged_kv_indptr = CpuGpuBuffer(
            max_num_reqs + 1, dtype=torch.int32, device=self.device, pin_memory=False
        )
        self.paged_kv_indices = torch.zeros(
            max_num_pages, dtype=torch.int32, device=self.device
        )
        self.paged_kv_last_page_len = CpuGpuBuffer(
            max_num_reqs, dtype=torch.int32, device=self.device, pin_memory=False
        )''', '''        # Preparing persistent buffers (sm75 spec-as-decode: one entry per verify token)
        _fl = 1 + (self._num_speculative_tokens or 0) if _turing_spec_as_decode() else 1
        self.paged_kv_indptr = CpuGpuBuffer(
            max_num_reqs * _fl + 1, dtype=torch.int32, device=self.device, pin_memory=False
        )
        self.paged_kv_indices = torch.zeros(
            max_num_pages * _fl, dtype=torch.int32, device=self.device
        )
        self.paged_kv_last_page_len = CpuGpuBuffer(
            max_num_reqs * _fl, dtype=torch.int32, device=self.device, pin_memory=False
        )''')

# CUDA graph support: uniform batches are graph-safe with the flattening
sub('''        """Get the cudagraph support level for FlashInfer attention."""
''', '''        """Get the cudagraph support level for FlashInfer attention."""
        if _turing_spec_as_decode():
            return AttentionCGSupport.UNIFORM_BATCH
''')

# classify uniform verify requests as decodes
sub('''            supports_spec_as_decode=(
                self.flashinfer_trtllm_api_decode_kernel is not None
            ),''', '''            supports_spec_as_decode=(
                self.flashinfer_trtllm_api_decode_kernel is not None
                or _turing_spec_as_decode()
            ),''')

# build(): flattening decision once cascade is known
sub('''        use_cascade = common_prefix_len > 0
        uses_spec_reorder = self.reorder_batch_threshold > 1
''', '''        use_cascade = common_prefix_len > 0
        uses_spec_reorder = self.reorder_batch_threshold > 1
        # sm75 (vllm_turing): uniform verify decodes with k > 1 query tokens per request run as k single-token
        # decodes over the same pages; next to prefills / with cascade they take the prefill path instead
        spec_k = 1
        num_reqs_kv = num_reqs
        if (
            num_decodes > 0
            and num_decode_tokens > num_decodes
            and not self.use_trtllm_decode_attention
            and _turing_spec_as_decode()
        ):
            if (
                num_prefills > 0
                or use_cascade
                or self.use_dcp
                or num_decode_tokens % num_decodes != 0
                or num_reqs != num_decodes
            ):
                num_decodes, num_prefills = 0, num_reqs
                num_decode_tokens, num_prefill_tokens = 0, num_actual_tokens
            else:
                spec_k = num_decode_tokens // num_decodes
''')

# expand seq lens / page counts / block table rows per verify token (marker check: later steps edit inside this block)
if "if spec_k > 1 and seq_lens_np is not None" not in s:
  sub('''        if seq_lens_cpu is not None:
            seq_lens_np = seq_lens_cpu.numpy()
            num_blocks_np = (seq_lens_np + (page_size - 1)) // page_size
        else:
            seq_lens_np = None
            num_blocks_np = None
''', '''        if seq_lens_cpu is not None:
            seq_lens_np = seq_lens_cpu.numpy()
            num_blocks_np = (seq_lens_np + (page_size - 1)) // page_size
        else:
            seq_lens_np = None
            num_blocks_np = None
        if spec_k > 1 and seq_lens_np is not None:
            offs = np.tile(np.arange(spec_k - 1, -1, -1, dtype=seq_lens_np.dtype), num_decodes)
            seq_lens_np = np.maximum(np.repeat(seq_lens_np[:num_decodes], spec_k) - offs, 0)
            num_blocks_np = (seq_lens_np + (page_size - 1)) // page_size
            block_table_tensor = block_table_tensor[:num_decodes].repeat_interleave(spec_k, dim=0)
            num_reqs_kv = num_decode_tokens
''')

sub('''            paged_kv_indices = self._compute_flashinfer_kv_metadata(
                num_blocks_np,
                seq_lens_np,
                block_table_tensor,
                num_reqs,
                page_size,
            )''', '''            paged_kv_indices = self._compute_flashinfer_kv_metadata(
                num_blocks_np,
                seq_lens_np,
                block_table_tensor,
                num_reqs_kv,
                page_size,
            )''')

# non-causal drafts (DFlash): a uniform block of k non-causal queries per request is k single-token decodes that all
# see the whole sequence (no causal offsets), so it takes the same graph-safe decode path; without this the drafter's
# attention went through the non-causal prefill wrapper inside FULL CUDA graphs (stale plan: garbage drafts, 0 %
# acceptance, illegal memory accesses at long context)
sub('''        causal = common_attn_metadata.causal
        route_decode = causal or self.use_xqa
''', '''        causal = common_attn_metadata.causal
        _turing_nc_flat = not causal and not self.use_dcp and _turing_spec_as_decode()
        route_decode = causal or self.use_xqa or _turing_nc_flat
''')
sub('''            offs = np.tile(np.arange(spec_k - 1, -1, -1, dtype=seq_lens_np.dtype), num_decodes)
''', '''            offs = np.tile(np.arange(spec_k - 1, -1, -1, dtype=seq_lens_np.dtype), num_decodes)
            if not causal:   # non-causal block: every query sees all keys
                offs = np.zeros_like(offs)
''')

# uniform verify as ONE causal (or non-causal, DFlash) FA2 prefill per request on a CUDA-graph prefill wrapper with
# fixed planning buffers (one per batch size / query width / causality; approach of weicj/vLLM-2080Ti-Definitive),
# instead of the k flattened decodes above: those re-read and re-dequantize (FP8, in software on sm75) the request's
# whole KV cache once per verify token, so the verify attention grew k-fold with depth (MTP4 decode 122 -> 98.5 tok/s
# from 4K to 32K; the kernel alone 343 vs 1022 us at 32K, 933 vs 3137 at 131K). Graph-safe: tests/test_fi_specprefill.py
# (capture, re-plan with other lengths / pages, replay: exact). VLLM_TURING_FI_SPEC_PREFILL=0 restores the flattening.
if "def _turing_spec_prefill" not in s:
  sub('''

@dataclass
class FIDecode:
''', '''

def _turing_spec_prefill() -> bool:
    """sm75: uniform verify through a CUDA-graph FA2 prefill wrapper (vllm_turing), else flattened decodes."""
    import os as _os

    return _turing_spec_as_decode() and _os.environ.get("VLLM_TURING_FI_SPEC_PREFILL", "1") != "0"


@dataclass
class FIDecode:
''')
sub('''        if spec_k > 1 and seq_lens_np is not None:
            offs = np.tile(''', '''        if spec_k > 1 and seq_lens_np is not None and not _turing_spec_prefill():
            offs = np.tile(''')
sub('''                decode_wrapper = self._get_decode_wrapper(
                    num_input_tokens, use_cudagraph
                )
''', '''                if spec_k > 1 and _turing_spec_prefill():
                    attn_metadata.decode = FIDecode(
                        wrapper=self._turing_plan_spec_prefill(
                            num_decodes, spec_k, bool(causal), paged_kv_indices
                        )
                    )
                    return attn_metadata
                decode_wrapper = self._get_decode_wrapper(
                    num_input_tokens, use_cudagraph
                )
''')
sub('''    def _get_decode_wrapper(self, batch_size: int, use_cudagraph: bool = False):
''', '''    def _turing_plan_spec_prefill(self, num_reqs, k, causal, paged_kv_indices):
        """sm75 (vllm_turing): plan the uniform verify (k query tokens per request) on a CUDA-graph FA2 prefill
        wrapper whose planning buffers never move, one wrapper per (batch, k, causal)."""
        pws = self.__dict__.setdefault("_turing_spec_pws", {})
        key = (num_reqs, k, causal)
        w = pws.get(key)
        if w is None:
            max_pages = num_reqs * cdiv(self.model_config.max_model_len, self.page_size)

            def ib(n):
                return torch.empty(n, dtype=torch.int32, device=self.device)

            w = BatchPrefillWithPagedKVCacheWrapper(
                self._get_workspace_buffer(),
                get_flashinfer_layout_string(self.kv_cache_layout),
                backend="fa2",
                use_cuda_graph=True,
                qo_indptr_buf=ib(num_reqs + 1),
                paged_kv_indptr_buf=ib(num_reqs + 1),
                paged_kv_indices_buf=ib(max_pages),
                paged_kv_last_page_len_buf=ib(num_reqs),
            )
            pws[key] = w
        w.plan(
            torch.arange(0, num_reqs * k + 1, k, dtype=torch.int32),
            self.paged_kv_indptr.cpu[: num_reqs + 1],
            paged_kv_indices,
            self.paged_kv_last_page_len.cpu[:num_reqs],
            self.num_qo_heads * self.dcp_world_size,
            self.num_kv_heads,
            self.head_dim,
            self.page_size,
            causal=causal,
            pos_encoding_mode="NONE",
            sm_scale=self.sm_scale,
            window_left=self.window_left,
            logits_soft_cap=self.logits_soft_cap or 0.0,
            q_data_type=self.q_data_type_decode,
            kv_data_type=self.kv_cache_dtype,
            o_data_type=self.model_config.dtype,
        )
        return w

    def _get_decode_wrapper(self, batch_size: int, use_cudagraph: bool = False):
''')

# incremental re-plan for autoregressive draft steps (vllm_turing nosync.py): step i's decode metadata is step 1's with
# every length + (i - 1) and, unless a page boundary is crossed, the same pages, so later steps re-plan the same
# decode wrapper from the remembered step-1 plan (~0.2 ms) instead of a full build (~1.3 ms of Python per step)
sub('''                attn_metadata.decode = FIDecode(wrapper=decode_wrapper)
        return attn_metadata
''', '''                self._turing_dec = (
                    (decode_wrapper, seq_lens_np[:num_input_tokens].copy(),
                     num_blocks_np[:num_input_tokens].copy(), paged_kv_indices, o_dtype)
                    if pure_decode and seq_lens_np is not None and spec_k == 1
                    else None
                )
                attn_metadata.decode = FIDecode(wrapper=decode_wrapper)
        return attn_metadata

    def _turing_replan_decode(self, delta: int) -> bool:
        """Re-plan the last single-token decode plan with every length + delta (vllm_turing); False when the pages
        would change (the caller then does a full build)."""
        d = getattr(self, "_turing_dec", None)
        if d is None:
            return False
        wrapper, sl0, nb0, indices, o_dtype = d
        sl = np.minimum(sl0 + delta, self.model_config.max_model_len)
        nb = (sl + (self.page_size - 1)) // self.page_size
        if not np.array_equal(nb, nb0):
            return False
        indptr = np.zeros(len(nb) + 1, dtype=np.int32)
        np.cumsum(nb, out=indptr[1:])
        last = (sl - (nb - 1) * self.page_size).astype(np.int32)
        fast_plan_decode(
            wrapper,
            indptr_cpu=torch.from_numpy(indptr),
            indices=indices,
            last_page_len_cpu=torch.from_numpy(last),
            seq_lens_cpu=torch.from_numpy(sl.astype(np.int32)),
            num_qo_heads=self.num_qo_heads * self.dcp_world_size,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            page_size=self.page_size,
            pos_encoding_mode="NONE",
            sm_scale=self.sm_scale,
            window_left=self.window_left,
            logits_soft_cap=self.logits_soft_cap,
            q_data_type=self.q_data_type_decode,
            kv_data_type=self.kv_cache_dtype,
            o_data_type=o_dtype,
            fixed_split_size=self.decode_fixed_split_size,
            disable_split_kv=self.disable_split_kv,
        )
        return True
''')

# exact CPU sequence lengths supplied by the caller (vllm_turing nosync.py: the MTP draft steps compute them from one
# early non-blocking copy): no seq_lens.cpu() device sync in build(), which drained the GPU before every draft step
if "_TURING_EXACT_SEQ_LENS" not in s:
  sub('''

def _turing_spec_as_decode() -> bool:''', '''

# set by vllm_turing (nosync.py) while it builds metadata whose seq_lens_cpu_upper_bound is exact
_TURING_EXACT_SEQ_LENS = False


def _turing_spec_as_decode() -> bool:''')
sub('''            if (
                self._num_speculative_tokens == 0
                and common_attn_metadata.seq_lens_cpu_upper_bound is not None
            ):''', '''            if (
                (self._num_speculative_tokens == 0 or _TURING_EXACT_SEQ_LENS)
                and common_attn_metadata.seq_lens_cpu_upper_bound is not None
            ):''')

if s == s0:
    print("already patched:", path)
else:
    open(path, "w", encoding="utf-8").write(s)
    print("patched:", path)
