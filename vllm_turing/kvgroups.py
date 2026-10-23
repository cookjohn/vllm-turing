"""KV cache grouping fix for hybrid targets with a sliding-window drafter (vLLM 0.31).

vLLM's uniform-page-size grouping (kv_cache_utils._get_kv_cache_groups_uniform_page_size) uses the smallest layer
bucket as the group size. With Qwen3.8 (16 full-attention + 48 GDN layers) and the DFlash2 drafter (5 sliding-window
layers) that is 5: the target layers are cut into 4 + 10 groups (instead of 1 + 3) and every group claims its own
blocks for the whole context, so a 16k context needed 2.33 GiB of KV cache. Sliding-window (and chunked local) layers
only ever hold a window of blocks, so padding them is cheap: this version leaves those buckets out of the group size
choice (when other buckets exist) and pads them instead. Same grouping as upstream for everything else.
VLLM_TURING_KVGROUPS=0 disables."""
import logging
import os
from collections import defaultdict

logger = logging.getLogger("vllm.turing")


def install():
    if os.environ.get("VLLM_TURING_KVGROUPS", "1") == "0":
        return
    from vllm.utils.math_utils import cdiv
    from vllm.v1.core import kv_cache_utils as ku
    from vllm.v1.kv_cache_interface import ChunkedLocalAttentionSpec, SlidingWindowSpec

    if getattr(ku, "_vllm_turing_kvgroups", False):
        return
    orig = ku._get_kv_cache_groups_uniform_page_size

    def _get_kv_cache_groups_uniform_page_size(kv_cache_spec):
        same_type_layers = defaultdict(list)
        for layer_name, layer_spec in kv_cache_spec.items():
            same_type_layers[layer_spec].append(layer_name)
        layer_buckets, spec_buckets = [], []
        for layer_spec, layer_names in same_type_layers.items():
            for names, specs in zip(layer_buckets, spec_buckets):
                try:
                    type(specs[0]).merge([*specs, layer_spec])
                except (AssertionError, ValueError):
                    continue
                names.extend(layer_names)
                specs.append(layer_spec)
                break
            else:
                layer_buckets.append(list(layer_names))
                spec_buckets.append([layer_spec])
        windowed = [isinstance(s[0], (SlidingWindowSpec, ChunkedLocalAttentionSpec)) for s in spec_buckets]
        if not any(windowed) or all(windowed):
            return orig(kv_cache_spec)
        sizes = [len(l) for l, w in zip(layer_buckets, windowed) if not w]
        group_size = min(sizes)
        if max(sizes) < group_size * 1.5:
            group_size = max(sizes)
        grouped_layers = []
        for layers in layer_buckets:
            num_groups = cdiv(len(layers), group_size)
            for i in range(num_groups):
                grouped_layers.append(layers[i::num_groups])
        logger.info("vllm_turing: KV cache groups of %d layers: %s", group_size,
                    ", ".join(f"{type(s[0]).__name__} {len(l)} -> {cdiv(len(l), group_size)} group(s)"
                              for l, s in zip(layer_buckets, spec_buckets)))
        return ku.create_kv_cache_group_specs(kv_cache_spec, grouped_layers)

    ku._get_kv_cache_groups_uniform_page_size = _get_kv_cache_groups_uniform_page_size
    ku._vllm_turing_kvgroups = True

    # diagnostics: what each final group claims (VLLM_TURING_KVDEBUG=1)
    if os.environ.get("VLLM_TURING_KVDEBUG") == "1":
        orig_groups = ku.get_kv_cache_groups

        def get_kv_cache_groups(vllm_config, kv_cache_spec):
            groups = orig_groups(vllm_config, kv_cache_spec)
            types = defaultdict(int)
            for s in kv_cache_spec.values():
                types[type(s).__name__] += 1
            logger.info("vllm_turing kvdebug: layer specs %s", dict(types))
            for i, g in enumerate(groups):
                s = g.kv_cache_spec
                try:
                    need = s.max_memory_usage_bytes(vllm_config)
                except Exception as e:   # noqa: BLE001
                    need = f"? ({e})"
                logger.info("vllm_turing kvdebug: group %d %s layers=%d block=%s page=%s need/layer=%s %s", i,
                            type(s).__name__, len(g.layer_names), getattr(s, "block_size", "?"),
                            getattr(s, "page_size_bytes", "?"), need, g.layer_names[:2])
            return groups

        ku.get_kv_cache_groups = get_kv_cache_groups
