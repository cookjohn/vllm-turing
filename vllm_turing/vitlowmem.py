"""Low-memory Qwen3-VL / Qwen3.5 vision encoder forward (vLLM 0.31, TORCH_SDPA ViT attention: sm75 has no FA2).
vLLM profiles the encoder at startup with one image of the maximum size (16.7 MP: 65536 patches -> 16384 tokens) and
the activation peak it sees is reserved for the whole run, out of the KV cache. The stock forward keeps several
full-length tensors alive at once inside a block (norm output, qkv, a contiguous q/k copy, its rotated copy, the
attention output and its transposed copy, the projection output; fc1 and activation outputs in the MLP).
Same math with a bounded working set:
  - qkv is computed in row chunks into one buffer, rotary embedding applied in place on its q and k parts
    (apply_rotary_emb_qkv_: the same Triton kernel the stock path runs out of place);
  - attention runs per image / frame (cu_seqlens) over query-row chunks against the full keys (SDPA results per query
    row do not depend on the other rows), each chunk projected and added into the residual stream in place;
  - norm2 + MLP + residual, the patch embedding and the patch merger run on row chunks.
Images up to VLLM_TURING_VIT_CHUNK patches (default 8192, ~2 MP) take the stock path. VLLM_TURING_VIT_LOWMEM=0
disables."""
import logging
import os

import torch
import torch.nn.functional as F

logger = logging.getLogger("vllm.turing")
ENABLED = os.environ.get("VLLM_TURING_VIT_LOWMEM", "1") != "0"
CHUNK = max(64, int(os.environ.get("VLLM_TURING_VIT_CHUNK", "8192")) // 64 * 64)   # rows (patches), merge-unit aligned


def _rows(n, step):
    for a in range(0, n, step):
        yield a, min(a + step, n)


def _block(blk, x, bounds, cos, sin):
    """Qwen3_VisionBlock.forward on x [s, 1, D], x updated in place."""
    from vllm.vllm_flash_attn.layers.rotary import apply_rotary_emb_qkv_
    attn = blk.attn
    s = x.shape[0]
    hp, d = attn.num_attention_heads_per_partition, attn.hidden_size_per_attention_head
    qkv = None
    for a, b in _rows(s, CHUNK):
        y, _ = attn.qkv(blk.norm1(x[a:b]))
        if qkv is None:
            qkv = y.new_empty(s, *y.shape[1:])
        qkv[a:b] = y
        del y
    qkv5 = qkv.view(1, s, 3, hp, d)
    apply_rotary_emb_qkv_(qkv5, cos, sin)
    q, k, v = (t.transpose(1, 2) for t in qkv5.unbind(2))   # [1, hp, s, d] views
    scale = attn.attn.scale
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b <= a:
            continue
        kk, vv = k[:, :, a:b], v[:, :, a:b]
        for c0, c1 in _rows(b - a, CHUNK):
            o = F.scaled_dot_product_attention(q[:, :, a + c0:a + c1], kk, vv, dropout_p=0.0, scale=scale)
            o, _ = attn.proj(o.transpose(1, 2).reshape(c1 - c0, 1, hp * d))
            x[a + c0:a + c1] += o
            del o
    del q, k, v, qkv5, qkv
    for a, b in _rows(s, CHUNK):
        xc = x[a:b]
        xc += blk.mlp(blk.norm2(xc))
    return x


def _merger(m, x):
    """Qwen3_VisionPatchMerger.forward on row chunks (x [s, 1, D] -> [s / unit, out])."""
    if m.use_postshuffle_norm:
        return m(x)
    s, D = x.shape[0], x.shape[-1]
    unit = m.hidden_size // D
    out = None
    for a, b in _rows(s, CHUNK):
        y = m.norm(x[a:b]).view(-1, m.hidden_size)
        y, _ = m.linear_fc1(y)
        y = m.act_fn(y)
        y, _ = m.linear_fc2(y)
        if out is None:
            out = y.new_empty(s // unit, y.shape[-1])
        out[a // unit:b // unit] = y
        del y
    return out


def install():
    if not ENABLED:
        return
    try:
        from vllm.model_executor.models import qwen3_vl
        from vllm.v1.attention.backends.registry import AttentionBackendEnum
    except Exception:   # noqa: BLE001
        return
    T = qwen3_vl.Qwen3_VisionTransformer
    if getattr(T, "_vllm_turing_lowmem", False):
        return
    orig = T.forward

    def forward(self, x, grid_thw, *, encoder_metadata=None):
        if (x.shape[0] <= CHUNK or self.attn_backend != AttentionBackendEnum.TORCH_SDPA
                or torch.cuda.is_current_stream_capturing()
                or any(getattr(b.attn.attn, "fp8_enabled", False) for b in self.blocks[:1])):
            return orig(self, x, grid_thw, encoder_metadata=encoder_metadata)
        own_metadata = encoder_metadata is None
        if own_metadata:
            grid_thw_list = grid_thw if isinstance(grid_thw, list) else grid_thw.tolist()
            encoder_metadata = self.prepare_encoder_metadata(grid_thw_list)
        s = x.shape[0]
        hs = torch.empty(s, self.hidden_size, dtype=self.dtype, device=self.device)
        for a, b in _rows(s, CHUNK):
            hs[a:b] = self.patch_embed(x[a:b].to(device=self.device, dtype=self.dtype, non_blocking=True))
        hs += encoder_metadata["pos_embeds"]
        if own_metadata:
            encoder_metadata.pop("pos_embeds")   # only read here
        hs = hs.unsqueeze(1)
        bounds = encoder_metadata["cu_seqlens"].tolist()
        cos, sin = encoder_metadata["rotary_pos_emb_cos"], encoder_metadata["rotary_pos_emb_sin"]
        deepstack = []
        for i, blk in enumerate(self.blocks):
            hs = _block(blk, hs, bounds, cos, sin)
            if i in self.deepstack_visual_indexes:
                deepstack.append(_merger(self.deepstack_merger_list[self.deepstack_visual_indexes.index(i)], hs))
        out = _merger(self.merger, hs)
        return torch.cat([out] + deepstack, dim=1) if deepstack else out

    T.forward, T._vllm_turing_lowmem = forward, True
    logger.info("vllm_turing: low-memory vision encoder forward for images over %d patches", CHUNK)
