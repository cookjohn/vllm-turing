"""INT8 weight-only fc for DFlash drafters (vLLM 0.31): the fc that combines the 5 concatenated target hidden states
(25600 -> 5120) is kept fp16 by the drafter checkpoints (a bad fit for per-token activation quantization) and is
replicated on every TP rank, so each draft step reads its 250 MB in full. Per-row int8 weights with fp16 activations
(lmhead.quantize_rows; weight error 1.1-2.1 % per input segment) halve that and free 125 MB per rank; M <= 16 rows (the
decode verify) go through the W8A16 GEMV, prefill chunks dequantize 512 weight rows at a time.
VLLM_TURING_DFLASH_FC8=0 disables."""
import logging
import os

import torch

from . import lmhead

logger = logging.getLogger("vllm.turing")
CHUNK = 512


def _quantize_fc(fc) -> bool:
    w = getattr(fc, "weight", None)
    if w is None or w.dtype not in (torch.float16, torch.bfloat16) or w.dim() != 2 or getattr(fc, "bias", None) is not None:
        return False
    q, s = lmhead.quantize_rows(w.data, chunk=1024)   # 1024 x 25600 fp32 temporaries
    N, K = q.shape
    dtype = w.dtype
    fc.weight.data = torch.empty(0, dtype=dtype, device=w.device)   # free the fp16 copy
    fc.turing_q, fc.turing_s = q, s
    torch.cuda.empty_cache()

    def forward(x):
        shp = x.shape
        x2 = x.reshape(-1, K).contiguous()
        if x2.shape[0] <= 16:
            out = torch.ops.vllm_turing.w8a16_mm(x2, q, s, None)
        else:
            out = torch.empty(x2.shape[0], N, dtype=x2.dtype, device=x2.device)
            for j in range(0, N, CHUNK):
                out[:, j:j + CHUNK] = x2 @ (q[j:j + CHUNK].to(x2.dtype) * s[j:j + CHUNK, None].to(x2.dtype)).t()
        return out.reshape(*shp[:-1], N)

    fc.forward = forward
    return True


def install():
    if os.environ.get("VLLM_TURING_DFLASH_FC8", "1") == "0":
        return
    try:
        from vllm.v1.worker.gpu.spec_decode.dflash import speculator as ds
    except Exception:   # noqa: BLE001
        return
    S = ds.DFlashSpeculator
    if getattr(S, "_vllm_turing_fc8", False):
        return
    orig = S.load_draft_model

    def load_draft_model(self, target_model, target_attn_layer_names):
        model = orig(self, target_model, target_attn_layer_names)
        fc = getattr(getattr(model, "model", None), "fc", None)
        if fc is not None and _quantize_fc(fc):
            logger.info("vllm_turing: DFlash fc %s -> int8 weights (W8A16 GEMV)", tuple(fc.turing_q.shape))
        return model

    S.load_draft_model, S._vllm_turing_fc8 = load_draft_model, True
