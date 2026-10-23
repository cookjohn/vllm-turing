#!/usr/bin/env python3
"""vllm_turing vitlowmem: low-memory Qwen3.5 vision encoder forward vs the stock forward, with the model's real vision
weights at TP 1: output difference, peak memory and time for a max-size image (65536 patches), a two-image batch and
a two-frame video.
usage: test_vitlowmem.py [model dir]   (vLLM env python, one free GPU)"""
import glob
import json
import os
import sys
import time

import torch
from safetensors import safe_open
from transformers import AutoConfig

os.environ["VLLM_TURING_DISABLE"] = "1"   # the plugin entry point must not patch before the stock forward is saved
model = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/models/Qwen3.8-27B-SQ06")

from vllm.config import VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.distributed import init_distributed_environment, initialize_model_parallel  # noqa: E402

cfg = AutoConfig.from_pretrained(model)
vc = VllmConfig()
with set_current_vllm_config(vc):
    init_distributed_environment(world_size=1, rank=0, local_rank=0,
                                 distributed_init_method="tcp://127.0.0.1:29571", backend="nccl")
    initialize_model_parallel(1, 1)
torch.set_default_dtype(torch.float16)
with set_current_vllm_config(vc):
    from vllm.model_executor.models import qwen3_vl
    vit = qwen3_vl.Qwen3_VisionTransformer(cfg.vision_config, norm_eps=getattr(cfg, "rms_norm_eps", 1e-6),
                                           prefix="visual").cuda().eval()
torch.set_default_dtype(torch.float32)

index = json.load(open(os.path.join(model, "model.safetensors.index.json")))["weight_map"]
prefix = next(p for p in ("model.visual.", "visual.", "model.vision_tower.") if any(k.startswith(p) for k in index))


def weights():
    for f in sorted({index[k] for k in index if k.startswith(prefix)}):
        with safe_open(os.path.join(model, f), "pt") as sf:
            for k in sf.keys():
                if k.startswith(prefix):
                    yield k[len(prefix):], sf.get_tensor(k).to(torch.float16)


loaded = vit.load_weights(weights())
print(f"loaded {len(loaded)} vision weights ({prefix}*), attention backend {vit.attn_backend}")

stock = qwen3_vl.Qwen3_VisionTransformer.forward
from vllm_turing import vitlowmem  # noqa: E402

vitlowmem.ENABLED = True
vitlowmem.install()
lowmem = qwen3_vl.Qwen3_VisionTransformer.forward
assert lowmem is not stock


def run(fwd, x, grid):
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    with torch.inference_mode():
        out = fwd(vit, x, grid)
    torch.cuda.synchronize()
    return out, (torch.cuda.max_memory_allocated() - base) / 2**30, time.time() - t


torch.manual_seed(0)
patch_dim = 3 * cfg.vision_config.temporal_patch_size * cfg.vision_config.patch_size ** 2
for name, grid in (("max image 256x256 patches", [[1, 256, 256]]),
                   ("two images 128x128 + 96x160", [[1, 128, 128], [1, 96, 160]]),
                   ("video 2 frames 96x128", [[2, 96, 128]])):
    n = sum(t * h * w for t, h, w in grid)
    x = torch.randn(n, patch_dim, device="cuda", dtype=torch.float32)   # processor output dtype
    a, pa, ta = run(stock, x, grid)
    b, pb, tb = run(lowmem, x, grid)
    d = (a.float() - b.float()).abs()
    rel = d.max().item() / a.float().abs().max().item()
    print(f"{name}: {n} patches -> {tuple(b.shape)} | stock peak {pa:.3f} GiB {ta:.1f} s | low-mem peak {pb:.3f} GiB "
          f"{tb:.1f} s | max abs diff {d.max().item():.3e} (rel {rel:.2e}), mean {d.mean().item():.3e}")
    if len(grid) > 1:
        # stock's own variation from the batch shape (GEMM algorithm choice depends on M): image 1 alone vs batched
        n0 = grid[0][0] * grid[0][1] * grid[0][2]
        a0, _, _ = run(stock, x[:n0].contiguous(), grid[:1])
        m = a0.shape[0]
        d0 = (a[:m].float() - a0.float()).abs()
        d1 = (b[:m].float() - a0.float()).abs()
        print(f"  image 1 alone (stock) vs batched stock: max {d0.max().item():.3e} mean {d0.mean().item():.3e} | "
              f"vs batched low-mem: max {d1.max().item():.3e} mean {d1.mean().item():.3e}")
    assert rel < 2e-2, "outputs differ"
    del a, b, x
print("ok")
