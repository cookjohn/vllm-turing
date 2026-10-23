#!/usr/bin/env python3
"""Qwen3.8-27B BF16 (HF safetensors) -> INT8 W8A8 compressed-tensors checkpoint for vLLM (Turing: int8 tensor cores,
no FP8). Data-free: weights int8 symmetric per output channel (scale = amax / 127, stored f32), activations int8
dynamic per token at run time (config only). Quantized: the Linears of the text layers (GDN in_proj_qkv / z / out_proj,
attention q / k / v / o, MLP gate / up / down) and of the MTP layer. Kept as stored (bf16): embeddings, lm_head, the
vision tower, norms, GDN conv1d / A_log / dt_bias / in_proj_a / in_proj_b, mtp.fc (see the ignore list; per-tensor
errors from w8a8_errs.py: mean 1.2 %, mtp.fc 12.8 %, in_proj_a / b 2-3 %, everything else <= 2 %).
Streams one input shard at a time (the 54 GB model never sits in RAM), quantizes on a GPU, writes the same shard
layout and a new index, copies the tokenizer / processor files and writes config.json with the quantization_config.
Run with the vLLM env python (needs torch + safetensors).
With --w8a16 [group size, default 128; 0 = per channel]: weight-only int8 instead (activations stay fp16), the
compressed-tensors pack-quantized format of vLLM's WNA16 scheme (Marlin): weight_packed int32 [N, K/4] holding
q + 128 little-endian, weight_scale fp16 [N, K/group], weight_shape.
With --mixed: W8A16 group-128 for the projections whose inputs carry the activation outliers (down_proj, o_proj,
out_proj: no activation quantization there) and W8A8 for the rest (int8 tensor cores for most of the prefill FLOPs);
per-group formats in one checkpoint ("mixed-precision").
usage: w8a8_convert.py <src dir> <dst dir> [cuda device, default cuda:1] [--w8a16 [group] | --mixed]"""
import json
import os
import re
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

args = [a for a in sys.argv[1:]]
MIXED = "--mixed" in args
if MIXED:
    args.remove("--mixed")
MIXED_W16 = re.compile(r"\.(down_proj|o_proj|out_proj)\.weight$")
W8A16 = "--w8a16" in args
GROUP = 128
if W8A16:
    i = args.index("--w8a16")
    if i + 1 < len(args) and args[i + 1].isdigit():
        GROUP = int(args.pop(i + 1))
    args.pop(i)
SMOOTH, ALPHA = None, 0.5
if "--smooth" in args:
    i = args.index("--smooth")
    SMOOTH = args[i + 1]
    del args[i:i + 2]
if "--alpha" in args:
    i = args.index("--alpha")
    ALPHA = float(args[i + 1])
    del args[i:i + 2]
src, dst = args[0], args[1]
dev = args[2] if len(args) > 2 else "cuda:1"
os.makedirs(dst, exist_ok=True)

QUANT = re.compile(r"^(model\.language_model\.layers\.\d+\.(linear_attn\.(in_proj_qkv|in_proj_z|out_proj)"
                   r"|self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))"
                   r"|mtp\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj)))\.weight$")

index = json.load(open(os.path.join(src, "model.safetensors.index.json")))
shards = sorted(set(index["weight_map"].values()))

# --- SmoothQuant (--smooth <calibration dir from vllm_turing.calib> [--alpha a]) ---------------------------------
# Per group of linear layers sharing one input X, per input channel j: s_j = max|X_j|^a / max|W_j|^(1-a); consumers
# get W[:, j] * s_j, the producer of X absorbs 1/s_j: zero-centred RMSNorm (Qwen: x * (1 + w)) -> (1 + w) / s - 1,
# gated RMSNorm of the GDN output (w * x * silu(z)) -> w / s (s shared over heads), down_proj <- up_proj rows / s,
# o_proj <- v_proj rows / s (s shared over the q heads of one kv head). in_proj_a / in_proj_b stay fp16 but consume
# the smoothed input too (scaled columns). Changed unquantized tensors are stored fp32 (no bf16 rounding of the fold).
col_scale, row_scale, norm_fix = {}, {}, {}   # tensor name -> s (columns *), s (rows /), (s, "zc" | "plain")
if SMOOTH:
    cfg0 = json.load(open(os.path.join(src, "config.json")))
    tc = cfg0.get("text_config", cfg0)
    n_q, n_kv, hd = tc["num_attention_heads"], tc["num_key_value_heads"], tc["head_dim"]
    n_vh, vhd = tc["linear_num_value_heads"], tc["linear_value_head_dim"]
    st = {}
    for tag, fix in (("", lambda n: n.replace("language_model.model.layers.", "model.language_model.layers.", 1)),
                     ("_m1", lambda n: n.replace("model.layers.", "mtp.layers.", 1))):
        parts = [torch.load(os.path.join(SMOOTH, f"rank{r}{tag}.pt")) for r in (0, 1)]
        for k in parts[0]:
            if k.rsplit(".", 1)[1] in ("o_proj", "down_proj", "out_proj"):   # row-parallel: K shards in rank order
                st[fix(k)] = torch.cat([parts[0][k], parts[1][k]]).float()
            else:
                st[fix(k)] = torch.maximum(parts[0][k], parts[1][k]).float()
    groups = []   # (activation key, consumer weight names, producer name, producer kind, reduce)
    for k in st:
        p, role = k.rsplit(".", 1)
        blk = p.rsplit(".", 1)[0] + "."   # layer prefix
        if role == "qkv_proj":
            groups.append((k, [f"{p}.{x}.weight" for x in ("q_proj", "k_proj", "v_proj")], blk + "input_layernorm.weight", "zc", None))
        elif role == "in_proj_qkvz":
            groups.append((k, [f"{p}.{x}.weight" for x in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b")],
                           blk + "input_layernorm.weight", "zc", None))
        elif role == "gate_up_proj":
            groups.append((k, [f"{p}.gate_proj.weight", f"{p}.up_proj.weight"], blk + "post_attention_layernorm.weight", "zc", None))
        elif role == "down_proj":
            groups.append((k, [f"{p}.down_proj.weight"], f"{p}.up_proj.weight", "rows", None))
        elif role == "o_proj":
            groups.append((k, [f"{p}.o_proj.weight"], f"{p}.v_proj.weight", "rows", "gqa"))
        elif role == "out_proj":
            groups.append((k, [f"{p}.out_proj.weight"], f"{p}.norm.weight", "plain", "heads"))
    need = {c for g in groups for c in g[1]}
    wmax = {}
    for shard in shards:   # pass 1: per-input-channel weight maxima of the consumers
        with safe_open(os.path.join(src, shard), framework="pt") as f:
            for name in f.keys():
                if name in need:
                    wmax[name] = f.get_tensor(name).to(dev, torch.float32).abs().amax(dim=0).cpu()
    for k, cons, prod, kind, red in groups:
        x = st[k]
        w = torch.stack([wmax[c] for c in cons]).amax(dim=0)
        if red == "gqa":      # o_proj input channel h * hd + d uses v head h // (n_q / n_kv)
            x = x.view(n_kv, n_q // n_kv, hd).amax(dim=1)
            w = w.view(n_kv, n_q // n_kv, hd).amax(dim=1)
        elif red == "heads":  # GDN norm weight is per dimension, shared by all value heads
            x = x.view(n_vh, vhd).amax(dim=0)
            w = w.view(n_vh, vhd).amax(dim=0)
        s = (x.clamp(min=1e-5) ** ALPHA / w.clamp(min=1e-5) ** (1 - ALPHA)).clamp(1e-2, 1e2)
        if red == "gqa":
            s_cols, s_prod = s.repeat_interleave(n_q // n_kv, dim=0).flatten(), s.flatten()
        elif red == "heads":
            s_cols, s_prod = s.repeat(n_vh), s
        else:
            s_cols = s_prod = s
        for c in cons:
            col_scale[c] = s_cols
        if kind == "rows":
            row_scale[prod] = s_prod
        else:
            norm_fix[prod] = (s_prod, kind)
    print(f"smoothing (alpha {ALPHA}): {len(groups)} groups, {len(col_scale)} consumer weights", flush=True)


def smooth(name, t):
    """Apply the smoothing to one tensor (float32 out) or return None if it is untouched."""
    if name not in col_scale and name not in row_scale and name not in norm_fix:
        return None
    w = t.to(dev, torch.float32)
    if name in col_scale:
        w = w * col_scale[name].to(dev)[None, :]
    if name in row_scale:
        w = w / row_scale[name].to(dev)[:, None]
    if name in norm_fix:
        s, kind = norm_fix[name]
        s = s.to(dev)
        w = (1 + w) / s - 1 if kind == "zc" else w / s
    return w


new_map, total, nq, err_max, err_sum = {}, 0, 0, 0.0, 0.0
for si, shard in enumerate(shards):
    out = {}
    with safe_open(os.path.join(src, shard), framework="pt") as f:
        for name in f.keys():
            t = f.get_tensor(name)
            ts = smooth(name, t) if SMOOTH else None
            if QUANT.match(name):
                assert t.dim() == 2, name
                w = ts if ts is not None else t.to(dev, torch.float32)
                N, K = w.shape
                w16 = W8A16 or (MIXED and MIXED_W16.search(name) is not None)
                if w16 and GROUP:
                    wg = w.view(N, K // GROUP, GROUP)
                    scale = wg.abs().amax(dim=2, keepdim=True).clamp(min=1e-12) / 127.0   # [N, K/G, 1]
                    q = torch.round(wg / scale).clamp(-128, 127).view(N, K)
                    dq = (q.view(N, K // GROUP, GROUP) * scale).view(N, K)
                    scale = scale.view(N, K // GROUP)
                else:
                    scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 127.0
                    q = torch.round(w / scale).clamp(-128, 127)
                    dq = q * scale
                # relative reconstruction error (Frobenius) as a sanity check
                e = ((dq - w).norm() / w.norm().clamp(min=1e-12)).item()
                err_max, err_sum, nq = max(err_max, e), err_sum + e, nq + 1
                base = name[: -len(".weight")]
                if w16:   # pack-quantized: unsigned q + 128, 4 per int32 along K (little-endian)
                    out[base + ".weight_packed"] = (q + 128).to(torch.uint8).contiguous().view(torch.int32).cpu()
                    out[base + ".weight_scale"] = scale.to(torch.float16).contiguous().cpu()
                    out[base + ".weight_shape"] = torch.tensor([N, K], dtype=torch.int64)
                else:
                    out[name] = q.to(torch.int8).cpu()
                    out[name + "_scale"] = scale.cpu()
                del w, q, dq
            elif ts is not None:
                out[name] = ts.cpu()   # smoothed unquantized tensor (norm weights, in_proj_a / b): fp32
            else:
                out[name] = t
    for name, t in out.items():
        new_map[name] = shard
        total += t.numel() * t.element_size()
    save_file(out, os.path.join(dst, shard), metadata={"format": "pt"})
    print(f"[{si + 1}/{len(shards)}] {shard}: {len(out)} tensors, {nq} quantized so far, "
          f"max rel err {err_max:.4f}", flush=True)
    del out

json.dump({"metadata": {"total_size": total}, "weight_map": new_map},
          open(os.path.join(dst, "model.safetensors.index.json"), "w"), indent=2)

for fn in os.listdir(src):
    if fn.endswith(".safetensors") or fn in ("model.safetensors.index.json", "config.json") or fn.startswith("."):
        continue
    shutil.copy2(os.path.join(src, fn), os.path.join(dst, fn))

cfg = json.load(open(os.path.join(src, "config.json")))
w8a16_group = {
    "targets": ["Linear"],
    "weights": {"num_bits": 8, "type": "int", "symmetric": True, "strategy": "group" if GROUP else "channel",
                "dynamic": False, "group_size": GROUP if GROUP else None, "observer": "minmax"},
    "input_activations": None,
    "output_activations": None,
    "format": "pack-quantized",
}
cfg["quantization_config"] = {
    "quant_method": "compressed-tensors",
    "format": "int-quantized",
    "quantization_status": "compressed",
    "config_groups": {
        "group_0": {
            "targets": ["Linear"],
            "weights": {"num_bits": 8, "type": "int", "symmetric": True, "strategy": "channel", "dynamic": False,
                        "group_size": None, "observer": "minmax"},
            "input_activations": {"num_bits": 8, "type": "int", "symmetric": True, "strategy": "token",
                                  "dynamic": True, "observer": None},
            "output_activations": None,
            "format": "int-quantized",
        }
    },
    # kept as stored: mtp.fc (int8 per-channel error 12.8%: its [embedding; hidden] input halves differ in scale) and the
    # GDN gate projections in_proj_a / in_proj_b (2-3%, tiny, they set the decay / update gates of the recurrence)
    "ignore": ["lm_head", "re:.*visual.*", "re:.*conv1d.*", "re:.*mtp\\.fc.*", "re:.*linear_attn\\.in_proj_(a|b)$"],
    "kv_cache_scheme": None,
    "sparsity_config": {},
    "global_compression_ratio": None,
}
if W8A16:
    cfg["quantization_config"]["format"] = "pack-quantized"
    cfg["quantization_config"]["config_groups"] = {"group_0": w8a16_group}
elif MIXED:
    w8a8_group = cfg["quantization_config"]["config_groups"]["group_0"]
    cfg["quantization_config"]["format"] = "mixed-precision"
    cfg["quantization_config"]["config_groups"] = {
        "group_0": dict(w8a16_group, targets=["re:.*down_proj$", "re:.*o_proj$", "re:.*out_proj$"]),
        "group_1": w8a8_group,
    }
json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=2)
print(f"done: {nq} tensors quantized, mean rel err {err_sum / max(nq, 1):.4f}, max {err_max:.4f}, "
      f"{total / 2**30:.2f} GiB -> {dst}")
