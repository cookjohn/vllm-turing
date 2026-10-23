#!/usr/bin/env python3
"""Compute utilization and waiting time from one rank's torch-profiler trace (vLLM --profiler-config torch) of a
profiled phase. Over the window from the first to the last GPU activity:
  compute    = union of compute kernels (everything that is not communication)
  comm wait  = time with communication activity (host all-reduce / AR8 / DMA all-gather kernels, NCCL, D2H / H2D
               copies) and no compute kernel: the SMs wait for the other GPU / PCIe
  idle       = no GPU activity at all: the GPU waits for the CPU (Python step preparation, scheduling)
plus kernel time per category, and the work done in the window: prefill FLOPs (model math for N prompt tokens) against
the int8 tensor-core peak at the SM clock sampled during the phase, or decode bytes per step against 616 GB/s.
usage: trace_util.py <trace.json.gz> prefill <prompt tokens> <mean SM MHz>
       trace_util.py <trace.json.gz> decode <generated tokens> <mean SM MHz> [context tokens]"""
import gzip
import json
import sys

# Qwen3.8-27B per TP2 rank
LINEAR_PARAMS = 24.35e9                 # text decoder linear weights (64 layers: MLP 17.1B, GDN 5.57B, attention 1.68B)
ATTN_LAYERS, Q_HEADS, HEAD_DIM = 16, 12, 256
MTP_PARAMS = 0.47e9                     # MTP layer (attention + MLP + fc), whole
LM_HEAD = 248320 * 5120
SMS, INT8_OPS_PER_SM_CLK = 68, 2048     # Turing: 8 tensor cores x 128 int8 MAC per SM per clock
PEAK_BW = 616e9

COMM = ("har_kernel", "ar8_", "dag_wait", "dag_signal", "nccl", "AllGather", "AllReduce")
CATS = (("int8 GEMM (prefill, CUTLASS)", ("cutlass",)),
        ("int8 GEMV (decode / verify)", ("w8a8_gemv", "w8a16", "dhead")),
        ("attention", ("fa8_", "fd_attn", "fd_merge", "BatchPrefill", "BatchDecode", "PersistentVariableLength",
                       "MergeStates", "reshape_and_cache")),
        ("GDN (linear attention)", ("gdn_", "chunk_", "recompute_w_u", "merge_16x16", "fused_sigmoid", "causal_conv1d",
                                    "fused_post_conv", "solve_tril", "l2norm", "fused_recurrent")),
        ("communication kernels", COMM),
        ("quant / norm / activation / misc", ()))


def load(path):
    d = json.load(gzip.open(path)) if path.endswith(".gz") else json.load(open(path))
    ev = d["traceEvents"] if isinstance(d, dict) else d
    return [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]


def union(iv):
    iv = sorted(iv)
    tot, s0, e0 = 0.0, None, None
    out = []
    for s, e in iv:
        if e0 is None or s > e0:
            if e0 is not None:
                out.append((s0, e0))
            s0, e0 = s, e
        else:
            e0 = max(e0, e)
    if e0 is not None:
        out.append((s0, e0))
    return out


def length(iv):
    return sum(e - s for s, e in iv)


def intersect_len(a, b):
    i = j = 0
    tot = 0.0
    while i < len(a) and j < len(b):
        s, e = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if e > s:
            tot += e - s
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return tot


def main():
    path, mode, n, mhz = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
    ctx = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    ev = load(path)
    t0 = min(e["ts"] for e in ev)
    t1 = max(e["ts"] + e["dur"] for e in ev)
    span = t1 - t0
    is_comm = lambda e: (e.get("cat") == "gpu_memcpy" and ("DtoH" in e["name"] or "HtoD" in e["name"]
                                                            or "Device -> Pinned" in e["name"]
                                                            or "Pinned -> Device" in e["name"])) \
        or (e.get("cat") == "kernel" and any(c in e["name"] for c in COMM))
    comp = union([(e["ts"], e["ts"] + e["dur"]) for e in ev if e.get("cat") == "kernel" and not is_comm(e)])
    comm = union([(e["ts"], e["ts"] + e["dur"]) for e in ev if is_comm(e)])
    allv = union([(e["ts"], e["ts"] + e["dur"]) for e in ev])
    t_comp, t_all = length(comp), length(allv)
    t_comm_only = t_all - t_comp
    t_idle = span - t_all
    t_overlap = intersect_len(comp, comm)
    print(f"window {span / 1e3:9.1f} ms: compute {100 * t_comp / span:5.1f}% | waiting for communication "
          f"{100 * t_comm_only / span:5.1f}% | idle (waiting for the CPU) {100 * t_idle / span:5.1f}% "
          f"| communication hidden under compute {t_overlap / 1e3:.1f} ms")
    cat_t = {name: 0.0 for name, _ in CATS}
    for e in ev:
        if e.get("cat") != "kernel":
            continue
        for name, keys in CATS:
            if not keys or any(k in e["name"] for k in keys):
                cat_t[name] += e["dur"]
                break
    memcpy = sum(e["dur"] for e in ev if e.get("cat") == "gpu_memcpy")
    for name, t in cat_t.items():
        print(f"   {name:34s} {t / 1e3:9.1f} ms  {100 * t / span:5.1f}% of the window")
    print(f"   {'memcpy (DMA all-gather, staging)':34s} {memcpy / 1e3:9.1f} ms (copy engines, can overlap)")
    peak = SMS * INT8_OPS_PER_SM_CLK * mhz * 1e6
    if mode == "prefill":
        lin = 2 * LINEAR_PARAMS * n / 2 + 2 * MTP_PARAMS * n / 2          # per rank: half the weights
        att = 4 * Q_HEADS * HEAD_DIM * ATTN_LAYERS * n * n / 2            # causal, per rank (12 of 24 heads)
        flops = lin + att
        gemm = cat_t["int8 GEMM (prefill, CUTLASS)"]
        print(f"   work: {n} prompt tokens = {flops / 1e12:.1f} TFLOP on this GPU (linear {lin / 1e12:.1f}, "
              f"attention {att / 1e12:.1f})")
        print(f"   achieved {flops / (span * 1e-6) / 1e12:6.1f} TOPS = {100 * flops / (span * 1e-6) / peak:4.1f}% of "
              f"the int8 tensor-core peak {peak / 1e12:.0f} TOPS at {mhz:.0f} MHz (spec 215 TOPS at 1545 MHz); "
              f"while running the int8 GEMMs reach {lin / (gemm * 1e-6) / 1e12:6.1f} TOPS "
              f"({100 * lin / (gemm * 1e-6) / peak:4.1f}%)")
    else:
        steps = sum(1 for e in ev if e.get("cat") == "kernel" and "_rejection_kernel" in e["name"]) or 1
        nspec = 4
        weights = LINEAR_PARAMS / 2 + LM_HEAD / 2                          # int8 bytes read per verify, per rank
        draft = nspec * (MTP_PARAMS / 2 + 94e6)                            # MTP layer + 4-bit draft head per draft
        gdn = 48 * 1.5 * 2**20 * (1 + 1 + nspec)                           # fp32 states: 1 read + 5 writes
        kv = 16 * 2 * 2 * 256 * ctx * (1 + nspec)                          # FP8 KV read: verify + drafts
        per_step = weights + draft + gdn + kv
        step_ms = span / 1e3 / steps
        print(f"   work: {steps} steps, {n} tokens ({n / steps:.2f} tokens per step), {step_ms:.1f} ms per step, "
              f"{n / (span * 1e-6):.1f} tok/s")
        print(f"   memory traffic ~{per_step / 1e9:.1f} GB per step on this GPU (weights {weights / 1e9:.1f}, drafts "
              f"{draft / 1e9:.2f}, GDN states {gdn / 1e9:.2f}, KV {kv / 1e9:.2f}) = "
              f"{per_step * steps / (span * 1e-6) / 1e9:5.0f} GB/s = "
              f"{100 * per_step * steps / (span * 1e-6) / PEAK_BW:4.1f}% of 616 GB/s")
        flops = 2 * (LINEAR_PARAMS / 2) * (nspec + 1) * steps
        print(f"   compute: {flops / (span * 1e-6) / 1e12:5.1f} TOPS = {100 * flops / (span * 1e-6) / peak:4.1f}% of "
              f"the int8 peak (decode is bound by memory bandwidth)")


if __name__ == "__main__":
    main()
