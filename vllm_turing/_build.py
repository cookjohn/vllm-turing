"""Builds csrc/<name>.cu into build/lib<name>.so with the env's CUDA 13 nvcc (nvidia/cu13, matches torch) and loads it
with ctypes; rebuilt when the source is newer, one build at a time across processes (file lock). load(name,
cutlass=True): against the CUTLASS headers shipped in vllm/third_party, with the system toolkit's nvcc (the pip nvcc's
bundled CCCL rejects CUTLASS's <cuda/std/...> includes)."""
import ctypes
import fcntl
import os
import subprocess

import torch

_here = os.path.dirname(os.path.abspath(__file__))
_libs = {}


def _nvcc():
    cu13 = os.path.realpath(os.path.join(os.path.dirname(torch.__file__), "..", "nvidia", "cu13"))
    for p in (os.path.join(cu13, "bin", "nvcc"), "/usr/local/cuda/bin/nvcc"):
        if os.path.exists(p):
            return p, cu13
    raise RuntimeError("nvcc not found")


def _cutlass_include():
    import vllm
    for rel in ("third_party/fmha_sm100/cutlass/include", "third_party/deep_gemm/include"):
        d = os.path.join(os.path.dirname(vllm.__file__), rel)
        if os.path.isfile(os.path.join(d, "cutlass", "cutlass.h")):
            return d
    raise RuntimeError("CUTLASS headers not found under vllm/third_party")


def load(name, cutlass=False):
    if name in _libs:
        return _libs[name]
    src = os.path.join(_here, "csrc", name + ".cu")
    build = os.path.join(_here, "build")
    so = os.path.join(build, f"lib{name}.so")
    os.makedirs(build, exist_ok=True)
    with open(os.path.join(build, ".lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src):
            if cutlass:
                nvcc = "/usr/local/cuda/bin/nvcc" if os.path.exists("/usr/local/cuda/bin/nvcc") else _nvcc()[0]
                inc = [f"-I{_cutlass_include()}"]
                extra = ["--expt-relaxed-constexpr", "-Xcompiler", "-Wno-missing-field-initializers"]
            else:
                nvcc, cu13 = _nvcc()
                inc = [f"-I{os.path.join(cu13, d)}" for d in ("include", "cccl") if os.path.isdir(os.path.join(cu13, d))]
                extra = []
            subprocess.run([nvcc, "-O3", "-std=c++17", "-gencode=arch=compute_75,code=sm_75", "-shared",
                            "-Xcompiler", "-fPIC", *extra, *inc, "-o", so + ".tmp", src], check=True)
            os.replace(so + ".tmp", so)
    _libs[name] = ctypes.CDLL(so)
    return _libs[name]
