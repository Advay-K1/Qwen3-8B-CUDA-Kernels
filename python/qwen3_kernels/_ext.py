"""JIT-builds and loads the CUDA extension (cached in TORCH_EXTENSIONS_DIR)."""
import glob
import os

import torch
from torch.utils.cpp_extension import load

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CSRC = os.path.join(_ROOT, "csrc")
_MEGA = os.path.join(_ROOT, "megakernel")
_mod = None
_mega = None

_CUDA_FLAGS = ["-O3", "-std=c++17", "-lineinfo", "--expt-relaxed-constexpr", "-Xptxas=-v" if os.environ.get("QK_PTXAS_V") else "-DQK=1"]


def _arch_flags():
    if os.environ.get("TORCH_CUDA_ARCH_LIST"):
        return []
    major, minor = torch.cuda.get_device_capability()
    return [f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"]


def ext():
    global _mod
    if _mod is None:
        srcs = sorted(glob.glob(os.path.join(_CSRC, "*.cu"))) + [os.path.join(_CSRC, "bindings.cpp")]
        _mod = load(
            name="qwen3_kernels_C",
            sources=srcs,
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=_CUDA_FLAGS + _arch_flags(),
            verbose=bool(os.environ.get("QK_VERBOSE")),
        )
    return _mod


def mega():
    global _mega
    if _mega is None:
        srcs = sorted(glob.glob(os.path.join(_MEGA, "*.cu"))) + [os.path.join(_MEGA, "bindings.cpp")]
        _mega = load(
            name="qwen3_megakernel_C",
            sources=srcs,
            extra_include_paths=[_CSRC],
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=_CUDA_FLAGS + _arch_flags(),
            verbose=bool(os.environ.get("QK_VERBOSE")),
        )
    return _mega
