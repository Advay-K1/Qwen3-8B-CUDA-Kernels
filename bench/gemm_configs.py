"""Compare GEMM tile configs vs cuBLAS on Qwen3 shapes (fwd / dx / dW)."""
import os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.dirname(__file__))
from qwen3_kernels import ops
from bench import bench

T = int(os.environ.get("T", 4096))
for name, (N, K) in {"qkv": (6144, 4096), "o_proj": (4096, 4096), "gate_up": (24576, 4096), "down": (4096, 12288)}.items():
    x = torch.randn(T, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
    dy = torch.randn(T, N, device="cuda", dtype=torch.bfloat16)
    fl = 2 * T * N * K / 1e9
    out = []
    for tag, f, r in [("fwd", lambda: ops.linear_fwd(x, w), lambda: x @ w.T), ("dx", lambda: ops.linear_dx(dy, w), lambda: dy @ w),
                      ("dW", lambda: ops.linear_dw(dy, x), lambda: dy.T @ x)]:
        res = []
        for cfg in (0, 1):
            ops.ext().gemm_force_config(cfg)
            res.append(fl / bench(f))
        out.append(f"{tag} 128x128:{res[0]:5.0f} 128x256:{res[1]:5.0f} cuBLAS:{fl / bench(r):5.0f}")
    ops.ext().gemm_force_config(-1)
    print(f"{name:8s}", " | ".join(out))
