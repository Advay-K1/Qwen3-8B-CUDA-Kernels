"""Roofline report: every kernel vs min(peak_flops, achievable_bw * arithmetic_intensity).

Peaks are *measured* on the current GPU:
  - achievable HBM bandwidth: best of a large streaming reduction (read) and copy (read+write)
  - achievable bf16 tensor-core throughput: cuBLAS on an 8192^3 GEMM (also prints nominal)

python bench/roofline.py [--decode]
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.dirname(__file__))
from bench import bench  # noqa: E402
from qwen3_kernels import ops  # noqa: E402

DEV, BF = "cuda", torch.bfloat16
ROWS = []


def measure_peaks():
    x = torch.empty(2 * 1024**3 // 2, device=DEV, dtype=BF).normal_()
    t_sum = bench(lambda: x.sum(dtype=torch.float32), iters=20)
    y = torch.empty_like(x)
    t_cp = bench(lambda: y.copy_(x), iters=20)
    bw = max(x.numel() * 2 / t_sum, 2 * x.numel() * 2 / t_cp) / 1e6  # bytes/ms -> GB/s
    del x, y
    a = torch.randn(8192, 8192, device=DEV, dtype=BF)
    b = torch.randn(8192, 8192, device=DEV, dtype=BF)
    t = bench(lambda: a @ b, iters=10)
    tf = 2 * 8192**3 / t / 1e9
    return bw, tf  # GB/s, TFLOP/s


def row(name, ms, flops, nbytes, bw, tf):
    ach_tf = flops / ms / 1e9
    ach_bw = nbytes / ms / 1e6
    roof_ms = max(flops / (tf * 1e9), nbytes / (bw * 1e6))
    bound = "compute" if flops / (tf * 1e9) >= nbytes / (bw * 1e6) else "memory"
    pct = 100 * roof_ms / ms
    ROWS.append((name, ms, ach_tf, ach_bw, bound, pct))
    print(f"  {name:38s} {ms:8.3f} ms  {ach_tf:7.1f} TF/s  {ach_bw:7.0f} GB/s  [{bound:7s}] {pct:5.1f}% of roofline",
          flush=True)


def kernels(bw, tf, T=4096):
    print("\n== GEMMs (T=4096)")
    for name, (N, K) in {"qkv": (6144, 4096), "o_proj": (4096, 4096), "gate_up": (24576, 4096),
                         "down": (4096, 12288), "lm_head": (151936, 4096)}.items():
        x = torch.randn(T, K, device=DEV, dtype=BF)
        w = torch.randn(N, K, device=DEV, dtype=BF) * 0.02
        dy = torch.randn(T, N, device=DEV, dtype=BF)
        fl = 2 * T * N * K
        by = 2 * (T * K + N * K + T * N)
        row(f"{name} fwd", bench(lambda: ops.linear_fwd(x, w)), fl, by, bw, tf)
        row(f"{name} dx", bench(lambda: ops.linear_dx(dy, w)), fl, by, bw, tf)
        row(f"{name} dW", bench(lambda: ops.linear_dw(dy, x)), fl, by, bw, tf)
        row(f"{name} fwd [cuBLAS]", bench(lambda: x @ w.T), fl, by, bw, tf)
        del x, w, dy

    print("\n== Attention (B=1, S=4096, Hq=32, Hkv=8, causal)")
    B, S = 1, 4096
    q = torch.randn(B, 32, S, 128, device=DEV, dtype=BF, requires_grad=True)
    k = torch.randn(B, 8, S, 128, device=DEV, dtype=BF, requires_grad=True)
    v = torch.randn(B, 8, S, 128, device=DEV, dtype=BF, requires_grad=True)
    e = ops.ext()
    fl = 4 * B * 32 * S * S * 128 / 2
    by = 2 * (q.numel() * 2 + 2 * k.numel())
    o, lse = e.flash_fwd(q.detach(), k.detach(), v.detach(), 128 ** -0.5)
    do = torch.randn_like(o)
    row("flash fwd", bench(lambda: e.flash_fwd(q.detach(), k.detach(), v.detach(), 128 ** -0.5)), fl, by, bw, tf)
    row("flash bwd (incl. pre)", bench(lambda: e.flash_bwd(q.detach(), k.detach(), v.detach(), o, do, lse,
                                                            128 ** -0.5)), 2.5 * fl, 3 * by, bw, tf)
    row("flash fwd [SDPA]", bench(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)),
        fl, by, bw, tf)
    o2 = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    do2 = torch.randn_like(o2)
    row("flash bwd [SDPA]", bench(lambda: torch.autograd.grad(o2, (q, k, v), do2, retain_graph=True)), 2.5 * fl,
        3 * by, bw, tf)

    print("\n== Memory-bound kernels (T=4096)")
    H, I = 4096, 12288
    x = torch.randn(T, H, device=DEV, dtype=BF)
    r = torch.randn(T, H, device=DEV, dtype=BF)
    w = torch.ones(H, device=DEV, dtype=BF)
    y, rstd = e.rmsnorm_fwd(x, w, 1e-6)
    row("rmsnorm fwd", bench(lambda: e.rmsnorm_fwd(x, w, 1e-6)), 0, 2 * T * H * 2, bw, tf)
    row("add+rmsnorm fwd", bench(lambda: e.rmsnorm_fwd(x, w, 1e-6, r)), 0, 4 * T * H * 2, bw, tf)
    row("rmsnorm bwd", bench(lambda: e.rmsnorm_bwd(y, x, w, rstd)), 0, 3 * T * H * 2, bw, tf)
    gu = torch.randn(T, 2 * I, device=DEV, dtype=BF)
    h = e.swiglu_fwd(gu)
    row("swiglu fwd", bench(lambda: e.swiglu_fwd(gu)), 0, 3 * T * I * 2, bw, tf)
    row("swiglu bwd", bench(lambda: e.swiglu_bwd(h, gu)), 0, 5 * T * I * 2, bw, tf)
    qkv = torch.randn(T, 6144, device=DEV, dtype=BF)
    pos = torch.arange(T, device=DEV, dtype=torch.int32)
    wq = torch.ones(128, device=DEV, dtype=BF)
    row("qk-norm+rope fwd", bench(lambda: e.qk_norm_rope_fwd(qkv, pos, wq, wq, 1, T, 32, 8, 1e6, 1e-6)), 0,
        2 * T * 6144 * 2, bw, tf)
    qq, kk, vv, rq, rk = e.qk_norm_rope_fwd(qkv, pos, wq, wq, 1, T, 32, 8, 1e6, 1e-6)
    dqf = torch.randn(1, 32, T, 128, device=DEV)
    row("qk-norm+rope bwd", bench(lambda: e.qk_norm_rope_bwd(dqf, kk, vv, qkv, pos, wq, wq, rq, rk, 1, T, 32, 8, 1e6)),
        0, T * 4096 * 4 + T * 2048 * 2 + 2 * T * 6144 * 2, bw, tf)
    V = 151936
    logits = torch.randn(1024, V, device=DEV, dtype=BF)
    tgt = torch.randint(0, V, (1024,), device=DEV, dtype=torch.int32)
    row("cross-entropy (1024 x 151936, in place)", bench(lambda: e.cross_entropy_(logits, tgt, 1.0, -100)), 0,
        3 * 1024 * V * 2, bw, tf)
    lg = torch.randn(8, V, device=DEV)
    tmp = torch.ones(8, device=DEV)
    row("sample top-k=50 top-p=0.9 (B=8)", bench(lambda: e.sample(lg, tmp, torch.full((8,), 50, device=DEV,
        dtype=torch.int32), tmp * 0.9, 0, 0)), 0, 8 * V * 4, bw, tf)


def decode(bw, tf):
    from qwen3_kernels import Qwen3Model
    from qwen3_kernels.megakernel import MegaDecoder

    print("\n== Decode step, Qwen3-8B (memory-bound: weights + KV cache per token)")
    m = Qwen3Model.from_pretrained(os.environ["QWEN3_PATH"])
    c = m.cfg
    wbytes = sum(p.numel() * 2 for p in m.parameters()) - m.embed.numel() * 2
    for B in (1, 4):
        for ctx in (512, 4096):
            Smax = ctx + 8
            kc = torch.randn(36, B, 8, Smax, 128, device=DEV, dtype=BF) * 0.5
            vc = torch.randn_like(kc)
            tok = torch.randint(0, c.vocab, (B,), device=DEV)
            mk = MegaDecoder(m, B, Smax, kc, vc)
            nbytes = wbytes + 36 * 2 * B * 8 * (ctx + 1) * 128 * 2
            row(f"megakernel B={B} ctx={ctx}", bench(lambda: mk.step(tok, ctx), iters=20),
                2 * B * wbytes / 2, nbytes, bw, tf)
            del kc, vc, mk


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--decode", action="store_true")
    ap.add_argument("--only-decode", action="store_true")
    a = ap.parse_args()
    bw, tf = measure_peaks()
    print(f"{torch.cuda.get_device_name()}: achievable HBM {bw:.0f} GB/s, achievable bf16 MMA (cuBLAS 8192^3) {tf:.1f} TF/s")
    if not a.only_decode:
        kernels(bw, tf)
    if a.decode or a.only_decode:
        decode(bw, tf)
