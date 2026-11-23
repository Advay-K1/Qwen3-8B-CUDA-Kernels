"""Benchmarks: our kernels vs PyTorch/cuBLAS/SDPA, and decode throughput (layered vs megakernel).

python bench/bench.py [--decode] [--no-ops]
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from qwen3_kernels import ops  # noqa: E402

DEV, BF = "cuda", torch.bfloat16


def bench(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters  # ms


def gemm_bench(T=4096):
    print(f"\n== GEMM (T={T})  TFLOP/s  ours | cuBLAS")
    shapes = {"qkv": (6144, 4096), "o_proj": (4096, 4096), "gate_up": (24576, 4096), "down": (4096, 12288),
              "lm_head": (151936, 4096)}
    for name, (N, K) in shapes.items():
        x = torch.randn(T, K, device=DEV, dtype=BF)
        w = torch.randn(N, K, device=DEV, dtype=BF) * 0.02
        dy = torch.randn(T, N, device=DEV, dtype=BF)
        fl = 2 * T * N * K / 1e12
        row = []
        for tag, ours, ref in [
            ("fwd", lambda: ops.linear_fwd(x, w), lambda: x @ w.T),
            ("dx", lambda: ops.linear_dx(dy, w), lambda: dy @ w),
            ("dW", lambda: ops.linear_dw(dy, x), lambda: dy.T @ x),
        ]:
            a, b = bench(ours), bench(ref)
            row.append(f"{tag} {fl / a * 1e3:6.1f} | {fl / b * 1e3:6.1f}")
        print(f"  {name:8s} " + "   ".join(row))


def attn_bench(B=1, S=4096, Hq=32, Hkv=8):
    print(f"\n== Flash attention (B={B} S={S} Hq={Hq} Hkv={Hkv}, causal)  ms  ours | torch SDPA")
    q = torch.randn(B, Hq, S, 128, device=DEV, dtype=BF, requires_grad=True)
    k = torch.randn(B, Hkv, S, 128, device=DEV, dtype=BF, requires_grad=True)
    v = torch.randn(B, Hkv, S, 128, device=DEV, dtype=BF, requires_grad=True)
    do = torch.randn(B, S, Hq, 128, device=DEV, dtype=BF)
    do_t = do.transpose(1, 2).contiguous()
    fl_f = 4 * B * Hq * S * S * 128 / 2 / 1e12

    def ours_f():
        return ops.flash_attn(q, k, v)

    def ref_f():
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)

    o1 = ours_f()
    o2 = ref_f()
    tf, rf = bench(lambda: ours_f()), bench(lambda: ref_f())
    tb = bench(lambda: torch.autograd.grad(o1, (q, k, v), do, retain_graph=True))
    rb = bench(lambda: torch.autograd.grad(o2, (q, k, v), do_t, retain_graph=True))
    print(f"  fwd {tf:7.3f} | {rf:7.3f} ms   ({fl_f / tf * 1e3:5.1f} | {fl_f / rf * 1e3:5.1f} TFLOP/s)")
    print(f"  bwd {tb:7.3f} | {rb:7.3f} ms   ({2.5 * fl_f / tb * 1e3:5.1f} | {2.5 * fl_f / rb * 1e3:5.1f} TFLOP/s)")


def block_bench(B=1, S=4096):
    from qwen3_kernels import Qwen3Block, Qwen3Config

    print(f"\n== Full decoder layer fwd+bwd (B={B}, S={S}), random weights")
    cfg = Qwen3Config()
    blk = Qwen3Block(cfg)
    with torch.no_grad():
        for p in blk.parameters():
            if p.dim() == 2:
                p.normal_(0, 0.02)
    x = torch.randn(B * S, cfg.hidden, device=DEV, dtype=BF, requires_grad=True)
    pos = torch.arange(S, device=DEV, dtype=torch.int32).repeat(B)
    dy = torch.randn_like(x)
    tf = bench(lambda: blk(x, pos, B, S), iters=10)

    def fb():
        y = blk(x, pos, B, S)
        y.backward(dy)

    tfb = bench(fb, iters=10)
    flops = 2 * B * S * (cfg.qkv_dim * 4096 + 4096 * 4096 + 2 * 12288 * 4096 + 4096 * 12288)
    flops += 4 * B * 32 * S * S * 128 / 2
    print(f"  fwd {tf:.2f} ms ({flops / tf / 1e9:.1f} TFLOP/s)   fwd+bwd {tfb:.2f} ms ({3 * flops / tfb / 1e9:.1f} TFLOP/s)")


def decode_bench(ctx=512, n_tokens=64, B=1):
    from qwen3_kernels import Qwen3Model
    from qwen3_kernels.megakernel import MegaDecoder

    path = os.environ["QWEN3_PATH"]
    print(f"\n== Decode (Qwen3-8B real weights, B={B}, context {ctx})")
    m = Qwen3Model.from_pretrained(path)
    c = m.cfg
    L = len(m.layers)
    Smax = ctx + n_tokens + 8
    kc = torch.randn(L, B, c.n_kv_heads, Smax, 128, device=DEV, dtype=BF) * 0.5
    vc = torch.randn_like(kc)
    tok = torch.randint(0, c.vocab, (B,), device=DEV)
    e = ops.ext()

    def layered(pos):
        p = torch.full((B,), pos, dtype=torch.int32, device=DEV)
        lens = p + 1
        x = e.embedding_fwd(tok.to(torch.int32), m.embed)
        for i, blk in enumerate(m.layers):
            x = blk.decode(x, p, pos, kc[i], vc[i], lens)
        h, _ = e.rmsnorm_fwd(x, m.norm, c.eps)
        return e.gemv(m.lm_head, h, None, True)

    mk = MegaDecoder(m, B, Smax, kc, vc)
    t_l = bench(lambda: layered(ctx), iters=n_tokens // 4)
    t_m = bench(lambda: mk.step(tok, ctx), iters=n_tokens // 4)
    # CUDA graph of the layered path (removes launch overhead) for a fair comparison
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        layered(ctx)
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        layered(ctx)
    t_g = bench(lambda: g.replay(), iters=n_tokens // 4)
    nbytes = sum(p.numel() * 2 for p in m.parameters()) - m.embed.numel() * 2 + B * c.hidden * 2
    nbytes += L * 2 * B * c.n_kv_heads * (ctx + 1) * 128 * 2
    for name, t in [("layered kernels", t_l), ("layered + CUDA graph", t_g), ("megakernel", t_m)]:
        print(f"  {name:22s} {t:7.3f} ms/token  {B * 1e3 / t:7.1f} tok/s  {nbytes / t / 1e6:7.1f} GB/s effective")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--decode", action="store_true")
    ap.add_argument("--no-ops", action="store_true")
    ap.add_argument("--batch", type=int, default=1)
    a = ap.parse_args()
    print(torch.cuda.get_device_name())
    if not a.no_ops:
        gemm_bench()
        attn_bench()
        block_bench()
    if a.decode:
        decode_bench(B=a.batch)
