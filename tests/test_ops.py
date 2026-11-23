"""Correctness tests: every kernel (fwd + bwd) vs an fp32 PyTorch reference.

Run:  python tests/test_ops.py        (or pytest tests/test_ops.py)
"""
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from qwen3_kernels import ops  # noqa: E402
from qwen3_kernels._ext import ext  # noqa: E402

torch.manual_seed(0)
DEV = "cuda"
BF = torch.bfloat16
RESULTS = []


def check(name, out, ref, tol=2e-2):
    out, ref = out.float(), ref.float()
    err = (out - ref).abs().max().item()
    scale = ref.abs().max().item() + 1e-6
    cos = F.cosine_similarity(out.flatten(), ref.flatten(), dim=0).item()
    ok = (err / scale < tol) and cos > 0.999 and math.isfinite(err)
    RESULTS.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name:42s} rel_max_err={err / scale:.2e} cos={cos:.6f}")
    return ok


def rand(*s, scale=1.0):
    return (torch.randn(*s, device=DEV) * scale).to(BF)


# ----------------------------------------------------------------------------- GEMM
def test_gemm():
    for (T, N, K) in [(256, 512, 256), (77, 6144, 4096), (1000, 4096, 12288), (129, 136, 72)]:
        x, w, dy = rand(T, K), rand(N, K, scale=0.02), rand(T, N)
        r = rand(T, N)
        check(f"gemm fwd   T={T} N={N} K={K}", ops.linear_fwd(x, w), x.float() @ w.float().T)
        check(f"gemm fwd+res T={T} N={N}", ops.linear_fwd(x, w, r), x.float() @ w.float().T + r.float())
        check(f"gemm dx    T={T} N={N} K={K}", ops.linear_dx(dy, w), dy.float() @ w.float())
        check(f"gemm dW    T={T} N={N} K={K}", ops.linear_dw(dy, x), dy.float().T @ x.float())
        acc = torch.randn(N, K, device=DEV)
        ref = acc + dy.float().T @ x.float()
        ops.linear_dw(dy, x, out=acc, accumulate=True)
        check(f"gemm dW fp32 accumulate T={T}", acc, ref)


# ----------------------------------------------------------------------------- RMSNorm
def ref_rmsnorm(x, w, eps=1e-6):
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return w.float() * y


def test_rmsnorm():
    for T, H in [(37, 4096), (512, 4096), (8, 128)]:
        x = rand(T, H)
        w = (1 + 0.1 * torch.randn(H, device=DEV)).to(BF)
        dy = rand(T, H)
        xr = x.float().requires_grad_()
        wr = w.float().requires_grad_()
        yr = ref_rmsnorm(xr, wr)
        yr.backward(dy.float())
        xc, wc = x.clone().requires_grad_(), w.clone().requires_grad_()
        y = ops.rmsnorm(xc, wc)
        y.backward(dy)
        check(f"rmsnorm fwd T={T} H={H}", y, yr)
        check(f"rmsnorm dx  T={T} H={H}", xc.grad, xr.grad)
        check(f"rmsnorm dw  T={T} H={H}", wc.grad, wr.grad)
    # fused add
    T, H = 300, 4096
    x, res, dy, ds = rand(T, H), rand(T, H), rand(T, H), rand(T, H)
    w = (1 + 0.1 * torch.randn(H, device=DEV)).to(BF)
    xr, rr, wr = x.float().requires_grad_(), res.float().requires_grad_(), w.float().requires_grad_()
    sr = xr + rr
    yr = ref_rmsnorm(sr, wr)
    (yr * dy.float()).sum().add_((sr * ds.float()).sum()).backward()
    xc, rc, wc = x.clone().requires_grad_(), res.clone().requires_grad_(), w.clone().requires_grad_()
    y, s = ops.add_rmsnorm(xc, rc, wc)
    torch.autograd.backward([y, s], [dy, ds])
    check("add_rmsnorm y", y, yr)
    check("add_rmsnorm s", s, sr)
    check("add_rmsnorm dx", xc.grad, xr.grad)
    check("add_rmsnorm dres", rc.grad, rr.grad)
    check("add_rmsnorm dw", wc.grad, wr.grad)


# ----------------------------------------------------------------------------- Attention
def ref_rope(x, pos, theta):  # x [B, H, S, D] float, pos [S]
    D = x.shape[-1]
    inv = 1.0 / (theta ** (torch.arange(0, D, 2, device=x.device, dtype=torch.float) / D))
    ang = pos.float()[:, None] * inv[None]
    emb = torch.cat([ang, ang], -1)
    cos, sin = emb.cos(), emb.sin()
    x1, x2 = x[..., : D // 2], x[..., D // 2:]
    rot = torch.cat([-x2, x1], -1)
    return x * cos + rot * sin


def ref_attention_core(qkv, wq, wk, B, S, Hq, Hkv, theta=1e6, eps=1e-6):
    D = 128
    x = qkv.view(B, S, Hq + 2 * Hkv, D)
    q, k, v = x[:, :, :Hq], x[:, :, Hq:Hq + Hkv], x[:, :, Hq + Hkv:]
    q = ref_rmsnorm(q, wq, eps).transpose(1, 2)
    k = ref_rmsnorm(k, wk, eps).transpose(1, 2)
    v = v.transpose(1, 2)
    pos = torch.arange(S, device=qkv.device)
    q, k = ref_rope(q, pos, theta), ref_rope(k, pos, theta)
    rep = Hq // Hkv
    k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
    o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return o.transpose(1, 2).reshape(B * S, Hq * D)


def test_flash_attention():
    for B, S, Hq, Hkv in [(1, 64, 4, 1), (2, 200, 8, 2), (1, 1024, 32, 8), (2, 333, 32, 8)]:
        q, k, v = rand(B, Hq, S, 128), rand(B, Hkv, S, 128), rand(B, Hkv, S, 128)
        do = rand(B, S, Hq, 128)
        qr, kr, vr = (t.float().requires_grad_() for t in (q, k, v))
        rep = Hq // Hkv
        orf = F.scaled_dot_product_attention(qr, kr.repeat_interleave(rep, 1), vr.repeat_interleave(rep, 1),
                                             is_causal=True).transpose(1, 2)
        orf.backward(do.float())
        qc, kc, vc = (t.clone().requires_grad_() for t in (q, k, v))
        o = ops.flash_attn(qc, kc, vc)
        o.backward(do)
        tag = f"B={B} S={S} Hq={Hq} Hkv={Hkv}"
        check(f"flash fwd {tag}", o, orf)
        check(f"flash dq  {tag}", qc.grad, qr.grad)
        check(f"flash dk  {tag}", kc.grad, kr.grad)
        check(f"flash dv  {tag}", vc.grad, vr.grad)


def test_attention_core():
    for B, S in [(1, 128), (2, 257)]:
        Hq, Hkv = 32, 8
        qkv = rand(B * S, (Hq + 2 * Hkv) * 128)
        wq = (1 + 0.2 * torch.randn(128, device=DEV)).to(BF)
        wk = (1 + 0.2 * torch.randn(128, device=DEV)).to(BF)
        do = rand(B * S, Hq * 128)
        pos = torch.arange(S, device=DEV, dtype=torch.int32).repeat(B)
        xr, wqr, wkr = qkv.float().requires_grad_(), wq.float().requires_grad_(), wk.float().requires_grad_()
        orf = ref_attention_core(xr, wqr, wkr, B, S, Hq, Hkv)
        orf.backward(do.float())
        xc, wqc, wkc = qkv.clone().requires_grad_(), wq.clone().requires_grad_(), wk.clone().requires_grad_()
        o = ops.attention_core(xc, pos, wqc, wkc, B, S, Hq, Hkv)
        o.backward(do)
        tag = f"B={B} S={S}"
        check(f"qknorm+rope+attn fwd {tag}", o, orf)
        check(f"qknorm+rope+attn dqkv {tag}", xc.grad, xr.grad)
        check(f"qknorm+rope+attn dwq {tag}", wqc.grad, wqr.grad)
        check(f"qknorm+rope+attn dwk {tag}", wkc.grad, wkr.grad)


# ----------------------------------------------------------------------------- MLP pieces
def test_swiglu():
    T, I = 333, 12288
    gu, dh = rand(T, 2 * I), rand(T, I)
    gr = gu.float().requires_grad_()
    hr = F.silu(gr[:, :I]) * gr[:, I:]
    hr.backward(dh.float())
    gc = gu.clone().requires_grad_()
    h = ops.swiglu(gc)
    h.backward(dh)
    check("swiglu fwd", h, hr)
    check("swiglu bwd", gc.grad, gr.grad)


def test_embedding():
    V, H, T = 1000, 4096, 777
    table = rand(V, H)
    ids = torch.randint(0, V, (T,), device=DEV)
    dout = rand(T, H)
    tr = table.float().requires_grad_()
    out_r = F.embedding(ids, tr)
    out_r.backward(dout.float())
    tc = table.clone().requires_grad_()
    out = ops.embedding(ids, tc)
    out.backward(dout)
    check("embedding fwd", out, out_r)
    check("embedding bwd", tc.grad, tr.grad)


def test_lm_head_ce():
    T, H, V = 700, 4096, 151936
    h = rand(T, H)
    w = rand(V, H, scale=0.02)
    tg = torch.randint(0, V, (T,), device=DEV)
    tg[::7] = -100
    hr, wr = h.float().requires_grad_(), w.float().requires_grad_()
    loss_r = F.cross_entropy(hr @ wr.T, tg, ignore_index=-100)
    (loss_r * 2.0).backward()
    hc, wc = h.clone().requires_grad_(), w.clone().requires_grad_()
    loss = ops.lm_head_cross_entropy(hc, wc, tg, chunk=256)
    (loss * 2.0).backward()
    check("lm_head+CE loss", loss.view(1), loss_r.view(1), tol=5e-3)
    check("lm_head+CE dh", hc.grad, hr.grad)
    check("lm_head+CE dW", wc.grad, wr.grad)


# ----------------------------------------------------------------------------- Sampling
def test_sampling():
    B, V = 4, 151936
    logits = torch.randn(B, V, device=DEV) * 3
    ids = ops.sample(logits, temperature=0.0)
    ok = bool((ids == logits.argmax(-1)).all())
    ids = ops.sample(logits, temperature=1.0, top_k=1, seed=3)
    ok &= bool((ids == logits.argmax(-1)).all())
    RESULTS.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] greedy / top_k=1 == argmax")

    # Distribution test on a small vocab, many draws (rows = independent draws).
    Vs, N = 64, 20000
    base = torch.randn(Vs, device=DEV) * 1.5
    for (temp, k, p) in [(1.0, 0, 1.0), (0.7, 10, 1.0), (1.0, 0, 0.8), (1.3, 20, 0.9)]:
        lg = base.repeat(N, 1).contiguous()
        draws = ops.sample(lg, temp, k, p, seed=1234)
        emp = torch.bincount(draws, minlength=Vs).float() / N
        # expected
        pr = torch.softmax(base / temp, -1)
        keep = torch.ones(Vs, dtype=torch.bool, device=DEV)
        if k > 0:
            keep &= pr >= pr.topk(k).values[-1]
        if p < 1.0:
            sp, si = (pr * keep).sort(descending=True)
            csum = sp.cumsum(0) / (pr * keep).sum()
            nkeep = int((csum < p).sum().item()) + 1
            kp = torch.zeros_like(keep)
            kp[si[:nkeep]] = True
            keep &= kp
        exp = pr * keep
        exp /= exp.sum()
        tv = 0.5 * (emp - exp).abs().sum().item()
        outside = emp[~keep].sum().item()
        ok = tv < 0.03 and outside == 0.0
        RESULTS.append(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] sampling T={temp} k={k} p={p}: TV={tv:.4f} mass_outside={outside}")


# ----------------------------------------------------------------------------- Decode kernels
def test_decode():
    K, N = 4096, 6144
    w = rand(N, K, scale=0.02)
    for nb in (1, 3, 8):
        x = rand(nb, K)
        r = rand(nb, N)
        check(f"gemv NB={nb}", ops.gemv(w, x, out_f32=True), x.float() @ w.float().T)
        check(f"gemv NB={nb} +res bf16", ops.gemv(w, x, r), x.float() @ w.float().T + r.float())
    B, Hq, Hkv, Smax = 2, 32, 8, 3000
    for L in (1, 100, 2777):
        kc, vc = rand(B, Hkv, Smax, 128), rand(B, Hkv, Smax, 128)
        q = rand(B, Hq, 128)
        lens = torch.full((B,), L, dtype=torch.int32, device=DEV)
        out = ops.decode_attn(q, kc, vc, lens, L)
        kr = kc[:, :, :L].float().repeat_interleave(4, 1)
        vr = vc[:, :, :L].float().repeat_interleave(4, 1)
        ref = F.scaled_dot_product_attention(q.float()[:, :, None], kr, vr)[:, :, 0].reshape(B, -1)
        check(f"decode attn L={L}", out, ref)


if __name__ == "__main__":
    ext()
    for fn in [test_gemm, test_rmsnorm, test_flash_attention, test_attention_core, test_swiglu, test_embedding,
               test_lm_head_ce, test_sampling, test_decode]:
        print(f"--- {fn.__name__}")
        fn()
    n_fail = RESULTS.count(False)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    sys.exit(1 if n_fail else 0)
