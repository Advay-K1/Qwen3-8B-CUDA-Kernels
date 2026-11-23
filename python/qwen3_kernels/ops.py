"""Autograd wrappers around the custom CUDA kernels.

Every op here runs our own kernels in both forward and backward; torch is only
used for allocation, views and bookkeeping.
"""
import torch

from ._ext import ext


# ----------------------------------------------------------------------------- GEMM primitives
def linear_fwd(x2, w, residual=None):
    """y[T,N] = x[T,K] @ w[N,K]^T (+ residual[T,N])."""
    T, K = x2.shape
    N = w.shape[0]
    y = torch.empty(T, N, dtype=x2.dtype, device=x2.device)
    ext().gemm(x2, w, y, T, N, K, K, K, N, True, True, 1.0, residual, N, False)
    return y


def linear_dx(dy2, w):
    """dx[T,K] = dy[T,N] @ w[N,K]."""
    T, N = dy2.shape
    K = w.shape[1]
    dx = torch.empty(T, K, dtype=dy2.dtype, device=dy2.device)
    ext().gemm(dy2, w, dx, T, K, N, N, K, K, True, False, 1.0, None, 0, False)
    return dx


def linear_dw(dy2, x2, out=None, accumulate=False, alpha=1.0):
    """dW[N,K] (+)= alpha * dy[T,N]^T @ x[T,K]. `out` may be bf16 or fp32."""
    T, N = dy2.shape
    K = x2.shape[1]
    if out is None:
        out = torch.empty(N, K, dtype=dy2.dtype, device=dy2.device)
    ext().gemm(dy2, x2, out, N, K, T, N, K, K, False, False, alpha, None, 0, accumulate)
    return out


class LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, residual=None):
        shp = x.shape
        x2 = x.reshape(-1, shp[-1]).contiguous()
        r2 = residual.reshape(-1, w.shape[0]).contiguous() if residual is not None else None
        y = linear_fwd(x2, w, r2)
        ctx.save_for_backward(x2, w)
        ctx.has_res = residual is not None
        ctx.shp = shp
        return y.view(*shp[:-1], w.shape[0])

    @staticmethod
    def backward(ctx, dy):
        x2, w = ctx.saved_tensors
        dy2 = dy.reshape(-1, w.shape[0]).contiguous()
        dx = linear_dx(dy2, w).view(ctx.shp) if ctx.needs_input_grad[0] else None
        dw = linear_dw(dy2, x2) if ctx.needs_input_grad[1] else None
        dr = dy if ctx.has_res else None
        return dx, dw, dr


def linear(x, w, residual=None):
    return LinearFn.apply(x, w, residual)


# ----------------------------------------------------------------------------- RMSNorm
class RMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, eps):
        x = x.contiguous()
        y, rstd = ext().rmsnorm_fwd(x, w, eps)
        ctx.save_for_backward(x, w, rstd)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w, rstd = ctx.saved_tensors
        dx, dw = ext().rmsnorm_bwd(dy.contiguous(), x, w, rstd)
        return dx, dw.to(w.dtype), None


class AddRMSNormFn(torch.autograd.Function):
    """s = x + residual; y = rmsnorm(s). Returns (y, s)."""

    @staticmethod
    def forward(ctx, x, residual, w, eps):
        y, rstd, s = ext().rmsnorm_fwd(x.contiguous(), w, eps, residual.contiguous())
        ctx.save_for_backward(s, w, rstd)
        return y, s

    @staticmethod
    def backward(ctx, dy, ds):
        s, w, rstd = ctx.saved_tensors
        if dy is None:
            dy = torch.zeros_like(s)
        dx, dw = ext().rmsnorm_bwd(dy.contiguous(), s, w, rstd, ds.contiguous() if ds is not None else None)
        return dx, dx, dw.to(w.dtype), None


def rmsnorm(x, w, eps=1e-6):
    return RMSNormFn.apply(x, w, eps)


def add_rmsnorm(x, residual, w, eps=1e-6):
    return AddRMSNormFn.apply(x, residual, w, eps)


# ----------------------------------------------------------------------------- Attention
class AttentionCoreFn(torch.autograd.Function):
    """qkv [B*S, (Hq+2Hkv)*D] -> attention output [B*S, Hq*D].

    Forward: fused q/k RMSNorm + RoPE + split (one kernel), then causal GQA flash attention.
    Backward: flash-attn bwd (dq in fp32), then fused norm/rope bwd straight into dqkv.
    """

    @staticmethod
    def forward(ctx, qkv, pos, wq, wk, B, S, Hq, Hkv, theta, eps):
        e = ext()
        qkv = qkv.contiguous()
        q, k, v, rq, rk = e.qk_norm_rope_fwd(qkv, pos, wq, wk, B, S, Hq, Hkv, theta, eps)
        scale = 128 ** -0.5
        o, lse = e.flash_fwd(q, k, v, scale)
        ctx.save_for_backward(qkv, pos, wq, wk, q, k, v, o, lse, rq, rk)
        ctx.cfg = (B, S, Hq, Hkv, theta, scale)
        return o.view(B * S, Hq * 128)

    @staticmethod
    def backward(ctx, do):
        qkv, pos, wq, wk, q, k, v, o, lse, rq, rk = ctx.saved_tensors
        B, S, Hq, Hkv, theta, scale = ctx.cfg
        e = ext()
        dq, dk, dv = e.flash_bwd(q, k, v, o, do.contiguous().view_as(o), lse, scale)
        dqkv, dwq, dwk = e.qk_norm_rope_bwd(dq, dk, dv, qkv, pos, wq, wk, rq, rk, B, S, Hq, Hkv, theta)
        return dqkv, None, dwq.to(wq.dtype), dwk.to(wk.dtype), None, None, None, None, None, None


def attention_core(qkv, pos, wq, wk, B, S, Hq=32, Hkv=8, theta=1e6, eps=1e-6):
    return AttentionCoreFn.apply(qkv, pos, wq, wk, B, S, Hq, Hkv, theta, eps)


class FlashAttnFn(torch.autograd.Function):
    """Standalone flash attention: q [B,Hq,S,D], k/v [B,Hkv,S,D] -> o [B,S,Hq,D]."""

    @staticmethod
    def forward(ctx, q, k, v, scale):
        o, lse = ext().flash_fwd(q.contiguous(), k.contiguous(), v.contiguous(), scale)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.scale = scale
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        dq, dk, dv = ext().flash_bwd(q, k, v, o, do.contiguous(), lse, ctx.scale)
        return dq.to(q.dtype), dk, dv, None


def flash_attn(q, k, v, scale=None):
    return FlashAttnFn.apply(q, k, v, scale if scale is not None else q.shape[-1] ** -0.5)


# ----------------------------------------------------------------------------- MLP
class SwiGLUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gu):
        gu = gu.contiguous()
        ctx.save_for_backward(gu)
        return ext().swiglu_fwd(gu)

    @staticmethod
    def backward(ctx, dh):
        (gu,) = ctx.saved_tensors
        return ext().swiglu_bwd(dh.contiguous(), gu)


def swiglu(gu):
    return SwiGLUFn.apply(gu)


# ----------------------------------------------------------------------------- Embedding
class EmbeddingFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ids, table):
        ids32 = ids.to(torch.int32).contiguous()
        ctx.save_for_backward(ids32)
        ctx.V = table.shape[0]
        ctx.dtype = table.dtype
        return ext().embedding_fwd(ids32, table)

    @staticmethod
    def backward(ctx, dout):
        (ids32,) = ctx.saved_tensors
        dt = ext().embedding_bwd(ids32, dout.contiguous(), ctx.V)
        return None, dt.to(ctx.dtype)


def embedding(ids, table):
    return EmbeddingFn.apply(ids, table)


# ----------------------------------------------------------------------------- LM head + CE
class LMHeadCrossEntropyFn(torch.autograd.Function):
    """mean CE loss of (h @ W^T) vs targets, without materializing all logits.

    Rows are processed in chunks: GEMM -> in-place CE (loss + dlogits) ->
    dh = dlogits @ W and dW += dlogits^T h. Gradients are produced in forward and
    scaled by grad_output in backward.
    """

    @staticmethod
    def forward(ctx, h, w, targets, ignore_index=-100, chunk=2048):
        e = ext()
        H = h.shape[-1]
        h2 = h.reshape(-1, H).contiguous()
        tg = targets.reshape(-1).to(torch.int32).contiguous()
        T, V = h2.shape[0], w.shape[0]
        n_valid = int((tg != ignore_index).sum().item())
        scale = 1.0 / max(n_valid, 1)
        need_dw = w.requires_grad
        dh = torch.empty_like(h2)
        dw = torch.zeros(V, H, dtype=torch.float32, device=h.device) if need_dw else None
        total = torch.zeros((), dtype=torch.float32, device=h.device)
        for s in range(0, T, chunk):
            hc = h2[s : s + chunk]
            logits = linear_fwd(hc, w)
            loss, _ = e.cross_entropy_(logits, tg[s : s + chunk].contiguous(), scale, ignore_index)
            total += loss.sum()
            dh[s : s + chunk] = linear_dx(logits, w)
            if need_dw:
                linear_dw(logits, hc, out=dw, accumulate=True)
        ctx.save_for_backward(dh, dw if need_dw else torch.empty(0))
        ctx.need_dw = need_dw
        ctx.hshape = h.shape
        return total * scale

    @staticmethod
    def backward(ctx, g):
        dh, dw = ctx.saved_tensors
        gs = float(g.item())
        dh_out = (dh * gs).view(ctx.hshape) if gs != 1.0 else dh.view(ctx.hshape)
        dw_out = ext().scale_cast(dw, gs) if ctx.need_dw else None
        return dh_out, dw_out, None, None, None


def lm_head_cross_entropy(h, w, targets, ignore_index=-100, chunk=2048):
    return LMHeadCrossEntropyFn.apply(h, w, targets, ignore_index, chunk)


# ----------------------------------------------------------------------------- Sampling / decode
def sample(logits, temperature=0.0, top_k=0, top_p=1.0, seed=0, offset=0):
    """logits [B, V] (any float dtype). Returns int64 token ids [B]."""
    B = logits.shape[0]
    dev = logits.device
    lg = logits.float().contiguous()

    def per_row(v, dt):
        if torch.is_tensor(v):
            return v.to(device=dev, dtype=dt).reshape(B).contiguous()
        return torch.full((B,), v, dtype=dt, device=dev)

    return ext().sample(lg, per_row(temperature, torch.float32), per_row(top_k, torch.int32),
                        per_row(top_p, torch.float32), int(seed), int(offset))


def gemv(w, x, residual=None, out_f32=False):
    return ext().gemv(w, x.contiguous(), residual, out_f32)


def decode_attn(q, k_cache, v_cache, lens, max_len, chunk=256, scale=128 ** -0.5):
    return ext().decode_attn(q.contiguous(), k_cache, v_cache, lens, max_len, chunk, scale)
