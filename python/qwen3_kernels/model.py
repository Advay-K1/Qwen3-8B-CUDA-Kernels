"""Qwen3-8B built entirely on the custom kernels (training fwd/bwd + KV-cache generation)."""
import json
import math
import os
from dataclasses import dataclass

import torch
import torch.nn as nn

from . import ops
from ._ext import ext


@dataclass
class Qwen3Config:
    hidden: int = 4096
    intermediate: int = 12288
    n_layers: int = 36
    n_heads: int = 32
    n_kv_heads: int = 8
    head_dim: int = 128
    vocab: int = 151936
    rope_theta: float = 1e6
    eps: float = 1e-6

    @classmethod
    def from_hf(cls, path):
        with open(os.path.join(path, "config.json")) as f:
            c = json.load(f)
        assert c["head_dim"] == 128, "kernels assume head_dim == 128"
        return cls(c["hidden_size"], c["intermediate_size"], c["num_hidden_layers"], c["num_attention_heads"],
                   c["num_key_value_heads"], c["head_dim"], c["vocab_size"], c["rope_theta"], c["rms_norm_eps"])

    @property
    def qkv_dim(self):
        return (self.n_heads + 2 * self.n_kv_heads) * self.head_dim


class Qwen3Block(nn.Module):
    """One decoder layer. QKV and gate/up projections are fused into single weights."""

    def __init__(self, cfg: Qwen3Config, device="cuda", dtype=torch.bfloat16):
        super().__init__()
        self.cfg = cfg
        H, I = cfg.hidden, cfg.intermediate
        kw = dict(device=device, dtype=dtype)
        self.input_ln = nn.Parameter(torch.ones(H, **kw))
        self.qkv_w = nn.Parameter(torch.empty(cfg.qkv_dim, H, **kw))
        self.q_norm = nn.Parameter(torch.ones(cfg.head_dim, **kw))
        self.k_norm = nn.Parameter(torch.ones(cfg.head_dim, **kw))
        self.o_w = nn.Parameter(torch.empty(H, cfg.n_heads * cfg.head_dim, **kw))
        self.post_ln = nn.Parameter(torch.ones(H, **kw))
        self.gate_up_w = nn.Parameter(torch.empty(2 * I, H, **kw))
        self.down_w = nn.Parameter(torch.empty(H, I, **kw))

    @torch.no_grad()
    def load_hf(self, sd, prefix):
        g = lambda n: sd[prefix + n]
        self.input_ln.copy_(g("input_layernorm.weight"))
        self.qkv_w.copy_(torch.cat([g("self_attn.q_proj.weight"), g("self_attn.k_proj.weight"),
                                    g("self_attn.v_proj.weight")], 0))
        self.q_norm.copy_(g("self_attn.q_norm.weight"))
        self.k_norm.copy_(g("self_attn.k_norm.weight"))
        self.o_w.copy_(g("self_attn.o_proj.weight"))
        self.post_ln.copy_(g("post_attention_layernorm.weight"))
        self.gate_up_w.copy_(torch.cat([g("mlp.gate_proj.weight"), g("mlp.up_proj.weight")], 0))
        self.down_w.copy_(g("mlp.down_proj.weight"))

    def forward(self, x, pos, B, S):
        """x [B*S, H] bf16, pos int32 [B*S]. Differentiable through our kernels."""
        c = self.cfg
        h = ops.rmsnorm(x, self.input_ln, c.eps)
        qkv = ops.linear(h, self.qkv_w)
        a = ops.attention_core(qkv, pos, self.q_norm, self.k_norm, B, S, c.n_heads, c.n_kv_heads, c.rope_theta, c.eps)
        x = ops.linear(a, self.o_w, residual=x)  # o_proj + residual fused in the GEMM epilogue
        h = ops.rmsnorm(x, self.post_ln, c.eps)
        h = ops.swiglu(ops.linear(h, self.gate_up_w))
        return ops.linear(h, self.down_w, residual=x)  # down_proj + residual

    # ------------------------------------------------------------ inference (no autograd)
    @torch.no_grad()
    def prefill(self, x, pos, B, S, k_cache, v_cache):
        """Prompt pass that also writes K/V into the caches [B, Hkv, Smax, D] at positions [0, S)."""
        c, e = self.cfg, ext()
        h, _ = e.rmsnorm_fwd(x, self.input_ln, c.eps)
        qkv = ops.linear_fwd(h, self.qkv_w)
        q, k, v = e.qk_norm_rope_fwd(qkv, pos, self.q_norm, self.k_norm, B, S, c.n_heads, c.n_kv_heads,
                                     c.rope_theta, c.eps, k_cache, v_cache, 0, False)
        (o,) = e.flash_fwd(q, k_cache, v_cache, c.head_dim ** -0.5, False)
        x = ops.linear_fwd(o.view(B * S, -1), self.o_w, x)
        h, _ = e.rmsnorm_fwd(x, self.post_ln, c.eps)
        h = e.swiglu_fwd(ops.linear_fwd(h, self.gate_up_w))
        return ops.linear_fwd(h, self.down_w, x)

    @torch.no_grad()
    def decode(self, x, pos, cur, k_cache, v_cache, lens):
        """One token per sequence: x [B, H], writes K/V at position `cur` (same for all rows)."""
        c, e = self.cfg, ext()
        B = x.shape[0]
        h, _ = e.rmsnorm_fwd(x, self.input_ln, c.eps)
        qkv = e.gemv(self.qkv_w, h, None, False)
        q, _, _ = e.qk_norm_rope_fwd(qkv, pos, self.q_norm, self.k_norm, B, 1, c.n_heads, c.n_kv_heads,
                                     c.rope_theta, c.eps, k_cache, v_cache, cur, False)
        a = e.decode_attn(q, k_cache, v_cache, lens, cur + 1, 256, c.head_dim ** -0.5)
        x = e.gemv(self.o_w, a, x, False)
        h, _ = e.rmsnorm_fwd(x, self.post_ln, c.eps)
        h = e.swiglu_fwd(e.gemv(self.gate_up_w, h, None, False))
        return e.gemv(self.down_w, h, x, False)


def load_state_dict(path, keys=None, device="cuda"):
    from safetensors import safe_open

    with open(os.path.join(path, "model.safetensors.index.json")) as f:
        wmap = json.load(f)["weight_map"]
    want = list(wmap) if keys is None else [k for k in wmap if any(k.startswith(p) for p in keys)]
    by_file = {}
    for k in want:
        by_file.setdefault(wmap[k], []).append(k)
    sd = {}
    for fn, ks in by_file.items():
        with safe_open(os.path.join(path, fn), framework="pt", device=str(device)) as f:
            for k in ks:
                sd[k] = f.get_tensor(k)
    return sd


class Qwen3Model(nn.Module):
    def __init__(self, cfg: Qwen3Config, n_layers=None, device="cuda", dtype=torch.bfloat16):
        super().__init__()
        self.cfg = cfg
        L = cfg.n_layers if n_layers is None else n_layers
        kw = dict(device=device, dtype=dtype)
        self.embed = nn.Parameter(torch.empty(cfg.vocab, cfg.hidden, **kw))
        self.layers = nn.ModuleList([Qwen3Block(cfg, device, dtype) for _ in range(L)])
        self.norm = nn.Parameter(torch.ones(cfg.hidden, **kw))
        self.lm_head = nn.Parameter(torch.empty(cfg.vocab, cfg.hidden, **kw))

    @classmethod
    def from_pretrained(cls, path, n_layers=None, device="cuda"):
        cfg = Qwen3Config.from_hf(path)
        m = cls(cfg, n_layers, device)
        L = len(m.layers)
        prefixes = ["model.embed_tokens", "model.norm", "lm_head"] + [f"model.layers.{i}." for i in range(L)]
        sd = load_state_dict(path, prefixes, device)
        with torch.no_grad():
            m.embed.copy_(sd["model.embed_tokens.weight"])
            m.norm.copy_(sd["model.norm.weight"])
            m.lm_head.copy_(sd["lm_head.weight"])
            for i, blk in enumerate(m.layers):
                blk.load_hf(sd, f"model.layers.{i}.")
        del sd
        torch.cuda.empty_cache()
        return m

    def _positions(self, B, S, device, start=0):
        return (torch.arange(S, device=device, dtype=torch.int32) + start).repeat(B)

    def forward(self, ids, targets=None):
        """ids [B, S]. Returns logits [B, S, V] (bf16), or the mean CE loss if targets is given."""
        B, S = ids.shape
        pos = self._positions(B, S, ids.device)
        x = ops.embedding(ids.reshape(-1), self.embed)
        for blk in self.layers:
            x = blk(x, pos, B, S)
        h = ops.rmsnorm(x, self.norm, self.cfg.eps)
        if targets is not None:
            return ops.lm_head_cross_entropy(h, self.lm_head, targets)
        return ops.linear(h, self.lm_head).view(B, S, -1)

    # ------------------------------------------------------------ generation
    @torch.no_grad()
    def generate(self, ids, max_new_tokens=32, temperature=0.0, top_k=0, top_p=1.0, seed=0, eos=None,
                 use_megakernel=False):
        """ids [B, S0] (equal-length prompts). Returns [B, S0 + n] token ids."""
        c, e = self.cfg, ext()
        B, S0 = ids.shape
        dev = ids.device
        Smax = S0 + max_new_tokens
        L = len(self.layers)
        kc = torch.zeros(L, B, c.n_kv_heads, Smax, c.head_dim, device=dev, dtype=torch.bfloat16)
        vc = torch.zeros_like(kc)
        pos = self._positions(B, S0, dev)
        x = e.embedding_fwd(ids.reshape(-1).to(torch.int32).contiguous(), self.embed)
        for i, blk in enumerate(self.layers):
            x = blk.prefill(x, pos, B, S0, kc[i], vc[i])
        last = x.view(B, S0, -1)[:, -1].contiguous()
        h, _ = e.rmsnorm_fwd(last, self.norm, c.eps)
        logits = e.gemv(self.lm_head, h, None, True)
        nxt = ops.sample(logits, temperature, top_k, top_p, seed, 0)
        out = [ids, nxt.view(B, 1)]
        mk = None
        if use_megakernel:
            from .megakernel import MegaDecoder

            mk = MegaDecoder(self, B, Smax, kc, vc)
        for step in range(1, max_new_tokens):
            cur = S0 + step - 1  # position of the token being fed
            if eos is not None and bool((nxt == eos).all()):
                break
            if mk is not None:
                logits = mk.step(nxt, cur)
            else:
                lens = torch.full((B,), cur + 1, dtype=torch.int32, device=dev)
                p = torch.full((B,), cur, dtype=torch.int32, device=dev)
                x = e.embedding_fwd(nxt.to(torch.int32).contiguous(), self.embed)
                for i, blk in enumerate(self.layers):
                    x = blk.decode(x, p, cur, kc[i], vc[i], lens)
                h, _ = e.rmsnorm_fwd(x, self.norm, c.eps)
                logits = e.gemv(self.lm_head, h, None, True)
            nxt = ops.sample(logits, temperature, top_k, top_p, seed, step)
            out.append(nxt.view(B, 1))
        return torch.cat(out, 1)
