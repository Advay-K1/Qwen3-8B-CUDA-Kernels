"""Python driver for the decode megakernel (one cooperative launch per generated token)."""
import math

import torch

from ._ext import mega


class MegaDecoder:
    """Runs embed -> all layers -> final norm -> LM head in a single kernel launch.

    k_cache / v_cache: [n_layers, B, Hkv, Smax, D] bf16 (shared with the prefill path).
    """

    def __init__(self, model, B, Smax, k_cache, v_cache, layers=None, with_head=True, prefetch=0):
        self.prefetch = prefetch
        m = mega()
        c = model.cfg
        assert (c.hidden, c.intermediate, c.n_heads, c.n_kv_heads, c.head_dim) == (4096, 12288, 32, 8, 128)
        assert 1 <= B <= 4
        self.cfg, self.B, self.Smax = c, B, Smax
        blocks = model.layers if layers is None else layers
        self.n_layers = len(blocks)
        self._keep = [[b.input_ln, b.qkv_w, b.q_norm, b.k_norm, b.o_w, b.post_ln, b.gate_up_w, b.down_w]
                      for b in blocks]
        dev = k_cache.device
        self.layers = m.pack_layers([[t.data.contiguous() for t in L] for L in self._keep], k_cache)
        self.embed = model.embed.data if hasattr(model, "embed") else None
        self.final_norm = model.norm.data if hasattr(model, "norm") else None
        self.lm_head = model.lm_head.data if (with_head and hasattr(model, "lm_head")) else None
        self.kc, self.vc = k_cache, v_cache
        sms = torch.cuda.get_device_properties(dev).multi_processor_count
        self.grid = min(sms, m.max_grid(B))
        assert self.grid > 0, "megakernel does not fit on this GPU"
        self.max_split = max(1, self.grid // (B * c.n_kv_heads))
        bf = dict(device=dev, dtype=torch.bfloat16)
        f32 = dict(device=dev, dtype=torch.float32)
        self.x = torch.empty(B, c.hidden, **bf)
        self.qkv = torch.empty(B, c.qkv_dim, **bf)
        self.part = torch.empty(B, c.n_heads, self.max_split, 130, **f32)
        self.attn = torch.empty(B, c.n_heads * c.head_dim, **bf)
        self.h = torch.empty(B, c.intermediate, **bf)
        self.logits = torch.empty(B, c.vocab if self.lm_head is not None else 1, **f32)
        self.bar = torch.zeros(1, device=dev, dtype=torch.int32)
        self._dummy = torch.zeros(8, **bf)

    def _schedule(self, pos):
        L = pos + 1
        nsplit = max(1, min(self.max_split, math.ceil(L / 32)))
        chunk = math.ceil(L / nsplit)
        nsplit = math.ceil(L / chunk)
        return nsplit, chunk

    def _run(self, tokens, pos, with_head):
        c = self.cfg
        nsplit, chunk = self._schedule(pos)
        mega().decode_step(self.layers, self.n_layers, tokens,
                           self.embed if self.embed is not None else self._dummy,
                           self.final_norm if self.final_norm is not None else self._dummy,
                           self.lm_head if with_head else None, self.kc, self.vc, pos, nsplit, chunk, c.eps,
                           c.rope_theta, self.x, self.qkv, self.part, self.attn, self.h, self.logits, self.bar,
                           self.grid, self.prefetch)

    @torch.no_grad()
    def step(self, tokens, pos):
        """tokens int64 [B] at position `pos` -> fp32 logits [B, V]."""
        self._run(tokens.to(torch.int64).contiguous(), pos, self.lm_head is not None)
        return self.logits

    @torch.no_grad()
    def forward_hidden(self, x, pos):
        """Block(s)-only megakernel: hidden [B, H] -> hidden [B, H] (no embed / head)."""
        self.x.copy_(x)
        self._run(None, pos, False)
        return self.x.clone()
