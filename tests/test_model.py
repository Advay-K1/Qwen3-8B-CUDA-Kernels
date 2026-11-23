"""End-to-end tests against HuggingFace transformers on the real Qwen3-8B weights.

  1. one decoder layer, fwd + bwd (input grad and all weight grads) vs HF Qwen3DecoderLayer
  2. 2-layer truncated model: training loss + grads (embedding, LM head + CE) vs HF
  3. full 36-layer model: prefill logits vs HF
  4. greedy generation: our layered decode path and the megakernel vs HF generate
  5. megakernel single block vs the layer-by-layer decode kernels

Run: python tests/test_model.py [--skip-full]
"""
import argparse
import gc
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from qwen3_kernels import Qwen3Block, Qwen3Config, Qwen3Model, load_state_dict  # noqa: E402

PATH = os.environ.get("QWEN3_PATH")
DEV = "cuda"
BF = torch.bfloat16
RESULTS = []


def check(name, out, ref, tol=3e-2, cos_min=0.999):
    out, ref = out.float(), ref.float()
    err = (out - ref).abs().max().item()
    scale = ref.abs().max().item() + 1e-6
    cos = F.cosine_similarity(out.flatten(), ref.flatten(), dim=0).item()
    ok = err / scale < tol and cos > cos_min
    RESULTS.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name:44s} rel_max_err={err / scale:.2e} cos={cos:.6f}")


def report(name, ok, msg=""):
    RESULTS.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name:44s} {msg}")


def hf_config(n_layers=None):
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(PATH)
    if n_layers is not None:
        cfg.num_hidden_layers = n_layers
    cfg._attn_implementation = "sdpa"
    return cfg


# ----------------------------------------------------------------------------- 1. decoder layer
def test_block(layer_idx=0, B=2, S=256):
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding

    hcfg = hf_config()
    cfg = Qwen3Config.from_hf(PATH)
    pre = f"model.layers.{layer_idx}."
    sd = load_state_dict(PATH, [pre, "model.embed_tokens"], DEV)
    ref = Qwen3DecoderLayer(hcfg, layer_idx).to(DEV, BF)
    ref.load_state_dict({k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)})
    ours = Qwen3Block(cfg)
    ours.load_hf(sd, pre)

    ids = torch.randint(0, cfg.vocab, (B, S), device=DEV)
    x0 = F.embedding(ids, sd["model.embed_tokens.weight"]).detach()
    dy = torch.randn(B, S, cfg.hidden, device=DEV, dtype=BF)

    rope = Qwen3RotaryEmbedding(hcfg).to(DEV)
    pos_ids = torch.arange(S, device=DEV)[None].expand(B, S)
    xr = x0.clone().requires_grad_()
    pe = rope(xr, pos_ids)
    yr = ref(xr, attention_mask=None, position_ids=pos_ids, position_embeddings=pe)
    yr = yr[0] if isinstance(yr, tuple) else yr
    yr.backward(dy)

    xo = x0.reshape(B * S, -1).clone().requires_grad_()
    pos = torch.arange(S, device=DEV, dtype=torch.int32).repeat(B)
    yo = ours(xo, pos, B, S)
    yo.backward(dy.reshape(B * S, -1))

    tag = f"layer{layer_idx}"
    check(f"{tag} fwd", yo, yr.reshape(B * S, -1))
    check(f"{tag} d(hidden)", xo.grad, xr.grad.reshape(B * S, -1))
    g = {n: p.grad for n, p in ref.named_parameters()}
    check(f"{tag} d(input_layernorm)", ours.input_ln.grad, g["input_layernorm.weight"])
    check(f"{tag} d(qkv_proj)", ours.qkv_w.grad,
          torch.cat([g["self_attn.q_proj.weight"], g["self_attn.k_proj.weight"], g["self_attn.v_proj.weight"]]))
    check(f"{tag} d(q_norm)", ours.q_norm.grad, g["self_attn.q_norm.weight"])
    check(f"{tag} d(k_norm)", ours.k_norm.grad, g["self_attn.k_norm.weight"])
    check(f"{tag} d(o_proj)", ours.o_w.grad, g["self_attn.o_proj.weight"])
    check(f"{tag} d(post_attention_layernorm)", ours.post_ln.grad, g["post_attention_layernorm.weight"])
    check(f"{tag} d(gate_up_proj)", ours.gate_up_w.grad,
          torch.cat([g["mlp.gate_proj.weight"], g["mlp.up_proj.weight"]]))
    check(f"{tag} d(down_proj)", ours.down_w.grad, g["mlp.down_proj.weight"])
    del ref, ours, sd
    gc.collect()
    torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- 2. truncated model training step
def test_train_2layer(B=2, S=128):
    from transformers import Qwen3ForCausalLM

    L = 2
    hcfg = hf_config(L)
    sd = load_state_dict(PATH, ["model.embed_tokens", "model.norm", "lm_head"] + [f"model.layers.{i}." for i in range(L)])
    with torch.device(DEV):
        ref = Qwen3ForCausalLM(hcfg).to(BF)
    ref.load_state_dict(sd, strict=False)
    del sd
    ours = Qwen3Model.from_pretrained(PATH, n_layers=L)
    ids = torch.randint(0, hcfg.vocab_size, (B, S), device=DEV)
    tgt = torch.roll(ids, -1, 1)
    tgt[:, -1] = -100

    logits = ref(input_ids=ids).logits
    loss_r = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=-100)
    loss_r.backward()
    loss_o = ours(ids, tgt)
    loss_o.backward()
    report("2-layer train loss", abs(loss_o.item() - loss_r.item()) < 2e-2 * abs(loss_r.item()),
           f"ours={loss_o.item():.5f} hf={loss_r.item():.5f}")
    check("2-layer d(lm_head)", ours.lm_head.grad, ref.lm_head.weight.grad)
    check("2-layer d(final norm)", ours.norm.grad, ref.model.norm.weight.grad)
    check("2-layer d(embed)", ours.embed.grad, ref.model.embed_tokens.weight.grad)
    check("2-layer d(layer0 down_proj)", ours.layers[0].down_w.grad, ref.model.layers[0].mlp.down_proj.weight.grad)
    gq = ref.model.layers[0].self_attn
    check("2-layer d(layer0 qkv_proj)", ours.layers[0].qkv_w.grad,
          torch.cat([gq.q_proj.weight.grad, gq.k_proj.weight.grad, gq.v_proj.weight.grad]))
    del ref, ours
    gc.collect()
    torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- 3/4. full model
PROMPT = "The key idea behind FlashAttention is"


def hf_reference(n_new):
    from transformers import AutoTokenizer, Qwen3ForCausalLM

    tok = AutoTokenizer.from_pretrained(PATH)
    ids = tok(PROMPT, return_tensors="pt").input_ids.to(DEV)
    ref = Qwen3ForCausalLM.from_pretrained(PATH, dtype=BF, attn_implementation="sdpa").to(DEV)
    with torch.no_grad():
        logits = ref(input_ids=ids).logits.float()
        gen = ref.generate(ids, max_new_tokens=n_new, do_sample=False, temperature=None, top_p=None, top_k=None)
        gen_logits = ref(input_ids=gen).logits.float()[0]  # teacher-forced logits along HF's own sequence
    del ref
    gc.collect()
    torch.cuda.empty_cache()
    return tok, ids, logits, gen, gen_logits


def test_full_model(n_new=32):
    tok, ids, logits_r, gen_r, gen_logits = hf_reference(n_new)
    m = Qwen3Model.from_pretrained(PATH)
    with torch.no_grad():
        logits_o = m(ids).float()
    check("full model prefill logits", logits_o, logits_r, tol=6e-2, cos_min=0.995)
    top1 = (logits_o.argmax(-1) == logits_r.argmax(-1)).float().mean().item()
    report("full model prefill top-1 agreement", top1 >= 0.9, f"{top1 * 100:.1f}%")

    for name, mk in [("layered decode", False), ("megakernel", True)]:
        gen_o = m.generate(ids, max_new_tokens=n_new, use_megakernel=mk)
        n = min(gen_o.shape[1], gen_r.shape[1])
        same = (gen_o[0, :n] == gen_r[0, :n])
        first_diff = int((~same).nonzero()[0].item()) if (~same).any() else n
        n_match = first_diff - ids.shape[1]
        if first_diff < n:
            # A divergence is acceptable only at a bf16 near-tie in HF's own logits.
            lg = gen_logits[first_diff - 1]
            gap = abs(lg[gen_r[0, first_diff]] - lg[gen_o[0, first_diff]]).item()
            ok, why = gap <= 0.25, f"diverges at a near-tie (HF logit gap {gap:.3f})"
        else:
            ok, why = True, "identical"
        report(f"greedy generate ({name}) vs HF", ok, f"{n_match}/{n_new} tokens identical; {why}")
        print("   ours:", repr(tok.decode(gen_o[0, ids.shape[1]:])))
    print("   hf  :", repr(tok.decode(gen_r[0, ids.shape[1]:])))

    # 5. megakernel single block vs layered kernels, on a populated cache
    from qwen3_kernels.megakernel import MegaDecoder

    c = m.cfg
    B, Smax, P = 2, 512, 300
    kc = torch.randn(1, B, c.n_kv_heads, Smax, 128, device=DEV, dtype=BF)
    vc = torch.randn_like(kc)
    kc2, vc2 = kc.clone(), vc.clone()
    x = torch.randn(B, c.hidden, device=DEV, dtype=BF)
    blk = m.layers[3]
    pos = torch.full((B,), P, dtype=torch.int32, device=DEV)
    lens = torch.full((B,), P + 1, dtype=torch.int32, device=DEV)
    y_ref = blk.decode(x, pos, P, kc[0], vc[0], lens)
    mkd = MegaDecoder(m, B, Smax, kc2, vc2, layers=[blk], with_head=False)
    y_mk = mkd.forward_hidden(x, P)
    check("megakernel block vs layered decode", y_mk, y_ref, tol=2e-2)
    check("megakernel block K-cache append", kc2[0, :, :, P], kc[0, :, :, P], tol=2e-2)
    check("megakernel block V-cache append", vc2[0, :, :, P], vc[0, :, :, P], tol=1e-3)

    # Megakernel full step logits vs layered path at the same position, batch 3.
    B = 3
    ids3 = ids.repeat(B, 1)
    out_l = m.generate(ids3, max_new_tokens=4)
    out_m = m.generate(ids3, max_new_tokens=4, use_megakernel=True)
    report("megakernel batch=3 generate == layered", bool((out_l == out_m).all()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-full", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(0)
    print("--- test_block"); test_block(0); test_block(20)
    print("--- test_train_2layer"); test_train_2layer()
    if not args.skip_full:
        print("--- test_full_model"); test_full_model()
    n_fail = RESULTS.count(False)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    sys.exit(1 if n_fail else 0)
