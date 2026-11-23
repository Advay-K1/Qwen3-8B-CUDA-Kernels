"""Megakernel vs layer-by-layer decode kernels across batch sizes and context lengths.

Covers: KV append into a later sub-chunk, sub-chunk boundaries, multi-split combine, B=1..4,
all 36 layers + LM head (logits), and greedy generation equality.
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from qwen3_kernels import Qwen3Model  # noqa: E402
from qwen3_kernels._ext import ext  # noqa: E402
from qwen3_kernels.megakernel import MegaDecoder  # noqa: E402

RESULTS = []


def check(name, out, ref, tol=2e-2):
    out, ref = out.float(), ref.float()
    err = (out - ref).abs().max().item() / (ref.abs().max().item() + 1e-6)
    cos = F.cosine_similarity(out.flatten(), ref.flatten(), dim=0).item()
    ok = err < tol and cos > 0.9999
    RESULTS.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name:46s} rel_max_err={err:.2e} cos={cos:.6f}", flush=True)


def main():
    torch.manual_seed(0)
    m = Qwen3Model.from_pretrained(os.environ["QWEN3_PATH"])
    c = m.cfg
    e = ext()
    blk = m.layers[5]
    for B in (1, 2, 4):
        for P in (0, 1, 63, 64, 300, 1000, 4095):
            Smax = 4160
            kc = torch.randn(1, B, 8, Smax, 128, device="cuda", dtype=torch.bfloat16)
            vc = torch.randn_like(kc)
            kc2, vc2 = kc.clone(), vc.clone()
            x = torch.randn(B, c.hidden, device="cuda", dtype=torch.bfloat16)
            pos = torch.full((B,), P, dtype=torch.int32, device="cuda")
            y_ref = blk.decode(x, pos, P, kc[0], vc[0], pos + 1)
            mk = MegaDecoder(m, B, Smax, kc2, vc2, layers=[blk], with_head=False)
            y = mk.forward_hidden(x, P)
            check(f"block B={B} pos={P}", y, y_ref)
            RESULTS.append(bool(torch.equal(kc2[0, :, :, P], kc[0, :, :, P])) and bool(torch.equal(vc2, vc)))
            if not RESULTS[-1]:
                print(f"[FAIL] KV append B={B} pos={P}")
    # full stack + head, on a real (prefilled) KV cache
    for B in (1, 3):
        P = 777
        Smax = P + 8
        ids = torch.randint(0, c.vocab, (B, P), device="cuda")
        kc = torch.zeros(36, B, 8, Smax, 128, device="cuda", dtype=torch.bfloat16)
        vc = torch.zeros_like(kc)
        pos0 = torch.arange(P, device="cuda", dtype=torch.int32).repeat(B)
        with torch.no_grad():
            xx = e.embedding_fwd(ids.reshape(-1).to(torch.int32).contiguous(), m.embed)
            for i, b_ in enumerate(m.layers):
                xx = b_.prefill(xx, pos0, B, P, kc[i], vc[i])
        kc2, vc2 = kc.clone(), vc.clone()
        tok = torch.randint(0, c.vocab, (B,), device="cuda")
        pos = torch.full((B,), P, dtype=torch.int32, device="cuda")
        xx = e.embedding_fwd(tok.to(torch.int32), m.embed)
        for i, b_ in enumerate(m.layers):
            xx = b_.decode(xx, pos, P, kc[i], vc[i], pos + 1)
        h, _ = e.rmsnorm_fwd(xx, m.norm, c.eps)
        ref = e.gemv(m.lm_head, h, None, True)
        mk = MegaDecoder(m, B, Smax, kc2, vc2)
        out = mk.step(tok, P)
        check(f"36 layers + LM head logits B={B} (real cache)", out, ref, tol=3e-2)
        top = bool((out.argmax(-1) == ref.argmax(-1)).all())
        RESULTS.append(top)
        print(f"[{'PASS' if top else 'FAIL'}] top-1 token equal B={B}")
    n_fail = RESULTS.count(False)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
