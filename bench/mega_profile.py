"""Per-stage timing of the decode megakernel (block 0's %globaltimer at every grid barrier).

python bench/mega_profile.py [--batch B] [--ctx N]
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from qwen3_kernels import Qwen3Model  # noqa: E402
from qwen3_kernels.megakernel import MegaDecoder  # noqa: E402

STAGES = ["qkv", "attn", "comb+o_proj", "gate_up", "down"]
NS = len(STAGES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--ctx", type=int, nargs="+", default=[512, 4096])
    ap.add_argument("--bw", type=float, default=0, help="achievable GB/s for %% column (default: measure)")
    a = ap.parse_args()
    bw = a.bw
    if not bw:
        x = torch.empty(1024**3, device="cuda", dtype=torch.bfloat16)
        x.sum()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(10):
            x.sum(dtype=torch.float32)
        e.record()
        torch.cuda.synchronize()
        bw = 10 * x.numel() * 2 / s.elapsed_time(e) / 1e6
        del x
    m = Qwen3Model.from_pretrained(os.environ["QWEN3_PATH"])
    c = m.cfg
    B = a.batch
    wbytes = {"qkv": c.qkv_dim * 4096 * 2, "attn": 0, "comb+o_proj": 4096 * 4096 * 2,
              "gate_up": 2 * 12288 * 4096 * 2, "down": 4096 * 12288 * 2}
    print(f"achievable HBM read bandwidth (torch sum): {bw:.0f} GB/s")
    for ctx in a.ctx:
        Smax = ctx + 8
        kc = torch.randn(36, B, 8, Smax, 128, device="cuda", dtype=torch.bfloat16) * 0.5
        vc = torch.randn_like(kc)
        mk = MegaDecoder(m, B, Smax, kc, vc)
        mk.timing = torch.zeros(512 + (4 + NS * 36) * mk.grid, dtype=torch.int64, device="cuda")
        tok = torch.randint(0, c.vocab, (B,), device="cuda")
        for _ in range(30):
            mk.step(tok, ctx)
        acc = torch.zeros(NS, dtype=torch.float64)
        emb = head = tot = 0.0
        N = 20
        for _ in range(N):
            mk.step(tok, ctx)
            torch.cuda.synchronize()
            t = mk.timing.cpu().double() / 1e3  # us
            emb += t[1] - t[0]
            lay = t[1:1 + NS * 36 + 1]
            d = (lay[1:] - lay[:-1]).view(36, NS)
            acc += d.sum(0)
            head += t[2 + NS * 36] - t[1 + NS * 36]
            tot += t[2 + NS * 36] - t[0]
        acc /= N
        # per-CTA arrival times (last run): work time vs barrier wait per stage
        arr = mk.timing[512:512 + (2 + NS * 36) * mk.grid].view(-1, mk.grid).cpu().double() / 1e3
        rel = t[:2 + NS * 36]  # block-0 release stamps; arrivals of barrier k precede rel[k+1]
        work_med = torch.zeros(NS); work_max = torch.zeros(NS); wait = torch.zeros(NS)
        for L_ in range(36):
            for i in range(NS):
                k = 1 + L_ * NS + i  # tick index of the barrier closing this stage
                start = rel[k]
                a_ = arr[k + 1] - start
                work_med[i] += a_.median().item(); work_max[i] += a_.max().item()
                wait[i] += (rel[k + 1] - arr[k + 1].max()).item()
        kvb = 2 * B * 8 * (ctx + 1) * 128 * 2
        print(f"\nB={B} ctx={ctx}: total {tot / N / 1e3:.3f} ms (in-kernel)   embed {emb / N:.1f} us   "
              f"head {head / N:.1f} us ({1.24e9 / (head / N) / 1e3:.0f} GB/s)")
        print(f"  {'stage':12s} {'us/layer':>9s} {'x36 ms':>8s} {'GB/s':>7s} {'%bw':>5s} {'medCTA':>7s} {'maxCTA':>7s} {'barrier':>7s}")
        for i, name in enumerate(STAGES):
            us = acc[i].item() / 36
            by = wbytes[name] + (kvb if name == "attn" else 0)
            g = by / us / 1e3 if by else 0
            print(f"  {name:12s} {us:9.2f} {acc[i].item() / 1e3:8.3f} {g:7.0f} {100 * g / bw:5.1f} "
                  f"{work_med[i] / 36:7.2f} {work_max[i] / 36:7.2f} {wait[i] / 36:7.2f}")
        del kc, vc, mk


if __name__ == "__main__":
    main()
