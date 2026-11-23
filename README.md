# Qwen3-8B CUDA Kernels

Hand-written CUDA kernels covering Qwen3-8B training (forward + backward), inference, and a decode megakernel.
No cuBLAS or FlashAttention library calls are made: PyTorch is used only for allocation, autograd bookkeeping
and as a test reference.

## Kernels (`csrc/`)

| Kernel | File | Fwd | Bwd | Notes |
|---|---|---|---|---|
| Tensor-core GEMM | `gemm.cu` | ✓ | ✓ | mma.sync m16n8k16 bf16, ldmatrix(.trans), cp.async 3-stage, XOR-swizzled smem, 128x128x64 tiles. 4 layout variants cover `y=xWᵀ`, `dx=dyW`, `dW=dyᵀx`. Epilogue: residual add, fp32 out, accumulate |
| RMSNorm / fused add+RMSNorm | `rmsnorm.cu` | ✓ | ✓ | HF rounding semantics; bwd fuses the residual-stream grad add; dw via per-block fp32 atomics |
| QK-norm + RoPE + QKV split | `qk_norm_rope.cu` | ✓ | ✓ | per-head RMSNorm + rotate-half RoPE fused; writes attention layout / KV cache directly; bwd consumes fp32 dQ from attention |
| FlashAttention-2 (causal GQA, d=128) | `flash_attn.cu` | ✓ | ✓ | online softmax, P kept in registers; bwd: one CTA per KV block loops over its 4 query heads (dK/dV in registers), dQ via fp32 atomics |
| SwiGLU | `elementwise.cu` | ✓ | ✓ | on the fused `[gate|up]` GEMM output |
| Embedding | `elementwise.cu` | ✓ | ✓ | bwd scatter-add in fp32 |
| LM head + cross-entropy | `elementwise.cu` + `ops.py` | ✓ | ✓ | chunked: GEMM → in-place CE (loss + dlogits) → dh/dW GEMMs; never materializes full `[T, 151936]` logits |
| Sampling | `sampling.cu` | ✓ | – | greedy / temperature / exact top-k and top-p via 4-pass radix select, inverse-CDF Philox draw |
| GEMV, split-KV decode attention | `decode.cu` | ✓ | – | batch 1..8 decode path |

The decoder layer (`Qwen3Block.forward`) runs:
`rmsnorm → QKV GEMM → [qk-norm+RoPE → flash-attn] → O GEMM(+residual) → rmsnorm → gate/up GEMM → SwiGLU → down GEMM(+residual)`,
with every step differentiable through these kernels.

## Megakernel (`megakernel/`)

`decode_megakernel` runs **one full decode step in one persistent cooperative launch**: embedding, all 36 layers,
final norm and the LM head. It uses one 512-thread CTA per SM and a grid barrier between dependent stages,
6 barriers per layer:

1. input RMSNorm (each CTA does it redundantly) + QKV GEMV
2. split-KV attention partials, with q/k-norm, RoPE and the KV-cache append fused in
3. combine the partials
4. O-proj GEMV + residual
5. post RMSNorm + gate/up GEMV + SwiGLU
6. down GEMV + residual

In the GEMVs, each CTA owns a slice of output rows and its warps share fixed (row, 2048-wide K segment) units.
Partials are reduced in shared memory, so results are deterministic and the load is balanced.
Supports batch 1–4. You can also run it on a subset of layers with no embedding or head (`MegaDecoder(..., layers=[blk], with_head=False)`);
that is the single-transformer-block megakernel.

## Usage

```bash
source scripts/env.sh                       # Delta: torch 2.12 + CUDA 13, sets QWEN3_PATH
python tests/test_ops.py                    # every kernel fwd/bwd vs fp32 torch reference
python tests/test_model.py                  # layer/model/generation vs HF transformers, megakernel checks
python bench/bench.py --decode              # kernel + decode throughput benchmarks
```

```python
from qwen3_kernels import Qwen3Model
m = Qwen3Model.from_pretrained(path)
loss = m(ids, targets); loss.backward()                         # training step on custom kernels
out = m.generate(ids, 64, temperature=0.7, top_p=0.9, use_megakernel=True)
```

## Status / results (A100-SXM4-40GB)

Correctness: `tests/test_ops.py` 79/79, `tests/test_model.py` 34/34. The model tests compare against HF transformers on the real weights:
- every layer's fwd output and all of its weight grads
- 2-layer training loss and grads
- full-model prefill logits: cosine 0.99997, 100% top-1 agreement
- greedy generation: identical until a bf16 near-tie where HF's own logits are tied

The megakernel reproduces the layer-by-layer decode path token for token.

| | ours | reference |
|---|---|---|
| GEMM, Qwen3 shapes, T=4096 | 176–200 TFLOP/s | cuBLAS 250–264 |
| Flash-attn fwd, S=4096, causal GQA | 1.07 ms | SDPA 0.84 ms |
| Flash-attn bwd | 4.08 ms | SDPA 2.50 ms |
| Decoder layer fwd+bwd, S=4096 | 35.6 ms (145 TFLOP/s) | – |
| Decode B=1, ctx 512: layered / +CUDA graph / **megakernel** | 13.44 / 12.94 / **12.19 ms/token** (82 tok/s, ~1.25 TB/s ≈ 80% of HBM peak) | – |
| Decode B=4, ctx 512: layered+graph / **megakernel** | 15.17 / **13.13 ms/token** (305 tok/s) | – |

Full logs are in `bench/results/`.

Possible next steps:
- flash-attn bwd: get rid of register spills and the dQ atomics, e.g. with a separate dQ pass
- GEMM: double-buffer fragments in registers and stage the epilogue through shared memory
- Hopper-specific paths (wgmma/TMA) for the H200 nodes
