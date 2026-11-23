// Host interface of the Qwen3-8B decode megakernel.
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#define MK_THREADS 512

namespace mk {
using bf16 = __nv_bfloat16;

struct LayerW {
  const bf16 *in_ln, *qkv, *q_norm, *k_norm, *o, *post_ln, *gate_up, *down;
};

struct MegaParams {
  const LayerW* layers;   // device array [n_layers]
  int n_layers;
  int B;                  // sequences (1..4)
  const long* tokens;     // [B] input tokens, or nullptr to start from x
  const bf16* embed;      // [V, H]
  const bf16* final_norm; // [H]
  const bf16* lm_head;    // [V, H] or nullptr (skip head)
  int vocab;
  bf16* k_cache;          // [n_layers, B, Hkv, Smax, D]
  bf16* v_cache;
  int Smax;
  int pos;                // position of the token being decoded (cache length before this step)
  int nsplit, chunk;      // split-KV schedule for attention
  float eps, theta, scale_log2;
  // scratch
  bf16* x;                // [B, H] residual stream (in/out)
  bf16* qkv;              // [B, QKV]
  float* attn_part;       // [B, Hq, nsplit, D + 2]
  bf16* attn_out;         // [B, Hq * D]
  bf16* h;                // [B, I]
  float* logits;          // [B, V]
  unsigned* bar;          // grid barrier counter (zeroed before launch)
  unsigned long long* timing;  // optional: block 0 writes %globaltimer at each stage boundary
  int prefetch;           // bit0: L2-prefetch O-proj slice during attention; bit1: next-stage rows before barriers.
                          // Measured slower on A100 (L2 contention with the streaming GEMVs); off by default.
};

cudaError_t launch(const MegaParams& p, int grid, cudaStream_t st);
int max_grid(int B, int num_sms);
}  // namespace mk
