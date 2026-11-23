// Host-side launchers for all kernels (raw pointers; torch glue lives in bindings.cpp).
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace qk {
using bf16 = __nv_bfloat16;

struct GemmArgs {
  const bf16* A;
  const bf16* B;
  void* C;
  int M, N, K;
  int lda, ldb, ldc;
  bool a_kmajor, b_kmajor;
  float alpha = 1.f;
  const bf16* R = nullptr;
  int ldr = 0;
  bool out_f32 = false;
  bool accumulate = false;
};
void gemm(const GemmArgs& a, cudaStream_t st);
void gemm_force_config(int cfg);  // -1 heuristic, 0 = 128x128, 1 = 128x256 (benchmarking)

void rmsnorm_fwd(const bf16* x, const bf16* res, const bf16* w, bf16* y, bf16* s_out, float* rstd, int rows, int H,
                 float eps, cudaStream_t st);
int rmsnorm_bwd_parts(int rows, int num_sms);  // size of dw_part's first dim
void rmsnorm_bwd(const bf16* dy, const bf16* x, const bf16* w, const float* rstd, const bf16* dres_in, bf16* dx,
                 float* dw, float* dw_part, int rows, int H, int num_sms, cudaStream_t st);

void qk_norm_rope_fwd(const bf16* qkv, const int* pos, const bf16* wq, const bf16* wk, bf16* q, bf16* k, bf16* v,
                      float* rstd_q, float* rstd_k, int B, int S, int Hq, int Hkv, int kv_S, int kv_off, float theta,
                      float eps, cudaStream_t st);
void qk_norm_rope_bwd(const float* dq, const bf16* dk, const bf16* dv, const bf16* qkv, const int* pos,
                      const bf16* wq, const bf16* wk, const float* rstd_q, const float* rstd_k, bf16* dqkv,
                      float* dwq, float* dwk, int B, int S, int Hq, int Hkv, float theta, int num_sms,
                      cudaStream_t st);

void flash_attn_fwd(const bf16* q, const bf16* k, const bf16* v, bf16* o, float* lse, int B, int S, int Hq, int Hkv,
                    int kv_S, float scale, cudaStream_t st);
void flash_attn_bwd(const bf16* q, const bf16* k, const bf16* v, const bf16* o, const bf16* dout, const float* lse,
                    float* dvec, float* dq_acc, bf16* dk, bf16* dv, int B, int S, int Hq, int Hkv, float scale,
                    cudaStream_t st);

void flash_fwd_set_warps(int nw);  // 4 (64-row Q tiles, default) or 8 (128-row)
void flash_bwd_set_warps(int nw);  // 4 (64-key blocks) or 8 (128-key blocks, default)

void swiglu_fwd(const bf16* gu, bf16* h, long T, int I, cudaStream_t st);
void swiglu_bwd(const bf16* dh, const bf16* gu, bf16* dgu, long T, int I, cudaStream_t st);
void embedding_fwd(const int* ids, const bf16* table, bf16* out, long T, int H, cudaStream_t st);
void embedding_bwd(const int* ids, const bf16* dout, float* dtable, long T, int H, cudaStream_t st);
void cross_entropy(bf16* logits, const int* target, float* loss, float* lse, long rows, int V, long ld,
                   float grad_scale, int ignore_index, cudaStream_t st);
void scale_cast(const float* in, bf16* out, long n, float s, cudaStream_t st);

void sample_tokens(const float* logits, long B, int V, long ld, const float* temperature, const int* top_k,
                   const float* top_p, unsigned long long seed, unsigned long long offset, long* out,
                   cudaStream_t st);

void gemv(const bf16* W, const bf16* x, void* y, const bf16* R, int NB, int N, int K, int ldy, bool out_f32,
          int num_sms, cudaStream_t st);
int decode_attn_num_splits(int max_len, int chunk);
void decode_attn(const bf16* q, const bf16* kc, const bf16* vc, const int* lens, float* part, bf16* out, int B,
                 int Hq, int Hkv, int Smax, int max_len, int chunk, float scale, cudaStream_t st);

}  // namespace qk
