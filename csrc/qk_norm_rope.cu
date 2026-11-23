// Fused per-head RMSNorm (q_norm / k_norm) + rotary embedding + QKV split.
//
// Input  qkv [T, (Hq + 2*Hkv) * D]  (output of the fused QKV projection), T = B*S
// Output q [B, Hq, S, D], k [B, Hkv, S, D], v [B, Hkv, S, D]   (attention layout)
//
// RoPE (HF rotate_half convention, D = 128, half = 64):
//   out[i]      = x[i]*cos_i - x[i+64]*sin_i
//   out[i+64]   = x[i+64]*cos_i + x[i]*sin_i,     angle_i = pos * theta^(-2i/D)
// One warp per (token, head); lane handles d = 2l, 2l+1, 64+2l, 64+2l+1.
#include "common.cuh"
#include "kernels.h"

namespace qk {
namespace {

constexpr int D = 128;

QK_DEVICE void rope_angles(int pos, int lane, float theta, float* c, float* s) {
  // inv_freq_i = theta^(-2i/D), i = 2*lane + {0,1}
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    int i = 2 * lane + j;
    float inv = 1.0f / powf(theta, (float)(2 * i) / (float)D);  // same fp32 recipe as HF
    float ang = (float)pos * inv;
    sincosf(ang, &s[j], &c[j]);
  }
}

__global__ void qk_norm_rope_fwd_kernel(const bf16* __restrict__ qkv, const int* __restrict__ pos,
                                        const bf16* __restrict__ wq, const bf16* __restrict__ wk,
                                        bf16* __restrict__ q, bf16* __restrict__ k, bf16* __restrict__ v,
                                        float* __restrict__ rstd_q, float* __restrict__ rstd_k, int T, int S,
                                        int Hq, int Hkv, int kv_S, int kv_off, float theta, float eps) {
  const int lane = threadIdx.x & 31;
  const int nh = Hq + 2 * Hkv;
  const long item = (long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (item >= (long)T * nh) return;
  const int t = item / nh, h = item % nh;
  const int b = t / S, sidx = t % S;
  const bf16* src = qkv + (size_t)t * nh * D + h * D;
  float x[4];
  {
    float2 lo = __bfloat1622float2(reinterpret_cast<const bf162*>(src)[lane]);
    float2 hi = __bfloat1622float2(reinterpret_cast<const bf162*>(src + 64)[lane]);
    x[0] = lo.x; x[1] = lo.y; x[2] = hi.x; x[3] = hi.y;
  }
  if (h >= Hq + Hkv) {  // V: plain copy
    int hv = h - Hq - Hkv;
    bf16* dst = v + (((size_t)b * Hkv + hv) * kv_S + kv_off + sidx) * D;
    reinterpret_cast<bf162*>(dst)[lane] = reinterpret_cast<const bf162*>(src)[lane];
    reinterpret_cast<bf162*>(dst + 64)[lane] = reinterpret_cast<const bf162*>(src + 64)[lane];
    return;
  }
  const bool isq = h < Hq;
  const bf16* w = isq ? wq : wk;
  float ss = warp_sum(x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]);
  float r = rsqrtf(ss / D + eps);
  float2 wlo = __bfloat1622float2(reinterpret_cast<const bf162*>(w)[lane]);
  float2 whi = __bfloat1622float2(reinterpret_cast<const bf162*>(w + 64)[lane]);
  float n0 = bf16_round(x[0] * r) * wlo.x, n1 = bf16_round(x[1] * r) * wlo.y;
  float n2 = bf16_round(x[2] * r) * whi.x, n3 = bf16_round(x[3] * r) * whi.y;
  float c[2], s[2];
  rope_angles(pos[t], lane, theta, c, s);
  float o0 = n0 * c[0] - n2 * s[0], o1 = n1 * c[1] - n3 * s[1];
  float o2 = n2 * c[0] + n0 * s[0], o3 = n3 * c[1] + n1 * s[1];
  bf16* dst;
  if (isq) {
    dst = q + (((size_t)b * Hq + h) * S + sidx) * D;
    if (lane == 0 && rstd_q) rstd_q[(size_t)t * Hq + h] = r;
  } else {
    int hk = h - Hq;
    dst = k + (((size_t)b * Hkv + hk) * kv_S + kv_off + sidx) * D;
    if (lane == 0 && rstd_k) rstd_k[(size_t)t * Hkv + hk] = r;
  }
  reinterpret_cast<bf162*>(dst)[lane] = __floats2bfloat162_rn(o0, o1);
  reinterpret_cast<bf162*>(dst + 64)[lane] = __floats2bfloat162_rn(o2, o3);
}

// Backward. dq is fp32 (flash-attn bwd accumulator), dk/dv bf16, all in attention layout.
// Writes dqkv [T, nh*D] and accumulates dwq/dwk (fp32 [D]).
constexpr int BWD_WARPS = 8;
__global__ void __launch_bounds__(BWD_WARPS * 32)
qk_norm_rope_bwd_kernel(const float* __restrict__ dq, const bf16* __restrict__ dk, const bf16* __restrict__ dv,
                        const bf16* __restrict__ qkv, const int* __restrict__ pos, const bf16* __restrict__ wq,
                        const bf16* __restrict__ wk, const float* __restrict__ rstd_q,
                        const float* __restrict__ rstd_k, bf16* __restrict__ dqkv, float* __restrict__ dwq,
                        float* __restrict__ dwk, int T, int S, int Hq, int Hkv, float theta) {
  __shared__ float red[2][BWD_WARPS][D];
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  const int nh = Hq + 2 * Hkv;
  const long total = (long)T * nh;
  float accq[4] = {0, 0, 0, 0}, acck[4] = {0, 0, 0, 0};
  float2 wql = __bfloat1622float2(reinterpret_cast<const bf162*>(wq)[lane]);
  float2 wqh = __bfloat1622float2(reinterpret_cast<const bf162*>(wq + 64)[lane]);
  float2 wkl = __bfloat1622float2(reinterpret_cast<const bf162*>(wk)[lane]);
  float2 wkh = __bfloat1622float2(reinterpret_cast<const bf162*>(wk + 64)[lane]);

  for (long item = (long)blockIdx.x * BWD_WARPS + wid; item < total; item += (long)gridDim.x * BWD_WARPS) {
    const int t = item / nh, h = item % nh;
    const int b = t / S, sidx = t % S;
    bf16* dst = dqkv + (size_t)t * nh * D + h * D;
    if (h >= Hq + Hkv) {
      int hv = h - Hq - Hkv;
      const bf16* g = dv + (((size_t)b * Hkv + hv) * S + sidx) * D;
      reinterpret_cast<bf162*>(dst)[lane] = reinterpret_cast<const bf162*>(g)[lane];
      reinterpret_cast<bf162*>(dst + 64)[lane] = reinterpret_cast<const bf162*>(g + 64)[lane];
      continue;
    }
    const bool isq = h < Hq;
    float g[4];
    float r;
    float w[4];
    if (isq) {
      const float* gp = dq + (((size_t)b * Hq + h) * S + sidx) * D;
      float2 lo = reinterpret_cast<const float2*>(gp)[lane];
      float2 hi = reinterpret_cast<const float2*>(gp + 64)[lane];
      g[0] = lo.x; g[1] = lo.y; g[2] = hi.x; g[3] = hi.y;
      r = rstd_q[(size_t)t * Hq + h];
      w[0] = wql.x; w[1] = wql.y; w[2] = wqh.x; w[3] = wqh.y;
    } else {
      int hk = h - Hq;
      const bf16* gp = dk + (((size_t)b * Hkv + hk) * S + sidx) * D;
      float2 lo = __bfloat1622float2(reinterpret_cast<const bf162*>(gp)[lane]);
      float2 hi = __bfloat1622float2(reinterpret_cast<const bf162*>(gp + 64)[lane]);
      g[0] = lo.x; g[1] = lo.y; g[2] = hi.x; g[3] = hi.y;
      r = rstd_k[(size_t)t * Hkv + hk];
      w[0] = wkl.x; w[1] = wkl.y; w[2] = wkh.x; w[3] = wkh.y;
    }
    // Inverse rotation (R^T).
    float c[2], s[2];
    rope_angles(pos[t], lane, theta, c, s);
    float gn[4];
    gn[0] = g[0] * c[0] + g[2] * s[0];
    gn[1] = g[1] * c[1] + g[3] * s[1];
    gn[2] = g[2] * c[0] - g[0] * s[0];
    gn[3] = g[3] * c[1] - g[1] * s[1];
    // RMSNorm backward over D.
    const bf16* src = qkv + (size_t)t * nh * D + h * D;
    float2 lo = __bfloat1622float2(reinterpret_cast<const bf162*>(src)[lane]);
    float2 hi = __bfloat1622float2(reinterpret_cast<const bf162*>(src + 64)[lane]);
    float xh[4] = {lo.x * r, lo.y * r, hi.x * r, hi.y * r};
    float dot = 0.f;
    float* acc = isq ? accq : acck;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      acc[i] += gn[i] * xh[i];
      gn[i] *= w[i];
      dot += gn[i] * xh[i];
    }
    dot = warp_sum(dot) / D;
    float o[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) o[i] = r * (gn[i] - xh[i] * dot);
    reinterpret_cast<bf162*>(dst)[lane] = __floats2bfloat162_rn(o[0], o[1]);
    reinterpret_cast<bf162*>(dst + 64)[lane] = __floats2bfloat162_rn(o[2], o[3]);
  }
  // Reduce dw across warps then one atomic per element per block.
  red[0][wid][2 * lane] = accq[0];
  red[0][wid][2 * lane + 1] = accq[1];
  red[0][wid][64 + 2 * lane] = accq[2];
  red[0][wid][64 + 2 * lane + 1] = accq[3];
  red[1][wid][2 * lane] = acck[0];
  red[1][wid][2 * lane + 1] = acck[1];
  red[1][wid][64 + 2 * lane] = acck[2];
  red[1][wid][64 + 2 * lane + 1] = acck[3];
  __syncthreads();
  for (int i = threadIdx.x; i < 2 * D; i += blockDim.x) {
    int which = i / D, d = i % D;
    float sum = 0.f;
#pragma unroll
    for (int w2 = 0; w2 < BWD_WARPS; ++w2) sum += red[which][w2][d];
    atomicAdd((which ? dwk : dwq) + d, sum);
  }
}

}  // namespace

void qk_norm_rope_fwd(const bf16* qkv, const int* pos, const bf16* wq, const bf16* wk, bf16* q, bf16* k, bf16* v,
                      float* rstd_q, float* rstd_k, int B, int S, int Hq, int Hkv, int kv_S, int kv_off, float theta,
                      float eps, cudaStream_t st) {
  long items = (long)B * S * (Hq + 2 * Hkv);
  if (items == 0) return;
  constexpr int WPB = 8;
  int grid = (items + WPB - 1) / WPB;
  qk_norm_rope_fwd_kernel<<<grid, WPB * 32, 0, st>>>(qkv, pos, wq, wk, q, k, v, rstd_q, rstd_k, B * S, S, Hq, Hkv,
                                                     kv_S, kv_off, theta, eps);
}

void qk_norm_rope_bwd(const float* dq, const bf16* dk, const bf16* dv, const bf16* qkv, const int* pos,
                      const bf16* wq, const bf16* wk, const float* rstd_q, const float* rstd_k, bf16* dqkv,
                      float* dwq, float* dwk, int B, int S, int Hq, int Hkv, float theta, int num_sms,
                      cudaStream_t st) {
  long items = (long)B * S * (Hq + 2 * Hkv);
  if (items == 0) return;
  long g1 = (items + BWD_WARPS - 1) / BWD_WARPS, g2 = (long)num_sms * 8;
  int grid = (int)(g1 < g2 ? g1 : g2);
  qk_norm_rope_bwd_kernel<<<grid, BWD_WARPS * 32, 0, st>>>(dq, dk, dv, qkv, pos, wq, wk, rstd_q, rstd_k, dqkv, dwq,
                                                            dwk, B * S, S, Hq, Hkv, theta);
}

}  // namespace qk
