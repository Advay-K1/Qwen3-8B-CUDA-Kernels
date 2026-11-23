// RMSNorm forward / backward, with optional fused residual add.
//
// Qwen3RMSNorm:  y = w * bf16( x * rsqrt(mean(x^2) + eps) )
// Fused add:     s = bf16(x + res);  y = rmsnorm(s)   (s is the new residual stream)
#include "common.cuh"
#include "kernels.h"

namespace qk {
namespace {

constexpr int THREADS = 256;
constexpr int MAX_VEC = 4;  // up to 4 x 8 elements per thread -> H <= 8192 at 256 threads

// One block per row. H % 8 == 0, H <= THREADS*8*MAX_VEC.
__global__ void __launch_bounds__(THREADS)
rmsnorm_fwd_kernel(const bf16* __restrict__ x, const bf16* __restrict__ res, const bf16* __restrict__ w,
                   bf16* __restrict__ y, bf16* __restrict__ s_out, float* __restrict__ rstd_out,
                   int H, float eps) {
  __shared__ float red[32];
  const int row = blockIdx.x;
  const bf16* xr = x + (size_t)row * H;
  const int nvec = H / 8;
  float v[MAX_VEC][8];
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < MAX_VEC; ++i) {
    int c = threadIdx.x + i * THREADS;
    if (c < nvec) {
      unpack8(reinterpret_cast<const uint4*>(xr)[c], v[i]);
      if (res) {
        float r[8];
        unpack8(reinterpret_cast<const uint4*>(res + (size_t)row * H)[c], r);
#pragma unroll
        for (int k = 0; k < 8; ++k) v[i][k] = bf16_round(v[i][k] + r[k]);
        reinterpret_cast<uint4*>(s_out + (size_t)row * H)[c] = pack8(v[i]);
      }
#pragma unroll
      for (int k = 0; k < 8; ++k) ss += v[i][k] * v[i][k];
    }
  }
  ss = block_sum(ss, red);
  const float rstd = rsqrtf(ss / H + eps);
  if (threadIdx.x == 0 && rstd_out) rstd_out[row] = rstd;
#pragma unroll
  for (int i = 0; i < MAX_VEC; ++i) {
    int c = threadIdx.x + i * THREADS;
    if (c < nvec) {
      float wv[8], o[8];
      unpack8(reinterpret_cast<const uint4*>(w)[c], wv);
#pragma unroll
      for (int k = 0; k < 8; ++k) o[k] = wv[k] * bf16_round(v[i][k] * rstd);
      reinterpret_cast<uint4*>(y + (size_t)row * H)[c] = pack8(o);
    }
  }
}

// Backward. x is the normalized input (s for the fused variant).
//   xhat = x*rstd, gw = dy*w
//   dx = rstd * (gw - xhat * mean(gw*xhat))   (+ dres_in, the residual-stream grad)
//   dw += sum_rows dy*xhat                   (fp32 atomics, one per column per block)
// Each block processes rows blockIdx.x, blockIdx.x+gridDim.x, ...
__global__ void __launch_bounds__(THREADS)
rmsnorm_bwd_kernel(const bf16* __restrict__ dy, const bf16* __restrict__ x, const bf16* __restrict__ w,
                   const float* __restrict__ rstd, const bf16* __restrict__ dres_in,
                   bf16* __restrict__ dx, float* __restrict__ dw, int rows, int H) {
  __shared__ float red[32];
  const int nvec = H / 8;
  float wv[MAX_VEC][8], dwacc[MAX_VEC][8];
#pragma unroll
  for (int i = 0; i < MAX_VEC; ++i) {
    int c = threadIdx.x + i * THREADS;
#pragma unroll
    for (int k = 0; k < 8; ++k) dwacc[i][k] = 0.f;
    if (c < nvec) unpack8(reinterpret_cast<const uint4*>(w)[c], wv[i]);
  }
  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    const float r = rstd[row];
    float xh[MAX_VEC][8], gw[MAX_VEC][8];
    float dot = 0.f;
#pragma unroll
    for (int i = 0; i < MAX_VEC; ++i) {
      int c = threadIdx.x + i * THREADS;
      if (c < nvec) {
        float g[8];
        unpack8(reinterpret_cast<const uint4*>(x + (size_t)row * H)[c], xh[i]);
        unpack8(reinterpret_cast<const uint4*>(dy + (size_t)row * H)[c], g);
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          xh[i][k] *= r;
          gw[i][k] = g[k] * wv[i][k];
          dwacc[i][k] += g[k] * xh[i][k];
          dot += gw[i][k] * xh[i][k];
        }
      }
    }
    dot = block_sum(dot, red) / H;
#pragma unroll
    for (int i = 0; i < MAX_VEC; ++i) {
      int c = threadIdx.x + i * THREADS;
      if (c < nvec) {
        float o[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) o[k] = r * (gw[i][k] - xh[i][k] * dot);
        if (dres_in) {
          float d2[8];
          unpack8(reinterpret_cast<const uint4*>(dres_in + (size_t)row * H)[c], d2);
#pragma unroll
          for (int k = 0; k < 8; ++k) o[k] += d2[k];
        }
        reinterpret_cast<uint4*>(dx + (size_t)row * H)[c] = pack8(o);
      }
    }
  }
#pragma unroll
  for (int i = 0; i < MAX_VEC; ++i) {
    int c = threadIdx.x + i * THREADS;
    if (c < nvec)
#pragma unroll
      for (int k = 0; k < 8; ++k) atomicAdd(dw + c * 8 + k, dwacc[i][k]);
  }
}

}  // namespace

void rmsnorm_fwd(const bf16* x, const bf16* res, const bf16* w, bf16* y, bf16* s_out, float* rstd, int rows,
                 int H, float eps, cudaStream_t st) {
  if (rows == 0) return;
  rmsnorm_fwd_kernel<<<rows, THREADS, 0, st>>>(x, res, w, y, s_out, rstd, H, eps);
}

void rmsnorm_bwd(const bf16* dy, const bf16* x, const bf16* w, const float* rstd, const bf16* dres_in, bf16* dx,
                 float* dw, int rows, int H, int num_sms, cudaStream_t st) {
  if (rows == 0) return;
  int grid = min(rows, num_sms * 4);
  rmsnorm_bwd_kernel<<<grid, THREADS, 0, st>>>(dy, x, w, rstd, dres_in, dx, dw, rows, H);
}

}  // namespace qk
