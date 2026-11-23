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
//   dw = sum_rows dy*xhat : each CTA writes its column partials to dw_part[blockIdx.x]
//        (deterministic, no atomics), reduced by rmsnorm_dw_reduce_kernel.
// One wave of CTAs; each walks rows blockIdx.x, +gridDim.x, ... with the next row's
// loads issued before the current row's reduction.
template <int VEC>
__global__ void __launch_bounds__(THREADS)
rmsnorm_bwd_kernel(const bf16* __restrict__ dy, const bf16* __restrict__ x, const bf16* __restrict__ w,
                   const float* __restrict__ rstd, const bf16* __restrict__ dres_in, bf16* __restrict__ dx,
                   float* __restrict__ dw_part, int rows, int H) {
  __shared__ float red[32];
  const int nvec = H / 8;
  float wv[VEC][8], dwacc[VEC][8];
#pragma unroll
  for (int i = 0; i < VEC; ++i) {
    int c = threadIdx.x + i * THREADS;
#pragma unroll
    for (int k = 0; k < 8; ++k) dwacc[i][k] = 0.f;
    if (c < nvec) unpack8(reinterpret_cast<const uint4*>(w)[c], wv[i]);
  }
  uint4 nx[VEC], ng[VEC];
  auto fetch = [&](int row) {
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
      int c = threadIdx.x + i * THREADS;
      if (c < nvec && row < rows) {
        nx[i] = reinterpret_cast<const uint4*>(x + (size_t)row * H)[c];
        ng[i] = reinterpret_cast<const uint4*>(dy + (size_t)row * H)[c];
      }
    }
  };
  int row = blockIdx.x;
  fetch(row);
  for (; row < rows; row += gridDim.x) {
    const float r = rstd[row];
    float xh[VEC][8], gw[VEC][8];
    float dot = 0.f;
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
      int c = threadIdx.x + i * THREADS;
      if (c < nvec) {
        float g[8];
        unpack8(nx[i], xh[i]);
        unpack8(ng[i], g);
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          xh[i][k] *= r;
          gw[i][k] = g[k] * wv[i][k];
          dwacc[i][k] += g[k] * xh[i][k];
          dot += gw[i][k] * xh[i][k];
        }
      }
    }
    fetch(row + gridDim.x);  // in flight during the reduction below
    dot = block_sum(dot, red) / H;
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
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
  float* part = dw_part + (size_t)blockIdx.x * H;
#pragma unroll
  for (int i = 0; i < VEC; ++i) {
    int c = threadIdx.x + i * THREADS;
    if (c < nvec) {
      reinterpret_cast<float4*>(part + c * 8)[0] = make_float4(dwacc[i][0], dwacc[i][1], dwacc[i][2], dwacc[i][3]);
      reinterpret_cast<float4*>(part + c * 8)[1] = make_float4(dwacc[i][4], dwacc[i][5], dwacc[i][6], dwacc[i][7]);
    }
  }
}

__global__ void rmsnorm_dw_reduce_kernel(const float* __restrict__ part, float* __restrict__ dw, int nparts, int H) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= H) return;
  float s = 0.f;
  for (int b = 0; b < nparts; ++b) s += part[(size_t)b * H + c];
  dw[c] = s;
}

}  // namespace

void rmsnorm_fwd(const bf16* x, const bf16* res, const bf16* w, bf16* y, bf16* s_out, float* rstd, int rows,
                 int H, float eps, cudaStream_t st) {
  if (rows == 0) return;
  rmsnorm_fwd_kernel<<<rows, THREADS, 0, st>>>(x, res, w, y, s_out, rstd, H, eps);
}

int rmsnorm_bwd_parts(int rows, int num_sms) { return rows < num_sms * 2 ? rows : num_sms * 2; }

void rmsnorm_bwd(const bf16* dy, const bf16* x, const bf16* w, const float* rstd, const bf16* dres_in, bf16* dx,
                 float* dw, float* dw_part, int rows, int H, int num_sms, cudaStream_t st) {
  if (rows == 0) return;
  const int vec = (H / 8 + THREADS - 1) / THREADS;
  const int grid = rmsnorm_bwd_parts(rows, num_sms);
  if (vec <= 1) rmsnorm_bwd_kernel<1><<<grid, THREADS, 0, st>>>(dy, x, w, rstd, dres_in, dx, dw_part, rows, H);
  else if (vec == 2) rmsnorm_bwd_kernel<2><<<grid, THREADS, 0, st>>>(dy, x, w, rstd, dres_in, dx, dw_part, rows, H);
  else rmsnorm_bwd_kernel<4><<<grid, THREADS, 0, st>>>(dy, x, w, rstd, dres_in, dx, dw_part, rows, H);
  rmsnorm_dw_reduce_kernel<<<(H + 255) / 256, 256, 0, st>>>(dw_part, dw, grid, H);
}

}  // namespace qk
