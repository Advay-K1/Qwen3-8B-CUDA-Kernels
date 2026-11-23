// Decode-time kernels (small batch): GEMV and split-KV attention over a KV cache.
#include "common.cuh"
#include "kernels.h"

namespace qk {
namespace {

constexpr int D = 128;

// y[b, n] = sum_k W[n, k] x[b, k]  (+ R[b, n]),  b < NB, one warp per output row.
// W bf16 [N, K] row-major; x bf16 [NB, K]; y fp32 or bf16 [NB, ldy].
template <int NB>
__global__ void __launch_bounds__(256)
gemv_kernel(const bf16* __restrict__ W, const bf16* __restrict__ x, void* __restrict__ y, const bf16* __restrict__ R,
            int N, int K, int ldy, bool out_f32) {
  const int lane = threadIdx.x & 31;
  const int warps = (gridDim.x * blockDim.x) >> 5;
  const int nv = K / 8;
  for (int n = (blockIdx.x * blockDim.x + threadIdx.x) >> 5; n < N; n += warps) {
    const uint4* wr = reinterpret_cast<const uint4*>(W + (size_t)n * K);
    float acc[NB];
#pragma unroll
    for (int b = 0; b < NB; ++b) acc[b] = 0.f;
#pragma unroll 4
    for (int c = lane; c < nv; c += 32) {
      float wf[8];
      unpack8(__ldg(wr + c), wf);
#pragma unroll
      for (int b = 0; b < NB; ++b) {
        float xf[8];
        unpack8(__ldg(reinterpret_cast<const uint4*>(x + (size_t)b * K) + c), xf);
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[b] = fmaf(wf[i], xf[i], acc[b]);
      }
    }
#pragma unroll
    for (int b = 0; b < NB; ++b) {
      float v = warp_sum(acc[b]);
      if (lane == 0) {
        if (R) v += __bfloat162float(R[(size_t)b * ldy + n]);
        if (out_f32) reinterpret_cast<float*>(y)[(size_t)b * ldy + n] = v;
        else reinterpret_cast<bf16*>(y)[(size_t)b * ldy + n] = __float2bfloat16(v);
      }
    }
  }
}

// Split-KV decode attention. grid (nsplit, Hkv, B), 4 warps; each CTA handles all
// n_rep (=Hq/Hkv, <= 8) query heads of one KV head over one key chunk.
// q [B, Hq, D] bf16;  k/v cache [B, Hkv, Smax, D];  lens[b] = number of valid keys.
// part: [B, Hq, nsplit, D + 2]  (o unnormalized, then m, l)
constexpr int DW = 4;
template <int NREP>
__global__ void __launch_bounds__(DW * 32)
decode_attn_kernel(const bf16* __restrict__ q, const bf16* __restrict__ kc, const bf16* __restrict__ vc,
                   const int* __restrict__ lens, float* __restrict__ part, int Hq, int Hkv, int Smax, int chunk,
                   int nsplit, float scale_log2) {
  __shared__ float sq[NREP][D];
  __shared__ float so[DW][NREP][D];
  __shared__ float sm[DW][NREP], sl[DW][NREP];
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const int split = blockIdx.x, hk = blockIdx.y, b = blockIdx.z;
  const int L = lens[b];
  const int k_beg = split * chunk, k_end = min(L, k_beg + chunk);
  for (int i = threadIdx.x; i < NREP * D; i += blockDim.x)
    sq[i / D][i % D] = __bfloat162float(q[((size_t)b * Hq + hk * NREP + i / D) * D + i % D]);
  __syncthreads();

  float m[NREP], l[NREP], o[NREP][4];
#pragma unroll
  for (int h = 0; h < NREP; ++h) {
    m[h] = -INFINITY;
    l[h] = 0.f;
    o[h][0] = o[h][1] = o[h][2] = o[h][3] = 0.f;
  }
  const bf16* kb = kc + (size_t)(b * Hkv + hk) * Smax * D;
  const bf16* vb = vc + (size_t)(b * Hkv + hk) * Smax * D;
  for (int base = k_beg + w * 32; base < k_end; base += DW * 32) {
    const int key = base + lane;
    const bool valid = key < k_end;
    float s[NREP];
#pragma unroll
    for (int h = 0; h < NREP; ++h) s[h] = 0.f;
    if (valid) {
      const uint4* kr = reinterpret_cast<const uint4*>(kb + (size_t)key * D);
#pragma unroll
      for (int c = 0; c < D / 8; ++c) {
        float kf[8];
        unpack8(kr[c], kf);
#pragma unroll
        for (int h = 0; h < NREP; ++h)
#pragma unroll
          for (int i = 0; i < 8; ++i) s[h] = fmaf(kf[i], sq[h][c * 8 + i], s[h]);
      }
    }
    float p[NREP];
#pragma unroll
    for (int h = 0; h < NREP; ++h) {
      float v = valid ? s[h] * scale_log2 : -INFINITY;
      float mx = warp_max(v);
      float mn = fmaxf(m[h], mx);
      float a = exp2f(m[h] - mn);
      p[h] = valid ? exp2f(v - mn) : 0.f;
      l[h] = l[h] * a + warp_sum(p[h]);
      m[h] = mn;
#pragma unroll
      for (int i = 0; i < 4; ++i) o[h][i] *= a;
    }
    const int nk = min(32, k_end - base);
    for (int j = 0; j < nk; ++j) {
      float2 v01, v23;
      {
        const bf162* vr = reinterpret_cast<const bf162*>(vb + (size_t)(base + j) * D + lane * 4);
        v01 = __bfloat1622float2(vr[0]);
        v23 = __bfloat1622float2(vr[1]);
      }
#pragma unroll
      for (int h = 0; h < NREP; ++h) {
        float pj = __shfl_sync(0xffffffff, p[h], j);
        o[h][0] = fmaf(pj, v01.x, o[h][0]);
        o[h][1] = fmaf(pj, v01.y, o[h][1]);
        o[h][2] = fmaf(pj, v23.x, o[h][2]);
        o[h][3] = fmaf(pj, v23.y, o[h][3]);
      }
    }
  }
  // Combine warps.
#pragma unroll
  for (int h = 0; h < NREP; ++h) {
#pragma unroll
    for (int i = 0; i < 4; ++i) so[w][h][lane * 4 + i] = o[h][i];
    if (lane == 0) {
      sm[w][h] = m[h];
      sl[w][h] = l[h];
    }
  }
  __syncthreads();
  for (int i = threadIdx.x; i < NREP * D; i += blockDim.x) {
    int h = i / D, d = i % D;
    float M = -INFINITY;
#pragma unroll
    for (int ww = 0; ww < DW; ++ww) M = fmaxf(M, sm[ww][h]);
    float acc = 0.f, lsum = 0.f;
#pragma unroll
    for (int ww = 0; ww < DW; ++ww) {
      float a = (sm[ww][h] == -INFINITY) ? 0.f : exp2f(sm[ww][h] - M);
      acc += so[ww][h][d] * a;
      lsum += sl[ww][h] * a;
    }
    float* pp = part + (((size_t)b * Hq + hk * NREP + h) * nsplit + split) * (D + 2);
    pp[d] = acc;
    if (d == 0) {
      pp[D] = M;
      pp[D + 1] = lsum;
    }
  }
}

// out [B, Hq, D] bf16 (== [B, Hq*D]) from split partials.
__global__ void decode_combine_kernel(const float* __restrict__ part, bf16* __restrict__ out, int nsplit) {
  const int bh = blockIdx.x, d = threadIdx.x;  // blockDim = D
  const float* pp = part + (size_t)bh * nsplit * (D + 2);
  float M = -INFINITY;
  for (int s = 0; s < nsplit; ++s) M = fmaxf(M, pp[s * (D + 2) + D]);
  float acc = 0.f, l = 0.f;
  for (int s = 0; s < nsplit; ++s) {
    float ms = pp[s * (D + 2) + D];
    float a = (ms == -INFINITY) ? 0.f : exp2f(ms - M);
    acc += pp[s * (D + 2) + d] * a;
    l += pp[s * (D + 2) + D + 1] * a;
  }
  out[(size_t)bh * D + d] = __float2bfloat16(l > 0.f ? acc / l : 0.f);
}

}  // namespace

void gemv(const bf16* W, const bf16* x, void* y, const bf16* R, int NB, int N, int K, int ldy, bool out_f32,
          int num_sms, cudaStream_t st) {
  int grid = min((N + 7) / 8, num_sms * 8);
  switch (NB) {
#define CASE(n) \
  case n: gemv_kernel<n><<<grid, 256, 0, st>>>(W, x, y, R, N, K, ldy, out_f32); break;
    CASE(1) CASE(2) CASE(3) CASE(4) CASE(5) CASE(6) CASE(7) CASE(8)
#undef CASE
    default: break;
  }
}

int decode_attn_num_splits(int max_len, int chunk) { return (max_len + chunk - 1) / chunk; }

void decode_attn(const bf16* q, const bf16* kc, const bf16* vc, const int* lens, float* part, bf16* out, int B,
                 int Hq, int Hkv, int Smax, int max_len, int chunk, float scale, cudaStream_t st) {
  int nsplit = decode_attn_num_splits(max_len, chunk);
  dim3 grid(nsplit, Hkv, B);
  const float sl2 = scale * 1.4426950408889634f;
  switch (Hq / Hkv) {
#define CASE(n) \
  case n: decode_attn_kernel<n><<<grid, DW * 32, 0, st>>>(q, kc, vc, lens, part, Hq, Hkv, Smax, chunk, nsplit, sl2); break;
    CASE(1) CASE(2) CASE(4) CASE(8)
#undef CASE
    default: break;
  }
  decode_combine_kernel<<<B * Hq, D, 0, st>>>(part, out, nsplit);
}

}  // namespace qk
