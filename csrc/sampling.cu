// Token sampling from fp32 logits [B, V]: greedy, temperature, top-k, top-p.
//
// One 1024-thread block per row. Top-k / top-p thresholds are found exactly with
// a 4-pass 8-bit radix select over order-preserving uint32 keys of the logits
// (count histogram for top-k, probability-mass histogram for top-p), so no sort
// is needed. The token is then drawn by inverse CDF over the kept set in index
// order (block prefix scan), with Philox randomness keyed on (seed, row, offset).
#include <curand_kernel.h>

#include "common.cuh"
#include "kernels.h"

namespace qk {
namespace {

constexpr int T = 1024;
constexpr int CAP = 11264;   // max candidates kept in shared memory
constexpr float CUT = 30.f;  // candidates: scaled logit >= max - CUT (exp(-30) ~ 9e-14 weight each)

QK_DEVICE uint32_t ord_key(float x) {
  uint32_t b = __float_as_uint(x);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

struct Shared {
  float fhist[256];
  unsigned ihist[256];
  float red[32];
  float scan[T];
  uint32_t prefix;
  float above;
  int result;
};

// Find the largest key threshold `thr` such that the weight of {key >= thr} >= target.
// weight(i) = use_mass ? exp(x_i - M) : 1, restricted to key >= floor_key.
// Histograms are warp-private (smem `wh`: [32 warps][256] floats + [32][256] uints) so
// atomics only collide within a warp.
QK_DEVICE uint32_t radix_select(const float* x, int V, float inv_temp, float M, uint32_t floor_key, bool use_mass,
                                float target, Shared& sh, float* wh) {
  float* fh = wh + (threadIdx.x >> 5) * 256;
  unsigned* ih = reinterpret_cast<unsigned*>(wh + 32 * 256) + (threadIdx.x >> 5) * 256;
  uint32_t prefix = 0, mask = 0;
  float above = 0.f;
  for (int pass = 0; pass < 4; ++pass) {
    const int shift = 24 - 8 * pass;
    for (int i = threadIdx.x; i < 32 * 256 * 2; i += T) wh[i] = 0.f;
    __syncthreads();
    for (int i0 = 0; i0 < V; i0 += T) {
      const int i = i0 + threadIdx.x;
      int d = -1;
      float wgt = 0.f;
      if (i < V) {
        const float v = x[i] * inv_temp;
        const uint32_t u = ord_key(v);
        if ((u & mask) == prefix && u >= floor_key) {
          d = (u >> shift) & 255;
          if (use_mass) wgt = __expf(v - M);
        }
      }
      if (d >= 0) {
        if (use_mass) atomicAdd(&fh[d], wgt);
        else atomicAdd(&ih[d], 1u);
      }
    }
    __syncthreads();
    if (threadIdx.x < 256) {
      float fs = 0.f;
      unsigned is = 0;
#pragma unroll 8
      for (int w = 0; w < 32; ++w) {
        fs += wh[w * 256 + threadIdx.x];
        is += reinterpret_cast<unsigned*>(wh + 32 * 256)[w * 256 + threadIdx.x];
      }
      sh.fhist[threadIdx.x] = fs;
      sh.ihist[threadIdx.x] = is;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      float cum = above;
      int digit = 0;
      for (int dd = 255; dd >= 0; --dd) {
        float h = use_mass ? sh.fhist[dd] : (float)sh.ihist[dd];
        if (cum + h >= target) {
          digit = dd;
          break;
        }
        cum += h;
      }
      sh.prefix = prefix | ((uint32_t)digit << shift);
      sh.above = cum;
    }
    __syncthreads();
    prefix = sh.prefix;
    above = sh.above;
    mask |= 255u << shift;
    __syncthreads();
  }
  return prefix;
}

__global__ void __launch_bounds__(T)
sample_kernel(const float* __restrict__ logits, int V, long ld, const float* __restrict__ temperature,
              const int* __restrict__ top_k, const float* __restrict__ top_p, unsigned long long seed,
              unsigned long long offset, long* __restrict__ out) {
  __shared__ Shared sh;
  const int row = blockIdx.x;
  const float* x = logits + row * ld;
  const float temp = temperature ? temperature[row] : 0.f;

  if (!(temp > 0.f)) {  // greedy argmax (lowest index on ties)
    float best = -INFINITY;
    int bi = 0x7fffffff;
    for (int i = threadIdx.x; i < V; i += T) {
      float v = x[i];
      if (v > best) { best = v; bi = i; }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      float ov = __shfl_xor_sync(0xffffffff, best, o);
      int oi = __shfl_xor_sync(0xffffffff, bi, o);
      if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
    }
    __shared__ float bv[32];
    __shared__ int bix[32];
    if ((threadIdx.x & 31) == 0) { bv[threadIdx.x >> 5] = best; bix[threadIdx.x >> 5] = bi; }
    __syncthreads();
    if (threadIdx.x < 32) {
      best = bv[threadIdx.x];
      bi = bix[threadIdx.x];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        float ov = __shfl_xor_sync(0xffffffff, best, o);
        int oi = __shfl_xor_sync(0xffffffff, bi, o);
        if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
      }
      if (threadIdx.x == 0) out[row] = bi;
    }
    return;
  }

  const float inv_temp = 1.f / temp;
  float mx = -INFINITY;
  for (int i = threadIdx.x; i < V; i += T) mx = fmaxf(mx, x[i] * inv_temp);
  const float M = block_max(mx, sh.red);

  // Compact candidates (index order, deterministic) into shared memory. Tokens below
  // max - CUT carry < V * e^-30 ~ 1e-8 of the mass in total: below fp32 summation error.
  // If there are too many (very flat distribution), fall back to the global-memory path.
  extern __shared__ float cand[];
  int* cidx = reinterpret_cast<int*>(cand + CAP);
  float* wh = cand + 2 * CAP;  // warp-private radix histograms
  const float* src = x;
  int n = V;
  float inv = inv_temp;
  bool compact = false;
  {
    // Thread-major order over coalesced strided indices (i = tid + k*T): a fixed,
    // deterministic permutation, which is all inverse-CDF sampling needs.
    const float cut = M - CUT;
    int cnt = 0;
    for (int i = threadIdx.x; i < V; i += T) cnt += (x[i] * inv_temp >= cut);
    sh.scan[threadIdx.x] = (float)cnt;
    __syncthreads();
    for (int o = 1; o < T; o <<= 1) {
      float add = threadIdx.x >= o ? sh.scan[threadIdx.x - o] : 0.f;
      __syncthreads();
      sh.scan[threadIdx.x] += add;
      __syncthreads();
    }
    const int total = (int)sh.scan[T - 1];
    if (total <= CAP) {
      int o = (int)sh.scan[threadIdx.x] - cnt;
      for (int i = threadIdx.x; i < V; i += T) {
        const float v = x[i] * inv_temp;
        if (v >= cut) {
          cand[o] = v;
          cidx[o] = i;
          ++o;
        }
      }
      src = cand;
      n = total;
      inv = 1.f;
      compact = true;
    }
    __syncthreads();
  }

  uint32_t thr = 0;
  const int k = top_k ? top_k[row] : 0;
  if (k > 0 && k < n) thr = radix_select(src, n, inv, M, 0u, false, (float)k, sh, wh);

  const float p = top_p ? top_p[row] : 1.f;
  if (p < 1.f) {
    float z = 0.f;
    for (int i = threadIdx.x; i < n; i += T) {
      float v = src[i] * inv;
      if (ord_key(v) >= thr) z += __expf(v - M);
    }
    z = block_sum(z, sh.red);
    uint32_t thr_p = radix_select(src, n, inv, M, thr, true, p * z, sh, wh);
    thr = max(thr, thr_p);
  }

  // Inverse-CDF draw over kept tokens, in index order. Thread i owns a contiguous segment.
  float local = 0.f;
  for (int i = threadIdx.x; i < n; i += T) {
    float v = src[i] * inv;
    if (ord_key(v) >= thr) local += __expf(v - M);
  }
  sh.scan[threadIdx.x] = local;
  if (threadIdx.x == 0) sh.result = -1;
  __syncthreads();
  // Inclusive Hillis-Steele scan.
  for (int o = 1; o < T; o <<= 1) {
    float add = threadIdx.x >= o ? sh.scan[threadIdx.x - o] : 0.f;
    __syncthreads();
    sh.scan[threadIdx.x] += add;
    __syncthreads();
  }
  const float total = sh.scan[T - 1];
  float u;
  {
    curandStatePhilox4_32_10_t rs;
    curand_init(seed, row, offset, &rs);
    u = curand_uniform(&rs);  // (0, 1]
  }
  const float r = (1.f - u) * total;  // [0, total)
  const float incl = sh.scan[threadIdx.x];
  const float excl = incl - local;
  if (local > 0.f && r >= excl && r < incl) {
    float c = excl;
    int pick = -1, last = -1;
    for (int i = threadIdx.x; i < n; i += T) {
      float v = src[i] * inv;
      if (ord_key(v) >= thr) {
        last = i;
        c += __expf(v - M);
        if (r < c) { pick = i; break; }
      }
    }
    sh.result = pick >= 0 ? pick : last;  // rounding fallback
  }
  __syncthreads();
  if (sh.result < 0) {  // r fell past the end through rounding: last kept token in the order
    __shared__ int last_t;
    if (threadIdx.x == 0) last_t = -1;
    __syncthreads();
    if (local > 0.f) atomicMax(&last_t, (int)threadIdx.x);
    __syncthreads();
    if ((int)threadIdx.x == last_t) {
      int last = -1;
      for (int i = threadIdx.x; i < n; i += T)
        if (ord_key(src[i] * inv) >= thr) last = i;
      sh.result = last;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) out[row] = (compact && sh.result >= 0) ? cidx[sh.result] : sh.result;
}

}  // namespace

void sample_tokens(const float* logits, long B, int V, long ld, const float* temperature, const int* top_k,
                   const float* top_p, unsigned long long seed, unsigned long long offset, long* out,
                   cudaStream_t st) {
  if (B == 0) return;
  constexpr int smem = CAP * 8 + 32 * 256 * 8;
  static bool cfg = false;
  if (!cfg) {
    cudaFuncSetAttribute(sample_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    cfg = true;
  }
  sample_kernel<<<B, T, smem, st>>>(logits, V, ld, temperature, top_k, top_p, seed, offset, out);
}

}  // namespace qk
