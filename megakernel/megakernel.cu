// Qwen3-8B decode megakernel.
//
// One persistent cooperative kernel runs a full decode step for NB (1..4) sequences:
//
//   [embed] -> for each layer:
//       S1  input RMSNorm (per CTA, redundant) + QKV GEMV                       | grid barrier
//       S2  attention partials: q/k RMSNorm + RoPE + KV-cache append fused in,  | grid barrier
//           split-KV online softmax over the cache
//       S3  combine split partials -> attention output                         | grid barrier
//       S4  O-proj GEMV + residual                                              | grid barrier
//       S5  post RMSNorm + gate/up GEMV + SwiGLU                                | grid barrier
//       S6  down GEMV + residual                                                | grid barrier
//   -> [final RMSNorm + LM-head GEMV -> fp32 logits]
//
// GEMV work: each CTA owns a contiguous slice of output rows; inside a CTA the
// warps share (row, K-segment) units and partial dot products are reduced in
// shared memory in a fixed order (deterministic, well balanced).
// bf16 rounding points match the layer-by-layer kernels (Qwen3Block.decode).
#include <cooperative_groups.h>

#include "common.cuh"
#include "megakernel.h"

namespace mk {
using namespace qk;

constexpr int H = 4096, I = 12288, HQ = 32, HKV = 8, D = 128, NREP = HQ / HKV, QKV = (HQ + 2 * HKV) * D;
constexpr int THREADS = MK_THREADS, WARPS = THREADS / 32;
constexpr int SEG = 2048;       // K elements per GEMV work unit (8 x 16B per lane)
constexpr int RPP = 256;        // rows per pass in block_gemv
constexpr int MAX_SEGS = I / SEG;

// ------------------------------------------------------------------ memory helpers
QK_DEVICE uint4 ld_stream(const void* p) {  // weights: read once, don't pollute L1
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p));
  return r;
}
QK_DEVICE uint4 ld_cg(const void* p) { return __ldcg(reinterpret_cast<const uint4*>(p)); }
QK_DEVICE float2 ld_cg_bf2(const bf16* p) {
  unsigned u = __ldcg(reinterpret_cast<const unsigned*>(p));
  return __bfloat1622float2(*reinterpret_cast<bf162*>(&u));
}

// Pull `bytes` starting at p into L2 (no registers / smem consumed).
QK_DEVICE void prefetch_l2(const void* p, size_t bytes) {
  const char* c = reinterpret_cast<const char*>(p);
  for (size_t off = (size_t)threadIdx.x * 128; off < bytes; off += (size_t)THREADS * 128)
    asm volatile("prefetch.global.L2 [%0];\n" ::"l"(c + off));
}

QK_DEVICE unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
  return t;
}

QK_DEVICE void grid_sync(unsigned* bar, unsigned& target) {
  __syncthreads();
  if (threadIdx.x == 0) {
    target += gridDim.x;
    __threadfence();
    atomicAdd(bar, 1u);
    unsigned v;
    do {
      asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n" : "=r"(v) : "l"(bar) : "memory");
    } while (v < target);
    __threadfence();
  }
  __syncthreads();
}

QK_DEVICE void rows_of(int N, int& r0, int& r1) {
  int per = (N + gridDim.x - 1) / gridDim.x;
  r0 = min(N, (int)blockIdx.x * per);
  r1 = min(N, r0 + per);
}

// Prefetch the first `rows` rows of this CTA's slice of an [N, K] weight.
QK_DEVICE void prefetch_slice(const bf16* W, int N, int K, int rows) {
  int r0, r1;
  rows_of(N, r0, r1);
  int n = min(rows, r1 - r0);
  if (n > 0) prefetch_l2(W + (size_t)r0 * K, (size_t)n * K * 2);
}

// ------------------------------------------------------------------ GEMV building blocks
// acc[b] = dot(W[row, seg], sx[b, seg]) over SEG elements; result valid in all lanes.
template <int NB>
QK_DEVICE void gemv_unit(const bf16* __restrict__ w, const bf16* sx, int K, float* acc) {
  const int lane = threadIdx.x & 31;
  uint4 wv[SEG / 256];
#pragma unroll
  for (int i = 0; i < SEG / 256; ++i) wv[i] = ld_stream(w + (i * 32 + lane) * 8);
#pragma unroll
  for (int b = 0; b < NB; ++b) acc[b] = 0.f;
#pragma unroll
  for (int i = 0; i < SEG / 256; ++i) {
    float wf[8];
    unpack8(wv[i], wf);
#pragma unroll
    for (int b = 0; b < NB; ++b) {
      float xf[8];
      unpack8(*reinterpret_cast<const uint4*>(sx + (size_t)b * K + (i * 32 + lane) * 8), xf);
#pragma unroll
      for (int k = 0; k < 8; ++k) acc[b] = fmaf(wf[k], xf[k], acc[b]);
    }
  }
#pragma unroll
  for (int b = 0; b < NB; ++b) acc[b] = warp_sum(acc[b]);
}

// y[row, b] = W[row, :] . sx[b, :] for rows [r0, r1); epi(row, b, value) per output.
template <int NB, class Epi>
QK_DEVICE void block_gemv(const bf16* __restrict__ W, int K, int r0, int r1, const bf16* sx, float* spart, Epi epi) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int nseg = K / SEG;
  for (int base = r0; base < r1; base += RPP) {
    const int nr = min(RPP, r1 - base);
    const int units = nr * nseg;
    for (int u = warp; u < units; u += WARPS) {
      const int r = u / nseg, s = u - r * nseg;
      float acc[NB];
      gemv_unit<NB>(W + (size_t)(base + r) * K + s * SEG, sx + s * SEG, K, acc);
      if (lane < NB) {
        float v = acc[0];
#pragma unroll
        for (int b = 1; b < NB; ++b)
          if (lane == b) v = acc[b];
        spart[(r * nseg + s) * NB + lane] = v;
      }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < nr * NB; i += THREADS) {
      const int r = i / NB, b = i - r * NB;
      float v = 0.f;
      for (int s = 0; s < nseg; ++s) v += spart[(r * nseg + s) * NB + b];
      epi(base + r, b, v);
    }
    __syncthreads();
  }
}

// sx[b, :] = bf16( w * bf16(x[b,:] * rstd_b) )  (Qwen3RMSNorm), x read through L2.
template <int NB>
QK_DEVICE void load_norm(const bf16* x, const bf16* __restrict__ w, bf16* sx, float eps, float* red) {
  static_assert(H / 8 == THREADS, "one 16B chunk per thread");
  const int c = threadIdx.x;
  float wv[8];
  unpack8(__ldg(reinterpret_cast<const uint4*>(w) + c), wv);
#pragma unroll
  for (int b = 0; b < NB; ++b) {
    float v[8];
    unpack8(ld_cg(x + (size_t)b * H + c * 8), v);
    float ss = 0.f;
#pragma unroll
    for (int k = 0; k < 8; ++k) ss += v[k] * v[k];
    ss = block_sum(ss, red);
    const float r = rsqrtf(ss / H + eps);
    float o[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) o[k] = wv[k] * bf16_round(v[k] * r);
    reinterpret_cast<uint4*>(sx + (size_t)b * H)[c] = pack8(o);
  }
  __syncthreads();
}

template <int NB>
QK_DEVICE void load_vec(const bf16* src, int K, bf16* sx) {
  for (int i = threadIdx.x; i < NB * K / 8; i += THREADS)
    reinterpret_cast<uint4*>(sx)[i] = ld_cg(src + (size_t)i * 8);
  __syncthreads();
}

// ------------------------------------------------------------------ attention pieces
// Per-warp head RMSNorm + RoPE (lane owns d = 2l, 2l+1, 64+2l, 64+2l+1). Returns bf16-rounded values.
QK_DEVICE void head_norm_rope(const bf16* src, const bf16* __restrict__ w, int pos, float theta, float eps,
                              float* o) {
  const int lane = threadIdx.x & 31;
  float2 lo = ld_cg_bf2(src + 2 * lane), hi = ld_cg_bf2(src + 64 + 2 * lane);
  float x[4] = {lo.x, lo.y, hi.x, hi.y};
  float ss = warp_sum(x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]);
  float r = rsqrtf(ss / D + eps);
  float2 wl = __bfloat1622float2(reinterpret_cast<const bf162*>(w)[lane]);
  float2 wh = __bfloat1622float2(reinterpret_cast<const bf162*>(w + 64)[lane]);
  float n0 = bf16_round(x[0] * r) * wl.x, n1 = bf16_round(x[1] * r) * wl.y;
  float n2 = bf16_round(x[2] * r) * wh.x, n3 = bf16_round(x[3] * r) * wh.y;
  float c[2], s[2];
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    int i = 2 * lane + j;
    float inv = 1.0f / powf(theta, (float)(2 * i) / (float)D);
    sincosf((float)pos * inv, &s[j], &c[j]);
  }
  o[0] = bf16_round(n0 * c[0] - n2 * s[0]);
  o[1] = bf16_round(n1 * c[1] - n3 * s[1]);
  o[2] = bf16_round(n2 * c[0] + n0 * s[0]);
  o[3] = bf16_round(n3 * c[1] + n1 * s[1]);
}

struct AttnSmem {
  float q[NREP][D];
  float o[WARPS][NREP][D];
  float m[WARPS][NREP], l[WARPS][NREP];
};

template <int NB>
QK_DEVICE void attn_item(const MegaParams& p, const LayerW& L, int layer, int item, AttnSmem& sm) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const int split = item % p.nsplit;
  const int bh = item / p.nsplit;
  const int hk = bh % HKV, b = bh / HKV;
  const int pos = p.pos, Lk = pos + 1;
  const int k_beg = split * p.chunk, k_end = min(Lk, k_beg + p.chunk);
  const size_t cache_off = (((size_t)layer * p.B + b) * HKV + hk) * p.Smax * D;
  bf16* kc = p.k_cache + cache_off;
  bf16* vc = p.v_cache + cache_off;
  const bf16* qkv = p.qkv + (size_t)b * QKV;

  if (w < NREP) {
    float o[4];
    head_norm_rope(qkv + (hk * NREP + w) * D, L.q_norm, pos, p.theta, p.eps, o);
    sm.q[w][2 * lane] = o[0];
    sm.q[w][2 * lane + 1] = o[1];
    sm.q[w][64 + 2 * lane] = o[2];
    sm.q[w][64 + 2 * lane + 1] = o[3];
  } else if (w == NREP && k_end == Lk) {  // this split owns the new token: append K
    float o[4];
    head_norm_rope(qkv + (HQ + hk) * D, L.k_norm, pos, p.theta, p.eps, o);
    bf16* dst = kc + (size_t)pos * D;
    reinterpret_cast<bf162*>(dst)[lane] = __floats2bfloat162_rn(o[0], o[1]);
    reinterpret_cast<bf162*>(dst + 64)[lane] = __floats2bfloat162_rn(o[2], o[3]);
  } else if (w == NREP + 1 && k_end == Lk) {  // append V
    const bf16* src = qkv + (HQ + HKV + hk) * D;
    bf16* dst = vc + (size_t)pos * D;
    reinterpret_cast<unsigned*>(dst)[lane] = __ldcg(reinterpret_cast<const unsigned*>(src) + lane);
    reinterpret_cast<unsigned*>(dst)[32 + lane] = __ldcg(reinterpret_cast<const unsigned*>(src) + 32 + lane);
  }
  __syncthreads();

  const float sl2 = p.scale_log2;
  float m[NREP], l[NREP], o[NREP][4];
#pragma unroll
  for (int h = 0; h < NREP; ++h) {
    m[h] = -INFINITY;
    l[h] = 0.f;
    o[h][0] = o[h][1] = o[h][2] = o[h][3] = 0.f;
  }
  for (int base = k_beg + w * 32; base < k_end; base += WARPS * 32) {
    const int key = base + lane;
    const bool valid = key < k_end;
    float s[NREP];
#pragma unroll
    for (int h = 0; h < NREP; ++h) s[h] = 0.f;
    if (valid) {
      const uint4* kr = reinterpret_cast<const uint4*>(kc + (size_t)key * D);
#pragma unroll 4
      for (int c = 0; c < D / 8; ++c) {
        float kf[8];
        unpack8(__ldcg(kr + c), kf);
#pragma unroll
        for (int h = 0; h < NREP; ++h)
#pragma unroll
          for (int i = 0; i < 8; ++i) s[h] = fmaf(kf[i], sm.q[h][c * 8 + i], s[h]);
      }
    }
    float pr[NREP];
#pragma unroll
    for (int h = 0; h < NREP; ++h) {
      float v = valid ? s[h] * sl2 : -INFINITY;
      float mx = warp_max(v);
      float mn = fmaxf(m[h], mx);
      float a = exp2f(m[h] - mn);
      pr[h] = valid ? exp2f(v - mn) : 0.f;
      l[h] = l[h] * a + warp_sum(pr[h]);
      m[h] = mn;
#pragma unroll
      for (int i = 0; i < 4; ++i) o[h][i] *= a;
    }
    const int nk = min(32, k_end - base);
    for (int j = 0; j < nk; ++j) {
      const bf162* vr = reinterpret_cast<const bf162*>(vc + (size_t)(base + j) * D + lane * 4);
      unsigned u0 = __ldcg(reinterpret_cast<const unsigned*>(vr));
      unsigned u1 = __ldcg(reinterpret_cast<const unsigned*>(vr) + 1);
      float2 v01 = __bfloat1622float2(*reinterpret_cast<bf162*>(&u0));
      float2 v23 = __bfloat1622float2(*reinterpret_cast<bf162*>(&u1));
#pragma unroll
      for (int h = 0; h < NREP; ++h) {
        float pj = __shfl_sync(0xffffffff, pr[h], j);
        o[h][0] = fmaf(pj, v01.x, o[h][0]);
        o[h][1] = fmaf(pj, v01.y, o[h][1]);
        o[h][2] = fmaf(pj, v23.x, o[h][2]);
        o[h][3] = fmaf(pj, v23.y, o[h][3]);
      }
    }
  }
#pragma unroll
  for (int h = 0; h < NREP; ++h) {
#pragma unroll
    for (int i = 0; i < 4; ++i) sm.o[w][h][lane * 4 + i] = o[h][i];
    if (lane == 0) {
      sm.m[w][h] = m[h];
      sm.l[w][h] = l[h];
    }
  }
  __syncthreads();
  for (int i = threadIdx.x; i < NREP * D; i += THREADS) {
    const int h = i / D, d = i % D;
    float M = -INFINITY;
#pragma unroll
    for (int ww = 0; ww < WARPS; ++ww) M = fmaxf(M, sm.m[ww][h]);
    float acc = 0.f, ls = 0.f;
#pragma unroll
    for (int ww = 0; ww < WARPS; ++ww) {
      float a = (sm.m[ww][h] == -INFINITY) ? 0.f : exp2f(sm.m[ww][h] - M);
      acc += sm.o[ww][h][d] * a;
      ls += sm.l[ww][h] * a;
    }
    float* pp = p.attn_part + (((size_t)b * HQ + hk * NREP + h) * p.nsplit + split) * (D + 2);
    pp[d] = acc;
    if (d == 0) {
      pp[D] = M;
      pp[D + 1] = ls;
    }
  }
  __syncthreads();
}

QK_DEVICE void combine_item(const MegaParams& p, int bh, int d) {
  const float* pp = p.attn_part + (size_t)bh * p.nsplit * (D + 2);
  float M = -INFINITY;
  for (int s = 0; s < p.nsplit; ++s) M = fmaxf(M, __ldcg(pp + s * (D + 2) + D));
  float acc = 0.f, l = 0.f;
  for (int s = 0; s < p.nsplit; ++s) {
    float ms = __ldcg(pp + s * (D + 2) + D);
    float a = (ms == -INFINITY) ? 0.f : exp2f(ms - M);
    acc += __ldcg(pp + s * (D + 2) + d) * a;
    l += __ldcg(pp + s * (D + 2) + D + 1) * a;
  }
  p.attn_out[(size_t)bh * D + d] = __float2bfloat16(l > 0.f ? acc / l : 0.f);
}

QK_DEVICE float sigmoidf_(float x) { return 1.f / (1.f + __expf(-x)); }

// ------------------------------------------------------------------ the kernel
template <int NB>
__global__ void __launch_bounds__(THREADS, 1) decode_megakernel(MegaParams p) {
  extern __shared__ __align__(128) uint8_t smem[];
  // Layout: [vec area: max(NB*I bf16, AttnSmem)] [spart] [sg] [red]
  constexpr size_t VEC_BYTES = (size_t)NB * I * 2 > sizeof(AttnSmem) ? (size_t)NB * I * 2 : sizeof(AttnSmem);
  bf16* sx = reinterpret_cast<bf16*>(smem);
  AttnSmem& sa = *reinterpret_cast<AttnSmem*>(smem);
  float* spart = reinterpret_cast<float*>(smem + VEC_BYTES);
  float* sg = spart + RPP * MAX_SEGS * NB;
  float* red = sg + RPP * NB;

  unsigned target = 0;
  bf16* x = p.x;
  int tick = 0;
  auto stamp = [&]() {
    if (p.timing && blockIdx.x == 0 && threadIdx.x == 0) p.timing[tick] = gtimer();
    ++tick;
  };
  stamp();

  if (p.tokens) {  // embedding lookup
    for (int i = blockIdx.x * THREADS + threadIdx.x; i < NB * H / 8; i += gridDim.x * THREADS) {
      int b = i / (H / 8), c = i % (H / 8);
      reinterpret_cast<uint4*>(x + (size_t)b * H)[c] =
          __ldg(reinterpret_cast<const uint4*>(p.embed + (size_t)p.tokens[b] * H) + c);
    }
    grid_sync(p.bar, target);
    stamp();
  }

  for (int layer = 0; layer < p.n_layers; ++layer) {
    const LayerW L = p.layers[layer];
    int r0, r1;

    // S1: input norm + QKV projection
    load_norm<NB>(x, L.in_ln, sx, p.eps, red);
    rows_of(QKV, r0, r1);
    block_gemv<NB>(L.qkv, H, r0, r1, sx, spart,
                   [&](int row, int b, float v) { p.qkv[(size_t)b * QKV + row] = __float2bfloat16(v); });
    grid_sync(p.bar, target);
    stamp();

    // S2: attention partials (+ q/k norm, rope, cache append)
    // HBM is mostly idle in S2/S3: stream this CTA's whole O-proj slice into L2 meanwhile.
    if (p.prefetch & 1) prefetch_slice(L.o, H, H, H);
    for (int item = blockIdx.x; item < NB * HKV * p.nsplit; item += gridDim.x) attn_item<NB>(p, L, layer, item, sa);
    grid_sync(p.bar, target);
    stamp();

    // S3: combine
    for (int i = blockIdx.x * THREADS + threadIdx.x; i < NB * HQ * D; i += gridDim.x * THREADS)
      combine_item(p, i / D, i % D);
    grid_sync(p.bar, target);
    stamp();

    // S4: O projection + residual
    load_vec<NB>(p.attn_out, H, sx);
    rows_of(H, r0, r1);
    block_gemv<NB>(L.o, H, r0, r1, sx, spart, [&](int row, int b, float v) {
      bf16* xp = x + (size_t)b * H + row;
      *xp = __float2bfloat16(v + __bfloat162float(__ldcg(xp)));
    });
    if (p.prefetch & 2) {  // cover the barrier with the head of the next stage's weights
      prefetch_slice(L.gate_up, I, H, 8);
      prefetch_slice(L.gate_up + (size_t)I * H, I, H, 8);
    }
    grid_sync(p.bar, target);
    stamp();

    // S5: post norm + gate/up + SwiGLU
    load_norm<NB>(x, L.post_ln, sx, p.eps, red);
    rows_of(I, r0, r1);
    for (int base = r0; base < r1; base += RPP) {
      const int e = min(r1, base + RPP);
      block_gemv<NB>(L.gate_up, H, base, e, sx, spart,
                     [&](int row, int b, float v) { sg[(row - base) * NB + b] = bf16_round(v); });
      block_gemv<NB>(L.gate_up + (size_t)I * H, H, base, e, sx, spart, [&](int row, int b, float v) {
        float g = sg[(row - base) * NB + b];
        float u = bf16_round(v);
        p.h[(size_t)b * I + row] = __float2bfloat16(bf16_round(g * sigmoidf_(g)) * u);
      });
    }
    if (p.prefetch & 2) prefetch_slice(L.down, H, I, 4);
    grid_sync(p.bar, target);
    stamp();

    // S6: down projection + residual
    load_vec<NB>(p.h, I, sx);
    rows_of(H, r0, r1);
    block_gemv<NB>(L.down, I, r0, r1, sx, spart, [&](int row, int b, float v) {
      bf16* xp = x + (size_t)b * H + row;
      *xp = __float2bfloat16(v + __bfloat162float(__ldcg(xp)));
    });
    if (p.prefetch & 2) {
      if (layer + 1 < p.n_layers) prefetch_slice(p.layers[layer + 1].qkv, QKV, H, 16);
      else if (p.lm_head) prefetch_slice(p.lm_head, p.vocab, H, 16);
    }
    grid_sync(p.bar, target);
    stamp();
  }

  if (p.lm_head) {  // final norm + LM head
    load_norm<NB>(x, p.final_norm, sx, p.eps, red);
    int r0, r1;
    rows_of(p.vocab, r0, r1);
    block_gemv<NB>(p.lm_head, H, r0, r1, sx, spart,
                   [&](int row, int b, float v) { p.logits[(size_t)b * p.vocab + row] = v; });
    __syncthreads();
    stamp();
  }
}

template <int NB>
constexpr size_t smem_bytes() {
  constexpr size_t vec = (size_t)NB * I * 2 > sizeof(AttnSmem) ? (size_t)NB * I * 2 : sizeof(AttnSmem);
  return vec + (RPP * MAX_SEGS * NB + RPP * NB + 32) * 4;
}

template <int NB>
cudaError_t launch_nb(const MegaParams& p, int grid, cudaStream_t st) {
  auto k = decode_megakernel<NB>;
  constexpr size_t smem = smem_bytes<NB>();
  cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  MegaParams pp = p;
  void* args[] = {&pp};
  return cudaLaunchCooperativeKernel((void*)k, dim3(grid), dim3(THREADS), args, smem, st);
}

cudaError_t launch(const MegaParams& p, int grid, cudaStream_t st) {
  switch (p.B) {
    case 1: return launch_nb<1>(p, grid, st);
    case 2: return launch_nb<2>(p, grid, st);
    case 3: return launch_nb<3>(p, grid, st);
    case 4: return launch_nb<4>(p, grid, st);
    default: return cudaErrorInvalidValue;
  }
}

int max_grid(int B, int num_sms) {
  int per_sm = 0;
  switch (B) {
#define CASE(n)                                                                                               \
  case n:                                                                                                     \
    cudaFuncSetAttribute(decode_megakernel<n>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes<n>()); \
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, decode_megakernel<n>, THREADS, smem_bytes<n>());   \
    break;
    CASE(1) CASE(2) CASE(3) CASE(4)
#undef CASE
  }
  return per_sm * num_sms;
}

}  // namespace mk
