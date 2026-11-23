// FlashAttention-2 style causal GQA attention, head_dim = 128, bf16 in / fp32 accumulate.
//
// Layouts:
//   q  [B, Hq,  S, D]      k, v [B, Hkv, S, D]        (written by qk_norm_rope)
//   o  [B, S, Hq, D]       (== [T, Hq*D], directly consumable by o_proj)
//   lse[B, Hq, S]  fp32, natural log of the softmax denominator of scale*QK^T
// Backward:
//   dO [B, S, Hq, D]; dq_acc fp32 [B, Hq, S, D] (must be zeroed); dk, dv [B, Hkv, S, D]
#include "common.cuh"
#include "kernels.h"

namespace qk {
namespace {

constexpr int D = 128;
constexpr int NW = 4;          // warps per CTA
constexpr int BR = 16 * NW;    // 64 query rows per CTA (fwd)
constexpr int BC = 64;         // keys per tile
constexpr int TILE = 64 * D;   // elements in a 64 x 128 tile
constexpr float LOG2E = 1.4426950408889634f;
constexpr float LN2 = 0.6931471805599453f;

// Async-load a 64 x 128 tile; rows >= nrows are zero-filled. row_stride in elements.
QK_DEVICE void load_tile64(bf16* s, const bf16* g, size_t row_stride, int nrows, int tid) {
#pragma unroll
  for (int i = 0; i < (64 * 16) / (NW * 32); ++i) {
    int id = tid + i * NW * 32;
    int r = id >> 4, c = id & 15;
    bool p = r < nrows;
    const bf16* src = p ? g + (size_t)r * row_stride + c * 8 : g;
    cp_async16(smem_u32(s + swz<D>(r, c * 8)), src, p);
  }
}

// Generic: ROWS x 128 tile loaded by NT threads; rows >= nrows zero-filled.
template <int ROWS, int NT>
QK_DEVICE void load_rows(bf16* s, const bf16* g, size_t row_stride, int nrows, int tid) {
#pragma unroll
  for (int i = 0; i < (ROWS * 16) / NT; ++i) {
    int id = tid + i * NT;
    int r = id >> 4, c = id & 15;
    bool p = r < nrows;
    const bf16* src = p ? g + (size_t)r * row_stride + c * 8 : g;
    cp_async16(smem_u32(s + swz<D>(r, c * 8)), src, p);
  }
}

// ============================================================== forward
// NWF warps x 16 query rows. smem: 2 stages of [K 64x128 | V 64x128]; the Q tile
// (<= 128 rows = one stage) is staged in stage 1 and moved to registers before
// stage 1 is first refilled.
template <int NWF>
__global__ void __launch_bounds__(NWF * 32)
flash_fwd_kernel(const bf16* __restrict__ Q, const bf16* __restrict__ K, const bf16* __restrict__ V,
                 bf16* __restrict__ O, float* __restrict__ LSE, int S, int Hq, int Hkv, int kv_S,
                 float scale_log2) {
  constexpr int BRF = 16 * NWF, NT = NWF * 32;
  extern __shared__ __align__(128) uint8_t smem_raw[];
  bf16* sbase = reinterpret_cast<bf16*>(smem_raw);
  auto sK = [&](int st) { return sbase + st * 2 * TILE; };
  auto sV = [&](int st) { return sbase + st * 2 * TILE + TILE; };
  bf16* sQ = sbase + 2 * TILE;  // aliases stage 1

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t = lane & 3, jj = lane >> 3;
  const int qb = gridDim.x - 1 - blockIdx.x;  // heaviest blocks first
  const int h = blockIdx.y, b = blockIdx.z;
  const int hk = h / (Hq / Hkv);
  const int q0 = qb * BRF;

  const bf16* qp = Q + ((size_t)(b * Hq + h) * S + q0) * D;
  const bf16* kp = K + (size_t)(b * Hkv + hk) * kv_S * D;
  const bf16* vp = V + (size_t)(b * Hkv + hk) * kv_S * D;

  const int nkv = min((S + BC - 1) / BC, (q0 + BRF + BC - 1) / BC);

  load_rows<BRF, NT>(sQ, qp, D, S - q0, tid);
  load_rows<64, NT>(sK(0), kp, D, S, tid);
  load_rows<64, NT>(sV(0), vp, D, S, tid);
  cp_async_commit();

  uint32_t qf[D / 16][4];
  float acc[D / 8][4];
#pragma unroll
  for (int n = 0; n < D / 8; ++n) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f;
  float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};

  for (int j = 0; j < nkv; ++j) {
    cp_async_wait<0>();
    __syncthreads();
    if (j == 0) {
#pragma unroll
      for (int kk = 0; kk < D / 16; ++kk) {
        int r = warp * 16 + (lane & 15), c = kk * 16 + (lane >> 4) * 8;
        ldmatrix_x4(qf[kk][0], qf[kk][1], qf[kk][2], qf[kk][3], smem_u32(sQ + swz<D>(r, c)));
      }
      __syncthreads();  // Q (stage 1) fully in registers before stage 1 is refilled
    }
    if (j + 1 < nkv) {
      int st = (j + 1) & 1;
      load_rows<64, NT>(sK(st), kp + (size_t)(j + 1) * BC * D, D, S - (j + 1) * BC, tid);
      load_rows<64, NT>(sV(st), vp + (size_t)(j + 1) * BC * D, D, S - (j + 1) * BC, tid);
    }
    cp_async_commit();
    const bf16* tK = sK(j & 1);
    const bf16* tV = sV(j & 1);

    // S = Q K^T  (16 x 64 per warp)
    float s[BC / 8][4];
#pragma unroll
    for (int n = 0; n < BC / 8; ++n) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f;
#pragma unroll
    for (int kk = 0; kk < D / 16; ++kk) {
#pragma unroll
      for (int p = 0; p < BC / 16; ++p) {
        uint32_t bb[4];
        int r = p * 16 + (jj >> 1) * 8 + (lane & 7), c = kk * 16 + (jj & 1) * 8;
        ldmatrix_x4(bb[0], bb[1], bb[2], bb[3], smem_u32(tK + swz<D>(r, c)));
        mma_bf16_16816(s[2 * p], qf[kk], bb);
        mma_bf16_16816(s[2 * p + 1], qf[kk], bb + 2);
      }
    }
    // Mask (diagonal block / sequence tail).
    const int k0 = j * BC;
    if (k0 + BC > q0 + warp * 16 || k0 + BC > S) {
#pragma unroll
      for (int n = 0; n < BC / 8; ++n)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          int kidx = k0 + n * 8 + 2 * t + (e & 1);
          int qidx = q0 + warp * 16 + g + (e >> 1) * 8;
          if (kidx > qidx || kidx >= S) s[n][e] = -INFINITY;
        }
    }
    // Online softmax.
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      float mx = -INFINITY;
#pragma unroll
      for (int n = 0; n < BC / 8; ++n) mx = fmaxf(mx, fmaxf(s[n][2 * hh], s[n][2 * hh + 1]));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 2));
      float m_new = fmaxf(m[hh], mx * scale_log2);
      float m_use = (m_new == -INFINITY) ? 0.f : m_new;
      float alpha = ex2(m[hh] - m_use);
      float rs = 0.f;
#pragma unroll
      for (int n = 0; n < BC / 8; ++n) {
        s[n][2 * hh] = ex2(fmaf(s[n][2 * hh], scale_log2, -m_use));
        s[n][2 * hh + 1] = ex2(fmaf(s[n][2 * hh + 1], scale_log2, -m_use));
        rs += s[n][2 * hh] + s[n][2 * hh + 1];
      }
      l[hh] = l[hh] * alpha + rs;
      m[hh] = m_new;
#pragma unroll
      for (int n = 0; n < D / 8; ++n) {
        acc[n][2 * hh] *= alpha;
        acc[n][2 * hh + 1] *= alpha;
      }
    }
    // O += P V
#pragma unroll
    for (int kk = 0; kk < BC / 16; ++kk) {
      uint32_t a[4] = {pack_bf162(s[2 * kk][0], s[2 * kk][1]), pack_bf162(s[2 * kk][2], s[2 * kk][3]),
                       pack_bf162(s[2 * kk + 1][0], s[2 * kk + 1][1]), pack_bf162(s[2 * kk + 1][2], s[2 * kk + 1][3])};
#pragma unroll
      for (int p = 0; p < D / 16; ++p) {
        uint32_t bb[4];
        int r = kk * 16 + (jj & 1) * 8 + (lane & 7), c = p * 16 + (jj >> 1) * 8;
        ldmatrix_x4_trans(bb[0], bb[1], bb[2], bb[3], smem_u32(tV + swz<D>(r, c)));
        mma_bf16_16816(acc[2 * p], a, bb);
        mma_bf16_16816(acc[2 * p + 1], a, bb + 2);
      }
    }
  }
  cp_async_wait<0>();

#pragma unroll
  for (int hh = 0; hh < 2; ++hh) {
    l[hh] += __shfl_xor_sync(0xffffffff, l[hh], 1);
    l[hh] += __shfl_xor_sync(0xffffffff, l[hh], 2);
    int qidx = q0 + warp * 16 + g + hh * 8;
    if (qidx >= S) continue;
    float inv = l[hh] > 0.f ? 1.f / l[hh] : 0.f;
    bf16* op = O + (((size_t)b * S + qidx) * Hq + h) * D;
#pragma unroll
    for (int n = 0; n < D / 8; ++n)
      *reinterpret_cast<bf162*>(op + n * 8 + 2 * t) =
          __floats2bfloat162_rn(acc[n][2 * hh] * inv, acc[n][2 * hh + 1] * inv);
    if (t == 0 && LSE) LSE[((size_t)b * Hq + h) * S + qidx] = (m[hh] + log2f(l[hh])) * LN2;
  }
}

// ============================================================== backward
// Dvec[b,h,s] = sum_d dO * O
__global__ void flash_bwd_pre_kernel(const bf16* __restrict__ dO, const bf16* __restrict__ O,
                                     float* __restrict__ Dvec, int B, int S, int Hq) {
  const int lane = threadIdx.x & 31;
  long row = (long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);  // index into [B, S, Hq]
  if (row >= (long)B * S * Hq) return;
  const bf16* a = dO + row * D;
  const bf16* c = O + row * D;
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    float2 x = __bfloat1622float2(reinterpret_cast<const bf162*>(a)[lane + 32 * i]);
    float2 y = __bfloat1622float2(reinterpret_cast<const bf162*>(c)[lane + 32 * i]);
    s += x.x * y.x + x.y * y.y;
  }
  s = warp_sum(s);
  if (lane == 0) {
    int h = row % Hq;
    long bs = row / Hq;
    int sidx = bs % S, b = bs / S;
    Dvec[((size_t)b * Hq + h) * S + sidx] = s;
  }
}

template <int NWB>
constexpr int bwd_smem() { return (2 * 16 * NWB * D + 4 * TILE + 16 * NWB * 64) * 2 + 4 * 64 * 4; }

// NWB warps; each warp owns 16 keys, so a CTA covers BCB = 16*NWB keys of one KV head.
template <int NWB>
__global__ void __launch_bounds__(NWB * 32)
flash_bwd_kernel(const bf16* __restrict__ Q, const bf16* __restrict__ K, const bf16* __restrict__ V,
                 const bf16* __restrict__ dO, const float* __restrict__ LSE, const float* __restrict__ Dvec,
                 float* __restrict__ dQacc, bf16* __restrict__ dK, bf16* __restrict__ dV, int S, int Hq, int Hkv,
                 float scale, int dbg) {
  constexpr int BCB = 16 * NWB, NT = NWB * 32;
  extern __shared__ __align__(128) uint8_t smem_raw[];
  bf16* sK = reinterpret_cast<bf16*>(smem_raw);
  bf16* sV = sK + BCB * D;
  bf16* sQ = sV + BCB * D;     // 2 stages
  bf16* sdO = sQ + 2 * TILE;   // 2 stages
  bf16* sdS = sdO + 2 * TILE;  // [BCB keys][64 q]
  float* sL = reinterpret_cast<float*>(sdS + BCB * 64);  // [2][64] lse * log2e
  float* sD = sL + 2 * 64;                               // [2][64]

  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  const int g = lane >> 2, t = lane & 3, jj = lane >> 3;
  const int kb = blockIdx.x, hk = blockIdx.y, b = blockIdx.z;
  const int n_rep = Hq / Hkv;
  const int kv0 = kb * BCB;
  const int nqb = (S + BR - 1) / BR;
  const int qb_start = kv0 / BR;  // causal: queries >= kv0
  const int nq_per = nqb - qb_start;
  const int iters = n_rep * nq_per;
  const float scale_log2 = scale * LOG2E;

  const bf16* kp = K + ((size_t)(b * Hkv + hk) * S + kv0) * D;
  const bf16* vp = V + ((size_t)(b * Hkv + hk) * S + kv0) * D;
  load_rows<BCB, NT>(sK, kp, D, S - kv0, tid);
  load_rows<BCB, NT>(sV, vp, D, S - kv0, tid);

  auto prefetch = [&](int it, int st) {
    int hq = hk * n_rep + it / nq_per;
    int q0 = (qb_start + it % nq_per) * BR;
    load_rows<64, NT>(sQ + st * TILE, Q + ((size_t)(b * Hq + hq) * S + q0) * D, D, S - q0, tid);
    load_rows<64, NT>(sdO + st * TILE, dO + (((size_t)b * S + q0) * Hq + hq) * D, (size_t)Hq * D, S - q0, tid);
    if (tid < 64) {
      int q = q0 + tid;
      sL[st * 64 + tid] = q < S ? LSE[((size_t)b * Hq + hq) * S + q] * LOG2E : INFINITY;
    } else if (tid < 128) {
      int q = q0 + tid - 64;
      sD[st * 64 + tid - 64] = q < S ? Dvec[((size_t)b * Hq + hq) * S + q] : 0.f;
    }
  };
  if (iters > 0) prefetch(0, 0);
  cp_async_commit();

  float dk[D / 8][4], dv[D / 8][4];
#pragma unroll
  for (int n = 0; n < D / 8; ++n)
#pragma unroll
    for (int e = 0; e < 4; ++e) dk[n][e] = dv[n][e] = 0.f;

  for (int it = 0; it < iters; ++it) {
    const int st = it & 1;
    cp_async_wait<0>();
    __syncthreads();
    if (it + 1 < iters) prefetch(it + 1, st ^ 1);
    cp_async_commit();

    const int hq = hk * n_rep + it / nq_per;
    const int qb = qb_start + it % nq_per;
    const int q0 = qb * BR;
    const bf16* tQ = sQ + st * TILE;
    const bf16* tdO = sdO + st * TILE;
    const float* tL = sL + st * 64;
    const float* tD = sD + st * 64;

    // 1. S^T = K Q^T  (16 keys x 64 queries per warp)
    float pt[8][4];
#pragma unroll
    for (int n = 0; n < 8; ++n) pt[n][0] = pt[n][1] = pt[n][2] = pt[n][3] = 0.f;
#pragma unroll
    for (int kk = 0; kk < D / 16; ++kk) {
      uint32_t af[4];
      ldmatrix_x4(af[0], af[1], af[2], af[3], smem_u32(sK + swz<D>(w * 16 + (lane & 15), kk * 16 + (lane >> 4) * 8)));
#pragma unroll
      for (int p = 0; p < 4; ++p) {
        uint32_t bb[4];
        int r = p * 16 + (jj >> 1) * 8 + (lane & 7), c = kk * 16 + (jj & 1) * 8;
        ldmatrix_x4(bb[0], bb[1], bb[2], bb[3], smem_u32(tQ + swz<D>(r, c)));
        mma_bf16_16816(pt[2 * p], af, bb);
        mma_bf16_16816(pt[2 * p + 1], af, bb + 2);
      }
    }
    // 2. P^T = exp(scale*S^T - lse)
    const bool need_mask = (q0 < kv0 + BCB) || (kv0 + BCB > S);
#pragma unroll
    for (int n = 0; n < 8; ++n)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        int ql = n * 8 + 2 * t + (e & 1);
        int key = kv0 + w * 16 + g + (e >> 1) * 8;
        float p = ex2(fmaf(pt[n][e], scale_log2, -tL[ql]));
        if (need_mask && (key > q0 + ql || key >= S)) p = 0.f;
        pt[n][e] = p;
      }
    // 3. dV += P^T dO
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
      uint32_t a[4] = {pack_bf162(pt[2 * kk][0], pt[2 * kk][1]), pack_bf162(pt[2 * kk][2], pt[2 * kk][3]),
                       pack_bf162(pt[2 * kk + 1][0], pt[2 * kk + 1][1]),
                       pack_bf162(pt[2 * kk + 1][2], pt[2 * kk + 1][3])};
#pragma unroll
      for (int p = 0; p < D / 16; ++p) {
        uint32_t bb[4];
        int r = kk * 16 + (jj & 1) * 8 + (lane & 7), c = p * 16 + (jj >> 1) * 8;
        ldmatrix_x4_trans(bb[0], bb[1], bb[2], bb[3], smem_u32(tdO + swz<D>(r, c)));
        mma_bf16_16816(dv[2 * p], a, bb);
        mma_bf16_16816(dv[2 * p + 1], a, bb + 2);
      }
    }
    // 4. dP^T = V dO^T
    float dpt[8][4];
#pragma unroll
    for (int n = 0; n < 8; ++n) dpt[n][0] = dpt[n][1] = dpt[n][2] = dpt[n][3] = 0.f;
#pragma unroll
    for (int kk = 0; kk < D / 16; ++kk) {
      uint32_t af[4];
      ldmatrix_x4(af[0], af[1], af[2], af[3], smem_u32(sV + swz<D>(w * 16 + (lane & 15), kk * 16 + (lane >> 4) * 8)));
#pragma unroll
      for (int p = 0; p < 4; ++p) {
        uint32_t bb[4];
        int r = p * 16 + (jj >> 1) * 8 + (lane & 7), c = kk * 16 + (jj & 1) * 8;
        ldmatrix_x4(bb[0], bb[1], bb[2], bb[3], smem_u32(tdO + swz<D>(r, c)));
        mma_bf16_16816(dpt[2 * p], af, bb);
        mma_bf16_16816(dpt[2 * p + 1], af, bb + 2);
      }
    }
    // 5. dS^T = P^T * (dP^T - Dvec)   (stored back into pt)
#pragma unroll
    for (int n = 0; n < 8; ++n)
#pragma unroll
      for (int e = 0; e < 4; ++e) pt[n][e] = pt[n][e] * (dpt[n][e] - tD[n * 8 + 2 * t + (e & 1)]);
    // 6. dK += dS^T Q ; 7. stash dS^T in smem
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
      uint32_t a[4] = {pack_bf162(pt[2 * kk][0], pt[2 * kk][1]), pack_bf162(pt[2 * kk][2], pt[2 * kk][3]),
                       pack_bf162(pt[2 * kk + 1][0], pt[2 * kk + 1][1]),
                       pack_bf162(pt[2 * kk + 1][2], pt[2 * kk + 1][3])};
      *reinterpret_cast<uint32_t*>(sdS + swz<64>(w * 16 + g, kk * 16 + 2 * t)) = a[0];
      *reinterpret_cast<uint32_t*>(sdS + swz<64>(w * 16 + g + 8, kk * 16 + 2 * t)) = a[1];
      *reinterpret_cast<uint32_t*>(sdS + swz<64>(w * 16 + g, kk * 16 + 8 + 2 * t)) = a[2];
      *reinterpret_cast<uint32_t*>(sdS + swz<64>(w * 16 + g + 8, kk * 16 + 8 + 2 * t)) = a[3];
#pragma unroll
      for (int p = 0; p < D / 16; ++p) {
        uint32_t bb[4];
        int r = kk * 16 + (jj & 1) * 8 + (lane & 7), c = p * 16 + (jj >> 1) * 8;
        ldmatrix_x4_trans(bb[0], bb[1], bb[2], bb[3], smem_u32(tQ + swz<D>(r, c)));
        mma_bf16_16816(dk[2 * p], a, bb);
        mma_bf16_16816(dk[2 * p + 1], a, bb + 2);
      }
    }
    __syncthreads();
    // 8. dQ += scale * dS K. Row group rg (16 queries) x one or both 64-wide halves of D per warp.
    const int rg = w & 3;
#pragma unroll
    for (int hi = 0; hi < 8 / NWB; ++hi) {
      const int half = (NWB == 8) ? (w >> 2) : hi;
      float dq[8][4];
#pragma unroll
      for (int n = 0; n < 8; ++n) dq[n][0] = dq[n][1] = dq[n][2] = dq[n][3] = 0.f;
#pragma unroll
      for (int kk = 0; kk < BCB / 16; ++kk) {
        uint32_t af[4];
        {
          int r = kk * 16 + (jj >> 1) * 8 + (lane & 7), c = rg * 16 + (jj & 1) * 8;
          ldmatrix_x4_trans(af[0], af[1], af[2], af[3], smem_u32(sdS + swz<64>(r, c)));
        }
#pragma unroll
        for (int p = 0; p < 4; ++p) {
          uint32_t bb[4];
          int r = kk * 16 + (jj & 1) * 8 + (lane & 7), c = half * 64 + p * 16 + (jj >> 1) * 8;
          ldmatrix_x4_trans(bb[0], bb[1], bb[2], bb[3], smem_u32(sK + swz<D>(r, c)));
          mma_bf16_16816(dq[2 * p], af, bb);
          mma_bf16_16816(dq[2 * p + 1], af, bb + 2);
        }
      }
      // Sector-friendly atomics: lanes g and g^4 swap their odd-column values so each RED
      // instruction covers 4 full rows x 8 contiguous floats (4 sectors) instead of
      // 8 rows x 4 strided floats (8 half-used sectors).
      if (!(dbg & 1)) {
        const bool lo4 = g < 4;
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const int rbase = q0 + rg * 16 + hh * 8;  // 8-row group
          const int rA = lo4 ? g : g - 4;           // instruction A: rows 0..3 of the group
          const int rB = lo4 ? g + 4 : g;           // instruction B: rows 4..7
          const int cA = lo4 ? 2 * t : 2 * t + 1;
          const int cB = lo4 ? 2 * t + 1 : 2 * t;
          float* rowA = dQacc + ((size_t)(b * Hq + hq) * S + rbase + rA) * D + half * 64 + cA;
          float* rowB = dQacc + ((size_t)(b * Hq + hq) * S + rbase + rB) * D + half * 64 + cB;
          const bool okA = rbase + rA < S, okB = rbase + rB < S;
#pragma unroll
          for (int n = 0; n < 8; ++n) {
            const float v0 = dq[n][2 * hh] * scale, v1 = dq[n][2 * hh + 1] * scale;
            const float other = __shfl_xor_sync(0xffffffff, v1, 16);
            if (okA) atomicAdd(rowA + n * 8, lo4 ? v0 : other);
            if (okB) atomicAdd(rowB + n * 8, lo4 ? other : v0);
          }
        }
      }
    }
  }
  cp_async_wait<0>();

  // Write dK (scaled) and dV.
#pragma unroll
  for (int hh = 0; hh < 2; ++hh) {
    int key = kv0 + w * 16 + g + hh * 8;
    if (key >= S) continue;
    size_t off = ((size_t)(b * Hkv + hk) * S + key) * D + 2 * t;
#pragma unroll
    for (int n = 0; n < D / 8; ++n) {
      *reinterpret_cast<bf162*>(dK + off + n * 8) = __floats2bfloat162_rn(dk[n][2 * hh] * scale, dk[n][2 * hh + 1] * scale);
      *reinterpret_cast<bf162*>(dV + off + n * 8) = __floats2bfloat162_rn(dv[n][2 * hh], dv[n][2 * hh + 1]);
    }
  }
}

}  // namespace

static int g_fwd_warps = 4;
void flash_fwd_set_warps(int nw) { g_fwd_warps = nw; }

void flash_attn_fwd(const bf16* q, const bf16* k, const bf16* v, bf16* o, float* lse, int B, int S, int Hq, int Hkv,
                    int kv_S, float scale, cudaStream_t st) {
  constexpr int smem = 4 * TILE * 2;
  if (g_fwd_warps == 8) {
    static bool cfg = false;
    if (!cfg) { cudaFuncSetAttribute(flash_fwd_kernel<8>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); cfg = true; }
    dim3 grid((S + 127) / 128, Hq, B);
    flash_fwd_kernel<8><<<grid, 256, smem, st>>>(q, k, v, o, lse, S, Hq, Hkv, kv_S, scale * LOG2E);
  } else {
    static bool cfg = false;
    if (!cfg) { cudaFuncSetAttribute(flash_fwd_kernel<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); cfg = true; }
    dim3 grid((S + 63) / 64, Hq, B);
    flash_fwd_kernel<4><<<grid, 128, smem, st>>>(q, k, v, o, lse, S, Hq, Hkv, kv_S, scale * LOG2E);
  }
}

static int g_bwd_warps = 8;
static int g_bwd_dbg = 0;
void flash_bwd_set_debug(int f) { g_bwd_dbg = f; }

void flash_attn_bwd(const bf16* q, const bf16* k, const bf16* v, const bf16* o, const bf16* dout, const float* lse,
                    float* dvec, float* dq_acc, bf16* dk, bf16* dv, int B, int S, int Hq, int Hkv, float scale,
                    cudaStream_t st) {
  long rows = (long)B * S * Hq;
  flash_bwd_pre_kernel<<<(rows + 7) / 8, 256, 0, st>>>(dout, o, dvec, B, S, Hq);
  if (g_bwd_warps == 4) {
    constexpr int smem = bwd_smem<4>();
    static bool cfg = false;
    if (!cfg) { cudaFuncSetAttribute(flash_bwd_kernel<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); cfg = true; }
    dim3 grid((S + 63) / 64, Hkv, B);
    flash_bwd_kernel<4><<<grid, 128, smem, st>>>(q, k, v, dout, lse, dvec, dq_acc, dk, dv, S, Hq, Hkv, scale, g_bwd_dbg);
  } else {
    constexpr int smem = bwd_smem<8>();
    static bool cfg = false;
    if (!cfg) { cudaFuncSetAttribute(flash_bwd_kernel<8>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); cfg = true; }
    dim3 grid((S + 127) / 128, Hkv, B);
    flash_bwd_kernel<8><<<grid, 256, smem, st>>>(q, k, v, dout, lse, dvec, dq_acc, dk, dv, S, Hq, Hkv, scale, g_bwd_dbg);
  }
}

void flash_bwd_set_warps(int nw) { g_bwd_warps = nw; }

}  // namespace qk
