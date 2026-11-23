// Qwen3-8B decode megakernel (v2).
//
// One persistent cooperative kernel runs a full decode step for NB (1..4) sequences:
//
//   [embed] -> for each layer:
//       S1  input RMSNorm (per CTA, redundant) + QKV GEMV                         | grid barrier
//       S2  attention partials: q/k RMSNorm + RoPE + KV-cache append fused in;    | grid barrier
//           K/V sub-chunks staged in smem with cp.async (double-buffered)
//       S3  combine split partials (per CTA, redundant) + O-proj GEMV + residual   | grid barrier
//       S4  post RMSNorm + gate/up GEMV + SwiGLU                                  | grid barrier
//       S5  down GEMV + residual                                                  | grid barrier
//   -> [final RMSNorm + LM-head GEMV -> fp32 logits]
//
// GEMV scheduling: warps pull whole output rows from a per-stage atomic counter
// (the next index is fetched one row ahead, hiding the atomic's latency). Each row
// is reduced by a single warp in a fixed order, so results are deterministic and
// the stage tail is under one row. Weights are streamed with L1::no_allocate and
// an L2 evict_first policy. bf16 rounding points match the layer-by-layer path.
#include <type_traits>

#include "common.cuh"
#include "megakernel.h"

namespace mk {
using namespace qk;

constexpr int H = 4096, I = 12288, HQ = 32, HKV = 8, D = 128, NREP = HQ / HKV, QKV = (HQ + 2 * HKV) * D;
constexpr int THREADS = MK_THREADS, WARPS = THREADS / 32;
constexpr int SUB = 64;  // keys per attention sub-chunk
constexpr int KV_STAGES = 3;
constexpr int CTRS_PER_LAYER = 8 + 4 * HKV;  // 4 GEMV row counters (+pad) + per-(b, kv head) split counters

// ------------------------------------------------------------------ memory helpers
QK_DEVICE uint64_t evict_first_policy() {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol));
  return pol;
}
QK_DEVICE uint4 ld_stream(const void* p, uint64_t pol) {  // weights: read once
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;\n"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p), "l"(pol));
  return r;
}
QK_DEVICE uint4 ld_cg(const void* p) { return __ldcg(reinterpret_cast<const uint4*>(p)); }
QK_DEVICE float2 ld_cg_bf2(const bf16* p) {
  unsigned u = __ldcg(reinterpret_cast<const unsigned*>(p));
  return __bfloat1622float2(*reinterpret_cast<bf162*>(&u));
}

QK_DEVICE unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
  return t;
}

QK_DEVICE void grid_sync(unsigned* bar, unsigned& target, unsigned long long* arrive = nullptr) {
  __syncthreads();
  if (threadIdx.x == 0) {
    if (arrive) arrive[blockIdx.x] = gtimer();
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

// ------------------------------------------------------------------ GEMV
// acc[b] = W[row, :] . sx[b, :]; K compile-time. All lanes end with the full sums.
template <int NB, int K>
QK_DEVICE void row_dot(const bf16* __restrict__ w, const bf16* sx, float* acc, uint64_t pol) {
  const int lane = threadIdx.x & 31;
  constexpr int NV = K / 256;  // 16B loads per lane for the row
  constexpr int CH = 16;       // loads in flight per chunk
#pragma unroll
  for (int b = 0; b < NB; ++b) acc[b] = 0.f;
#pragma unroll
  for (int c0 = 0; c0 < NV; c0 += CH) {
    uint4 wv[CH];
#pragma unroll
    for (int i = 0; i < CH; ++i) wv[i] = ld_stream(w + ((c0 + i) * 32 + lane) * 8, pol);
#pragma unroll
    for (int i = 0; i < CH; ++i) {
      float wf[8];
      unpack8(wv[i], wf);
#pragma unroll
      for (int b = 0; b < NB; ++b) {
        float xf[8];
        unpack8(*reinterpret_cast<const uint4*>(sx + (size_t)b * K + ((c0 + i) * 32 + lane) * 8), xf);
#pragma unroll
        for (int k = 0; k < 8; ++k) acc[b] = fmaf(wf[k], xf[k], acc[b]);
      }
    }
  }
#pragma unroll
  for (int b = 0; b < NB; ++b) acc[b] = warp_sum(acc[b]);
}

// Rows [0, N): warp i of the grid takes row i statically (no burst of same-address
// atomics at stage start), then pulls rows >= #warps from the stage's atomic counter
// (fetched one row ahead, so the atomic's latency hides behind the row's stream).
template <class Body>
QK_DEVICE void for_rows(unsigned* ctr, int N, Body body) {
  const int lane = threadIdx.x & 31;
  const unsigned nwarps = gridDim.x * WARPS;
  unsigned row = blockIdx.x * WARPS + (threadIdx.x >> 5);
  while (row < (unsigned)N) {
    unsigned nxt = 0;
    if (lane == 0) nxt = nwarps + atomicAdd(ctr, 1u);
    body((int)row);
    row = __shfl_sync(0xffffffff, nxt, 0);
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

// ------------------------------------------------------------------ attention
// Per-warp head RMSNorm + RoPE (lane owns d = 2l, 2l+1, 64+2l, 64+2l+1). Returns bf16-rounded values.
QK_DEVICE void head_norm_rope(const bf16* src, const bf16* __restrict__ w, const float* rc, const float* rs,
                              float eps, float* o) {
  const int lane = threadIdx.x & 31;
  float2 lo = ld_cg_bf2(src + 2 * lane), hi = ld_cg_bf2(src + 64 + 2 * lane);
  float x[4] = {lo.x, lo.y, hi.x, hi.y};
  float ss = warp_sum(x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]);
  float r = rsqrtf(ss / D + eps);
  float2 wl = __bfloat1622float2(reinterpret_cast<const bf162*>(w)[lane]);
  float2 wh = __bfloat1622float2(reinterpret_cast<const bf162*>(w + 64)[lane]);
  float n0 = bf16_round(x[0] * r) * wl.x, n1 = bf16_round(x[1] * r) * wl.y;
  float n2 = bf16_round(x[2] * r) * wh.x, n3 = bf16_round(x[3] * r) * wh.y;
  const float c0 = rc[2 * lane], c1 = rc[2 * lane + 1], s0 = rs[2 * lane], s1 = rs[2 * lane + 1];
  o[0] = bf16_round(n0 * c0 - n2 * s0);
  o[1] = bf16_round(n1 * c1 - n3 * s1);
  o[2] = bf16_round(n2 * c0 + n0 * s0);
  o[3] = bf16_round(n3 * c1 + n1 * s1);
}

// Attention smem. K/V rows padded to 136 elements (272 B): 16-byte loads from 8
// consecutive rows hit disjoint banks.
constexpr int KP = D + 8;
constexpr int QP = 64 + 4;
constexpr int PST = D + 4;  // partial record stride: o[D], m, l, pad (16B aligned)
struct AttnSmem {
  bf16 k[KV_STAGES][SUB][KP];
  bf16 v[KV_STAGES][SUB][KP];
  float q[2 * NREP][QP];  // row = head*2 + half (64 floats each), padded: conflict-free per-thread reads
  float s[NREP][SUB];   // scores, then probabilities
  float m[NREP], l[NREP], a[NREP];
  int last;
};

// cp.async `nrows` K and V rows (global rows base..) into stage `st`; skips global row `skip`.
QK_DEVICE void load_sub(AttnSmem& sm, int st, const bf16* kc, const bf16* vc, int base, int nrows, int skip) {
  for (int i = threadIdx.x; i < nrows * 16 * 2; i += THREADS) {
    const int which = i / (nrows * 16), j = i % (nrows * 16);
    const int r = j >> 4, c = j & 15;
    if (base + r == skip) continue;
    const bf16* src = (which ? vc : kc) + (size_t)(base + r) * D + c * 8;
    bf16* dst = which ? &sm.v[st][r][c * 8] : &sm.k[st][r][c * 8];
    cp_async16(smem_u32(dst), src, true);
  }
}

struct ItemGeom {
  int split, hk, b, k_beg, k_end, nsub, new_sub, new_row;
  bool owns_new, new_early;
  bf16 *kc, *vc;
};
QK_DEVICE ItemGeom item_geom(const MegaParams& p, int layer, int item) {
  ItemGeom g;
  g.split = item % p.nsplit;
  const int bh = item / p.nsplit;
  g.hk = bh % HKV;
  g.b = bh / HKV;
  const int Lk = p.pos + 1;
  g.k_beg = g.split * p.chunk;
  g.k_end = min(Lk, g.k_beg + p.chunk);
  g.nsub = (g.k_end - g.k_beg + SUB - 1) / SUB;
  const size_t cache_off = (((size_t)layer * p.B + g.b) * HKV + g.hk) * p.Smax * D;
  g.kc = p.k_cache + cache_off;
  g.vc = p.v_cache + cache_off;
  // The new token's K/V row is produced in S2. If it falls in one of the two sub-chunks
  // fetched up front, it is written into smem directly (and skipped by the async copy);
  // later sub-chunks are copied after the append is visible.
  g.owns_new = (g.k_end == Lk);
  g.new_sub = g.owns_new ? (p.pos - g.k_beg) / SUB : -1;
  g.new_early = g.owns_new && g.new_sub < KV_STAGES - 1;
  g.new_row = p.pos - g.k_beg - g.new_sub * SUB;
  return g;
}

// First two K/V sub-chunks of an item (two cp.async groups). Independent of this step's
// token, so it is issued a whole stage early (at the start of S1).
QK_DEVICE void attn_prefetch(const MegaParams& p, const ItemGeom& g, AttnSmem& sm) {
  const int skip = g.new_early ? p.pos : -1;
  load_sub(sm, 0, g.kc, g.vc, g.k_beg, min(SUB, g.k_end - g.k_beg), skip);
  cp_async_commit();
  if (g.nsub > 1) load_sub(sm, 1, g.kc, g.vc, g.k_beg + SUB, min(SUB, g.k_end - g.k_beg - SUB), skip);
  cp_async_commit();
}

// One split of one (sequence, KV head): all NREP=4 query heads, keys [k_beg, k_end).
// Per 64-key sub-chunk (3-stage cp.async pipeline):
//   scores : thread = (key, head, half of D), q half in registers, one shuffle joins halves
//   softmax: warp h (< 4) updates the running max / sum of head h
//   PV     : thread = (head, 4 dims, key quarter), 8-byte V loads, 2-shuffle reduce
template <int NB>
QK_DEVICE void attn_item(const MegaParams& p, const LayerW& L, int layer, int item, AttnSmem& sm, bool prefetched,
                         const float* rc, const float* rs) {
  static_assert(THREADS == NREP * D && THREADS == SUB * NREP * 2, "thread mappings");
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, tid = threadIdx.x;
  const ItemGeom g = item_geom(p, layer, item);
  const int pos = p.pos;
  const int split = g.split, hk = g.hk, b = g.b, k_beg = g.k_beg, k_end = g.k_end, nsub = g.nsub;
  bf16* kc = g.kc;
  bf16* vc = g.vc;
  const bf16* qkv = p.qkv + (size_t)b * QKV;
  const bool owns_new = g.owns_new, new_early = g.new_early;
  const int new_sub = g.new_sub, new_row = g.new_row;
  int dbg = 0;
  auto dstamp = [&]() {  // fine-grained timing: block 0, first layer, first item
    if (p.timing && layer == 0 && item == blockIdx.x && blockIdx.x == 0 && threadIdx.x == 0)
      p.timing[256 + dbg] = gtimer();
    ++dbg;
  };
  dstamp();
  if (!prefetched) attn_prefetch(p, g, sm);

  if (w < NREP) {
    float o[4];
    head_norm_rope(qkv + (hk * NREP + w) * D, L.q_norm, rc, rs, p.eps, o);
    const float sl2 = p.scale_log2;  // fold the softmax scale (log2 domain) into q
    sm.q[2 * w][2 * lane] = o[0] * sl2;
    sm.q[2 * w][2 * lane + 1] = o[1] * sl2;
    sm.q[2 * w + 1][2 * lane] = o[2] * sl2;
    sm.q[2 * w + 1][2 * lane + 1] = o[3] * sl2;
    if (lane == 0) {
      sm.m[w] = -INFINITY;
      sm.l[w] = 0.f;
    }
  } else if (w == NREP && owns_new) {  // append K (global + smem)
    float o[4];
    head_norm_rope(qkv + (HQ + hk) * D, L.k_norm, rc, rs, p.eps, o);
    bf162 lo = __floats2bfloat162_rn(o[0], o[1]), hi = __floats2bfloat162_rn(o[2], o[3]);
    bf16* dst = kc + (size_t)pos * D;
    reinterpret_cast<bf162*>(dst)[lane] = lo;
    reinterpret_cast<bf162*>(dst + 64)[lane] = hi;
    if (new_early) {
      reinterpret_cast<bf162*>(&sm.k[new_sub][new_row][0])[lane] = lo;
      reinterpret_cast<bf162*>(&sm.k[new_sub][new_row][64])[lane] = hi;
    }
    __threadfence();
  } else if (w == NREP + 1 && owns_new) {  // append V
    const unsigned* src = reinterpret_cast<const unsigned*>(qkv + (HQ + HKV + hk) * D);
    unsigned a0 = __ldcg(src + lane), a1 = __ldcg(src + 32 + lane);
    unsigned* dst = reinterpret_cast<unsigned*>(vc + (size_t)pos * D);
    dst[lane] = a0;
    dst[32 + lane] = a1;
    if (new_early) {
      reinterpret_cast<unsigned*>(&sm.v[new_sub][new_row][0])[lane] = a0;
      reinterpret_cast<unsigned*>(&sm.v[new_sub][new_row][0])[32 + lane] = a1;
    }
    __threadfence();
  }
  __syncthreads();
  dstamp();  // 1: q/k/v ready

  // thread roles
  const int s_key = tid >> 3, s_head = (tid >> 1) & 3, s_half = tid & 1;   // scores
  const int o_head = tid >> 7, o_d4 = (tid >> 2) & 31, o_kq = tid & 3;      // PV
  float qreg[64];
#pragma unroll
  for (int j = 0; j < 64; j += 4) {
    const float4 t4 = *reinterpret_cast<const float4*>(&sm.q[2 * s_head + s_half][j]);
    qreg[j] = t4.x; qreg[j + 1] = t4.y; qreg[j + 2] = t4.z; qreg[j + 3] = t4.w;
  }
  float oacc[4] = {0.f, 0.f, 0.f, 0.f};

  for (int sc = 0; sc < nsub; ++sc) {
    const int st = sc % KV_STAGES;
    cp_async_wait<KV_STAGES - 2>();
    __syncthreads();  // sub-chunk sc resident; stage (sc+2)%3 free (its PV finished last iteration)
    if (sc + 2 < nsub) {
      const int nb = k_beg + (sc + 2) * SUB;
      load_sub(sm, (sc + 2) % KV_STAGES, kc, vc, nb, min(SUB, k_end - nb), -1);
    }
    cp_async_commit();
    if (sc < 4) dstamp();  // 2+4sc: sub-chunk resident
    const int nrow = min(SUB, k_end - k_beg - sc * SUB);
    {  // scores (4 independent accumulators)
      float acc[4] = {0.f, 0.f, 0.f, 0.f};
      const bf16* kr = &sm.k[st][s_key][s_half * 64];
#pragma unroll
      for (int c = 0; c < 8; ++c) {
        float kf[8];
        unpack8(*reinterpret_cast<const uint4*>(kr + c * 8), kf);
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[i & 3] = fmaf(qreg[c * 8 + i], kf[i], acc[i & 3]);
      }
      float tot = (acc[0] + acc[1]) + (acc[2] + acc[3]);
      tot += __shfl_xor_sync(0xffffffff, tot, 1);
      if (s_half == 0) sm.s[s_head][s_key] = s_key < nrow ? tot : -INFINITY;
    }
    __syncthreads();
    if (sc < 4) dstamp();  // scores done
    if (w < NREP) {  // online softmax for head w over this sub-chunk (lane owns keys lane, lane+32)
      float s0 = sm.s[w][lane], s1 = sm.s[w][lane + 32];
      float mx = warp_max(fmaxf(s0, s1));
      const float m_old = sm.m[w];
      const float mn = fmaxf(m_old, mx);
      const float p0 = ex2(s0 - mn), p1 = ex2(s1 - mn);
      const float ps = warp_sum(p0 + p1);
      sm.s[w][lane] = p0;
      sm.s[w][lane + 32] = p1;
      __syncwarp();
      if (lane == 0) {
        const float a = ex2(m_old - mn);
        sm.a[w] = a;
        sm.l[w] = sm.l[w] * a + ps;
        sm.m[w] = mn;
      }
    }
    __syncthreads();
    if (sc < 4) dstamp();  // softmax done
    {  // PV: keys o_kq, o_kq+4, ... ; rows >= nrow are skipped (stale smem may hold NaN)
      const float* pr = sm.s[o_head];
      const bf16* vb = &sm.v[st][0][o_d4 * 4];
      float a4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int j = 0; j < SUB / 4; ++j) {
        const int r = o_kq + 4 * j;
        if (r < nrow) {
          const float pk = pr[r];
          uint2 u = *reinterpret_cast<const uint2*>(vb + r * KP);
          float2 v01 = __bfloat1622float2(*reinterpret_cast<bf162*>(&u.x));
          float2 v23 = __bfloat1622float2(*reinterpret_cast<bf162*>(&u.y));
          a4[0] = fmaf(pk, v01.x, a4[0]);
          a4[1] = fmaf(pk, v01.y, a4[1]);
          a4[2] = fmaf(pk, v23.x, a4[2]);
          a4[3] = fmaf(pk, v23.y, a4[3]);
        }
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        a4[i] += __shfl_xor_sync(0xffffffff, a4[i], 1);
        a4[i] += __shfl_xor_sync(0xffffffff, a4[i], 2);
      }
      const float a = sm.a[o_head];
#pragma unroll
      for (int i = 0; i < 4; ++i) oacc[i] = fmaf(oacc[i], a, a4[i]);
    }
    if (sc < 4) dstamp();  // PV issued (not synced)
  }
  cp_async_wait<0>();
  if (o_kq == 0) {
    float* pp = p.attn_part + (((size_t)b * HQ + hk * NREP + o_head) * p.nsplit + split) * PST;
    *reinterpret_cast<float4*>(pp + o_d4 * 4) = make_float4(oacc[0], oacc[1], oacc[2], oacc[3]);
    if (o_d4 == 0) {
      pp[D] = sm.m[o_head];
      pp[D + 1] = sm.l[o_head];
    }
    __threadfence();
  }
  __syncthreads();
  // Last split to finish for this (sequence, KV head) combines all splits of its 4 heads.
  if (tid == 0) {
    unsigned* cnt = p.bar + 1 + layer * CTRS_PER_LAYER + 8 + b * HKV + hk;
    sm.last = (atomicAdd(cnt, 1u) == (unsigned)p.nsplit - 1);
  }
  __syncthreads();
  if (sm.last) {
    __threadfence();
    const int h = tid >> 7, d = tid & (D - 1);
    const float* base = p.attn_part + ((size_t)b * HQ + hk * NREP + h) * p.nsplit * PST;
    float M = -INFINITY;
    for (int s2 = 0; s2 < p.nsplit; ++s2) M = fmaxf(M, __ldcg(base + s2 * PST + D));
    float num = 0.f, den = 0.f;
    for (int s2 = 0; s2 < p.nsplit; ++s2) {
      const float ms = __ldcg(base + s2 * PST + D);
      const float a = (ms == -INFINITY) ? 0.f : ex2(ms - M);
      num = fmaf(a, __ldcg(base + s2 * PST + d), num);
      den = fmaf(a, __ldcg(base + s2 * PST + D + 1), den);
    }
    p.attn_out[((size_t)b * HQ + hk * NREP + h) * D + d] = __float2bfloat16(den > 0.f ? num / den : 0.f);
  }
  __syncthreads();
  if (p.timing && layer == 0 && item == blockIdx.x && blockIdx.x == 0 && threadIdx.x == 0) p.timing[255] = gtimer();
}

// sx[b, h*D + d] = bf16(combined attention output).
template <int NB>
QK_DEVICE void load_attn_combined(const MegaParams& p, bf16* sx, float* swt) {
  const int ns = p.nsplit;
  // 1) all (b,h,split) m/l values in parallel
  for (int i = threadIdx.x; i < NB * HQ * ns; i += THREADS) {
    float2 ml = __ldcg(reinterpret_cast<const float2*>(p.attn_part + (size_t)i * PST + D));
    swt[2 * i] = ml.x;
    swt[2 * i + 1] = ml.y;
  }
  __syncthreads();
  // 2) per (b,h): normalized split weights, in place over the m slots
  for (int bh = threadIdx.x; bh < NB * HQ; bh += THREADS) {
    float* r = swt + 2 * bh * ns;
    float M = -INFINITY;
    for (int s = 0; s < ns; ++s) M = fmaxf(M, r[2 * s]);
    float l = 0.f;
    for (int s = 0; s < ns; ++s) {
      const float a = (r[2 * s] == -INFINITY) ? 0.f : ex2(r[2 * s] - M);
      r[2 * s] = a;
      l += r[2 * s + 1] * a;
    }
    const float inv = l > 0.f ? 1.f / l : 0.f;
    for (int s = 0; s < ns; ++s) r[2 * s] *= inv;
  }
  __syncthreads();
  // 3) weighted sum of the partial outputs, 4 elements per thread, splits unrolled
  for (int i = threadIdx.x; i < NB * HQ * D / 4; i += THREADS) {
    const int bh = i / (D / 4), d4 = (i % (D / 4)) * 4;
    const float* pp = p.attn_part + (size_t)bh * ns * PST + d4;
    const float* r = swt + 2 * bh * ns;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    int s = 0;
    for (; s + 4 <= ns; s += 4) {
      float4 x[4];
#pragma unroll
      for (int u = 0; u < 4; ++u) x[u] = __ldcg(reinterpret_cast<const float4*>(pp + (s + u) * PST));
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const float wgt = r[2 * (s + u)];
        acc.x = fmaf(wgt, x[u].x, acc.x);
        acc.y = fmaf(wgt, x[u].y, acc.y);
        acc.z = fmaf(wgt, x[u].z, acc.z);
        acc.w = fmaf(wgt, x[u].w, acc.w);
      }
    }
    for (; s < ns; ++s) {
      const float4 x = __ldcg(reinterpret_cast<const float4*>(pp + s * PST));
      const float wgt = r[2 * s];
      acc.x = fmaf(wgt, x.x, acc.x);
      acc.y = fmaf(wgt, x.y, acc.y);
      acc.z = fmaf(wgt, x.z, acc.z);
      acc.w = fmaf(wgt, x.w, acc.w);
    }
    bf162* dst = reinterpret_cast<bf162*>(sx + (size_t)bh * D + d4);
    dst[0] = __floats2bfloat162_rn(acc.x, acc.y);
    dst[1] = __floats2bfloat162_rn(acc.z, acc.w);
  }
  __syncthreads();
}

QK_DEVICE float sigmoidf_(float x) { return 1.f / (1.f + __expf(-x)); }

template <int NB>
QK_DEVICE float pick(const float* acc, int lane) {
  float v = acc[0];
#pragma unroll
  for (int b = 1; b < NB; ++b)
    if (lane == b) v = acc[b];
  return v;
}

// ------------------------------------------------------------------ the kernel
template <int NB>
struct SmemLayout {
  // [0, NB*H*2)          : sx during S1/S3/S4 (input vector of the GEMV)
  // [NB*H*2, ...)        : AttnSmem (the next layer's first K/V sub-chunks are prefetched
  //                        here during S1, so it must not overlap the S1 vector)
  //                        also reused for the combine weights in S3 and the tail of the
  //                        NB*I vector in S5
  static constexpr size_t ATT = (size_t)NB * H * 2;
  static constexpr size_t A_END = ATT + sizeof(AttnSmem);
  static constexpr size_t VEC = (size_t)NB * I * 2;
  static constexpr size_t MAIN = VEC > A_END ? VEC : A_END;
  static constexpr size_t TOTAL = MAIN + 2 * 64 * 4 + 32 * 4;  // + rope table + reduction scratch
  static_assert((size_t)NB * HQ * MK_MAX_SPLIT * 2 * 4 <= sizeof(AttnSmem), "combine weights fit");
};

template <int NB>
__global__ void __launch_bounds__(THREADS, 1) decode_megakernel(MegaParams p) {
  extern __shared__ __align__(128) uint8_t smem[];
  using SL = SmemLayout<NB>;
  bf16* sx = reinterpret_cast<bf16*>(smem);
  AttnSmem& sa = *reinterpret_cast<AttnSmem*>(smem + SL::ATT);
  float* swt = reinterpret_cast<float*>(smem + SL::ATT);
  float* rc = reinterpret_cast<float*>(smem + SL::MAIN);
  float* rs = rc + 64;
  float* red = rs + 64;
  const uint64_t pol = evict_first_policy();
  const int lane = threadIdx.x & 31;
  if (threadIdx.x < 64) {  // RoPE cos/sin for this step's position (same for every layer and head)
    const float inv = 1.0f / powf(p.theta, (float)(2 * threadIdx.x) / (float)D);
    sincosf((float)p.pos * inv, &rs[threadIdx.x], &rc[threadIdx.x]);
  }

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
  }

  grid_sync(p.bar, target, p.timing ? p.timing + 512 + (size_t)tick * gridDim.x : nullptr);
  stamp();

  for (int layer = 0; layer < p.n_layers; ++layer) {
    const LayerW L = p.layers[layer];
    unsigned* ctr = p.bar + 1 + layer * CTRS_PER_LAYER;

    // S1: input norm + QKV projection. Also start this CTA's attention K/V loads (they
    // do not depend on this step's token), so S2 finds them resident.
    if ((int)blockIdx.x < NB * HKV * p.nsplit) attn_prefetch(p, item_geom(p, layer, blockIdx.x), sa);
    load_norm<NB>(x, L.in_ln, sx, p.eps, red);
    for_rows(ctr + 0, QKV, [&](int row) {
      float acc[NB];
      row_dot<NB, H>(L.qkv + (size_t)row * H, sx, acc, pol);
      if (lane < NB) p.qkv[(size_t)lane * QKV + row] = __float2bfloat16(pick<NB>(acc, lane));
    });
    grid_sync(p.bar, target, p.timing ? p.timing + 512 + (size_t)tick * gridDim.x : nullptr);
    stamp();

    // S2: attention partials (+ q/k norm, rope, cache append)
    for (int item = blockIdx.x; item < NB * HKV * p.nsplit; item += gridDim.x)
      attn_item<NB>(p, L, layer, item, sa, item == (int)blockIdx.x, rc, rs);
    grid_sync(p.bar, target, p.timing ? p.timing + 512 + (size_t)tick * gridDim.x : nullptr);
    stamp();

    // S3: O projection + residual (attention output was combined by the last split of each head group)
    load_vec<NB>(p.attn_out, H, sx);
    for_rows(ctr + 1, H, [&](int row) {
      float acc[NB];
      row_dot<NB, H>(L.o + (size_t)row * H, sx, acc, pol);
      if (lane < NB) {
        bf16* xp = x + (size_t)lane * H + row;
        *xp = __float2bfloat16(pick<NB>(acc, lane) + __bfloat162float(__ldcg(xp)));
      }
    });
    grid_sync(p.bar, target, p.timing ? p.timing + 512 + (size_t)tick * gridDim.x : nullptr);
    stamp();

    // S4: post norm + gate/up + SwiGLU
    load_norm<NB>(x, L.post_ln, sx, p.eps, red);
    for_rows(ctr + 2, I, [&](int row) {
      float g[NB], u[NB];
      row_dot<NB, H>(L.gate_up + (size_t)row * H, sx, g, pol);
      row_dot<NB, H>(L.gate_up + (size_t)(I + row) * H, sx, u, pol);
      if (lane < NB) {
        const float gv = bf16_round(pick<NB>(g, lane)), uv = bf16_round(pick<NB>(u, lane));
        p.h[(size_t)lane * I + row] = __float2bfloat16(bf16_round(gv * sigmoidf_(gv)) * uv);
      }
    });
    grid_sync(p.bar, target, p.timing ? p.timing + 512 + (size_t)tick * gridDim.x : nullptr);
    stamp();

    // S5: down projection + residual
    load_vec<NB>(p.h, I, sx);
    for_rows(ctr + 3, H, [&](int row) {
      float acc[NB];
      row_dot<NB, I>(L.down + (size_t)row * I, sx, acc, pol);
      if (lane < NB) {
        bf16* xp = x + (size_t)lane * H + row;
        *xp = __float2bfloat16(pick<NB>(acc, lane) + __bfloat162float(__ldcg(xp)));
      }
    });
    grid_sync(p.bar, target, p.timing ? p.timing + 512 + (size_t)tick * gridDim.x : nullptr);
    stamp();
  }

  if (p.lm_head) {  // final norm + LM head
    load_norm<NB>(x, p.final_norm, sx, p.eps, red);
    for_rows(p.bar + 1 + p.n_layers * CTRS_PER_LAYER, p.vocab, [&](int row) {
      float acc[NB];
      row_dot<NB, H>(p.lm_head + (size_t)row * H, sx, acc, pol);
      if (lane < NB) p.logits[(size_t)lane * p.vocab + row] = pick<NB>(acc, lane);
    });
    __syncthreads();
    stamp();
  }
}

template <int NB>
cudaError_t launch_nb(const MegaParams& p, int grid, cudaStream_t st) {
  auto k = decode_megakernel<NB>;
  constexpr size_t smem = SmemLayout<NB>::TOTAL;
  static bool configured = false;  // attribute calls are not free; do it once per instantiation
  if (!configured) {
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    configured = true;
  }
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

size_t counter_words(int n_layers) { return 1 + (size_t)(n_layers + 1) * CTRS_PER_LAYER; }

int max_grid(int B, int num_sms) {
  int per_sm = 0;
  switch (B) {
#define CASE(n)                                                                              \
  case n:                                                                                    \
    cudaFuncSetAttribute(decode_megakernel<n>, cudaFuncAttributeMaxDynamicSharedMemorySize, \
                         (int)SmemLayout<n>::TOTAL);                                         \
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, decode_megakernel<n>, THREADS,    \
                                                  SmemLayout<n>::TOTAL);                     \
    break;
    CASE(1) CASE(2) CASE(3) CASE(4)
#undef CASE
  }
  return per_sm * num_sms;
}

}  // namespace mk
