// bf16 tensor-core GEMM (mma.sync m16n8k16, ldmatrix, cp.async, multistage).
//
//   C[M,N] = alpha * sum_k A[m,k] * B[n,k]  (+ R[M,N])  (+ C if accumulate)
//
// Layouts (the contiguous dim must be a multiple of 8 elements):
//   A_KMAJOR : A[m,k] at A[m*lda + k]     else  A[k*lda + m]
//   B_KMAJOR : B[n,k] at B[n*ldb + k]     else  B[k*ldb + n]
// This covers every linear-layer GEMM:
//   fwd  y  = x W^T  : A=x  (K-major), B=W  (K-major)
//   bwd  dx = dy W   : A=dy (K-major), B=W  (N-major)
//   bwd  dW = dy^T x : A=dy (M-major), B=x  (N-major)
#include "common.cuh"
#include "kernels.h"

namespace qk {
namespace {

// CTA tile BM x BN x BK, 8 warps arranged 2 (M) x 4 (N), warp tile WM x WN (WN = BN / 4).
constexpr int BM = 128, BK = 64, THREADS = 256, WM = 64;
constexpr int A_TILE = BM * BK;
template <int BN, int STAGES>
constexpr int smem_bytes() {
  return STAGES * (A_TILE + BN * BK) * 2 > BM * (BN + 4) * 4 ? STAGES * (A_TILE + BN * BK) * 2 : BM * (BN + 4) * 4;
}

struct Epi {
  float alpha;
  const bf16* R;  // optional residual, same shape as C
  int ldr;
  bool out_f32;
  bool accumulate;  // C += result (C must be fp32 or bf16 per out_f32)
};

// cp.async tile loader with all per-thread addressing precomputed once.
// KMAJOR: smem tile is [rows=MN][cols=BK]; else [rows=BK][cols=MN].
// Every thread copies the same chunk column in every k-tile, so a single base
// pointer + a row-valid bitmask + a smem offset describe all of its copies.
template <bool KMAJOR, int MN>
struct TileLoader {
  static constexpr int COLS = KMAJOR ? BK : MN;
  static constexpr int CPR = COLS / 8;
  static constexpr int RPI = THREADS / CPR;  // rows per iteration
  static constexpr int ITERS = (KMAJOR ? MN : BK) / RPI;
  static_assert(ITERS <= 32, "row mask");
  const bf16* g;
  int ld, c8, r0;
  uint32_t s_off, mask;

  QK_DEVICE void init(const bf16* g0, int ld_, int mn0, int MNsz, int tid) {
    ld = ld_;
    r0 = tid / CPR;
    c8 = (tid % CPR) * 8;
    s_off = swz<COLS>(r0, c8) * 2;
    mask = 0;
    if constexpr (KMAJOR) {
      g = g0 + (size_t)(mn0 + r0) * ld + c8;
#pragma unroll
      for (int it = 0; it < ITERS; ++it) mask |= (uint32_t)(mn0 + r0 + it * RPI < MNsz) << it;
    } else {
      g = g0 + (size_t)r0 * ld + mn0 + c8;
      mask = (mn0 + c8 < MNsz) ? 0xffffffffu : 0u;
    }
  }
  QK_DEVICE void load(uint32_t stage_base, int k0, int Ksz) const {
#pragma unroll
    for (int it = 0; it < ITERS; ++it) {
      bool p;
      const bf16* src;
      if constexpr (KMAJOR) {
        p = ((mask >> it) & 1) && (k0 + c8 < Ksz);
        src = g + (size_t)it * RPI * ld + k0;
      } else {
        p = (mask & 1) && (k0 + r0 + it * RPI < Ksz);
        src = g + (size_t)(k0 + it * RPI) * ld;
      }
      cp_async16(stage_base + s_off + it * RPI * COLS * 2, src, p);  // src-size 0 when !p: nothing read
    }
  }
};

template <bool A_KMAJOR, bool B_KMAJOR, int BN, int STAGES>
__global__ void __launch_bounds__(THREADS)
gemm_kernel(const bf16* __restrict__ A, const bf16* __restrict__ B, void* __restrict__ C,
            int M, int N, int K, int lda, int ldb, int ldc, Epi epi) {
  constexpr int WN = BN / 4;
  constexpr int B_TILE = BN * BK;
  extern __shared__ __align__(128) uint8_t smem_raw[];
  bf16* sA = reinterpret_cast<bf16*>(smem_raw);
  bf16* sB = sA + STAGES * A_TILE;

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int warp_m = warp >> 2, warp_n = warp & 3;

  // Grouped rasterization for L2 reuse.
  const int grid_m = (M + BM - 1) / BM, grid_n = (N + BN - 1) / BN;
  constexpr int GROUP = 8;
  int bid = blockIdx.x;
  int group_size = GROUP * grid_n;
  int group = bid / group_size;
  int first_m = group * GROUP;
  int gsz = min(grid_m - first_m, GROUP);
  int bm = first_m + (bid % group_size) % gsz;
  int bn = (bid % group_size) / gsz;
  const int m0 = bm * BM, n0 = bn * BN;

  const int KT = (K + BK - 1) / BK;

  TileLoader<A_KMAJOR, BM> la;
  TileLoader<B_KMAJOR, BN> lb;
  la.init(A, lda, m0, M, tid);
  lb.init(B, ldb, n0, N, tid);
  const uint32_t sA0 = smem_u32(sA), sB0 = smem_u32(sB);
  auto load_stage = [&](int stage, int kt) {
    la.load(sA0 + stage * (A_TILE * 2), kt * BK, K);
    lb.load(sB0 + stage * (B_TILE * 2), kt * BK, K);
  };

#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < KT) load_stage(s, s);
    cp_async_commit();
  }

  float acc[WM / 16][WN / 8][4];
#pragma unroll
  for (int i = 0; i < WM / 16; ++i)
#pragma unroll
    for (int j = 0; j < WN / 8; ++j)
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[i][j][r] = 0.f;

  const int wm0 = warp_m * WM, wn0 = warp_n * WN;

  // Precomputed per-lane ldmatrix byte offsets. The XOR swizzle only touches the low 3
  // chunk bits, which are disjoint from the k-step / tile-index bits, so it is linear:
  // address = stage_base + lane_base + compile-time offset + x[step or tile].
  const int l7 = lane & 7, jq = lane >> 3;
  uint32_t a_lane, b_lane, a_x[4], b_x[4];
  if constexpr (A_KMAJOR) {  // tile [BM][BK]
    a_lane = (wm0 + (lane & 15)) * BK * 2;
    const int xa = (lane >> 4) ^ l7;
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) a_x[kk] = ((2 * kk) ^ xa) << 4;
  } else {  // tile [BK][BM]
    a_lane = ((jq >> 1) * 8 + l7) * BM * 2 + wm0 * 2;
    const int ya = (jq & 1) ^ l7;
#pragma unroll
    for (int i = 0; i < 4; ++i) a_x[i] = ((2 * i) ^ ya) << 4;
  }
  if constexpr (B_KMAJOR) {  // tile [BN][BK]
    b_lane = (wn0 + (jq >> 1) * 8 + l7) * BK * 2;
    const int xb = (jq & 1) ^ l7;
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) b_x[kk] = ((2 * kk) ^ xb) << 4;
  } else {  // tile [BK][BN]
    const int lo = (wn0 / 8) & 7, hi = (wn0 / 8) & ~7;
    b_lane = ((jq & 1) * 8 + l7) * BN * 2 + hi * 16;
    const int yb = lo ^ (jq >> 1) ^ l7;
#pragma unroll
    for (int p = 0; p < 4; ++p) b_x[p] = ((2 * p) ^ yb) << 4;
  }
  const uint32_t sA_u = smem_u32(sA) + a_lane, sB_u = smem_u32(sB) + b_lane;

  // Register fragments, double-buffered across k16 steps.
  uint32_t af[2][WM / 16][4];
  uint32_t bfr[2][WN / 8][2];
  auto load_frags = [&](int buf, int stage, int kk) {
    const uint32_t ta = sA_u + stage * (A_TILE * 2), tb = sB_u + stage * (B_TILE * 2);
#pragma unroll
    for (int i = 0; i < WM / 16; ++i) {
      if constexpr (A_KMAJOR)
        ldmatrix_x4(af[buf][i][0], af[buf][i][1], af[buf][i][2], af[buf][i][3], ta + i * 16 * BK * 2 + a_x[kk]);
      else
        ldmatrix_x4_trans(af[buf][i][0], af[buf][i][1], af[buf][i][2], af[buf][i][3],
                          ta + kk * 16 * BM * 2 + a_x[i]);
    }
#pragma unroll
    for (int p = 0; p < WN / 16; ++p) {
      if constexpr (B_KMAJOR)
        ldmatrix_x4(bfr[buf][2 * p][0], bfr[buf][2 * p][1], bfr[buf][2 * p + 1][0], bfr[buf][2 * p + 1][1],
                    tb + p * 16 * BK * 2 + b_x[kk]);
      else
        ldmatrix_x4_trans(bfr[buf][2 * p][0], bfr[buf][2 * p][1], bfr[buf][2 * p + 1][0], bfr[buf][2 * p + 1][1],
                          tb + kk * 16 * BN * 2 + b_x[p]);
    }
  };

  constexpr int KSTEPS = BK / 16;
  cp_async_wait<STAGES - 2>();
  __syncthreads();
  load_frags(0, 0, 0);

  int stage = 0;
  for (int kt = 0; kt < KT; ++kt) {
#pragma unroll
    for (int kk = 0; kk < KSTEPS; ++kk) {
      int fstage = stage;
      if (kk == KSTEPS - 1) {
        // Next k-tile must be resident before its first fragments are loaded.
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        fstage = (stage + 1 == STAGES) ? 0 : stage + 1;
      }
      load_frags((kk + 1) & 1, fstage, (kk + 1) % KSTEPS);  // prefetch next k16 step
      if (kk == 0) {
        // Refill the stage consumed in the previous iteration (all warps passed its last read).
        int nk = kt + STAGES - 1;
        if (nk < KT) load_stage(nk % STAGES, nk);
        cp_async_commit();
      }
#pragma unroll
      for (int i = 0; i < WM / 16; ++i)
#pragma unroll
        for (int j = 0; j < WN / 8; ++j) mma_bf16_16816(acc[i][j], af[kk & 1][i], bfr[kk & 1][j]);
    }
    stage = (stage + 1 == STAGES) ? 0 : stage + 1;
  }
  cp_async_wait<0>();
  __syncthreads();

  // Epilogue: stage the fp32 tile in smem, then 16-byte coalesced stores with fused residual / accumulate.
  constexpr int CST = BN + 4;  // padded row stride (floats)
  float* sC = reinterpret_cast<float*>(smem_raw);
  {
    const int g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int i = 0; i < WM / 16; ++i)
#pragma unroll
      for (int j = 0; j < WN / 8; ++j)
#pragma unroll
        for (int h = 0; h < 2; ++h)
          *reinterpret_cast<float2*>(sC + (wm0 + i * 16 + g + h * 8) * CST + wn0 + j * 8 + 2 * t) =
              make_float2(acc[i][j][2 * h] * epi.alpha, acc[i][j][2 * h + 1] * epi.alpha);
  }
  __syncthreads();
  constexpr int CPR = BN / 8;
#pragma unroll 4
  for (int id = tid; id < BM * CPR; id += THREADS) {
    const int r = id / CPR, c8 = (id % CPR) * 8;
    const int row = m0 + r, col = n0 + c8;
    if (row >= M || col >= N) continue;
    float v[8];
    *reinterpret_cast<float4*>(v) = *reinterpret_cast<const float4*>(sC + r * CST + c8);
    *reinterpret_cast<float4*>(v + 4) = *reinterpret_cast<const float4*>(sC + r * CST + c8 + 4);
    const size_t off = (size_t)row * ldc + col;
    if (col + 8 <= N) {
      if (epi.R) {
        float rr[8];
        unpack8(*reinterpret_cast<const uint4*>(epi.R + (size_t)row * epi.ldr + col), rr);
#pragma unroll
        for (int k = 0; k < 8; ++k) v[k] += rr[k];
      }
      if (epi.out_f32) {
        float4* pc = reinterpret_cast<float4*>(reinterpret_cast<float*>(C) + off);
        if (epi.accumulate) {
          float4 a0 = pc[0], a1 = pc[1];
          v[0] += a0.x; v[1] += a0.y; v[2] += a0.z; v[3] += a0.w;
          v[4] += a1.x; v[5] += a1.y; v[6] += a1.z; v[7] += a1.w;
        }
        pc[0] = make_float4(v[0], v[1], v[2], v[3]);
        pc[1] = make_float4(v[4], v[5], v[6], v[7]);
      } else {
        uint4* pc = reinterpret_cast<uint4*>(reinterpret_cast<bf16*>(C) + off);
        if (epi.accumulate) {
          float o[8];
          unpack8(*pc, o);
#pragma unroll
          for (int k = 0; k < 8; ++k) v[k] += o[k];
        }
        *pc = pack8(v);
      }
    } else {  // ragged N tail (N even): pairs
      for (int k = 0; k < 8 && col + k < N; k += 2) {
        float v0 = v[k], v1 = v[k + 1];
        if (epi.R) {
          float2 rr = __bfloat1622float2(*reinterpret_cast<const bf162*>(epi.R + (size_t)row * epi.ldr + col + k));
          v0 += rr.x; v1 += rr.y;
        }
        if (epi.out_f32) {
          float2* pc = reinterpret_cast<float2*>(reinterpret_cast<float*>(C) + off + k);
          if (epi.accumulate) { float2 o = *pc; v0 += o.x; v1 += o.y; }
          *pc = make_float2(v0, v1);
        } else {
          bf162* pc = reinterpret_cast<bf162*>(reinterpret_cast<bf16*>(C) + off + k);
          if (epi.accumulate) { float2 o = __bfloat1622float2(*pc); v0 += o.x; v1 += o.y; }
          *pc = __floats2bfloat162_rn(v0, v1);
        }
      }
    }
  }
}

template <bool AK, bool BK_, int BN, int STAGES>
void launch_cfg(const GemmArgs& a, cudaStream_t st) {
  auto k = gemm_kernel<AK, BK_, BN, STAGES>;
  constexpr int smem = smem_bytes<BN, STAGES>();
  static bool configured = false;  // one flag per instantiation
  if (!configured) {
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    configured = true;
  }
  int grid = ((a.M + BM - 1) / BM) * ((a.N + BN - 1) / BN);
  Epi e{a.alpha, a.R, a.ldr, a.out_f32, a.accumulate};
  k<<<grid, THREADS, smem, st>>>(a.A, a.B, a.C, a.M, a.N, a.K, a.lda, a.ldb, a.ldc, e);
}

int g_num_sms = 0;
int g_force_cfg = -1;  // -1: heuristic, 0: 128x128, 1: 128x256

template <bool AK, bool BK_>
void launch(const GemmArgs& a, cudaStream_t st) {
  if (!g_num_sms) {
    int dev;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&g_num_sms, cudaDevAttrMultiProcessorCount, dev);
  }
  // Big tiles when there are enough of them to fill the GPU at least twice.
  long big_tiles = (long)((a.M + BM - 1) / BM) * ((a.N + 255) / 256);
  bool big = g_force_cfg >= 0 ? g_force_cfg == 1 : big_tiles >= 2L * g_num_sms;
  if (big) launch_cfg<AK, BK_, 256, 3>(a, st);
  else launch_cfg<AK, BK_, 128, 4>(a, st);
}

}  // namespace

void gemm_force_config(int cfg) { g_force_cfg = cfg; }

void gemm(const GemmArgs& a, cudaStream_t st) {
  if (a.M <= 0 || a.N <= 0) return;
  if (a.a_kmajor && a.b_kmajor) launch<true, true>(a, st);
  else if (a.a_kmajor && !a.b_kmajor) launch<true, false>(a, st);
  else if (!a.a_kmajor && a.b_kmajor) launch<false, true>(a, st);
  else launch<false, false>(a, st);
}

}  // namespace qk
