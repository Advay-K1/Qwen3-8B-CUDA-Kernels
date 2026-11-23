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
constexpr int smem_bytes() { return STAGES * (A_TILE + BN * BK) * 2; }

struct Epi {
  float alpha;
  const bf16* R;  // optional residual, same shape as C
  int ldr;
  bool out_f32;
  bool accumulate;  // C += result (C must be fp32 or bf16 per out_f32)
};

// Load one BMxBK (or BNxBK) operand tile into swizzled smem.
// KMAJOR: smem tile is [rows=MN][cols=BK]; else [rows=BK][cols=MN].
template <bool KMAJOR, int MN>
QK_DEVICE void load_tile(bf16* s, const bf16* g, int ld, int mn0, int k0, int MNsz, int Ksz, int tid) {
  if constexpr (KMAJOR) {
    constexpr int CPR = BK / 8;  // chunks per row
    constexpr int TOTAL = MN * CPR;
#pragma unroll
    for (int i = 0; i < TOTAL / THREADS; ++i) {
      int id = tid + i * THREADS;
      int r = id / CPR, c = id % CPR;
      int gm = mn0 + r, gk = k0 + c * 8;
      bool p = (gm < MNsz) && (gk < Ksz);
      const bf16* src = p ? g + (size_t)gm * ld + gk : g;
      cp_async16(smem_u32(s + swz<BK>(r, c * 8)), src, p);
    }
  } else {
    constexpr int CPR = MN / 8;
    constexpr int TOTAL = BK * CPR;
#pragma unroll
    for (int i = 0; i < TOTAL / THREADS; ++i) {
      int id = tid + i * THREADS;
      int r = id / CPR, c = id % CPR;
      int gk = k0 + r, gm = mn0 + c * 8;
      bool p = (gk < Ksz) && (gm < MNsz);
      const bf16* src = p ? g + (size_t)gk * ld + gm : g;
      cp_async16(smem_u32(s + swz<MN>(r, c * 8)), src, p);
    }
  }
}

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

  auto load_stage = [&](int stage, int kt) {
    load_tile<A_KMAJOR, BM>(sA + stage * A_TILE, A, lda, m0, kt * BK, M, K, tid);
    load_tile<B_KMAJOR, BN>(sB + stage * B_TILE, B, ldb, n0, kt * BK, N, K, tid);
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

  for (int kt = 0; kt < KT; ++kt) {
    cp_async_wait<STAGES - 2>();
    __syncthreads();
    {
      int nk = kt + STAGES - 1;
      if (nk < KT) load_stage(nk % STAGES, nk);
      cp_async_commit();
    }
    const bf16* tA = sA + (kt % STAGES) * A_TILE;
    const bf16* tB = sB + (kt % STAGES) * B_TILE;

#pragma unroll
    for (int kk = 0; kk < BK / 16; ++kk) {
      uint32_t af[WM / 16][4];
      uint32_t bfr[WN / 8][2];
#pragma unroll
      for (int i = 0; i < WM / 16; ++i) {
        if constexpr (A_KMAJOR) {
          int r = wm0 + i * 16 + (lane & 15);
          int c = kk * 16 + (lane >> 4) * 8;
          ldmatrix_x4(af[i][0], af[i][1], af[i][2], af[i][3], smem_u32(tA + swz<BK>(r, c)));
        } else {
          int j = lane >> 3;
          int r = kk * 16 + (j >> 1) * 8 + (lane & 7);
          int c = wm0 + i * 16 + (j & 1) * 8;
          ldmatrix_x4_trans(af[i][0], af[i][1], af[i][2], af[i][3], smem_u32(tA + swz<BM>(r, c)));
        }
      }
#pragma unroll
      for (int p = 0; p < WN / 16; ++p) {
        int j = lane >> 3;
        if constexpr (B_KMAJOR) {
          int r = wn0 + p * 16 + (j >> 1) * 8 + (lane & 7);
          int c = kk * 16 + (j & 1) * 8;
          ldmatrix_x4(bfr[2 * p][0], bfr[2 * p][1], bfr[2 * p + 1][0], bfr[2 * p + 1][1],
                      smem_u32(tB + swz<BK>(r, c)));
        } else {
          int r = kk * 16 + (j & 1) * 8 + (lane & 7);
          int c = wn0 + p * 16 + (j >> 1) * 8;
          ldmatrix_x4_trans(bfr[2 * p][0], bfr[2 * p][1], bfr[2 * p + 1][0], bfr[2 * p + 1][1],
                            smem_u32(tB + swz<BN>(r, c)));
        }
      }
#pragma unroll
      for (int i = 0; i < WM / 16; ++i)
#pragma unroll
        for (int j = 0; j < WN / 8; ++j) mma_bf16_16816(acc[i][j], af[i], bfr[j]);
    }
  }
  cp_async_wait<0>();

  // Epilogue.
  const int g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int i = 0; i < WM / 16; ++i) {
#pragma unroll
    for (int j = 0; j < WN / 8; ++j) {
      int col = n0 + wn0 + j * 8 + 2 * t;
      if (col >= N) continue;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        int row = m0 + wm0 + i * 16 + g + h * 8;
        if (row >= M) continue;
        float v0 = acc[i][j][2 * h] * epi.alpha;
        float v1 = acc[i][j][2 * h + 1] * epi.alpha;
        if (epi.R) {
          float2 r = __bfloat1622float2(*reinterpret_cast<const bf162*>(epi.R + (size_t)row * epi.ldr + col));
          v0 += r.x;
          v1 += r.y;
        }
        size_t off = (size_t)row * ldc + col;
        if (epi.out_f32) {
          float2* p = reinterpret_cast<float2*>(reinterpret_cast<float*>(C) + off);
          if (epi.accumulate) {
            float2 o = *p;
            v0 += o.x;
            v1 += o.y;
          }
          *p = make_float2(v0, v1);
        } else {
          bf162* p = reinterpret_cast<bf162*>(reinterpret_cast<bf16*>(C) + off);
          if (epi.accumulate) {
            float2 o = __bfloat1622float2(*p);
            v0 += o.x;
            v1 += o.y;
          }
          *p = __floats2bfloat162_rn(v0, v1);
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
