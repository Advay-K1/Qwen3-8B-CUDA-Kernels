// Shared helpers for the Qwen3-8B kernels.
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

#define QK_DEVICE __device__ __forceinline__

namespace qk {

using bf16 = __nv_bfloat16;
using bf162 = __nv_bfloat162;

// ---------------------------------------------------------------- reductions
QK_DEVICE float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
  return v;
}
QK_DEVICE float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
  return v;
}

// Block-wide sum. `smem` needs >= 32 floats. All threads get the result.
QK_DEVICE float block_sum(float v, float* smem) {
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  const int nw = (blockDim.x + 31) >> 5;
  v = warp_sum(v);
  __syncthreads();
  if (lane == 0) smem[wid] = v;
  __syncthreads();
  v = (lane < nw) ? smem[lane] : 0.f;
  v = warp_sum(v);
  return v;
}
QK_DEVICE float block_max(float v, float* smem) {
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  const int nw = (blockDim.x + 31) >> 5;
  v = warp_max(v);
  __syncthreads();
  if (lane == 0) smem[wid] = v;
  __syncthreads();
  v = (lane < nw) ? smem[lane] : -INFINITY;
  v = warp_max(v);
  return v;
}

// ---------------------------------------------------------------- vector io
struct __align__(16) bf16x8 { bf162 h[4]; };

QK_DEVICE void unpack8(const uint4& u, float* f) {
  const bf162* h = reinterpret_cast<const bf162*>(&u);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float2 t = __bfloat1622float2(h[i]);
    f[2 * i] = t.x;
    f[2 * i + 1] = t.y;
  }
}
QK_DEVICE uint4 pack8(const float* f) {
  uint4 u;
  bf162* h = reinterpret_cast<bf162*>(&u);
#pragma unroll
  for (int i = 0; i < 4; ++i) h[i] = __floats2bfloat162_rn(f[2 * i], f[2 * i + 1]);
  return u;
}
QK_DEVICE float bf16_round(float x) { return __bfloat162float(__float2bfloat16(x)); }

QK_DEVICE uint32_t pack_bf162(float a, float b) {
  bf162 h = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&h);
}

// ---------------------------------------------------------------- PTX wrappers
QK_DEVICE uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// 16-byte async copy global->shared; bytes beyond src_bytes are zero-filled.
QK_DEVICE void cp_async16(uint32_t dst, const void* src, bool pred) {
  const int n = pred ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(n));
}
QK_DEVICE void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
QK_DEVICE void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

QK_DEVICE void ldmatrix_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}
QK_DEVICE void ldmatrix_x4_trans(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}

// D = A(16x16, row) * B(16x8, col) + D, bf16 inputs, fp32 accumulate.
QK_DEVICE void mma_bf16_16816(float* d, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Swizzled shared-memory tile: ROWS x COLS bf16, COLS a multiple of 64.
// 16-byte chunk c of row r lives at chunk (c ^ (r & 7)): ldmatrix over 8
// consecutive rows at the same logical chunk is bank-conflict free.
template <int COLS>
QK_DEVICE int swz(int r, int c /*element col*/) {
  static_assert(COLS % 64 == 0, "swizzled tile needs COLS % 64 == 0");
  return r * COLS + ((((c >> 3) ^ (r & 7))) << 3) + (c & 7);
}

}  // namespace qk
