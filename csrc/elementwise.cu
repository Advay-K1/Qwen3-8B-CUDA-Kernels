// SwiGLU, embedding, and misc elementwise kernels.
#include "common.cuh"
#include "kernels.h"

namespace qk {
namespace {

QK_DEVICE float sigmoidf_(float x) { return 1.f / (1.f + __expf(-x)); }

// gu [T, 2I] = [gate | up]  ->  h [T, I] = silu(gate) * up
// HF: act_fn(gate_proj(x)) * up_proj(x), each op rounded to bf16.
__global__ void swiglu_fwd_kernel(const bf16* __restrict__ gu, bf16* __restrict__ h, long T, int I) {
  const int nv = I / 8;
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= T * nv) return;
  long t = idx / nv;
  int c = idx % nv;
  float g[8], u[8], o[8];
  unpack8(reinterpret_cast<const uint4*>(gu + t * 2 * I)[c], g);
  unpack8(reinterpret_cast<const uint4*>(gu + t * 2 * I + I)[c], u);
#pragma unroll
  for (int i = 0; i < 8; ++i) o[i] = bf16_round(g[i] * sigmoidf_(g[i])) * u[i];
  reinterpret_cast<uint4*>(h + t * I)[c] = pack8(o);
}

// dgate = dh * up * silu'(gate),  dup = dh * silu(gate)
// silu'(x) = s * (1 + x * (1 - s)),  s = sigmoid(x)
__global__ void swiglu_bwd_kernel(const bf16* __restrict__ dh, const bf16* __restrict__ gu, bf16* __restrict__ dgu,
                                  long T, int I) {
  const int nv = I / 8;
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= T * nv) return;
  long t = idx / nv;
  int c = idx % nv;
  float g[8], u[8], d[8], dg[8], du[8];
  unpack8(reinterpret_cast<const uint4*>(gu + t * 2 * I)[c], g);
  unpack8(reinterpret_cast<const uint4*>(gu + t * 2 * I + I)[c], u);
  unpack8(reinterpret_cast<const uint4*>(dh + t * I)[c], d);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    float s = sigmoidf_(g[i]);
    float silu = g[i] * s;
    du[i] = d[i] * silu;
    dg[i] = d[i] * u[i] * s * (1.f + g[i] * (1.f - s));
  }
  reinterpret_cast<uint4*>(dgu + t * 2 * I)[c] = pack8(dg);
  reinterpret_cast<uint4*>(dgu + t * 2 * I + I)[c] = pack8(du);
}

__global__ void embedding_fwd_kernel(const int* __restrict__ ids, const bf16* __restrict__ table, bf16* __restrict__ out,
                                     int H) {
  const long t = blockIdx.x;
  const int id = ids[t];
  const uint4* src = reinterpret_cast<const uint4*>(table + (size_t)id * H);
  uint4* dst = reinterpret_cast<uint4*>(out + t * H);
  for (int c = threadIdx.x; c < H / 8; c += blockDim.x) dst[c] = src[c];
}

// dtable (fp32, zeroed) [V, H] += dout rows.
__global__ void embedding_bwd_kernel(const int* __restrict__ ids, const bf16* __restrict__ dout,
                                     float* __restrict__ dtable, int H) {
  const long t = blockIdx.x;
  const int id = ids[t];
  float* dst = dtable + (size_t)id * H;
  for (int c = threadIdx.x; c < H / 8; c += blockDim.x) {
    float f[8];
    unpack8(reinterpret_cast<const uint4*>(dout + t * H)[c], f);
#pragma unroll
    for (int i = 0; i < 8; ++i) atomicAdd(dst + c * 8 + i, f[i]);
  }
}

// Cross-entropy over the vocab, one block per row. Computes the loss and,
// in place, dlogits = (softmax(x) - onehot(target)) * grad_scale.
// Rows with target == ignore_index get loss 0 and zero grad.
constexpr int CE_THREADS = 1024;
__global__ void __launch_bounds__(CE_THREADS)
cross_entropy_kernel(bf16* __restrict__ logits, const int* __restrict__ target, float* __restrict__ loss,
                     float* __restrict__ lse_out, int V, long ld, float grad_scale, int ignore_index) {
  __shared__ float red[32];
  const long row = blockIdx.x;
  bf16* x = logits + row * ld;
  const int tgt = target[row];
  const int nv = V / 8;
  // Pass 1: online max/sum per thread.
  float m = -INFINITY, s = 0.f;
  for (int c = threadIdx.x; c < nv; c += CE_THREADS) {
    float f[8];
    unpack8(reinterpret_cast<const uint4*>(x)[c], f);
    float lm = f[0];
#pragma unroll
    for (int i = 1; i < 8; ++i) lm = fmaxf(lm, f[i]);
    float nm = fmaxf(m, lm);
    s *= __expf(m - nm);
#pragma unroll
    for (int i = 0; i < 8; ++i) s += __expf(f[i] - nm);
    m = nm;
  }
  float gm = block_max(m, red);
  s = (m == -INFINITY) ? 0.f : s * __expf(m - gm);
  s = block_sum(s, red);
  const float lse = gm + logf(s);
  const bool ignored = (tgt == ignore_index);
  if (threadIdx.x == 0) {
    loss[row] = ignored ? 0.f : lse - __bfloat162float(x[tgt]);
    if (lse_out) lse_out[row] = lse;
  }
  __syncthreads();  // x[tgt] read before overwrite
  // Pass 2: gradient in place.
  const float gs = ignored ? 0.f : grad_scale;
  for (int c = threadIdx.x; c < nv; c += CE_THREADS) {
    float f[8];
    unpack8(reinterpret_cast<const uint4*>(x)[c], f);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      float p = __expf(f[i] - lse);
      if (c * 8 + i == tgt) p -= 1.f;
      f[i] = p * gs;
    }
    reinterpret_cast<uint4*>(x)[c] = pack8(f);
  }
}

__global__ void scale_cast_kernel(const float* __restrict__ in, bf16* __restrict__ out, long n, float s) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __float2bfloat16(in[i] * s);
}

}  // namespace

void swiglu_fwd(const bf16* gu, bf16* h, long T, int I, cudaStream_t st) {
  long n = T * (I / 8);
  if (n == 0) return;
  swiglu_fwd_kernel<<<(n + 255) / 256, 256, 0, st>>>(gu, h, T, I);
}
void swiglu_bwd(const bf16* dh, const bf16* gu, bf16* dgu, long T, int I, cudaStream_t st) {
  long n = T * (I / 8);
  if (n == 0) return;
  swiglu_bwd_kernel<<<(n + 255) / 256, 256, 0, st>>>(dh, gu, dgu, T, I);
}
void embedding_fwd(const int* ids, const bf16* table, bf16* out, long T, int H, cudaStream_t st) {
  if (T == 0) return;
  embedding_fwd_kernel<<<T, 256, 0, st>>>(ids, table, out, H);
}
void embedding_bwd(const int* ids, const bf16* dout, float* dtable, long T, int H, cudaStream_t st) {
  if (T == 0) return;
  embedding_bwd_kernel<<<T, 256, 0, st>>>(ids, dout, dtable, H);
}
void cross_entropy(bf16* logits, const int* target, float* loss, float* lse, long rows, int V, long ld,
                   float grad_scale, int ignore_index, cudaStream_t st) {
  if (rows == 0) return;
  cross_entropy_kernel<<<rows, CE_THREADS, 0, st>>>(logits, target, loss, lse, V, ld, grad_scale, ignore_index);
}
void scale_cast(const float* in, bf16* out, long n, float s, cudaStream_t st) {
  if (n == 0) return;
  scale_cast_kernel<<<(n + 255) / 256, 256, 0, st>>>(in, out, n, s);
}

}  // namespace qk
