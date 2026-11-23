// PyTorch bindings. All tensors must be CUDA + contiguous; activations/weights bf16.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "kernels.h"

using torch::Tensor;
using c10::optional;

namespace {

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIG(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_BF16(x) TORCH_CHECK((x).scalar_type() == torch::kBFloat16, #x " must be bfloat16")
#define CHECK_IN(x) \
  do {              \
    CHECK_CUDA(x);  \
    CHECK_CONTIG(x); \
  } while (0)
#define CHECK_BF16_IN(x) \
  do {                   \
    CHECK_IN(x);         \
    CHECK_BF16(x);       \
  } while (0)

inline qk::bf16* bp(const Tensor& t) { return reinterpret_cast<qk::bf16*>(t.data_ptr()); }
inline const qk::bf16* cbp(const optional<Tensor>& t) {
  return t.has_value() ? reinterpret_cast<const qk::bf16*>(t->data_ptr()) : nullptr;
}
inline cudaStream_t stream() { return at::cuda::getCurrentCUDAStream(); }
inline int num_sms() { return at::cuda::getCurrentDeviceProperties()->multiProcessorCount; }

// Generic GEMM: C[M,N] (+)= alpha * A.B (+ R); C dtype (bf16/fp32) selects the output type.
void gemm(const Tensor& A, const Tensor& B, Tensor& C, int64_t M, int64_t N, int64_t K, int64_t lda, int64_t ldb,
          int64_t ldc, bool a_kmajor, bool b_kmajor, double alpha, const optional<Tensor>& R, int64_t ldr,
          bool accumulate) {
  CHECK_BF16_IN(A);
  CHECK_BF16_IN(B);
  CHECK_IN(C);
  TORCH_CHECK((a_kmajor ? K : M) % 8 == 0 && lda % 8 == 0, "A contiguous dim must be a multiple of 8");
  TORCH_CHECK((b_kmajor ? K : N) % 8 == 0 && ldb % 8 == 0, "B contiguous dim must be a multiple of 8");
  TORCH_CHECK(N % 2 == 0 && ldc % 2 == 0, "N must be even");
  const c10::cuda::CUDAGuard g(A.device());
  qk::GemmArgs a;
  a.A = bp(A);
  a.B = bp(B);
  a.C = C.data_ptr();
  a.M = M; a.N = N; a.K = K;
  a.lda = lda; a.ldb = ldb; a.ldc = ldc;
  a.a_kmajor = a_kmajor; a.b_kmajor = b_kmajor;
  a.alpha = alpha;
  a.R = cbp(R);
  a.ldr = ldr;
  a.out_f32 = C.scalar_type() == torch::kFloat32;
  a.accumulate = accumulate;
  qk::gemm(a, stream());
}

std::vector<Tensor> rmsnorm_fwd(const Tensor& x, const Tensor& w, double eps, const optional<Tensor>& residual) {
  CHECK_BF16_IN(x);
  CHECK_BF16_IN(w);
  const int H = x.size(-1);
  TORCH_CHECK(H % 8 == 0 && H <= 8192, "hidden size must be a multiple of 8 and <= 8192");
  const int rows = x.numel() / H;
  const c10::cuda::CUDAGuard g(x.device());
  auto y = torch::empty_like(x);
  auto rstd = torch::empty({rows}, x.options().dtype(torch::kFloat32));
  Tensor s;
  if (residual.has_value()) {
    CHECK_BF16_IN((*residual));
    s = torch::empty_like(x);
  }
  qk::rmsnorm_fwd(bp(x), cbp(residual), bp(w), bp(y), residual.has_value() ? bp(s) : nullptr,
                  rstd.data_ptr<float>(), rows, H, eps, stream());
  if (residual.has_value()) return {y, rstd, s};
  return {y, rstd};
}

std::vector<Tensor> rmsnorm_bwd(const Tensor& dy, const Tensor& x, const Tensor& w, const Tensor& rstd,
                                const optional<Tensor>& dres) {
  CHECK_BF16_IN(dy);
  CHECK_BF16_IN(x);
  CHECK_BF16_IN(w);
  CHECK_IN(rstd);
  const int H = x.size(-1);
  const int rows = x.numel() / H;
  const c10::cuda::CUDAGuard g(x.device());
  auto dx = torch::empty_like(x);
  auto dw = torch::empty({H}, x.options().dtype(torch::kFloat32));
  auto part = torch::empty({qk::rmsnorm_bwd_parts(rows, num_sms()), H}, x.options().dtype(torch::kFloat32));
  if (dres.has_value()) CHECK_BF16_IN((*dres));
  qk::rmsnorm_bwd(bp(dy), bp(x), bp(w), rstd.data_ptr<float>(), cbp(dres), bp(dx), dw.data_ptr<float>(),
                  part.data_ptr<float>(), rows, H, num_sms(), stream());
  return {dx, dw};
}

// qkv [B*S, (Hq+2Hkv)*D] -> q [B,Hq,S,D], k/v [B,Hkv,kv_S,D] (written at kv_off), rstd_q [T,Hq], rstd_k [T,Hkv]
// If k_cache/v_cache are given they are written in place (kv_S = their dim 2).
std::vector<Tensor> qk_norm_rope_fwd(const Tensor& qkv, const Tensor& pos, const Tensor& wq, const Tensor& wk,
                                     int64_t B, int64_t S, int64_t Hq, int64_t Hkv, double theta, double eps,
                                     const optional<Tensor>& k_cache, const optional<Tensor>& v_cache,
                                     int64_t kv_off, bool save_rstd) {
  CHECK_BF16_IN(qkv);
  CHECK_IN(pos);
  TORCH_CHECK(pos.scalar_type() == torch::kInt32, "pos must be int32");
  const int D = 128;
  TORCH_CHECK(qkv.size(-1) == (Hq + 2 * Hkv) * D, "qkv width mismatch (head_dim must be 128)");
  const c10::cuda::CUDAGuard g(qkv.device());
  auto q = torch::empty({B, Hq, S, D}, qkv.options());
  Tensor k, v;
  int kv_S = S;
  if (k_cache.has_value()) {
    k = *k_cache;
    v = *v_cache;
    kv_S = k.size(2);
  } else {
    k = torch::empty({B, Hkv, S, D}, qkv.options());
    v = torch::empty({B, Hkv, S, D}, qkv.options());
  }
  auto f32 = qkv.options().dtype(torch::kFloat32);
  Tensor rq = save_rstd ? torch::empty({B * S, Hq}, f32) : Tensor();
  Tensor rk = save_rstd ? torch::empty({B * S, Hkv}, f32) : Tensor();
  qk::qk_norm_rope_fwd(bp(qkv), pos.data_ptr<int>(), bp(wq), bp(wk), bp(q), bp(k), bp(v),
                       save_rstd ? rq.data_ptr<float>() : nullptr, save_rstd ? rk.data_ptr<float>() : nullptr, B, S,
                       Hq, Hkv, kv_S, kv_off, theta, eps, stream());
  if (save_rstd) return {q, k, v, rq, rk};
  return {q, k, v};
}

std::vector<Tensor> qk_norm_rope_bwd(const Tensor& dq, const Tensor& dk, const Tensor& dv, const Tensor& qkv,
                                     const Tensor& pos, const Tensor& wq, const Tensor& wk, const Tensor& rq,
                                     const Tensor& rk, int64_t B, int64_t S, int64_t Hq, int64_t Hkv, double theta) {
  CHECK_IN(dq);
  TORCH_CHECK(dq.scalar_type() == torch::kFloat32, "dq must be fp32");
  CHECK_BF16_IN(dk);
  CHECK_BF16_IN(dv);
  CHECK_BF16_IN(qkv);
  const c10::cuda::CUDAGuard g(qkv.device());
  auto dqkv = torch::empty_like(qkv);
  auto f32 = qkv.options().dtype(torch::kFloat32);
  auto dwq = torch::zeros({128}, f32), dwk = torch::zeros({128}, f32);
  qk::qk_norm_rope_bwd(dq.data_ptr<float>(), bp(dk), bp(dv), bp(qkv), pos.data_ptr<int>(), bp(wq), bp(wk),
                       rq.data_ptr<float>(), rk.data_ptr<float>(), bp(dqkv), dwq.data_ptr<float>(),
                       dwk.data_ptr<float>(), B, S, Hq, Hkv, theta, num_sms(), stream());
  return {dqkv, dwq, dwk};
}

// q [B,Hq,S,D], k/v [B,Hkv,kv_S,D] (first S keys used) -> o [B,S,Hq,D], lse [B,Hq,S]
std::vector<Tensor> flash_fwd(const Tensor& q, const Tensor& k, const Tensor& v, double scale, bool need_lse) {
  CHECK_BF16_IN(q);
  CHECK_BF16_IN(k);
  CHECK_BF16_IN(v);
  const int B = q.size(0), Hq = q.size(1), S = q.size(2), Hkv = k.size(1), kv_S = k.size(2);
  TORCH_CHECK(q.size(3) == 128 && k.size(3) == 128, "head_dim must be 128");
  TORCH_CHECK(Hq % Hkv == 0, "Hq must be a multiple of Hkv");
  const c10::cuda::CUDAGuard g(q.device());
  auto o = torch::empty({B, S, Hq, 128}, q.options());
  Tensor lse = need_lse ? torch::empty({B, Hq, S}, q.options().dtype(torch::kFloat32)) : Tensor();
  qk::flash_attn_fwd(bp(q), bp(k), bp(v), bp(o), need_lse ? lse.data_ptr<float>() : nullptr, B, S, Hq, Hkv, kv_S,
                     scale, stream());
  if (need_lse) return {o, lse};
  return {o};
}

// Returns dq (fp32, [B,Hq,S,D]), dk, dv ([B,Hkv,S,D]).
std::vector<Tensor> flash_bwd(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& o, const Tensor& dout,
                              const Tensor& lse, double scale) {
  CHECK_BF16_IN(q);
  CHECK_BF16_IN(k);
  CHECK_BF16_IN(v);
  CHECK_BF16_IN(o);
  CHECK_BF16_IN(dout);
  CHECK_IN(lse);
  const int B = q.size(0), Hq = q.size(1), S = q.size(2), Hkv = k.size(1);
  TORCH_CHECK(k.size(2) == S, "flash_bwd needs k/v with exactly S keys");
  const c10::cuda::CUDAGuard g(q.device());
  auto f32 = q.options().dtype(torch::kFloat32);
  auto dq = torch::zeros({B, Hq, S, 128}, f32);
  auto dvec = torch::empty({B, Hq, S}, f32);
  auto dk = torch::empty_like(k), dv = torch::empty_like(v);
  qk::flash_attn_bwd(bp(q), bp(k), bp(v), bp(o), bp(dout), lse.data_ptr<float>(), dvec.data_ptr<float>(),
                     dq.data_ptr<float>(), bp(dk), bp(dv), B, S, Hq, Hkv, scale, stream());
  return {dq, dk, dv};
}

Tensor swiglu_fwd(const Tensor& gu) {
  CHECK_BF16_IN(gu);
  const int I = gu.size(-1) / 2;
  TORCH_CHECK(I % 8 == 0, "intermediate size must be a multiple of 8");
  const long T = gu.numel() / (2 * I);
  const c10::cuda::CUDAGuard g(gu.device());
  auto sizes = gu.sizes().vec();
  sizes.back() = I;
  auto h = torch::empty(sizes, gu.options());
  qk::swiglu_fwd(bp(gu), bp(h), T, I, stream());
  return h;
}

Tensor swiglu_bwd(const Tensor& dh, const Tensor& gu) {
  CHECK_BF16_IN(dh);
  CHECK_BF16_IN(gu);
  const int I = gu.size(-1) / 2;
  const long T = gu.numel() / (2 * I);
  const c10::cuda::CUDAGuard g(gu.device());
  auto dgu = torch::empty_like(gu);
  qk::swiglu_bwd(bp(dh), bp(gu), bp(dgu), T, I, stream());
  return dgu;
}

Tensor embedding_fwd(const Tensor& ids, const Tensor& table) {
  CHECK_IN(ids);
  CHECK_BF16_IN(table);
  TORCH_CHECK(ids.scalar_type() == torch::kInt32, "ids must be int32");
  const int H = table.size(1);
  const c10::cuda::CUDAGuard g(table.device());
  auto sizes = ids.sizes().vec();
  sizes.push_back(H);
  auto out = torch::empty(sizes, table.options());
  qk::embedding_fwd(ids.data_ptr<int>(), bp(table), bp(out), ids.numel(), H, stream());
  return out;
}

Tensor embedding_bwd(const Tensor& ids, const Tensor& dout, int64_t V) {
  CHECK_IN(ids);
  CHECK_BF16_IN(dout);
  const int H = dout.size(-1);
  const c10::cuda::CUDAGuard g(dout.device());
  auto dt = torch::zeros({V, H}, dout.options().dtype(torch::kFloat32));
  qk::embedding_bwd(ids.data_ptr<int>(), bp(dout), dt.data_ptr<float>(), ids.numel(), H, stream());
  return dt;
}

// In place: logits [rows, V] bf16 become dlogits * grad_scale. Returns (loss [rows], lse [rows]).
std::vector<Tensor> cross_entropy_(Tensor& logits, const Tensor& target, double grad_scale, int64_t ignore_index) {
  CHECK_BF16(logits);
  CHECK_CUDA(logits);
  CHECK_IN(target);
  TORCH_CHECK(target.scalar_type() == torch::kInt32, "target must be int32");
  TORCH_CHECK(logits.stride(-1) == 1, "logits must be row-contiguous");
  const int V = logits.size(-1);
  TORCH_CHECK(V % 8 == 0, "vocab must be a multiple of 8");
  const long rows = logits.numel() / V;
  const c10::cuda::CUDAGuard g(logits.device());
  auto f32 = logits.options().dtype(torch::kFloat32);
  auto loss = torch::empty({rows}, f32), lse = torch::empty({rows}, f32);
  qk::cross_entropy(bp(logits), target.data_ptr<int>(), loss.data_ptr<float>(), lse.data_ptr<float>(), rows, V,
                    logits.stride(-2), grad_scale, ignore_index, stream());
  return {loss, lse};
}

Tensor scale_cast(const Tensor& in, double s) {
  CHECK_IN(in);
  TORCH_CHECK(in.scalar_type() == torch::kFloat32, "fp32 input expected");
  const c10::cuda::CUDAGuard g(in.device());
  auto out = torch::empty(in.sizes(), in.options().dtype(torch::kBFloat16));
  qk::scale_cast(in.data_ptr<float>(), bp(out), in.numel(), s, stream());
  return out;
}

// logits fp32 [B, V]; per-row temperature (<=0: greedy), top_k (0: off), top_p (1: off).
Tensor sample(const Tensor& logits, const Tensor& temperature, const Tensor& top_k, const Tensor& top_p,
              int64_t seed, int64_t offset) {
  CHECK_CUDA(logits);
  TORCH_CHECK(logits.scalar_type() == torch::kFloat32 && logits.stride(-1) == 1, "fp32 row-contiguous logits");
  CHECK_IN(temperature);
  CHECK_IN(top_k);
  CHECK_IN(top_p);
  const long B = logits.size(0);
  const int V = logits.size(1);
  const c10::cuda::CUDAGuard g(logits.device());
  auto out = torch::empty({B}, logits.options().dtype(torch::kInt64));
  qk::sample_tokens(logits.data_ptr<float>(), B, V, logits.stride(0), temperature.data_ptr<float>(),
                    top_k.data_ptr<int>(), top_p.data_ptr<float>(), (unsigned long long)seed,
                    (unsigned long long)offset, reinterpret_cast<long*>(out.data_ptr<int64_t>()), stream());
  return out;
}

// y [NB, N] = x [NB, K] @ W[N, K]^T (+ R)
Tensor gemv(const Tensor& W, const Tensor& x, const optional<Tensor>& R, bool out_f32) {
  CHECK_BF16_IN(W);
  CHECK_BF16_IN(x);
  const int N = W.size(0), K = W.size(1);
  TORCH_CHECK(K % 8 == 0, "K must be a multiple of 8");
  const int NB = x.numel() / K;
  TORCH_CHECK(NB >= 1 && NB <= 8, "gemv supports 1..8 rows");
  const c10::cuda::CUDAGuard g(W.device());
  auto y = torch::empty({NB, N}, x.options().dtype(out_f32 ? torch::kFloat32 : torch::kBFloat16));
  if (R.has_value()) CHECK_BF16_IN((*R));
  qk::gemv(bp(W), bp(x), y.data_ptr(), cbp(R), NB, N, K, N, out_f32, num_sms(), stream());
  return y;
}

// q [B,Hq,D]; caches [B,Hkv,Smax,D]; lens int32 [B] -> out [B, Hq*D]
Tensor decode_attn(const Tensor& q, const Tensor& kc, const Tensor& vc, const Tensor& lens, int64_t max_len,
                   int64_t chunk, double scale) {
  CHECK_BF16_IN(q);
  CHECK_BF16_IN(kc);
  CHECK_BF16_IN(vc);
  CHECK_IN(lens);
  const int B = kc.size(0), Hkv = kc.size(1), Smax = kc.size(2);
  const int Hq = q.numel() / (B * 128);
  const c10::cuda::CUDAGuard g(q.device());
  const int nsplit = qk::decode_attn_num_splits(max_len, chunk);
  auto part = torch::empty({B, Hq, nsplit, 130}, q.options().dtype(torch::kFloat32));
  auto out = torch::empty({B, Hq * 128}, q.options());
  qk::decode_attn(bp(q), bp(kc), bp(vc), lens.data_ptr<int>(), part.data_ptr<float>(), bp(out), B, Hq, Hkv, Smax,
                  max_len, chunk, scale, stream());
  return out;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemm", &gemm);
  m.def("gemm_force_config", &qk::gemm_force_config);
  m.def("rmsnorm_fwd", &rmsnorm_fwd, py::arg("x"), py::arg("w"), py::arg("eps"), py::arg("residual") = py::none());
  m.def("rmsnorm_bwd", &rmsnorm_bwd, py::arg("dy"), py::arg("x"), py::arg("w"), py::arg("rstd"),
        py::arg("dres") = py::none());
  m.def("qk_norm_rope_fwd", &qk_norm_rope_fwd, py::arg("qkv"), py::arg("pos"), py::arg("wq"), py::arg("wk"),
        py::arg("B"), py::arg("S"), py::arg("Hq"), py::arg("Hkv"), py::arg("theta"), py::arg("eps"),
        py::arg("k_cache") = py::none(), py::arg("v_cache") = py::none(), py::arg("kv_off") = 0,
        py::arg("save_rstd") = true);
  m.def("qk_norm_rope_bwd", &qk_norm_rope_bwd);
  m.def("flash_fwd", &flash_fwd, py::arg("q"), py::arg("k"), py::arg("v"), py::arg("scale"),
        py::arg("need_lse") = true);
  m.def("flash_bwd", &flash_bwd);
  m.def("flash_bwd_set_warps", &qk::flash_bwd_set_warps);
  m.def("flash_fwd_set_warps", &qk::flash_fwd_set_warps);
  m.def("flash_bwd_set_debug", &qk::flash_bwd_set_debug);
  m.def("swiglu_fwd", &swiglu_fwd);
  m.def("swiglu_bwd", &swiglu_bwd);
  m.def("embedding_fwd", &embedding_fwd);
  m.def("embedding_bwd", &embedding_bwd);
  m.def("cross_entropy_", &cross_entropy_);
  m.def("scale_cast", &scale_cast);
  m.def("sample", &sample);
  m.def("gemv", &gemv, py::arg("W"), py::arg("x"), py::arg("R") = py::none(), py::arg("out_f32") = false);
  m.def("decode_attn", &decode_attn);
}
