// PyTorch bindings for the decode megakernel.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "megakernel.h"

using torch::Tensor;

namespace {
inline mk::bf16* bp(const Tensor& t) { return reinterpret_cast<mk::bf16*>(t.data_ptr()); }

// layers: list of 8-tuples (in_ln, qkv, q_norm, k_norm, o, post_ln, gate_up, down) -> device buffer.
Tensor pack_layers(const std::vector<std::vector<Tensor>>& layers, const Tensor& like) {
  std::vector<mk::LayerW> h(layers.size());
  for (size_t i = 0; i < layers.size(); ++i) {
    const auto& L = layers[i];
    TORCH_CHECK(L.size() == 8, "each layer needs 8 tensors");
    for (const auto& t : L) TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == torch::kBFloat16,
                                        "layer weights must be contiguous bf16 CUDA tensors");
    h[i] = {bp(L[0]), bp(L[1]), bp(L[2]), bp(L[3]), bp(L[4]), bp(L[5]), bp(L[6]), bp(L[7])};
  }
  auto buf = torch::empty({(long)(h.size() * sizeof(mk::LayerW))}, like.options().dtype(torch::kUInt8));
  cudaMemcpy(buf.data_ptr(), h.data(), h.size() * sizeof(mk::LayerW), cudaMemcpyHostToDevice);
  return buf;
}

int64_t max_grid(int64_t B) {
  return mk::max_grid(B, at::cuda::getCurrentDeviceProperties()->multiProcessorCount);
}

void decode_step(const Tensor& layers, int64_t n_layers, const c10::optional<Tensor>& tokens, const Tensor& embed,
                 const Tensor& final_norm, const c10::optional<Tensor>& lm_head, Tensor& k_cache, Tensor& v_cache,
                 int64_t pos, int64_t nsplit, int64_t chunk, double eps, double theta, Tensor& x, Tensor& qkv,
                 Tensor& attn_part, Tensor& attn_out, Tensor& h, Tensor& logits, Tensor& bar, int64_t grid) {
  const c10::cuda::CUDAGuard g(x.device());
  TORCH_CHECK(k_cache.dim() == 5 && k_cache.size(0) >= n_layers, "k_cache must be [L, B, Hkv, Smax, D]");
  TORCH_CHECK(pos < k_cache.size(3), "cache full");
  TORCH_CHECK((pos + 1 + chunk - 1) / chunk <= nsplit, "nsplit too small for chunk");
  mk::MegaParams p{};
  p.layers = reinterpret_cast<const mk::LayerW*>(layers.data_ptr());
  p.n_layers = n_layers;
  p.B = x.size(0);
  TORCH_CHECK(p.B >= 1 && p.B <= 4, "megakernel supports batch 1..4");
  p.tokens = tokens.has_value() ? reinterpret_cast<const long*>(tokens->data_ptr<int64_t>()) : nullptr;
  p.embed = bp(embed);
  p.final_norm = bp(final_norm);
  p.lm_head = lm_head.has_value() ? bp(*lm_head) : nullptr;
  p.vocab = lm_head.has_value() ? lm_head->size(0) : 0;
  p.k_cache = bp(k_cache);
  p.v_cache = bp(v_cache);
  p.Smax = k_cache.size(3);
  p.pos = pos;
  p.nsplit = nsplit;
  p.chunk = chunk;
  p.eps = eps;
  p.theta = theta;
  p.scale_log2 = (1.0 / std::sqrt(128.0)) * 1.4426950408889634;
  p.x = bp(x);
  p.qkv = bp(qkv);
  p.attn_part = attn_part.data_ptr<float>();
  p.attn_out = bp(attn_out);
  p.h = bp(h);
  p.logits = logits.data_ptr<float>();
  p.bar = reinterpret_cast<unsigned*>(bar.data_ptr<int32_t>());
  auto st = at::cuda::getCurrentCUDAStream();
  cudaMemsetAsync(p.bar, 0, sizeof(unsigned), st);
  mk::launch(p, grid, st);
  auto err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "megakernel launch failed: ", cudaGetErrorString(err));
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pack_layers", &pack_layers);
  m.def("max_grid", &max_grid);
  m.def("decode_step", &decode_step);
}
