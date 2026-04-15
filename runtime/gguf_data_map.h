/**
 * GGUFNamedDataMap — NamedDataMap backed by a GGUF file.
 *
 * Implements the ExecuTorch NamedDataMap interface so that backends can
 * resolve weight data directly from a .gguf file via get_data(key).
 * The GGUF tensor data is memory-mapped; no copy is made until the
 * backend requests it via get_data().
 */

#pragma once

#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include <executorch/runtime/core/named_data_map.h>
#include <ggml.h>
#include <gguf.h>

namespace executorch_ggml {

class GGUFNamedDataMap : public executorch::runtime::NamedDataMap {
 public:
  ~GGUFNamedDataMap() override {
    if (tensor_ctx_) ggml_free(tensor_ctx_);
    if (gguf_ctx_) gguf_free(gguf_ctx_);
  }

  /// Open a GGUF file. Returns Error::InvalidArgument on failure.
  static executorch::runtime::Result<std::unique_ptr<GGUFNamedDataMap>>
  load(const std::string& gguf_path) {
    struct ggml_context* tctx = nullptr;
    struct gguf_init_params gp = {/*no_alloc=*/false, /*ctx=*/&tctx};
    struct gguf_context* ctx = gguf_init_from_file(gguf_path.c_str(), gp);
    if (!ctx) {
      return executorch::runtime::Error::InvalidArgument;
    }

    auto map = std::make_unique<GGUFNamedDataMap>();
    map->gguf_ctx_ = ctx;
    map->tensor_ctx_ = tctx;
    map->path_ = gguf_path;

    // Build key→tensor index.
    int64_t n = gguf_get_n_tensors(ctx);
    for (int64_t i = 0; i < n; ++i) {
      map->keys_.push_back(gguf_get_tensor_name(ctx, i));
      map->key_to_idx_[map->keys_.back()] = static_cast<uint32_t>(i);
    }
    return map;
  }

  executorch::runtime::Result<const executorch::runtime::TensorLayout>
  get_tensor_layout(executorch::aten::string_view key) const override {
    (void)key;
    return executorch::runtime::Error::NotFound;
  }

  /// Get the ggml type of a GGUF tensor by name (for quantized weight loading).
  ggml_type get_tensor_type(const char* name) const {
    struct ggml_tensor* gt = ggml_get_tensor(tensor_ctx_, name);
    return gt ? gt->type : GGML_TYPE_F32;
  }

  executorch::runtime::Result<executorch::runtime::FreeableBuffer>
  get_data(executorch::aten::string_view key) const override {
    std::string k(key.data(), key.size());

    // Fused projection lookup: if key matches a fuseable pattern (attn_q →
    // Q+K+V, ffn_gate → gate+up), return the concatenated tensor.
    // This serves both fused PTEs (which need the full concat) and non-fused
    // PTEs (backend copies only the first N bytes it needs, ignoring the rest,
    // since the size check is buf.size() >= nbytes).
    {
      auto fused = try_fused_lookup(k);
      if (fused.ok()) return std::move(fused.get());
    }

    // Direct GGUF tensor lookup.
    auto it = key_to_idx_.find(k);
    if (it != key_to_idx_.end()) {
      struct ggml_tensor* gt = ggml_get_tensor(tensor_ctx_, k.c_str());
      if (gt && gt->data) {
        size_t nbytes = ggml_nbytes(gt);
        // Non-owning buffer (GGUF data is mmap'd).
        return executorch::runtime::FreeableBuffer(gt->data, nbytes, /*free_fn=*/nullptr);
      }
    }

    // Derived constant: inv_freq — compute from GGUF metadata.
    // inv_freq[i] = 1 / (freq_base ** (2i / dim)), shape [dim/2].
    if (k.find("inv_freq") != std::string::npos) {
      return compute_inv_freq();
    }

    return executorch::runtime::Error::NotFound;
  }

 private:
  executorch::runtime::Result<executorch::runtime::FreeableBuffer>
  compute_inv_freq() const {
    // Read rope parameters from GGUF metadata.
    float freq_base = 10000.0f;
    int64_t dim = 128;  // default head_dim

    int64_t arch_key = gguf_find_key(gguf_ctx_, "general.architecture");
    if (arch_key >= 0) {
      const char* arch = gguf_get_val_str(gguf_ctx_, arch_key);
      std::string fb_key = std::string(arch) + ".rope.freq_base";
      int64_t fb_id = gguf_find_key(gguf_ctx_, fb_key.c_str());
      if (fb_id >= 0) freq_base = gguf_get_val_f32(gguf_ctx_, fb_id);

      std::string emb_key = std::string(arch) + ".embedding_length";
      std::string head_key = std::string(arch) + ".attention.head_count";
      int64_t emb_id = gguf_find_key(gguf_ctx_, emb_key.c_str());
      int64_t head_id = gguf_find_key(gguf_ctx_, head_key.c_str());
      if (emb_id >= 0 && head_id >= 0) {
        int emb = (int)gguf_get_val_u32(gguf_ctx_, emb_id);
        int heads = (int)gguf_get_val_u32(gguf_ctx_, head_id);
        if (heads > 0) dim = emb / heads;
      }
    }

    int half_dim = (int)(dim / 2);
    // Allocate owning buffer.
    float* buf = new float[half_dim];
    for (int i = 0; i < half_dim; ++i) {
      buf[i] = 1.0f / std::pow(freq_base, (float)(2 * i) / (float)dim);
    }
    auto deleter = [](void* /*ctx*/, void* data, size_t /*size*/) {
      delete[] static_cast<float*>(data);
    };
    return executorch::runtime::FreeableBuffer(
        buf, half_dim * sizeof(float), deleter);
  }

 public:

  executorch::runtime::Error load_data_into(
      executorch::aten::string_view key,
      void* buffer,
      size_t size) const override {
    auto res = get_data(key);
    if (!res.ok()) return res.error();
    auto buf = std::move(res.get());
    size_t copy = (buf.size() < size) ? buf.size() : size;
    std::memcpy(buffer, buf.data(), copy);
    return executorch::runtime::Error::Ok;
  }

  executorch::runtime::Result<uint32_t> get_num_keys() const override {
    return static_cast<uint32_t>(keys_.size());
  }

  executorch::runtime::Result<const char*> get_key(uint32_t index) const override {
    if (index >= keys_.size()) return executorch::runtime::Error::InvalidArgument;
    return keys_[index].c_str();
  }

  const std::string& path() const { return path_; }
  int64_t num_tensors() const { return static_cast<int64_t>(keys_.size()); }

 private:
  /// Try to serve a fused projection weight by concatenating component tensors.
  /// Handles QKV fusion (attn_q + attn_k + attn_v) and gate/up fusion
  /// (ffn_gate + ffn_up).
  executorch::runtime::Result<executorch::runtime::FreeableBuffer>
  try_fused_lookup(const std::string& key) const {
    // QKV fusion: key ends with attn_q.weight → concat attn_q + attn_k + attn_v
    std::string suffix_q = ".attn_q.weight";
    if (key.size() > suffix_q.size() &&
        key.compare(key.size() - suffix_q.size(), suffix_q.size(), suffix_q) == 0) {
      std::string prefix = key.substr(0, key.size() - suffix_q.size());
      std::string k_key = prefix + ".attn_k.weight";
      std::string v_key = prefix + ".attn_v.weight";
      return concat_tensors({key, k_key, v_key});
    }
    // Gate/Up fusion: key ends with ffn_gate.weight → concat ffn_gate + ffn_up
    std::string suffix_gate = ".ffn_gate.weight";
    if (key.size() > suffix_gate.size() &&
        key.compare(key.size() - suffix_gate.size(), suffix_gate.size(), suffix_gate) == 0) {
      std::string prefix = key.substr(0, key.size() - suffix_gate.size());
      std::string up_key = prefix + ".ffn_up.weight";
      return concat_tensors({key, up_key});
    }
    return executorch::runtime::Error::NotFound;
  }

  /// Concatenate multiple GGUF tensors into a single owning buffer.
  executorch::runtime::Result<executorch::runtime::FreeableBuffer>
  concat_tensors(const std::vector<std::string>& names) const {
    size_t total = 0;
    std::vector<std::pair<const void*, size_t>> parts;
    for (const auto& name : names) {
      struct ggml_tensor* gt = ggml_get_tensor(tensor_ctx_, name.c_str());
      if (!gt || !gt->data) return executorch::runtime::Error::NotFound;
      size_t nb = ggml_nbytes(gt);
      parts.push_back({gt->data, nb});
      total += nb;
    }
    uint8_t* buf = new uint8_t[total];
    size_t offset = 0;
    for (const auto& [data, nb] : parts) {
      std::memcpy(buf + offset, data, nb);
      offset += nb;
    }
    auto deleter = [](void* /*ctx*/, void* data, size_t /*size*/) {
      delete[] static_cast<uint8_t*>(data);
    };
    return executorch::runtime::FreeableBuffer(buf, total, deleter);
  }

  struct gguf_context* gguf_ctx_ = nullptr;
  struct ggml_context* tensor_ctx_ = nullptr;
  std::string path_;
  std::vector<std::string> keys_;
  std::unordered_map<std::string, uint32_t> key_to_idx_;
};

}  // namespace executorch_ggml
