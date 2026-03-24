/**
 * benchmark_voxtral.cpp — Voxtral decoder decode benchmark.
 *
 * Measures single-token decode throughput with proper CUDA sync.
 * Supports GGML_SKIP_OUTPUT_COPY=1 + CUDA argmax.
 *
 * Usage:
 *   ./benchmark_voxtral <model.pte> [--n-decode 100]
 *   GGML_SKIP_OUTPUT_COPY=1 ./benchmark_voxtral <model.pte>
 */

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/core/result.h>

using executorch::extension::Module;
using executorch::extension::TensorPtr;
using executorch::extension::from_blob;
using executorch::runtime::Error;

extern "C" int64_t cuda_argmax_f32(const void* gpu_data, int64_t n);

static double now_ms() {
  return std::chrono::duration<double, std::milli>(
      std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <model.pte> [--n-decode N] [--dim D]\n", argv[0]);
    return 1;
  }
  const char* model_path = argv[1];
  int n_decode = 100;
  int dim = 3072;  // Voxtral model dim

  bool skip_copy = false;
  {
    const char* env = std::getenv("GGML_SKIP_OUTPUT_COPY");
    skip_copy = env && std::string(env) != "0";
  }

  for (int i = 2; i < argc; i++) {
    if (strcmp(argv[i], "--n-decode") == 0 && i + 1 < argc)
      n_decode = atoi(argv[++i]);
    else if (strcmp(argv[i], "--dim") == 0 && i + 1 < argc)
      dim = atoi(argv[++i]);
  }

  fprintf(stderr, "Loading %s (dim=%d, skip_copy=%d)...\n",
          model_path, dim, skip_copy);
  Module model(model_path);

  auto load_err = model.load_method("text_decoder");
  if (load_err != Error::Ok) {
    // Try "forward" as fallback
    load_err = model.load_method("forward");
    if (load_err != Error::Ok) {
      fprintf(stderr, "load_method failed: %d\n", (int)load_err);
      return 1;
    }
  }

  // Detect model input dtype (BF16 vs F32) from method metadata
  auto meta = model.method_meta("text_decoder");
  bool is_bf16 = false;
  if (meta.ok()) {
    auto input_meta = meta->input_tensor_meta(0);
    if (input_meta.ok()) {
      is_bf16 = (input_meta->scalar_type() == executorch::aten::ScalarType::BFloat16);
    }
  }
  auto embed_dtype = is_bf16 ? executorch::aten::ScalarType::BFloat16
                             : executorch::aten::ScalarType::Float;
  fprintf(stderr, "Model input dtype: %s\n", is_bf16 ? "BF16" : "F32");

  // Random embeddings for decode (generate in F32, convert if needed)
  std::mt19937 rng(42);
  std::normal_distribution<float> dist(0.0f, 0.02f);
  std::vector<float> embeds_f32(dim);
  for (auto& x : embeds_f32) x = dist(rng);

  // BF16 storage (F32 bits >> 16, stored as uint16_t)
  std::vector<uint16_t> embeds_bf16(dim);
  if (is_bf16) {
    for (int i = 0; i < dim; i++) {
      uint32_t bits;
      memcpy(&bits, &embeds_f32[i], sizeof(bits));
      embeds_bf16[i] = static_cast<uint16_t>(bits >> 16);
    }
  }

  // Warmup (5 steps)
  fprintf(stderr, "Warmup ...\n");
  for (int pos = 0; pos < 5; pos++) {
    TensorPtr emb;
    if (is_bf16) {
      emb = from_blob(embeds_bf16.data(), {1, 1, dim}, embed_dtype);
    } else {
      emb = from_blob(embeds_f32.data(), {1, 1, dim}, embed_dtype);
    }
    int64_t cache_pos = pos;
    auto pos_t = from_blob(&cache_pos, {1},
                           executorch::aten::ScalarType::Long);
    auto r = model.execute("text_decoder", {*emb, *pos_t});
    if (!r.ok()) {
      fprintf(stderr, "Warmup step %d failed: %d\n", pos, (int)r.error());
      return 1;
    }
  }
  fprintf(stderr, "Warmup done.\n");

  // Decode benchmark
  fprintf(stderr, "Decode (%d tokens, skip_copy=%s) ...\n",
          n_decode, skip_copy ? "ON" : "OFF");

  double t_total = 0;
  int64_t vocab_size = 0;
  std::vector<double> step_times;

  for (int step = 0; step < n_decode; step++) {
    int64_t pos = 10 + step;
    TensorPtr emb;
    if (is_bf16) {
      emb = from_blob(embeds_bf16.data(), {1, 1, dim}, embed_dtype);
    } else {
      emb = from_blob(embeds_f32.data(), {1, 1, dim}, embed_dtype);
    }
    auto pos_t = from_blob(&pos, {1}, executorch::aten::ScalarType::Long);

    double t0 = now_ms();
    auto result = model.execute("text_decoder", {*emb, *pos_t});
    if (!result.ok()) {
      fprintf(stderr, "Step %d failed\n", step);
      return 1;
    }

    auto& logits = result->at(0).toTensor();
    if (vocab_size == 0) vocab_size = logits.size(logits.dim() - 1);

    int64_t token;
    if (skip_copy) {
      token = cuda_argmax_f32(logits.const_data_ptr<float>(), vocab_size);
    } else if (logits.scalar_type() == executorch::aten::ScalarType::BFloat16) {
      // BF16 logits: convert to F32 for argmax
      const uint16_t* ld = static_cast<const uint16_t*>(logits.const_data_ptr());
      token = 0;
      float max_val = -1e30f;
      for (int64_t v = 0; v < vocab_size; v++) {
        uint32_t bits = static_cast<uint32_t>(ld[v]) << 16;
        float val;
        memcpy(&val, &bits, sizeof(val));
        if (val > max_val) { max_val = val; token = v; }
      }
    } else {
      const float* ld = logits.const_data_ptr<float>();
      token = 0;
      float max_val = ld[0];
      for (int64_t v = 1; v < vocab_size; v++) {
        if (ld[v] > max_val) { max_val = ld[v]; token = v; }
      }
    }
    double dt = now_ms() - t0;
    step_times.push_back(dt);
    t_total += dt;
  }

  // Stats
  std::sort(step_times.begin(), step_times.end());
  int trim = n_decode / 10;
  double trimmed_total = 0;
  int trimmed_count = 0;
  for (int i = trim; i < n_decode - trim; i++) {
    trimmed_total += step_times[i];
    trimmed_count++;
  }
  double avg = trimmed_total / trimmed_count;
  double p50 = step_times[n_decode / 2];

  fprintf(stderr, "Decode: avg=%.2f ms/tok, p50=%.2f ms/tok, "
          "min=%.2f ms/tok (%.1f tok/s)\n",
          avg, p50, step_times[0],
          trimmed_count / (trimmed_total / 1000.0));

  // Machine-readable output
  printf("voxtral_decoder,%s,%d,%.2f,%.1f\n",
         skip_copy ? "skip_copy" : "standard",
         n_decode, avg,
         trimmed_count / (trimmed_total / 1000.0));

  return 0;
}
