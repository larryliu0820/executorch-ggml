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

  // Random embeddings for decode
  std::mt19937 rng(42);
  std::normal_distribution<float> dist(0.0f, 0.02f);
  std::vector<float> embeds(dim);
  for (auto& x : embeds) x = dist(rng);

  // Warmup (5 steps)
  fprintf(stderr, "Warmup ...\n");
  for (int pos = 0; pos < 5; pos++) {
    auto emb = from_blob(embeds.data(), {1, 1, dim},
                         executorch::aten::ScalarType::Float);
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
    auto emb = from_blob(embeds.data(), {1, 1, dim},
                         executorch::aten::ScalarType::Float);
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
