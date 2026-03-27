/**
 * benchmark_llm.cpp — Minimal LLM decode benchmark for executorch-ggml.
 *
 * Loads a .pte file, runs prefill + N decode steps, and reports tok/s.
 * No tokenizer needed — uses raw token IDs.
 *
 * Usage:
 *   ./benchmark_llm <model.pte> [--n-decode 32] [--prompt-len 5]
 */

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/core/result.h>

using executorch::extension::Module;
using executorch::extension::TensorPtr;
using executorch::extension::from_blob;
using executorch::runtime::Error;

// CUDA argmax for GPU-resident logits (when GGML_SKIP_OUTPUT_COPY=1)
#ifdef GGML_FUSED_KERNELS
extern "C" int64_t cuda_argmax_f32(const void* gpu_data, int64_t n);
#endif

// CPU argmax for Metal unified memory and CPU backends
static int64_t cpu_argmax_f32(const float* data, int64_t n) {
  int64_t idx = 0; float mx = data[0];
  for (int64_t v = 1; v < n; v++) { if (data[v] > mx) { mx = data[v]; idx = v; } }
  return idx;
}

static double now_ms() {
  auto t = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::milli>(t.time_since_epoch()).count();
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <model.pte> [--n-decode N] [--prompt-len N]\n", argv[0]);
    return 1;
  }
  const char* model_path = argv[1];
  int n_decode = 32;
  int prompt_len = 5;
  // When CUDA fused kernels are available, skip the D2H logits copy and
  // do argmax on GPU.  Set the env var so the runtime also skips.
#ifdef GGML_FUSED_KERNELS
  bool skip_copy = true;
  {
    const char* env = std::getenv("GGML_SKIP_OUTPUT_COPY");
    if (env && std::string(env) == "0") {
      skip_copy = false;
    } else if (!env) {
      setenv("GGML_SKIP_OUTPUT_COPY", "1", 0);
    }
  }
#else
  bool skip_copy = false;
#endif

  for (int i = 2; i < argc; i++) {
    if (strcmp(argv[i], "--n-decode") == 0 && i + 1 < argc) {
      n_decode = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--prompt-len") == 0 && i + 1 < argc) {
      prompt_len = atoi(argv[++i]);
    }
  }

  fprintf(stderr, "Loading %s ...\n", model_path);
  Module model(model_path);

  // Verify model loaded and method exists
  auto method_names = model.method_names();
  if (!method_names.ok()) {
    fprintf(stderr, "Failed to get method names: %d\n", (int)method_names.error());
    return 1;
  }
  fprintf(stderr, "Methods:");
  for (auto& name : *method_names) {
    fprintf(stderr, " %s", name.c_str());
  }
  fprintf(stderr, "\n");

  auto load_err = model.load_method("forward");
  if (load_err != Error::Ok) {
    fprintf(stderr, "load_method('forward') failed: %d (0x%x)\n", (int)load_err, (int)load_err);
    return 1;
  }

  // Build prompt token IDs [1, 2, 3, ..., prompt_len]
  std::vector<int64_t> prompt_ids(prompt_len);
  for (int i = 0; i < prompt_len; i++) prompt_ids[i] = i + 1;

  // Cache positions [0, 1, ..., prompt_len-1]
  std::vector<int64_t> cache_pos(prompt_len);
  for (int i = 0; i < prompt_len; i++) cache_pos[i] = i;

  // --- Warmup (triggers graph build + sched_alloc for both shapes) ---
  {
    fprintf(stderr, "Warmup ...\n");
    auto w_ids = from_blob(prompt_ids.data(), {1, prompt_len},
                           executorch::aten::ScalarType::Long);
    auto w_pos = from_blob(cache_pos.data(), {prompt_len},
                           executorch::aten::ScalarType::Long);
    (void)model.forward({*w_ids, *w_pos});  // prefill shape
    int64_t tok = 1, pos = prompt_len;
    auto w_tok = from_blob(&tok, {1, 1}, executorch::aten::ScalarType::Long);
    auto w_p = from_blob(&pos, {1}, executorch::aten::ScalarType::Long);
    (void)model.forward({*w_tok, *w_p});     // decode shape
    fprintf(stderr, "Warmup done.\n");
  }

  // --- Prefill ---
  fprintf(stderr, "Prefill (%d tokens) ...\n", prompt_len);
  auto input_ids = from_blob(
      prompt_ids.data(), {1, prompt_len},
      executorch::aten::ScalarType::Long);
  auto pos_tensor = from_blob(
      cache_pos.data(), {prompt_len},
      executorch::aten::ScalarType::Long);

  double t0 = now_ms();
  auto prefill_result = model.forward({*input_ids, *pos_tensor});
  double t_prefill = now_ms() - t0;

  if (!prefill_result.ok()) {
    fprintf(stderr, "Prefill failed: %d\n", (int)prefill_result.error());
    return 1;
  }

  // Get logits and find argmax for last token
  auto& logits = prefill_result->at(0).toTensor();
  int64_t vocab_size = logits.size(2);
  int64_t next_token = 0;
  if (skip_copy) {
    const float* data = logits.const_data_ptr<float>();
    int64_t offset = (prompt_len - 1) * vocab_size;
#ifdef GGML_FUSED_KERNELS
    next_token = cuda_argmax_f32(data + offset, vocab_size);
#else
    next_token = cpu_argmax_f32(data + offset, vocab_size);
#endif
  } else {
    const float* logits_data = logits.const_data_ptr<float>();
    int64_t offset = (prompt_len - 1) * vocab_size;
    float max_val = logits_data[offset];
    for (int64_t v = 1; v < vocab_size; v++) {
      if (logits_data[offset + v] > max_val) {
        max_val = logits_data[offset + v];
        next_token = v;
      }
    }
  }

  fprintf(stderr, "Prefill: %.1f ms (%.1f tok/s), first token=%lld\n",
          t_prefill, prompt_len / (t_prefill / 1000.0), (long long)next_token);

  // --- Decode ---
  fprintf(stderr, "Decode (%d tokens) ...\n", n_decode);
  std::vector<int64_t> generated;
  generated.push_back(next_token);

  double t_decode_total = 0;
  for (int step = 0; step < n_decode - 1; step++) {
    int64_t pos = prompt_len + step;
    auto tok = from_blob(&next_token, {1, 1}, executorch::aten::ScalarType::Long);
    auto pos_t = from_blob(&pos, {1}, executorch::aten::ScalarType::Long);

    double t_step = now_ms();
    auto result = model.forward({*tok, *pos_t});
    t_decode_total += now_ms() - t_step;

    if (!result.ok()) {
      fprintf(stderr, "Decode step %d failed: %d\n", step, (int)result.error());
      return 1;
    }

    auto& step_logits = result->at(0).toTensor();
    if (skip_copy) {
#ifdef GGML_FUSED_KERNELS
      next_token = cuda_argmax_f32(step_logits.const_data_ptr<float>(), vocab_size);
#else
      next_token = cpu_argmax_f32(step_logits.const_data_ptr<float>(), vocab_size);
#endif
    } else {
      const float* ld = step_logits.const_data_ptr<float>();
      next_token = 0;
      float max_val = ld[0];
      for (int64_t v = 1; v < vocab_size; v++) {
        if (ld[v] > max_val) {
          max_val = ld[v];
          next_token = v;
        }
      }
    }
    generated.push_back(next_token);

    if (next_token == 2) break; // EOS
  }

  int actual_decode = (int)generated.size() - 1;
  fprintf(stderr, "Decode: %.1f ms total, %.1f ms/tok (%.1f tok/s)\n",
          t_decode_total,
          actual_decode > 0 ? t_decode_total / actual_decode : 0,
          actual_decode > 0 ? actual_decode / (t_decode_total / 1000.0) : 0);

  // Print summary
  printf("| model | prefill tok/s | decode tok/s | decode ms/tok |\n");
  printf("| --- | --- | --- | --- |\n");
  printf("| %s | %.1f | %.1f | %.1f |\n",
         model_path,
         prompt_len / (t_prefill / 1000.0),
         actual_decode > 0 ? actual_decode / (t_decode_total / 1000.0) : 0,
         actual_decode > 0 ? t_decode_total / actual_decode : 0);

  fprintf(stderr, "Tokens: ");
  for (auto t : generated) fprintf(stderr, "%lld ", (long long)t);
  fprintf(stderr, "\n");

  return 0;
}
