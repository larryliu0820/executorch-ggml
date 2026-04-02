/**
 * benchmark_llm.cpp — Minimal LLM decode benchmark for executorch-ggml.
 *
 * Loads a .pte file, runs prefill + N decode steps, and reports tok/s.
 * When --gguf is provided, weights are loaded from the GGUF file via
 * GGUFNamedDataMap instead of from the PTE.
 *
 * Usage:
 *   ./benchmark_llm <model.pte> [--gguf model.gguf] [--n-decode 32] [--prompt-len 5]
 */

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/extension/memory_allocator/malloc_memory_allocator.h>
#include <executorch/extension/data_loader/mmap_data_loader.h>
#include <executorch/runtime/core/result.h>
#include <gguf.h>

#include "../runtime/gguf_data_map.h"

using executorch::extension::Module;
using executorch::extension::MmapDataLoader;
using executorch::extension::MallocMemoryAllocator;
using executorch::extension::TensorPtr;
using executorch::extension::from_blob;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::HierarchicalAllocator;
using executorch::runtime::MemoryManager;
using executorch::runtime::Method;
using executorch::runtime::Program;
using executorch::runtime::Span;

// CUDA argmax for GPU-resident logits (when GGML_SKIP_OUTPUT_COPY=1)
#ifdef GGML_FUSED_KERNELS
extern "C" int64_t cuda_argmax_f32(const void* gpu_data, int64_t n);
#endif

static int64_t cpu_argmax_f32(const float* data, int64_t n) {
  int64_t idx = 0; float mx = data[0];
  for (int64_t v = 1; v < n; v++) { if (data[v] > mx) { mx = data[v]; idx = v; } }
  return idx;
}

static double now_ms() {
  auto t = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::milli>(t.time_since_epoch()).count();
}

// --- Abstract interface for running forward() on either Module or GGUF. ---
struct Runner {
  virtual ~Runner() = default;
  virtual Error forward(const std::vector<EValue>& inputs,
                        std::vector<EValue>& outputs) = 0;
};

// Module-based runner (weights in PTE).
struct ModuleRunner : Runner {
  Module module;
  ModuleRunner(const char* pte_path) : module(pte_path) {}

  Error forward(const std::vector<EValue>& inputs,
                std::vector<EValue>& outputs) override {
    auto result = module.forward(inputs);
    if (!result.ok()) return result.error();
    outputs = std::move(*result);
    return Error::Ok;
  }
};

// GGUF-based runner (weights from .gguf via GGUFNamedDataMap).
struct GGUFRunner : Runner {
  std::unique_ptr<MmapDataLoader> loader;
  std::unique_ptr<Program> program;
  std::unique_ptr<executorch_ggml::GGUFNamedDataMap> gguf_map;
  std::unique_ptr<Method> method;

  // Memory management (kept alive for method lifetime).
  std::unique_ptr<MallocMemoryAllocator> mem_alloc;
  std::unique_ptr<MallocMemoryAllocator> tmp_alloc;
  std::vector<std::vector<uint8_t>> planned_bufs;
  std::vector<Span<uint8_t>> planned_spans;
  std::unique_ptr<HierarchicalAllocator> planned_mem;
  std::unique_ptr<MemoryManager> mem_mgr;

  static std::unique_ptr<GGUFRunner> create(const char* pte_path,
                                             const char* gguf_path) {
    auto r = std::make_unique<GGUFRunner>();

    // Load GGUF.
    auto gguf_res = executorch_ggml::GGUFNamedDataMap::load(gguf_path);
    if (!gguf_res.ok()) {
      fprintf(stderr, "Failed to open GGUF: %s\n", gguf_path);
      return nullptr;
    }
    r->gguf_map = std::move(gguf_res.get());
    fprintf(stderr, "GGUF weights: %s (%lld tensors)\n",
            gguf_path, (long long)r->gguf_map->num_tensors());

    // Load PTE program.
    auto loader_res = MmapDataLoader::from(
        pte_path, MmapDataLoader::MlockConfig::UseMlockIgnoreErrors);
    if (!loader_res.ok()) {
      fprintf(stderr, "Failed to mmap PTE: %s\n", pte_path);
      return nullptr;
    }
    r->loader = std::make_unique<MmapDataLoader>(std::move(loader_res.get()));

    auto prog_res = Program::load(r->loader.get());
    if (!prog_res.ok()) {
      fprintf(stderr, "Failed to load program: 0x%x\n", (int)prog_res.error());
      return nullptr;
    }
    r->program = std::make_unique<Program>(std::move(prog_res.get()));

    // Allocate planned memory from method metadata.
    auto meta_res = r->program->method_meta("forward");
    if (!meta_res.ok()) {
      fprintf(stderr, "Failed to get method meta\n");
      return nullptr;
    }
    auto& meta = meta_res.get();
    for (size_t i = 0; i < meta.num_memory_planned_buffers(); ++i) {
      size_t sz = meta.memory_planned_buffer_size(i).get();
      r->planned_bufs.emplace_back(sz);
      r->planned_spans.emplace_back(r->planned_bufs.back().data(), sz);
    }
    r->planned_mem = std::make_unique<HierarchicalAllocator>(
        Span<Span<uint8_t>>(r->planned_spans.data(), r->planned_spans.size()));
    r->mem_alloc = std::make_unique<MallocMemoryAllocator>();
    r->tmp_alloc = std::make_unique<MallocMemoryAllocator>();
    r->mem_mgr = std::make_unique<MemoryManager>(
        r->mem_alloc.get(), r->planned_mem.get(), r->tmp_alloc.get());

    // Load forward method with GGUF as the NamedDataMap.
    auto method_res = r->program->load_method(
        "forward", r->mem_mgr.get(), nullptr, r->gguf_map.get());
    if (!method_res.ok()) {
      fprintf(stderr, "load_method('forward') failed: 0x%x\n",
              (int)method_res.error());
      return nullptr;
    }
    r->method = std::make_unique<Method>(std::move(method_res.get()));
    return r;
  }

  Error forward(const std::vector<EValue>& inputs,
                std::vector<EValue>& outputs) override {
    auto err = method->set_inputs(
        executorch::aten::ArrayRef<EValue>(inputs.data(), inputs.size()));
    if (err != Error::Ok) return err;

    err = method->execute();
    if (err != Error::Ok) return err;

    outputs.clear();
    for (size_t i = 0; i < method->outputs_size(); ++i) {
      outputs.push_back(method->get_output(i));
    }
    return Error::Ok;
  }
};

// --- Helpers ---

static int64_t argmax(const float* data, int64_t vocab_size, bool skip_copy) {
  if (skip_copy) {
#ifdef GGML_FUSED_KERNELS
    return cuda_argmax_f32(data, vocab_size);
#else
    return cpu_argmax_f32(data, vocab_size);
#endif
  }
  return cpu_argmax_f32(data, vocab_size);
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr,
      "Usage: %s <model.pte> [--gguf model.gguf] [--n-decode N] [--prompt-len N]\n",
      argv[0]);
    return 1;
  }
  const char* model_path = argv[1];
  const char* gguf_path = nullptr;
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
    if (strcmp(argv[i], "--n-decode") == 0 && i + 1 < argc)
      n_decode = atoi(argv[++i]);
    else if (strcmp(argv[i], "--prompt-len") == 0 && i + 1 < argc)
      prompt_len = atoi(argv[++i]);
    else if (strcmp(argv[i], "--gguf") == 0 && i + 1 < argc)
      gguf_path = argv[++i];
  }

  // --- Create runner ---
  std::unique_ptr<Runner> runner;
  if (gguf_path) {
    auto gr = GGUFRunner::create(model_path, gguf_path);
    if (!gr) return 1;
    runner = std::move(gr);
  } else {
    auto mr = std::make_unique<ModuleRunner>(model_path);
    auto err = mr->module.load_method("forward");
    if (err != Error::Ok) {
      fprintf(stderr, "load_method('forward') failed: 0x%x\n", (int)err);
      return 1;
    }
    runner = std::move(mr);
  }

  // Build prompt token IDs [1, 2, ..., prompt_len]
  std::vector<int64_t> prompt_ids(prompt_len);
  for (int i = 0; i < prompt_len; i++) prompt_ids[i] = i + 1;
  std::vector<int64_t> cache_pos(prompt_len);
  for (int i = 0; i < prompt_len; i++) cache_pos[i] = i;

  // --- Warmup ---
  {
    fprintf(stderr, "Warmup ...\n");
    auto w_ids = from_blob(prompt_ids.data(), {1, prompt_len},
                           executorch::aten::ScalarType::Long);
    auto w_pos = from_blob(cache_pos.data(), {prompt_len},
                           executorch::aten::ScalarType::Long);
    std::vector<EValue> in = {EValue(*w_ids), EValue(*w_pos)};
    std::vector<EValue> out;
    runner->forward(in, out);

    int64_t tok = 1, pos_val = prompt_len;
    auto w_tok = from_blob(&tok, {1, 1}, executorch::aten::ScalarType::Long);
    auto w_p = from_blob(&pos_val, {1}, executorch::aten::ScalarType::Long);
    in = {EValue(*w_tok), EValue(*w_p)};
    runner->forward(in, out);
    fprintf(stderr, "Warmup done.\n");
  }

  // --- Prefill ---
  fprintf(stderr, "Prefill (%d tokens) ...\n", prompt_len);
  auto input_ids = from_blob(prompt_ids.data(), {1, prompt_len},
                             executorch::aten::ScalarType::Long);
  auto pos_tensor = from_blob(cache_pos.data(), {prompt_len},
                              executorch::aten::ScalarType::Long);

  std::vector<EValue> inputs = {EValue(*input_ids), EValue(*pos_tensor)};
  std::vector<EValue> outputs;

  double t0 = now_ms();
  auto err = runner->forward(inputs, outputs);
  double t_prefill = now_ms() - t0;

  if (err != Error::Ok) {
    fprintf(stderr, "Prefill failed: 0x%x\n", (int)err);
    return 1;
  }

  auto& logits = outputs.at(0).toTensor();
  int64_t vocab_size = logits.size(2);
  int64_t offset = (prompt_len - 1) * vocab_size;
  int64_t next_token = argmax(logits.const_data_ptr<float>() + offset,
                               vocab_size, skip_copy);

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
    inputs = {EValue(*tok), EValue(*pos_t)};

    double t_step = now_ms();
    err = runner->forward(inputs, outputs);
    t_decode_total += now_ms() - t_step;

    if (err != Error::Ok) {
      fprintf(stderr, "Decode step %d failed: 0x%x\n", step, (int)err);
      return 1;
    }

    auto& step_logits = outputs.at(0).toTensor();
    next_token = argmax(step_logits.const_data_ptr<float>(), vocab_size, skip_copy);
    generated.push_back(next_token);
    if (next_token == 2) break; // EOS
  }

  int actual_decode = (int)generated.size() - 1;
  fprintf(stderr, "Decode: %.1f ms total, %.1f ms/tok (%.1f tok/s)\n",
          t_decode_total,
          actual_decode > 0 ? t_decode_total / actual_decode : 0,
          actual_decode > 0 ? actual_decode / (t_decode_total / 1000.0) : 0);

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
