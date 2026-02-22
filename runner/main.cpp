/*
 * C++ runner for executorch-ggml benchmark comparison with llama.cpp.
 *
 * Usage:
 *   ./ggml_runner <model.pte> [--prompt "text"] [--tokens N] [--warmup N]
 */

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/result.h>

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::extension::from_blob;
using executorch::extension::Module;
using executorch::runtime::EValue;
using executorch::runtime::Result;

struct BenchmarkStats {
  double prefill_time_ms = 0;
  double decode_time_ms = 0;
  int prefill_tokens = 0;
  int decode_tokens = 0;

  double prefill_tokens_per_sec() const {
    return prefill_time_ms > 0 ? (prefill_tokens * 1000.0 / prefill_time_ms) : 0;
  }

  double decode_tokens_per_sec() const {
    return decode_time_ms > 0 ? (decode_tokens * 1000.0 / decode_time_ms) : 0;
  }

  void print() const {
    std::cout << "\n=== Benchmark Results ===" << std::endl;
    std::cout << "Prefill: " << prefill_tokens << " tokens in "
              << prefill_time_ms << " ms ("
              << prefill_tokens_per_sec() << " tokens/sec)" << std::endl;
    std::cout << "Decode:  " << decode_tokens << " tokens in "
              << decode_time_ms << " ms ("
              << decode_tokens_per_sec() << " tokens/sec)" << std::endl;
    std::cout << "Total:   " << (prefill_tokens + decode_tokens) << " tokens in "
              << (prefill_time_ms + decode_time_ms) << " ms" << std::endl;
  }
};

int64_t sample_greedy(const float* logits, size_t vocab_size) {
  int64_t max_idx = 0;
  float max_val = logits[0];
  for (size_t i = 1; i < vocab_size; ++i) {
    if (logits[i] > max_val) {
      max_val = logits[i];
      max_idx = static_cast<int64_t>(i);
    }
  }
  return max_idx;
}

BenchmarkStats run_benchmark(
    Module& model,
    const std::vector<int64_t>& prompt_tokens,
    int max_new_tokens,
    int warmup_iterations) {
  BenchmarkStats stats;

  // Warmup
  if (warmup_iterations > 0) {
    std::cout << "Running " << warmup_iterations << " warmup iterations..." << std::endl;
    for (int w = 0; w < warmup_iterations; ++w) {
      int64_t token = prompt_tokens[0];
      auto input = from_blob(&token, {1, 1}, ScalarType::Long);
      int64_t pos = 0;
      auto pos_tensor = from_blob(&pos, {1}, ScalarType::Long);
      model.forward(input, pos_tensor);
    }
    std::cout << "Warmup complete." << std::endl;
  }

  // Prefill phase - process prompt tokens one by one
  std::cout << "Prefill: " << prompt_tokens.size() << " tokens" << std::endl;
  auto prefill_start = std::chrono::high_resolution_clock::now();

  std::vector<EValue> outputs;
  for (size_t i = 0; i < prompt_tokens.size(); ++i) {
    int64_t token = prompt_tokens[i];
    auto input = from_blob(&token, {1, 1}, ScalarType::Long);
    int64_t pos = static_cast<int64_t>(i);
    auto pos_tensor = from_blob(&pos, {1}, ScalarType::Long);

    auto result = model.forward(input, pos_tensor);
    if (!result.ok()) {
      std::cerr << "Error during prefill at position " << i << std::endl;
      return stats;
    }
    outputs = result.get();
  }

  auto prefill_end = std::chrono::high_resolution_clock::now();
  stats.prefill_time_ms = std::chrono::duration<double, std::milli>(
      prefill_end - prefill_start).count();
  stats.prefill_tokens = static_cast<int>(prompt_tokens.size());

  // Decode phase - generate new tokens
  std::cout << "Decode: generating " << max_new_tokens << " tokens" << std::endl;
  auto decode_start = std::chrono::high_resolution_clock::now();

  std::vector<int64_t> generated_tokens;
  int64_t current_pos = static_cast<int64_t>(prompt_tokens.size());

  for (int i = 0; i < max_new_tokens; ++i) {
    // Get logits from last forward pass
    Tensor logits_tensor = outputs[0].toTensor();
    const float* logits = logits_tensor.data_ptr<float>();
    size_t vocab_size = static_cast<size_t>(logits_tensor.numel());

    // Sample next token
    int64_t next_token = sample_greedy(logits, vocab_size);
    generated_tokens.push_back(next_token);

    // Forward pass for next token
    auto input = from_blob(&next_token, {1, 1}, ScalarType::Long);
    auto pos_tensor = from_blob(&current_pos, {1}, ScalarType::Long);

    auto result = model.forward(input, pos_tensor);
    if (!result.ok()) {
      std::cerr << "Error during decode at position " << current_pos << std::endl;
      break;
    }
    outputs = result.get();
    current_pos++;
  }

  auto decode_end = std::chrono::high_resolution_clock::now();
  stats.decode_time_ms = std::chrono::duration<double, std::milli>(
      decode_end - decode_start).count();
  stats.decode_tokens = static_cast<int>(generated_tokens.size());

  // Print generated token IDs
  std::cout << "Generated tokens: ";
  for (auto t : generated_tokens) {
    std::cout << t << " ";
  }
  std::cout << std::endl;

  return stats;
}

void print_usage(const char* prog) {
  std::cout << "Usage: " << prog << " <model.pte> [options]" << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << "  --tokens N     Number of tokens to generate (default: 32)" << std::endl;
  std::cout << "  --warmup N     Number of warmup iterations (default: 3)" << std::endl;
  std::cout << "  --prompt-ids   Comma-separated token IDs for prompt" << std::endl;
  std::cout << "                 (default: 785,6722,315,9625,374 = 'The capital of France is')" << std::endl;
}

std::vector<int64_t> parse_token_ids(const std::string& ids_str) {
  std::vector<int64_t> tokens;
  std::string current;
  for (char c : ids_str) {
    if (c == ',') {
      if (!current.empty()) {
        tokens.push_back(std::stoll(current));
        current.clear();
      }
    } else if (c >= '0' && c <= '9') {
      current += c;
    }
  }
  if (!current.empty()) {
    tokens.push_back(std::stoll(current));
  }
  return tokens;
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    print_usage(argv[0]);
    return 1;
  }

  std::string model_path = argv[1];
  int max_tokens = 32;
  int warmup = 3;
  std::vector<int64_t> prompt_tokens = {785, 6722, 315, 9625, 374}; // "The capital of France is"

  // Parse arguments
  for (int i = 2; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--tokens" && i + 1 < argc) {
      max_tokens = std::stoi(argv[++i]);
    } else if (arg == "--warmup" && i + 1 < argc) {
      warmup = std::stoi(argv[++i]);
    } else if (arg == "--prompt-ids" && i + 1 < argc) {
      prompt_tokens = parse_token_ids(argv[++i]);
    } else if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      return 0;
    }
  }

  std::cout << "=== ExecuTorch GGML Runner ===" << std::endl;
  std::cout << "Model: " << model_path << std::endl;
  std::cout << "Prompt tokens: " << prompt_tokens.size() << std::endl;
  std::cout << "Max new tokens: " << max_tokens << std::endl;

  // Load model
  std::cout << "\nLoading model..." << std::endl;
  auto load_start = std::chrono::high_resolution_clock::now();

  Module model(model_path, Module::LoadMode::MmapUseMlockIgnoreErrors);

  auto load_end = std::chrono::high_resolution_clock::now();
  double load_time_ms = std::chrono::duration<double, std::milli>(
      load_end - load_start).count();
  std::cout << "Model loaded in " << load_time_ms << " ms" << std::endl;

  // Run benchmark
  BenchmarkStats stats = run_benchmark(model, prompt_tokens, max_tokens, warmup);
  stats.print();

  return 0;
}
