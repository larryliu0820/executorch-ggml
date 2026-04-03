# GGUF + PTE Pipeline

Run any GGUF model through ExecuTorch's GGML backend: export a lightweight PTE (graph only, ~200 KB), load weights from the original GGUF at runtime.

## Quick Start

```python
from executorch_ggml import export_gguf_to_pte, GGUFExportConfig, GGUFModule

# Export: GGUF → weight-less PTE
config = GGUFExportConfig(max_seq_len=128, preserve_dynamic_shapes=True, enable_quantization=True)
export_gguf_to_pte("model.gguf", "model.pte", config)

# Run: PTE (graph) + GGUF (weights)
module = GGUFModule("model.pte", "model.gguf")
out = module.forward(input_ids, cache_position)
```

C++ benchmark:
```bash
./build/benchmark/benchmark_llm model.pte --gguf model.gguf --n-decode 64
```

## Architecture

```
Export (Python):
  GGUF file → GGUFAnalyzer → model config
                            → WeightNameMapper (PyTorch FQN ↔ GGUF tensor names)
                            → PyTorch model from config (no weights loaded)
                            → torch.export + GGML partitioner
                            → PTE with GGUF tensor names as data_keys, no weight data

Runtime (C++):
  PTE (graph IR)  → Program::load_method("forward", ..., &gguf_data_map)
  GGUF (weights)  → GGUFNamedDataMap (implements NamedDataMap interface)
                  → Backend reads weights via get_data(key) — same as CUDA backend
                  → inv_freq computed from GGUF metadata, KV caches zero-init'd
```

## Performance (Qwen3-0.6B Q8_0, A100)

| Path | PTE Size | Decode tok/s | ms/tok |
|------|----------|-------------|--------|
| Weights in PTE | 762 MB | 410 | 2.4 |
| GGUF (weights external) | 213 KB | 410 | 2.4 |

Zero overhead from GGUF weight loading. Graph caching enabled by INDEX_PUT
and I32-to-F32 cast fixes that eliminated input-derived eager constants.

## Components

| File | Purpose |
|------|---------|
| `runtime/gguf_data_map.h` | `GGUFNamedDataMap` — NamedDataMap backed by .gguf, computes inv_freq |
| `python/executorch_ggml/_ggml_backend_pybind.cpp` | `GGUFRuntime` — C++ class: PTE + GGUF → load_method → execute |
| `python/executorch_ggml/gguf_module.py` | `GGUFModule` — Python API wrapping GGUFRuntime |
| `python/executorch_ggml/export_gguf.py` | `export_gguf_to_pte()` — GGUF → weight-less PTE |
| `python/executorch_ggml/weight_mapping.py` | `WeightNameMapper` — bidirectional FQN ↔ GGUF names (310/310 Qwen3) |
| `python/executorch_ggml/gguf_analyzer.py` | `GGUFAnalyzer` — GGUF metadata extraction |
| `benchmark/benchmark_llm.cpp` | `--gguf` flag: uses GGUFRunner with GGUFNamedDataMap |

## Next Steps

### 1. Expand architecture support

Currently Qwen3 only. To add a new architecture:
1. Add mapping rules in `WeightNameMapper._ARCH_MAPPINGS`
2. Add config parser in `GGUFAnalyzer._get_<arch>_config()`
3. Add model creator in `export_gguf._create_<arch>_model_from_config()`

Candidates: Llama 3, Mistral, Phi, Gemma.

### 3. GGUF quantization type matching

The export currently creates a PyTorch model with F32 weights, then applies
Q8_0 quantization during GGML partitioning. The PTE IR types must match
the GGUF tensor types. Currently handles Q8_0; extend for Q4_0, Q6_K, etc.

### 4. Multi-method support

Voxtral/Parakeet have multiple methods (preprocessor, encoder, decoder).
Each would need its own PTE with shared GGUF weights.

### 5. Benchmark: benchmark_llm with --gguf for C++ runner

The `GGUFRunner` in benchmark_llm uses `Program::load_method` directly
(bypasses Module). Consider whether the main runner (`runner/main.cpp`)
should also support `--gguf` for production use.
