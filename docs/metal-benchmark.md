# Metal Benchmark & Performance Analysis

## Results (Qwen3-0.6B Q8_0, Apple M4 Max)

| Backend | PTE Size | Decode tok/s | Prefill tok/s | ms/tok |
|---------|----------|-------------:|--------------:|-------:|
| executorch-ggml (fused, GGUF weights) | 213 KB | **331** | **776** | 3.0 |
| executorch-ggml (unfused, GGUF weights) | 213 KB | 323 | 922 | 3.1 |
| executorch-ggml (fused, embedded weights) | 762 MB | **328** | — | 3.0 |
| llama.cpp | N/A | 299 | 744 | 3.3 |

executorch-ggml is **111% of llama.cpp** on decode with QKV + gate/up projection
fusion, using the same .gguf weights and the same ggml compute kernels.

For comparison, on CUDA (A100-SXM4-40GB):

| Backend | Decode tok/s | vs llama.cpp |
|---------|-------------:|:-------------|
| executorch-ggml | **411** | **108%** |
| llama.cpp | 380 | baseline |

## Why executorch-ggml matches or exceeds llama.cpp

Both use the exact same ggml kernels (`third-party/llama.cpp/ggml`). The
difference comes from graph construction and per-step overhead:

### 1. RMSNorm weight folding (biggest factor)

llama.cpp: `rms_norm(x) * weight` then `linear(result, W)` — two ops.

executorch-ggml: the AOT pass `fold_rms_norm_weights` absorbs the norm weight
into the downstream linear at export time (`W_new = W * weight`). This
eliminates 57 MUL nodes per decode step — 57 fewer kernel dispatches and 57
fewer weight tensor memory reads.

The remaining 56 q_norm/k_norm weights can't be folded (RoPE follows, not a
linear), but the Metal graph optimizer auto-fuses `rms_norm + mul` into a
single `kernel_rms_norm_mul_f32_4` dispatch for those.

### 2. Zero per-step graph overhead

llama.cpp rebuilds portions of its graph each step (KV cache views, position
embeddings, attention masks). executorch-ggml's graph cache reuses the entire
`ggml_cgraph` + scheduler allocation across calls with the same input shape:

```
[perf] execute #10: build=0.00ms alloc=0.00ms compute=0.33ms output=2.80ms | HIT
```

Graph cache is enabled by default. The `has_input_derived_eager` flag ensures
correctness: graphs with input-dependent constants are rebuilt automatically.

### 3. No abstraction layer overhead

llama.cpp: `llama_model → llama_context → llama_batch → llama_graph_build →
ggml_backend_sched_graph_compute`.

executorch-ggml: `Module::forward → ggml_backend_sched_graph_compute_async`.

Less indirection per step. The ExecuTorch runtime is a thin wrapper over the
delegate's execute() call.

### 4. AOT-optimized graph topology

torch.export + GGML partitioner produces a graph that's been through multiple
optimization passes. The resulting op ordering may be slightly more favorable
for the Metal graph optimizer's concurrency reordering.

### What does NOT explain the difference

- **Kernel performance** — identical ggml kernels from the same source tree.
- **Quantization** — same Q8_0 weights from the same .gguf file.
- **Memory bandwidth** — same hardware, same data sizes.
- **Graph cache** — llama.cpp also reuses its graph; our cache is slightly
  more aggressive (zero rebuild vs partial rebuild).

The gains are modest (4–8%) and consistent across platforms. On larger models
where compute dominates, the gap would likely shrink toward zero.

## Per-step timing breakdown

Steady-state decode (graph cache HIT):

| Phase | Time | Notes |
|-------|-----:|-------|
| build_graph | 0.0 ms | Cached — no IR parse, no tensor creation |
| sched_alloc | 0.0 ms | Cached — scheduler stays allocated |
| input copy | 0.0 ms | Single token (8 bytes) |
| compute | 0.3 ms | Metal command buffer submission (async) |
| output sync | 2.8 ms | Wait for Metal GPU + read logits |
| **total** | **~3.1 ms** | **~323 tok/s** |

The 0.3 ms compute time is just command buffer submission — the GPU is still
running. The 2.8 ms output time includes waiting for Metal to finish computing
plus reading the logits tensor (151936 × 4 = 608 KB). On Apple Silicon with
unified memory, the "read" is a memcpy from shared memory, not a DMA transfer.

## Per-op profile (decode step, with projection fusion)

Captured with `GGML_PROFILE=1` (forces per-node Metal sync, so times are
inflated by ~90 us/node sync overhead — use for relative comparison only):

| Op | Count | % of dispatches | Notes |
|----|------:|----------------:|-------|
| MUL_MAT | 113 | 22% | Q8_0 matmul (28×(1 fused QKV + 1 fused gate/up + o_proj + down_proj) + lm_head) |
| RMS_NORM | 113 | 22% | 28×4 norms + 1 final (auto-fused with MUL by Metal) |
| ADD | 57 | 11% | Residual connections |
| ROPE | 56 | 11% | Fused RoPE (28×2 for Q,K) |
| MUL | 56 | 11% | q/k norm weight multiply |
| SET_ROWS | 56 | 11% | KV cache scatter updates |
| FLASH_ATTN_EXT | 28 | 5% | One per layer, GQA-aware |
| GLU (swiglu) | 28 | 5% | Fused SiLU-gate per layer |
| Other | 6 | 1% | GET_ROWS, CPY, UNARY, SUB |
| **Dispatches** | **513** | | Actual Metal kernel launches |
| VIEW/RESHAPE | 311 | — | Zero-cost metadata (no Metal dispatch) |
| **Graph nodes** | **824** | | Including zero-cost nodes |

MUL_MAT reduced from 197 → 113 (−43%) by QKV + gate/up projection fusion.
The 311 VIEW/RESHAPE nodes are ggml tensor metadata ops — they don't launch
any Metal compute commands.

## Optimizations applied

All optimizations from the [CUDA performance guide](cuda-performance-optimizations.md)
are active on Metal:

| # | Optimization | Status | Effect |
|---|---|---|---|
| 1 | Graph cache | Enabled by default | 0 ms build on HIT |
| 2 | CUDA graphs | N/A — Metal dispatch overhead is ~0.5 us/node vs CUDA's 3–5 us | Metal graph optimizer handles concurrency |
| 3 | Fused RMSNorm | `swap_rms_norm` | 8 ops/norm → 1 |
| 4 | Fused RoPE | `fuse_rope_in_graph` | 9 ops/Q,K → 1 |
| 5 | GQA strip | `strip_gqa_expand` | Remove expand/repeat |
| 6 | RMSNorm weight fold | `fold_rms_norm_weights` | 57 MUL nodes eliminated |
| 7 | Mask conversion cache | `causal_mask_cache` | Deduplicate across 28 layers |
| 8 | RESHAPE collapse + PERMUTE compose | VIEW handler + `safe_ggml_permute` | Eliminate redundant layout ops |
| 9 | QKV + gate/up projection fusion | `fuse_qkv_projections` + `fuse_gate_up_projections` + CSE | MUL_MAT 197 → 113 (−43%) |

Metal-specific features (from ggml, enabled by default):
- `use fusion = true` — kernel fusion (rms_norm + mul, etc.)
- `use concurrency = true` — concurrent command encoding
- `use graph optimize = true` — node reordering for parallelism

## How to reproduce

### Download

```bash
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download('larryliu0820/Qwen3-0.6B-Q8_0-ExecuTorch-GGML',
                  local_dir='models/qwen3')
"
```

### Build (macOS)

```bash
cmake -B build_native \
    -DEXECUTORCH_GGML_BUILD_LLAMA_RUNNER=ON \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build_native --target benchmark_llm --parallel 16
```

### Run executorch-ggml

```bash
./build_native/benchmark/benchmark_llm \
    models/qwen3/qwen3_q8_0.pte \
    --gguf models/qwen3/Qwen3-0.6B-Q8_0.gguf \
    --n-decode 128 --prompt-len 5
```

### Run llama.cpp baseline

```bash
cd third-party/llama.cpp
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-bench --parallel 16
cd ../..

third-party/llama.cpp/build/bin/llama-bench \
    -m models/qwen3/Qwen3-0.6B-Q8_0.gguf \
    -ngl 99 -p 5 -n 128 -r 5
```

### Profile

```bash
# Per-call timing (build / alloc / compute / output / HIT|MISS)
GGML_PERF_LOG=1 ./build_native/benchmark/benchmark_llm \
    models/qwen3/qwen3_q8_0.pte \
    --gguf models/qwen3/Qwen3-0.6B-Q8_0.gguf \
    --n-decode 32

# Per-op timing (forces Metal sync — inflated absolute times)
GGML_PROFILE=1 ./build_native/benchmark/benchmark_llm \
    models/qwen3/qwen3_q8_0.pte \
    --gguf models/qwen3/Qwen3-0.6B-Q8_0.gguf \
    --n-decode 5
```

### Python

```python
import torch
from executorch_ggml.gguf_module import GGUFModule

module = GGUFModule("models/qwen3/qwen3_q8_0.pte",
                    "models/qwen3/Qwen3-0.6B-Q8_0.gguf")

# Prefill
input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
cache_pos = torch.arange(5, dtype=torch.long)
logits = module.forward(input_ids, cache_pos)
next_token = logits[0][:, -1, :].argmax(dim=-1).item()

# Decode
import time
tokens = [next_token]
t0 = time.time()
for i in range(127):
    out = module.forward(
        torch.tensor([[next_token]], dtype=torch.long),
        torch.tensor([5 + i], dtype=torch.long),
    )
    next_token = out[0][0, 0, :].argmax(dim=-1).item()
    tokens.append(next_token)
print(f"{len(tokens) / (time.time() - t0):.0f} tok/s")
```

## Environment variables

| Variable | Values | Description |
|---|---|---|
| `GGML_PERF_LOG` | `1` | Per-call timing breakdown |
| `GGML_PROFILE` | `1` | Per-op timing (adds sync overhead) |
| `GGML_NO_GRAPH_CACHE` | `1` | Disable graph caching (debug) |
| `GGML_DEBUG_DUMP` | `<path>` | Per-node tensor stats to file |
| `GGML_SKIP_OUTPUT_COPY` | `1` | Skip logits GPU→CPU copy (CUDA only) |
