# CUDA Benchmark & Profiling Guide

## Quick Results (Qwen3-0.6B, A100)

| Backend | Model Size | Decode tok/s | Compute ms/tok | Build ms/tok |
|---|---|---|---|---|
| llama.cpp (Q8_0 GGUF) | 604 MB | 364 | ~2.7 | ~0 (reused) |
| executorch-ggml (F32) | 2275 MB | 10.8 | 10.4 | 105 |
| executorch-ggml (Q8_0) | 606 MB | 11.8 | 24.8 | 108 |

## 1. Build llama.cpp Baseline

```bash
cd third-party/llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build --target llama-bench --parallel 16
```

Download a GGUF model:
```bash
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('Qwen/Qwen3-0.6B-GGUF', 'Qwen3-0.6B-Q8_0.gguf',
                local_dir='/home/dev/models/Qwen3-GGUF')
"
```

Benchmark:
```bash
third-party/llama.cpp/build/bin/llama-bench \
    -m /home/dev/models/Qwen3-GGUF/Qwen3-0.6B-Q8_0.gguf \
    -ngl 99 -p 128 -n 32
```

## 2. Export Qwen3 to executorch-ggml

### F32 Export
```bash
GGML_BACKEND_DEVICE=cuda python3 -c "
import torch
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower, ExecutorchBackendConfig
from executorch_ggml import GgmlPartitioner
from executorch_ggml.passes import RemoveGraphAssertsPass
from executorch_ggml.passes.replace_copy_ops_pass import ReplaceCopyOpsPass
from optimum.exporters.executorch.integrations import CausalLMExportableModule
from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig
from executorch.exir.passes import MemoryPlanningPass

config = AutoConfig.from_pretrained('Qwen/Qwen3-0.6B')
if hasattr(config, 'rope_scaling') and config.rope_scaling is not None:
    config.rope_scaling['type'] = 'default'
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B', device_map='cpu',
    torch_dtype=torch.float32, config=config, attn_implementation='sdpa',
    generation_config=GenerationConfig(use_cache=True, cache_implementation='static',
    max_length=256, cache_config={'batch_size': 1, 'max_cache_len': 256}))

exportable = CausalLMExportableModule(model, max_seq_len=256,
    use_custom_kv_cache=False, use_custom_sdpa=False, disable_dynamic_shapes=False)
ep = exportable.export()['model']

edge_mgr = to_edge_transform_and_lower(ep, partitioner=[GgmlPartitioner()],
    compile_config=EdgeCompileConfig(_check_ir_validity=False, _skip_dim_order=True),
    transform_passes=[ReplaceCopyOpsPass(), RemoveGraphAssertsPass()],
    constant_methods=exportable.metadata)
et = edge_mgr.to_executorch(config=ExecutorchBackendConfig(
    extract_delegate_segments=True,
    memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False)))

with open('/tmp/qwen3_f32.pte', 'wb') as f:
    f.write(et.buffer)
print(f'Saved {len(et.buffer)/(1024*1024):.0f} MB')
"
```

### Q8_0 Export
Same as above but add quantization config to the partitioner:
```python
from executorch_ggml import GgmlQuantConfig, GgmlQuantType
quant_config = GgmlQuantConfig(quant_type=GgmlQuantType.Q8_0)
# ... then pass to partitioner:
partitioner=[GgmlPartitioner(quant_config=quant_config)]
```

## 3. Python Benchmark

```bash
GGML_BACKEND_DEVICE=cuda python3 -c "
import torch, time, executorch_ggml
from executorch.extension.pybindings.portable_lib import _load_for_executorch
from transformers import AutoTokenizer

pte = _load_for_executorch('/tmp/qwen3_f32.pte')
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')
ids = tok.encode('The capital of France is', return_tensors='pt')
seq_len = ids.shape[1]

# Prefill
out = pte.forward((ids, torch.arange(seq_len, dtype=torch.long)))
next_tok = out[0][:, -1, :].argmax(dim=-1).item()
tokens = [next_tok]

# Decode 32 tokens
t0 = time.time()
for i in range(31):
    out = pte.forward((torch.tensor([[next_tok]]), torch.tensor([seq_len + i])))
    next_tok = out[0][:, -1, :].argmax(dim=-1).item()
    tokens.append(next_tok)
dt = time.time() - t0
print(f'{32/dt:.1f} tok/s | {tok.decode(tokens)}')
"
```

## 4. C++ Benchmark

Build (requires ExecuTorch from source):
```bash
cmake -B build_runner \
    -DEXECUTORCH_GGML_BUILD_LLAMA_RUNNER=ON \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=80 \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF
cmake --build build_runner --target benchmark_llm --parallel 16
```

Run:
```bash
GGML_BACKEND_DEVICE=cuda ./build_runner/benchmark/benchmark_llm \
    /tmp/qwen3_f32.pte --n-decode 32 --prompt-len 5
```

## 5. Profiling

### Per-op timing
```bash
GGML_BACKEND_DEVICE=cuda GGML_PROFILE=1 python3 -c "
import torch, executorch_ggml
from executorch.extension.pybindings.portable_lib import _load_for_executorch
pte = _load_for_executorch('/tmp/qwen3_f32.pte')
pte.forward((torch.tensor([[1,2,3,4,5]]), torch.arange(5, dtype=torch.long)))
# Profile a decode step:
pte.forward((torch.tensor([[6]]), torch.tensor([5])))
"
```

**Warning**: `GGML_PROFILE=1` forces per-node CUDA synchronization, adding
massive overhead. Use it to identify which op types are slow, not for
absolute timing.

### Per-node tensor stats
```bash
GGML_BACKEND_DEVICE=cuda GGML_DEBUG_DUMP=/tmp/dump.txt python3 -c "..."
```
Writes per-node mean/std/min/max. Compare CPU vs CUDA dumps to find
divergence.

### Built-in execute timing
The C++ backend prints `[timing]` lines for the first few calls:
```
[timing] build=105ms alloc=1ms compute=10ms total=148ms splits=1 nodes=2837
```
- **build**: IR parsing + ggml tensor/graph creation
- **alloc**: Scheduler buffer allocation
- **compute**: Actual `ggml_backend_sched_graph_compute`
- **splits**: Number of CPU↔GPU sync barriers (should be 1)
- **nodes**: Graph size

## 6. Performance Analysis

### Current bottleneck breakdown (decode step, Qwen3-0.6B)

| Phase | Time | Notes |
|---|---|---|
| Graph rebuild | 105 ms | Re-parse IR, create ggml tensors |
| Sched alloc | 1 ms | Trivial |
| Compute | 10 ms | Actual GPU work |
| **Total** | **~115 ms** | ~93% overhead |

The **graph rebuild** is 10x the actual compute. llama.cpp also rebuilds
the graph every call, but their rebuild is ~0.1ms because they construct
the graph directly via ggml API calls, not by parsing a serialized IR.

### Opportunities for improvement

#### High impact
1. **Skip graph rebuild for same-shape calls** (~10x decode speedup).
   Cache the ggml context + graph when input shapes haven't changed.
   The graph cache infrastructure already exists (`graph_cache` keyed by
   shape). Just skip `build_graph()` when `active->ctx` is valid.
   Expected: 105ms → 0ms overhead → ~95 tok/s decode.

2. **Graph caching with input patching**. Instead of rebuilding from IR,
   reuse the existing graph and only update the input tensor data pointers.
   This is how llama.cpp works — the graph structure is static, only
   input buffers change between calls.

#### Medium impact
3. **Q8_0 matmul kernels**. Currently Q8_0 weights are dequantized to F32
   during matmul. Using ggml's native Q8_0 dot product kernels would give
   ~4x bandwidth improvement. Need to verify the dequant+matmul path
   actually uses quantized kernels (check `ggml_mul_mat` with Q8_0 `a`).

4. **Eliminate unnecessary CONT nodes**. 156 contiguity-enforcement copies
   per decode step (793ms with profiling). Many are after reshape/permute
   views that are already contiguous. Add `ggml_is_contiguous()` checks
   before inserting `ensure_cont()`.

5. **Reduce type conversion overhead**. F32↔F16 ping-pong in mixed-type
   models (FP16 weights + F32 activations). Each conversion adds a CPY
   node. Consider keeping all activations in F32 or all in F16.

#### Lower impact
6. **Batch graph compute**. Currently each ExecuTorch `execute()` call
   rebuilds and computes one graph. Batching multiple decode steps into
   a single graph compute would amortize the rebuild cost.

7. **Persistent scheduler allocation**. The scheduler `reset + alloc`
   cycle is cheap (1-3ms) but could be eliminated entirely by reusing
   the buffer assignments across calls with the same graph shape.

## 7. Environment Variables

| Variable | Values | Description |
|---|---|---|
| `GGML_BACKEND_DEVICE` | `cpu`, `cuda`, `metal` | Force backend selection |
| `GGML_PROFILE` | `1` | Per-op timing (adds sync overhead) |
| `GGML_DEBUG_DUMP` | `<path>` | Per-node tensor stats to file |
| `GGML_NATIVE_CMP_OPS` | `0`, `1` | Override comparison op mode |
