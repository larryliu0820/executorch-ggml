# CUDA Benchmark & Profiling Guide

## Quick Results (Qwen3-0.6B, A100)

| Backend | Model Size | Decode tok/s | Compute ms/tok | Overhead ms/tok |
|---|---|---|---|---|
| llama.cpp (Q8_0 GGUF) | 604 MB | 364 | ~2.7 | ~0 (reused) |
| executorch-ggml (F32) | 2275 MB | **62** | 10.5 | **0** (graph cached) |
| executorch-ggml (Q8_0) | 606 MB | 11 | 9.9 | 46 (rebuild) |

With `GGML_GRAPH_CACHE=1`: F32=67 tok/s, Q8_0=~100 tok/s (0ms overhead).
Without graph cache (default): F32=10.8 tok/s, Q8_0=11.0 tok/s (46-105ms rebuild per call).

Note: Q8_0 MUL_MAT is 9.6x faster than F32 (native quantized kernels work correctly).
The similar tok/s is because attention + element-wise ops (not MUL_MAT) dominate compute.

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
The C++ backend prints `[timing]` lines for the first few decode calls:
```
[timing] build=0.0ms alloc=0.0ms compute=10.5ms total=10.7ms splits=1 nodes=2837
```

With `GGML_PERF_LOG=1`, detailed timing is printed for the first 5 calls
and every 100th call (plus all cache misses):
```
[perf] execute #0: build=47.10ms alloc=2.06ms input=33.13ms compute=1547.17ms output=0.84ms total=1630.30ms | MISS(build+alloc) nodes=2904 splits=1
[perf] execute #1: build=85.05ms alloc=1.89ms input=32.36ms compute=72.25ms output=0.20ms total=191.75ms | MISS(build+alloc) nodes=2837 splits=1
[perf] execute #2: build=0.00ms alloc=0.00ms input=0.00ms compute=10.54ms output=0.16ms total=10.71ms | HIT nodes=2837 splits=1
```

Fields:
- **build**: IR parsing + ggml tensor/graph creation (0 on cache HIT)
- **alloc**: Scheduler buffer allocation (0 on cache HIT)
- **input**: Input data copy to backend tensors
- **compute**: Actual `ggml_backend_sched_graph_compute`
- **output**: Output data copy from backend tensors
- **HIT/MISS**: Graph cache hit (reuse) or miss (rebuild)
- **splits**: Number of CPU↔GPU sync barriers (should be 1)
- **nodes**: Graph size

Init timing is also printed:
```
[perf] init: backends=2.3ms constants=1169.8ms buffers=685.1ms graph=47.4ms total=1904.6ms
```

## 6. Performance Analysis

### Current performance (with graph caching, decode step, Qwen3-0.6B)

| Phase | Time | Notes |
|---|---|---|
| Graph cache lookup | ~0 ms | Hash-based shape key lookup |
| Input copy | ~0 ms | Single token, negligible |
| Compute | 10.5 ms | Actual GPU work (bandwidth-limited) |
| Output copy | 0.15 ms | Logits D→H transfer |
| **Total** | **~10.7 ms** | **~62 tok/s** |

Graph caching eliminates the 105ms rebuild overhead that dominated before.
The remaining time is pure GPU compute, which is bandwidth-limited at
~217 GB/s (A100 theoretical: ~2 TB/s). The F32 model's 2275 MB / 10.5ms
matches llama.cpp's bandwidth utilization (604 MB / 2.7ms = ~224 GB/s).

### Cache behavior
- First call for each unique input shape: **MISS** — full `build_graph()`
  (85ms for Qwen3 decode shape) + scheduler allocation
- Subsequent calls with the same shape: **HIT** — zero overhead, direct compute
- For LLMs: typically 2 shapes (prefill + decode), so after 2 misses,
  all subsequent decode calls are cache hits

### Remaining opportunities for improvement

#### Medium impact
1. ~~**Q8_0 matmul kernels**~~ — **Verified.** Q8_0 `ggml_mul_mat` uses
   native MMVQ kernels (9.6x faster than F32 matmul, 4.5ms vs 43.1ms).
   The bottleneck is now FLASH_ATTN_EXT (15.9ms, 36%) + element-wise ops.

2. **Reduce CONT nodes**. ~156 contiguity-enforcement copies per decode
   step. The `ensure_cont()` helper already checks `ggml_is_contiguous()`,
   so remaining CONTs are genuinely needed after permute/transpose. Could
   be reduced by restructuring ops to avoid unnecessary transposes.

3. **Reduce type conversion overhead**. F32↔F16 ping-pong in mixed-type
   models (FP16 weights + F32 activations). Each conversion adds a CPY
   node. Consider keeping all activations in F32 or all in F16.

#### Lower impact
4. **Pre-warm decode shape during init**. The first decode call triggers
   a MISS (~85ms build + ~72ms CUDA kernel warmup). Could pre-build the
   decode-shape graph during init to eliminate this one-time cost.

5. **Output copy optimization**. The 0.15ms logits D→H copy is small
   but could be eliminated by doing argmax on GPU (requires framework changes).

#### Completed
- ~~Skip graph rebuild for same-shape calls~~ — **Done.** Graph + scheduler
  + buffer assignments are fully reused across calls with the same shape.
  105ms → 0ms overhead. 10.8 → 62 tok/s.
- ~~Persistent scheduler allocation~~ — **Done.** Scheduler stays in
  `is_alloc=true` state between cached calls, skipping reset+alloc entirely.

## 7. Environment Variables

| Variable | Values | Description |
|---|---|---|
| `GGML_BACKEND_DEVICE` | `cpu`, `cuda`, `metal` | Force backend selection |
| `GGML_PERF_LOG` | `1` | Detailed per-call timing breakdown |
| `GGML_NO_GRAPH_CACHE` | `1` | Disable graph caching (for debugging) |
| `GGML_PROFILE` | `1` | Per-op timing (adds sync overhead) |
| `GGML_DEBUG_DUMP` | `<path>` | Per-node tensor stats to file |
| `GGML_NATIVE_CMP_OPS` | `0`, `1` | Override comparison op mode |
