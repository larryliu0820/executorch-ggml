# CUDA Performance Optimizations

Decode throughput progression for **Qwen3-0.6B Q8_0 on A100-SXM4-40GB**:

| Configuration | Nodes | ms/tok | tok/s | vs llama.cpp |
|---|---|---|---|---|
| Baseline (rebuild every call) | 2844 | 15.2 | 66 | 0.20x |
| + Graph cache | 2844 | 10.2 | 98 | 0.29x |
| + CUDA graphs | 2844 | 6.6 | 151 | 0.45x |
| + Fused RMSNorm | 2166 | 5.4 | 184 | 0.55x |
| + Fused RoPE (AOT pass) | 1643 | 5.6 | 170 | 0.51x |
| + GQA REPEAT elimination | 1475 | 4.9 | 203 | 0.61x |
| + RMSNorm weight folding | 1418 | 4.8 | 209 | 0.62x |
| + Mask conversion cache | 1310 | 4.3 | 231 | 0.69x |
| **llama.cpp** (reference) | ~700 | ~3.0 | 335 | 1.0x |

## 1. Graph Cache (`GGML_GRAPH_CACHE=1`)

**Problem**: `build_graph()` parses the IR flatbuffer and creates ggml tensors on every
`execute()` call. For Qwen3-0.6B this takes ~2-5ms (build + alloc + input copy), which
is significant relative to the 10ms compute time.

**Solution**: Cache the `GraphInstance` (context, graph, scheduler, tensor mappings) keyed
by input shapes. On subsequent calls with the same shapes, skip `build_graph` entirely and
reuse the cached graph.

**Blocker that was fixed**: Input-derived eager constants (e.g. `float(cache_position[0])`)
had their values baked in at build time. Reusing the graph meant stale position values.

**Fix** (`input_derived` set): A `std::unordered_set<ggml_tensor*>` tracks tensors
transitively derived from graph inputs during Phase B. This set is:

- **Seeded** with input tensors when `t->is_input()` is true
- **Propagated** generically after each op: if any source is in the set, the result is added
- **Carried** through helper functions via `HostDataAccessor::input_derived` pointer
- **Checked** in guards that would otherwise bake in values:
  - CAST handler eager scalar path (skips baking, falls through to `safe_ggml_cast`)
  - `safe_ggml_cast` I32->F32 eager scalar (skips, emits `ggml_cast` graph op instead)
  - MUL handler `ggml_scale` optimization (skips, uses `ggml_mul` instead)

The result: input-derived scalars become graph ops (`ggml_cast(I32, F32)`) rather than
eager constants with frozen data. Phase C sets `has_input_derived_eager` on the
`GraphInstance` to force rebuild if any input-derived eager constants remain.

For Qwen3: `input_derived=no` (zero input-derived eager constants), so graph cache HIT
skips build entirely. Saves ~5.7ms/call.

**Files**: `runtime/ggml_backend.cpp` (HostDataAccessor, Phase B propagation, Phase C flag)

## 2. CUDA Graphs (`GGML_CUDA_GRAPHS=ON`)

**Problem**: Each ggml graph node is a separate CUDA kernel launch. With 2844 nodes,
kernel launch overhead (~3-5us each) adds up to ~8-14ms.

**Solution**: ggml's built-in CUDA graph support captures the entire kernel sequence on
first execution and replays it as a single GPU operation on subsequent calls. This
eliminates per-kernel launch overhead.

**Prerequisite**: Graph cache (above). CUDA graph capture requires stable tensor data
pointers across calls, which graph cache provides by reusing the same allocated graph.

**Build flag**: `cmake -DGGML_CUDA_GRAPHS=ON` (off by default in ggml).

**Files**: `third-party/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu` (ggml upstream)

## 3. Fused RMSNorm (`swap_rms_norm`)

**Problem**: HuggingFace RMSNorm decomposes during `torch.export` into 8 primitive ops
per instance: REPEAT(1.0) + SQR + MEAN + ADD(eps) + SQRT + DIV + MUL(x, rsqrt) + MUL(weight).
Qwen3-0.6B has 113 RMSNorm instances (28 layers x 4 norms + 1 final), producing 904
graph nodes just for normalization.

**Solution**: `swap_rms_norm(model)` replaces HF RMSNorm modules with `FusedRMSNorm`
that calls `torch.ops.aten.rms_norm.default` directly. The partitioner's
`ops_to_not_decompose` prevents re-decomposition during export. The C++ backend maps
`RMS_NORM` to a single `ggml_rms_norm()` call.

**Usage** (before export):
```python
from executorch_ggml.modules.rms_norm import swap_rms_norm
n = swap_rms_norm(model)  # returns count of swapped modules
```

**Result**: 113 instances x 8 ops -> 113 x 2 ops (ggml_rms_norm + ggml_mul).
Nodes: 2844 -> 2166 (-24%).

**Files**: `python/executorch_ggml/modules/rms_norm.py`, `runtime/ggml_backend.cpp` (RMS_NORM handler)

## 4. Fused RoPE (AOT graph pass)

**Problem**: HuggingFace `apply_rotary_pos_emb` + `rotate_half` decomposes into 9 ops
per Q/K: MUL(x, cos) + SLICE + SLICE + NEG + CAT + MUL(rotated, sin) + ADD.
56 instances (28 layers x 2 for Q and K) = 504 nodes.

**Solution**: `FuseRoPEPass` matches the decomposed pattern in the FX graph after
`torch.export` and replaces it with `torch.ops.ggml.rope(x, positions, head_dim,
freq_base, mode=2)`. The C++ backend maps this to a single `ggml_rope_ext()` call.

Applied on the `ExportedProgram` graph BEFORE `to_edge_transform_and_lower`:
```python
from executorch_ggml.passes.fuse_rope_pass import fuse_rope_in_graph
n = fuse_rope_in_graph(ep.graph_module, head_dim=128, freq_base=1e6)
```

The pass pattern-matches `ADD(MUL(x, cos), MUL(CAT(NEG(SLICE(x)), SLICE(x)), sin))`
and traces back from `cos` to find `cache_position`. Safe: if the pattern doesn't match,
the original decomposed ops are kept (no correctness risk).

**Result**: 56 instances x 9 ops -> 56 x 1 op. Nodes: 2166 -> 1643 (-24%).

**Files**: `python/executorch_ggml/passes/fuse_rope_pass.py`, `python/executorch_ggml/rope_op.py`

## 5. GQA REPEAT Elimination

**Problem**: PyTorch's SDPA expands K/V from `num_kv_heads` to `num_attention_heads` via
REPEAT before passing to attention. For Qwen3 (8 KV heads, 16 Q heads), each layer has
2 REPEATs + 4 RESHAPEs = 168 nodes total.

**Solution**: The C++ SDPA handler strips the REPEAT by tracing back through
RESHAPE → REPEAT → RESHAPE and passing the un-expanded K/V directly to
`ggml_flash_attn_ext`, which handles GQA natively (`gqa_ratio = Q.ne[2] / K.ne[2]`).

**Result**: 56 REPEAT + 112 RESHAPE eliminated. Nodes: 1643 -> 1475 (-10%).

**Files**: `runtime/ggml_backend.cpp` (LLAMA_ATTENTION handler, `strip_gqa_repeat` lambda)

## 6. RMSNorm Weight Folding

**Problem**: Each fused `ggml_rms_norm(x, eps)` is followed by `ggml_mul(result, weight)`.
113 separate MUL ops for the weight multiply.

**Solution**: `fold_rms_norm_weights(model)` absorbs the norm weight into the downstream
linear projection weights at export time. For `rms_norm(x) * w → linear(result, W)`,
we set `W_new = W * w` and remove the weight from the norm.

```python
from executorch_ggml.passes.fold_rms_norm_weights import fold_rms_norm_weights
n = fold_rms_norm_weights(model)  # after swap_rms_norm, before export
```

Works for:
- `input_layernorm` → q_proj, k_proj, v_proj (28 layers)
- `post_attention_layernorm` → gate_proj, up_proj (28 layers)
- `final_norm` → lm_head (1)

Does NOT apply to q_norm/k_norm (56) since RoPE follows, not a linear.

**Result**: 57 MUL ops eliminated. Nodes: 1475 -> 1418 (-4%).

**Files**: `python/executorch_ggml/passes/fold_rms_norm_weights.py`

## 7. Mask Conversion Cache

**Problem**: The C++ SDPA handler converts the boolean attention mask to F16 additive
format (SCALE + ADD + CPY = 4 ops). The same source mask tensor is shared across all
28 layers in the IR, but the conversion was done separately for each SDPA call,
producing 28 x 4 = 112 duplicate graph ops.

**Solution**: Cache the converted F16 mask keyed by the source tensor pointer in the
`causal_mask_cache`. On subsequent SDPA calls with the same source mask, reuse the
cached converted tensor.

**Result**: 27 x 4 = 108 duplicate mask conversion ops eliminated.
Nodes: 1418 -> 1310 (-8%).

**Files**: `runtime/ggml_backend.cpp` (LLAMA_ATTENTION handler, mask cache lookup)

## Remaining gap to llama.cpp

We have 1310 nodes vs llama.cpp's ~700. The remaining 610-node difference comes from:

- **View ops** (564 RESHAPE + 140 PERMUTE = 704): Zero-compute layout operations that
  ExecuTorch's IR generates from PyTorch's decomposed attention reshaping. llama.cpp
  builds Q/K/V views directly via the C API. These add CUDA graph overhead but no
  GPU compute.
- **56 q_norm/k_norm weight MULs**: Can't be folded (RoPE follows, not a linear).
- **ExecuTorch Python overhead**: `method.execute()` goes through Python → pybind → C++
  on each call (~0.3ms).

## Full export pipeline

```python
from executorch_ggml.modules.rms_norm import swap_rms_norm
from executorch_ggml.passes.fold_rms_norm_weights import fold_rms_norm_weights
from executorch_ggml.passes.fuse_rope_pass import fuse_rope_in_graph

# 1. Module swaps (before export)
swap_rms_norm(model)
fold_rms_norm_weights(model)

# 2. Export
ep = exportable.export()["model"]

# 3. AOT graph passes (before edge lowering)
fuse_rope_in_graph(ep.graph_module, head_dim, freq_base)

# 4. Edge lowering (C++ handler applies GQA strip + mask cache automatically)
edge_mgr = to_edge_transform_and_lower(ep, ...)
```
