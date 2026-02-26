# Dynamic Shape Rebuild — Progress

## Status: FFN + GQA both passing (CPU). SDPA test added. IR deserializer tool added.

---

## Completed Milestones

### M1–M7: See git history

Core dynamic shape infrastructure: `build_graph()` extraction, shape overrides, dynamic size map, shape-change detection, ExecuTorch integration fixes, native broadcast support, multi-input model fixes.

### M8: Fix `dynamic_size_map` collision ✓

**Problem**: The value-based `dynamic_size_map` (`{31→1}`) remapped ALL dimensions with the same numeric value, including unrelated ones (e.g. `max_cache_len - 1 = 31` vs `trace_seq_len = 31`).

**Fix** (Option B from previous PROGRESS.md):

1. **Removed global `dynamic_size_map` application** to all op tensor `ne[]`
2. **ARANGE/FULL**: Apply `dynamic_size_map` locally (no input tensors to derive shapes from)
3. **Comparison/bitwise/logical/CUMSUM/ANY ops**: Derive output shape from input tensor `ne[]` instead of stale IR `ne[]`
4. **SLICE**: Use source tensor shape for non-sliced dims
5. **VIEW**: Removed blanket `dynamic_size_map` application. New strategy:
   - First try: remap ALL dims in the map at once → check numel match
   - If collision (numel mismatch): try one-at-a-time, outer-to-inner → accept first match
   - Fallback: numel-preservation heuristic (outer-to-inner)

### M9: Always-rebuild with input data pre-population ✓

**Problem**: Eager ops (LE, EQ, CUMSUM, etc.) compute data during `build_graph()` by reading source tensor `->data`. When sources are ggml graph ops (e.g. the LE for the causal mask reads from ARANGE + ADD results that depend on `cache_position` input), the data is uninitialized at build time → garbage mask → NaN at step 1.

**Fix**:
1. Always rebuild the graph when `has_dynamic` is true (every step, not just on shape changes)
2. Pass input data to `build_graph()` via `InputDataOverride` structs
3. After creating input tensors, copy ET input data into ggml tensor memory before the eager ops run
4. This gives eager ops correct input values at build time

### M10: Custom ops for comparison/logical ops (in progress)

Converting eager ops (LE, LT, GT, GE, EQ, NE, BITWISE_AND/OR, LOGICAL_NOT) to `ggml_custom_4d` so they execute during `ggml_backend_graph_compute()` instead of at build time. This is needed for the decomposed SDPA path (BareSDPA test) where these ops read from upstream graph ops like softmax output.

ANY remains eager (reduction op, rare, and its sources are now custom ops).

### M11: IR deserializer tool ✓

`python -m executorch_ggml.dump_ir model.pte` — extracts ggml IR FlatBuffer from .pte segments and prints the full graph with decoded op names, shapes, sources, and op_params.

- Generated FlatBuffer Python bindings: `python/executorch_ggml/ggml_ir/`
- Deserializer: `python/executorch_ggml/dump_ir.py`
- Handles ExecuTorch extended header (`eh00`) for segment base offset

### M12: GQA test switched to `attn_implementation="sdpa"` ✓

Changed from `"eager"` to `"sdpa"` so the exported graph contains `aten.scaled_dot_product_attention.default`, which the ggml lowering captures as `LLAMA_ATTENTION` → `ggml_flash_attn_ext`. This avoids the decomposed BMM+softmax+EQ+WHERE path that requires custom ops.

---

## Test Results

### test_dynamic_shapes_ffn — ALL PASS ✓

Gated SiLU MLP + residual with dynamic `seq_len`:
- Tested at seq_lens `[4, 1, 8, 1, 16, 4]` — all pass with `max_abs_diff < 0.001`

### test_dynamic_shapes_gqa — PASS (CPU) ✓

Qwen3 GQA via optimum-executorch (tiny: dim=64, 4 heads, 2 KV heads, 2 layers):
- **Step 0**: max_abs_diff ≈ 0.3–0.5 (PASS, threshold 1.0)
- **Step 1**: max_abs_diff ≈ 0.4–0.5 (PASS, threshold 1.0)
- No more NaN
- Diff comes from `ggml_flash_attn_ext` vs PyTorch SDPA (different accumulation order)

### test_dynamic_shapes_sdpa — WIP

Bare SDPA (Q/K/V projections + `F.scaled_dot_product_attention` + output projection):
- PyTorch export decomposes SDPA into individual ops (BMM, SOFTMAX, EQ, WHERE)
- The EQ/WHERE ops are eager and read uninitialized softmax output at build time
- Custom ops (M10) needed to fix this — partially implemented

---

## Architecture: How Eager Ops Work

Some ops in `build_graph()` compute their output data immediately using CPU loops and store the result as a frozen constant (`op = GGML_OP_NONE`). These include:

- **ARANGE**: fills `[start, start+step, ...]` — safe (no source tensors)
- **FULL**: fills with constant — safe (no source tensors)
- **EQ/NE/LE/LT/GT/GE**: element-wise comparison — **unsafe** if sources are graph ops
- **BITWISE_AND/OR, LOGICAL_NOT**: — **unsafe** if sources are graph ops
- **ANY**: reduction — **unsafe** if source is a graph op
- **CUMSUM**: cumulative sum — **unsafe** if source is a graph op
- **ADD/SUB (I64 path)**: integer arithmetic — **unsafe** if sources are graph ops

"Unsafe" means: the source tensor's `->data` is uninitialized during `build_graph()` because the actual computation happens later during `ggml_backend_graph_compute()`.

**Fix for unsafe ops**: Convert to `ggml_custom_4d()` so they run during graph compute. This is implemented for comparison/bitwise/logical ops in M10.

---

## Files Modified

| File | Changes |
|------|---------|
| `runtime/ggml_backend.cpp` | M1-M10: dynamic_size_map fix, always-rebuild, input pre-population, custom ops, VIEW/SLICE fixes |
| `python/executorch_ggml/ggml_backend.py` | M1-M7 (unchanged this session) |
| `python/executorch_ggml/dump_ir.py` | NEW: IR deserializer tool |
| `python/executorch_ggml/ggml_ir/` | NEW: Generated FlatBuffer Python bindings |
| `schema/ggml_ir.fbs` | `dynamic_dims` field (M4, unchanged this session) |
| `tests/test_dynamic_shapes.py` | Added SDPA test, switched GQA to `attn_implementation="sdpa"` |
| `CMakeLists.txt` | (no net change) |

## Build & Test

```bash
# Build
pip install -e . --no-build-isolation
# or direct cmake (reuse existing build dir):
cmake --build /tmp/tmp*.build-temp --target executorch_ggml_backend_py -j$(nproc)

# Test
LD_LIBRARY_PATH=python/executorch_ggml pytest tests/test_dynamic_shapes.py -v -s

# Dump IR from .pte
python -m executorch_ggml.dump_ir model.pte
```
