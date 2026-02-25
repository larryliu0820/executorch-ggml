# Dynamic Shape Rebuild — Progress

## Status: Dynamic shapes working end-to-end on FFN. Qwen3 GQA initial build works, rebuild has input mapping issue.

---

## Completed Milestones

### M1: Extract `build_graph()` — Pure Refactor ✓

- `build_graph()` static function at `runtime/ggml_backend.cpp:288`
- Signature: `static Error build_graph(GgmlDelegateHandle* handle, const int64_t* input_ne_overrides, size_t n_overrides)`
- Tears down previous ctx/galloc, parses IR from `handle->ir_copy`, rebuilds graph
- `init()` reduced to: copy IR → create backend → load constants → call `build_graph()` → populate handle
- `GgmlDelegateHandle` extended with `ir_copy` (FlatBuffer copy) and `constant_data` (vector of `SavedConstant`)
- Constants loaded from `NamedDataMap` once in `init()`, then `build_graph()` reads from `handle->constant_data`

### M2: Shape Override for Input Tensors ✓

- 7 lines at `ggml_backend.cpp:427-434`
- After reading `ne[]` from IR, checks `t->is_input() && t->dynamic_dims()`
- Replaces `ne[d]` with `input_ne_overrides[input_idx * 4 + d]` for dynamic dims
- Runs before `n_dims` computation so overridden shape flows through naturally

### M3: Handle Ops with Baked-In Shapes ✓

Three changes:

1. **Dynamic size map** (`ggml_backend.cpp:383-408`):
   - Before tensor loop, scans input tensors with `dynamic_dims`
   - For each dynamic dim where `trace_ne != runtime_ne`, records `{trace_val → runtime_val}`
   - Applied to ALL op tensor `ne[]` (not inputs, not leaves)
   - Universally fixes ARANGE, FULL, comparison ops, etc.

2. **VIEW fix** (`ggml_backend.cpp:940-951`):
   - After reading `new_ne` from `op_params`, applies `dynamic_size_map` to the target shape
   - Then compares `src_numel` vs `view_numel` and infers one mismatched dim (numel-preservation)

3. **SLICE fix** (`ggml_backend.cpp:1052-1054`):
   - Clamps `end` to actual source dim size
   - Recomputes `ne[ax] = end - start`

### M4: Shape-Change Detection in `execute()` ✓

- `GgmlDelegateHandle` extended with:
  - `has_dynamic` (bool)
  - `input_dynamic_dims` (per-input vector of 4 bools)
  - `last_input_ne` (flattened, 4 values per input)
- `init()` step 5: reads `dynamic_dims` from IR for each input, sets `has_dynamic`
- `init()` step 7: initializes `last_input_ne` from initial graph's input tensor `ne[]`
- `execute()` top: converts ET tensor shapes to ggml ne order, compares dynamic dims only, calls `build_graph()` on mismatch

### M5: ExecuTorch Integration Fixes ✓

Five bugs found and fixed to make dynamic shapes work end-to-end:

1. **SymInt placeholder skip** (`ggml_backend.py:296-299`):
   - ExecuTorch passes `sym_size` as a delegate input placeholder with `fake_val` of type `SymInt`
   - Preprocess was creating a dummy `[1,1,1,1]` IR input, corrupting input count and shape overrides
   - Fix: skip non-tensor placeholders (`if fake_val is None or not hasattr(fake_val, "shape"): continue`)

2. **Non-tensor args in execute()** (`ggml_backend.cpp:2364-2380`):
   - With dynamic shapes, ExecuTorch inserts `sym_size` integers in the args span alongside tensor inputs
   - `execute()` assumed all args before outputs were tensors
   - Fix: scan args for tensor-only inputs (skip non-tensors), index outputs from the end of args span

3. **Output tensor resize** (`ggml_backend.cpp:2514-2530`):
   - ExecuTorch pre-allocates output tensors at the upper-bound shape (e.g., seq_len=max)
   - After computing at the actual shape, the output must be resized
   - Fix: call `resize_tensor()` on the ET output tensor using the ggml output's actual shape

4. **VIEW op_params not updated** (`ggml_backend.cpp:974-984`):
   - The `dynamic_size_map` was applied to IR `ne[]` but NOT to the VIEW op's target shape in `op_params`
   - The trace-time seq_len was baked in, causing the numel-preservation heuristic to corrupt wrong dims
   - Fix: apply `dynamic_size_map` to VIEW `new_ne` before the reshape

5. **UNSQUEEZE stale IR ne** (`ggml_backend.cpp:1076-1098`):
   - UNSQUEEZE used the IR `ne[]` (after `dynamic_size_map`) which could have stale values
   - Fix: check if IR ne matches source numel; if not, use source shape directly

### M6: Native Broadcast Support ✓

Removed the requirement for `BroadcastCanonicalizationPass` for MUL/ADD/SUB:

1. **Python preprocess**: Removed shape-match assertions for MUL, ADD, and SUB ops.
   These assertions required all inputs to have matching shapes, forcing the use of
   BroadcastCanonicalizationPass. Now broadcasts are passed through to the C++ backend.

2. **C++ MUL handler** (`ggml_backend.cpp:710-720`):
   - `ggml_mul(a, b)` requires `ggml_can_repeat(b, a)` — b broadcasts to a.
   - Fix: swap operands so the larger tensor is always first.

3. **C++ WHERE handler** (`ggml_backend.cpp:1602-1628`):
   - Bug: called `ggml_repeat(cond, x)` when `ggml_can_repeat(cond, x)` was FALSE (instant crash).
   - Fix: find the largest tensor as target, repeat all smaller operands to it.

4. **C++ REPEAT handler** (`ggml_backend.cpp:722-750`):
   - The "like" tensor (static shape from export) could mismatch the source at runtime.
   - Fix: derive target shape from source's actual shape + expansion ratios from the IR.

5. **expand_copy lowering** (`ggml_backend.py:1210-1265`):
   - Pass through source when shapes are identical (no-op expand).
   - Only emit REPEAT when actual expansion is needed.

---

## Test Results

### test_dynamic_shapes_ffn — ALL PASS ✓

Gated SiLU MLP + residual (LLM feedforward block) with dynamic `seq_len`:
- Ops: 3x linear (MUL_MAT), SiLU, element-wise MUL (gating), ADD (residual), multiple VIEW/RESHAPE
- Lowered to **1 delegate**
- Tested at seq_lens `[4, 1, 8, 1, 16, 4]` — all pass with `max_abs_diff < 0.001`
- Graph rebuilds triggered on every shape change, logged correctly

### test_dynamic_shapes_gqa — Initial build works, rebuild blocked

Qwen3-style GQA attention via optimum-executorch (tiny config: dim=64, 4 heads, 2 KV heads, 2 layers):
- Export with dynamic shapes via `CausalLMExportableModule` (no BroadcastCanonicalizationPass)
- Lowered to **1 delegate** (398KB .pte)
- **Initial build succeeds** — all VIEW, REPEAT, MUL, ADD, SOFTMAX, BMM ops build clean
- **First forward pass succeeds** (step 0: input_ids=[[42]], cache_position=[0])
- **Second forward triggers rebuild → crash in `ggml_set_rows`**:
  - Root cause: input arg mapping is wrong for the 7-input model (input_ids, cache_position × 2, KV cache × 4)
  - The shape-change detection code matches ggml inputs to ET args incorrectly when KV cache tensors shift position
  - See "Current Blocker" below

---

## Current Blocker: Input arg mapping for multi-input models

### Symptom

On the second `execute()` call, the rebuild triggers with wrong input shapes:
```
input[2]: last=(31,1,1,1) cur=(16,32,2,1)
```
Input 2 should be a cache_position (1D, dynamic), but `cur=(16,32,2,1)` is a KV cache tensor shape.

### Root cause

The `execute()` function maps ggml inputs to ET args by scanning non-output args for tensors.
With 7 inputs (input_ids + cache_position variants + KV cache buffers), the 1:1 mapping
assumption breaks — the args may be reordered or include non-tensor `sym_size` values that
shift the mapping.

### Next step

The input mapping needs to account for the fact that the IR input order (from `input_index`)
may not match the ET args order. One approach: use the IR tensor shapes to match ggml inputs
to ET args by finding the best shape match (or use names if available).

---

## Files Modified

| File | Changes |
|------|---------|
| `runtime/ggml_backend.cpp` | M1-M6 implementation (~300 lines added/modified) |
| `runtime/ggml_ir_generated.h` | Regenerated with `flatc --scoped-enums` |
| `python/executorch_ggml/ggml_backend.py` | SymInt skip, broadcast assertion removal, expand_copy passthrough |
| `tests/test_dynamic_shapes.py` | FFN + Qwen3 GQA dynamic shape test |

## Important Notes

- **No BroadcastCanonicalizationPass needed**: MUL/ADD/SUB handle broadcasts natively. The pass was previously required but caused issues with dynamic shapes (SymInt in expand_copy args).
- **Build**: `pip install -e . -v --no-build-isolation`
- **Test**: `LD_LIBRARY_PATH=python/executorch_ggml python tests/test_dynamic_shapes.py`
- **Editable install works now**: `pip install -e .` via `pyproject.toml` + `setup.py`
- **Debug prints**: `init()` has `fprintf(stderr, "[ggml_backend] input[%d]...")` and `execute()` has `"[ggml_backend] Dynamic shapes changed..."` — kept for development.
