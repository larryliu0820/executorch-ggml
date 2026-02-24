# Plan: Dynamic Shape Support — Full C++ Graph Rebuild

## Context

The ggml backend serializes concrete shapes (trace-time hints, e.g. seq_len=127) into the IR. At runtime, a 7-token prompt still computes 127 positions because ggml ops bake output shapes into tensors at graph-creation time (`ggml_mul_mat` reads `a->ne` when creating the result tensor, not at compute time). The only correct fix is rebuilding the ggml graph from the IR when input shapes change.

**Already done** (Python/schema side):
- `schema/ggml_ir.fbs`: Added `dynamic_dims:[bool]` field to Tensor table
- `python/executorch_ggml/serialize.py`: `IrTensor` emits `dynamic_dims`; `_create_bool_vector` helper
- `python/executorch_ggml/ggml_backend.py`: `_detect_dynamic_dims()` checks for SymInt; preprocess emits `dynamic_dims` for runtime inputs
- `runtime/ggml_ir_generated.h`: Regenerated from updated schema
- `runtime/ggml_backend.cpp`: `GgmlDelegateHandle` extended with `ir_copy`, `constant_data`, `input_dynamic_dims`, `last_input_ne`, `has_dynamic`; `init()` populates them

**Remaining**: C++ runtime refactor of `runtime/ggml_backend.cpp` (2300 lines).

## File: `runtime/ggml_backend.cpp`

### Current Architecture

```
init() [lines 285-2166]:
  Phase A: Parse FlatBuffer, calculate ctx size, create ctx     [285-350]
  Phase B: Tensor creation loop — leaf tensors + op switch      [362-1980]
    - Leaf tensor creation (constants + inputs)                 [387-464]
    - Op-building switch (40+ cases)                            [479-1972]
    - Output collection                                         [1977-1980]
  Phase C: Graph build + allocate + backend                     [1982-2105]
  Phase D: Handle creation + dynamic state                      [2107-2166]

execute() [lines 2168-2285]:
  Input copy → deferred casts → graph compute → output copy

destroy() [lines 2287-2301]:
  Free gallocr, backend, context, handle
```

### Target Architecture

```
init():
  1. Parse FlatBuffer, select backend (one-time)
  2. Load constant data from NamedDataMap (one-time)
  3. Build initial graph with serialized shapes (calls build_graph())
  4. Store IR + constants + dynamic metadata in handle

build_graph(handle, input_ne_overrides):   ← NEW extracted function
  1. Create fresh ggml context (sized for max shapes)
  2. Create leaf tensors (constants use stored data, inputs use override shapes)
  3. Run op-building switch (UNCHANGED — just in new function)
  4. Build forward graph
  5. gallocr allocate
  6. Restore constants to backend buffers
  7. Update handle->ctx, handle->graph, handle->inputs, handle->outputs, handle->galloc

execute():
  1. Check if dynamic input shapes changed vs last_input_ne
  2. If changed → call build_graph() with actual shapes
  3. Copy inputs, deferred casts, graph compute, copy outputs (UNCHANGED)

destroy():
  Free everything (UNCHANGED)
```

### Detailed Steps

#### Step 1: Add `build_graph()` static function

**What it does**: Takes a handle (with backend, IR, constants already populated) + optional shape overrides, and (re)builds the ggml context + graph.

**Signature**:
```cpp
// Rebuild the ggml compute graph, optionally overriding input shapes.
// On success, updates handle->ctx, graph, inputs, outputs, deferred_i64_to_i32, galloc.
// Frees any previously existing ctx/galloc.
static Error build_graph(
    GgmlDelegateHandle* handle,
    const int64_t* input_ne_overrides,  // nullptr = use serialized shapes
    size_t n_overrides);                // number of inputs × 4
```

**Body** — move these phases from init() into build_graph():
- Phase A (partial): calculate ctx size, create ctx — **lines 300-350**
- Phase B: full tensor creation loop + op switch — **lines 362-1980**
- Phase C: graph build + allocate — **lines 1982-2105**

**Key modifications inside build_graph()**:
- Parse IR from `handle->ir_copy` (not from `processed` buffer)
- For leaf tensors with `data_key`: load from `handle->constant_data` (not from NamedDataMap)
- For input tensors: if `input_ne_overrides` provided AND the input has `dynamic_dims`, use the override shape for those dims; otherwise use serialized `ne[]`
- Backend is reused from `handle->backend` (not recreated)
- Before building, free old `handle->ctx` and `handle->galloc` if they exist

#### Step 2: Refactor `init()` to call `build_graph()`

New `init()` flow:
```
1. Parse FlatBuffer from processed buffer
2. Select backend (GPU/CPU) — move lines 2057-2087 here (one-time)
3. Load ALL constants from NamedDataMap into handle->constant_data
   Walk IR tensors: for each leaf with data_key, read from ndm, store bytes
4. Copy IR buffer: handle->ir_copy = copy of processed->data()
5. Read dynamic_dims from IR for each input → handle->input_dynamic_dims
6. Set handle->has_dynamic, handle->n_threads
7. Call build_graph(handle, nullptr, 0) for initial graph with serialized shapes
8. Initialize handle->last_input_ne from handle->inputs
9. Return handle
```

This makes init() much shorter (~100 lines instead of ~1900).

#### Step 3: Add shape-change detection + rebuild to `execute()`

Insert at the top of execute(), before the input copy loop:

```cpp
if (handle->has_dynamic) {
    // Build current input shapes from ET tensors
    std::vector<int64_t> current_ne;
    for (size_t i = 0; i < n_inputs; ++i) {
        if (!args[i]->isTensor()) { /* error */ }
        const auto& et = args[i]->toTensor();
        // Convert ET shape (PyTorch order: outermost first) to ggml order (innermost first)
        int ndim = et.dim();
        for (int d = ndim - 1; d >= 0; --d) {
            current_ne.push_back(et.size(d));
        }
        // Pad to 4 dims
        for (int d = ndim; d < 4; ++d) {
            current_ne.push_back(1);
        }
    }

    // Compare with last-seen shapes (only dynamic dims matter)
    bool shapes_changed = false;
    for (size_t i = 0; i < n_inputs && !shapes_changed; ++i) {
        const auto& dd = handle->input_dynamic_dims[i];
        for (size_t d = 0; d < dd.size() && d < 4; ++d) {
            if (dd[d] && current_ne[i*4 + d] != handle->last_input_ne[i*4 + d]) {
                shapes_changed = true;
                break;
            }
        }
    }

    if (shapes_changed) {
        Error err = build_graph(handle, current_ne.data(), current_ne.size());
        if (err != Error::Ok) return err;
        handle->last_input_ne = current_ne;
    }
}
```

The rest of execute() (input copy, compute, output copy) stays the same — it already reads from `handle->inputs` and `handle->outputs` which build_graph() updated.

#### Step 4: Update `destroy()`

No structural changes needed. It already frees ctx, galloc, backend.

### Constant Loading Strategy

**In init()** (one-time): Walk IR tensors. For each leaf with `data_key`:
- Read bytes from `context.get_named_data_map()->get_data(key)`
- Store as `SavedConstant{ir_tensor_id, data_bytes}` in `handle->constant_data`

**In build_graph()** (each rebuild): When creating leaf tensors with `data_key`:
- Find matching entry in `handle->constant_data` by IR tensor ID
- `memcpy(gt->data, saved.data.data(), nbytes)` (into the ggml context buffer, before gallocr clears it)

This avoids touching NamedDataMap after init().

### Input Shape Override Logic in build_graph()

When creating an input leaf tensor (line ~460):
```cpp
int64_t ne[4] = {1, 1, 1, 1};
// Read serialized shape
for (size_t d = 0; d < t->ne()->size() && d < 4; ++d) {
    ne[d] = t->ne()->Get(d);
}
// Override dynamic dims with runtime shapes
if (input_ne_overrides && t->dynamic_dims()) {
    int input_idx = t->input_index();
    for (size_t d = 0; d < t->dynamic_dims()->size() && d < 4; ++d) {
        if (t->dynamic_dims()->Get(d)) {
            ne[d] = input_ne_overrides[input_idx * 4 + d];
        }
    }
}
```

The rest of the op-building switch is unchanged — ggml ops like `ggml_mul_mat(a, b)` auto-compute output shapes from source tensor `ne[]`.

### Ops That Read Shape from `op_params` (Not Auto-Sized)

These ops have compile-time shapes baked into `op_params` and need special handling:

| Op | Issue | Fix in build_graph() |
|----|-------|---------------------|
| **VIEW** (line ~867) | Target `ne[]` in op_params is compile-time | Read actual source numel and recompute view shape. For the common case `[1, seq, hidden]→[1, seq, heads, head_dim]`, only the dynamic dim changes. Use `ggml_reshape_4d(ctx, src, ne0, ne1, ne2, ne3)` with recomputed dims. |
| **ARANGE** (line ~1504) | Output size in ne[] is compile-time | Use input-derived length from the runtime cache_position size |
| **FULL** (line ~1546) | Output size in ne[] is compile-time | Use dynamic dim from corresponding input |
| **SLICE** (line ~956) | start/end in op_params may be compile-time | If end equals serialized dim size, treat as "slice-to-end" (use actual size) |

**Practical approach**: For VIEW, instead of using the serialized `ne[]` from `op_params`, compute `new_ne` by replacing the dynamic dim with the actual runtime value. The `_resolve_shape()` in Python stored the hint; the C++ code substitutes the actual value for dynamic dims.

Alternatively (simpler): skip reading `ne` from `op_params` for VIEW and instead let ggml infer the view shape. Use `ggml_reshape_4d(ctx, src, ne0, ne1, ne2, ne3)` where we compute ne from the source tensor's actual shape + the known static dims.

### Performance: Shape Cache

After the first rebuild, cache the graph. On subsequent calls:
- If `current_ne == last_input_ne` → reuse cached graph (no rebuild)
- If shapes differ → rebuild, update cache

For LLM inference this means: prefill (seq_len=7) rebuilds once, then decode (seq_len=1) rebuilds once. After that, all decode steps reuse the seq_len=1 graph. At most 2 rebuilds per generation.

### Files to Modify

| File | Status | Change |
|------|--------|--------|
| `schema/ggml_ir.fbs` | DONE | `dynamic_dims:[bool]` |
| `runtime/ggml_ir_generated.h` | DONE | Regenerated |
| `python/executorch_ggml/serialize.py` | DONE | `dynamic_dims` in IrTensor + serialization |
| `python/executorch_ggml/ggml_backend.py` | DONE | `_detect_dynamic_dims()` + emit in preprocess |
| **`runtime/ggml_backend.cpp`** | **TODO** | Extract `build_graph()`, refactor `init()`, add rebuild in `execute()` |

---

## Milestones

### Milestone 1: Extract `build_graph()` — Pure Refactor (No Behavior Change)

**Goal**: Move the graph-building code from `init()` into a standalone `build_graph()` function. `init()` calls it. No dynamic shape logic yet — behavior is identical to before.

**Deliverables**:
- `build_graph(handle, nullptr, 0)` function containing phases B+C from init()
- `init()` reduced to: parse IR → select backend → load constants → store IR copy → call `build_graph()` → populate handle
- `destroy()` unchanged

**Test**: Re-export + run existing `test_qwen3_optimum.py::test_qwen3_dynamic_shape_export`. The .pte should serialize identically. Run `llm_main` — output should be the same as before (Flam!!!! garbage, but no crash). This proves the refactor is correct.

**Estimated scope**: ~200 lines of new scaffolding around the existing 1600-line op switch (which moves verbatim).

---

### Milestone 2: Shape Override in `build_graph()` for Input Tensors

**Goal**: `build_graph()` accepts `input_ne_overrides` and applies them to input tensors with `dynamic_dims`. The op switch still runs unchanged — ggml auto-infers downstream shapes.

**Deliverables**:
- `build_graph()` reads `dynamic_dims` from IR for each input
- When `input_ne_overrides` is provided, replaces `ne[d]` for dynamic dims
- Unit test: call `build_graph()` twice with different shapes, verify input tensor `ne[]` differ

**Test**: Write a C++ unit test (or Python-driven test) that:
1. Exports a simple dynamic-shape model (e.g., `linear(x)` with dynamic batch dim)
2. Calls `build_graph()` with seq_len=3, verifies output tensor ne[0]=3
3. Calls `build_graph()` with seq_len=7, verifies output tensor ne[0]=7

**Estimated scope**: ~30 lines in `build_graph()` leaf-tensor creation + test.

---

### Milestone 3: Handle Ops with Baked-In Shapes (VIEW, ARANGE, FULL, SLICE)

**Goal**: Make ops that read shapes from `op_params` adapt to the actual runtime input shapes.

**Deliverables**:
- **VIEW**: Instead of reading `ne` from op_params, compute from source tensor's actual `ne[]` + known static dims. The Python side already stores concrete hint shapes; the C++ side replaces dynamic dims proportionally.
- **ARANGE**: Output shape derived from the dynamic input (cache_position) size.
- **FULL**: Output shape matches the corresponding dynamic tensor.
- **SLICE**: When `end` matches the serialized max dim, treat as "to-end" and use actual dim.

**Test**: Export Qwen3 with dynamic shapes, call `build_graph()` with seq_len=7, inspect VIEW output tensors — they should have shapes proportional to 7 (e.g., `[1, 7, 16, 64]` instead of `[1, 127, 16, 64]`).

**Estimated scope**: ~100 lines of changes in the op switch cases.

---

### Milestone 4: Shape-Change Detection + Rebuild in `execute()`

**Goal**: `execute()` detects when dynamic input shapes change and calls `build_graph()` to rebuild.

**Deliverables**:
- Shape comparison logic at top of `execute()`
- Conversion from ExecuTorch tensor dims (PyTorch order) to ggml ne order
- `build_graph()` called on mismatch; `last_input_ne` updated
- Shape cache: skip rebuild when shapes match

**Test**: Run `llm_main` with the dynamic .pte:
- Prefill (7 tokens): `numel = 7 × 151936` in output
- Decode (1 token): `numel = 1 × 151936` in output
- Full generation produces coherent text

```
GGML_BACKEND_DEVICE=cpu llm_main \
  --model_path qwen3_dynamic_export.pte \
  --tokenizer_path .../tokenizer.json \
  --prompt "The answer to the ultimate question is" \
  --temperature 0 --max_new_tokens 32
```

**Estimated scope**: ~50 lines in `execute()`.

---

### Milestone 5: End-to-End Numerical Accuracy Test

**Goal**: Verify the ggml backend produces numerically accurate output matching PyTorch eager.

**Deliverables**:
- `test_qwen3_numerical.py` passes (cosine similarity > 0.99, argmax match)
- Rebuild ggml CPU libs without AVX-512 (or use the debug build's libs)

**Test**: `pytest tests/test_qwen3_numerical.py -v -s`

**Estimated scope**: Build system fix + test run.

---

### Summary

| Milestone | What | Lines Changed | Risk |
|-----------|------|---------------|------|
| M1 | Extract build_graph() — pure refactor | ~200 new + move 1600 | Low (no behavior change) |
| M2 | Shape overrides for inputs | ~30 | Low (isolated change) |
| M3 | Fix VIEW/ARANGE/FULL/SLICE for dynamic dims | ~100 | Medium (need to handle each op) |
| M4 | execute() shape detection + rebuild | ~50 | Low (well-defined) |
| M5 | Numerical accuracy e2e | Build fix | Low |
