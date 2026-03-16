# Two-Graph Architecture: Known Issues

## Issue 1: Dynamic shapes — decode graph corruption when switching back

**Status**: Open — actively debugging
**Affects**: `test_dynamic_shapes_ffn` (seq_len=4 at end, after prefill cycles)
**Baseline**: Also fails (RuntimeError — shape mismatch on graph rebuild). Different error mode.

### Minimal repro

```python
class FFN(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.w1 = nn.Linear(d, h, bias=False)
        self.w2 = nn.Linear(h, d, bias=False)
    def forward(self, x):
        return self.w2(torch.nn.functional.silu(self.w1(x)))

# Export with dynamic seq dim, lower to ggml
# Call 1: x=[1,4,64] → max_diff=0.0  (decode graph, first use)
# Call 2: x=[1,1,64] → max_diff=0.0  (prefill graph, built + used)
# Call 3: x=[1,4,64] → max_diff=0.67 (decode graph, SECOND use — WRONG)
```

### What we've verified

1. **const_buf weights are intact** — sampled first 4 floats of both weight matrices before/after switch. Values are bitwise identical.
2. **Shared leaf pointers are stable** — `tensor->buffer == const_buf` and `tensor->data` offsets match after switch. The scheduler does NOT overwrite pre-set buffer pointers on INPUT-flagged tensors.
3. **Input data is correctly copied** — verified by reading back input tensor data after copy. Values are correct.
4. **Graph structure is intact** — ctx, graph, nodes count (8 for this FFN), inputs (1), outputs (1), shared_leaves (2) all match expected values. No null buffers on any graph node or source tensor.
5. **No eager constants** — this FFN model has zero eager constants (only 2 weights, 1 input, ops are mul_mat and silu).
6. **IR is clean** — exported graph is just: 2 weight placeholders → 2 linears → 1 silu → output.

### What goes wrong

The decode graph's **compute produces wrong output values** on its second use (after a prefill graph cycle). Everything LOOKS correct: weights, input data, graph structure, buffer pointers. But the computed result is numerically wrong (diff ~0.6–4.0).

### Current hypothesis

The ggml scheduler's internal state may not be fully reset between graph switches. When `sched_reset` + `sched_alloc_graph` is called for the decode graph the second time, the scheduler may:
- Assign different backend buffer layouts than the first time
- Have stale internal mapping tables that affect op dispatch
- Route ops to the wrong backend (GPU vs CPU) differently than the first time

Alternatively, there may be a subtle issue with how `ggml_backend_sched` handles graphs with pre-assigned buffers (const_buf) that belong to a different allocation than the scheduler's managed pool. The scheduler might need the pre-assigned buffers to be registered as a "backend buffer type" it knows about.

### Fix options

1. **Separate scheduler per graph** — each GraphInstance gets its own `ggml_backend_sched`. No sharing, no state leaks. Cost: ~2x scheduler overhead.

2. **Rebuild decode graph on switch-back** — always rebuild from IR when switching back to a previously-used graph. Ensures clean scheduler state. Cost: rebuild latency (but graph is cached for same-shape reuse).

3. **Investigate scheduler internals** — dig into `ggml_backend_sched_alloc_graph` to understand exactly what state is preserved across `sched_reset` and whether pre-assigned buffers (const_buf) interact with the scheduler's backend assignment logic.

## Issue 2: Eager constants in build_graph

**Status**: Workaround in place (save/restore in `eager_constants` vector)

### Problem

Several ops compute values eagerly during `build_graph`:
- `make_f32_scalar` — creates F32 scalar tensors in the context pool
- `try_eager_scalar_binop` — computes scalar ADD/MUL/DIV/SUB eagerly
- `eager_i32_to_i64` / `eager_i64_to_i32` — type casts
- `try_eager_scalar_cast` — I32→F32 scalar conversion

These create GGML_OP_NONE tensors with data allocated inline in the context pool. The data must be preserved across scheduler reset/alloc cycles.

### Bug found during implementation

When `const_buf == nullptr` (model has no shared-buffer constants), the check `t->buffer == handle->const_buf` evaluates to `nullptr == nullptr == true`, causing the code to SKIP context-pool-allocated scalars in both the eager constant save loop AND the tensor clearing loop. This caused segfaults in models with eager ops but no shared buffers (e.g. WHERE op test).

Fix: changed to `if (handle->const_buf && t->buffer == handle->const_buf)` — only skip when the buffer pointer is actually set.

### Current workaround

Phase C of `build_graph` saves eager constants into `GraphInstance::eager_constants`. After each `sched_alloc_graph` in execute, the data is restored via `ggml_backend_tensor_set`.

### Desired fix

Eliminate ALL eager computation during build_graph. Instead:
- Scalar constants should be serialized in the IR with `data_key` and loaded into `const_buf`
- Type casts should be done as ggml graph ops (not host-side eager casts)
- This eliminates the `eager_constants` mechanism entirely

## Issue 3: Mutable buffer detection in delegated subgraphs

**Status**: Partially resolved

### Problem

When `GgmlBackend.preprocess` receives the delegated subgraph's `ExportedProgram`, `graph_signature.buffers_to_mutate` is empty and `inputs_to_buffers` is also empty. The partitioner lifts buffer data into the constant store, so the subgraph doesn't know which constants were originally mutable buffers.

### What we found

- The **edge program** (before partitioning) correctly reports `buffers_to_mutate: {'getitem': 'cache'}` and `OutputKind.BUFFER_MUTATION`
- The **delegated subgraph** has all these fields empty — the buffer is lifted as a runtime input
- For the KV cache test model, the `cache` buffer appears as a `placeholder` named `b_cache` but is NOT in `buffer_map`, `param_map`, or `state_dict`

### Current approach

Use `buffers_to_mutate` from the graph signature (populated for non-partitioned case). For the partitioned case, the buffer is a runtime input (no `data_key`), so the scheduler manages it directly and the buffer persists in scheduler memory across calls with the same shape.

The `is_mutable` flag in the IR schema is set based on `mutated_buffer_fqns = set(sig.buffers_to_mutate.values())`. For models that serialize KV caches as constants with `data_key`, this correctly routes them to `mutable_buf`.

## Test status (March 2026)

| Test file | Status | Notes |
|-----------|--------|-------|
| test_ops.py (48 tests) | PASS | All ops |
| test_mask_ops.py (13 tests) | PASS | WHERE, BMM, softmax, trig, pow, sigmoid |
| test_kv_cache.py (2 tests) | PASS | index_put + multi-token generation |
| test_conv_ops.py | PASS | conv1d, conv2d |
| test_update_cache_op.py | PASS | UPDATE_CACHE op |
| test_linear_leaky_relu.py | PASS | |
| test_index_multi.py | PASS | |
| test_parakeet_ops.py | PASS | |
| test_quantize.py | PASS | Q8_0 quantization |
| test_bn_folding_rewrite_pass.py | PASS | |
| test_dynamic_shapes.py | **FAIL** | Issue 1 (decode graph re-entry). Baseline also fails (RuntimeError). |
| test_sym_expr.py::test_run_linear_dynamic | **FAIL** | Pre-existing: tolerance too tight (0.0006 > 0.0001) |
