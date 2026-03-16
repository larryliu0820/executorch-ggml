# build_graph() Architecture

## Overview

The GGML backend rebuilds its entire compute graph on **every `execute()` call**. This mirrors llama.cpp's approach and is necessary because ggml contexts are immutable once allocated — you can't resize tensors or change graph topology in-place. Despite sounding expensive, this costs ~0.007ms per call because the rebuild is almost entirely metadata manipulation with no actual data movement.

## Memory Hierarchy

There are four distinct memory tiers, each with different lifetimes:

### Tier 1: Permanent (init → destroy)

Allocated once in `init()`, never freed until the delegate is destroyed.

| Object | Location | Contents | Size (Voxtral 26L) |
|--------|----------|----------|---------------------|
| `const_buf` | GPU (Metal) | Model weights, RoPE freqs, embedding tables | ~6.5 GB |
| `mutable_buf` | CPU¹ | KV caches (K and V for all layers) | ~13 MB |
| `ir_copy` | CPU | Serialized IR FlatBuffer (graph description) | ~few MB |
| `constant_data` | CPU | Raw weight bytes loaded from NamedDataMap | freed after buf copy |
| `leaf_buf_map` | CPU | Per-tensor `{buffer, offset, nbytes}` index into const/mutable_buf | trivial |

¹ Allocated on CPU backend to avoid scheduler copy-buffer aliasing. On Apple Silicon, CPU `malloc` returns unified memory anyway, so Metal can read it directly.

### Tier 2: Per-shape, preserved across calls

Created once per unique input shape combination. Preserved across `execute()` calls for the same shape.

| Object | Contents |
|--------|----------|
| `GraphInstance::sched` | `ggml_backend_sched` — wraps gallocr with internal buffer pools |
| gallocr internal buffers | Backend memory for intermediate tensors (activations, scratch) |

The scheduler's gallocr tracks buffer allocations internally. On the first `ggml_backend_sched_alloc_graph()` for a shape, it allocates backend memory. On subsequent calls (after `ggml_backend_sched_reset()`), it **reuses** those buffers — no new allocations occur unless the graph grows.

### Tier 3: Rebuilt every execute() (the cheap part)

Freed and reallocated on every `execute()` call via `build_graph()`.

| Object | What it is | Allocation method |
|--------|-----------|-------------------|
| `GraphInstance::ctx` | ggml_context — arena for tensor metadata + graph structure | Single `malloc` of ~14 MB (bump-pointer) |
| `ggml_tensor` structs | ~3665 tensor headers (ne, nb, op, src pointers) | Bump-pointer from ctx arena |
| `ggml_cgraph` | Node array + leaf array (~3058 nodes for 26L decoder) | Bump-pointer from ctx arena |
| Eager constant data | Scalar values, bool masks, causal attention masks | Bump-pointer from ctx arena |

**Why this is fast**: `ggml_init()` does a single `malloc` for the entire context pool. Every `ggml_new_tensor_*()` call is just a pointer bump — no system allocator involvement. Tearing down is a single `ggml_free()` → `free()`. The entire Phase B tensor creation loop is a single O(n_tensors) pass through the IR.

### Tier 4: Ephemeral (within build_graph only)

Created and freed within a single `build_graph()` call.

| Object | Purpose |
|--------|---------|
| `host_buf` | Temporary CPU buffer for leaf tensor data during Phase B. Freed in Phase C step 4 before returning. |

## execute() Call Flow

```
execute(args)
│
├── 1. Gather input shapes from ExecuTorch tensors
│   └── Build current_ne[] vector (4 dims × n_inputs)
│
├── 2. Graph instance selection (dynamic shapes only)
│   ├── Hash current_ne → shape_key
│   ├── Cache hit  → reuse existing GraphInstance (sched preserved)
│   └── Cache miss → create empty GraphInstance (sched=nullptr)
│
├── 3. build_graph()                               ◄── ~0.007ms
│   ├── Free previous ctx (single free())
│   ├── Re-parse IR from ir_copy
│   ├── Phase A: compute ctx_size (shape-aware)
│   ├── ggml_init() — single malloc for arena
│   ├── Create scheduler if first build (per-shape)
│   ├── Phase B: tensor creation + op dispatch
│   │   ├── Leaves: create tensor, point at const_buf/mutable_buf
│   │   ├── Inputs: create tensor, mark as input
│   │   └── Ops: ggml_mul_mat, ggml_add, etc. (builds graph edges)
│   └── Phase C: prepare for scheduler
│       ├── Save eager constants (memcpy data out of arena)
│       ├── Clear data/buffer on non-shared tensors
│       ├── Mark inputs/outputs
│       └── Free host_buf
│
├── 4. ggml_backend_sched_reset()                   ◄── resets gallocr state
├── 5. ggml_backend_sched_alloc_graph()             ◄── reuses gallocr buffers
│
├── 6. Restore shared-buffer leaf pointers
│   ├── const_buf leaves: data = base + offset
│   └── mutable_buf leaves: data = base + offset
│
├── 7. Restore eager constants
│   └── memcpy saved data back into scheduler-allocated tensor memory
│
├── 8. Copy input data (ET tensors → ggml backend tensors)
│   └── Type conversion if needed (F32→BF16, I64→I32, etc.)
│
├── 9. ggml_backend_sched_graph_compute()           ◄── actual GPU work
│
└── 10. Copy output data (ggml backend tensors → ET tensors)
```

## What's Reused vs Reallocated

```
                    REUSED                          REALLOCATED
                    (zero cost)                     (near-zero cost)
              ┌─────────────────┐             ┌──────────────────────┐
              │  const_buf      │             │  ggml_context (arena) │
              │  (6.5 GB GPU)   │             │  (14 MB, 1 malloc)   │
              │                 │             │                      │
              │  mutable_buf    │             │  tensor metadata     │
              │  (13 MB CPU)    │             │  (bump-pointer)      │
              │                 │             │                      │
              │  scheduler      │             │  graph structure     │
              │  (gallocr pool) │             │  (bump-pointer)      │
              │                 │             │                      │
              │  ir_copy        │             │  eager constants     │
              │  (FlatBuffer)   │             │  (small memcpys)     │
              └─────────────────┘             └──────────────────────┘
```

## Performance Breakdown

Measured on Apple M4 Max, Voxtral 26-layer decoder (3665 IR tensors, 3058 graph nodes):

| Step | Time | Notes |
|------|------|-------|
| `build_graph()` total | ~0.007ms | Arena malloc + O(n) tensor creation |
| `sched_reset + alloc` | ~0.001ms | Reuses existing gallocr buffers |
| Input copy | ~0.01ms | BF16 embeddings (1, 1, 3072) for decode |
| **Graph compute** | **~2-5ms** | Actual Metal GPU work |
| Output copy | ~0.01ms | F32 logits readback |

The rebuild is <0.3% of total `execute()` time. The alternative — mutating tensor shapes in-place — would save ~0.007ms but add significant complexity (ggml contexts aren't designed for mutation, and the graph topology can change with dynamic shapes).

## Shape-Keyed Graph Cache

Dynamic-shape models (e.g., prefill seq_len=128 vs decode seq_len=1) produce different graph topologies. The graph cache maps input shapes to dedicated `GraphInstance` objects:

```
graph_cache = {
    hash([3072,4,1,1, 4,1,1,1]) → GraphInstance (prefill, T=4)
    hash([3072,1,1,1, 1,1,1,1]) → GraphInstance (decode,  T=1)
}
```

**The cache does NOT avoid `build_graph()`** — that's called every `execute()` regardless. What the cache preserves is the **scheduler and its gallocr buffer pools**.

### What the cache actually saves

Each `ggml_backend_sched` owns a gallocr that manages GPU memory for intermediate tensors (activations, scratch space). On first use, `sched_alloc_graph()` allocates these buffers from the Metal backend. On subsequent calls for the same shape, `sched_reset()` + `sched_alloc_graph()` reuses the existing buffers — no GPU memory allocation.

Without the cache, every shape transition would cause gallocr buffer churn:

```
execute(prefill T=4)  → allocate GPU buffers for T=4 activations
execute(decode  T=1)  → FREE T=4 buffers, allocate new T=1 buffers
execute(decode  T=1)  → FREE T=1 buffers, allocate new T=1 buffers  (same shape!)
execute(prefill T=8)  → FREE T=1 buffers, allocate new T=8 buffers
```

With the cache, each shape keeps its own scheduler with pre-allocated buffers:

```
execute(prefill T=4)  → cache miss: build sched, alloc GPU buffers (first time)
execute(decode  T=1)  → cache miss: build sched, alloc GPU buffers (first time)
execute(decode  T=1)  → cache HIT:  sched_reset + reuse buffers   (free)
execute(prefill T=8)  → cache miss: build sched, alloc GPU buffers (new shape)
execute(prefill T=4)  → cache HIT:  sched_reset + reuse buffers   (free)
```

For a 26-layer decoder, the gallocr pool holds hundreds of MB of activation memory. Avoiding reallocation on every decode step is significant — GPU buffer allocation involves kernel calls and memory mapping that can take milliseconds, vs the ~0.001ms cost of `sched_reset()` on an existing pool.

### Why per-shape schedulers?

A single shared scheduler would also avoid reallocation for same-shape calls, but it causes a subtler problem: gallocr's internal buffer **splits** encode the graph's tensor lifetime analysis. Different shapes produce different lifetime patterns, so gallocr's split boundaries from shape A would be wrong for shape B, leading to either:
- Incorrect memory reuse (data corruption), or
- Conservative fallback to full reallocation anyway

Per-shape schedulers avoid this entirely — each shape's gallocr has splits tuned to its own graph.

## Why Rebuild Every Call?

1. **ggml contexts are immutable** — tensor shapes are baked at creation time. There's no `ggml_tensor_resize()`.
2. **It's how llama.cpp works** — battle-tested pattern. `llama_decode()` rebuilds the graph every call.
3. **It's negligible** — 0.007ms vs 2-5ms compute. The rebuild is lost in the noise.
4. **It simplifies state management** — no need to track which tensors changed shape, which graph edges are stale, etc.

## Shared Buffer Mechanism

Leaf tensors with `data_key` (weights, KV caches) don't allocate from the scheduler pool. Instead, they point directly into `const_buf` or `mutable_buf`:

```cpp
// During build_graph Phase B:
auto it = handle->leaf_buf_map.find(t->id());
if (it != handle->leaf_buf_map.end()) {
    gt->data   = base + slot.offset;
    gt->buffer = slot.buf;
    shared_leaves.push_back({gt, slot.buf, slot.offset});
}
```

After `sched_alloc_graph()` clears all tensor data pointers (so it can allocate intermediates), the shared leaves are restored:

```cpp
// During execute(), after sched_alloc:
for (auto& sl : active->shared_leaves) {
    sl.tensor->data   = base + sl.offset;
    sl.tensor->buffer = sl.buf;
}
```

This ensures model weights and KV caches are never copied or reallocated — they stay in their permanent buffers while the scheduler manages only intermediate activations.
