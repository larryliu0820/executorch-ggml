# ggml Context Size Estimation Problem

## Background

`ggml_init()` requires a fixed-size memory pool (`ctx_size`) upfront. This pool holds:
1. **Tensor metadata** — `ggml_tensor` structs (~368 bytes each)
2. **Graph structure** — `ggml_cgraph` with node/edge arrays
3. **Eager data** — data for tensors computed at graph-build time (not deferred to compute)

ggml uses an arena allocator with no dynamic growth. If the pool runs out during tensor creation, `ggml_new_object` aborts.

## The Problem

### Which ops allocate eagerly?

Certain ops create data-filled tensors during `build_graph()` (before `ggml_graph_compute`):
- `EQ`/`NE` with scalar: creates a full-size tensor filled with the scalar value
- `ARANGE`: creates a filled range tensor
- `FULL`: creates a filled constant tensor
- `CUMSUM`: computes cumulative sum eagerly
- `ANY`: reduces and fills result tensor
- `INDEX_PUT`: copies data eagerly
- `LE`/`LT`/`GT`/`GE` with scalar: same pattern as EQ
- `ADD`/`SUB` on I64 tensors: eager element-wise computation
- `CAST` to/from I64: eager conversion
- `LLAMA_ATTENTION` with `is_causal`: builds an F16 causal mask

### Why max-shape estimates are wrong

The serialized IR stores **max shapes** (from export-time dynamic shape bounds). For a model with `max_seq_len=4096`:
- EQ mask ops have shape `[4096, 4096]` = 16M elements × 8 bytes = 128 MB each
- With 26 decoder layers, each potentially creating multiple mask ops, the estimate explodes to tens of GB

At **runtime**, actual shapes are much smaller (e.g., `[10, 10]` for a 10-token sequence), so actual memory usage is tiny.

### The chicken-and-egg problem

- `ggml_init()` happens **before** we know actual input shapes
- `build_graph()` allocates eager tensors at max shape during `init()`
- Even with deferred build (skip init-time build), the ctx_size estimation in `build_graph()` still uses max shapes from the serialized IR

### Why just overallocating doesn't work

- For `max_seq_len=4096`: the EQ/NE ops alone need ~17 GB at max shape
- macOS can handle this via virtual memory (no physical RAM needed for unused pages), but `ggml_init` pre-allocates the full pool
- On memory-constrained devices (mobile), this is unacceptable

## Potential Solutions

### 1. Two-pass build (recommended)
First pass: walk the IR and count tensors without allocating data. Use actual input shapes (available at execute time). Second pass: allocate with accurate size.

### 2. Dynamic context growth
Patch ggml to support context pool growth (realloc). This is invasive but eliminates the estimation problem.

### 3. Separate eager allocation
Don't allocate eager data from the ggml context pool. Instead, use a separate heap allocation for each eager tensor's data. This decouples metadata size from data size.

### 4. Lazy eager evaluation
Instead of computing EQ/NE/etc. at build time, encode them as graph ops and let `ggml_graph_compute` handle them. This moves the allocation to the backend scheduler, which uses its own allocator.

### 5. Shape-aware estimation at execute time
When `build_graph()` is called during `execute()` with actual shapes, compute ctx_size from those shapes instead of the serialized max shapes. This requires parsing the IR twice (once for size estimation, once for building).

## Current Status

The current code uses option 5 partially (deferred build for dynamic models) but still estimates from max shapes. The estimation caps per-op at 512K elements, which underestimates for large models.
