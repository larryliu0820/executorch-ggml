# KV Cache Strategies for SDPA

Two strategies for handling the KV cache in `ggml_flash_attn_ext`, selectable at runtime via `GGML_KV_SLICE=1`.

## Background

During LLM decode, the KV cache grows by one position per step. The SDPA (flash attention) kernel must only attend to **valid** positions — uninitialized cache entries contain garbage/zeros that corrupt attention weights.

The challenge: `ggml_flash_attn_ext`'s CUDA kernel grid dimensions depend on `K->ne[1]` (number of K positions). Changing `ne[1]` every step breaks CUDA graph replay, adding ~4ms overhead per step.

## Default: Dynamic Mask (CUDA-graph-friendly)

K/V tensors always use the full cache size (`ne[1] = max_seq_len`). A per-call F16 mask blocks uninitialized positions:

```
K ne[1] = 256 (fixed)      Mask [256, 1]:
                            pos 0..5  → 0.0    (attend)
                            pos 6..255 → -inf   (block)
```

- Tensor shapes never change → CUDA graph replays every step
- Mask data (~256 bytes for decode) uploaded via `ggml_backend_tensor_set`
- CUDA graphs natively pick up updated buffer contents on replay
- **433 tok/s** on A100 (Qwen3-0.6B Q8_0, 128 decode tokens)

Best for: small-to-medium caches where attention compute over masked positions is negligible.

## GGML_KV_SLICE=1: Auto-Slice with 256-Aligned Padding

K/V views are sliced to `GGML_PAD(kv_valid_len, 256)` — the nearest multiple of 256 (matching llama.cpp's `GGML_PAD(n_ctx_seq, 256)`). A dynamic mask handles positions within each 256-block:

```
Step   kv_valid_len  padded_kv   K ne[1]   Graph action
-----  ------------  ---------   -------   ------------
  1         1          256        256      BUILD (first time)
  2         2          256        256      CUDA graph REPLAY
  ...
255       255          256        256      CUDA graph REPLAY
256       256          512        512      REBUILD (boundary crossing)
257       257          512        512      CUDA graph REPLAY
  ...
511       511          512        512      CUDA graph REPLAY
512       512          768        768      REBUILD
```

- CUDA graphs replay within each 256-step window (255 out of 256 steps)
- Graph rebuilds only at boundary crossings (~4ms, amortized to 0.016ms/step)
- Flash attention processes fewer positions than full-cache approach
- **403 tok/s** on A100 (Qwen3-0.6B Q8_0, 128 decode tokens)

Best for: large caches (2K+) where attention compute savings outweigh the per-boundary rebuild cost.

## Performance Comparison

Qwen3-0.6B Q8_0 on NVIDIA A100, 128 decode tokens, max_seq_len=128 (padded to 256):

| Mode | tok/s | vs llama.cpp | CUDA graphs |
|------|------:|:-------------|:------------|
| Dynamic mask (default) | **433** | **114%** | Full replay every step |
| Auto-slice GGML_KV_SLICE=1 | **403** | **106%** | Replay within 256-step windows |
| llama.cpp (baseline) | 379 | 100% | Rebuilds graph each step |

Without CUDA graphs (GGML_CUDA_DISABLE_GRAPHS=1):

| Mode | tok/s | Notes |
|------|------:|:------|
| Dynamic mask | 342 | Mask upload overhead (~0.3ms/step) |
| Auto-slice | 367 | Zero per-step overhead (just ne[] patch) |

Auto-slice is faster without CUDA graphs because it has no per-step mask upload. Dynamic mask wins with CUDA graphs because shapes are perfectly fixed.

## Why 256?

The number comes from `FATTN_KQ_STRIDE` in ggml's flash attention CUDA kernel (`fattn-common.cuh`). The kernel processes K positions in tiles of 256 elements. Aligning to this boundary:

1. **Enables `KV_max` optimization**: When `K->ne[1] % 256 == 0`, the kernel scans the mask to find the last valid 256-block per tile and skips fully-masked blocks entirely (early exit in the inner loop)
2. **Matches llama.cpp**: `llama-context.cpp:180` pads `n_ctx_seq = GGML_PAD(n_ctx_seq, 256)`, and `llama-kv-cache.cpp:1009` pads runtime `n_kv` to `max(n_pad, 256)`
3. **CUDA graph alignment**: Kernel grid dimensions derived from `K->ne[1]` stay constant within each 256-step window

## Scaling Analysis for Large Caches

For a 4096-position cache at step 100:

| Mode | Positions processed | Wasted compute |
|------|--------------------:|:---------------|
| Dynamic mask | 4096 | 97.6% masked |
| Auto-slice (padded) | 256 | 60.9% masked |

At step 2000:

| Mode | Positions processed | Wasted compute |
|------|--------------------:|:---------------|
| Dynamic mask | 4096 | 51.2% masked |
| Auto-slice (padded) | 2048 | 2.4% masked |

Auto-slice's advantage grows with cache size and diminishes as the cache fills.

## Implementation Details

### Build time (`ops_special.h`, `build_op_llama_attention`)

The SDPA handler detects KV cache reads (`k_is_mutable`) and computes `kv_valid_len` from the `cache_position` input:

- **Default**: creates a full-cache F16 mask `[T_kv, T_q]`, registers it as `PatchableKVView` for per-call value updates
- **GGML_KV_SLICE=1**: creates K/V `ggml_view_4d` sliced to `padded_kv = GGML_PAD(kv_valid_len, 256)`, plus a mask `[padded_kv, T_q]` for positions within the block

### Execute time (`ggml_backend.cpp`)

On each forward call:

1. Read `cache_position` from I32 inputs (no GPU sync — reads from ExecuTorch CPU tensor)
2. **Default**: rewrite mask values for the current position, upload to GPU
3. **GGML_KV_SLICE=1**: check if `GGML_PAD(kv_valid_len, 256)` crossed a boundary. If yes: force graph rebuild. If no: rewrite mask values within the current block

### Graph dependency

The IR preserves the data dependency from INDEX_PUT (KV write) to SDPA (KV read) through `id_to_tensor` mapping. INDEX_PUT uses `ggml_set_rows` which creates a view of the cache leaf in `mutable_buf` — prior positions are preserved because `ggml_set_rows` only writes the scattered rows.

### Why not pure auto-slice (no mask)?

The model's IR has `is_causal=false` (PyTorch exports SDPA with explicit mask, not the is_causal flag). Dropping the mask removes all causal masking. The `flash_attn_ext` CUDA kernel also requires contiguous masks (`nb[1] == ne[0] * sizeof(F16)`), so a mask sized to the exact `kv_valid_len` (which changes every step) can't be graph-cached.
