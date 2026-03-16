# AOT BF16-to-F32 Cast Pass

## Problem

Several ggml CPU kernels crash (`GGML_ABORT`) on BF16 tensors because they only support F32 (or F32/F16). When running the Voxtral BF16 model, this causes runtime aborts for gelu, silu, layer_norm, softmax, pad, and conv (im2col).

### BF16-Unsafe Ops

| ggml op | Supported types | PyTorch ATen target |
|---|---|---|
| gelu | F32, F16 | `aten.gelu.default` |
| silu | F32, F16 | `aten.silu.default` |
| norm (layer_norm) | F32 only | `aten.native_layer_norm.default` |
| soft_max | F32 only | `aten._softmax.default` |
| pad | F32 only | `aten.constant_pad_nd.default` |
| im2col (conv) | F16/F32 | `aten.convolution.default` |

BF16-safe (no cast needed): relu, tanh, sigmoid, clamp, mul_mat, add, mul.

## Solution

`BF16UnsafeOpsCastPass` (`python/executorch_ggml/passes/bf16_cast_pass.py`) walks the exported graph at AOT time and inserts `aten._to_copy` dtype casts around each unsafe op:

1. **Before**: `_to_copy(input, dtype=float32)` for each BF16 tensor arg
2. **After**: `_to_copy(output, dtype=bfloat16)` to restore the graph dtype

This keeps the graph in BF16 everywhere else while giving these ops F32 inputs.

### Key implementation details

- **Edge dialect matching**: Op targets are matched by name string (`"aten.gelu.default" in target_str`) rather than identity comparison, because the Edge dialect wraps ops in `EdgeOpOverload` objects that don't compare equal to `torch.ops.aten.*`.

- **Multi-output ops**: `native_layer_norm` returns a tuple `(output, mean, rstd)`. Only the main output (index 0, accessed via `operator.getitem`) is cast back to BF16. The mean/rstd outputs are always F32.

- **Per-op arg selection**: `convolution` casts both input (arg 0) and weight (arg 1). `native_layer_norm` casts input, weight, and bias (args 0-2). Other ops cast only arg 0.

## Companion fixes

### C++ CAST handler missing BF16 (`runtime/ggml_backend.cpp`)

The CAST op's target-type switch statement had no case for `GGML_TYPE_BF16` (IR enum value 5). BF16 cast targets silently fell through to the F32 default, causing downstream shape/type mismatches. Fixed by adding `case 5: target_type = GGML_TYPE_BF16;`.

### C++ runtime BF16 workaround removal

With the AOT pass handling casts, the runtime workarounds for BF16 inputs to activations, softmax, layer_norm, pad, and conv output cast-back were removed. The conv input casts (F16 kernel / F32 input for im2col) remain — those are ggml format requirements, not BF16-specific.

## Known issues

### `auto_functionalized_v2` wrapping of custom ops

Newer PyTorch/ExecuTorch versions wrap mutating custom ops (e.g. `llama.update_cache`) in `auto_functionalized_v2`. The partitioner recognizes `llama.update_cache.default` by name but not the `auto_functionalized_v2` wrapper, causing `getitem` nodes on unprocessed sources during `preprocess`. Current workaround: export with `use_standard_attention=True`. Proper fix: update the partitioner and backend to unwrap `auto_functionalized_v2` nodes.

### Empty transcription with `use_standard_attention=True`

The model produces empty transcription output when exported with `use_standard_attention=True`. The causal mask (`le(arange(4096), cache_position)`) is correctly computed as a ggml graph node and converted to an F16 additive mask for `flash_attn_ext`, but the model behavior differs from the custom-attention path. Needs investigation.

### Decode speed

Text decoder decode runs at ~0.6 tok/s on M4 Max for the 4B BF16 model. The graph is rebuilt on shape changes (prefill vs decode), but reused for same-shape decode steps. Further optimization: eliminate eager ops (arange, full) by making them proper ggml compute nodes, enabling full graph reuse without any rebuilds.

## Usage

```python
from executorch_ggml.passes import BF16UnsafeOpsCastPass

et_prog = to_edge_transform_and_lower(
    programs,
    transform_passes=[BF16UnsafeOpsCastPass(), ReplaceCopyOpsPass(), RemoveGraphAssertsPass()],
    partitioner=partitioner,
    ...
)
```

The pass should run **before** `ReplaceCopyOpsPass` since it inserts `_to_copy` nodes that the copy pass may optimize.
