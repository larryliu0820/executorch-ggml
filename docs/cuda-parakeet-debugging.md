# CUDA Parakeet Debugging: All Issues Fixed

## Status
- **Comparison ops type mismatch**: FIXED (all backends)
- **CUDA depthwise conv graph splits**: FIXED (native conv2d_dw_direct)
- **HostDataAccessor staging buffer reuse**: FIXED (per-call staging)
- **Eager constant memory aliasing**: FIXED (dedicated buffer)
- **CPU transcription**: Working correctly
- **CUDA transcription**: Working correctly, matches CPU output exactly

## Bug 1: Comparison Ops Type Mismatch (FIXED)

### Symptom
On CPU (and Metal), the Parakeet encoder's self-attention produced **all-zero output**.
The softmax attention weights were perfectly uniform (every value = 1/T), meaning
the attention mask was completely broken. Despite this, the model still produced
reasonable transcription because residual connections bypass the zeroed attention.

### Root Cause
The non-native comparison callbacks (`ggml_custom_lt`, `ggml_custom_eq`, etc.)
read tensor data as `int32_t*` regardless of the actual tensor type:

```cpp
const int32_t av = *reinterpret_cast<const int32_t*>(a_base + ao);
const int32_t bv = *reinterpret_cast<const int32_t*>(b_base + bo);
```

When inputs are F32 (e.g., `arange` output), the float bit pattern is
reinterpreted as an integer:
- `float(1.0)` → `int32(0x3F800000)` = `1,065,353,216`
- `float(375.0)` → `int32(0x43BB8000)` = `1,139,998,720`

### Fix
Force `use_native_cmp_ops = true` for ALL backends. The native decompositions
cast inputs to F32 first, avoiding the bug.

## Bug 2: CUDA Depthwise Conv1d Graph Splits (FIXED)

### Symptom
On CUDA, the Conformer encoder's depthwise conv1d (groups=C) used a custom CPU
callback, creating graph splits. The scheduler didn't copy custom op output back
to CUDA, causing downstream ops to read stale GPU memory.

### Root Cause
`ggml_conv_1d` doesn't support groups>1, so a custom CPU callback
(`ggml_custom_conv_1d_dw_f32`) was used. On CUDA, this caused:
1. Graph split at the custom op
2. Missing CPU→CUDA copy of the result

### Fix
Replaced the custom op with native `ggml_conv_2d_dw_direct` by reshaping
1D tensors to 2D with H=1:
```cpp
// weight: [K, 1, C, 1] → [K, 1, 1, C]
// input:  [L, C, B, 1] → [L, 1, C, B]
struct ggml_tensor* w2d = ggml_reshape_4d(ctx, weight, K, 1, 1, C);
struct ggml_tensor* inp2d = ggml_reshape_4d(ctx, ensure_cont(ctx, input), L, 1, C, B);
struct ggml_tensor* conv_out = ggml_conv_2d_dw_direct(ctx, w2d, inp2d, stride, 1, pad, 0, dilation, 1);
// output: [L_out, 1, C, B] → [L_out, C, B, 1]
gt = ggml_reshape_4d(ctx, conv_out, L_out, C, B, 1);
```
`ggml_conv_2d_dw_direct` has a dedicated CUDA kernel in `conv2d-dw.cu`.

## Bug 3: HostDataAccessor Staging Buffer Reuse (FIXED)

### Symptom
CUDA encoder output had wrong batch norm parameters. The first depthwise conv
layer's MUL node (batch norm scale) produced wildly wrong output despite the
conv output being correct. Mean of scale was 3.23 on CUDA vs 1.78 on CPU.

### Root Cause
`HostDataAccessor` used a single `std::vector<uint8_t> staging_` buffer for all
GPU→host tensor copies. The BATCH_NORM handler called `get()` four times:

```cpp
const float* w_data = (const float*)host_acc.get(weight);  // staging_ = weight
const float* b_data = (const float*)host_acc.get(bias);    // staging_ = bias (w_data STALE)
const float* m_data = (const float*)host_acc.get(mean);    // staging_ = mean (b_data STALE)
const float* v_data = (const float*)host_acc.get(var);     // staging_ = var  (m_data STALE)
```

On CPU, `get()` returns direct pointers (no staging), so all 4 are independent.
On CUDA, all 4 point to the same staging buffer containing only var data.
This produced: `scale = var/sqrt(var+eps)` instead of `gamma/sqrt(var+eps)`.

### Fix
Changed `staging_` from `std::vector<uint8_t>` to `std::deque<std::vector<uint8_t>>`.
Each `get()` call now appends a new buffer, keeping all previous pointers valid.

## Bug 4: Eager Constant Memory Aliasing (FIXED)

### Symptom
Even after fixing Bug 3, the scheduler's gallocr allocated intermediate tensor
outputs overlapping with eager constant memory (batch norm params, masks),
overwriting constant data during execution.

### Root Cause
Eager constants (leaf tensors without `data_key`, computed during `build_graph`)
were marked with `ggml_set_input()` but still allocated in the scheduler's pool.
The gallocr didn't properly prevent reuse of their memory. Specifically, a large
mel-processing tensor (node 12, 128x3001) was assigned to memory overlapping
with a 1x1024 batch norm weight.

### Fix
Eager constants now get their own dedicated `eager_const_buf` allocated on the
primary backend BEFORE `ggml_backend_sched_alloc_graph()`. With `data` and
`buffer` already set, the scheduler treats them like shared leaves (pre-allocated
externally) and won't overlap their memory.

Shared leaf restore was also moved before `sched_alloc_graph` for the same reason.

## Debug Method
The `GGML_DEBUG_DUMP=<path>` env var writes per-node tensor statistics
(mean, std, min, max) during graph execution. Comparing CPU vs CUDA dumps
revealed each divergence point.

## Files Modified
- `runtime/ggml_backend.cpp`:
  - `HostDataAccessor`: `staging_` → `std::deque` for per-call buffers
  - CONV_1D_DW: native `ggml_conv_2d_dw_direct` instead of custom callback
  - `GraphInstance`: added `eager_const_buf` for isolated eager constant storage
  - Execute path: shared leaves + eager constants restored before `sched_alloc_graph`
