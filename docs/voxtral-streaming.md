# Voxtral Realtime Streaming on GGML Backend

## Status

**Offline mode**: Working. Correct transcription, 65-84 tok/s Q8_0 on A100.

**Streaming mode**: Export, load, and multi-chunk execution work with GGML delegation. NaN bug fixed — ring buffer mask now computes entirely on the host via eager ops. Next step: end-to-end transcription accuracy validation.

**Key proof point**: Eager (PyTorch) streaming encoder + GGML Q8_0 decoder produces correct transcription ("Mr. Quilter is the apostle of the middle classes..."). This proves the decoder is compatible with streaming encoder embeddings.

## Architecture

Voxtral streaming uses separate `.pte` files:

| File | Method | Description | Status |
|------|--------|-------------|--------|
| `streaming_enc_test.pte` | `e` | Streaming encoder: conv frontend + 32-layer transformer with ring buffer KV cache | Exports + runs multi-chunk. Output valid (NaN fixed). |
| `model_q8_0.pte` | `audio_encoder` | Offline encoder (all mel frames at once) | Working |
| `model_q8_0.pte` | `text_decoder` | 26-layer decoder with KV cache | Working (65-84 tok/s) |
| `model_q8_0.pte` | `token_embedding` | Token embedding lookup | Working |
| `preprocessor.pte` | `forward` | XNNPACK mel spectrogram | Working |

## Bugs Fixed

### 1. INDEX_PUT dependency bug (decoder garbage output)

When `ggml_scatter_axis != 1`, the handler did `gt = dst` which discarded the dependency on `ggml_cpy`. The scheduler could execute reads before writes, corrupting decoder attention.

**Fix**: Replace `ggml_cpy` with transpose + `ggml_set_rows` + transpose back.

### 2. KV valid length formula

`kv_valid_len = max_pos + seq_len` overcounted. Fixed to `max_pos + 1`.

### 3. C++ KV cache auto-slice

The decoder uses full 4096-position KV cache. The C++ SDPA handler auto-slices K/V to the valid range at runtime:
1. `execute()` scans integer inputs for max position
2. Computes `kv_valid_len = max(pos) + 1`
3. SDPA handler creates `ggml_view_4d` to slice K/V + `ensure_cont`

### 4. Negative SLICE index

`mel_chunk[:, :, -2:]` baked `start=-2`. Added `if (start < 0) start += dim_size`.

### 5. select_copy ndim mismatch

`pack_slice_params` defaulted `ndim=4` for a 1D tensor. Fixed by passing `ndim=len(src_shape)`.

### 6. Streaming encoder `copy_()` delegation

`copy_()` on conv state buffers causes `SpecViolationError`. Fixed with `_patch_conv_state_copy()` which replaces `copy_()` with `index_copy_()`.

### 7. Missing partitioner ops

Added to `_SUPPORTED_OP_NAMES`: `rms_norm.default`, `ge.Scalar`, `lt.Scalar`, `gt.Scalar`, `remainder.Scalar`, `div.Tensor_mode`, `select_copy.int`.

### 8. Missing backend preprocess ops

- `remainder.Scalar`: OpCode 84, eager CPU evaluation for I64/I32
- `ge/lt/gt.Scalar`: Materialize scalar from op_params, C++ handlers create F32 repeat tensor
- `select_copy.int`: Recognized alongside `select.int`

### 9. Integer types on CUDA

Ring buffer mask computation uses I64/I32 arithmetic. CUDA binary kernels only support F32/F16/BF16. Added I32/I64 -> F32 eager casts in `safe_ggml_cast`, plus casts in ADD, SUB, DIV, MUL handlers.

### 10. SDPA mask boolean conversion (`was_boolean`)

The mask conversion code applied `mask*65504 - 65504` to ALL F32 masks. For additive masks from WHERE (values `{0, -65504}`), this turns valid positions into masked ones (`0 * 65504 - 65504 = -65504`).

**Fix**: Check `was_boolean = (mask->type == GGML_TYPE_I32 || GGML_TYPE_I64)`. Only apply the scale+offset to boolean masks. F32 additive masks just get cast to F16.

### 11. Named data store collision

Multi-method `.pte` export fails with duplicate `_lifted_tensor_constant` keys.

**Fix**: Prefix auto-generated names with `__ggml_sg{id}_` in preprocess.

### 12. Floor div rounding mode

`aten.div.Tensor_mode(mode='floor')` lowered as plain DIV, losing floor semantics.

**Fix**: Encode `rounding_mode` in `op_params`. C++ DIV handler reads it and applies `ggml_floor`. I64 eager DIV uses Python-style floor division.

### 13. WHERE 0*inf=NaN

Arithmetic emulation `cond*x + (1-cond)*y` produces NaN when `y=-inf` and `cond=1` (`0*-inf=NaN`).

**Fix**: Eager WHERE path when all inputs have host data (with -inf clamping to -65504). Graph path adds `ggml_clamp` on x/y before arithmetic.

### 14. Eager I64 ARANGE

`ggml_arange` only supports F32. Ring buffer positions need I64. Added eager host computation for I64/I32 ARANGE with `ggml_set_output` to prevent scheduler aliasing.

### 15. Eager I64 MUL/DIV

Ring buffer computation chain: `pos * stride`, `pos / window`. I64 MUL and DIV with host data computed eagerly. DIV supports Python floor-division semantics.

### 16. try_eager_f32_binop

General eager F32 binary op for tensors up to 10M elements. Computes on host during graph build, avoiding all CUDA buffer aliasing issues. Supports +, -, *, /, ge, lt, le, gt, and.

### 17. GE/LT/GT scalar variant crash

C++ comparison handlers assumed 2 sources. Scalar variants (ge.Scalar, lt.Scalar, gt.Scalar) only have 1 source with scalar in op_params. Fixed to materialize scalar as F32 repeat tensor.

### 18. REMAINDER scalar variant crash

C++ handler accessed `bc.srcs[1]` unconditionally. Scalar remainder only has 1 source. Fixed bounds check.

### 19. I32→F32 eager cast blocked by `input_derived` guard

Position input (I32) marked `input_derived` caused `safe_ggml_cast` to skip eager path → graph node with no host data → downstream mask ops all fall through to CUDA → NaN. Fix: remove guard; `has_input_derived_eager` forces rebuild.

### 20. Eager F32 binop not retried after type normalization

`try_eager_f32_binop` in ADD/SUB/MUL/DIV ran before I32/I64→F32 casts (→ skip). After casts, both inputs had F32 host data, but code had passed the eager window. Fix: retry `try_eager_f32_binop` after casts.

### 21. Scalar comparisons used REPEAT (no host data)

GE/LT/GT scalar variants wrapped constant in `ggml_repeat_4d` (graph node). `try_eager_f32_binop` couldn't read it. Fix: pass scalar directly; eager broadcasting handles it via modulo indexing.

## Fixed: NaN output from streaming encoder

The ring buffer mask computation chain uses ~15 input-derived IR ops that must be evaluated eagerly on the host. Three bugs prevented this:

1. **I32→F32 eager cast blocked by `input_derived` guard**: The position input (I32) was marked `input_derived`, which caused `safe_ggml_cast` to skip the eager path and create a graph node (no host data). Fix: remove the guard — `has_input_derived_eager` already forces graph rebuilding, so eager input-derived constants are safe.

2. **`try_eager_f32_binop` called before I32/I64→F32 casts**: In ADD/SUB/MUL/DIV, the eager check ran first (seeing I32/I64 types → skip), then the types were cast to F32 with host data, but the code had already passed the eager window. Fix: add a second `try_eager_f32_binop` call after the type normalization.

3. **Scalar comparisons used `ggml_repeat_4d`**: GE/LT/GT scalar variants wrapped the constant in REPEAT (a graph node with no host data), preventing `try_eager_f32_binop` from reading it. Fix: pass the scalar directly — `try_eager_f32_binop` handles broadcasting via modulo indexing.

## Next Steps

- End-to-end streaming transcription accuracy validation (GGML encoder → GGML decoder)
- Performance benchmarking of streaming encoder on A100

## How to Export + Run

```bash
# Build
cmake -B build -G Ninja -DGGML_CUDA=ON -DGGML_CUDA_GRAPHS=ON
cmake --build build --parallel 16

# Export streaming encoder
python3 << 'EOF'
import sys, os; sys.path.insert(0, '.')
import torch
from executorch.examples.models.voxtral_realtime.model import load_model, StreamingAudioEncoderExport
from torch.export import export
from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge_transform_and_lower
from executorch.exir.passes import MemoryPlanningPass
from executorch_ggml import GgmlPartitioner
from executorch_ggml.passes.replace_copy_ops_pass import ReplaceCopyOpsPass
from executorch_ggml.passes import RemoveGraphAssertsPass
from export_voxtral_ggml import _patch_conv_state_copy

model = load_model(os.path.expanduser('~/models/Voxtral-Mini-4B-Realtime-2602'),
                   max_seq_len=4096, dtype=torch.float32)
model.config.use_standard_attention = True
enc = StreamingAudioEncoderExport(model, max_enc_len=750)
_patch_conv_state_copy(enc)
enc.eval()
ep = export(enc, (torch.randn(1, 128, 8), torch.arange(4, dtype=torch.long)), strict=False)
edge = to_edge_transform_and_lower(
    {'e': ep},
    transform_passes=[ReplaceCopyOpsPass(), RemoveGraphAssertsPass()],
    partitioner={'e': [GgmlPartitioner()]},
    compile_config=EdgeCompileConfig(_check_ir_validity=False, _skip_dim_order=True))
et = edge.to_executorch(config=ExecutorchBackendConfig(
    extract_delegate_segments=True,
    memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False)))
with open('streaming_enc_test.pte', 'wb') as f:
    et.write_to_file(f)
EOF

# Test multi-chunk execution
python3 << 'EOF'
import sys, os; sys.path.insert(0, '.')
import torch, executorch_ggml
from executorch.runtime import Runtime
rt = Runtime.get()
prog = rt.load_program('streaming_enc_test.pte')
enc = prog.load_method('e')
for i in range(3):
    chunk = torch.randn(1, 128, 8)
    pos = torch.arange(i*4, (i+1)*4, dtype=torch.long)
    out = enc.execute([chunk, pos])
    print(f'Chunk {i}: {out[0].shape} NaN={torch.isnan(out[0]).any().item()}')
EOF
```

## Files Changed

| File | Changes |
|------|---------|
| `runtime/ops/helpers.h` | `try_eager_f32_binop` for arbitrary-size eager F32 binary ops |
| `runtime/ops/host_data_accessor.h` | Eager I64→F32, I32→F32 casts (including input-derived) in `safe_ggml_cast` |
| `runtime/ops/ops_arithmetic.h` | `try_eager_f32_binop` in ADD/SUB/MUL/DIV (before AND after type casts), I64 eager MUL/DIV, floor div |
| `runtime/ops/ops_comparison.h` | `try_eager_f32_binop` in GE/LT/GT/LE/AND, scalar as direct F32 (not REPEAT) |
| `runtime/ops/ops_creation.h` | Eager I64 ARANGE, eager WHERE with -inf clamping, REMAINDER handler |
| `runtime/ops/ops_shape.h` | Negative SLICE index resolution |
| `runtime/ops/ops_special.h` | `was_boolean` SDPA mask fix |
| `runtime/ggml_backend.cpp` | REMAINDER case in switch, resize_tensor warning fix |
| `python/executorch_ggml/ggml_backend.py` | `__ggml_sg` prefix for lifted tensor constants |
| `python/executorch_ggml/ggml_partitioner.py` | ge/lt/gt.Scalar, remainder.Scalar, select_copy.int, int64 MUL/DIV/remainder |
| `python/executorch_ggml/ops/arithmetic.py` | DIV rounding_mode encoding for floor div |
| `python/executorch_ggml/ops/comparison.py` | ge/lt/gt.Scalar, remainder.Scalar handlers |
| `python/executorch_ggml/ops/shape.py` | select_copy.int alias, ndim fix for select |
| `python/executorch_ggml/serialize.py` | OP_REMAINDER = 84 |
| `python/executorch_ggml/ggml_ir/OpCode.py` | ROPE, REMAINDER enums |
| `schema/ggml_ir.fbs` | REMAINDER = 84 |
| `schema/ggml_ir_generated.h` | REMAINDER enum |
| `export_voxtral_ggml.py` | `_patch_conv_state_copy`, GGML delegation for streaming encoder |
