# Eager Ops to Graph Nodes Migration + Graph Reuse

## Motivation

The GGML backend rebuilds the entire compute graph on every `execute()` for dynamic
models. With voxtral's static KV cache, tensor shapes are fixed during decode
(T_q=1, T_kv=max_seq_len). Despite this, `build_graph()` is expensive because eager
ops allocate and fill data from the context pool. The fix has two parts:

1. **Migrate eager ops to graph nodes** — make `build_graph()` purely descriptor
   creation (no data allocation for ARANGE, FULL, CUMSUM, scalar comparisons)
2. **Add graph reuse** — skip `build_graph()` entirely when shapes haven't changed

## Changes

### Files Modified

- `runtime/ggml_backend.cpp` — Eager-to-graph migration + graph reuse logic
- `python/executorch_ggml/ggml_backend.py` — Set `elem_size=0` for migrated ops

### Part 1: Eager Ops to Graph Nodes

#### 1a. ARANGE → `ggml_arange()`

Replaced the eager fill loop with native `ggml_arange(ctx, start, stop, step)`.
Output stays F32 regardless of IR type — ggml doesn't support I32/I64 binary ops,
and all downstream consumers (add, sub, comparisons) work on F32.

```cpp
// Before: eager alloc + fill loop
ggml_set_no_alloc(ctx, false);
gt = ggml_new_tensor_4d(ctx, out_type, ...);
ggml_set_no_alloc(ctx, true);
for (i = 0; i < nelem; i++) data[i] = start + i * step;

// After: graph node
gt = ggml_arange(ctx, (float)start, stop, (float)step);
```

#### 1b. FULL → `ggml_repeat_4d()`

Replaced the eager fill loop with `make_f32_scalar()` + `ggml_repeat_4d()`.
Stays F32 for I32/I64/BOOL targets (same reasoning as ARANGE). Only casts to F16
when the IR type is F16.

```cpp
// Before: eager alloc + fill loop
ggml_set_no_alloc(ctx, false);
gt = ggml_new_tensor_4d(ctx, out_type, ...);
ggml_set_no_alloc(ctx, true);
for (i = 0; i < nelem; i++) data[i] = fill_value;

// After: graph nodes
gt = ggml_repeat_4d(ctx, make_f32_scalar(ctx, fill_value), ne[0], ne[1], ne[2], ne[3]);
```

#### 1c. Scalar Comparison Broadcast → `ggml_repeat_4d()`

For EQ/NE/LE scalar ops, replaced the eager broadcast tensor (alloc + fill loop)
with `ggml_repeat_4d(ctx, make_f32_scalar(ctx, scalar), ...)`.

#### 1d. LLAMA_ATTENTION Causal Mask — KEPT EAGER

**Not migrated.** The `ggml_tri()` + `ggml_log()` approach causes Metal compute hangs
(see Known Issues). The eager F16 triangular mask fill is retained.

#### 1e. Bool→Additive Mask — Removed Eager Path

Deleted the `mask->op == GGML_OP_NONE && mask->data != nullptr` eager conversion
block in LLAMA_ATTENTION. With eager ops migrated to graph nodes, pre-computed
boolean masks no longer exist. The fallback paths (custom op for I32/I64, `ggml_log`
for F32) handle all cases.

#### 1f. CUMSUM → `ggml_custom_4d` Callback

Created `ggml_custom_cumsum` callback function. The CUMSUM op now uses
`ggml_custom_4d()` pinned to CPU. I64 inputs are pre-cast to I32 (ggml has no I64
compute). The ggml axis is stored in `op_params` for the callback.

#### 1g. I64 ADD/SUB — Conditional Eager

Changed from `a->data && b->data` (any data present) to
`a->op == GGML_OP_NONE && a->data != nullptr` (true compile-time constants only).
Non-constant I64 tensors (e.g. from ARANGE graph nodes) are cast to F32 and use
`ggml_add`/`ggml_sub`.

#### 1h. Python `elem_size=0`

Set `elem_size=0` for: FULL (2 sites), ARANGE (2 sites), CUMSUM, EQ scalar,
NE scalar, LE scalar. This tells the C++ ctx_size estimator these ops no longer
allocate eager data.

#### 1i. ctx_size Safety Margin

Kept at `2x + 4MB` (not reduced as originally planned). The LLAMA_ATTENTION causal
mask estimate is retained since the causal mask stays eager.

### Part 2: Graph Reuse

#### 2a. Shape-Change Detection (already existed)

The existing `last_input_ne` comparison in `execute()` already detects shape changes
and only calls `build_graph()` when needed.

#### 2b. Skip Scheduler Reset/Alloc

Added `graph_needs_alloc` flag to `GgmlDelegateHandle`. After `build_graph()` sets
it to `true`, the scheduler reset + alloc runs once, then the flag is cleared.
Subsequent executions with the same shapes skip the expensive `sched_reset` +
`sched_alloc_graph`.

#### 2c. Constant Restoration Optimization

Added `constants_need_restore` flag. After a graph rebuild, constants are fully
restored and the probe cycle resets. When reusing the graph:
- **First call after rebuild**: full restore, set execute_count=1
- **Second call**: probe which constants are corrupted by compute, cache result
- **Steady state**: only restore known-corrupted constants

#### 2d. New Handle Fields

```cpp
struct GgmlDelegateHandle {
  // ... existing fields ...
  bool graph_needs_alloc = true;      // true after build_graph()
  bool constants_need_restore = true;  // true after build_graph()
};
```

## Known Issues

### 1. `ggml_tri` + `ggml_log` Causal Mask Causes Metal Hang

**Symptom**: `ggml_backend_sched_graph_compute` hangs indefinitely on Metal when
the causal mask is built as graph nodes via `ggml_tri()` + `ggml_log()`.

**Root cause**: `ggml_log(0.0)` produces `-inf` in F32, which is then cast to F16.
The resulting `-inf` F16 value appears to cause the Metal flash attention kernel to
hang or loop infinitely. This affects every SDPA layer with `is_causal=True`.

**Workaround**: Causal mask stays as eager F16 fill (Part 1d not migrated).

**Reproduce**:
```bash
# In runtime/ggml_backend.cpp, replace the eager causal mask block with:
#   struct ggml_tensor* ones = ggml_repeat_4d(ctx, make_f32_scalar(ctx, 1.0f), T_kv, T_q, 1, 1);
#   struct ggml_tensor* tri = ggml_tri(ctx, ones, GGML_TRI_TYPE_LOWER_DIAG);
#   struct ggml_tensor* causal = safe_ggml_cast(ctx, ggml_log(ctx, tri), GGML_TYPE_F16);
# Then:
cmake --build build_native
python runner/export_voxtral_rt.py --model-path $MODEL_PATH --dtype BF16
python -c "
import executorch_ggml
from executorch.runtime import Runtime
import torch
rt = Runtime.get()
prog = rt.load_program('./voxtral_ggml/model_bf16.pte')
dec = prog.load_method('text_decoder')
out = dec.execute([torch.randn(1,1,3072,dtype=torch.bfloat16), torch.tensor([0])])[0]
# ^ hangs here
"
```

### 2. Audio Encoder Hangs on Metal with `backend="cuda"` Export

**Symptom**: The `audio_encoder` method hangs during `ggml_backend_sched_graph_compute`
on Metal. The `text_decoder` works fine.

**Root cause**: Unknown. The `CausalWhisperEncoder` has 32 self-attention layers with
`is_causal=True` and no KV cache. The specific op pattern (conv1d → 32×[RoPE + SDPA
+ SwiGLU] → RMSNorm) may trigger a Metal kernel deadlock. This is a pre-existing
issue unrelated to the eager-to-graph migration.

**Workaround**: Run the encoder eagerly in PyTorch, feed its output to the GGML
text_decoder.

**Reproduce**:
```bash
python runner/export_voxtral_rt.py --model-path $MODEL_PATH --dtype BF16
python -c "
import executorch_ggml
from executorch.runtime import Runtime
import torch
rt = Runtime.get()
prog = rt.load_program('./voxtral_ggml/model_bf16.pte')
enc = prog.load_method('audio_encoder')
mel = torch.randn(1, 128, 8, dtype=torch.bfloat16)
out = enc.execute([mel])[0]  # hangs here
"
```

### 3. Subprocess GGML Runs Crash on `im2col` Assert

**Symptom**: When running GGML inference in a Python subprocess, conv1d ops crash
with `GGML_ASSERT(src0->type == GGML_TYPE_F16)` in `ggml_compute_forward_im2col`.

**Root cause**: The subprocess's ggml scheduler routes conv1d to the CPU backend
(instead of Metal). The CPU `im2col` implementation only supports F16 input, but
BF16 input is provided. In the main process, Metal handles conv1d natively.

**Workaround**: Run inference in the main process, not subprocesses.

## Verification

```bash
# Build
cmake --build build_native

# Export (requires model weights)
MODEL_PATH=~/.cache/huggingface/hub/models--mistralai--Voxtral-Mini-4B-Realtime-2602/snapshots/2769294da9567371363522aac9bbcfdd19447add
python runner/export_voxtral_rt.py --model-path $MODEL_PATH --dtype BF16

# Test text_decoder (works)
python -c "
import executorch_ggml
from executorch.runtime import Runtime
import torch, time
rt = Runtime.get()
prog = rt.load_program('./voxtral_ggml/model_bf16.pte')
dec = prog.load_method('text_decoder')
out = dec.execute([torch.randn(1,1,3072,dtype=torch.bfloat16), torch.tensor([0])])[0]
print(f'OK: shape={list(out.shape)}, nan={torch.isnan(out).any().item()}')
"

# Test eager encoder + GGML decoder (end-to-end transcription)
python -c "
import executorch_ggml, torch, os, sys
sys.path.insert(0, '.')
from executorch.runtime import Runtime
from executorch.examples.models.voxtral_realtime.model import load_model
from executorch.examples.models.voxtral_realtime.export_voxtral_rt import AudioEncoderExport

MODEL_PATH = '$MODEL_PATH'
model = load_model(MODEL_PATH, max_seq_len=4096, dtype=torch.bfloat16, backend='cuda')
enc = AudioEncoderExport(model); enc.eval()

import torchaudio, scipy.io.wavfile as wavfile, numpy as np
sr, a = wavfile.read('test_audio.wav')
a = a[:sr*2].astype('float32') / 32768.0
audio = torch.from_numpy(a).float().unsqueeze(0)
mel = torchaudio.transforms.MelSpectrogram(16000, 400, hop_length=160, n_mels=128, power=2.0)(audio)
mel = torch.clamp(mel, min=1e-10).log10(); mel = torch.maximum(mel, mel.max()-8.0); mel = (mel+4)/4
T = mel.shape[2]; mel = torch.nn.functional.pad(mel, (0, ((T+7)//8)*8 - T)).to(torch.bfloat16)

with torch.no_grad(): audio_embeds = enc(mel)
print(f'Encoder: {list(audio_embeds.shape)}')

rt = Runtime.get()
prog = rt.load_program('./voxtral_ggml/model_bf16.pte')
dec = prog.load_method('text_decoder')
T_a = audio_embeds.shape[1]
logits = dec.execute([audio_embeds, torch.arange(T_a)])[0]
print(f'Prefill logits: {list(logits.shape)}, nan={torch.isnan(logits).any().item()}')
"
```
