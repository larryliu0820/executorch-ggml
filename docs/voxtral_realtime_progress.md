# Voxtral Realtime GGML Backend — Progress

## What Works

### GGML Backend Additions
- **GELU op**: Added across the full stack (schema → serialize → partitioner → backend → runtime). Maps to `ggml_gelu()`.
- **is_causal SDPA**: Python backend reads `is_causal` from SDPA args, packs into `op_params`. C++ runtime builds a lower-triangular F16 causal mask at runtime when `is_causal=true` and no explicit mask is provided.
- **add.Scalar**: New op support — creates scalar constant via NamedDataStore + ADD.
- **le.Scalar**: New op support — comparison with scalar packed in `op_params`, C++ creates filled tensor.
- **Partition completeness**: All three methods (audio_encoder, text_decoder, token_embedding) fully delegated to a single GGML delegate each. Only unavoidable `getitem`/`sym_size` residual ops remain.

### Export Pipeline
- `export_voxtral_ggml.py`: Core export logic using `to_edge_transform_and_lower` + `ReplaceCopyOpsPass` + `RemoveGraphAssertsPass`. Preprocessor computed in Python (torchaudio), not exported to PTE.
- `runner/export_voxtral_rt.py`: CLI entry point with `--dtype BF16|Q8_0`, `--streaming`, `--max-seq-len`.
- BF16 export succeeds: 8.5 GB PTE (4B parameter model).
- Lowering takes ~75 seconds.

### Inference Runner
- `runner/run_voxtral_rt.py`: Supports offline, file-streaming, and microphone modes.
- Mel spectrogram computed via torchaudio in Python.
- Tokenizer loaded from model directory (`tekken.json` via `mistral_common`).

## Blocking Issues

### 1. Context Size Estimation (critical)
The ggml context allocator requires a fixed-size pool at init. The current estimation massively overallocates for models with dynamic shapes because eager ops (EQ, NE, ARANGE, etc.) are estimated at their max-shape sizes.

**Impact**: For `max_seq_len=4096`, the decoder's mask ops need `[4096, 4096]` tensors at build time → estimated ~17 GB context pool. Even with deferred build (skip init-time graph build for dynamic models), the execute-time rebuild uses the same overestimate.

**Solution needed**: Two-pass build or shape-aware estimation at execute time. See `docs/ctx_size_estimation.md`.

### 2. 5D RoPE Intermediates
The encoder and decoder use RoPE with `torch.stack([a, b], dim=-1).flatten(-2)` which creates 5D intermediate tensors `[B, T, H, D/2, 2]`. The GGML backend collapses these into 4D but the `ANY` op's axis mapping was computing `ggml_axis = (ndim-1) - dim = 4` (out of range for ggml's 4D limit).

**Fix applied**: Clamped `ndim` to 4 in the ANY handler. The VIEW handler's >4D→4D collapsing already handles the reshape correctly. The `safe_ggml_permute` wrapper prevents crashes from invalid axes.

**Status**: Fix applied but needs validation with a successful full run.

### 3. Encoder Max Shape
Audio encoder exported with `max_t_mel=24000` → `T_enc=12000`. This makes the serialized IR shapes very large, contributing to the ctx_size overestimation. May need to reduce `max_t_mel` or use `Dim.AUTO` more carefully.

## What's Next

1. **Fix ctx_size estimation** — implement shape-aware estimation using actual input shapes at execute time
2. **Validate inference end-to-end** — get a transcription from test_audio.wav
3. **Q8_0 export** — test quantized export
4. **Streaming mode** — test encode_audio_chunk streaming export and inference
5. **Performance profiling** — measure encode/decode latency on Apple M4 Max
