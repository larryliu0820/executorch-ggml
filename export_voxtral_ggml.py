"""Export Voxtral-Mini-4B-Realtime-2602 to ExecuTorch with GGML backend.

Reuses model wrappers from the upstream Voxtral export script and adds
GGML-specific lowering.

Usage:
    python export_voxtral_ggml.py --model-path ~/models/Voxtral-Mini-4B-Realtime-2602
"""

import os

import torch
import torch.nn as nn
from torch.export import Dim, export

from executorch.examples.models.voxtral_realtime.export_voxtral_rt import (
    AudioEncoderExport,
    TextDecoderExport,
    TokenEmbeddingExport,
)
from executorch.examples.models.voxtral_realtime.model import (
    StreamingAudioEncoderExport,
    load_model,
)

from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)
from executorch.exir.passes import MemoryPlanningPass

from executorch_ggml import GgmlPartitioner
from executorch_ggml.passes import BF16UnsafeOpsCastPass, RemoveGraphAssertsPass
from executorch_ggml.passes.replace_copy_ops_pass import ReplaceCopyOpsPass


class MelPreprocessor(nn.Module):
    """Wraps torchaudio MelSpectrogram for portable-op export."""

    def __init__(self, sample_rate=16000, n_fft=400, hop_length=160, n_mels=128):
        super().__init__()
        import torchaudio

        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: (1, N_samples) -> mel: (1, n_mels, T_mel)
        mel = self.mel_spec(waveform)
        mel = torch.clamp(mel, min=1e-10).log10()
        mel = torch.maximum(mel, mel.max() - 8.0)
        mel = (mel + 4.0) / 4.0
        return mel


def export_all_ggml(model, max_seq_len, dtype=torch.float32):
    """Export all Voxtral components as ExportedPrograms."""
    # Replace llama custom ops (update_cache, custom_sdpa) with standard ATen
    # ops (index_copy_, F.scaled_dot_product_attention) to avoid
    # auto_functionalized_v2 wrapping that breaks ExecuTorch delegation.
    from executorch_ggml.modules.voxtral_attention import swap_voxtral_attention
    swap_voxtral_attention(model)

    programs = {}
    # Use target dtype for float tensors, but preserve integer dtypes for samples
    float_dtype = dtype  # BF16 for float sample inputs
    # Integer sample inputs should always use appropriate integer dtypes

    # Preprocessor (MelSpectrogram) is computed in Python at inference time
    # rather than exported into the PTE — torchaudio STFT ops don't have
    # out-variant registrations needed by ExecuTorch's portable runtime.

    # --- Audio encoder ---
    print("\nExporting audio_encoder...")
    audio_encoder = AudioEncoderExport(model)
    audio_encoder.eval()
    max_t_mel = 24000  # 3000 * 8
    sample_mel = torch.randn(
        1, model.config.num_mel_bins, max_t_mel, dtype=float_dtype  # Use target float dtype
    )
    programs["audio_encoder"] = export(
        audio_encoder,
        (sample_mel,),
        dynamic_shapes={"mel": {2: Dim.AUTO}},
        strict=True,
    )
    print(f"  audio_encoder exported (sample input: {sample_mel.shape})")

    # --- Text decoder ---
    print("\nExporting text_decoder...")
    text_decoder = TextDecoderExport(model)
    text_decoder.eval()
    seq_dim = Dim("seq_len", min=1, max=max_seq_len)
    sample_embeds = torch.randn(1, 4, model.config.dim, dtype=float_dtype)
    sample_pos = torch.arange(4, dtype=torch.long)
    programs["text_decoder"] = export(
        text_decoder,
        (sample_embeds, sample_pos),
        dynamic_shapes={
            "input_embeds": {1: seq_dim},
            "cache_position": {0: seq_dim},
        },
        strict=True,
    )
    print(f"  text_decoder exported (sample input: {sample_embeds.shape})")

    # --- Token embedding ---
    print("\nExporting token_embedding...")
    tok_emb = TokenEmbeddingExport(model)
    tok_emb.eval()
    tok_seq_dim = Dim("tok_seq_len", min=1, max=max_seq_len)
    sample_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    programs["token_embedding"] = export(
        tok_emb,
        (sample_ids,),
        dynamic_shapes={"token_ids": {1: tok_seq_dim}},
        strict=True,
    )
    print(f"  token_embedding exported (sample input: {sample_ids.shape})")

    metadata = {
        "sample_rate": 16000,
        "num_mel_bins": model.config.num_mel_bins,
        "hop_length": 160,
        "window_size": 400,
        "downsample_factor": model.config.downsample_factor,
        "dim": model.config.dim,
        "vocab_size": model.config.vocab_size,
        "max_seq_len": max_seq_len,
    }

    return programs, metadata


def export_streaming_ggml(model, max_seq_len, max_enc_len=750, dtype=torch.float32):
    """Export streaming Voxtral components as ExportedPrograms."""
    programs = {}
    param_dtype = dtype
    float_dtype = dtype  # Use target float dtype for sample inputs

    # Preprocessor (MelSpectrogram) computed in Python at inference time.

    # --- Streaming audio encoder ---
    print("\nExporting encode_audio_chunk...")
    streaming_enc = StreamingAudioEncoderExport(model, max_enc_len=max_enc_len)

    # CRITICAL FIX: Only convert float parameters to BF16, preserve integer types
    if param_dtype == torch.bfloat16:
        print("  Converting only float parameters to BF16, preserving integer types...")
        for name, param in streaming_enc.named_parameters():
            if param.dtype.is_floating_point:
                param.data = param.data.to(param_dtype)
                print(f"    {name}: {param.dtype} -> converted to BF16")
            else:
                print(f"    {name}: {param.dtype} -> preserved (integer type)")

        # Also handle buffers (position embeddings, etc.)
        for name, buffer in streaming_enc.named_buffers():
            if buffer.dtype.is_floating_point:
                buffer.data = buffer.data.to(param_dtype)
                print(f"    {name}: {buffer.dtype} -> converted to BF16")
            else:
                print(f"    {name}: {buffer.dtype} -> preserved (integer type)")
    else:
        streaming_enc.to(dtype=param_dtype)

    streaming_enc.eval()
    sample_mel_chunk = torch.randn(
        1, model.config.num_mel_bins, 8, dtype=float_dtype
    )
    sample_enc_pos = torch.arange(4, dtype=torch.long)
    programs["encode_audio_chunk"] = export(
        streaming_enc,
        (sample_mel_chunk, sample_enc_pos),
        dynamic_shapes=None,
        strict=True,
    )
    print(f"  encode_audio_chunk exported (fixed shapes: mel_chunk={sample_mel_chunk.shape})")

    # --- Text decoder ---
    print("\nExporting text_decoder...")
    text_decoder = TextDecoderExport(model)
    text_decoder.eval()
    seq_dim = Dim("seq_len", min=1, max=max_seq_len)
    sample_embeds = torch.randn(1, 4, model.config.dim, dtype=float_dtype)
    sample_pos = torch.arange(4, dtype=torch.long)
    programs["text_decoder"] = export(
        text_decoder,
        (sample_embeds, sample_pos),
        dynamic_shapes={
            "input_embeds": {1: seq_dim},
            "cache_position": {0: seq_dim},
        },
        strict=True,
    )
    print(f"  text_decoder exported (sample input: {sample_embeds.shape})")

    # --- Token embedding ---
    print("\nExporting token_embedding...")
    tok_emb = TokenEmbeddingExport(model)
    tok_emb.eval()
    tok_seq_dim = Dim("tok_seq_len", min=1, max=max_seq_len)
    sample_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    programs["token_embedding"] = export(
        tok_emb,
        (sample_ids,),
        dynamic_shapes={"token_ids": {1: tok_seq_dim}},
        strict=True,
    )
    print(f"  token_embedding exported (sample input: {sample_ids.shape})")

    hop_length = 160
    n_fft = 400
    sample_rate = 16000
    frame_rate = 12.5
    step_samples = int(sample_rate / frame_rate)
    stft_left_overlap = ((n_fft // 2 + hop_length - 1) // hop_length) * hop_length
    mel_skip_frames = stft_left_overlap // hop_length
    chunk_mel_len = 8
    stft_right_lookahead = (
        (chunk_mel_len - 1) * hop_length + n_fft // 2 - chunk_mel_len * hop_length
    )

    metadata = {
        "sample_rate": sample_rate,
        "num_mel_bins": model.config.num_mel_bins,
        "hop_length": hop_length,
        "window_size": n_fft,
        "downsample_factor": model.config.downsample_factor,
        "dim": model.config.dim,
        "enc_dim": model.config.enc_dim,
        "vocab_size": model.config.vocab_size,
        "max_seq_len": max_seq_len,
        "streaming": 1,
        "step_samples": step_samples,
        "chunk_mel_len": chunk_mel_len,
        "max_enc_len": max_enc_len,
        "conv1_pad": 2,
        "conv2_pad": 2,
        "stft_left_overlap": stft_left_overlap,
        "stft_right_lookahead": stft_right_lookahead,
        "mel_skip_frames": mel_skip_frames,
    }

    return programs, metadata


def lower_to_ggml(programs, metadata=None, quant_config=None, target_dtype=None):
    """Lower exported programs to ExecuTorch with GGML backend."""
    print("\nLowering to ExecuTorch with GGML backend...")

    partitioner = {key: [GgmlPartitioner(quant_config=quant_config)] for key in programs}

    constant_methods = {}
    if metadata:
        for key, value in metadata.items():
            constant_methods[key] = value

    # Build transform passes based on target dtype
    transform_passes = [ReplaceCopyOpsPass(), RemoveGraphAssertsPass()]

    # Precision-safe cast pass: protects integer index operations from BF16 corruption
    if target_dtype == "BF16":
        from executorch_ggml.passes.bf16_cast_pass import BF16UnsafeOpsCastPass
        transform_passes.insert(0, BF16UnsafeOpsCastPass())
        print("  Using BF16UnsafeOpsCastPass to protect index operations from corruption")
    elif target_dtype == "FP16":
        print("  FP16 cast pass not yet implemented, using F32")

    et_prog = to_edge_transform_and_lower(
        programs,
        transform_passes=transform_passes,
        partitioner=partitioner,
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=True,
        ),
        constant_methods=constant_methods if constant_methods else None,
    )
    return et_prog.to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
        ),
    )
