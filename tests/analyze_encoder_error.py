"""Analyze encoder error sources: F16 weight precision, output distributions, etc.

Usage:
    source .venv/bin/activate
    python3 tests/analyze_encoder_error.py
"""

import os
import time

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.export import Dim, export

from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig
from executorch.exir.passes import MemoryPlanningPass
from executorch.extension.pybindings.portable_lib import (
    _load_for_executorch_from_buffer,
)

from executorch_ggml import GgmlPartitioner, to_edge_rewrite_and_lower
from executorch_ggml.passes.replace_copy_ops_pass import ReplaceCopyOpsPass


def load_audio(audio_path: str, sample_rate: int = 16000) -> torch.Tensor:
    data, sr = sf.read(audio_path)
    waveform = torch.from_numpy(np.array(data, dtype=np.float32))
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    return waveform


def load_model():
    import nemo.collections.asr as nemo_asr
    model = nemo_asr.models.ASRModel.from_pretrained(
        "nvidia/parakeet-tdt-0.6b-v3", map_location="cpu"
    )
    model.eval()
    model.cpu()
    return model


# ---- Part 1: F16 weight analysis (no export, instant) ----

def analyze_f16_weights(encoder):
    """Check F16 round-trip error for every parameter in every layer."""
    print("=" * 80)
    print("PART 1: F16 WEIGHT PRECISION PER LAYER")
    print("=" * 80)
    print(f"\n{'Layer':<10} {'Param':<45} {'Shape':<20} {'F16 max_err':>12} {'F16 mean_err':>12} {'Weight mag':>12}")
    print("-" * 115)

    layer_summary = {}

    for layer_idx, layer in enumerate(encoder.layers):
        worst_err = 0.0
        worst_name = ""
        total_f16_err = 0.0
        total_params = 0

        for name, param in layer.named_parameters():
            w = param.data.float()
            w_f16 = w.half().float()
            err = (w - w_f16).abs()
            max_err = err.max().item()
            mean_err = err.mean().item()
            mag = w.abs().max().item()

            if max_err > worst_err:
                worst_err = max_err
                worst_name = name

            total_f16_err += err.sum().item()
            total_params += param.numel()

            # Only print params with significant error
            if max_err > 0.01:
                print(f"{layer_idx:<10} {name:<45} {str(list(param.shape)):<20} {max_err:>12.6f} {mean_err:>12.8f} {mag:>12.4f}")

        avg_err = total_f16_err / total_params if total_params > 0 else 0
        layer_summary[layer_idx] = (worst_err, worst_name, avg_err)

    print(f"\n{'Layer':<8} {'Worst F16 err':>14} {'Avg F16 err':>14} {'Worst param'}")
    print("-" * 80)
    for layer_idx in sorted(layer_summary.keys()):
        worst_err, worst_name, avg_err = layer_summary[layer_idx]
        print(f"{layer_idx:<8} {worst_err:>14.6f} {avg_err:>14.8f} {worst_name}")


# ---- Part 2: Eager intermediate analysis ----

def analyze_eager_intermediates(encoder, mel, length):
    """Track tensor magnitudes and statistics through the encoder layer by layer."""
    print("\n" + "=" * 80)
    print("PART 2: EAGER INTERMEDIATE TENSOR STATISTICS")
    print("=" * 80)

    with torch.no_grad():
        x = mel.transpose(1, 2)
        x, length = encoder.pre_encode(x=x, lengths=length)
        length = length.to(torch.int64)
        x, pos_emb = encoder.pos_enc(x=x, cache_len=0)

        max_audio_length = x.size(1)
        pad_mask, att_mask = encoder._create_masks(
            att_context_size=encoder.att_context_size,
            padding_length=length,
            max_audio_length=max_audio_length,
            offset=None,
            device=x.device,
        )

    print(f"\nAfter subsampling+pos_enc: shape={list(x.shape)}")
    print(f"  x: mean={x.mean():.4f} std={x.std():.4f} min={x.min():.4f} max={x.max():.4f}")
    print(f"  pos_emb: mean={pos_emb.mean():.4f} std={pos_emb.std():.4f}")

    print(f"\n{'Layer':<8} {'x_mean':>10} {'x_std':>10} {'x_min':>10} {'x_max':>10} {'delta_norm':>12} {'has_nan':>8}")
    print("-" * 80)

    with torch.no_grad():
        for i, layer in enumerate(encoder.layers):
            x_prev = x.clone()
            x = layer(x=x, att_mask=att_mask, pos_emb=pos_emb, pad_mask=pad_mask)
            delta = (x - x_prev).norm().item()
            has_nan = torch.isnan(x).any().item()
            print(f"{i:<8} {x.mean().item():>10.4f} {x.std().item():>10.4f} "
                  f"{x.min().item():>10.4f} {x.max().item():>10.4f} "
                  f"{delta:>12.4f} {'YES' if has_nan else 'no':>8}")


# ---- Part 3: Enhanced prefix scan with error distribution ----

class EncoderPrefix(nn.Module):
    def __init__(self, encoder, num_layers):
        super().__init__()
        self.encoder = encoder
        self.num_layers = num_layers

    def forward(self, audio_signal: torch.Tensor, length: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        enc = self.encoder
        x = audio_signal.transpose(1, 2)
        x, length = enc.pre_encode(x=x, lengths=length)
        length = length.to(torch.int64)
        x, pos_emb = enc.pos_enc(x=x, cache_len=0)
        max_audio_length = x.size(1)
        pad_mask, att_mask = enc._create_masks(
            att_context_size=enc.att_context_size,
            padding_length=length,
            max_audio_length=max_audio_length,
            offset=None,
            device=x.device,
        )
        for layer in enc.layers[: self.num_layers]:
            x = layer(x=x, att_mask=att_mask, pos_emb=pos_emb, pad_mask=pad_mask)
        return x, length


def export_and_run(wrapper, mel, length):
    wrapper.eval()
    with torch.no_grad():
        eager_out, _ = wrapper(audio_signal=mel, length=length)

    ep = export(
        wrapper, (),
        kwargs={"audio_signal": mel, "length": length},
        dynamic_shapes={"audio_signal": {2: Dim.AUTO}, "length": {}},
        strict=False,
    )
    edge_mgr = to_edge_rewrite_and_lower(
        ep,
        transform_passes=[ReplaceCopyOpsPass()],
        partitioner=[GgmlPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False, _skip_dim_order=True),
    )
    et = edge_mgr.to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
        ),
    )
    pte = _load_for_executorch_from_buffer(et.buffer)
    result = pte.forward((mel, length))
    return eager_out, result[0]


def analyze_error_distribution(eager, ggml, label):
    """Deep analysis of the error between eager and ggml outputs."""
    e = eager.float()
    g = ggml.float()
    diff = (e - g).abs()

    cos = F.cosine_similarity(e.flatten().unsqueeze(0), g.flatten().unsqueeze(0)).item()
    has_nan_eager = torch.isnan(e).any().item()
    has_nan_ggml = torch.isnan(g).any().item()
    has_inf_ggml = torch.isinf(g).any().item()

    print(f"\n--- {label} ---")
    print(f"  cos_sim={cos:.6f}  has_nan_eager={has_nan_eager}  has_nan_ggml={has_nan_ggml}  has_inf_ggml={has_inf_ggml}")
    print(f"  eager: mean={e.mean():.4f} std={e.std():.4f} min={e.min():.4f} max={e.max():.4f}")
    print(f"  ggml:  mean={g.mean():.4f} std={g.std():.4f} min={g.min():.4f} max={g.max():.4f}")
    print(f"  diff:  mean={diff.mean():.6f} std={diff.std():.6f} max={diff.max():.4f} "
          f"median={diff.median():.6f}")

    # Per-position (time-step) error
    # Shape is [B, T, D] — compute error per time step
    if diff.dim() == 3:
        per_time = diff[0].mean(dim=-1)  # [T]
        per_chan = diff[0].mean(dim=0)    # [D]
        top5_time = per_time.topk(min(5, per_time.size(0)))
        top5_chan = per_chan.topk(min(5, per_chan.size(0)))
        print(f"  per-time-step error (top 5): positions={top5_time.indices.tolist()} "
              f"errors={[f'{v:.4f}' for v in top5_time.values.tolist()]}")
        print(f"  per-channel error (top 5): channels={top5_chan.indices.tolist()} "
              f"errors={[f'{v:.4f}' for v in top5_chan.values.tolist()]}")

        # Check what fraction of elements have large error
        for thresh in [0.1, 1.0, 5.0, 10.0]:
            frac = (diff > thresh).float().mean().item()
            if frac > 0:
                print(f"  fraction with |error| > {thresh}: {frac:.4%}")

    return cos


def run_enhanced_prefix_scan(encoder, mel, length, stages):
    """Prefix scan with detailed error analysis at each stage."""
    print("\n" + "=" * 80)
    print("PART 3: ENHANCED PREFIX SCAN WITH ERROR DISTRIBUTION")
    print("=" * 80)

    for k in stages:
        label = f"Prefix {k} layers" if k > 0 else "Subsampling+pos_enc"
        t0 = time.time()
        wrapper = EncoderPrefix(encoder, num_layers=k)
        try:
            eager_out, ggml_out = export_and_run(wrapper, mel, length)
        except Exception as e:
            print(f"\n--- {label}: FAILED ({e}) ---")
            continue
        elapsed = time.time() - t0
        cos = analyze_error_distribution(eager_out, ggml_out, f"{label} ({elapsed:.1f}s)")


def main():
    print("Loading model...")
    model = load_model()
    encoder = model.encoder
    n_layers = len(encoder.layers)
    print(f"Encoder has {n_layers} Conformer layers")

    # Load mel
    audio_path = os.path.join(os.path.dirname(__file__), "..", "test_audio.wav")
    sample_rate = model.preprocessor._cfg.sample_rate
    audio = load_audio(audio_path, sample_rate=sample_rate)
    with torch.no_grad():
        mel, mel_len = model.preprocessor(
            input_signal=audio, length=torch.tensor([audio.shape[1]])
        )
    length = mel_len.to(torch.int64)
    T_mel = mel.shape[2]
    print(f"Mel shape: {list(mel.shape)} (T_mel={T_mel})")
    encoder.update_max_seq_length(seq_length=T_mel, device=mel.device)

    # Part 1: F16 weight analysis (instant)
    analyze_f16_weights(encoder)

    # Part 2: Eager intermediate analysis (instant)
    analyze_eager_intermediates(encoder, mel, length)

    # Part 3: Enhanced prefix scan (slow but diagnostic)
    # Test at key stages to understand error distribution
    run_enhanced_prefix_scan(encoder, mel, length, [0, 6, 12, 18, 24])


if __name__ == "__main__":
    with torch.no_grad():
        main()
