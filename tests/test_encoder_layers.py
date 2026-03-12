"""Layer-by-layer encoder quality diagnostic.

Exports prefix-encoders (first k layers) through GGML and compares to eager,
isolating which Conformer layer(s) introduce the most numerical error.

Usage:
    source .venv/bin/activate

    # Coarse prefix scan (layers 0-24 cumulative through GGML)
    python3 tests/test_encoder_layers.py --layers 0,6,12,18,24,proj

    # Isolate: run each layer individually through GGML (eager inputs)
    # This is MUCH faster and isolates per-layer GGML error
    python3 tests/test_encoder_layers.py --isolate --layers 18,19,20,21,22,23

    # Isolate all 24 layers
    python3 tests/test_encoder_layers.py --isolate
"""

import argparse
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
    elif waveform.dim() == 2:
        waveform = waveform.T
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        import torchaudio
        waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
    return waveform


def load_model():
    import nemo.collections.asr as nemo_asr
    model = nemo_asr.models.ASRModel.from_pretrained(
        "nvidia/parakeet-tdt-0.6b-v3", map_location="cpu"
    )
    model.eval()
    model.cpu()
    return model


# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------

class EncoderPrefix(nn.Module):
    """Run encoder up to layer `num_layers` (0 = just subsampling + pos_enc)."""

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


class EncoderFull(nn.Module):
    """Full encoder + transpose + projection."""

    def __init__(self, encoder, project_encoder):
        super().__init__()
        self.encoder = encoder
        self.project_encoder = project_encoder

    def forward(self, audio_signal: torch.Tensor, length: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded, enc_len = self.encoder(audio_signal=audio_signal, length=length)
        encoded_t = encoded.transpose(1, 2)
        f_proj = self.project_encoder(encoded_t)
        return f_proj, enc_len


class SingleLayerWrapper(nn.Module):
    """Run a single ConformerLayer through GGML.

    Takes pre-computed (x, att_mask, pos_emb, pad_mask) as input so that only
    this one layer is traced/exported. The att_mask and pad_mask are baked in
    as buffers (they don't depend on the dynamic input).
    """

    def __init__(self, layer, att_mask, pad_mask):
        super().__init__()
        self.layer = layer
        self.register_buffer("att_mask", att_mask)
        self.register_buffer("pad_mask", pad_mask)

    def forward(self, x: torch.Tensor, pos_emb: torch.Tensor) -> torch.Tensor:
        return self.layer(
            x=x, att_mask=self.att_mask, pos_emb=pos_emb, pad_mask=self.pad_mask,
        )


# ---------------------------------------------------------------------------
# Export + run helpers
# ---------------------------------------------------------------------------

def lower_and_run(ep, inputs):
    """Lower an ExportedProgram through GGML and run it."""
    edge_mgr = to_edge_rewrite_and_lower(
        ep,
        transform_passes=[ReplaceCopyOpsPass()],
        partitioner=[GgmlPartitioner()],
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=True,
        ),
    )
    et = edge_mgr.to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
        ),
    )
    pte = _load_for_executorch_from_buffer(et.buffer)
    result = pte.forward(inputs)
    return result


def export_and_run_prefix(wrapper, mel, length):
    """Export prefix wrapper through GGML, return (eager_out, ggml_out)."""
    wrapper.eval()
    with torch.no_grad():
        eager_out, eager_len = wrapper(audio_signal=mel, length=length)

    ep = export(
        wrapper, (),
        kwargs={"audio_signal": mel, "length": length},
        dynamic_shapes={"audio_signal": {2: Dim.AUTO}, "length": {}},
        strict=False,
    )
    result = lower_and_run(ep, (mel, length))
    return eager_out, result[0]


def export_and_run_single_layer(wrapper, x, pos_emb):
    """Export single-layer wrapper through GGML, return (eager_out, ggml_out).

    Uses static shapes since we're testing numerical accuracy at a fixed size,
    not shape flexibility.
    """
    wrapper.eval()
    with torch.no_grad():
        eager_out = wrapper(x, pos_emb)

    ep = export(wrapper, (x, pos_emb), strict=False)
    result = lower_and_run(ep, (x, pos_emb))
    return eager_out, result[0]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def cosine_sim(a, b):
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    return F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()


def print_row(stage_str, cos, max_diff, delta, elapsed):
    print(f"{stage_str:<35} {cos:>8.4f} {max_diff:>10.4f} {delta:>10} {elapsed:>6.1f}s")


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def run_prefix_mode(model, mel, length, stages):
    """Original prefix mode: export layers 0..k through GGML."""
    encoder = model.encoder

    print(f"\n{'Stage':<35} {'cos_sim':>8} {'max_diff':>10} {'delta_cos':>10} {'time':>7}")
    print("-" * 80)

    prev_cos = None
    results = []

    for stage_k, label, is_proj in stages:
        t0 = time.time()
        if is_proj:
            wrapper = EncoderFull(encoder, model.joint.project_encoder)
        else:
            wrapper = EncoderPrefix(encoder, num_layers=stage_k)

        try:
            eager_out, ggml_out = export_and_run_prefix(wrapper, mel, length)
        except Exception as e:
            elapsed = time.time() - t0
            print(f"Stage {stage_k:>2} ({label:<25}): FAILED ({e})")
            results.append((stage_k, label, None, None, None))
            continue

        elapsed = time.time() - t0
        cos = cosine_sim(eager_out, ggml_out)
        max_diff = (eager_out.float() - ggml_out.float()).abs().max().item()
        delta = f"{cos - prev_cos:+.4f}" if prev_cos is not None else "  ---"
        prev_cos = cos
        print_row(f"Stage {stage_k:>2} ({label})", cos, max_diff, delta, elapsed)
        results.append((stage_k, label, cos, max_diff, elapsed))

    return results


def run_isolate_mode(model, mel, length, layer_indices):
    """Isolate mode: run each layer individually through GGML with eager inputs."""
    encoder = model.encoder
    n_layers = len(encoder.layers)

    # Run the encoder front-end eagerly to get intermediate state
    print("Running encoder front-end eagerly...")
    with torch.no_grad():
        x = mel.transpose(1, 2)
        x, enc_length = encoder.pre_encode(x=x, lengths=length)
        enc_length = enc_length.to(torch.int64)
        x, pos_emb = encoder.pos_enc(x=x, cache_len=0)
        max_audio_length = x.size(1)
        pad_mask, att_mask = encoder._create_masks(
            att_context_size=encoder.att_context_size,
            padding_length=enc_length,
            max_audio_length=max_audio_length,
            offset=None,
            device=x.device,
        )

    print(f"Intermediate shape: x={list(x.shape)}, pos_emb={list(pos_emb.shape)}")
    print(f"att_mask={'None' if att_mask is None else list(att_mask.shape)}, "
          f"pad_mask={'None' if pad_mask is None else list(pad_mask.shape)}")

    print(f"\n{'Layer':<35} {'cos_sim':>8} {'max_diff':>10} {'mean_diff':>10} {'time':>7}")
    print("-" * 80)

    results = []

    # Run each layer eagerly up to the target, then test target through GGML
    cur_x = x.clone()
    cur_layer_idx = 0

    for target_idx in layer_indices:
        if target_idx < 0 or target_idx >= n_layers:
            print(f"Layer {target_idx}: out of range [0, {n_layers - 1}], skipping")
            continue

        # Advance eagerly to reach the target layer's input
        while cur_layer_idx < target_idx:
            with torch.no_grad():
                cur_x = encoder.layers[cur_layer_idx](
                    x=cur_x, att_mask=att_mask, pos_emb=pos_emb, pad_mask=pad_mask,
                )
            cur_layer_idx += 1

        # Now cur_x is the eager input to layer[target_idx]
        layer_input = cur_x.clone()

        t0 = time.time()
        wrapper = SingleLayerWrapper(encoder.layers[target_idx], att_mask, pad_mask)

        try:
            eager_out, ggml_out = export_and_run_single_layer(wrapper, layer_input, pos_emb)
        except Exception as e:
            elapsed = time.time() - t0
            print(f"Layer {target_idx:>2}: FAILED ({e})")
            results.append((target_idx, None, None, None))
            # Still advance eagerly so we can test later layers
            with torch.no_grad():
                cur_x = encoder.layers[target_idx](
                    x=cur_x, att_mask=att_mask, pos_emb=pos_emb, pad_mask=pad_mask,
                )
            cur_layer_idx = target_idx + 1
            continue

        elapsed = time.time() - t0
        cos = cosine_sim(eager_out, ggml_out)
        max_diff = (eager_out.float() - ggml_out.float()).abs().max().item()
        mean_diff = (eager_out.float() - ggml_out.float()).abs().mean().item()
        print_row(f"Layer {target_idx:>2}", cos, max_diff, f"{mean_diff:.6f}", elapsed)
        results.append((target_idx, cos, max_diff, mean_diff))

        # Advance past this layer eagerly for the next iteration
        with torch.no_grad():
            cur_x = encoder.layers[target_idx](
                x=cur_x, att_mask=att_mask, pos_emb=pos_emb, pad_mask=pad_mask,
            )
        cur_layer_idx = target_idx + 1

    return results


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_layers_prefix(layers_str, n_layers):
    """Parse --layers for prefix mode."""
    if layers_str is None:
        stages = [(k, f"+ layer {k - 1}" if k > 0 else "subsampling+pos_enc", False)
                  for k in range(n_layers + 1)]
        stages.append((n_layers + 1, "+ projection", True))
        return stages

    stages = []
    for tok in layers_str.split(","):
        tok = tok.strip()
        if tok in ("proj", "projection"):
            stages.append((n_layers + 1, "+ projection", True))
        else:
            k = int(tok)
            if k == 0:
                stages.append((0, "subsampling+pos_enc", False))
            elif 0 < k <= n_layers:
                stages.append((k, f"+ layer {k - 1}", False))
            else:
                print(f"Warning: layer {k} out of range [0, {n_layers}], skipping")
    return stages


def parse_layers_isolate(layers_str, n_layers):
    """Parse --layers for isolate mode."""
    if layers_str is None:
        return list(range(n_layers))
    return [int(tok.strip()) for tok in layers_str.split(",") if tok.strip().isdigit()]


def main():
    parser = argparse.ArgumentParser(description="Layer-by-layer encoder quality diagnostic")
    parser.add_argument(
        "--layers", type=str, default=None,
        help="Comma-separated layers. Prefix mode: '0,6,12,18,24,proj'. "
             "Isolate mode: '18,19,20,21,22,23'."
    )
    parser.add_argument(
        "--isolate", action="store_true",
        help="Isolate mode: test each layer individually through GGML "
             "(eager inputs). Much faster and more diagnostic."
    )
    args = parser.parse_args()

    print("Loading model...")
    model = load_model()
    encoder = model.encoder
    n_layers = len(encoder.layers)
    print(f"Encoder has {n_layers} Conformer layers")

    # Load mel spectrogram
    audio_path = os.path.join(os.path.dirname(__file__), "..", "test_audio.wav")
    if os.path.exists(audio_path):
        print(f"Loading audio from {audio_path}")
        sample_rate = model.preprocessor._cfg.sample_rate
        audio = load_audio(audio_path, sample_rate=sample_rate)
        with torch.no_grad():
            mel, mel_len = model.preprocessor(
                input_signal=audio, length=torch.tensor([audio.shape[1]])
            )
        T_mel = mel.shape[2]
        print(f"Mel spectrogram shape: {mel.shape} (T_mel={T_mel})")
    else:
        print(f"test_audio.wav not found, using random input")
        torch.manual_seed(42)
        T_mel = 744
        mel = torch.randn(1, 128, T_mel)
        mel_len = torch.tensor([T_mel], dtype=torch.int64)

    length = mel_len.to(torch.int64)
    encoder.update_max_seq_length(seq_length=T_mel, device=mel.device)

    if args.isolate:
        layer_indices = parse_layers_isolate(args.layers, n_layers)
        print(f"\nISOLATE MODE: testing layers {layer_indices} individually through GGML")
        results = run_isolate_mode(model, mel, length, layer_indices)

        # Summary
        valid = [(idx, cos, md, mean) for idx, cos, md, mean in results if cos is not None]
        if valid:
            valid.sort(key=lambda r: r[1])  # sort by cos_sim ascending (worst first)
            print("\n" + "=" * 80)
            print("SUMMARY: Layers sorted by cos_sim (worst first)")
            print("=" * 80)
            for idx, cos, max_diff, mean_diff in valid:
                print(f"  Layer {idx:>2}: cos={cos:.6f}  max_diff={max_diff:.4f}  mean_diff={mean_diff:.6f}")
    else:
        stages = parse_layers_prefix(args.layers, n_layers)
        print(f"Testing {len(stages)} stages: {[s[0] for s in stages]}")
        results = run_prefix_mode(model, mel, length, stages)

        # Summary
        drops = []
        for i in range(1, len(results)):
            prev = results[i - 1]
            curr = results[i]
            if prev[2] is not None and curr[2] is not None:
                drop = curr[2] - prev[2]
                drops.append((drop, curr[0], curr[1], curr[2]))
        drops.sort()
        if drops:
            print("\n" + "=" * 80)
            print("SUMMARY: Stages sorted by cos_sim drop (worst first)")
            print("=" * 80)
            for drop, stage_k, label, cos in drops[:10]:
                print(f"  Stage {stage_k:>2} ({label:<25}): cos={cos:.4f}  delta={drop:+.6f}")


if __name__ == "__main__":
    with torch.no_grad():
        main()
