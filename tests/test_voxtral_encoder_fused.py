#!/usr/bin/env python3
"""Voxtral encoder test with fused RoPE + RMS norm.

Loads model, applies swap_encoder_rope, exports truncated encoders
(1, 2, 4, 8, 16, 32 layers) via GGML backend, compares against eager.

Usage:
    DYLD_LIBRARY_PATH=python/executorch_ggml python3 tests/test_voxtral_encoder_fused.py
"""

import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.export import Dim, export

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _repo_root)

MODEL_PATH = (
    "/Users/larryliu/.cache/huggingface/hub/"
    "models--mistralai--Voxtral-Mini-4B-Realtime-2602/"
    "snapshots/2769294da9567371363522aac9bbcfdd19447add"
)
DTYPE = torch.float32


class TruncatedEncoder(nn.Module):
    def __init__(self, full_encoder, n_layers: int):
        super().__init__()
        self.conv_layers = full_encoder.conv_layers
        self.layers = nn.ModuleList(list(full_encoder.layers)[:n_layers])
        self.norm = full_encoder.norm
        self.register_buffer("freqs_cos", full_encoder.freqs_cos)
        self.register_buffer("freqs_sin", full_encoder.freqs_sin)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.conv_layers[0](mel))
        x = F.gelu(self.conv_layers[1](x))
        x = x.transpose(1, 2)
        T = x.shape[1]
        freqs_cos = self.freqs_cos[:T]
        freqs_sin = self.freqs_sin[:T]
        for layer in self.layers:
            x = layer(x, freqs_cos, freqs_sin)
        return self.norm(x)


class TruncatedAudioEncoder(nn.Module):
    def __init__(self, model, n_layers: int):
        super().__init__()
        self.encoder = TruncatedEncoder(model.encoder, n_layers)
        self.adapter = model.adapter
        self.downsample_factor = model.config.downsample_factor

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.encoder(mel)
        B, T, D = x.shape
        x = x.reshape(B, T // self.downsample_factor, D * self.downsample_factor)
        return self.adapter(x)


def export_and_run(wrapper, mel):
    from executorch.exir import (
        EdgeCompileConfig,
        ExecutorchBackendConfig,
        to_edge_transform_and_lower,
    )
    from executorch.exir.passes import MemoryPlanningPass
    from executorch.extension.pybindings.portable_lib import (
        _load_for_executorch_from_buffer,
    )
    from executorch_ggml import GgmlPartitioner
    from executorch_ggml.passes import BF16UnsafeOpsCastPass, RemoveGraphAssertsPass
    from executorch_ggml.passes.replace_copy_ops_pass import ReplaceCopyOpsPass

    wrapper.eval()
    with torch.no_grad():
        eager_out = wrapper(mel)

    prog = export(wrapper, (mel,), dynamic_shapes={"mel": {2: Dim.AUTO}}, strict=True)

    # Count ggml.rope nodes in the exported graph
    rope_nodes = sum(
        1 for n in prog.graph.nodes
        if n.op == "call_function" and "ggml.rope" in str(n.target)
    )
    rms_nodes = sum(
        1 for n in prog.graph.nodes
        if n.op == "call_function" and "rms_norm" in str(n.target)
    )

    et_prog = to_edge_transform_and_lower(
        {"forward": prog},
        partitioner=[GgmlPartitioner()],
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=True,
        ),
        transform_passes=[BF16UnsafeOpsCastPass(), RemoveGraphAssertsPass(), ReplaceCopyOpsPass()],
    ).to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
        ),
    )

    pte = _load_for_executorch_from_buffer(et_prog.buffer)
    ggml_out = pte.forward((mel,))[0]

    return eager_out, ggml_out, rope_nodes, rms_nodes


def main():
    from executorch.examples.models.voxtral_realtime.model import load_model
    from executorch_ggml.modules.rope import swap_encoder_rope

    print(f"Loading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH, max_seq_len=4096, dtype=DTYPE)

    total_layers = len(model.encoder.layers)
    freq_base = model.config.enc_rope_theta
    print(f"Encoder: {total_layers} layers, head_dim={model.config.enc_head_dim}, "
          f"freq_base={freq_base}")

    # Apply fused RoPE swap
    print("Applying swap_encoder_rope...")
    swap_encoder_rope(model, freq_base=freq_base)

    # Verify swap worked
    from executorch_ggml.modules.rope import GgmlEncoderAttention
    assert isinstance(model.encoder.layers[0].attention, GgmlEncoderAttention)
    print("  All encoder attention modules swapped to GgmlEncoderAttention")

    torch.manual_seed(42)
    mel = torch.randn(1, 128, 8, dtype=DTYPE)

    test_points = [1, 2, 4, 8, 16, 32]
    test_points = [n for n in test_points if n <= total_layers]

    print(f"\n{'Layers':<10} {'cos_sim':>10} {'max_diff':>12} {'mean_diff':>12} "
          f"{'rope_ops':>10} {'rms_ops':>10} {'time':>8}")
    print("-" * 82)

    for n in test_points:
        t0 = time.time()
        try:
            wrapper = TruncatedAudioEncoder(model, n)
            eager_out, ggml_out, rope_nodes, rms_nodes = export_and_run(wrapper, mel)
        except Exception as e:
            elapsed = time.time() - t0
            print(f"{n:<10} {'FAILED':>10}   {str(e)[:50]}")
            continue

        elapsed = time.time() - t0
        cos = F.cosine_similarity(
            eager_out.float().flatten().unsqueeze(0),
            ggml_out.float().flatten().unsqueeze(0),
        ).item()
        diff = (eager_out.float() - ggml_out.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        has_nan = torch.isnan(ggml_out).any().item()

        status = "NaN!" if has_nan else ""
        print(f"{n:<10} {cos:>10.6f} {max_diff:>12.6f} {mean_diff:>12.6f} "
              f"{rope_nodes:>10} {rms_nodes:>10} {elapsed:>7.1f}s {status}")

    print("\nDone.")


if __name__ == "__main__":
    with torch.no_grad():
        main()
