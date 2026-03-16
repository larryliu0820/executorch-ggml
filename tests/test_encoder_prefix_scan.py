#!/usr/bin/env python3
"""Prefix scan test for audio_encoder: export with 0..N encoder layers,
run each via GGML backend, compare against eager PyTorch.

Usage:
    source .venv/bin/activate
    python tests/test_encoder_prefix_scan.py
"""

import json
import os
import subprocess
import sys
import tempfile

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
DTYPE = torch.float16


class TruncatedEncoder(nn.Module):
    """CausalWhisperEncoder truncated to n_layers transformer layers."""

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
    """AudioEncoderExport with truncated encoder."""

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


def export_pte(model, n_layers: int, mel: torch.Tensor, pte_path: str):
    """Export truncated encoder to .pte file."""
    from executorch.exir import (
        EdgeCompileConfig,
        ExecutorchBackendConfig,
        to_edge_transform_and_lower,
    )
    from executorch.exir.passes import MemoryPlanningPass
    from executorch_ggml import GgmlPartitioner
    from executorch_ggml.passes import BF16UnsafeOpsCastPass, RemoveGraphAssertsPass
    from executorch_ggml.passes.replace_copy_ops_pass import ReplaceCopyOpsPass

    enc = TruncatedAudioEncoder(model, n_layers)
    enc.eval()

    prog = export(enc, (mel,), dynamic_shapes={"mel": {2: Dim.AUTO}}, strict=True)

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

    with open(pte_path, "wb") as f:
        et_prog.write_to_file(f)


def run_pte_subprocess(pte_path: str, mel_path: str) -> dict:
    """Run .pte in a subprocess with saved mel tensor, return result dict."""
    dtype_str = "torch.float16" if DTYPE == torch.float16 else "torch.bfloat16"
    script = f"""
import sys, json, torch
sys.path.insert(0, {_repo_root!r})
import executorch_ggml
from executorch.runtime import Runtime
rt = Runtime.get()
prog = rt.load_program({pte_path!r})
method = prog.load_method("forward")
mel = torch.load({mel_path!r}, weights_only=True)
out = method.execute([mel])[0]
has_nan = torch.isnan(out).any().item()
result = {{"has_nan": has_nan, "shape": list(out.shape)}}
if not has_nan:
    vals = out.float()
    result["min"] = vals.min().item()
    result["max"] = vals.max().item()
    # Save output for comparison
    torch.save(out, {pte_path!r} + ".out.pt")
print("RESULT:" + json.dumps(result))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=120,
    )
    for line in result.stdout.splitlines():
        if line.startswith("RESULT:"):
            return json.loads(line[7:])
    stderr = result.stderr
    for sline in stderr.splitlines():
        if "GGML_ASSERT" in sline:
            return {"error": sline.strip()}
    if result.returncode != 0:
        return {"error": f"exit code {result.returncode}"}
    return {"error": "no RESULT line in output"}


def run_eager(model, n_layers: int, mel: torch.Tensor) -> torch.Tensor:
    """Run truncated encoder eagerly in PyTorch."""
    enc = TruncatedAudioEncoder(model, n_layers)
    enc.eval()
    with torch.no_grad():
        return enc(mel)


def main():
    from executorch.examples.models.voxtral_realtime.model import load_model

    print(f"Loading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH, max_seq_len=4096, dtype=DTYPE, backend="cuda")

    total_layers = len(model.encoder.layers)
    print(f"Encoder has {total_layers} transformer layers, dtype={DTYPE}")

    # Create deterministic mel input and save it
    torch.manual_seed(42)
    mel = torch.randn(1, 128, 8, dtype=DTYPE)
    tmpdir = tempfile.mkdtemp(prefix="enc_scan_")
    mel_path = os.path.join(tmpdir, "mel.pt")
    torch.save(mel, mel_path)

    test_points = [0, 1, 2, 4, 8, 16, 32]
    test_points = [n for n in test_points if n <= total_layers]

    print()
    first_bad = None
    for n in test_points:
        label = f"layers={n:2d}"

        # 1. Run eager
        eager_out = run_eager(model, n, mel)
        eager_nan = torch.isnan(eager_out).any().item()

        # 2. Export .pte
        pte_path = os.path.join(tmpdir, f"enc_{n}.pte")
        try:
            export_pte(model, n, mel, pte_path)
        except Exception as e:
            print(f"  {label}: EXPORT ERROR  {e}")
            continue

        # 3. Run in subprocess
        result = run_pte_subprocess(pte_path, mel_path)

        if "error" in result:
            print(f"  {label}: CRASH    {result['error']}")
            if first_bad is None:
                first_bad = n
            continue

        ggml_nan = result["has_nan"]

        # 4. Compare
        status = []
        if eager_nan:
            status.append("eager=NaN")
        if ggml_nan:
            status.append("ggml=NaN")
            if first_bad is None:
                first_bad = n

        if not eager_nan and not ggml_nan:
            # Load ggml output and compare
            ggml_out_path = pte_path + ".out.pt"
            if os.path.exists(ggml_out_path):
                ggml_out = torch.load(ggml_out_path, weights_only=True)
                diff = (eager_out.float() - ggml_out.float()).abs()
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()
                status.append(f"max_diff={max_diff:.6f}")
                status.append(f"mean_diff={mean_diff:.6f}")

            print(f"  {label}: OK       {', '.join(status)}")
        else:
            print(f"  {label}: {'  '.join(status)}")

    # Narrow if needed
    if first_bad is not None and first_bad > 0:
        lo = test_points[test_points.index(first_bad) - 1]
        hi = first_bad
        print(f"\nNarrowing between layers={lo} and layers={hi}...")
        for n in range(lo + 1, hi):
            label = f"layers={n:2d}"
            eager_out = run_eager(model, n, mel)
            pte_path = os.path.join(tmpdir, f"enc_{n}.pte")
            try:
                export_pte(model, n, mel, pte_path)
            except Exception as e:
                print(f"  {label}: EXPORT ERROR  {e}")
                continue
            result = run_pte_subprocess(pte_path, mel_path)
            if "error" in result:
                print(f"  {label}: CRASH    {result['error']}")
            elif result["has_nan"]:
                print(f"  {label}: ggml=NaN")
            else:
                ggml_out_path = pte_path + ".out.pt"
                if os.path.exists(ggml_out_path):
                    ggml_out = torch.load(ggml_out_path, weights_only=True)
                    diff = (eager_out.float() - ggml_out.float()).abs()
                    print(f"  {label}: OK       max_diff={diff.max().item():.6f}")

    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)
    print("\nDone.")


if __name__ == "__main__":
    main()
