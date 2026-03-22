#!/usr/bin/env python3
"""Profile Voxtral encoder on CUDA via GGML backend.

Exports the audio encoder, runs on CUDA with profiling, reports:
- Node count and op breakdown
- Per-call timing (build/compute/total)
- Graph cache hit/miss behavior

Usage:
    GGML_PERF_LOG=1 GGML_BACKEND_DEVICE=cuda python tests/profile_voxtral_encoder.py
    GGML_PROFILE=1 GGML_BACKEND_DEVICE=cuda python tests/profile_voxtral_encoder.py
"""

import os
import sys
import time

import torch
import torch.nn as nn
from torch.export import Dim, export

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _repo_root)

MODEL_PATH = "/home/dev/models/Voxtral-Mini-4B-Realtime-2602"
DTYPE = torch.float32


def count_ir_ops(et_prog, method_name="forward"):
    """Count ops in the lowered IR by inspecting the delegate blob."""
    # We can't easily inspect the flatbuffer, but we can count nodes
    # in the edge program before final lowering
    pass


def main():
    from executorch.examples.models.voxtral_realtime.export_voxtral_rt import (
        AudioEncoderExport,
    )
    from executorch.examples.models.voxtral_realtime.model import load_model
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

    print(f"Loading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH, max_seq_len=4096, dtype=DTYPE)

    # Apply encoder optimizations
    from executorch_ggml.modules.rope import swap_encoder_rope
    swap_encoder_rope(model, freq_base=model.config.enc_rope_theta)
    print("Applied swap_encoder_rope")

    from executorch_ggml.modules.voxtral_conv import swap_causal_conv1d
    n_conv = swap_causal_conv1d(model)
    print(f"Applied swap_causal_conv1d ({n_conv} swapped)")

    from executorch_ggml.passes.fold_encoder_rms_norm_weights import fold_encoder_rms_norm_weights
    n_fold = fold_encoder_rms_norm_weights(model)
    print(f"Applied fold_encoder_rms_norm_weights ({n_fold} folded)")

    # Wrap for export
    wrapper = AudioEncoderExport(model)
    wrapper.eval()

    # Use a realistic input size: 10s audio = 1000 mel frames
    T_mel = 1000  # ~6.25s of audio (1000 mel frames at 160 hop / 16kHz)
    # Must be multiple of 8
    T_mel = (T_mel // 8) * 8  # 1000 -> 1000 (already multiple of 8)
    mel = torch.randn(1, 128, T_mel, dtype=DTYPE)
    print(f"Input mel shape: {mel.shape}")

    # Export
    print("\nExporting...")
    t0 = time.time()
    prog = export(wrapper, (mel,), dynamic_shapes={"mel": {2: Dim.AUTO}}, strict=True)
    t1 = time.time()
    print(f"Export took {t1-t0:.1f}s")

    # Count nodes in exported graph
    node_counts = {}
    for n in prog.graph.nodes:
        if n.op == "call_function":
            op = str(n.target).split(".")[-1]
            node_counts[op] = node_counts.get(op, 0) + 1
    total_nodes = sum(node_counts.values())
    print(f"\nExported graph: {total_nodes} call_function nodes")
    for op, count in sorted(node_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {op:<40} {count:>5}")

    # Check for ggml.rope nodes
    rope_nodes = sum(1 for n in prog.graph.nodes
                     if n.op == "call_function" and "rope" in str(n.target))
    rms_nodes = sum(1 for n in prog.graph.nodes
                    if n.op == "call_function" and "rms_norm" in str(n.target))
    print(f"\nFused ops: rope={rope_nodes}, rms_norm={rms_nodes}")

    # Lower to GGML
    print("\nLowering to GGML...")
    t0 = time.time()
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
    t1 = time.time()
    print(f"Lowering took {t1-t0:.1f}s")

    # Load and run
    print("\nLoading PTE...")
    pte = _load_for_executorch_from_buffer(et_prog.buffer)

    # Warmup (5 runs to stabilize CUDA graphs)
    print("Warmup (5 runs)...")
    for i in range(5):
        t0 = time.time()
        out = pte.forward((mel,))[0]
        torch.cuda.synchronize()
        t1 = time.time()
        print(f"  Run {i}: {(t1-t0)*1000:.1f}ms, output shape: {list(out.shape)}")

    # Benchmark with proper CUDA sync
    n_runs = 20
    print(f"\nBenchmark ({n_runs} runs, CUDA synced)...")
    times = []
    for i in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.time()
        out = pte.forward((mel,))[0]
        torch.cuda.synchronize()
        t1 = time.time()
        times.append((t1-t0) * 1000)

    times.sort()
    # Drop top/bottom 10%
    trimmed = times[2:-2]
    avg = sum(trimmed) / len(trimmed)
    mn = min(times)
    mx = max(times)
    p50 = times[len(times)//2]
    print(f"  Avg(trimmed): {avg:.1f}ms, p50: {p50:.1f}ms, Min: {mn:.1f}ms, Max: {mx:.1f}ms")
    print(f"  Output shape: {list(out.shape)}")


if __name__ == "__main__":
    with torch.no_grad():
        main()
