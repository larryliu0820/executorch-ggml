#!/usr/bin/env python3
"""Profile Voxtral decoder on CUDA via GGML backend.

Usage:
    GGML_PERF_LOG=1 GGML_BACKEND_DEVICE=cuda GGML_GRAPH_CACHE=1 python tests/profile_voxtral_decoder.py
    GGML_PROFILE=1 GGML_BACKEND_DEVICE=cuda GGML_GRAPH_CACHE=1 python tests/profile_voxtral_decoder.py
"""

import os
import sys
import time

import torch
from torch.export import Dim, export

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _repo_root)

MODEL_PATH = "/home/dev/models/Voxtral-Mini-4B-Realtime-2602"
DTYPE = torch.float32
MAX_SEQ_LEN = 4096


def main():
    from executorch.examples.models.voxtral_realtime.export_voxtral_rt import (
        TextDecoderExport,
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
    model = load_model(MODEL_PATH, max_seq_len=MAX_SEQ_LEN, dtype=DTYPE)

    # Apply decoder optimizations
    from executorch_ggml.modules.voxtral_attention import swap_voxtral_attention
    swap_voxtral_attention(model)
    print("Applied swap_voxtral_attention")

    from executorch_ggml.modules.voxtral_decoder_rope import swap_decoder_rope
    n_rope = swap_decoder_rope(model, freq_base=model.config.rope_theta)
    print(f"Applied swap_decoder_rope ({n_rope} layers)")

    from executorch_ggml.passes.fold_decoder_rms_norm_weights import fold_decoder_rms_norm_weights
    n_fold = fold_decoder_rms_norm_weights(model)
    print(f"Applied fold_decoder_rms_norm_weights ({n_fold} folded)")

    # Wrap for export
    wrapper = TextDecoderExport(model)
    wrapper.eval()

    # Export with dynamic shapes
    dim = model.config.dim  # 3072
    seq_dim = Dim("seq_len", min=1, max=MAX_SEQ_LEN)
    sample_embeds = torch.randn(1, 4, dim, dtype=DTYPE)
    sample_pos = torch.arange(4, dtype=torch.long)

    print(f"\nExporting decoder (dim={dim}, seq_dim=1..{MAX_SEQ_LEN})...")
    t0 = time.time()
    prog = export(
        wrapper,
        (sample_embeds, sample_pos),
        dynamic_shapes={
            "input_embeds": {1: seq_dim},
            "cache_position": {0: seq_dim},
        },
        strict=True,
    )
    t1 = time.time()
    print(f"Export took {t1-t0:.1f}s")

    # Count nodes
    node_counts = {}
    for n in prog.graph.nodes:
        if n.op == "call_function":
            op = str(n.target).split(".")[-1]
            node_counts[op] = node_counts.get(op, 0) + 1
    total_nodes = sum(node_counts.values())
    print(f"\nExported graph: {total_nodes} call_function nodes")
    for op, count in sorted(node_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {op:<40} {count:>5}")

    rope_nodes = sum(1 for n in prog.graph.nodes
                     if n.op == "call_function" and "rope" in str(n.target))
    rms_nodes = sum(1 for n in prog.graph.nodes
                    if n.op == "call_function" and "rms_norm" in str(n.target))
    print(f"\nFused ops: rope={rope_nodes}, rms_norm={rms_nodes}")

    # Lower
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

    # Load
    print("\nLoading PTE...")
    pte = _load_for_executorch_from_buffer(et_prog.buffer)

    # Simulate autoregressive decoding: single-token steps at different positions
    print("\nSimulating decode (single-token steps)...")

    # Warmup
    for pos in range(5):
        embeds = torch.randn(1, 1, dim, dtype=DTYPE)
        cache_pos = torch.tensor([pos], dtype=torch.long)
        t0 = time.time()
        out = pte.forward((embeds, cache_pos))[0]
        torch.cuda.synchronize()
        t1 = time.time()
        print(f"  pos={pos}: {(t1-t0)*1000:.1f}ms, output shape: {list(out.shape)}")

    # Benchmark decode steps (standard: copy logits to CPU)
    n_steps = 50
    print(f"\nBenchmark ({n_steps} decode steps, pos=10..{10+n_steps-1})...")
    times = []
    for i in range(n_steps):
        pos = 10 + i
        embeds = torch.randn(1, 1, dim, dtype=DTYPE)
        cache_pos = torch.tensor([pos], dtype=torch.long)
        torch.cuda.synchronize()
        t0 = time.time()
        out = pte.forward((embeds, cache_pos))[0]
        torch.cuda.synchronize()
        t1 = time.time()
        times.append((t1 - t0) * 1000)

    times.sort()
    trimmed = times[5:-5]
    avg = sum(trimmed) / len(trimmed)
    mn = min(times)
    p50 = times[len(times)//2]
    tok_s = 1000.0 / avg
    print(f"  Avg(trimmed): {avg:.1f}ms/tok, p50: {p50:.1f}ms/tok")
    print(f"  Min: {mn:.1f}ms/tok, tok/s: {tok_s:.0f}")
    print(f"  Output shape: {list(out.shape)}")

    # Note: With GGML_SKIP_OUTPUT_COPY=1, output copy drops from ~13ms to ~0.01ms.
    # Combined with cuda_argmax_f32 (from fused_kernels.cu), this adds ~0.1ms
    # for the argmax kernel + single int64 D2H copy. Net savings: ~12.9ms/tok.
    # Use the C++ benchmark_llm for the full skip_copy + argmax pipeline.

    # Skip prefill: loading a second PTE exhausts GPU memory for FP32 4B model


if __name__ == "__main__":
    with torch.no_grad():
        main()
