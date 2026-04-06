#!/usr/bin/env python3
"""Export Qwen3-0.6B with Q8_0 quantization.

Usage (from repo root):
    source .venv/bin/activate
    python runner/export_qwen3_q8.py
"""

import os
import time

import torch

# Must import executorch_ggml first for RTLD_GLOBAL symbol resolution
import executorch_ggml  # noqa: F401, I001  # isort: skip


def export_qwen3_q8(max_seq_len: int = 128):
    """Export Qwen3-0.6B with Q8_0 quantization."""
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
    from executorch_ggml import GgmlPartitioner, GgmlQuantConfig
    from executorch_ggml.passes import RemoveGraphAssertsPass
    from executorch_ggml.passes.replace_copy_ops_pass import ReplaceCopyOpsPass
    from optimum.exporters.executorch.integrations import CausalLMExportableModule
    from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig

    model_id = "Qwen/Qwen3-0.6B"
    config = AutoConfig.from_pretrained(model_id)
    if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
        config.rope_scaling["type"] = "default"

    print(f"Loading {model_id}...")
    eager_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cpu",
        torch_dtype=torch.float32,
        config=config,
        attn_implementation="sdpa",
        generation_config=GenerationConfig(
            use_cache=True,
            cache_implementation="static",
            max_length=max_seq_len,
            cache_config={"batch_size": 1, "max_cache_len": max_seq_len},
        ),
    )

    # --- Performance optimizations (before export) ---
    from executorch_ggml.modules.rms_norm import swap_rms_norm
    from executorch_ggml.passes.fold_rms_norm_weights import fold_rms_norm_weights

    from executorch_ggml.passes.fuse_projections import (
        fuse_qkv_projections,
        fuse_gate_up_projections,
    )

    n_rms = swap_rms_norm(eager_model)
    print(f"  swap_rms_norm: {n_rms} modules")
    n_fold = fold_rms_norm_weights(eager_model)
    print(f"  fold_rms_norm_weights: {n_fold} norms folded")
    n_qkv = fuse_qkv_projections(eager_model)
    print(f"  fuse_qkv_projections: {n_qkv} attention modules")
    n_mlp = fuse_gate_up_projections(eager_model)
    print(f"  fuse_gate_up_projections: {n_mlp} MLP modules")

    print("Exporting with torch.export...")
    exportable = CausalLMExportableModule(
        eager_model,
        max_seq_len=max_seq_len,
        use_custom_kv_cache=False,
        use_custom_sdpa=False,
        disable_dynamic_shapes=False,
    )
    ep = exportable.export()["model"]

    # --- AOT graph passes (before edge lowering) ---
    from executorch_ggml.passes.fuse_rope_pass import fuse_rope_in_graph
    from executorch_ggml.passes.strip_gqa_expand_pass import strip_gqa_expand

    head_dim = config.hidden_size // config.num_attention_heads
    freq_base = getattr(config, "rope_theta", 1000000.0)
    n_rope = fuse_rope_in_graph(ep.graph_module, head_dim=head_dim, freq_base=freq_base)
    print(f"  fuse_rope: {n_rope} instances")
    n_gqa = strip_gqa_expand(ep.graph_module)
    print(f"  strip_gqa_expand: {n_gqa} expansions removed")

    from executorch_ggml.passes.cse_pass import eliminate_common_subexpressions
    n_cse = eliminate_common_subexpressions(ep.graph_module)
    print(f"  CSE: {n_cse} duplicate nodes eliminated")

    print("Lowering with Q8_0 quantization...")
    t0 = time.time()
    quant_config = GgmlQuantConfig()
    edge_mgr = to_edge_transform_and_lower(
        ep,
        partitioner=[GgmlPartitioner(quant_config=quant_config)],
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=True,
        ),
        transform_passes=[ReplaceCopyOpsPass(), RemoveGraphAssertsPass()],
        constant_methods=exportable.metadata,
    )
    t1 = time.time()
    print(f"Lowering took {t1 - t0:.1f}s")

    return edge_mgr


def main():
    max_seq_len = 128
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "qwen3")
    q8_path = os.path.join(out_dir, "qwen3_q8_0.pte")
    f32_path = os.path.join(out_dir, "qwen3.pte")

    edge_mgr = export_qwen3_q8(max_seq_len=max_seq_len)
    et = edge_mgr.to_executorch()
    pte_bytes = et.buffer

    os.makedirs(out_dir, exist_ok=True)
    with open(q8_path, "wb") as f:
        f.write(pte_bytes)

    q8_size_mb = len(pte_bytes) / (1024 * 1024)
    print(f"\nSaved Q8_0 .pte to {q8_path} ({q8_size_mb:.1f} MB)")

    if os.path.exists(f32_path):
        f32_size_mb = os.path.getsize(f32_path) / (1024 * 1024)
        print(f"F32 .pte: {f32_size_mb:.1f} MB")
        print(f"Compression: {f32_size_mb / q8_size_mb:.2f}x")

    print("\nTo run inference:")
    print(f"  python runner/run_qwen3.py --model {q8_path}")


if __name__ == "__main__":
    main()
