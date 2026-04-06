"""GGUF-to-PTE export pipeline for ExecuTorch GGML backend.

This module provides functions to export GGUF files to weight-less PTE files
using the existing ExecuTorch infrastructure, enabling separation of model
graph structure from weight data.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch_ggml import GgmlPartitioner, GgmlQuantConfig
from executorch_ggml.gguf_analyzer import GGUFAnalyzer
from executorch_ggml.passes import RemoveGraphAssertsPass
from executorch_ggml.passes.replace_copy_ops_pass import ReplaceCopyOpsPass
from executorch_ggml.weight_mapping import WeightNameMapper
from optimum.exporters.executorch.integrations import CausalLMExportableModule
from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig


class GGUFExportConfig:
    """Configuration for GGUF-to-PTE export."""

    def __init__(
        self,
        max_seq_len: int = 128,
        enable_quantization: bool = False,
        quant_config: Optional[GgmlQuantConfig] = None,
        preserve_dynamic_shapes: bool = False,
        use_custom_kv_cache: bool = False,
        use_custom_sdpa: bool = False,
        skip_weight_data: bool = True,
    ):
        self.max_seq_len = max_seq_len
        self.enable_quantization = enable_quantization
        self.quant_config = quant_config or GgmlQuantConfig()
        self.preserve_dynamic_shapes = preserve_dynamic_shapes
        self.use_custom_kv_cache = use_custom_kv_cache
        self.use_custom_sdpa = use_custom_sdpa
        self.skip_weight_data = skip_weight_data


def _create_pytorch_model_from_gguf_metadata(
    analyzer: GGUFAnalyzer,
    max_seq_len: int = 128
) -> torch.nn.Module:
    """Create a PyTorch model from GGUF metadata without loading weights.

    Args:
        analyzer: GGUFAnalyzer instance with GGUF file metadata
        max_seq_len: Maximum sequence length for model configuration

    Returns:
        PyTorch model ready for export (without weights loaded)

    Raises:
        ValueError: If architecture is not supported
    """
    arch = analyzer.get_model_architecture()
    config_dict = analyzer.get_model_config()

    if arch == "qwen3":
        return _create_qwen3_model_from_config(config_dict, max_seq_len)
    elif arch == "llama":
        return _create_llama_model_from_config(config_dict, max_seq_len)
    else:
        raise ValueError(f"Unsupported architecture for PTE export: {arch}")


def _create_qwen3_model_from_config(config_dict: Dict[str, Any], max_seq_len: int) -> torch.nn.Module:
    """Create a Qwen3 PyTorch model from configuration dictionary."""

    # Create transformers config
    config = AutoConfig.for_model("qwen3", **{
        "hidden_size": config_dict["embedding_length"],
        "num_hidden_layers": config_dict["block_count"],
        "num_attention_heads": config_dict["attention_head_count"],
        "num_key_value_heads": config_dict["attention_head_count_kv"],
        "intermediate_size": config_dict["feed_forward_length"],
        "vocab_size": config_dict["vocab_size"],
        "rms_norm_eps": config_dict["attention_layer_norm_rms_epsilon"],
        "rope_theta": config_dict.get("rope_freq_base", 1000000.0),
        "max_position_embeddings": max_seq_len,
    })

    # Handle rope scaling
    if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
        config.rope_scaling["type"] = "default"

    # Create model without loading pretrained weights
    model = AutoModelForCausalLM.from_config(
        config,
        torch_dtype=torch.float32,
        attn_implementation="sdpa",
    )

    # Set generation config
    model.generation_config = GenerationConfig(
        use_cache=True,
        cache_implementation="static",
        max_length=max_seq_len,
        cache_config={"batch_size": 1, "max_cache_len": max_seq_len},
    )

    model.eval()
    return model


def _create_llama_model_from_config(config_dict: Dict[str, Any], max_seq_len: int) -> torch.nn.Module:
    """Create a Llama PyTorch model from configuration dictionary."""

    # Create transformers config for Llama
    config = AutoConfig.for_model("llama", **{
        "hidden_size": config_dict["embedding_length"],
        "num_hidden_layers": config_dict["block_count"],
        "num_attention_heads": config_dict["attention_head_count"],
        "num_key_value_heads": config_dict["attention_head_count_kv"],
        "intermediate_size": config_dict["feed_forward_length"],
        "vocab_size": config_dict["vocab_size"],
        "rms_norm_eps": config_dict["attention_layer_norm_rms_epsilon"],
        "rope_theta": config_dict.get("rope_freq_base", 10000.0),
        "max_position_embeddings": max_seq_len,
    })

    # Create model without loading pretrained weights
    model = AutoModelForCausalLM.from_config(
        config,
        torch_dtype=torch.float32,
        attn_implementation="sdpa",
    )

    # Set generation config
    model.generation_config = GenerationConfig(
        use_cache=True,
        cache_implementation="static",
        max_length=max_seq_len,
        cache_config={"batch_size": 1, "max_cache_len": max_seq_len},
    )

    model.eval()
    return model


def _apply_model_optimizations(model: torch.nn.Module, skip_weight_data: bool = True) -> int:
    """Apply ExecuTorch-specific optimizations to the model.

    Args:
        model: PyTorch model to optimize
        skip_weight_data: If True (GGUF external weights path), skip
            fold_rms_norm_weights since the GGUF provides original weights
            that haven't been folded.

    Returns:
        Number of optimizations applied
    """
    optimizations_count = 0

    # Import optimization passes
    try:
        from executorch_ggml.modules.rms_norm import swap_rms_norm

        # Fuse decomposed RMSNorm into single op (graph-only, no weight change)
        n_rms = swap_rms_norm(model)
        optimizations_count += n_rms

        # fold_rms_norm_weights modifies linear weight values (W *= norm_weight).
        # Only safe when weights are embedded in the PTE. When using external
        # GGUF weights, the GGUF provides the original unfolded weights, so
        # folding the graph would create a mismatch.
        if not skip_weight_data:
            from executorch_ggml.passes.fold_rms_norm_weights import fold_rms_norm_weights
            n_fold = fold_rms_norm_weights(model)
            optimizations_count += n_fold

    except ImportError:
        pass

    # Fuse parallel projections (after fold so folded weights get fused)
    try:
        from executorch_ggml.passes.fuse_projections import (
            fuse_gate_up_projections,
            fuse_qkv_projections,
        )

        n_qkv = fuse_qkv_projections(model)
        n_mlp = fuse_gate_up_projections(model)
        optimizations_count += n_qkv + n_mlp

    except ImportError:
        pass

    return optimizations_count


def _apply_graph_passes(ep, config_dict: Dict[str, Any]) -> None:
    """Apply graph-level passes after export."""

    try:
        from executorch_ggml.passes.fuse_rope_pass import fuse_rope_in_graph
        from executorch_ggml.passes.strip_gqa_expand_pass import strip_gqa_expand

        # RoPE fusion
        head_dim = config_dict["embedding_length"] // config_dict["attention_head_count"]
        freq_base = config_dict.get("rope_freq_base", 1000000.0)
        fuse_rope_in_graph(ep.graph_module, head_dim=head_dim, freq_base=freq_base)

        # GQA optimization
        strip_gqa_expand(ep.graph_module)

    except ImportError:
        pass

    # CSE: merge duplicate linears from fused projections
    try:
        from executorch_ggml.passes.cse_pass import eliminate_common_subexpressions

        eliminate_common_subexpressions(ep.graph_module)
    except ImportError:
        pass


def export_gguf_to_pte(
    gguf_path: str,
    output_pte_path: str,
    export_config: Optional[GGUFExportConfig] = None,
) -> str:
    """Export a GGUF file to a weight-less PTE file.

    This function analyzes a GGUF file, creates a PyTorch model from its metadata,
    and exports it to a PTE file with external weight references that match the
    original GGUF tensor names.

    Args:
        gguf_path: Path to the input GGUF file
        output_pte_path: Path for the output PTE file
        export_config: Configuration for the export process

    Returns:
        Path to the generated PTE file

    Raises:
        FileNotFoundError: If GGUF file doesn't exist
        ValueError: If architecture is not supported
    """
    if export_config is None:
        export_config = GGUFExportConfig()

    print(f"Exporting {gguf_path} -> {output_pte_path}")

    # Analyze GGUF file
    analyzer = GGUFAnalyzer(gguf_path)

    # Create weight mapper
    arch = analyzer.get_model_architecture()
    config_dict = analyzer.get_model_config()
    n_blocks = config_dict["block_count"]

    weight_mapper = WeightNameMapper(arch, n_blocks)

    # Create PyTorch model from metadata (no weights)
    model = _create_pytorch_model_from_gguf_metadata(analyzer, export_config.max_seq_len)

    # Apply model optimizations
    _apply_model_optimizations(model, skip_weight_data=True)

    # Export model to ExportedProgram
    exportable = CausalLMExportableModule(
        model,
        max_seq_len=export_config.max_seq_len,
        use_custom_kv_cache=export_config.use_custom_kv_cache,
        use_custom_sdpa=export_config.use_custom_sdpa,
        disable_dynamic_shapes=not export_config.preserve_dynamic_shapes,
    )
    ep = exportable.export()["model"]

    # Apply graph passes
    _apply_graph_passes(ep, config_dict)

    # Build GGUF weight map (PyTorch FQN -> GGUF tensor name).
    # Use the ExportedProgram's state_dict keys (which have the wrapper
    # module prefix like "model.model.") rather than model.named_parameters().
    pytorch_names = list(ep.state_dict.keys())
    gguf_names = analyzer.get_tensor_names()

    # Build PyTorch FQN -> GGUF name mapping
    gguf_weight_map = {}
    gguf_name_set = set(gguf_names)
    for pytorch_name in pytorch_names:
        gguf_name = weight_mapper.pytorch_to_gguf(pytorch_name)
        if gguf_name in gguf_name_set:
            gguf_weight_map[pytorch_name] = gguf_name

    # Handle tied weights (e.g. lm_head shares token_embd in Qwen3).
    for pytorch_name in pytorch_names:
        if pytorch_name not in gguf_weight_map:
            if "lm_head" in pytorch_name and "token_embd.weight" in gguf_name_set:
                gguf_weight_map[pytorch_name] = "token_embd.weight"

    # Lower to ExecuTorch with GGUF weight names as data_keys
    # Configure partitioner with GGUF weight map
    quant_cfg = export_config.quant_config if export_config.enable_quantization else None
    partitioner = GgmlPartitioner(
        quant_config=quant_cfg,
        gguf_weight_map=gguf_weight_map,
        skip_weight_data=export_config.skip_weight_data,
    )

    # Lower to edge with GGML delegation
    edge_mgr = to_edge_transform_and_lower(
        ep,
        partitioner=[partitioner],
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=True,
        ),
        transform_passes=[ReplaceCopyOpsPass(), RemoveGraphAssertsPass()],
        constant_methods=exportable.metadata,
    )

    # Convert to ExecuTorch
    et_program = edge_mgr.to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=True,
        )
    )

    # Save PTE file
    output_dir = os.path.dirname(output_pte_path)
    if output_dir:  # Only create directory if path contains a directory
        os.makedirs(output_dir, exist_ok=True)

    pte_bytes = et_program.buffer
    with open(output_pte_path, "wb") as f:
        f.write(pte_bytes)

    pte_size_mb = len(pte_bytes) / (1024 * 1024)
    print(f"Saved {output_pte_path} ({pte_size_mb:.1f} MB)")

    return output_pte_path