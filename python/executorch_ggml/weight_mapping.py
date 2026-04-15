"""Weight name mapping utilities for GGUF-to-PTE conversion.

This module provides bidirectional mapping between PyTorch Fully Qualified Names (FQNs)
and GGUF tensor names, enabling seamless weight reference conversion during export
and runtime loading.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Dict, List, Optional, Tuple

# Add llama.cpp gguf-py to path for GGUF utilities
_gguf_py_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "third-party", "llama.cpp", "gguf-py"
)
if _gguf_py_path not in sys.path:
    sys.path.append(_gguf_py_path)

from gguf.constants import MODEL_ARCH
from gguf.tensor_mapping import get_tensor_name_map


class WeightNameMapper:
    """Bidirectional mapper between PyTorch FQNs and GGUF tensor names.

    This class handles the conversion between PyTorch parameter names (used in
    ExecuTorch models) and GGUF tensor names (used in GGUF files) for various
    model architectures.
    """

    # Architecture-specific name mappings.
    # Each tuple is (gguf_pattern, pytorch_pattern).
    # Order matters: more specific patterns MUST come before less specific ones
    # to avoid partial matches (e.g. attn_q_norm before attn_q).
    _ARCH_MAPPINGS = {
        "qwen3": [
            # Global names (no block prefix)
            # CausalLMExportableModule wraps the HF model, adding an extra
            # "model." prefix (model.model.embed_tokens vs model.embed_tokens).
            ("token_embd", "model.model.embed_tokens"),
            ("output_norm", "model.model.norm"),

            # Block prefix: blk.N -> model.model.layers.N
            ("blk.", "model.model.layers."),

            # Attention Q/K norms (BEFORE generic attn_q/attn_k!)
            ("attn_q_norm", "self_attn.q_norm"),
            ("attn_k_norm", "self_attn.k_norm"),

            # Attention projections (order: longest match first)
            ("attn_output", "self_attn.o_proj"),
            ("attn_q", "self_attn.q_proj"),
            ("attn_k", "self_attn.k_proj"),
            ("attn_v", "self_attn.v_proj"),

            # Norms
            ("attn_norm", "input_layernorm"),
            ("ffn_norm", "post_attention_layernorm"),

            # Feed forward
            ("ffn_gate", "mlp.gate_proj"),
            ("ffn_up", "mlp.up_proj"),
            ("ffn_down", "mlp.down_proj"),
        ],
        "llama": [
            # Global names
            ("token_embd", "model.embed_tokens"),
            ("output_norm", "model.norm"),

            # Block prefix
            ("blk.", "model.layers."),

            # Attention projections (longest match first)
            ("attn_output", "self_attn.o_proj"),
            ("attn_q", "self_attn.q_proj"),
            ("attn_k", "self_attn.k_proj"),
            ("attn_v", "self_attn.v_proj"),

            # Norms
            ("attn_norm", "input_layernorm"),
            ("ffn_norm", "post_attention_layernorm"),

            # Feed forward
            ("ffn_gate", "mlp.gate_proj"),
            ("ffn_up", "mlp.up_proj"),
            ("ffn_down", "mlp.down_proj"),
        ],
        "qwen35moe": [
            # Global names (direct export — no model.model. prefix)
            ("token_embd", "embed_tokens"),
            # Note: output_norm -> norm is handled specially in pytorch_to_gguf
            # to avoid matching attn.norm. Only matches the final "norm.weight".
            ("output", "lm_head"),

            # Block prefix: blk.N -> layers.N
            ("blk.", "layers."),

            # Attention norms
            ("attn_norm", "ln_1"),
            ("post_attention_norm", "ln_2"),

            # SSM / linear attention (GatedDeltaNet layers)
            ("ssm_a", "attn.A_log"),
            ("ssm_dt.bias", "attn.dt_bias"),
            ("ssm_conv1d", "attn.conv1d"),
            ("ssm_norm", "attn.norm"),
            ("ssm_out", "attn.out_proj"),
            # Note: attn.in_proj is a fused projection that concatenates
            # multiple GGUF tensors (ssm_alpha, ssm_beta, attn_qkv).
            # The C++ runtime must concatenate them at load time.
            # For now, map to ssm_alpha as primary data_key.
            ("ssm_alpha", "attn.in_proj"),

            # Full attention layers (every 4th layer)
            ("attn_qkv", "attn.in_proj"),     # fused QKV + gate
            ("attn_gate", "attn.gate_proj"),
            ("attn_output", "attn.out_proj"),
            ("attn_q_norm", "attn.q_norm"),
            ("attn_k_norm", "attn.k_norm"),

            # MoE expert weights (stacked 3D tensors)
            # w1_weight = concat(gate, up) — mapped to gate_exps for data_key,
            # runtime concatenates gate_exps + up_exps
            ("ffn_gate_exps.weight", "mlp.experts.w1_weight"),
            ("ffn_down_exps.weight", "mlp.experts.w2_weight"),

            # MoE router
            ("ffn_gate_inp", "mlp.gate"),

            # Shared expert (unfused gate/up)
            ("ffn_gate_shexp", "mlp.shared_expert.gate_proj"),
            ("ffn_up_shexp", "mlp.shared_expert.up_proj"),
            ("ffn_down_shexp", "mlp.shared_expert.down_proj"),
            ("ffn_gate_inp_shexp", "mlp.shared_expert_gate"),
        ],
    }

    def __init__(self, arch: str, n_blocks: int):
        """Initialize the weight name mapper.

        Args:
            arch: Model architecture string (e.g., "qwen3", "llama")
            n_blocks: Number of transformer blocks in the model
        """
        self.arch = arch
        self.n_blocks = n_blocks

        # Get architecture-specific mappings
        if arch in self._ARCH_MAPPINGS:
            self.mappings = self._ARCH_MAPPINGS[arch]
        else:
            # Use llama as fallback for unknown architectures
            self.mappings = self._ARCH_MAPPINGS.get("llama", [])

        # Build reverse mapping (PyTorch -> GGUF)
        self.pytorch_to_gguf_map = {}
        self.gguf_to_pytorch_map = {}
        self._build_bidirectional_mappings()

    def _build_bidirectional_mappings(self) -> None:
        """Build bidirectional mapping tables from the replacement rules."""

        # For each replacement pair, create both directions
        for gguf_pattern, pytorch_pattern in self.mappings:
            self.gguf_to_pytorch_map[gguf_pattern] = pytorch_pattern
            self.pytorch_to_gguf_map[pytorch_pattern] = gguf_pattern

    def pytorch_to_gguf(self, pytorch_name: str) -> str:
        """Convert a PyTorch FQN to GGUF tensor name."""
        # Use explicit regex mapping for qwen35moe (string replace is too fragile)
        if self.arch in ("qwen35moe", "qwen3_5_moe"):
            return self._qwen35moe_pytorch_to_gguf(pytorch_name)

        result = pytorch_name
        for gguf_pattern, pytorch_pattern in self.mappings:
            result = result.replace(pytorch_pattern, gguf_pattern)
        return result

    def _qwen35moe_pytorch_to_gguf(self, name: str) -> str:
        """Explicit PyTorch→GGUF mapping for Qwen3.5 MoE."""
        # Global tensors
        if name == "embed_tokens.weight":
            return "token_embd.weight"
        if name == "norm.weight":
            return "output_norm.weight"
        if name == "lm_head.weight":
            return "output.weight"

        # Block tensors: layers.N.xxx -> blk.N.yyy
        m = re.match(r"layers\.(\d+)\.(.*)", name)
        if not m:
            return name
        blk = m.group(1)
        rest = m.group(2)

        # Norms
        _map = {
            "ln_1.weight": "attn_norm.weight",
            "ln_2.weight": "post_attention_norm.weight",
            # SSM / GatedDeltaNet (unfused projections)
            "attn.A_log": "ssm_a",
            "attn.dt_bias": "ssm_dt.bias",
            "attn.conv1d.weight": "ssm_conv1d.weight",
            "attn.norm.weight": "ssm_norm.weight",
            "attn.out_proj.weight": "ssm_out.weight",
            "attn._in_proj_qkv.weight": "attn_qkv.weight",
            "attn._in_proj_z.weight": "attn_gate.weight",   # z/output gate
            "attn._in_proj_b.weight": "ssm_beta.weight",    # beta projection
            "attn._in_proj_a.weight": "ssm_alpha.weight",   # alpha/time-step projection
            # Full attention (every 4th layer)
            "attn.q_norm.weight": "attn_q_norm.weight",
            "attn.k_norm.weight": "attn_k_norm.weight",
            "attn.gate_proj.weight": "attn_gate.weight",
            # MoE experts (unfused gate/up)
            "mlp.experts.gate_weight": "ffn_gate_exps.weight",
            "mlp.experts.up_weight": "ffn_up_exps.weight",
            "mlp.experts.down_weight": "ffn_down_exps.weight",
            "mlp.gate.weight": "ffn_gate_inp.weight",
            # Shared expert (unfused gate/up)
            "mlp.shared_expert.gate_proj.weight": "ffn_gate_shexp.weight",
            "mlp.shared_expert.up_proj.weight": "ffn_up_shexp.weight",
            "mlp.shared_expert.down_proj.weight": "ffn_down_shexp.weight",
            "mlp.shared_expert_gate.weight": "ffn_gate_inp_shexp.weight",
        }
        gguf_rest = _map.get(rest)
        if gguf_rest:
            return f"blk.{blk}.{gguf_rest}"
        return name  # unmapped (runtime state buffers etc.)

    def gguf_to_pytorch(self, gguf_name: str) -> str:
        """Convert a GGUF tensor name to PyTorch FQN.

        Args:
            gguf_name: GGUF tensor name (e.g., "blk.0.attn_q.weight")

        Returns:
            PyTorch parameter name (e.g., "model.layers.0.self_attn.q_proj.weight")
        """
        result = gguf_name

        # Apply GGUF -> PyTorch transformations in FORWARD order.
        # Mappings are ordered with specific patterns before generic ones
        # (e.g. attn_q_norm before attn_q), so forward order is correct.
        for gguf_pattern, pytorch_pattern in self.mappings:
            result = result.replace(gguf_pattern, pytorch_pattern)

        return result

    def is_weight_parameter(self, node_name: str) -> bool:
        """Check if a node name represents a weight parameter.

        This function determines if a given node in the computation graph
        represents a model parameter that should be externalized to GGUF.

        Args:
            node_name: Name of the graph node

        Returns:
            True if the node represents a weight parameter
        """
        # Common weight parameter patterns
        weight_patterns = [
            r"\.weight$",           # Parameters ending with .weight
            r"\.bias$",             # Parameters ending with .bias
            r"tok_embeddings$",     # Token embeddings
            r"norm\.weight$",       # Normalization weights
            r"attention\.w[qkvo]",  # Attention projections
            r"feed_forward\.w[123]", # Feed-forward weights
        ]

        for pattern in weight_patterns:
            if re.search(pattern, node_name):
                return True

        return False

    def get_gguf_tag_generator(self):
        """Get a tag generator function for use with ExecutorchBackendConfig.

        Returns:
            Function that takes a torch.fx.Node and returns GGUF tensor name or None
        """
        def gguf_tag_generator(node) -> Optional[str]:
            """Generate GGUF tensor name tag for external constants pass."""
            # Check if this is a weight parameter
            if hasattr(node, 'name') and self.is_weight_parameter(node.name):
                return self.pytorch_to_gguf(node.name)
            return None

        return gguf_tag_generator

    def validate_mapping(self, pytorch_names: List[str], gguf_names: List[str]) -> Dict[str, List[str]]:
        """Validate bidirectional mapping between PyTorch and GGUF names.

        Args:
            pytorch_names: List of PyTorch parameter names from model
            gguf_names: List of tensor names from GGUF file

        Returns:
            Dictionary with validation results including missing mappings
        """
        results = {
            "pytorch_to_gguf_missing": [],
            "gguf_to_pytorch_missing": [],
            "successful_mappings": [],
            "pytorch_unmapped": [],
            "gguf_unmapped": []
        }

        # Check PyTorch -> GGUF mappings
        for pytorch_name in pytorch_names:
            if self.is_weight_parameter(pytorch_name):
                gguf_name = self.pytorch_to_gguf(pytorch_name)
                if gguf_name in gguf_names:
                    results["successful_mappings"].append((pytorch_name, gguf_name))
                else:
                    results["pytorch_to_gguf_missing"].append((pytorch_name, gguf_name))
            else:
                results["pytorch_unmapped"].append(pytorch_name)

        # Check for unmapped GGUF tensors
        mapped_gguf_names = set(pair[1] for pair in results["successful_mappings"])
        for gguf_name in gguf_names:
            if gguf_name not in mapped_gguf_names:
                results["gguf_unmapped"].append(gguf_name)

        return results

    def print_mapping_summary(self, validation_results: Dict[str, List]) -> None:
        """Print a summary of the mapping validation results."""

        successful = len(validation_results["successful_mappings"])
        missing_pytorch = len(validation_results["pytorch_to_gguf_missing"])
        missing_gguf = len(validation_results["gguf_to_pytorch_missing"])
        unmapped_pytorch = len(validation_results["pytorch_unmapped"])
        unmapped_gguf = len(validation_results["gguf_unmapped"])

        print(f"Weight Name Mapping Summary ({self.arch}):")
        print(f"  Successful mappings: {successful}")
        print(f"  PyTorch parameters without GGUF match: {missing_pytorch}")
        print(f"  GGUF tensors without PyTorch match: {missing_gguf}")
        print(f"  Unmapped PyTorch parameters: {unmapped_pytorch}")
        print(f"  Unmapped GGUF tensors: {unmapped_gguf}")

        if missing_pytorch > 0:
            print(f"\n  Missing GGUF tensors:")
            for pytorch_name, expected_gguf in validation_results["pytorch_to_gguf_missing"][:5]:
                print(f"    {pytorch_name} -> {expected_gguf} (missing)")
            if missing_pytorch > 5:
                print(f"    ... and {missing_pytorch - 5} more")

        if unmapped_gguf > 0:
            print(f"\n  Unmapped GGUF tensors:")
            for gguf_name in validation_results["gguf_unmapped"][:5]:
                print(f"    {gguf_name}")
            if unmapped_gguf > 5:
                print(f"    ... and {unmapped_gguf - 5} more")


class MultiArchWeightMapper:
    """Multi-architecture weight name mapper supporting multiple model types."""

    def __init__(self):
        """Initialize multi-architecture mapper."""
        self.mappers = {}

    def get_mapper(self, arch: str, n_blocks: int) -> WeightNameMapper:
        """Get or create a mapper for a specific architecture.

        Args:
            arch: Model architecture string
            n_blocks: Number of transformer blocks

        Returns:
            WeightNameMapper instance for the architecture
        """
        key = f"{arch}_{n_blocks}"
        if key not in self.mappers:
            self.mappers[key] = WeightNameMapper(arch, n_blocks)
        return self.mappers[key]

    def detect_architecture_from_tensor_names(self, tensor_names: List[str]) -> str:
        """Detect model architecture from GGUF tensor names.

        Args:
            tensor_names: List of tensor names from GGUF file

        Returns:
            Detected architecture string

        Raises:
            ValueError: If architecture cannot be detected
        """
        # Architecture detection heuristics based on tensor name patterns
        tensor_set = set(tensor_names)

        # Look for architecture-specific patterns
        if any("token_embd" in name for name in tensor_names):
            # This is the GGUF naming convention
            if any("blk." in name and "attn_" in name for name in tensor_names):
                # Generic transformer architecture - could be Llama, Qwen, etc.
                # Try to be more specific
                return "llama"  # Default to llama for now

        # If we can't detect, default to llama
        return "llama"

    def infer_block_count(self, tensor_names: List[str]) -> int:
        """Infer the number of transformer blocks from tensor names.

        Args:
            tensor_names: List of tensor names from GGUF file

        Returns:
            Number of blocks detected
        """
        max_block = -1

        # Look for block indices in tensor names
        for name in tensor_names:
            # Match patterns like "blk.5.attn_q.weight"
            match = re.search(r'blk\.(\d+)\.', name)
            if match:
                block_idx = int(match.group(1))
                max_block = max(max_block, block_idx)

        # Return block count (max index + 1), default to 32 if none found
        return max_block + 1 if max_block >= 0 else 32