"""GGUF file analyzer for ExecuTorch GGML backend.

This module provides utilities to analyze GGUF files and extract model metadata,
architecture information, and tensor names for use in ExecuTorch export pipeline.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List

# Add llama.cpp gguf-py to path for GGUF utilities
_gguf_py_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "third-party", "llama.cpp", "gguf-py"
)
if _gguf_py_path not in sys.path:
    sys.path.append(_gguf_py_path)

from gguf import GGUFReader
from gguf.constants import MODEL_ARCH


class GGUFAnalyzer:
    """Analyzer for GGUF files that extracts model metadata and architecture info.

    This class wraps the GGUFReader to provide convenient access to model
    information needed for ExecuTorch export, including architecture detection,
    model configuration extraction, and tensor name enumeration.
    """

    def __init__(self, gguf_path: str):
        """Initialize analyzer with a GGUF file.

        Args:
            gguf_path: Path to the GGUF file to analyze

        Raises:
            FileNotFoundError: If the GGUF file doesn't exist
            ValueError: If the file is not a valid GGUF file
        """
        if not os.path.exists(gguf_path):
            raise FileNotFoundError(f"GGUF file not found: {gguf_path}")

        self.gguf_path = gguf_path
        self.reader = GGUFReader(gguf_path, "r")
        self.metadata = self._extract_metadata()

    def _extract_metadata(self) -> Dict[str, Any]:
        """Extract all metadata fields from the GGUF file."""
        metadata = {}
        for field in self.reader.fields.values():
            try:
                metadata[field.name] = field.contents()
            except Exception as e:
                # Some fields might not be readable - skip them
                print(f"Warning: Could not read field '{field.name}': {e}")
        return metadata

    def get_model_architecture(self) -> str:
        """Get the model architecture string.

        Returns:
            Architecture string (e.g., "qwen3", "llama", etc.)

        Raises:
            ValueError: If architecture is not found in metadata
        """
        arch = self.metadata.get("general.architecture")
        if arch is None:
            raise ValueError("Model architecture not found in GGUF metadata")
        return arch

    def get_model_arch_enum(self) -> MODEL_ARCH:
        """Get the MODEL_ARCH enum value for this model.

        Returns:
            MODEL_ARCH enum value

        Raises:
            ValueError: If architecture is not supported
        """
        arch_str = self.get_model_architecture()

        # Find the corresponding MODEL_ARCH enum
        for arch_enum in MODEL_ARCH:
            if arch_enum.name.lower() == arch_str.upper():
                return arch_enum

        # Try alternative matching
        arch_mapping = {
            "qwen3": MODEL_ARCH.QWEN3,
            "llama": MODEL_ARCH.LLAMA,
            "qwen2": MODEL_ARCH.QWEN2,
            "mistral": MODEL_ARCH.MISTRAL,
        }

        if arch_str in arch_mapping:
            return arch_mapping[arch_str]

        raise ValueError(f"Unsupported architecture: {arch_str}")

    def get_tensor_names(self) -> List[str]:
        """Get all tensor names in the GGUF file.

        Returns:
            List of tensor names
        """
        return [tensor.name for tensor in self.reader.tensors]

    def get_tensor_count(self) -> int:
        """Get the number of tensors in the GGUF file.

        Returns:
            Number of tensors
        """
        return len(self.reader.tensors)

    def get_model_config(self) -> Dict[str, Any]:
        """Extract model configuration parameters for the detected architecture.

        Returns:
            Dictionary containing model configuration parameters

        Raises:
            ValueError: If required configuration parameters are missing
        """
        arch = self.get_model_architecture()

        if arch == "qwen3":
            return self._get_qwen3_config()
        elif arch == "llama":
            return self._get_llama_config()
        else:
            # Generic configuration extraction
            return self._get_generic_config(arch)

    def _get_qwen3_config(self) -> Dict[str, Any]:
        """Extract Qwen3-specific configuration parameters."""
        config = {}

        # Required Qwen3 parameters
        required_params = {
            "embedding_length": "qwen3.embedding_length",
            "block_count": "qwen3.block_count",
            "feed_forward_length": "qwen3.feed_forward_length",
            "attention_head_count": "qwen3.attention.head_count",
            "attention_head_count_kv": "qwen3.attention.head_count_kv",
            "attention_layer_norm_rms_epsilon": "qwen3.attention.layer_norm_rms_epsilon",
        }

        for param_name, metadata_key in required_params.items():
            value = self.metadata.get(metadata_key)
            if value is None:
                raise ValueError(f"Missing required Qwen3 parameter: {metadata_key}")
            config[param_name] = value

        # Optional parameters with defaults
        optional_params = {
            "rope_freq_base": ("qwen3.rope.freq_base", 1000000.0),
            "rope_dimension_count": ("qwen3.rope.dimension_count", None),
        }

        for param_name, (metadata_key, default_value) in optional_params.items():
            config[param_name] = self.metadata.get(metadata_key, default_value)

        # Extract vocabulary info
        tokens = self.metadata.get("tokenizer.ggml.tokens")
        if tokens is not None:
            config["vocab_size"] = len(tokens)
        else:
            # Fallback - try to infer from tensors
            config["vocab_size"] = self._infer_vocab_size()

        return config

    def _get_llama_config(self) -> Dict[str, Any]:
        """Extract Llama-specific configuration parameters."""
        config = {}

        # Required Llama parameters
        required_params = {
            "embedding_length": "llama.embedding_length",
            "block_count": "llama.block_count",
            "feed_forward_length": "llama.feed_forward_length",
            "attention_head_count": "llama.attention.head_count",
            "attention_head_count_kv": "llama.attention.head_count_kv",
            "attention_layer_norm_rms_epsilon": "llama.attention.layer_norm_rms_epsilon",
        }

        for param_name, metadata_key in required_params.items():
            value = self.metadata.get(metadata_key)
            if value is None:
                raise ValueError(f"Missing required Llama parameter: {metadata_key}")
            config[param_name] = value

        # Optional parameters
        config["rope_freq_base"] = self.metadata.get("llama.rope.freq_base", 10000.0)

        # Vocabulary
        tokens = self.metadata.get("tokenizer.ggml.tokens")
        if tokens is not None:
            config["vocab_size"] = len(tokens)
        else:
            config["vocab_size"] = self._infer_vocab_size()

        return config

    def _get_generic_config(self, arch: str) -> Dict[str, Any]:
        """Extract generic configuration parameters for any architecture."""
        config = {"architecture": arch}

        # Try common parameter names
        common_params = [
            f"{arch}.embedding_length",
            f"{arch}.block_count",
            f"{arch}.feed_forward_length",
            f"{arch}.attention.head_count",
            f"{arch}.attention.head_count_kv",
            f"{arch}.attention.layer_norm_rms_epsilon",
            f"{arch}.rope.freq_base",
        ]

        for param_key in common_params:
            value = self.metadata.get(param_key)
            if value is not None:
                # Convert key to standard format (remove architecture prefix)
                param_name = param_key.replace(f"{arch}.", "").replace("attention.", "")
                config[param_name] = value

        # Vocabulary
        tokens = self.metadata.get("tokenizer.ggml.tokens")
        if tokens is not None:
            config["vocab_size"] = len(tokens)
        else:
            config["vocab_size"] = self._infer_vocab_size()

        return config

    def _infer_vocab_size(self) -> int:
        """Infer vocabulary size from token embedding tensor shape."""
        # Look for token embedding tensor
        for tensor in self.reader.tensors:
            if "token_embd" in tensor.name or "tok_embeddings" in tensor.name:
                # Vocab size is typically the first dimension
                return tensor.shape[0] if tensor.shape else 32000

        # Default fallback
        return 32000

    def get_file_info(self) -> Dict[str, Any]:
        """Get basic file information.

        Returns:
            Dictionary with file size, tensor count, and architecture
        """
        file_size = os.path.getsize(self.gguf_path)

        return {
            "file_path": self.gguf_path,
            "file_size_bytes": file_size,
            "file_size_mb": file_size / (1024 * 1024),
            "architecture": self.get_model_architecture(),
            "tensor_count": self.get_tensor_count(),
            "gguf_version": getattr(self.reader, "version", "unknown"),
        }

    def print_summary(self) -> None:
        """Print a summary of the GGUF file contents."""
        info = self.get_file_info()
        config = self.get_model_config()

        print(f"GGUF File: {info['file_path']}")
        print(f"Size: {info['file_size_mb']:.1f} MB")
        print(f"Architecture: {info['architecture']}")
        print(f"GGUF Version: {info['gguf_version']}")
        print(f"Tensors: {info['tensor_count']}")

        print("\nModel Configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")

        print(f"\nFirst 10 tensors:")
        tensor_names = self.get_tensor_names()
        for i, name in enumerate(tensor_names[:10]):
            print(f"  {i+1}. {name}")
        if len(tensor_names) > 10:
            print(f"  ... and {len(tensor_names) - 10} more")