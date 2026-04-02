"""GGUFModule runtime class for ExecuTorch GGML backend.

This module provides runtime loading of PTE files with external weight references
that are dynamically resolved from GGUF files, enabling memory-efficient inference
with separated graph structure and weight data.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch

# Add llama.cpp gguf-py to path for GGUF utilities
_gguf_py_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "third-party", "llama.cpp", "gguf-py"
)
if _gguf_py_path not in sys.path:
    sys.path.append(_gguf_py_path)

from gguf import GGUFReader

from executorch_ggml.gguf_analyzer import GGUFAnalyzer
from executorch_ggml.weight_mapping import WeightNameMapper


class GGUFTensor:
    """Wrapper for lazy-loaded GGUF tensor data."""

    def __init__(self, gguf_reader: GGUFReader, tensor_name: str):
        """Initialize GGUF tensor wrapper.

        Args:
            gguf_reader: GGUFReader instance
            tensor_name: Name of the tensor in GGUF file
        """
        self.gguf_reader = gguf_reader
        self.tensor_name = tensor_name
        self._tensor = None
        self._tensor_info = None

        # Find tensor info
        for tensor in gguf_reader.tensors:
            if tensor.name == tensor_name:
                self._tensor_info = tensor
                break

        if self._tensor_info is None:
            raise ValueError(f"Tensor '{tensor_name}' not found in GGUF file")

    @property
    def shape(self) -> Tuple[int, ...]:
        """Get tensor shape (GGUF format - reversed from PyTorch)."""
        return tuple(self._tensor_info.shape)

    @property
    def pytorch_shape(self) -> Tuple[int, ...]:
        """Get tensor shape in PyTorch format (reversed from GGUF)."""
        return tuple(reversed(self._tensor_info.shape))

    @property
    def dtype(self):
        """Get tensor data type."""
        return self._tensor_info.tensor_type

    def load(self) -> torch.Tensor:
        """Load tensor data as a raw-bytes torch.uint8 tensor.

        The GGML backend handles all formats (F32, F16, Q8_0, etc.) natively,
        so we always return the raw bytes without interpretation.
        """
        if self._tensor is None:
            self._tensor = torch.from_numpy(self._tensor_info.data.copy())
        return self._tensor

    def __repr__(self) -> str:
        return f"GGUFTensor('{self.tensor_name}', shape={self.pytorch_shape}, dtype={self.dtype})"


class GGUFModule:
    """Runtime module for loading PTE files with GGUF weight resolution.

    This class provides a PyTorch-like interface for models exported as
    weight-less PTE files with external GGUF weight references.
    """

    def __init__(self, pte_path: str, gguf_path: str):
        """Initialize GGUFModule with PTE graph and GGUF weights.

        Args:
            pte_path: Path to the PTE file (graph structure)
            gguf_path: Path to the GGUF file (weights)

        Raises:
            FileNotFoundError: If PTE or GGUF files don't exist
            ImportError: If ExecuTorch runtime is not available
        """
        if not os.path.exists(pte_path):
            raise FileNotFoundError(f"PTE file not found: {pte_path}")
        if not os.path.exists(gguf_path):
            raise FileNotFoundError(f"GGUF file not found: {gguf_path}")

        self.pte_path = pte_path
        self.gguf_path = gguf_path

        # Load PTE + GGUF via GGUFRuntime: the GGUF file provides the
        # NamedDataMap so the GGML backend resolves weights from it.
        import executorch_ggml as _ggml_pkg  # noqa: F401  # registers GgmlBackend
        from executorch_ggml._ggml_backend import GGUFRuntime
        self._runtime = GGUFRuntime(pte_path, os.path.abspath(gguf_path))
        self.et_program = self._runtime  # keep for API compat

        # Load GGUF file
        self.gguf_reader = GGUFReader(gguf_path, "r")
        self.gguf_analyzer = GGUFAnalyzer(gguf_path)

        # Initialize weight mapper
        arch = self.gguf_analyzer.get_model_architecture()
        config = self.gguf_analyzer.get_model_config()
        n_blocks = config["block_count"]
        self.weight_mapper = WeightNameMapper(arch, n_blocks)

        # Weight caching
        self.weight_cache: Dict[str, GGUFTensor] = {}

    def get_gguf_tensor(self, gguf_name: str) -> GGUFTensor:
        """Get a GGUF tensor by name with lazy loading.

        Args:
            gguf_name: Name of the tensor in GGUF file

        Returns:
            GGUFTensor wrapper for the tensor

        Raises:
            KeyError: If tensor not found in GGUF file
        """
        if gguf_name not in self.weight_cache:
            self.weight_cache[gguf_name] = GGUFTensor(self.gguf_reader, gguf_name)
        return self.weight_cache[gguf_name]

    def load_weight_tensor(self, gguf_name: str) -> torch.Tensor:
        """Load a weight tensor from GGUF file.

        Args:
            gguf_name: Name of the tensor in GGUF file

        Returns:
            PyTorch tensor with loaded data
        """
        gguf_tensor = self.get_gguf_tensor(gguf_name)
        return gguf_tensor.load()

    def list_gguf_tensors(self) -> List[str]:
        """List all tensor names in the GGUF file.

        Returns:
            List of tensor names
        """
        return self.gguf_analyzer.get_tensor_names()

    def forward(self, *inputs) -> Tuple[torch.Tensor, ...]:
        """Forward pass through the model.

        Args:
            inputs: Input tensors

        Returns:
            Output tensors
        """
        # GGUFRuntime.forward expects a list, returns numpy arrays.
        np_outputs = self._runtime.forward(list(inputs))
        return tuple(torch.from_numpy(arr) for arr in np_outputs)

    def __call__(self, *inputs) -> Tuple[torch.Tensor, ...]:
        """Make the module callable."""
        return self.forward(*inputs)

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        gguf_info = self.gguf_analyzer.get_file_info()
        config = self.gguf_analyzer.get_model_config()

        return {
            "pte_path": self.pte_path,
            "gguf_path": self.gguf_path,
            "pte_size_mb": os.path.getsize(self.pte_path) / (1024 * 1024),
            "gguf_size_mb": gguf_info["file_size_mb"],
            "architecture": gguf_info["architecture"],
            "model_config": config,
            "tensor_count": gguf_info["tensor_count"],
            "cached_tensors": len(self.weight_cache),
        }

    def print_info(self) -> None:
        """Print detailed information about the loaded model."""
        info = self.get_model_info()

        print(f"GGUFModule Information:")
        print(f"  PTE File: {info['pte_path']}")
        print(f"  PTE Size: {info['pte_size_mb']:.1f} MB")
        print(f"  GGUF File: {info['gguf_path']}")
        print(f"  GGUF Size: {info['gguf_size_mb']:.1f} MB")
        print(f"  Architecture: {info['architecture']}")
        print(f"  Tensor Count: {info['tensor_count']}")
        print(f"  Cached Tensors: {info['cached_tensors']}")

        print(f"\nModel Configuration:")
        for key, value in info['model_config'].items():
            print(f"  {key}: {value}")

    def validate_weight_references(self) -> Dict[str, Any]:
        """Validate that PTE external weight references match GGUF tensors.

        Returns:
            Dictionary with validation results
        """
        # TODO: Implement validation by inspecting PTE metadata
        # For now, return basic info
        gguf_tensors = self.list_gguf_tensors()

        return {
            "gguf_tensor_count": len(gguf_tensors),
            "weight_parameters": [name for name in gguf_tensors if self.weight_mapper.is_weight_parameter(name)],
            "validation_status": "partial",  # TODO: Implement full validation
            "gguf_tensors": gguf_tensors[:10],  # First 10 for inspection
        }

    def clear_weight_cache(self) -> None:
        """Clear the weight tensor cache to free memory."""
        self.weight_cache.clear()

    def preload_weights(self, tensor_names: Optional[List[str]] = None) -> None:
        """Preload weight tensors into cache.

        Args:
            tensor_names: Specific tensor names to load, or None for all weights
        """
        if tensor_names is None:
            # Load all weight tensors
            tensor_names = [
                name for name in self.list_gguf_tensors()
                if self.weight_mapper.is_weight_parameter(name)
            ]

        for name in tensor_names:
            try:
                self.get_gguf_tensor(name)  # This caches the tensor wrapper
            except Exception as e:
                import warnings
                warnings.warn(f"Could not preload {name}: {e}")