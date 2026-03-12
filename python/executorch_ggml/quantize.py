"""GGML weight quantization for the executorch-ggml backend.

Implements block-based quantization formats matching llama.cpp/ggml.
Weights are quantized at export time during GgmlBackend.preprocess(),
not through PyTorch's observer/fake-quantize workflow.
"""

import enum
import struct
from dataclasses import dataclass, field
from typing import Set

import numpy as np

# Q8_0 constants (from ggml-common.h)
QK8_0 = 32  # elements per block
BLOCK_Q8_0_BYTES = 34  # 2 (fp16 scale) + 32 (int8 quants)


class GgmlQuantType(enum.Enum):
    Q8_0 = "q8_0"
    # Q6_K = "q6_k"  # future
    # Q4_0 = "q4_0"  # future


@dataclass
class GgmlQuantConfig:
    """Configuration for GGML weight quantization at export time.

    quant_type: which GGML quantization format to use.
    skip_patterns: FQN substrings that should NOT be quantized.
    min_elements: minimum number of elements for a tensor to be quantized.
    """

    quant_type: GgmlQuantType = GgmlQuantType.Q8_0
    skip_patterns: Set[str] = field(
        default_factory=lambda: {"norm", "layernorm", "rmsnorm", "bias"}
    )
    min_elements: int = 1024


def _block_size_for_type(quant_type: GgmlQuantType) -> int:
    if quant_type == GgmlQuantType.Q8_0:
        return QK8_0
    raise ValueError(f"Unknown quant type: {quant_type}")


def quantize_tensor_q8_0(data_f32: np.ndarray) -> bytes:
    """Quantize a flat F32 array to Q8_0 block format.

    Matches the reference C implementation in ggml-quants.c:quantize_row_q8_0_ref.

    Each block of 32 floats is quantized as:
      d = max(|x|) / 127          (stored as float16, 2 bytes)
      qs[i] = round(x[i] / d)    (stored as int8, 32 bytes)
    Total: 34 bytes per block of 32 elements.

    Args:
        data_f32: Flat float32 array. Length must be a multiple of 32.

    Returns:
        Raw bytes in ggml block_q8_0 layout.
    """
    assert data_f32.dtype == np.float32
    k = data_f32.size
    assert k % QK8_0 == 0, f"tensor size {k} not divisible by {QK8_0}"
    nb = k // QK8_0

    blocks = data_f32.reshape(nb, QK8_0)

    # Per-block absolute max
    amax = np.max(np.abs(blocks), axis=1)  # (nb,)

    # Scale: d = amax / 127
    d = amax / 127.0

    # Inverse scale (avoid div-by-zero for all-zero blocks)
    d_safe = np.where(d == 0, 1.0, d)
    id_ = np.where(d != 0, 1.0 / d_safe, 0.0)

    # Quantize to int8: round(x * id)
    qs = np.clip(np.round(blocks * id_[:, None]), -128, 127).astype(np.int8)

    # Convert scale to fp16
    d_f16 = d.astype(np.float16)

    # Pack: interleave fp16 scale + int8 quants per block
    # Layout per block: [d_f16: 2 bytes][qs: 32 bytes] = 34 bytes
    out = bytearray(nb * BLOCK_Q8_0_BYTES)
    d_bytes = d_f16.view(np.uint8).reshape(nb, 2)
    qs_bytes = qs.view(np.uint8).reshape(nb, QK8_0)
    for i in range(nb):
        off = i * BLOCK_Q8_0_BYTES
        out[off : off + 2] = d_bytes[i].tobytes()
        out[off + 2 : off + BLOCK_Q8_0_BYTES] = qs_bytes[i].tobytes()
    return bytes(out)


def dequantize_q8_0(data: bytes, n_elements: int) -> np.ndarray:
    """Dequantize Q8_0 bytes back to float32. Useful for testing."""
    assert n_elements % QK8_0 == 0
    nb = n_elements // QK8_0
    assert len(data) == nb * BLOCK_Q8_0_BYTES

    result = np.empty(n_elements, dtype=np.float32)
    for i in range(nb):
        off = i * BLOCK_Q8_0_BYTES
        d = np.frombuffer(data[off : off + 2], dtype=np.float16)[0].astype(
            np.float32
        )
        qs = np.frombuffer(
            data[off + 2 : off + BLOCK_Q8_0_BYTES], dtype=np.int8
        ).astype(np.float32)
        result[i * QK8_0 : (i + 1) * QK8_0] = d * qs
    return result


def should_quantize(fqn: str, shape: tuple, dtype, numel: int, config: GgmlQuantConfig) -> bool:
    """Decide if a weight tensor should be quantized.

    Args:
        fqn: Fully-qualified parameter name (e.g. "model.layers.0.self_attn.q_proj.weight").
        shape: Tensor shape tuple.
        dtype: Tensor dtype (torch.dtype).
        numel: Number of elements.
        config: Quantization configuration.

    Returns:
        True if the tensor should be quantized.
    """
    import torch

    # Check skip patterns (case-insensitive substring match)
    fqn_lower = fqn.lower()
    for pattern in config.skip_patterns:
        if pattern in fqn_lower:
            return False

    # Only quantize 2D+ float tensors (linear/embedding weights)
    ndim = len(shape)
    if ndim < 2:
        return False
    if dtype not in (torch.float32, torch.float16, torch.bfloat16):
        return False
    if numel < config.min_elements:
        return False

    # ggml requires the innermost dim (last in PyTorch) to be a multiple of block_size
    inner_dim = shape[-1]
    block_size = _block_size_for_type(config.quant_type)
    if inner_dim % block_size != 0:
        return False

    return True


def quantize_tensor(data_f32: np.ndarray, quant_type: GgmlQuantType) -> bytes:
    """Quantize a flat F32 array to the specified GGML format."""
    if quant_type == GgmlQuantType.Q8_0:
        return quantize_tensor_q8_0(data_f32)
    raise ValueError(f"Unsupported quant type: {quant_type}")
