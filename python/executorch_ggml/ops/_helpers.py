"""Shared helper functions for ggml backend operator lowering."""

from typing import List

import torch

from executorch_ggml.serialize import (
    TYPE_F32,
    TYPE_F16,
    TYPE_I64,
    TYPE_I32,
    TYPE_BOOL,
    TYPE_BF16,
    TYPE_Q8_0,
    TYPE_Q4_0,
)


def _concrete_int(s) -> int:
    """Convert a SymInt (or plain int) to a concrete int without installing
    guards on the shape environment.  This avoids mutating
    ``shape_env.var_to_range`` during ``preprocess``."""
    if isinstance(s, int):
        return s
    # FX Node — resolve from FakeTensor metadata (scalar_tensor output).
    if isinstance(s, torch.fx.Node):
        fv = s.meta.get("val")
        if fv is not None and hasattr(fv, "item"):
            return int(fv.item())
        # Fall back to a large sentinel for "end" values
        return 2**62
    # torch.SymInt — read the hint (concrete value used during tracing)
    # without creating a guard that narrows var_to_range.
    return s.node.hint


def _resolve_shape(fake_val) -> List[int]:
    """Get a concrete integer shape list from a FakeTensor without guarding."""
    if fake_val is None or not hasattr(fake_val, "shape"):
        return []
    return [_concrete_int(s) for s in fake_val.shape]


def _type_elem_size(ir_type: int) -> int:
    """Return bytes per element for an IR tensor type."""
    return {
        TYPE_F32: 4, TYPE_F16: 2, TYPE_I64: 8, TYPE_I32: 4,
        TYPE_BOOL: 4, TYPE_BF16: 2, TYPE_Q8_0: 1, TYPE_Q4_0: 1,
    }.get(ir_type, 4)


def _pytorch_shape_to_ggml_ne(shape: List[int]) -> List[int]:
    """PyTorch [d0, d1, ..., dn] → ggml ne [dn, ..., d1, d0], padded/collapsed to 4D.

    For >4D tensors, collapse leading dimensions: [a,b,c,d,e] → [a*b, c, d, e] (reversed).
    """
    if len(shape) <= 4:
        ne = list(reversed(shape))
        while len(ne) < 4:
            ne.append(1)
        return ne[:4]
    else:
        leading_prod = 1
        for d in shape[:-4]:
            leading_prod *= d
        ne = [leading_prod] + list(shape[-4:])
        return list(reversed(ne))


def _torch_dtype_to_ir_type(dtype: torch.dtype) -> int:
    if dtype == torch.float16:
        return TYPE_F16
    if dtype == torch.float32:
        return TYPE_F32
    if dtype == torch.bfloat16:
        return TYPE_BF16
    if dtype == torch.int64:
        return TYPE_I64
    if dtype == torch.int32:
        return TYPE_I32
    if dtype == torch.bool:
        return TYPE_BOOL
    return TYPE_F32
