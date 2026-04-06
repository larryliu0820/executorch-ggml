"""Fuse parallel linear projections into shared-weight modules.

Replaces groups of linear layers that share the same input with modules
that share a single concatenated weight parameter. After torch.export + CSE,
this produces a single aten.linear + N aten.slice nodes instead of N separate
aten.linear nodes.

Two fusions:
  - QKV: q_proj + k_proj + v_proj → one matmul + 3 slices
  - Gate/Up: gate_proj + up_proj → one matmul + 2 slices

Usage (before torch.export, after fold_rms_norm_weights):
    from executorch_ggml.passes.fuse_projections import (
        fuse_qkv_projections,
        fuse_gate_up_projections,
    )
    n_qkv = fuse_qkv_projections(model)
    n_mlp = fuse_gate_up_projections(model)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SlicedLinear(nn.Module):
    """Linear that computes a full fused matmul and returns a slice.

    All _SlicedLinear instances in a fusion group share the SAME nn.Parameter.
    torch.export lifts shared parameters as a single placeholder, so the
    exported graph has N identical aten.linear calls with the same weight arg.
    A post-export CSE pass merges these into one linear + N slices.
    """

    def __init__(self, weight: nn.Parameter, start: int, end: int):
        super().__init__()
        self.weight = weight
        self._start = start
        self._end = end

    @property
    def in_features(self) -> int:
        return self.weight.shape[1]

    @property
    def out_features(self) -> int:
        return self._end - self._start

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)[..., self._start : self._end].contiguous()


def fuse_qkv_projections(model: nn.Module) -> int:
    """Fuse q_proj + k_proj + v_proj into shared-weight projections.

    Walks all modules looking for ones with q_proj, k_proj, v_proj attributes
    (all nn.Linear with the same in_features). Concatenates their weights into
    a single nn.Parameter shared across three _SlicedLinear replacements.

    Works with Qwen3, LLaMA, Mistral, Gemma, Phi, and any architecture using
    the q_proj/k_proj/v_proj naming convention.

    Must be called AFTER fold_rms_norm_weights (so folded weights get fused).
    Must be called BEFORE torch.export.

    Args:
        model: PyTorch model to modify in-place.

    Returns:
        Number of attention modules fused.
    """
    count = 0
    for _name, module in list(model.named_modules()):
        if not _has_linear_attrs(module, ("q_proj", "k_proj", "v_proj")):
            continue

        q_proj = module.q_proj
        k_proj = module.k_proj
        v_proj = module.v_proj

        if q_proj.in_features != k_proj.in_features:
            continue
        if q_proj.in_features != v_proj.in_features:
            continue

        q_size = q_proj.out_features
        k_size = k_proj.out_features
        v_size = v_proj.out_features

        fused_weight = nn.Parameter(
            torch.cat(
                [q_proj.weight.data, k_proj.weight.data, v_proj.weight.data],
                dim=0,
            )
        )

        module.q_proj = _SlicedLinear(fused_weight, 0, q_size)
        module.k_proj = _SlicedLinear(fused_weight, q_size, q_size + k_size)
        module.v_proj = _SlicedLinear(
            fused_weight, q_size + k_size, q_size + k_size + v_size
        )
        count += 1

    return count


def fuse_gate_up_projections(model: nn.Module) -> int:
    """Fuse gate_proj + up_proj into shared-weight projections.

    Same approach as QKV fusion but for the MLP gate/up projections.
    gate_proj and up_proj always have identical shapes, so the fused
    output is split evenly.

    Works with Qwen3, LLaMA, Mistral, and any architecture using
    the gate_proj/up_proj naming convention.

    Args:
        model: PyTorch model to modify in-place.

    Returns:
        Number of MLP modules fused.
    """
    count = 0
    for _name, module in list(model.named_modules()):
        if not _has_linear_attrs(module, ("gate_proj", "up_proj")):
            continue

        gate_proj = module.gate_proj
        up_proj = module.up_proj

        if gate_proj.in_features != up_proj.in_features:
            continue

        gate_size = gate_proj.out_features
        up_size = up_proj.out_features

        fused_weight = nn.Parameter(
            torch.cat([gate_proj.weight.data, up_proj.weight.data], dim=0)
        )

        module.gate_proj = _SlicedLinear(fused_weight, 0, gate_size)
        module.up_proj = _SlicedLinear(fused_weight, gate_size, gate_size + up_size)
        count += 1

    return count


def _has_linear_attrs(module: nn.Module, attr_names: tuple) -> bool:
    """Check that module has all named attributes and they are nn.Linear."""
    for name in attr_names:
        attr = getattr(module, name, None)
        if attr is None or not isinstance(attr, nn.Linear):
            return False
    return True
