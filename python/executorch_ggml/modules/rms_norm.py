"""Module swap: replace decomposed RMSNorm with fused aten.rms_norm op.

Usage (before export):
    from executorch_ggml.modules.rms_norm import swap_rms_norm
    swap_rms_norm(model)

This replaces any Qwen3RMSNorm (or similar) modules with a version that
calls torch.ops.aten.rms_norm.default directly, preventing decomposition
during export. The ggml backend lowers this to a single ggml_rms_norm call.
"""

import torch
import torch.nn as nn


class FusedRMSNorm(nn.Module):
    """RMSNorm that calls the fused aten op instead of decomposing."""

    def __init__(self, weight: torch.Tensor, eps: float):
        super().__init__()
        self.weight = nn.Parameter(weight.clone())
        self.eps = eps
        self.normalized_shape = list(weight.shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.aten.rms_norm.default(
            x, self.normalized_shape, self.weight, self.eps
        )


def swap_rms_norm(model: nn.Module) -> int:
    """Replace all RMSNorm-like modules with FusedRMSNorm.

    Detects modules by checking for .weight + .variance_epsilon attributes
    (the standard HuggingFace RMSNorm pattern).

    Returns the number of modules swapped.
    """
    count = 0
    for name, module in list(model.named_modules()):
        if not (hasattr(module, "weight") and hasattr(module, "variance_epsilon")):
            continue
        fused = FusedRMSNorm(module.weight.data, module.variance_epsilon)
        # Navigate to parent and replace
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            parent = model.get_submodule(parts[0])
            setattr(parent, parts[1], fused)
        else:
            setattr(model, name, fused)
        count += 1
    return count
