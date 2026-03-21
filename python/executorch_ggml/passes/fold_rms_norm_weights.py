"""Fold RMSNorm weights into downstream linear projections.

rms_norm(x) * w  →  linear(result, W)
becomes:
rms_norm(x)      →  linear(result, W * w)

Eliminates the weight MUL op from the graph. Applied before export.

Usage:
    from executorch_ggml.passes.fold_rms_norm_weights import fold_rms_norm_weights
    n = fold_rms_norm_weights(model)
"""

import torch
import torch.nn as nn


def _clear_norm_weight(norm: nn.Module) -> None:
    """Remove weight from a FusedRMSNorm so aten.rms_norm emits weight=None."""
    # Replace the module's forward to pass weight=None
    eps = norm.eps
    normalized_shape = list(norm.weight.shape)
    # Delete the weight parameter so it doesn't appear in the graph
    del norm.weight
    norm.weight = None  # attribute exists but is not a Parameter
    norm.forward = lambda x: torch.ops.aten.rms_norm.default(
        x, normalized_shape, None, eps
    )


def fold_rms_norm_weights(model: nn.Module) -> int:
    """Fold RMSNorm weights into downstream linear layers.

    For each decoder layer:
      - input_layernorm.weight → folded into q_proj, k_proj, v_proj
      - post_attention_layernorm.weight → folded into gate_proj, up_proj

    Also folds the final model norm into lm_head if present.

    After folding, the norm weight is set to 1.0 so the MUL becomes identity
    (and aten.rms_norm with all-ones weight produces no MUL in the graph).

    Returns the number of norms folded.
    """
    count = 0

    # Find decoder layers
    layers = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers

    if layers is None:
        return 0

    for layer in layers:
        # input_layernorm → q_proj, k_proj, v_proj
        norm = getattr(layer, "input_layernorm", None)
        attn = getattr(layer, "self_attn", None)
        if norm is not None and attn is not None and hasattr(norm, "weight"):
            w = norm.weight.data  # [hidden_size]
            for proj_name in ("q_proj", "k_proj", "v_proj"):
                proj = getattr(attn, proj_name, None)
                if proj is not None and hasattr(proj, "weight"):
                    # proj.weight: [out_features, in_features]
                    # Multiply each row by w (broadcast over out_features)
                    proj.weight.data.mul_(w.unsqueeze(0))
            _clear_norm_weight(norm)
            count += 1

        # post_attention_layernorm → gate_proj, up_proj (in MLP)
        post_norm = getattr(layer, "post_attention_layernorm", None)
        mlp = getattr(layer, "mlp", None)
        if post_norm is not None and mlp is not None and hasattr(post_norm, "weight"):
            w = post_norm.weight.data
            for proj_name in ("gate_proj", "up_proj"):
                proj = getattr(mlp, proj_name, None)
                if proj is not None and hasattr(proj, "weight"):
                    proj.weight.data.mul_(w.unsqueeze(0))
            _clear_norm_weight(post_norm)
            count += 1

    # Final norm → lm_head
    final_norm = None
    lm_head = None
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        final_norm = model.model.norm
    elif hasattr(model, "norm"):
        final_norm = model.norm
    if hasattr(model, "lm_head"):
        lm_head = model.lm_head

    if (
        final_norm is not None
        and lm_head is not None
        and hasattr(final_norm, "weight")
        and hasattr(lm_head, "weight")
    ):
        w = final_norm.weight.data
        lm_head.weight.data.mul_(w.unsqueeze(0))
        _clear_norm_weight(final_norm)
        count += 1

    return count
