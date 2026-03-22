"""Fold encoder RMSNorm weights into downstream linear projections.

For each encoder layer:
  attention_norm.weight → folded into attention.wq, attention.wk, attention.wv
  ffn_norm.weight      → folded into feed_forward.w1, feed_forward.w3
  final norm           → folded into adapter.w_in (with downsample expansion)

Eliminates 65 MUL ops (32 layers * 2 norms + 1 final) from the encoder graph.

Usage (before export, after swap_encoder_rope):
    from executorch_ggml.passes.fold_encoder_rms_norm_weights import fold_encoder_rms_norm_weights
    n = fold_encoder_rms_norm_weights(model)
"""

import torch
import torch.nn as nn


def _clear_norm_weight(norm: nn.Module) -> None:
    """Remove norm weight so rms_norm emits no MUL op."""
    import torch.nn.functional as F
    dim = norm.dim
    eps = norm.eps
    del norm.weight
    norm.weight = None
    norm.forward = lambda x: F.rms_norm(x, (dim,), None, eps)


def fold_encoder_rms_norm_weights(model: nn.Module) -> int:
    """Fold encoder RMSNorm weights into downstream linear layers.

    Returns the number of norms folded.
    """
    encoder = model.encoder if hasattr(model, "encoder") else model
    adapter = model.adapter if hasattr(model, "adapter") else None

    if not hasattr(encoder, "layers"):
        return 0

    count = 0

    for layer in encoder.layers:
        # attention_norm → wq, wk, wv
        norm = getattr(layer, "attention_norm", None)
        attn = getattr(layer, "attention", None)
        if norm is not None and attn is not None and hasattr(norm, "weight"):
            w = norm.weight.data  # [enc_dim]
            for proj_name in ("wq", "wk", "wv"):
                proj = getattr(attn, proj_name, None)
                if proj is not None and hasattr(proj, "weight"):
                    proj.weight.data.mul_(w.unsqueeze(0))
            _clear_norm_weight(norm)
            count += 1

        # ffn_norm → w1, w3
        ffn_norm = getattr(layer, "ffn_norm", None)
        ff = getattr(layer, "feed_forward", None)
        if ffn_norm is not None and ff is not None and hasattr(ffn_norm, "weight"):
            w = ffn_norm.weight.data
            for proj_name in ("w1", "w3"):
                proj = getattr(ff, proj_name, None)
                if proj is not None and hasattr(proj, "weight"):
                    proj.weight.data.mul_(w.unsqueeze(0))
            _clear_norm_weight(ffn_norm)
            count += 1

    # Final encoder norm → adapter.w_in
    # After norm, output is [B, T, enc_dim]. Then reshaped to [B, T//ds, enc_dim*ds].
    # The norm weight [enc_dim] repeats ds times in the reshaped dimension.
    final_norm = getattr(encoder, "norm", None)
    ds = getattr(model.config, "downsample_factor", None) if hasattr(model, "config") else None
    if (
        final_norm is not None
        and adapter is not None
        and ds is not None
        and hasattr(final_norm, "weight")
        and hasattr(adapter, "w_in")
    ):
        w = final_norm.weight.data  # [enc_dim]
        w_expanded = w.repeat(ds)  # [enc_dim * ds]
        adapter.w_in.weight.data.mul_(w_expanded.unsqueeze(0))
        _clear_norm_weight(final_norm)
        count += 1

    return count
