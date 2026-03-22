"""Fold decoder attention_norm weights into downstream linear projections.

For each decoder layer:
  attention_norm.weight → folded into attention.wq, attention.wk, attention.wv

Does NOT fold ffn_norm (adaptive scale from ada_rms_norm_t_cond prevents it).
Does NOT fold final norm (output weights are tied with embeddings).

Eliminates 26 MUL ops from the decoder graph.

Usage (before export, after swap_decoder_rope):
    from executorch_ggml.passes.fold_decoder_rms_norm_weights import fold_decoder_rms_norm_weights
    n = fold_decoder_rms_norm_weights(model)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def fold_decoder_rms_norm_weights(model: nn.Module) -> int:
    """Fold decoder attention_norm weights into wq/wk/wv.

    Returns the number of norms folded.
    """
    decoder = model.decoder if hasattr(model, "decoder") else model
    if not hasattr(decoder, "layers"):
        return 0

    count = 0
    for layer in decoder.layers:
        norm = getattr(layer, "attention_norm", None)
        attn = getattr(layer, "attention", None)
        if norm is None or attn is None:
            continue
        if not hasattr(norm, "weight") or norm.weight is None:
            continue

        w = norm.weight.data  # [dim]
        for proj_name in ("wq", "wk", "wv"):
            proj = getattr(attn, proj_name, None)
            if proj is not None and hasattr(proj, "weight"):
                proj.weight.data.mul_(w.unsqueeze(0))

        # Clear norm weight
        dim = norm.dim
        eps = norm.eps
        del norm.weight
        norm.weight = None
        norm.forward = lambda x, _d=dim, _e=eps: F.rms_norm(x, (_d,), None, _e)
        count += 1

    return count
