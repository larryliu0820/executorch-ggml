"""BatchNorm folding utilities.

This implements standard inference-time folding of BatchNorm2d into Conv2d.

Given:
  y = BN(conv(x; W, b))
We produce W', b' such that:
  y == conv(x; W', b')

Assumes bn is in eval mode (running_mean/var are used).
"""

from __future__ import annotations

import torch


def fold_conv_bn_weights(
    conv_w: torch.Tensor,
    conv_b: torch.Tensor | None,
    bn_weight: torch.Tensor | None,
    bn_bias: torch.Tensor | None,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (folded_w, folded_b).

    Shapes:
      conv_w: [out_channels, in_channels/groups, kH, kW]
      conv_b: [out_channels] or None
      bn_weight (gamma): [out_channels] or None
      bn_bias (beta): [out_channels] or None
      running_mean/var: [out_channels]
    """

    w = conv_w.detach().to(dtype=torch.float32)
    if conv_b is None:
        b = torch.zeros(w.shape[0], device=w.device, dtype=torch.float32)
    else:
        b = conv_b.detach().to(dtype=torch.float32)

    if bn_weight is None:
        gamma = torch.ones_like(running_mean, dtype=torch.float32)
    else:
        gamma = bn_weight.detach().to(dtype=torch.float32)

    if bn_bias is None:
        beta = torch.zeros_like(running_mean, dtype=torch.float32)
    else:
        beta = bn_bias.detach().to(dtype=torch.float32)

    mean = running_mean.detach().to(dtype=torch.float32)
    var = running_var.detach().to(dtype=torch.float32)

    invstd = torch.rsqrt(var + eps)
    scale = gamma * invstd  # [out]

    # Fold into conv weights
    w_fold = w * scale.reshape(-1, 1, 1, 1)

    # Fold into bias
    b_fold = (b - mean) * scale + beta

    return w_fold, b_fold
