"""Register torch.ops.ggml.ssm_conv custom op for fused SSM conv via ggml_ssm_conv.

Replaces the manual kernel-unroll loop in GatedDeltaNet with llama.cpp's native
ggml_ssm_conv kernel.

Usage:
    conv_input: (B, C, K-1 + T)  — concatenated [old_state, new_tokens]
    weight:     (C, K)           — squeezed depthwise conv1d kernel
    Returns:    (B, T, C)        — convolved output, in ggml_ssm_conv native layout
                                   (no permute needed in C++ runtime)
"""

import torch

ggml_lib = torch.library.Library("ggml", "FRAGMENT")
ggml_lib.define("ssm_conv(Tensor conv_input, Tensor weight) -> Tensor")


@torch.library.impl(ggml_lib, "ssm_conv", "CompositeExplicitAutograd")
def ssm_conv_impl(conv_input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Eager fallback: depthwise conv1d via kernel-unroll, returned as [B, T, C]."""
    B, C, L = conv_input.shape
    K = weight.shape[1]
    T = L - K + 1
    acc = torch.zeros(B, C, T, dtype=torch.float32, device=conv_input.device)
    for k in range(K):
        acc = acc + conv_input[:, :, k:k + T].float() * weight[:, k:k + 1]
    return acc.transpose(1, 2).contiguous()


@torch.library.impl_abstract("ggml::ssm_conv")
def ssm_conv_abstract(conv_input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    B, C, L = conv_input.shape
    K = weight.shape[1]
    T = L - K + 1
    return conv_input.new_empty((B, T, C), dtype=torch.float32)
