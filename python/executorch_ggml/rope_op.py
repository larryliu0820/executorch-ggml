"""Register torch.ops.ggml.rope custom op for fused RoPE via ggml_rope_ext.

Usage:
    import executorch_ggml.rope_op  # registers the op
    out = torch.ops.ggml.rope(x, positions, n_dims, freq_base)
    out = torch.ops.ggml.rope(x, positions, n_dims, freq_base, mode=2)  # NeoX half-rotation

x:         (B, T, n_heads, head_dim)  — the query or key tensor
positions: (T,)                       — I32/I64 position indices
n_dims:    int                        — number of dims to rotate (head_dim)
freq_base: float                      — RoPE frequency base (e.g. 1e6)
mode:      int                        — 0=interleaved pairs, 2=NeoX half-rotation

Returns:   same shape/dtype as x
"""

import torch

ggml_lib = torch.library.Library("ggml", "DEF")
ggml_lib.define(
    "rope(Tensor x, Tensor positions, int n_dims, float freq_base, int mode=0) -> Tensor"
)


@torch.library.impl(ggml_lib, "rope", "CPU")
def rope_cpu(
    x: torch.Tensor,
    positions: torch.Tensor,
    n_dims: int,
    freq_base: float,
    mode: int = 0,
) -> torch.Tensor:
    """Eager fallback: rotary position embedding."""
    # x: (B, T, n_heads, head_dim), positions: (T,)
    half = n_dims // 2
    freqs = 1.0 / (
        freq_base ** (torch.arange(0, n_dims, 2, device=x.device).float() / n_dims)
    )
    # (T, half)
    angles = torch.outer(positions.float(), freqs)
    cos = angles.cos().unsqueeze(0).unsqueeze(2)  # (1, T, 1, half)
    sin = angles.sin().unsqueeze(0).unsqueeze(2)

    x_f = x.float()

    if mode == 2:
        # NeoX half-rotation: split first/second half, rotate
        x1 = x_f[..., :half]
        x2 = x_f[..., half:n_dims]
        rotated = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        if n_dims < x.shape[-1]:
            rotated = torch.cat([rotated, x_f[..., n_dims:]], dim=-1)
    else:
        # mode=0: interleaved pairs (x0,x1), (x2,x3), ...
        x_r = x_f[..., :n_dims].reshape(x_f.shape[:-1] + (half, 2))
        x_re, x_im = x_r.unbind(-1)
        out_re = x_re * cos - x_im * sin
        out_im = x_re * sin + x_im * cos
        rotated = torch.stack([out_re, out_im], dim=-1).flatten(-2)
        if n_dims < x.shape[-1]:
            rotated = torch.cat([rotated, x_f[..., n_dims:]], dim=-1)

    return rotated.to(x.dtype)


@torch.library.impl(ggml_lib, "rope", "Meta")
def rope_meta(
    x: torch.Tensor,
    positions: torch.Tensor,
    n_dims: int,
    freq_base: float,
    mode: int = 0,
) -> torch.Tensor:
    return torch.empty_like(x)
