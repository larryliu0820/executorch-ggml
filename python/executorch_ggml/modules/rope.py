"""Module swap: replace encoder apply_rotary_emb with fused ggml.rope op.

Usage (before export):
    from executorch_ggml.modules.rope import swap_encoder_rope
    swap_encoder_rope(model, freq_base=model.config.enc_rope_theta)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import executorch_ggml.rope_op  # noqa: F401 — registers torch.ops.ggml.rope


class GgmlEncoderAttention(nn.Module):
    """Drop-in replacement for EncoderAttention that uses ggml.rope.

    Same interface: forward(x, freqs_cos, freqs_sin) -> Tensor.
    freqs_cos/freqs_sin are accepted for signature compatibility but ignored;
    positions are generated internally as arange(0, T).
    """

    def __init__(self, n_heads: int, head_dim: int, freq_base: float):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.freq_base = freq_base
        attn_dim = n_heads * head_dim
        self.wq = nn.Linear(attn_dim, attn_dim)  # placeholder, weights copied
        self.wk = nn.Linear(attn_dim, attn_dim)
        self.wv = nn.Linear(attn_dim, attn_dim)
        self.wo = nn.Linear(attn_dim, attn_dim)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_heads, self.head_dim)

        positions = torch.arange(T, dtype=torch.int32, device=x.device)
        q = torch.ops.ggml.rope(q, positions, self.head_dim, self.freq_base)
        k = torch.ops.ggml.rope(k, positions, self.head_dim, self.freq_base)

        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(y.transpose(1, 2).contiguous().view(B, T, -1))


def swap_encoder_rope(model: nn.Module, freq_base: float) -> None:
    """Replace each encoder layer's attention with GgmlEncoderAttention.

    Walks model.encoder.layers (CausalWhisperEncoder) and swaps in-place.
    """
    encoder = model.encoder if hasattr(model, "encoder") else model
    for layer in encoder.layers:
        old = layer.attention
        new_attn = GgmlEncoderAttention(
            n_heads=old.n_heads,
            head_dim=old.head_dim,
            freq_base=freq_base,
        )
        # Copy weights (preserving device/dtype)
        new_attn.wq.weight = old.wq.weight
        new_attn.wq.bias = old.wq.bias
        new_attn.wk.weight = old.wk.weight
        new_attn.wk.bias = old.wk.bias
        new_attn.wv.weight = old.wv.weight
        new_attn.wv.bias = old.wv.bias
        new_attn.wo.weight = old.wo.weight
        new_attn.wo.bias = old.wo.bias
        layer.attention = new_attn
