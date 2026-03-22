"""Module swap: replace decoder decomposed RoPE with fused ggml.rope.

The decoder's LMAttention calls apply_rotary_emb which decomposes into
~9 ops per Q/K (reshape, unbind, mul, stack, flatten, etc.) × 26 layers.
This replaces it with a single ggml.rope call (mode=0, interleaved).

Must be called AFTER swap_voxtral_attention (which swaps KV cache + SDPA).

Usage:
    from executorch_ggml.modules.voxtral_decoder_rope import swap_decoder_rope
    swap_decoder_rope(model, freq_base=model.config.rope_theta)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import executorch_ggml.rope_op  # noqa: F401 — registers torch.ops.ggml.rope


class GgmlDecoderAttention(nn.Module):
    """Decoder attention with fused RoPE, index_copy KV cache, native GQA SDPA."""

    def __init__(self, n_heads, n_kv_heads, head_dim, dim, freq_base):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.dim = dim
        self.freq_base = freq_base
        self.enable_gqa = n_kv_heads != n_heads

        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)

        # KV cache buffers are registered during swap from existing modules.

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        input_pos: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        # Fused RoPE (interleaved mode=0)
        q = torch.ops.ggml.rope(q, input_pos.int(), self.head_dim, self.freq_base)
        k = torch.ops.ggml.rope(k, input_pos.int(), self.head_dim, self.freq_base)

        # KV cache update
        self.k_cache.index_copy_(1, input_pos, k)
        self.v_cache.index_copy_(1, input_pos, v)

        # SDPA with native GQA
        q = q.transpose(1, 2)
        k_full = self.k_cache.transpose(1, 2)
        v_full = self.v_cache.transpose(1, 2)

        if attn_mask is None:
            y = F.scaled_dot_product_attention(
                q, k_full, v_full, is_causal=True, enable_gqa=self.enable_gqa)
        else:
            y = F.scaled_dot_product_attention(
                q, k_full, v_full, attn_mask=attn_mask, enable_gqa=self.enable_gqa)

        attn_dim = self.n_heads * self.head_dim
        return self.wo(y.transpose(1, 2).contiguous().view(B, T, attn_dim))


def swap_decoder_rope(model: nn.Module, freq_base: float) -> int:
    """Replace decoder LMAttention modules with GgmlDecoderAttention.

    Copies weights and KV cache buffers from the existing attention modules.
    Must be called AFTER swap_voxtral_attention.

    Returns the number of layers swapped.
    """
    decoder = model.decoder if hasattr(model, "decoder") else model
    if not hasattr(decoder, "layers"):
        return 0

    count = 0
    for layer in decoder.layers:
        old = layer.attention
        new_attn = GgmlDecoderAttention(
            n_heads=old.n_heads,
            n_kv_heads=old.n_kv_heads,
            head_dim=old.head_dim,
            dim=old.dim,
            freq_base=freq_base,
        )
        # Copy weights
        new_attn.wq.weight = old.wq.weight
        new_attn.wk.weight = old.wk.weight
        new_attn.wv.weight = old.wv.weight
        new_attn.wo.weight = old.wo.weight

        # Copy KV cache buffers (from IndexCopyKVCache or KVCache)
        kv = old.kv_cache
        new_attn.register_buffer("k_cache", kv.k_cache)
        new_attn.register_buffer("v_cache", kv.v_cache)

        layer.attention = new_attn
        count += 1

    return count
