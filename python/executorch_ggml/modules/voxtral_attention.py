"""Module swap: replace Voxtral's llama custom ops with standard ATen ops.

Replaces llama.update_cache with index_copy_ and llama.custom_sdpa with
F.scaled_dot_product_attention.  This avoids auto_functionalized_v2
wrapping that breaks ExecuTorch delegation.

Usage (before export):
    from executorch_ggml.modules.voxtral_attention import swap_voxtral_attention
    swap_voxtral_attention(model)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class IndexCopyKVCache(nn.Module):
    """KV cache using index_copy_ instead of llama.update_cache."""

    def __init__(self, max_seq_len: int, n_kv_heads: int, head_dim: int):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        cache_shape = (1, max_seq_len, n_kv_heads, head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape))
        self.register_buffer("v_cache", torch.zeros(cache_shape))

    def update(
        self, input_pos: torch.Tensor, k_val: torch.Tensor, v_val: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.k_cache.index_copy_(1, input_pos, k_val)
        self.v_cache.index_copy_(1, input_pos, v_val)
        return self.k_cache, self.v_cache


class StandardSDPA(nn.Module):
    """SDPA using F.scaled_dot_product_attention instead of llama.custom_sdpa.

    Expects KV cache in [B, S, H, D] layout; transposes to [B, H, S, D]
    for the standard SDPA call.  GQA handled natively by enable_gqa=True
    (ggml's flash_attn_ext supports GQA via gqa_ratio).
    """

    def __init__(self, n_heads: int, n_kv_heads: int, head_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.dim = n_heads * head_dim
        self.enable_gqa = n_kv_heads != n_heads

    def forward(
        self,
        input_pos: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bsz: int,
        seqlen: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if mask is None:
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, enable_gqa=self.enable_gqa)
        else:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, enable_gqa=self.enable_gqa)

        return y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)


def swap_voxtral_attention(model: nn.Module) -> None:
    """Replace KVCache and SDPA in all decoder attention layers.

    Walks model.decoder.layers and swaps kv_cache and sdpa in-place,
    copying existing cache buffer data.  Idempotent — skips layers
    already swapped (e.g. by swap_decoder_rope).
    """
    from executorch.examples.models.voxtral_realtime.model import KVCache, SDPA

    decoder = model.decoder if hasattr(model, "decoder") else model
    for layer in decoder.layers:
        attn = layer.attention

        # Skip if already replaced by swap_decoder_rope (GgmlDecoderAttention)
        if not hasattr(attn, "kv_cache"):
            continue

        # Swap KVCache
        if isinstance(attn.kv_cache, KVCache):
            old_kv = attn.kv_cache
            new_kv = IndexCopyKVCache(
                old_kv.max_seq_len, old_kv.n_kv_heads, old_kv.head_dim,
            )
            new_kv.k_cache = old_kv.k_cache
            new_kv.v_cache = old_kv.v_cache
            attn.kv_cache = new_kv

        # Swap SDPA
        if isinstance(attn.sdpa, SDPA):
            attn.sdpa = StandardSDPA(
                attn.n_heads, attn.n_kv_heads, attn.head_dim,
            )
