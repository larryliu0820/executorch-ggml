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

    def __init__(self, n_heads, n_kv_heads, head_dim, dim, freq_base, max_seq_len=4096):
        super().__init__()
        self.max_seq_len = max_seq_len
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

        # SDPA with native GQA. Slice cache to valid positions only —
        # attending to zero-filled entries produces wrong output.
        # Use is_causal only during prefill (S_q == S_kv); during decode
        # the cache already enforces causality.
        kv_len = input_pos[0].item() + T
        torch._check(kv_len > 0)
        torch._check(kv_len >= T)
        torch._check(kv_len <= self.max_seq_len)
        q = q.transpose(1, 2)
        k_slice = self.k_cache[:, :kv_len, :, :].transpose(1, 2)
        v_slice = self.v_cache[:, :kv_len, :, :].transpose(1, 2)

        # Manual SDPA: avoids PyTorch >= 2.7 decomposition guards.
        # KV cache is already sliced to valid positions — no causal mask needed.
        scale = 1.0 / (q.shape[-1] ** 0.5)
        if self.enable_gqa:
            n_rep = self.n_heads // self.n_kv_heads
            k_slice = k_slice.repeat_interleave(n_rep, dim=1)
            v_slice = v_slice.repeat_interleave(n_rep, dim=1)
        attn_weights = torch.matmul(q, k_slice.transpose(-2, -1)) * scale
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask
        attn_weights = F.softmax(attn_weights, dim=-1)
        y = torch.matmul(attn_weights, v_slice)

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
            max_seq_len=old.kv_cache.max_seq_len if hasattr(old, 'kv_cache') else 4096,
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
