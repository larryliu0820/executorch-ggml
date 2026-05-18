"""PyTorch port of Cactus Compute's Needle (Simple Attention Network).

Needle is a 26M-parameter encoder-decoder transformer (12 encoder layers,
8 decoder layers) trained in JAX/Flax. The original architecture lives at
third-party/needle/needle/model/architecture.py. This module mirrors that
architecture in PyTorch so we can torch.export and lower to the GGML backend.

Key features mirrored from the JAX original:
  * ZCRMSNorm: zero-centred RMSNorm — `(1 + scale) * x / RMS(x)`.
  * GQA self-attention (8 heads / 4 KV heads in the published checkpoint).
  * RoPE with theta=10000, head_dim=64, applied to the first half of the dim
    pair (cat[x1*cos - x2*sin, x2*cos + x1*sin]).
  * Per-layer learnable `attn_gate` / `cross_attn_gate` / `self_attn_gate`
    (sigmoided, scalar-per-layer) on the residual.
  * No FFN (`no_feedforward=True` in the published checkpoint).
  * Decoder cross-attention into encoder output; logits via tied embedding.

The JAX checkpoint stacks per-layer params on a leading axis (nn.scan).
`load_jax_checkpoint` slices that out and transposes Dense kernels (JAX uses
[in, out] while PyTorch's nn.Linear weight is [out, in]).
"""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config


@dataclass
class NeedleConfig:
    vocab_size: int = 8192
    d_model: int = 512
    num_heads: int = 8
    num_kv_heads: int = 4
    num_encoder_layers: int = 12
    num_decoder_layers: int = 8
    d_ff: int = 2048
    max_seq_len: int = 1024
    pad_token_id: int = 0
    rope_theta: float = 10000.0
    activation: str = "swiglu"
    no_feedforward: bool = True
    contrastive_dim: int = 128

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads

    @staticmethod
    def from_jax_config(cfg: dict) -> "NeedleConfig":
        valid = {f for f in NeedleConfig.__dataclass_fields__}
        return NeedleConfig(**{k: v for k, v in cfg.items() if k in valid})


# ---------------------------------------------------------------------------
# Modules


class ZCRMSNorm(nn.Module):
    """Zero-centred RMSNorm: `(1 + scale) * x / RMS(x)`. Scale init=0."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.rms_norm is preserved by GgmlPartitioner and maps to ggml_rms_norm
        # for CUDA fusion. Multiplier is (1 + scale) per the JAX implementation.
        w = (1.0 + self.scale.float())
        return F.rms_norm(x.float(), [x.shape[-1]], w, self.eps).type_as(x)


def precompute_rope_freqs(head_dim: int, max_seq_len: int, theta: float):
    """Pre-compute cos/sin table with the same layout as the JAX original.

    JAX builds a [T, head_dim/2] table via outer(arange(T), 1/theta^(2k/d)).
    We expose it as cos/sin buffers indexed by absolute position.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    angles = torch.outer(t, freqs)  # [T, head_dim/2]
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Match `apply_rope` from the JAX original.

    x: [B, H, T, D]. cos/sin: [T, D/2]. Splits x along the last dim into
    [first half, second half] and rotates with cat[x1*cos - x2*sin,
    x2*cos + x1*sin].
    """
    half = x.shape[-1] // 2
    cos_b = cos[None, None, :, :].to(x.dtype)
    sin_b = sin[None, None, :, :].to(x.dtype)
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([x1 * cos_b - x2 * sin_b, x2 * cos_b + x1 * sin_b], dim=-1)


class MultiHeadAttention(nn.Module):
    """GQA multi-head attention with QK-norm and (optional) RoPE.

    Mirrors the JAX MultiHeadAttention block. Uses `repeat` to broadcast K/V
    from `num_kv_heads` up to `num_heads` so plain `matmul` works (the JAX
    original does the same with `jnp.repeat`).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        rope_keys_only: bool = False,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        assert num_heads % num_kv_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.kv_dim = num_kv_heads * self.head_dim
        self.rope_keys_only = rope_keys_only

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.kv_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.q_norm = ZCRMSNorm(self.head_dim)
        self.k_norm = ZCRMSNorm(self.head_dim)

    def forward(
        self,
        q_input: torch.Tensor,
        kv_input: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        B, T_q, _ = q_input.shape
        T_kv = kv_input.shape[1]

        q = self.q_proj(q_input).view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv_input).view(B, T_kv, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_input).view(B, T_kv, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if rope is not None:
            cos, sin = rope
            if not self.rope_keys_only:
                q = apply_rope(q, cos[:T_q], sin[:T_q])
            k = apply_rope(k, cos[:T_kv], sin[:T_kv])

        # JAX applies mask as `where(mask, attn, finfo.min)` then softmax.
        # F.sdpa's attn_mask, when bool, does the same: True positions
        # attend, False positions are masked. We pass it directly so the
        # partitioner can preserve the SDPA op end-to-end and lower it to
        # ggml_flash_attn_ext (preserve_sdpa=True is the partitioner default).
        # GQA: pre-expand K/V via unsqueeze+expand+reshape (matches the
        # qwen3 strip_gqa_expand_pass pattern). enable_gqa=True in
        # F.sdpa is known to diverge from manual decomposition for GQA in
        # some cases (see CLAUDE.md: "is_causal=True GQA divergence");
        # we keep our own GQA expand and let the runtime keep the heads.
        repeats = self.num_heads // self.num_kv_heads
        if repeats > 1:
            B, H_kv, T_k, D = k.shape
            k = k.unsqueeze(2).expand(B, H_kv, repeats, T_k, D).reshape(B, H_kv * repeats, T_k, D)
            v = v.unsqueeze(2).expand(B, H_kv, repeats, T_k, D).reshape(B, H_kv * repeats, T_k, D)

        scale = 1.0 / math.sqrt(float(self.head_dim))
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, scale=scale,
        )

        y = y.transpose(1, 2).reshape(B, T_q, self.d_model)
        return self.out_proj(y)


class SelfAttnKVCache(nn.Module):
    """Per-layer self-attention KV cache, [1, num_kv_heads, max_seq_len, head_dim].

    `update(input_pos, k_val, v_val)` writes the new K/V at `input_pos` and
    returns the full cache tensor for reading. The cache is registered as a
    buffer so it survives across decode steps in the exported runtime.
    """

    def __init__(self, num_kv_heads: int, head_dim: int, max_seq_len: int):
        super().__init__()
        # `persistent=True` (the default) is required so the buffer is
        # shared across exported methods of the same program. With
        # persistent=False, each method gets its own private copy and
        # writes from `prepare_cross_kv` would not be visible to `decoder`.
        self.register_buffer(
            "k_cache",
            torch.zeros(1, num_kv_heads, max_seq_len, head_dim),
        )
        self.register_buffer(
            "v_cache",
            torch.zeros(1, num_kv_heads, max_seq_len, head_dim),
        )

    def update(
        self,
        input_pos: torch.Tensor,
        k_val: torch.Tensor,
        v_val: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # k_val: [1, num_kv_heads, T_q, head_dim]; T_q is typically 1 for decode.
        # input_pos: [T_q] int — absolute positions the new tokens occupy.
        self.k_cache[:, :, input_pos] = k_val
        self.v_cache[:, :, input_pos] = v_val
        return self.k_cache, self.v_cache


class CachedSelfAttention(nn.Module):
    """Self-attention with KV cache for decoder use.

    Mirrors `MultiHeadAttention` (same projections + QK-norm + RoPE) but
    writes the new K/V into a per-layer cache and reads the full cache for
    attention. The mask must be broadcastable to [B, 1, T_q, max_seq_len]
    and is responsible for masking positions > current input_pos.
    """

    def __init__(self, config: "NeedleConfig"):
        super().__init__()
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, self.kv_dim, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.q_norm = ZCRMSNorm(self.head_dim)
        self.k_norm = ZCRMSNorm(self.head_dim)
        self.kv_cache = SelfAttnKVCache(
            self.num_kv_heads, self.head_dim, config.max_seq_len,
        )

    def forward(
        self,
        x: torch.Tensor,
        input_pos: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        B, T_q, _ = x.shape
        q = self.q_proj(x).view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T_q, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T_q, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE for the new tokens only — uses absolute positions from input_pos.
        cos = rope_cos[input_pos]
        sin = rope_sin[input_pos]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Write into cache, then read full cache for attention.
        k_full, v_full = self.kv_cache.update(input_pos, k, v)

        # GQA expand on the full cache (same pattern as MultiHeadAttention).
        repeats = self.num_heads // self.num_kv_heads
        if repeats > 1:
            B, H_kv, T_k, D = k_full.shape
            k_full = k_full.unsqueeze(2).expand(B, H_kv, repeats, T_k, D).reshape(B, H_kv * repeats, T_k, D)
            v_full = v_full.unsqueeze(2).expand(B, H_kv, repeats, T_k, D).reshape(B, H_kv * repeats, T_k, D)

        scale = 1.0 / math.sqrt(float(self.head_dim))
        y = F.scaled_dot_product_attention(
            q, k_full, v_full, attn_mask=mask, scale=scale,
        )
        y = y.transpose(1, 2).reshape(B, T_q, self.d_model)
        return self.out_proj(y)


class CachedDecoderBlock(nn.Module):
    """Decoder block using `CachedSelfAttention` + standard cross-attention.

    Cross-attention K/V are recomputed from `encoder_out` each step. We
    initially tried caching them across decode steps via a `prepare_cross_kv`
    method writing into per-layer mutable buffers, but the GgmlBackend
    allocates `mutable_buf` per delegated partition — buffers with the
    same FQN in different methods are *not* the same physical memory. So
    writes from `prepare_cross_kv` are invisible to `decoder` and the
    cache reads back zero. Recomputing every step is the simpler design
    until the runtime grows a cross-method shared mutable buffer.
    """

    def __init__(self, config: "NeedleConfig"):
        super().__init__()
        self.self_norm = ZCRMSNorm(config.d_model)
        self.self_attn = CachedSelfAttention(config)
        self.self_attn_gate = nn.Parameter(torch.zeros(()))

        self.cross_norm = ZCRMSNorm(config.d_model)
        self.cross_attn = MultiHeadAttention(
            config.d_model, config.num_heads, config.num_kv_heads,
        )
        self.cross_attn_gate = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        x: torch.Tensor,
        input_pos: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        encoder_out: torch.Tensor,
        self_mask: torch.Tensor,
        cross_mask: torch.Tensor,
    ) -> torch.Tensor:
        self_g = torch.sigmoid(self.self_attn_gate)
        h = self.self_norm(x)
        h = self.self_attn(h, input_pos, rope_cos, rope_sin, self_mask)
        x = x + self_g * h

        cross_g = torch.sigmoid(self.cross_attn_gate)
        h = self.cross_norm(x)
        h = self.cross_attn(h, encoder_out, mask=cross_mask, rope=None)
        x = x + cross_g * h
        return x


class EncoderBlock(nn.Module):
    """Pre-norm self-attention block with sigmoided learnable residual gate.

    Matches the JAX EncoderBlock with `no_feedforward=True`.
    """

    def __init__(self, config: NeedleConfig):
        super().__init__()
        self.norm = ZCRMSNorm(config.d_model)
        self.self_attn = MultiHeadAttention(
            config.d_model, config.num_heads, config.num_kv_heads,
        )
        self.attn_gate = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        rope: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        gate = torch.sigmoid(self.attn_gate)
        residual = x
        h = self.norm(x)
        h = self.self_attn(h, h, mask=mask, rope=rope)
        return residual + gate * h


class DecoderBlock(nn.Module):
    """Pre-norm self-attn + cross-attn with per-residual gates."""

    def __init__(self, config: NeedleConfig):
        super().__init__()
        self.self_norm = ZCRMSNorm(config.d_model)
        self.self_attn = MultiHeadAttention(
            config.d_model, config.num_heads, config.num_kv_heads,
        )
        self.self_attn_gate = nn.Parameter(torch.zeros(()))

        self.cross_norm = ZCRMSNorm(config.d_model)
        self.cross_attn = MultiHeadAttention(
            config.d_model, config.num_heads, config.num_kv_heads,
        )
        self.cross_attn_gate = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor,
        self_mask: Optional[torch.Tensor],
        cross_mask: Optional[torch.Tensor],
        rope: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        self_g = torch.sigmoid(self.self_attn_gate)
        h = self.self_norm(x)
        h = self.self_attn(h, h, mask=self_mask, rope=rope)
        x = x + self_g * h

        cross_g = torch.sigmoid(self.cross_attn_gate)
        h = self.cross_norm(x)
        # Cross-attention: queries from x, keys/values from encoder. JAX does
        # not apply RoPE to cross-attention.
        h = self.cross_attn(h, encoder_out, mask=cross_mask, rope=None)
        x = x + cross_g * h
        return x


class KVCacheDecoder(nn.Module):
    """Decoder-only model with self-attn KV cache, sharing weights with a
    parent `NeedleModel`. Built once via `from_needle_model` and exported
    as a `decoder(token_id, input_pos, encoder_out, cross_mask)` method.

    Cross-attention K/V is recomputed from `encoder_out` on every call —
    splitting that into a separate `prepare_cross_kv` method does not
    work with the current GgmlBackend because each delegated partition
    allocates its own `mutable_buf`, so writes in one method are not
    visible to reads in another even when the buffer FQNs match.
    """

    def __init__(self, config: NeedleConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [CachedDecoderBlock(config) for _ in range(config.num_decoder_layers)]
        )
        self.norm = ZCRMSNorm(config.d_model)
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.embed_scale = math.sqrt(float(config.d_model))
        cos, sin = precompute_rope_freqs(config.head_dim, config.max_seq_len, config.rope_theta)
        # Use a buffer name distinct from NeedleModel's `rope_cos` so the
        # two can coexist in the same .pte's named-data store.
        self.register_buffer("dec_rope_cos", cos, persistent=False)
        self.register_buffer("dec_rope_sin", sin, persistent=False)
        self.register_buffer(
            "cache_positions",
            torch.arange(config.max_seq_len, dtype=torch.long),
            persistent=False,
        )

    @staticmethod
    def from_needle_model(
        model: "NeedleModel",
        max_seq_len: Optional[int] = None,
    ) -> "KVCacheDecoder":
        cfg = model.config
        if max_seq_len is not None and max_seq_len != cfg.max_seq_len:
            from dataclasses import replace
            cfg = replace(cfg, max_seq_len=max_seq_len)
        kv = KVCacheDecoder(cfg)
        kv.embedding.load_state_dict(model.embedding.state_dict())
        kv.norm.load_state_dict(model.decoder_norm.state_dict())
        for src, dst in zip(model.decoder_layers, kv.layers):
            dst.self_norm.load_state_dict(src.self_norm.state_dict())
            dst.cross_norm.load_state_dict(src.cross_norm.state_dict())
            dst.cross_attn.load_state_dict(src.cross_attn.state_dict())
            dst.self_attn_gate.data.copy_(src.self_attn_gate.data)
            dst.cross_attn_gate.data.copy_(src.cross_attn_gate.data)
            dst.self_attn.q_proj.load_state_dict(src.self_attn.q_proj.state_dict())
            dst.self_attn.k_proj.load_state_dict(src.self_attn.k_proj.state_dict())
            dst.self_attn.v_proj.load_state_dict(src.self_attn.v_proj.state_dict())
            dst.self_attn.out_proj.load_state_dict(src.self_attn.out_proj.state_dict())
            dst.self_attn.q_norm.load_state_dict(src.self_attn.q_norm.state_dict())
            dst.self_attn.k_norm.load_state_dict(src.self_attn.k_norm.state_dict())
        return kv

    def reset_cache(self):
        for layer in self.layers:
            layer.self_attn.kv_cache.k_cache.zero_()
            layer.self_attn.kv_cache.v_cache.zero_()

    def forward(
        self,
        token_id: torch.Tensor,
        input_pos: torch.Tensor,
        encoder_out: torch.Tensor,
        cross_mask: torch.Tensor,
    ) -> torch.Tensor:
        """One decode step.

        token_id: [1, T_q] long — typically T_q=1 for incremental decode,
            but T_q>1 supported for prompt prefill.
        input_pos: [T_q] long — absolute position of each query token.
        encoder_out: [1, T_src, d_model].
        cross_mask: [1, 1, 1, T_src] bool.
        """
        x = self.embedding(token_id) * self.embed_scale
        self_mask = (
            self.cache_positions.unsqueeze(0) <= input_pos.unsqueeze(1)
        ).unsqueeze(0).unsqueeze(0)

        for layer in self.layers:
            x = layer(
                x, input_pos,
                self.dec_rope_cos, self.dec_rope_sin,
                encoder_out, self_mask, cross_mask,
            )
        x = self.norm(x)
        last = x[:, -1:, :].float()
        return last @ self.embedding.weight.float().T


class NeedleModel(nn.Module):
    """Encoder + decoder + tied LM head, end-to-end."""

    def __init__(self, config: NeedleConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.embed_scale = math.sqrt(float(config.d_model))

        self.encoder_layers = nn.ModuleList([EncoderBlock(config) for _ in range(config.num_encoder_layers)])
        self.encoder_norm = ZCRMSNorm(config.d_model)

        self.decoder_layers = nn.ModuleList([DecoderBlock(config) for _ in range(config.num_decoder_layers)])
        self.decoder_norm = ZCRMSNorm(config.d_model)

        cos, sin = precompute_rope_freqs(config.head_dim, config.max_seq_len, config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    # -- helpers --------------------------------------------------------

    def _rope(self, seq_len: int):
        return self.rope_cos[:seq_len], self.rope_sin[:seq_len]

    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """src: [B, T_src] int. src_mask: [B, 1, 1, T_src] bool."""
        x = self.embedding(src) * self.embed_scale
        rope = self._rope(src.shape[1])
        for layer in self.encoder_layers:
            x = layer(x, src_mask, rope)
        return self.encoder_norm(x)

    def decode(
        self,
        tgt: torch.Tensor,
        encoder_out: torch.Tensor,
        self_mask: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """tgt: [B, T_tgt] int. Returns logits [B, T_tgt, vocab]."""
        x = self.embedding(tgt) * self.embed_scale
        rope = self._rope(tgt.shape[1])
        for layer in self.decoder_layers:
            x = layer(x, encoder_out, self_mask, cross_mask, rope)
        x = self.decoder_norm(x)
        # Tied LM head, computed in F32 like the JAX original.
        return x.float() @ self.embedding.weight.float().T

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoder_out = self.encode(src, src_mask=src_mask)
        cm = cross_mask if cross_mask is not None else src_mask
        return self.decode(tgt, encoder_out, self_mask=tgt_mask, cross_mask=cm)


# ---------------------------------------------------------------------------
# Optional fusion: fold ZCRMSNorm weights into downstream linear projections


def fold_zcrmsnorm_weights(model: "NeedleModel") -> int:
    """Fold each pre-attention ZCRMSNorm's `(1 + scale)` into the linear
    weight of the projection it feeds.

    `(1 + scale) * (x / RMS(x))` followed by `Linear(W)` is mathematically
    equivalent to `x / RMS(x)` followed by `Linear(W * (1 + scale)[None, :])`.
    We zero out the norm scale so the rms_norm op emits no MUL after the
    normalization step. Saves one ggml MUL per fused norm at runtime
    without changing numerics.

    Folds:
      * encoder block self-attn pre-norms (12 in needle.pkl)
      * decoder block self-attn pre-norms (8)
      * decoder block cross-attn pre-norms — only into q_proj since k/v
        come from encoder_out, not the normed x

    Returns the number of norms folded.
    """
    n_folded = 0

    def fold(norm: ZCRMSNorm, projs):
        nonlocal n_folded
        # ZCRMSNorm forward applies `(1 + scale)` as the rms_norm weight.
        w = (1.0 + norm.scale.detach().float()).contiguous()  # [in_features]
        for proj in projs:
            if proj is None:
                continue
            # proj.weight: [out_features, in_features]; broadcast w over out.
            proj.weight.data.mul_(w.unsqueeze(0).to(proj.weight.dtype))
        # After folding, the norm should output `x / RMS(x)` only — i.e.
        # scale = 0 makes (1 + scale) = 1.
        norm.scale.data.zero_()
        n_folded += 1

    for layer in model.encoder_layers:
        fold(layer.norm, [
            layer.self_attn.q_proj,
            layer.self_attn.k_proj,
            layer.self_attn.v_proj,
        ])

    for layer in model.decoder_layers:
        fold(layer.self_norm, [
            layer.self_attn.q_proj,
            layer.self_attn.k_proj,
            layer.self_attn.v_proj,
        ])
        # cross_norm only feeds q_proj on the queries side; k/v come from
        # encoder_out which is NOT routed through cross_norm.
        fold(layer.cross_norm, [layer.cross_attn.q_proj])

    return n_folded


def fold_zcrmsnorm_weights_kv(kv_dec: "KVCacheDecoder") -> int:
    """Same as `fold_zcrmsnorm_weights` but for the KVCacheDecoder layers
    (which use `CachedDecoderBlock` containing `CachedSelfAttention`).
    """
    n_folded = 0

    def fold(norm: ZCRMSNorm, projs):
        nonlocal n_folded
        w = (1.0 + norm.scale.detach().float()).contiguous()
        for proj in projs:
            if proj is None:
                continue
            proj.weight.data.mul_(w.unsqueeze(0).to(proj.weight.dtype))
        norm.scale.data.zero_()
        n_folded += 1

    for layer in kv_dec.layers:
        fold(layer.self_norm, [
            layer.self_attn.q_proj,
            layer.self_attn.k_proj,
            layer.self_attn.v_proj,
        ])
        fold(layer.cross_norm, [layer.cross_attn.q_proj])

    return n_folded


# ---------------------------------------------------------------------------
# Mask helpers (match JAX: True = attend, False = mask)


def make_padding_mask(tokens: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """[B, T] -> [B, 1, 1, T]."""
    return (tokens != pad_token_id).unsqueeze(1).unsqueeze(1)


def make_causal_mask(seq_len: int, device=None) -> torch.Tensor:
    return torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)).unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# JAX → PyTorch checkpoint loader


def _to_torch(x):
    import numpy as np
    return torch.from_numpy(np.asarray(x).astype("float32"))


def load_jax_checkpoint(path: str) -> Tuple[NeedleModel, NeedleConfig]:
    """Load `needle.pkl` (the JAX/Flax pickle) into a `NeedleModel`.

    The JAX checkpoint stacks per-layer parameters on a leading axis (because
    the original training uses nn.scan). We slice each layer out, transpose
    Dense kernels from [in, out] to [out, in], and assemble a state_dict for
    the PyTorch port.
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
    cfg = NeedleConfig.from_jax_config(data["config"])
    params = data["params"]
    model = NeedleModel(cfg)

    state = {}

    # Embedding
    state["embedding.weight"] = _to_torch(params["embedding"]["embedding"])

    # Encoder/decoder layer stacks (JAX nn.scan -> [num_layers, ...])
    enc_layers = params["encoder"]["layers"]["EncoderBlock_0"]
    dec_layers = params["decoder"]["layers"]["DecoderBlock_0"]

    def slice_norm(stacked, layer_i):
        return _to_torch(stacked[layer_i])

    def slice_dense(stacked, layer_i):
        # Flax Dense kernel: [num_layers, in, out] -> torch Linear weight [out, in]
        return _to_torch(stacked[layer_i]).t().contiguous()

    for i in range(cfg.num_encoder_layers):
        prefix = f"encoder_layers.{i}."
        state[prefix + "norm.scale"] = slice_norm(enc_layers["ZCRMSNorm_0"]["scale"], i)
        state[prefix + "attn_gate"] = _to_torch(enc_layers["attn_gate"][i])

        sa = enc_layers["self_attn"]
        state[prefix + "self_attn.q_proj.weight"] = slice_dense(sa["q_proj"]["kernel"], i)
        state[prefix + "self_attn.k_proj.weight"] = slice_dense(sa["k_proj"]["kernel"], i)
        state[prefix + "self_attn.v_proj.weight"] = slice_dense(sa["v_proj"]["kernel"], i)
        state[prefix + "self_attn.out_proj.weight"] = slice_dense(sa["out_proj"]["kernel"], i)
        state[prefix + "self_attn.q_norm.scale"] = slice_norm(sa["q_norm"]["scale"], i)
        state[prefix + "self_attn.k_norm.scale"] = slice_norm(sa["k_norm"]["scale"], i)

    state["encoder_norm.scale"] = _to_torch(params["encoder"]["final_norm"]["scale"])

    for i in range(cfg.num_decoder_layers):
        prefix = f"decoder_layers.{i}."
        state[prefix + "self_norm.scale"] = slice_norm(dec_layers["ZCRMSNorm_0"]["scale"], i)
        state[prefix + "cross_norm.scale"] = slice_norm(dec_layers["ZCRMSNorm_1"]["scale"], i)
        state[prefix + "self_attn_gate"] = _to_torch(dec_layers["self_attn_gate"][i])
        state[prefix + "cross_attn_gate"] = _to_torch(dec_layers["cross_attn_gate"][i])

        for jax_name, pt_name in (("self_attn", "self_attn"), ("cross_attn", "cross_attn")):
            mod = dec_layers[jax_name]
            state[prefix + f"{pt_name}.q_proj.weight"] = slice_dense(mod["q_proj"]["kernel"], i)
            state[prefix + f"{pt_name}.k_proj.weight"] = slice_dense(mod["k_proj"]["kernel"], i)
            state[prefix + f"{pt_name}.v_proj.weight"] = slice_dense(mod["v_proj"]["kernel"], i)
            state[prefix + f"{pt_name}.out_proj.weight"] = slice_dense(mod["out_proj"]["kernel"], i)
            state[prefix + f"{pt_name}.q_norm.scale"] = slice_norm(mod["q_norm"]["scale"], i)
            state[prefix + f"{pt_name}.k_norm.scale"] = slice_norm(mod["k_norm"]["scale"], i)

    state["decoder_norm.scale"] = _to_torch(params["decoder"]["ZCRMSNorm_0"]["scale"])

    missing, unexpected = model.load_state_dict(state, strict=False)
    # Buffers (rope_cos/sin) are computed in __init__, not loaded.
    extra_missing = [k for k in missing if not k.startswith(("rope_cos", "rope_sin"))]
    if extra_missing:
        raise RuntimeError(f"Missing keys in state_dict: {extra_missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected keys: {unexpected}")
    return model, cfg
