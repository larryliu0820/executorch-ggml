"""Export Cactus Compute's Needle to ExecuTorch with the GGML backend.

Needle is a 26M-parameter encoder-decoder transformer (12 enc layers,
8 dec layers, GQA, RoPE, ZCRMSNorm, no FFN). The published checkpoint is a
JAX/Flax pickle (`needle.pkl`); we load it into a PyTorch port
(`executorch_ggml.models.needle`) and export three ExecuTorch methods:

  * `token_embedding(token_ids)` — embedding lookup (handy for prompt
    pre-processing or contrastive use; not required by the decode loop).
  * `encoder(src_tokens)` — produces encoder hidden states for the prompt.
  * `decoder(token_id, input_pos, encoder_out, cross_mask)` — single
    decode step. Self-attention KV cache (per-layer) is held in mutable
    buffers and persists across calls; `input_pos` indexes the cache and
    must be advanced 0, 1, 2, ... by the caller. Returns logits for the
    new token, shape [1, 1, vocab].

Usage:
    python export_needle_ggml.py --output needle/needle.pte
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn
from torch.export import Dim, export

import executorch_ggml  # noqa: F401  # registers backend via static init
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)
from executorch.exir.passes import MemoryPlanningPass
from executorch_ggml import GgmlPartitioner
from executorch_ggml.passes import RemoveGraphAssertsPass
from executorch_ggml.passes.replace_copy_ops_pass import ReplaceCopyOpsPass

from executorch_ggml.models.needle import (
    KVCacheDecoder,
    NeedleConfig,
    NeedleModel,
    fold_zcrmsnorm_weights,
    fold_zcrmsnorm_weights_kv,
    load_jax_checkpoint,
    make_padding_mask,
)


# ---------------------------------------------------------------------------
# Export wrappers


class TokenEmbeddingExport(nn.Module):
    def __init__(self, model: NeedleModel):
        super().__init__()
        self.embedding = model.embedding
        self.embed_scale = model.embed_scale

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(token_ids) * self.embed_scale


class EncoderExport(nn.Module):
    """Encoder: src_tokens -> encoder_out.

    Padding mask is built inside so torch.export captures only the token
    input. We pass the embedded tokens through the encoder stack rather than
    going via `model.encode` so the embedding lookup can be exported as its
    own method (matching how Voxtral splits its program).
    """

    def __init__(self, model: NeedleModel, pad_token_id: int):
        super().__init__()
        self.encoder_layers = model.encoder_layers
        self.encoder_norm = model.encoder_norm
        self.embedding = model.embedding
        self.embed_scale = model.embed_scale
        self.register_buffer("rope_cos", model.rope_cos.detach().clone())
        self.register_buffer("rope_sin", model.rope_sin.detach().clone())
        self.pad_token_id = pad_token_id

    def forward(self, src_tokens: torch.Tensor) -> torch.Tensor:
        # src_tokens: [1, T_src]
        T = src_tokens.shape[1]
        x = self.embedding(src_tokens) * self.embed_scale
        rope = (self.rope_cos[:T], self.rope_sin[:T])
        # Padding mask in the same [B, 1, 1, T] layout the modules expect.
        pad_mask = (src_tokens != self.pad_token_id).unsqueeze(1).unsqueeze(1)
        for layer in self.encoder_layers:
            x = layer(x, pad_mask, rope)
        return self.encoder_norm(x)


# ---------------------------------------------------------------------------
# Export pipeline


def export_needle(
    checkpoint_path: str,
    output_path: str,
    max_enc_len: int = 1024,
    max_gen_len: int = 512,
):
    print(f"Loading needle checkpoint from {checkpoint_path}...")
    model, cfg = load_jax_checkpoint(checkpoint_path)
    model.eval()

    print(f"  config: d_model={cfg.d_model}, heads={cfg.num_heads}/{cfg.num_kv_heads}, "
          f"enc_layers={cfg.num_encoder_layers}, dec_layers={cfg.num_decoder_layers}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params / 1e6:.1f}M")

    # --- Fusion passes (run BEFORE export so the fused modules are
    # captured in the FX graph).

    # 1) Fold pre-attn ZCRMSNorm weights into projections.
    n_folded = fold_zcrmsnorm_weights(model)
    print(f"  fold_zcrmsnorm_weights: {n_folded} pre-attn norms folded")

    # 2) Build the KV decoder before fusing Q/K/V — `from_needle_model`
    # copies state from `model.*.q_proj.weight` etc., and that breaks once
    # those are replaced by `_SlicedLinear` views over a shared parameter.
    kv_dec = KVCacheDecoder.from_needle_model(model, max_seq_len=max_gen_len).eval()

    # 3) Fuse Q/K/V projections in both the encoder/decoder model and the
    # KV decoder. After torch.export + DCE this becomes one aten.linear
    # plus three aten.slice ops per attention layer.
    from executorch_ggml.passes.fuse_projections import fuse_qkv_projections
    n_qkv_main = fuse_qkv_projections(model)
    n_qkv_kv = fuse_qkv_projections(kv_dec)
    print(f"  fuse_qkv_projections: {n_qkv_main} (NeedleModel) + {n_qkv_kv} (KVCacheDecoder)")

    programs = {}

    # --- Token embedding ---
    print("\nExporting token_embedding...")
    tok_emb = TokenEmbeddingExport(model).eval()
    sample_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    tok_seq_dim = Dim("tok_seq_len", min=1, max=max(max_enc_len, max_gen_len))
    programs["token_embedding"] = export(
        tok_emb,
        (sample_ids,),
        dynamic_shapes={"token_ids": {1: tok_seq_dim}},
        strict=False,
        prefer_deferred_runtime_asserts_over_guards=True,
    )

    # --- Encoder ---
    print("Exporting encoder...")
    enc = EncoderExport(model, pad_token_id=cfg.pad_token_id).eval()
    sample_src = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
    enc_seq_dim = Dim("enc_seq_len", min=1, max=max_enc_len)
    programs["encoder"] = export(
        enc,
        (sample_src,),
        dynamic_shapes={"src_tokens": {1: enc_seq_dim}},
        strict=False,
        prefer_deferred_runtime_asserts_over_guards=True,
    )

    # --- KV-cached decoder ---
    print("Exporting decoder (KV cache)...")
    sample_tok = torch.tensor([[1]], dtype=torch.long)
    sample_pos = torch.tensor([0], dtype=torch.long)
    sample_enc_out_kv = torch.randn(1, 8, cfg.d_model, dtype=torch.float32)
    sample_cross_mask_kv = torch.ones(1, 1, 1, 8, dtype=torch.bool)
    enc_out_dim = Dim("enc_out_len", min=1, max=max_enc_len)
    programs["decoder"] = export(
        kv_dec,
        (sample_tok, sample_pos, sample_enc_out_kv, sample_cross_mask_kv),
        dynamic_shapes={
            "token_id": None,
            "input_pos": None,
            "encoder_out": {1: enc_out_dim},
            "cross_mask": {3: enc_out_dim},
        },
        strict=False,
        prefer_deferred_runtime_asserts_over_guards=True,
    )

    # AOT graph passes:
    # 1) strip GQA expand (unsqueeze+expand+reshape) from SDPA K/V args.
    #    ggml_flash_attn_ext handles GQA natively via gqa_ratio so the
    #    expand is dead weight.
    # 2) Common-subexpression elimination — collapses the N identical
    #    aten.linear nodes that fuse_qkv emits (each `_SlicedLinear`
    #    produces a full `linear(x, fused_weight)` followed by a slice;
    #    CSE turns the N linears into one).
    from executorch_ggml.passes.strip_gqa_expand_pass import strip_gqa_expand
    from executorch_ggml.passes.cse_pass import eliminate_common_subexpressions
    for name, ep in programs.items():
        n_gqa = strip_gqa_expand(ep.graph_module)
        n_cse = eliminate_common_subexpressions(ep.graph_module)
        if n_gqa > 0 or n_cse > 0:
            print(f"  AOT passes ({name}): strip_gqa_expand={n_gqa}, CSE={n_cse}")

    # --- Metadata (constant methods) ---
    metadata = {
        "vocab_size": cfg.vocab_size,
        "d_model": cfg.d_model,
        "num_heads": cfg.num_heads,
        "num_kv_heads": cfg.num_kv_heads,
        "num_encoder_layers": cfg.num_encoder_layers,
        "num_decoder_layers": cfg.num_decoder_layers,
        "pad_token_id": cfg.pad_token_id,
        "max_enc_len": max_enc_len,
        "max_gen_len": max_gen_len,
    }

    # --- Lower to GGML ---
    print("\nLowering to ExecuTorch with GGML backend...")
    partitioner = {key: [GgmlPartitioner()] for key in programs}
    transform_passes = [ReplaceCopyOpsPass(), RemoveGraphAssertsPass()]

    edge = to_edge_transform_and_lower(
        programs,
        transform_passes=transform_passes,
        partitioner=partitioner,
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=True,
        ),
        constant_methods=metadata,
    )
    et = edge.to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
        ),
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(et.buffer)
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"\nSaved {output_path} ({size_mb:.1f} MB)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to needle.pkl. If omitted, downloads from HF.")
    p.add_argument("--output", type=str, default="needle/needle.pte")
    p.add_argument("--max-enc-len", type=int, default=1024)
    p.add_argument("--max-gen-len", type=int, default=512)
    return p.parse_args()


def main():
    args = parse_args()
    ckpt = args.checkpoint
    if ckpt is None:
        from huggingface_hub import hf_hub_download
        ckpt = hf_hub_download("Cactus-Compute/needle", "needle.pkl")
    export_needle(
        checkpoint_path=ckpt,
        output_path=args.output,
        max_enc_len=args.max_enc_len,
        max_gen_len=args.max_gen_len,
    )


if __name__ == "__main__":
    main()
