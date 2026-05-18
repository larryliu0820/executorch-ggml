"""Numerical equivalence test for the Needle PyTorch port.

Validates two things:
  1. The PyTorch port (`executorch_ggml.models.needle.NeedleModel`) loaded
     from the JAX checkpoint matches the JAX/Flax reference numerics.
  2. The GGML-lowered ExecuTorch program matches the PyTorch port (run after
     export — see test_needle_ggml_e2e.py).

Run:
    python -m pytest tests/test_needle_numerics.py -s
"""

import os
import pickle
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "third-party", "needle"))

from executorch_ggml.models.needle import (  # noqa: E402
    NeedleConfig,
    NeedleModel,
    load_jax_checkpoint,
    make_causal_mask,
    make_padding_mask,
)


def _checkpoint_path():
    """Download (or reuse) the published needle.pkl from HF."""
    from huggingface_hub import hf_hub_download
    return hf_hub_download("Cactus-Compute/needle", "needle.pkl")


class TestNeedleEagerVsJax(unittest.TestCase):
    """Compare PyTorch port outputs against the JAX/Flax reference."""

    @classmethod
    def setUpClass(cls):
        try:
            import jax  # noqa: F401
        except ModuleNotFoundError:
            raise unittest.SkipTest("jax not installed; skip JAX comparison")
        cls.ckpt_path = _checkpoint_path()
        torch.manual_seed(0)
        np.random.seed(0)

        cls.pt_model, cls.cfg = load_jax_checkpoint(cls.ckpt_path)
        cls.pt_model.eval()

        import jax
        import jax.numpy as jnp
        from needle.model.architecture import SimpleAttentionNetwork, TransformerConfig

        with open(cls.ckpt_path, "rb") as f:
            data = pickle.load(f)
        # Cast params to float32 to match the PT port (which loads as F32).
        cls.jax_params = jax.tree.map(lambda x: jnp.asarray(np.asarray(x), dtype=jnp.float32), data["config"] and data["params"])
        jax_cfg = TransformerConfig(**data["config"])
        # Force float32 in the JAX model so cast points line up with the PT
        # port (which keeps everything in F32 internally).
        jax_cfg.dtype = "float32"
        cls.jax_model = SimpleAttentionNetwork(jax_cfg)
        cls.jnp = jnp
        cls.jax = jax

    def _gen_inputs(self, T_src=16, T_tgt=8):
        cfg = self.cfg
        rng = np.random.default_rng(42)
        # Avoid pad token id=0 so the padding mask doesn't accidentally
        # short-circuit any rows.
        src = rng.integers(low=1, high=cfg.vocab_size, size=(1, T_src), dtype=np.int32)
        tgt = rng.integers(low=1, high=cfg.vocab_size, size=(1, T_tgt), dtype=np.int32)
        return src, tgt

    def test_encoder_matches_jax(self):
        src_np, _ = self._gen_inputs()
        cfg = self.cfg

        # PyTorch encode
        src_pt = torch.from_numpy(src_np).long()
        src_mask_pt = make_padding_mask(src_pt, cfg.pad_token_id)
        with torch.no_grad():
            enc_pt = self.pt_model.encode(src_pt, src_mask=src_mask_pt)
        enc_pt_np = enc_pt.float().cpu().numpy()

        # JAX encode
        src_jax = self.jnp.asarray(src_np)
        src_mask_jax = (src_jax != cfg.pad_token_id)[:, None, None, :]
        enc_jax, _ = self.jax_model.apply(
            {"params": self.jax_params}, src_jax, src_mask=src_mask_jax, method="encode_text",
        )
        enc_jax_np = np.asarray(enc_jax)

        max_abs = np.abs(enc_pt_np - enc_jax_np).max()
        cos = (enc_pt_np * enc_jax_np).sum() / (
            np.linalg.norm(enc_pt_np) * np.linalg.norm(enc_jax_np) + 1e-9
        )
        print(f"encoder max_abs={max_abs:.4e}, cosine={cos:.6f}")
        self.assertGreater(cos, 0.9999, f"encoder cosine too low: {cos}")
        self.assertLess(max_abs, 1e-2)

    def test_decoder_logits_match_jax(self):
        src_np, tgt_np = self._gen_inputs()
        cfg = self.cfg

        src_pt = torch.from_numpy(src_np).long()
        tgt_pt = torch.from_numpy(tgt_np).long()
        src_mask_pt = make_padding_mask(src_pt, cfg.pad_token_id)
        tgt_pad_pt = make_padding_mask(tgt_pt, cfg.pad_token_id)
        causal_pt = make_causal_mask(tgt_pt.shape[1])
        tgt_mask_pt = causal_pt & tgt_pad_pt
        with torch.no_grad():
            logits_pt = self.pt_model(
                src_pt, tgt_pt,
                src_mask=src_mask_pt, tgt_mask=tgt_mask_pt, cross_mask=src_mask_pt,
            )
        logits_pt_np = logits_pt.cpu().numpy()

        src_jax = self.jnp.asarray(src_np)
        tgt_jax = self.jnp.asarray(tgt_np)
        src_mask_jax = (src_jax != cfg.pad_token_id)[:, None, None, :]
        tgt_pad_jax = (tgt_jax != cfg.pad_token_id)[:, None, None, :]
        causal_jax = self.jnp.tril(self.jnp.ones((tgt_jax.shape[1], tgt_jax.shape[1]), dtype=self.jnp.bool_))[None, None]
        tgt_mask_jax = causal_jax & tgt_pad_jax
        logits_jax = self.jax_model.apply(
            {"params": self.jax_params}, src_jax, tgt_jax,
            src_mask=src_mask_jax, tgt_mask=tgt_mask_jax, cross_mask=src_mask_jax,
        )
        logits_jax_np = np.asarray(logits_jax)

        max_abs = np.abs(logits_pt_np - logits_jax_np).max()
        flat_pt = logits_pt_np.flatten()
        flat_jx = logits_jax_np.flatten()
        cos = (flat_pt @ flat_jx) / (np.linalg.norm(flat_pt) * np.linalg.norm(flat_jx) + 1e-9)
        # Argmax agreement at every position is the key metric for generation.
        argmax_pt = logits_pt_np.argmax(axis=-1)
        argmax_jx = logits_jax_np.argmax(axis=-1)
        argmax_match = float((argmax_pt == argmax_jx).mean())
        print(f"decoder max_abs={max_abs:.4e}, cosine={cos:.6f}, argmax_match={argmax_match:.4f}")
        self.assertGreater(cos, 0.9999, f"decoder cosine too low: {cos}")


if __name__ == "__main__":
    unittest.main()
