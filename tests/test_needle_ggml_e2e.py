"""End-to-end Needle test: PyTorch eager vs GGML-lowered .pte (via C++).

The Python `_ggml_backend.so` extension is currently incompatible with the
installed `_portable_lib.so` (different `Module::Module` ctor signatures —
the third-party submodule is ahead of the installed executorch). So we run
the .pte through the C++ `benchmark_needle` binary, which prints a stable
"NUMERICS:" summary line, and compare it against the PyTorch eager port
that the .pte was exported from.

Both sides use the same deterministic encoder prompt (see
`_make_prompt`) and the same greedy decode loop, so the numbers should
agree up to F32 reduction noise.

Run:
    python -m pytest tests/test_needle_ggml_e2e.py -s
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import unittest

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "third-party", "needle"))
sys.path.insert(0, REPO_ROOT)  # so `import export_needle_ggml` works

from executorch_ggml.models.needle import (  # noqa: E402
    NeedleModel,
    load_jax_checkpoint,
    make_causal_mask,
    make_padding_mask,
)


PTE_PATH = os.path.join(REPO_ROOT, "needle", "needle.pte")
BINARY = os.path.join(REPO_ROOT, "build_runner", "benchmark", "benchmark_needle")


def _checkpoint_path():
    from huggingface_hub import hf_hub_download
    return hf_hub_download("Cactus-Compute/needle", "needle.pkl")


def _ensure_pte():
    if os.path.exists(PTE_PATH):
        return
    from export_needle_ggml import export_needle  # noqa: E402
    ckpt = _checkpoint_path()
    export_needle(
        checkpoint_path=ckpt,
        output_path=PTE_PATH,
        max_enc_len=1024,
        max_gen_len=512,
    )


def _make_prompt(enc_len: int, vocab: int) -> np.ndarray:
    """Match `benchmark_needle.cpp`'s prompt: 1 + (i*7 + 11) % (vocab-1)."""
    i = np.arange(enc_len, dtype=np.int64)
    return 1 + (i * 7 + 11) % (vocab - 1)


def _run_binary(enc_len: int, n_decode: int, vocab: int):
    """Run benchmark_needle and parse the NUMERICS summary line."""
    if not os.path.exists(BINARY):
        raise unittest.SkipTest(
            f"benchmark_needle not built (expected at {BINARY}). "
            "Build with: cmake --build build_runner --parallel 16 -t benchmark_needle"
        )
    out = subprocess.check_output(
        [
            BINARY,
            PTE_PATH,
            "--enc-len", str(enc_len),
            "--n-decode", str(n_decode),
            "--vocab", str(vocab),
        ],
        stderr=subprocess.STDOUT,
        text=True,
    )
    m = re.search(
        r"NUMERICS: enc_sum=([-\d.eE+]+) enc_abs_sum=([-\d.eE+]+) "
        r"last_token=(-?\d+) last_logit_sum=([-\d.eE+]+)",
        out,
    )
    if not m:
        raise AssertionError(f"Could not parse NUMERICS line. Output:\n{out}")
    return {
        "enc_sum": float(m.group(1)),
        "enc_abs_sum": float(m.group(2)),
        "last_token": int(m.group(3)),
        "last_logit_sum": float(m.group(4)),
        "stdout": out,
    }


def _eager_numerics(enc_tokens_np: np.ndarray, n_decode: int):
    """Encode + n decode steps with the eager PyTorch model.

    Returns the same summary fields the C++ binary emits so the two can be
    compared directly. The decode loop matches `benchmark_needle.cpp`:
    starts from `[1]`, picks argmax at the last position each step, and
    appends. cross_mask is all-ones (no padding in the synthetic prompt).
    """
    ckpt = _checkpoint_path()
    model, cfg = load_jax_checkpoint(ckpt)
    model.eval()

    src = torch.from_numpy(enc_tokens_np).long().unsqueeze(0)  # [1, T_src]
    src_mask = make_padding_mask(src, cfg.pad_token_id)
    enc_len = src.shape[1]
    cross_mask = torch.ones(1, 1, 1, enc_len, dtype=torch.bool)

    with torch.no_grad():
        encoder_out = model.encode(src, src_mask=src_mask)
    enc_arr = encoder_out.float().cpu().numpy()
    enc_sum = float(np.sum(enc_arr[np.isfinite(enc_arr)]))
    enc_abs_sum = float(np.sum(np.abs(enc_arr[np.isfinite(enc_arr)])))

    dec_buffer = [1]
    last_token = 1
    last_logit_sum = 0.0
    for _ in range(n_decode):
        T = len(dec_buffer)
        tgt = torch.tensor([dec_buffer], dtype=torch.long)
        causal = make_causal_mask(T)
        tgt_pad = make_padding_mask(tgt, cfg.pad_token_id)
        self_mask = causal & tgt_pad
        with torch.no_grad():
            logits = model.decode(
                tgt, encoder_out, self_mask=self_mask, cross_mask=cross_mask,
            )
        last_row = logits[0, -1, :].cpu().numpy()
        last_token = int(np.argmax(last_row))
        last_logit_sum = float(np.sum(last_row))
        dec_buffer.append(last_token)

    return {
        "enc_sum": enc_sum,
        "enc_abs_sum": enc_abs_sum,
        "last_token": last_token,
        "last_logit_sum": last_logit_sum,
    }


class TestNeedleGgmlE2E(unittest.TestCase):
    """Compare the C++/ggml path against the PyTorch eager port on a shared
    deterministic prompt."""

    @classmethod
    def setUpClass(cls):
        _ensure_pte()
        cls.enc_len = 24
        cls.n_decode = 4
        cls.vocab = 8192
        cls.prompt = _make_prompt(cls.enc_len, cls.vocab)
        cls.ggml = _run_binary(cls.enc_len, cls.n_decode, cls.vocab)
        cls.eager = _eager_numerics(cls.prompt, cls.n_decode)
        print(f"\nggml: {cls.ggml}")
        print(f"eager: {cls.eager}")

    def test_encoder_sum_matches(self):
        # Encoder output sum: F32 reduction over ~12k values; tolerance
        # accounts for nondeterministic order between CUDA and CPU
        # reductions.
        self.assertAlmostEqual(
            self.ggml["enc_sum"], self.eager["enc_sum"],
            delta=max(1.0, abs(self.eager["enc_sum"]) * 5e-3),
            msg=f"enc_sum: ggml={self.ggml['enc_sum']} eager={self.eager['enc_sum']}",
        )

    def test_encoder_abs_sum_matches(self):
        self.assertAlmostEqual(
            self.ggml["enc_abs_sum"], self.eager["enc_abs_sum"],
            delta=max(1.0, abs(self.eager["enc_abs_sum"]) * 5e-3),
            msg=f"enc_abs_sum: ggml={self.ggml['enc_abs_sum']} eager={self.eager['enc_abs_sum']}",
        )

    def test_decode_argmax_matches(self):
        self.assertEqual(
            self.ggml["last_token"], self.eager["last_token"],
            msg=f"argmax disagree: ggml={self.ggml['last_token']} "
                f"eager={self.eager['last_token']}",
        )

    def test_decode_logit_sum_finite(self):
        self.assertTrue(np.isfinite(self.ggml["last_logit_sum"]))
        self.assertAlmostEqual(
            self.ggml["last_logit_sum"], self.eager["last_logit_sum"],
            delta=max(10.0, abs(self.eager["last_logit_sum"]) * 5e-3),
            msg=(
                f"logit_sum: ggml={self.ggml['last_logit_sum']} "
                f"eager={self.eager['last_logit_sum']}"
            ),
        )


if __name__ == "__main__":
    unittest.main()
