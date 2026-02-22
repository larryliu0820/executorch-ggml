"""
Numerical smoke test: Qwen3-0.6B with ggml backend.

Tests Qwen3-0.6B with SDPA preserved (not decomposed) and all ops
delegated to the ggml backend. Verifies numerical accuracy against eager.

Usage:
    source .venv/bin/activate
    pytest tests/test_qwen3_numerical.py -v -s
"""

import pytest
import torch


class TestQwen3WithSDPAPreservation:
    """Test Qwen3 with SDPA preserved (not decomposed).

    Uses to_edge_transform_and_lower which preserves SDPA via
    ops_to_not_decompose, with all ops delegated to the ggml backend.
    """

    def test_single_token_with_sdpa_preserved(self):
        """Test 1-token forward with SDPA preserved and verify numerical accuracy."""
        from executorch_ggml import GgmlPartitioner
        from executorch_ggml.passes.replace_copy_ops_pass import ReplaceCopyOpsPass
        from executorch.extension.pybindings.portable_lib import (
            _load_for_executorch_from_buffer,
        )
        from executorch.exir import to_edge_transform_and_lower

        ep = torch.export.load("/tmp/qwen3_0_6b_ggml_export/qwen3_0_6b.pt2")

        # Eager reference
        tokens = torch.tensor([[151644]], dtype=torch.long)
        input_pos = torch.tensor([0], dtype=torch.long)
        with torch.no_grad():
            eager_logits = ep.module()(tokens, {"input_pos": input_pos})

        # GGML backend
        edge_mgr = to_edge_transform_and_lower(
            ep,
            partitioner=[GgmlPartitioner()],
            transform_passes=[ReplaceCopyOpsPass()],
        )
        et_module = edge_mgr.to_executorch()
        pte_model = _load_for_executorch_from_buffer(et_module.buffer)
        output = pte_model.forward((tokens, input_pos))
        ggml_logits = output[0]

        # Shape check
        eager_flat = eager_logits.detach().view(-1)
        ggml_flat = ggml_logits.detach().view(-1)
        assert eager_flat.shape == ggml_flat.shape, (
            f"Shape mismatch: eager={eager_flat.shape} ggml={ggml_flat.shape}"
        )

        # Numerical checks
        cos_sim = torch.nn.functional.cosine_similarity(
            eager_flat.unsqueeze(0), ggml_flat.unsqueeze(0)
        ).item()
        max_diff = (eager_flat - ggml_flat).abs().max().item()
        eager_argmax = eager_flat.argmax().item()
        ggml_argmax = ggml_flat.argmax().item()

        print(f"Cosine similarity: {cos_sim:.6f}")
        print(f"Max |eager - ggml|: {max_diff:.6f}")
        print(f"Eager argmax: {eager_argmax}, GGML argmax: {ggml_argmax}")

        assert torch.isfinite(ggml_flat).all()
        assert cos_sim > 0.99, f"Cosine similarity too low: {cos_sim:.4f}"
        assert max_diff < 1.0, f"Max diff too large: {max_diff:.4f}"
        assert eager_argmax == ggml_argmax, (
            f"Argmax mismatch: eager={eager_argmax} ggml={ggml_argmax}"
        )


if __name__ == "__main__":
    # Standalone smoke run.
    print("Run with pytest for full checks:")
    print("  pytest tests/test_qwen3_numerical.py -v -s")
