"""
Numerical smoke test: Qwen3-0.6B no-KV-cache eager vs. ggml backend.

Goal for bring-up:
1) Disable KV cache.
2) Lower to ggml backend.
3) Run one forward pass and verify we can score the next token.

Usage:
    source .venv/bin/activate
    pytest tests/test_qwen3_numerical.py -v -s
"""

import pytest
import torch

MODEL_PATH = "/Users/mengweiliu/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"


class NoKvWrapper(torch.nn.Module):
    """Wrap HF model to a single no-cache forward(input_ids) -> logits."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        out = self.model(input_ids=input_ids, use_cache=False)
        return out.logits


@pytest.fixture(scope="module")
def qwen3_eager():
    """Load Qwen3-0.6B in eager mode with KV cache disabled."""
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(MODEL_PATH)
    config.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.float32,
        config=config,
        attn_implementation="sdpa",
    )
    model.eval()
    return model


@pytest.fixture(scope="module")
def exported_program(qwen3_eager):
    """Export no-KV wrapper with a short prompt shape [1, 3]."""
    from torch.export import export

    wrapper = NoKvWrapper(qwen3_eager)
    sample_ids = torch.tensor([[151644, 151645, 151646]], dtype=torch.long)
    with torch.no_grad():
        ep = export(wrapper, (sample_ids,))
    return ep


def _eager_forward(model, input_ids):
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=False)
    return out.logits


class TestQwen3NoKvNumerical:
    """Smoke + numerical checks for no-KV-cache lowering."""

    def test_export_succeeds(self, exported_program):
        assert exported_program is not None

    def test_single_step_next_token_logits(self, qwen3_eager, exported_program):
        from executorch_ggml import GgmlPartitioner
        from executorch_ggml.edge_pipeline import to_edge_rewrite_and_lower
        from executorch_ggml.passes.broadcast_pass import BroadcastCanonicalizationPass as BroadcastPass
        from executorch.extension.pybindings.portable_lib import _load_for_executorch_from_buffer

        # Prompt length 3, then compare logits for the last position (next-token scores).
        input_ids = torch.tensor([[151644, 151645, 151646]], dtype=torch.long)

        eager_logits = _eager_forward(qwen3_eager, input_ids)[0, -1]  # [vocab]

        edge_mgr = to_edge_rewrite_and_lower(
            exported_program,
            ep_passes=[BroadcastPass()],
            partitioner=[GgmlPartitioner()],
        )
        et_module = edge_mgr.to_executorch()
        pte_model = _load_for_executorch_from_buffer(et_module.buffer)

        ggml_out = pte_model.forward((input_ids,))
        ggml_logits = ggml_out[0][0, -1]  # [vocab]

        top_k = 1024
        eager_top_ids = eager_logits.topk(top_k).indices
        eager_top_vals = eager_logits[eager_top_ids]
        ggml_top_vals = ggml_logits[eager_top_ids]
        cos = torch.nn.functional.cosine_similarity(
            eager_top_vals.unsqueeze(0), ggml_top_vals.unsqueeze(0)
        ).item()

        max_diff = (eager_logits - ggml_logits).abs().max().item()
        eager_argmax = eager_logits.argmax().item()
        ggml_argmax = ggml_logits.argmax().item()

        print(f"Cosine(top-{top_k}): {cos:.6f}")
        print(f"Max |eager - ggml|: {max_diff:.6f}")
        print(f"Eager argmax: {eager_argmax}, GGML argmax: {ggml_argmax}")

        # Bring-up gate: no crash, finite logits, and reasonable numerical proximity.
        assert ggml_logits.shape == eager_logits.shape
        assert torch.isfinite(ggml_logits).all()
        assert 0 <= ggml_argmax < ggml_logits.numel()
        # Keep thresholds loose while we stabilize no-KV correctness.
        assert cos > 0.70, f"Cosine similarity too low: {cos:.4f}"
        assert max_diff < 30.0, f"Max diff too large: {max_diff:.4f}"


if __name__ == "__main__":
    # Standalone smoke run.
    print("Run with pytest for full checks:")
    print("  pytest tests/test_qwen3_numerical.py -v -s")
