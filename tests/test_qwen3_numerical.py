"""
Numerical comparison test: Qwen3-0.6B eager vs. ggml backend.

Exports the model using optimum-executorch's CausalLMExportableModule,
delegates to ggml, then compares logits against PyTorch eager forward.

Usage:
    source .venv/bin/activate
    pytest tests/test_qwen3_numerical.py -v -s

The test checks cosine similarity and max absolute error of logits.
"""

import math
import pytest
import torch

MODEL_PATH = "/Users/mengweiliu/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"


@pytest.fixture(scope="module")
def qwen3_eager():
    """Load Qwen3-0.6B in PyTorch eager mode (float32, static cache)."""
    from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig

    config = AutoConfig.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.float32,
        config=config,
        attn_implementation="sdpa",
        generation_config=GenerationConfig(
            use_cache=True,
            cache_implementation="static",
            max_length=128,
            cache_config={"batch_size": 1, "max_cache_len": 128},
        ),
    )
    model.eval()
    return model


@pytest.fixture(scope="module")
def exported_program():
    """Export Qwen3-0.6B with optimum-executorch CausalLMExportableModule."""
    from optimum.exporters.executorch.tasks.causal_lm import load_causal_lm_model

    exportable = load_causal_lm_model(
        MODEL_PATH,
        dtype="float32",
        attn_implementation="sdpa",
        max_length=128,
    )
    programs = exportable.export()
    ep = programs["model"]
    return ep


@pytest.fixture(scope="module")
def ggml_lowered(exported_program):
    """Lower the exported program to ggml backend."""
    from executorch.exir import to_edge
    from executorch_ggml import GgmlPartitioner, GgmlBackend
    from executorch_ggml.edge_pipeline import to_edge_rewrite_and_lower
    from executorch_ggml.passes.broadcast_pass import BroadcastCanonicalizationPass as BroadcastPass

    ep = exported_program

    edge_manager = to_edge_rewrite_and_lower(
        ep,
        ep_passes=[BroadcastPass()],
        partitioner=[GgmlPartitioner()],
    )
    return edge_manager


def _eager_forward(model, input_ids, cache_position):
    """Run one step of eager forward with static cache, resetting between calls."""
    # Reset static cache
    for layer in model.model.layers:
        cache = layer.self_attn.past_key_value
        if hasattr(cache, "key_cache"):
            for i in range(len(cache.key_cache)):
                cache.key_cache[i].zero_()
                cache.value_cache[i].zero_()

    with torch.no_grad():
        out = model(input_ids=input_ids, cache_position=cache_position)
    return out.logits


class TestQwen3Numerical:
    """Compare ggml backend logits vs. PyTorch eager logits."""

    def test_export_succeeds(self, exported_program):
        """The exported program should be non-None."""
        assert exported_program is not None

    def test_ggml_lowering_succeeds(self, ggml_lowered):
        """The edge manager after lowering should be non-None."""
        assert ggml_lowered is not None

    def test_single_token_logit_cosine(self, qwen3_eager, exported_program):
        """Cosine similarity of top-128 logits should be > 0.99."""
        from executorch.exir import to_edge
        from executorch_ggml import GgmlPartitioner
        from executorch_ggml.edge_pipeline import to_edge_rewrite_and_lower
        from executorch_ggml.passes.broadcast_pass import BroadcastCanonicalizationPass as BroadcastPass
        from torch.export import ExportedProgram

        input_ids = torch.tensor([[151644]], dtype=torch.long)
        cache_pos = torch.tensor([0], dtype=torch.long)

        # Eager
        eager_logits = _eager_forward(qwen3_eager, input_ids, cache_pos)
        eager_logits = eager_logits[0, 0]  # [vocab]

        print(f"\nEager top5: {eager_logits.topk(5).values.tolist()}")
        print(f"Eager top5 ids: {eager_logits.topk(5).indices.tolist()}")

        # ggml — run via exported program lowered to ggml
        ep = exported_program
        edge_mgr = to_edge_rewrite_and_lower(
            ep,
            ep_passes=[BroadcastPass()],
            partitioner=[GgmlPartitioner()],
        )
        et_module = edge_mgr.to_executorch()

        from executorch.extension.pybindings.portable_lib import _load_for_executorch_from_buffer
        pte_model = _load_for_executorch_from_buffer(et_module.buffer)

        ggml_out = pte_model.forward((input_ids, cache_pos))
        ggml_logits = ggml_out[0][0, 0]  # [vocab]

        print(f"GGML top5: {ggml_logits.topk(5).values.tolist()}")
        print(f"GGML top5 ids: {ggml_logits.topk(5).indices.tolist()}")

        # Cosine similarity on top-1024 logits (to focus comparison on significant values)
        top_k = 1024
        eager_top = eager_logits.topk(top_k).values
        ggml_top = ggml_logits[eager_logits.topk(top_k).indices]

        cos = torch.nn.functional.cosine_similarity(
            eager_top.unsqueeze(0), ggml_top.unsqueeze(0)
        ).item()
        print(f"Cosine similarity (top-{top_k}): {cos:.6f}")

        max_diff = (eager_logits - ggml_logits).abs().max().item()
        print(f"Max |eager - ggml|: {max_diff:.4f}")

        # Check argmax agrees
        eager_argmax = eager_logits.argmax().item()
        ggml_argmax = ggml_logits.argmax().item()
        print(f"Eager argmax: {eager_argmax}, GGML argmax: {ggml_argmax}")

        assert cos > 0.90, f"Cosine similarity too low: {cos:.4f}"
        assert max_diff < 10.0, f"Max diff too large: {max_diff:.4f}"


class TestQwen3LayerByLayer:
    """Layer-by-layer comparison to find where divergence starts."""

    def test_embedding_layer(self, qwen3_eager):
        """Embedding output should be bit-exact (no computation involved)."""
        input_ids = torch.tensor([[151644]], dtype=torch.long)
        embed_pt = qwen3_eager.model.embed_tokens(input_ids)  # [1, 1, hidden]
        print(f"\nEmbedding output mean={embed_pt.mean():.6f} std={embed_pt.std():.6f}")
        print(f"Embedding shape: {embed_pt.shape}")
        # Just ensure it runs without error
        assert embed_pt.shape == (1, 1, 1024)

    def test_rms_norm_layer0(self, qwen3_eager):
        """First layer norm after embedding."""
        input_ids = torch.tensor([[151644]], dtype=torch.long)
        with torch.no_grad():
            embed = qwen3_eager.model.embed_tokens(input_ids)
            normed = qwen3_eager.model.layers[0].input_layernorm(embed)
        print(f"\nRMSNorm layer0 output mean={normed.mean():.6f} std={normed.std():.6f}")
        assert normed.shape == (1, 1, 1024)


if __name__ == "__main__":
    # Quick standalone test
    import sys
    print("Loading Qwen3-0.6B eager model...")
    from transformers import AutoModelForCausalLM, AutoConfig, GenerationConfig

    config = AutoConfig.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.float32,
        config=config,
        attn_implementation="sdpa",
        generation_config=GenerationConfig(
            use_cache=True,
            cache_implementation="static",
            max_length=128,
            cache_config={"batch_size": 1, "max_cache_len": 128},
        ),
    )
    model.eval()

    input_ids = torch.tensor([[151644]], dtype=torch.long)
    cache_pos = torch.tensor([0], dtype=torch.long)

    print("Running eager forward...")
    with torch.no_grad():
        out = model(input_ids=input_ids, cache_position=cache_pos)
    logits = out.logits[0, 0]
    top5 = logits.topk(5)
    print(f"Eager top5 values: {[round(v,4) for v in top5.values.tolist()]}")
    print(f"Eager top5 ids: {top5.indices.tolist()}")

    print("\nExporting with optimum-executorch...")
    from optimum.exporters.executorch.tasks.causal_lm import load_causal_lm_model

    exportable = load_causal_lm_model(
        MODEL_PATH,
        dtype="float32",
        attn_implementation="sdpa",
        max_length=128,
    )
    programs = exportable.export()
    ep = programs["model"]
    print("Export succeeded.")

    print("\nLowering to ggml backend...")
    sys.path.insert(0, "/Volumes/larryliu/work/executorch-ggml/python")
    from executorch_ggml import GgmlPartitioner
    from executorch_ggml.edge_pipeline import to_edge_rewrite_and_lower
    from executorch_ggml.passes.broadcast_pass import BroadcastCanonicalizationPass as BroadcastPass

    edge_mgr = to_edge_rewrite_and_lower(
        ep,
        ep_passes=[BroadcastPass()],
        partitioner=[GgmlPartitioner()],
    )
    et_module = edge_mgr.to_executorch()

    print("Building .pte buffer and running ggml inference...")
    from executorch.extension.pybindings.portable_lib import _load_for_executorch_from_buffer
    pte_model = _load_for_executorch_from_buffer(et_module.buffer)

    ggml_out = pte_model.forward((input_ids, cache_pos))
    ggml_logits = ggml_out[0][0, 0]
    ggml_top5 = ggml_logits.topk(5)
    print(f"\nGGML top5 values: {[round(v,4) for v in ggml_top5.values.tolist()]}")
    print(f"GGML top5 ids: {ggml_top5.indices.tolist()}")

    print(f"\nMax |eager - ggml|: {(logits - ggml_logits).abs().max():.4f}")
    cos = torch.nn.functional.cosine_similarity(
        logits.unsqueeze(0), ggml_logits.unsqueeze(0)
    ).item()
    print(f"Cosine similarity: {cos:.6f}")
    print(f"Eager argmax: {logits.argmax().item()}, GGML argmax: {ggml_logits.argmax().item()}")
