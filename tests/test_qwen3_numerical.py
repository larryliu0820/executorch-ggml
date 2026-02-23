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
        from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig
        from optimum.exporters.executorch.integrations import CausalLMExportableModule
        from executorch_ggml import GgmlPartitioner
        from executorch_ggml.passes.replace_copy_ops_pass import ReplaceCopyOpsPass
        from executorch_ggml.passes import RemoveGraphAssertsPass, BroadcastCanonicalizationPass
        from executorch.extension.pybindings.portable_lib import (
            _load_for_executorch_from_buffer,
        )
        from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig

        model_id = "Qwen/Qwen3-0.6B"
        max_seq_len = 128

        print(f"Loading model: {model_id}")
        config = AutoConfig.from_pretrained(model_id)

        # Disable rope scaling to avoid data-dependent control flow
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            config.rope_scaling["type"] = "default"

        eager_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cpu",
            torch_dtype=torch.float32,
            config=config,
            attn_implementation="sdpa",
            generation_config=GenerationConfig(
                use_cache=True,
                cache_implementation="static",
                max_length=max_seq_len,
                cache_config={
                    "batch_size": 1,
                    "max_cache_len": max_seq_len,
                },
            ),
        )

        print("Creating exportable module...")
        exportable = CausalLMExportableModule(
            eager_model,
            max_seq_len=max_seq_len,
            use_custom_kv_cache=False,
            use_custom_sdpa=False,
            disable_dynamic_shapes=True,
        )

        # Test inputs
        tokens = torch.tensor([[151644]], dtype=torch.long)
        cache_position = torch.tensor([0], dtype=torch.long)

        # Eager reference
        print("Computing eager reference...")
        with torch.no_grad():
            eager_out = eager_model(tokens, cache_position=cache_position)
            eager_logits = eager_out.logits.clone()

        print("Exporting model...")
        exported_progs = exportable.export()
        ep = exported_progs["model"]

        # Apply BroadcastCanonicalizationPass to make broadcasts explicit
        ep = BroadcastCanonicalizationPass().run(ep)

        # GGML backend
        print("Lowering to ggml...")
        edge_mgr = to_edge_transform_and_lower(
            ep,
            partitioner=[GgmlPartitioner()],
            compile_config=EdgeCompileConfig(
                _check_ir_validity=False,
                _skip_dim_order=True,
            ),
            transform_passes=[ReplaceCopyOpsPass(), RemoveGraphAssertsPass()],
        )

        # Check delegation
        delegate_count = sum(
            1 for n in edge_mgr.exported_program().graph.nodes
            if n.op == "call_function" and "executorch_call_delegate" in str(n.target)
        )
        print(f"Delegated calls: {delegate_count}")

        et_module = edge_mgr.to_executorch()
        print(f"Serialized .pte size: {len(et_module.buffer)} bytes")

        pte_model = _load_for_executorch_from_buffer(et_module.buffer)

        print("Running with ggml backend...")
        output = pte_model.forward((tokens, cache_position))
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
