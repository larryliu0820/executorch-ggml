"""Test Qwen3 export using optimum-executorch with ggml backend.

This tests dynamic shape export from optimum-executorch and lowering to ggml.
"""

import unittest
import torch


class TestQwen3OptimumExport(unittest.TestCase):
    """Test Qwen3 export using optimum-executorch + GgmlPartitioner."""

    def test_qwen3_dynamic_shape_export(self):
        """Test Qwen3 dynamic shape export with ggml partitioner.

        Verifies that:
        1. Export with dynamic shapes succeeds
        2. BroadcastCanonicalizationPass handles the dynamic shapes
        3. GgmlPartitioner can lower the model
        4. Serialization to .pte succeeds
        """
        from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig
        from optimum.exporters.executorch.integrations import CausalLMExportableModule
        from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig
        from executorch_ggml import GgmlPartitioner
        from executorch_ggml.passes import RemoveGraphAssertsPass, BroadcastCanonicalizationPass
        from executorch_ggml.passes.replace_copy_ops_pass import ReplaceCopyOpsPass

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
            torch_dtype=torch.bfloat16,
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

        print("Creating exportable module with dynamic shapes...")
        exportable = CausalLMExportableModule(
            eager_model,
            max_seq_len=max_seq_len,
            use_custom_kv_cache=False,
            use_custom_sdpa=False,
            disable_dynamic_shapes=False,  # Enable dynamic shapes
        )

        print("Exporting with dynamic shapes...")
        exported_progs = exportable.export()
        ep = exported_progs["model"]

        print(f"Range constraints: {ep.range_constraints}")

        # Verify dynamic shapes in the graph
        graph = ep.graph_module.graph
        has_sym_size = any("sym_size" in str(n.target) for n in graph.nodes if n.op == "call_function")
        print(f"Graph has sym_size ops: {has_sym_size}")

        # Apply BroadcastCanonicalizationPass to make broadcasts explicit
        ep = BroadcastCanonicalizationPass().run(ep)

        print("Lowering to edge with GgmlPartitioner...")
        edge_manager = to_edge_transform_and_lower(
            ep,
            partitioner=[GgmlPartitioner()],
            compile_config=EdgeCompileConfig(
                _check_ir_validity=False,
                _skip_dim_order=True,
            ),
            transform_passes=[ReplaceCopyOpsPass(), RemoveGraphAssertsPass()],
        )

        # Check delegation
        edge_program = edge_manager.exported_program()
        delegate_count = sum(
            1 for n in edge_program.graph.nodes
            if n.op == "call_function" and "executorch_call_delegate" in str(n.target)
        )
        print(f"Delegated calls: {delegate_count}")
        self.assertGreater(delegate_count, 0, "Expected at least one delegated call")

        # Serialize
        et_program = edge_manager.to_executorch()
        print(f"Serialized .pte size: {len(et_program.buffer)} bytes")

        print("\nDynamic shape export test PASSED!")

    def test_qwen3_static_shape_export(self):
        """Test Qwen3 static shape export with optimum-executorch and ggml backend.

        Verifies that:
        1. Export with static shapes succeeds
        2. BroadcastCanonicalizationPass handles the graph correctly
        3. GgmlPartitioner can lower the model
        4. Serialization to .pte succeeds
        """
        from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig
        from optimum.exporters.executorch.integrations import CausalLMExportableModule
        from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig
        from executorch_ggml import GgmlPartitioner
        from executorch_ggml.passes import RemoveGraphAssertsPass, BroadcastCanonicalizationPass
        from executorch_ggml.passes.replace_copy_ops_pass import ReplaceCopyOpsPass

        model_id = "Qwen/Qwen3-0.6B"
        max_seq_len = 128

        print(f"Loading model: {model_id}")
        config = AutoConfig.from_pretrained(model_id)

        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            config.rope_scaling["type"] = "default"

        eager_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cpu",
            torch_dtype=torch.bfloat16,
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

        print("Creating exportable module (static shapes)...")
        exportable = CausalLMExportableModule(
            eager_model,
            max_seq_len=max_seq_len,
            use_custom_kv_cache=False,
            use_custom_sdpa=False,
            disable_dynamic_shapes=True,
        )

        print("Exporting with static shapes...")
        exported_progs = exportable.export()
        ep = exported_progs["model"]

        print(f"Range constraints: {ep.range_constraints}")

        # Apply BroadcastCanonicalizationPass
        ep = BroadcastCanonicalizationPass().run(ep)

        print("Lowering to edge with GgmlPartitioner...")
        edge_manager = to_edge_transform_and_lower(
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
            1 for n in edge_manager.exported_program().graph.nodes
            if n.op == "call_function" and "executorch_call_delegate" in str(n.target)
        )
        print(f"Delegated calls: {delegate_count}")
        self.assertGreater(delegate_count, 0, "Expected at least one delegated call")

        # Serialize
        et_program = edge_manager.to_executorch()
        print(f"Serialized .pte size: {len(et_program.buffer)} bytes")

        print("\nStatic shape export test PASSED!")


if __name__ == "__main__":
    unittest.main()
