"""Test Qwen3 export using optimum-executorch with ggml backend.

This tests dynamic shape export from optimum-executorch and lowering to ggml.
"""

import unittest
import torch


class TestQwen3OptimumExport(unittest.TestCase):
    """Test Qwen3 export using optimum-executorch + GgmlPartitioner."""

    def test_qwen3_dynamic_shape_export(self):
        """Test Qwen3 dynamic shape export with ggml partitioner."""
        from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig
        from optimum.exporters.executorch.integrations import CausalLMExportableModule
        from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig
        from executorch_ggml import GgmlPartitioner
        from executorch_ggml.passes import RemoveGraphAssertsPass, BroadcastCanonicalizationPass
        from executorch_ggml.passes.replace_copy_ops_pass import ReplaceCopyOpsPass

        model_id = "Qwen/Qwen3-0.6B"
        max_seq_len = 128  # Small for testing

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
            disable_dynamic_shapes=False,  # Enable dynamic shapes
        )

        print("Exporting with dynamic shapes...")
        exported_progs = exportable.export()
        ep = exported_progs["model"]

        print(f"Exported program graph signature: {ep.graph_signature}")
        print(f"Range constraints: {ep.range_constraints}")

        # Print some graph info
        graph = ep.graph_module.graph
        op_counts = {}
        for node in graph.nodes:
            if node.op == "call_function":
                op_name = str(node.target).split(".")[-1]
                op_counts[op_name] = op_counts.get(op_name, 0) + 1
        print(f"Op counts (top 10): {dict(sorted(op_counts.items(), key=lambda x: -x[1])[:10])}")

        # Apply BroadcastCanonicalizationPass to make broadcasts explicit
        ep = BroadcastCanonicalizationPass().run(ep)

        print("Lowering to edge with GgmlPartitioner...")
        try:
            edge_manager = to_edge_transform_and_lower(
                ep,
                partitioner=[GgmlPartitioner()],
                compile_config=EdgeCompileConfig(
                    _check_ir_validity=False,
                    _skip_dim_order=True,
                ),
                transform_passes=[ReplaceCopyOpsPass(), RemoveGraphAssertsPass()],
            )

            edge_program = edge_manager.exported_program()
            graph = edge_program.graph_module.graph

            delegate_count = 0
            non_delegate_ops = []
            for node in graph.nodes:
                if node.op == "call_function":
                    if "executorch_call_delegate" in str(node.target):
                        delegate_count += 1
                    elif "getitem" not in str(node.target):
                        non_delegate_ops.append(str(node.target))

            print(f"Delegated calls: {delegate_count}")
            print(f"Non-delegated ops ({len(non_delegate_ops)}): {non_delegate_ops[:20]}")

            # Serialize
            et_program = edge_manager.to_executorch()
            print(f"Serialized .pte size: {len(et_program.buffer)} bytes")

            self.assertGreater(delegate_count, 0, "Expected at least one delegated call")

        except Exception as e:
            print(f"Lowering failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    def test_qwen3_static_shape_export(self):
        """Test Qwen3 static shape export (disable_dynamic_shapes=True) with ggml partitioner."""
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

        print("Creating exportable module (static shapes)...")
        exportable = CausalLMExportableModule(
            eager_model,
            max_seq_len=max_seq_len,
            use_custom_kv_cache=False,
            use_custom_sdpa=False,
            disable_dynamic_shapes=True,  # Static shapes
        )

        print("Exporting with static shapes...")
        exported_progs = exportable.export()
        ep = exported_progs["model"]

        print(f"Range constraints: {ep.range_constraints}")

        # Apply BroadcastCanonicalizationPass to make broadcasts explicit
        ep = BroadcastCanonicalizationPass().run(ep)

        print("Lowering to edge with GgmlPartitioner...")
        try:
            edge_manager = to_edge_transform_and_lower(
                ep,
                partitioner=[GgmlPartitioner()],
                compile_config=EdgeCompileConfig(
                    _check_ir_validity=False,
                    _skip_dim_order=True,
                ),
                transform_passes=[ReplaceCopyOpsPass(), RemoveGraphAssertsPass()],
            )

            edge_program = edge_manager.exported_program()
            graph = edge_program.graph_module.graph

            delegate_count = 0
            non_delegate_ops = []
            for node in graph.nodes:
                if node.op == "call_function":
                    if "executorch_call_delegate" in str(node.target):
                        delegate_count += 1
                    elif "getitem" not in str(node.target):
                        non_delegate_ops.append(str(node.target))

            print(f"Delegated calls: {delegate_count}")
            print(f"Non-delegated ops ({len(non_delegate_ops)}): {non_delegate_ops[:20]}")

            et_program = edge_manager.to_executorch()
            print(f"Serialized .pte size: {len(et_program.buffer)} bytes")

            self.assertGreater(delegate_count, 0, "Expected at least one delegated call")

        except Exception as e:
            print(f"Lowering failed: {e}")
            import traceback
            traceback.print_exc()
            raise


if __name__ == "__main__":
    unittest.main()
