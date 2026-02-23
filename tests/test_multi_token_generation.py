"""Test multi-token generation with KV cache.

Uses a simpler test without custom KV cache to verify basic multi-token
generation works with the ggml backend.
"""

import pytest
import torch
import sys

sys.path.insert(0, "python")


class TestMultiTokenGeneration:
    """Test multi-token generation with ggml backend."""

    def test_simple_two_tokens(self):
        """Test 2 token generation using pre-exported model without custom KV cache."""
        from executorch_ggml import GgmlPartitioner
        from executorch_ggml.passes import RemoveGraphAssertsPass, BroadcastCanonicalizationPass
        from executorch_ggml.passes.replace_copy_ops_pass import ReplaceCopyOpsPass
        from executorch.extension.pybindings.portable_lib import (
            _load_for_executorch_from_buffer,
        )
        from executorch.exir import to_edge_transform_and_lower
        import os

        # Use pre-exported model (without custom KV cache for simplicity)
        export_path = "/tmp/qwen3_0_6b_ggml_export/qwen3_0_6b.pt2"
        if not os.path.exists(export_path):
            pytest.skip(f"Pre-exported model not found at {export_path}")

        ep = torch.export.load(export_path)

        # Test inputs
        token1 = torch.tensor([[151644]], dtype=torch.long)
        pos1 = torch.tensor([0], dtype=torch.long)

        token2 = torch.tensor([[872]], dtype=torch.long)
        pos2 = torch.tensor([1], dtype=torch.long)

        # Get eager reference
        print("Computing eager reference...")
        with torch.no_grad():
            ref1 = ep.module()(token1, {"input_pos": pos1})
            ref2 = ep.module()(token2, {"input_pos": pos2})

        print(f"  Ref1 shape: {ref1.shape}")
        print(f"  Ref2 shape: {ref2.shape}")

        # Apply BroadcastCanonicalizationPass to make broadcasts explicit
        ep = BroadcastCanonicalizationPass().run(ep)

        # Lower to ggml
        print("Lowering to ggml...")
        edge = to_edge_transform_and_lower(
            ep,
            partitioner=[GgmlPartitioner()],
            transform_passes=[ReplaceCopyOpsPass(), RemoveGraphAssertsPass()],
        )

        # Check delegation
        delegated = sum(
            1 for n in edge.exported_program().graph.nodes
            if n.op == "call_function" and "executorch_call_delegate" in str(n.target)
        )
        total = sum(
            1 for n in edge.exported_program().graph.nodes
            if n.op == "call_function"
        )
        print(f"Delegation: {delegated}/{total} ops")

        # Run with ggml
        print("Running with ggml backend...")
        et_program = edge.to_executorch()
        pte_model = _load_for_executorch_from_buffer(et_program.buffer)

        result1 = pte_model.forward((token1, pos1))[0]
        result2 = pte_model.forward((token2, pos2))[0]

        # Compare
        diff1 = (ref1 - result1).abs().max().item()
        diff2 = (ref2 - result2).abs().max().item()

        print(f"\nResults:")
        print(f"  Token 1 max diff: {diff1:.6f}")
        print(f"  Token 2 max diff: {diff2:.6f}")

        # Check argmax matches
        ref1_argmax = ref1.view(-1).argmax().item()
        result1_argmax = result1.view(-1).argmax().item()
        ref2_argmax = ref2.view(-1).argmax().item()
        result2_argmax = result2.view(-1).argmax().item()

        print(f"  Token 1 argmax: ref={ref1_argmax}, ggml={result1_argmax}")
        print(f"  Token 2 argmax: ref={ref2_argmax}, ggml={result2_argmax}")

        assert diff1 < 1.0, f"Token 1 diff too large: {diff1}"
        assert diff2 < 1.0, f"Token 2 diff too large: {diff2}"

        print("\nMulti-token generation test PASSED!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
