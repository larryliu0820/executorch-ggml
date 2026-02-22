"""Test MobileNetV2 end-to-end with BN folding.

This test verifies that MobileNetV2 can be exported, lowered with BN folding,
and the entire graph gets delegated to the ggml backend.
"""

import unittest

import torch
from torch.export import export
from torchvision.models import mobilenet_v2

from executorch_ggml import GgmlPartitioner, to_edge_rewrite_and_lower
from executorch_ggml.passes import BatchNormFoldingRewritePass


class TestMV2(unittest.TestCase):
    """Test MobileNetV2 for GGML backend with BN folding."""

    def test_mv2_e2e(self):
        """Test full MobileNetV2 export and delegation."""
        torch.manual_seed(42)
        model = mobilenet_v2(weights=None).eval()
        example_input = (torch.randn(1, 3, 224, 224),)

        # Compute reference output
        with torch.no_grad():
            ref_output = model(*example_input)

        # Export
        exported = export(model, example_input)

        # Edge transform and lower with BN folding
        edge_manager = to_edge_rewrite_and_lower(
            exported,
            ep_passes=[BatchNormFoldingRewritePass()],
            partitioner=[GgmlPartitioner()],
        )

        # Verify delegation
        edge_program = edge_manager.exported_program()
        graph = edge_program.graph_module.graph

        delegate_count = 0
        non_delegate_ops = []
        for node in graph.nodes:
            if node.op == "call_function":
                if "executorch_call_delegate" in str(node.target):
                    delegate_count += 1
                else:
                    # Track non-delegated ops (excluding getitem which is just tuple unpacking)
                    if "getitem" not in str(node.target):
                        non_delegate_ops.append(str(node.target))

        print(f"Reference output shape: {ref_output.shape}")
        print(f"Number of delegated calls: {delegate_count}")
        if non_delegate_ops:
            print(f"Non-delegated ops: {non_delegate_ops}")

        # The entire graph should be delegated as a single subgraph
        self.assertEqual(delegate_count, 1, "Expected exactly one delegated call (whole graph)")
        self.assertEqual(len(non_delegate_ops), 0, f"Expected no non-delegated ops, got: {non_delegate_ops}")

        # Serialize
        executorch_program = edge_manager.to_executorch()
        pte_bytes = executorch_program.buffer
        self.assertGreater(len(pte_bytes), 0)

        print(f"Serialized .pte size: {len(pte_bytes)} bytes")


if __name__ == "__main__":
    unittest.main()
