"""Test 3: MobileNetV2 partial forward comparison.

This test creates a partial MobileNetV2 block and verifies partitioning/lowering.
"""

import unittest

import torch
import torch.nn as nn
from torch.export import export

from executorch.exir import to_edge_transform_and_lower

from executorch_ggml import GgmlPartitioner


class InvertedResidualBlock(nn.Module):
    """Simplified MobileNetV2 InvertedResidual block.

    Structure: 1x1 conv (expand) -> 3x3 depthwise conv -> 1x1 conv (project) -> BN -> ReLU6
    """

    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=6):
        super().__init__()
        self.stride = stride
        hidden_dim = in_channels * expand_ratio
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []
        # Expansion phase (1x1 conv)
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            ])

        # Depthwise phase (3x3 depthwise conv)
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
        ])

        # Projection phase (1x1 conv, linear - no activation)
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class TestMV2PartialForward(unittest.TestCase):
    """Test partial MobileNetV2 forward pass."""

    def test_inverted_residual_block(self):
        """Test that an InvertedResidual block can be partitioned and lowered."""
        torch.manual_seed(42)
        model = InvertedResidualBlock(16, 24, stride=1, expand_ratio=6).eval()
        example_input = (torch.randn(1, 16, 56, 56),)

        # Compute reference output
        with torch.no_grad():
            ref_output = model(*example_input)

        # Export
        exported = export(model, example_input)

        # Edge transform and lower
        edge_manager = to_edge_transform_and_lower(
            exported,
            partitioner=[GgmlPartitioner()],
        )

        # Verify delegation
        edge_program = edge_manager.exported_program()
        graph = edge_program.graph_module.graph

        delegate_count = 0
        for node in graph.nodes:
            if node.op == "call_function" and "executorch_call_delegate" in str(node.target):
                delegate_count += 1

        print(f"Reference output shape: {ref_output.shape}")
        print(f"Number of delegated calls: {delegate_count}")

        self.assertGreater(delegate_count, 0, "Expected at least one delegated call")

        # Serialize
        executorch_program = edge_manager.to_executorch()
        pte_bytes = executorch_program.buffer
        self.assertGreater(len(pte_bytes), 0)

        print(f"Serialized .pte size: {len(pte_bytes)} bytes")

    def test_inverted_residual_block_no_expand(self):
        """Test InvertedResidual block without expansion (expand_ratio=1)."""
        torch.manual_seed(42)
        model = InvertedResidualBlock(16, 24, stride=2, expand_ratio=1).eval()
        example_input = (torch.randn(1, 16, 56, 56),)

        # Compute reference output
        with torch.no_grad():
            ref_output = model(*example_input)

        # Export
        exported = export(model, example_input)

        # Edge transform and lower
        edge_manager = to_edge_transform_and_lower(
            exported,
            partitioner=[GgmlPartitioner()],
        )

        # Verify delegation
        edge_program = edge_manager.exported_program()
        graph = edge_program.graph_module.graph

        delegate_count = 0
        for node in graph.nodes:
            if node.op == "call_function" and "executorch_call_delegate" in str(node.target):
                delegate_count += 1

        print(f"Reference output shape: {ref_output.shape}")
        print(f"Number of delegated calls: {delegate_count}")

        self.assertGreater(delegate_count, 0, "Expected at least one delegated call")

        # Serialize
        executorch_program = edge_manager.to_executorch()
        pte_bytes = executorch_program.buffer
        self.assertGreater(len(pte_bytes), 0)

        print(f"Serialized .pte size: {len(pte_bytes)} bytes")


if __name__ == "__main__":
    unittest.main()
