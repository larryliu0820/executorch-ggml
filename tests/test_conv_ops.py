"""Tests for MobileNetV2 conv ops (without BN for now)."""

import unittest

import torch
import torch.nn as nn
from torch.export import export

from executorch.exir import to_edge_transform_and_lower

from executorch_ggml import GgmlPartitioner


class ConvReLU6(nn.Module):
    """Test module: conv + relu6 (hardtanh) - NO BN for now."""

    def __init__(self, in_channels=3, out_channels=16):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu6 = nn.ReLU6()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu6(x)
        return x


class TestConvReLU6(unittest.TestCase):
    """Test conv + relu6 (without BN)."""

    def test_partition_and_lower(self):
        """Test that conv+relu6 can be partitioned and lowered."""
        torch.manual_seed(42)
        model = ConvReLU6().eval()
        example_input = (torch.randn(1, 3, 32, 32),)

        # Compute reference output
        with torch.no_grad():
            ref_output = model(*example_input)

        # Export
        exported = export(model, example_input)

        # Edge transform and lower with GgmlPartitioner
        edge_manager = to_edge_transform_and_lower(
            exported,
            partitioner=[GgmlPartitioner()],
        )

        # Verify that delegation happened
        edge_program = edge_manager.exported_program()
        graph = edge_program.graph_module.graph

        has_delegate = False
        for node in graph.nodes:
            if node.op == "call_function" and "executorch_call_delegate" in str(
                node.target
            ):
                has_delegate = True
                break

        self.assertTrue(
            has_delegate,
            "Expected at least one delegated call after partitioning",
        )

        # Serialize to .pte
        executorch_program = edge_manager.to_executorch()
        pte_bytes = executorch_program.buffer
        self.assertGreater(len(pte_bytes), 0, "Serialized .pte should be non-empty")

        print(f"Reference output shape: {ref_output.shape}")
        print(f"Serialized .pte size: {len(pte_bytes)} bytes")
        print("Conv+ReLU6 partition + serialization passed.")


class DepthwiseConvReLU(nn.Module):
    """Test depthwise convolution with ReLU."""

    def __init__(self, channels=32):
        super().__init__()
        # Depthwise conv: groups = in_channels
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))


class TestDepthwiseConv(unittest.TestCase):
    """Test depthwise convolution support."""

    def test_partition_and_lower(self):
        """Test that depthwise conv can be partitioned and lowered."""
        torch.manual_seed(42)
        model = DepthwiseConvReLU().eval()
        example_input = (torch.randn(1, 32, 28, 28),)

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

        has_delegate = False
        for node in graph.nodes:
            if node.op == "call_function" and "executorch_call_delegate" in str(node.target):
                has_delegate = True
                break

        self.assertTrue(has_delegate, "Expected at least one delegated call")

        # Serialize
        executorch_program = edge_manager.to_executorch()
        pte_bytes = executorch_program.buffer
        self.assertGreater(len(pte_bytes), 0)

        print(f"Depthwise conv output shape: {ref_output.shape}")
        print(f"Serialized .pte size: {len(pte_bytes)} bytes")


if __name__ == "__main__":
    unittest.main()
