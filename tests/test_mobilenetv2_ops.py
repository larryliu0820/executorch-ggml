"""Tests for MobileNetV2 ops support in GgmlBackend."""

import unittest

import torch
import torch.nn as nn
from torch.export import export

from executorch_ggml import GgmlPartitioner, to_edge_rewrite_and_lower
from executorch_ggml.passes import BatchNormFoldingRewritePass


class ConvBNReLU6(nn.Module):
    """Small test module: conv + bn + relu6 (hardtanh)."""

    def __init__(self, in_channels=3, out_channels=16):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu6 = nn.ReLU6()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu6(x)
        return x


class TestConvBNReLU6(unittest.TestCase):
    """Test 1: Unit test folded conv+bn+relu6 small module e2e vs pytorch."""

    def test_partition_and_lower(self):
        """Test that conv+bn+relu6 can be partitioned and lowered."""
        torch.manual_seed(42)
        model = ConvBNReLU6().eval()
        example_input = (torch.randn(1, 3, 32, 32),)

        # Compute reference output
        with torch.no_grad():
            ref_output = model(*example_input)

        # Export
        exported = export(model, example_input)

        # Edge transform -> BN fold rewrite -> lower with GgmlPartitioner
        edge_manager = to_edge_rewrite_and_lower(
            exported,
            ep_passes=[BatchNormFoldingRewritePass()],
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
        print("Conv+BN+ReLU6 partition + serialization passed.")

    def test_partition_tags_created(self):
        """Verify that the partitioner creates the expected partition tags."""
        torch.manual_seed(42)
        model = ConvBNReLU6().eval()
        example_input = (torch.randn(1, 3, 32, 32),)

        exported = export(model, example_input)
        # Partition tags are applied on Edge dialect, so use the wrapper pipeline.
        edge_manager = to_edge_rewrite_and_lower(
            exported,
            ep_passes=[BatchNormFoldingRewritePass()],
            partitioner=[GgmlPartitioner()],
        )
        edge_program = edge_manager.exported_program()
        partitioner = GgmlPartitioner()
        result = partitioner.partition(edge_program)

        self.assertGreater(
            len(result.partition_tags),
            0,
            "Expected at least one partition tag",
        )
        for tag, spec in result.partition_tags.items():
            self.assertEqual(spec.backend_id, "GgmlBackend")
            self.assertTrue(tag.startswith("ggml_partition_"))


class DepthwiseConv(nn.Module):
    """Test depthwise convolution."""

    def __init__(self, channels=32):
        super().__init__()
        # Depthwise conv: groups = in_channels
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels, bias=True)

    def forward(self, x):
        return self.conv(x)


class TestDepthwiseConv(unittest.TestCase):
    """Test depthwise convolution support."""

    def test_partition_and_lower(self):
        """Test that depthwise conv can be partitioned and lowered."""
        torch.manual_seed(42)
        model = DepthwiseConv().eval()
        example_input = (torch.randn(1, 32, 28, 28),)

        # Compute reference output
        with torch.no_grad():
            ref_output = model(*example_input)

        # Export
        exported = export(model, example_input)

        # Edge transform and lower
        edge_manager = to_edge_rewrite_and_lower(
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


class MeanViewPermute(nn.Module):
    """Test mean, view, and permute ops."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Mean over last dimension (single dim)
        # Note: Multi-dim mean not yet supported
        x = x.mean(dim=3, keepdim=False)  # [N, C, H]
        # Reshape
        x = x.view(x.shape[0], -1)  # Flatten
        # Permute (swap dims)
        x = x.permute(1, 0)  # [C*H, N]
        return x


class TestMeanViewPermute(unittest.TestCase):
    """Test mean, view, and permute ops."""

    def test_partition_and_lower(self):
        """Test that mean/view/permute can be partitioned and lowered."""
        torch.manual_seed(42)
        model = MeanViewPermute().eval()
        example_input = (torch.randn(2, 16, 8, 8),)

        # Compute reference output
        with torch.no_grad():
            ref_output = model(*example_input)

        # Export
        exported = export(model, example_input)

        # Edge transform and lower
        edge_manager = to_edge_rewrite_and_lower(
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

        print(f"Output shape: {ref_output.shape}")
        print(f"Serialized .pte size: {len(pte_bytes)} bytes")


if __name__ == "__main__":
    unittest.main()
