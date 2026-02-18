"""End-to-end test: nn.Linear + nn.LeakyReLU → GgmlBackend delegation."""

import unittest

import torch
import torch.nn as nn
from torch.export import export

from executorch.exir import to_edge_transform_and_lower

from executorch_ggml import GgmlPartitioner


class LinearLeakyReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 8)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leaky_relu(self.linear(x))


class TestLinearLeakyReLU(unittest.TestCase):

    def test_partition_and_lower(self):
        torch.manual_seed(42)
        model = LinearLeakyReLU().eval()
        example_input = (torch.randn(2, 4),)

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

        # Verify that delegation happened: check for lowered modules
        # After lowering, the graph should contain call_delegate nodes
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
            "Expected at least one delegated call after partitioning, "
            "but found none. Graph nodes: "
            + str([n.op + ":" + str(n.target) for n in graph.nodes]),
        )

        # Serialize to .pte (validates the full serialization pipeline)
        executorch_program = edge_manager.to_executorch()
        pte_bytes = executorch_program.buffer
        self.assertGreater(len(pte_bytes), 0, "Serialized .pte should be non-empty")

        print(f"Reference output shape: {ref_output.shape}")
        print(f"Reference output:\n{ref_output}")
        print(f"Serialized .pte size: {len(pte_bytes)} bytes")
        print("Partition + serialization passed.")

    def test_partition_tags_created(self):
        """Verify that the partitioner creates the expected partition tags."""
        torch.manual_seed(42)
        model = LinearLeakyReLU().eval()
        example_input = (torch.randn(2, 4),)

        exported = export(model, example_input)
        partitioner = GgmlPartitioner()
        result = partitioner.partition(exported)

        self.assertGreater(
            len(result.partition_tags),
            0,
            "Expected at least one partition tag",
        )
        for tag, spec in result.partition_tags.items():
            self.assertEqual(spec.backend_id, "GgmlBackend")
            self.assertTrue(tag.startswith("ggml_partition_"))


if __name__ == "__main__":
    unittest.main()
