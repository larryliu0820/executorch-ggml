"""Regression tests for aten.index.Tensor multi-index lowering to ggml."""

import unittest

import torch
from torch.export import export

from executorch_ggml import GgmlPartitioner
from executorch_ggml.edge_pipeline import to_edge_rewrite_and_lower
from executorch_ggml.passes.broadcast_pass import BroadcastCanonicalizationPass as BroadcastPass


def _run_ggml(model: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Export + lower + execute with ggml backend, returning first output tensor."""
    from executorch.extension.pybindings.portable_lib import _load_for_executorch_from_buffer

    exported = export(model, inputs)
    edge_mgr = to_edge_rewrite_and_lower(
        exported,
        ep_passes=[BroadcastPass()],
        partitioner=[GgmlPartitioner()],
    )

    # Ensure partitioning actually delegated a subgraph to ggml.
    graph = edge_mgr.exported_program().graph_module.graph
    has_delegate = any(
        n.op == "call_function" and "executorch_call_delegate" in str(n.target)
        for n in graph.nodes
    )
    assert has_delegate, "Expected delegated call in lowered graph"

    et_module = edge_mgr.to_executorch()
    pte_model = _load_for_executorch_from_buffer(et_module.buffer)
    return pte_model.forward(inputs)[0]


class MultiIndex2D(torch.nn.Module):
    def forward(self, x, idx0, idx1):
        return x[(idx0, idx1)]


class MultiIndex3D(torch.nn.Module):
    def forward(self, x, idx0, idx1, idx2):
        return x[(idx0, idx1, idx2)]


class TestMultiIndex(unittest.TestCase):
    def test_2d_broadcast_float(self):
        model = MultiIndex2D().eval()
        x = torch.randn(5, 7, dtype=torch.float32)
        idx0 = torch.tensor([[0], [3]], dtype=torch.long)      # [2, 1]
        idx1 = torch.tensor([[1, 2, 3]], dtype=torch.long)      # [1, 3]

        ref = model(x, idx0, idx1)
        out = _run_ggml(model, (x, idx0, idx1))

        self.assertEqual(tuple(out.shape), tuple(ref.shape))
        self.assertTrue(torch.equal(out, ref), f"max diff: {(out - ref).abs().max().item()}")

    def test_2d_broadcast_int64(self):
        model = MultiIndex2D().eval()
        x = torch.randint(-1000, 1000, (5, 7), dtype=torch.int64)
        idx0 = torch.tensor([[4], [1]], dtype=torch.long)       # [2, 1]
        idx1 = torch.tensor([[0, 2, 6]], dtype=torch.long)      # [1, 3]

        ref = model(x, idx0, idx1)
        out = _run_ggml(model, (x, idx0, idx1))

        self.assertEqual(tuple(out.shape), tuple(ref.shape))
        self.assertTrue(torch.equal(out, ref))

    def test_3d_broadcast_negative_indices(self):
        model = MultiIndex3D().eval()
        x = torch.randn(4, 5, 6, dtype=torch.float32)
        idx0 = torch.tensor([[[0]], [[-1]]], dtype=torch.long)   # [2, 1, 1]
        idx1 = torch.tensor([[[1], [3]]], dtype=torch.long)      # [1, 2, 1]
        idx2 = torch.tensor([[[2, 4, 5]]], dtype=torch.long)     # [1, 1, 3]

        ref = model(x, idx0, idx1, idx2)
        out = _run_ggml(model, (x, idx0, idx1, idx2))

        self.assertEqual(tuple(out.shape), tuple(ref.shape))
        self.assertTrue(torch.equal(out, ref), f"max diff: {(out - ref).abs().max().item()}")


if __name__ == "__main__":
    unittest.main()
