"""Phase 1 test: GgmlPartitioner correctly tags torch.cond nodes when
both branches are fully supported, and leaves them un-tagged otherwise.

Run:
    python -m pytest tests/test_cond_partition.py -s
"""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from executorch_ggml import GgmlPartitioner


class _SimpleCondModule(nn.Module):
    """Tiny module: torch.cond chooses between (x+y) and (x*y)."""

    def __init__(self):
        super().__init__()
        self.register_buffer("flag", torch.zeros(1, dtype=torch.bool))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        def t_branch(x, y):
            return x + y
        def f_branch(x, y):
            return x * y
        return torch.cond(self.flag.any(), t_branch, f_branch, (x, y))


class _CondWithUnsupportedOp(nn.Module):
    """Cond branch contains an op the partitioner doesn't support.

    `torch.special.i0e` lowers to `aten.special_i0e.default` which the
    partitioner does not list in `_SUPPORTED_OP_NAMES` and doesn't get
    decomposed into supported primitives.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("flag", torch.zeros(1, dtype=torch.bool))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        def t(x, y):
            return x + y
        def f(x, y):
            return torch.special.i0e(x) + y
        return torch.cond(self.flag.any(), t, f, (x, y))


def _is_cond(node) -> bool:
    if node.op != "call_function":
        return False
    t = node.target
    return t is torch.ops.higher_order.cond or (
        getattr(t, "__name__", "") == "cond" and type(t).__name__ == "CondOp"
    )


def _count_cond_nodes(ep) -> int:
    return sum(1 for node in ep.graph_module.graph.nodes if _is_cond(node))


class TestCondPartition(unittest.TestCase):

    def test_supported_cond_gets_tagged(self):
        """The partitioner stage tags the cond region for delegation.

        We verify partition tagging at the post-`to_edge` stage (just
        after `partition()` runs) rather than running the full
        `to_edge_transform_and_lower` pipeline, because the GgmlBackend
        preprocess does not yet emit IR for cond — that's Phase 3. Once
        Phase 3 lands, we can switch this test to run the full pipeline
        and check for `executorch_call_delegate`.
        """
        from executorch.exir import to_edge
        m = _SimpleCondModule().eval()
        ep = torch.export.export(m, (torch.randn(4), torch.randn(4)), strict=False)
        edge_unpartitioned = to_edge(
            ep,
            compile_config=EdgeCompileConfig(
                _check_ir_validity=False, _skip_dim_order=True),
        )
        gp = list(edge_unpartitioned._edge_programs.values())[0]
        # Run the partitioner directly so we don't trigger preprocess.
        result = GgmlPartitioner().partition(gp)
        tagged_cond = [
            n for n in result.tagged_exported_program.graph_module.graph.nodes
            if _is_cond(n) and "delegation_tag" in n.meta
        ]
        print(f"\n[supported] tagged cond nodes: {len(tagged_cond)}")
        self.assertEqual(len(tagged_cond), 1,
                         "expected the cond node to be tagged for delegation")
        # Both subgraph get_attr nodes should also be tagged.
        cond = tagged_cond[0]
        for branch_attr in (cond.args[1], cond.args[2]):
            self.assertIsInstance(branch_attr, torch.fx.Node)
            self.assertEqual(branch_attr.op, "get_attr")
            self.assertIn("delegation_tag", branch_attr.meta,
                          f"{branch_attr.name} subgraph attr was not tagged")

    def test_unsupported_cond_not_tagged(self):
        m = _CondWithUnsupportedOp().eval()
        ep = torch.export.export(m, (torch.randn(4), torch.randn(4)), strict=False)
        edge = to_edge_transform_and_lower(
            {"forward": ep},
            partitioner=[GgmlPartitioner()],
            compile_config=EdgeCompileConfig(
                _check_ir_validity=False, _skip_dim_order=True),
        )
        gp = list(edge._edge_programs.values())[0]
        n_cond = _count_cond_nodes(gp)
        # Cond should still be in the top-level graph (unhandled by us, ET
        # outer executor walks into the branches).
        print(f"\n[unsupported] cond nodes still in top-level graph: {n_cond}")
        self.assertEqual(n_cond, 1,
                         "unsupported cond should remain in the top-level graph")


if __name__ == "__main__":
    unittest.main()
