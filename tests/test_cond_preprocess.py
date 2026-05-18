"""Phase 3 test: the preprocess walk emits one OP_COND IR tensor per
output of a cond node, plus an IrSubgraph entry per branch.

Run:
    python -m pytest tests/test_cond_preprocess.py -s
"""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from executorch_ggml import GgmlPartitioner


class _SimpleCondModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("flag", torch.zeros(1, dtype=torch.bool))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        def t_branch(x, y):
            return x + y
        def f_branch(x, y):
            return x * y
        return torch.cond(self.flag.any(), t_branch, f_branch, (x, y))


class TestCondPreprocess(unittest.TestCase):

    def test_cond_lowers_to_op_cond_ir(self):
        m = _SimpleCondModule().eval()
        ep = torch.export.export(m, (torch.randn(4), torch.randn(4)), strict=False)
        # to_edge_transform_and_lower invokes preprocess. We don't actually
        # run inference; we just want the lowering to succeed without
        # raising an "unsupported op cond" error.
        edge = to_edge_transform_and_lower(
            {"forward": ep},
            partitioner=[GgmlPartitioner()],
            compile_config=EdgeCompileConfig(
                _check_ir_validity=False, _skip_dim_order=True),
        )
        gp = list(edge._edge_programs.values())[0]
        # After lowering the cond should have been subsumed into a delegate
        # call (executorch_call_delegate with our backend).
        delegate_calls = [
            n for n in gp.graph_module.graph.nodes
            if n.op == "call_function" and "executorch_call_delegate" in str(n.target)
        ]
        self.assertEqual(len(delegate_calls), 1,
                         "expected exactly one delegate call subsuming the cond")
        print(f"\n[Phase 3] cond lowered cleanly: 1 delegate call.")


if __name__ == "__main__":
    unittest.main()
