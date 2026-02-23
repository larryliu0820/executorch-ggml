import unittest

import torch
import torch.nn as nn
from torch.export import export

from executorch_ggml.passes import BatchNormFoldingRewritePass


class ConvNoBiasBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(8)

    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        return y


class TestBatchNormFoldingRewritePass(unittest.TestCase):
    def test_rewrite_adds_bias_and_removes_bn(self):
        torch.manual_seed(0)
        m = ConvNoBiasBN().eval()
        x = torch.randn(1, 3, 16, 16)

        ep = export(m, (x,))
        res = BatchNormFoldingRewritePass().run(ep)
        ep2 = res.ep

        self.assertGreaterEqual(res.num_patterns, 1)
        self.assertEqual(res.num_folded, 1)

        # Ensure BN op is gone.
        targets = [str(n.target) for n in ep2.graph_module.graph.nodes if n.op == "call_function"]
        self.assertFalse(any("_native_batch_norm_legit_no_training" in t or "aten.batch_norm" in t for t in targets))

        # Ensure conv now has a bias argument that is a placeholder.
        conv_nodes = [
            n
            for n in ep2.graph_module.graph.nodes
            if n.op == "call_function"
            and ("aten.convolution" in str(n.target) or "aten.conv2d" in str(n.target))
        ]
        self.assertTrue(conv_nodes)
        conv = conv_nodes[0]
        self.assertIsInstance(conv.args[2], torch.fx.Node)
        self.assertEqual(conv.args[2].op, "placeholder")

        # Basic numerical sanity: the updated EP should run and match eager.
        with torch.no_grad():
            ref = m(x)
            out = ep2.module()(x)
        self.assertTrue(torch.allclose(out, ref, rtol=1e-4, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
