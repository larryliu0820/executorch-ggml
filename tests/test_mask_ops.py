"""
Tests for mask computation ops added to support LLM attention masking.

These ops enable delegation of the mask computation chain:
arange -> cumsum -> comparisons -> boolean ops -> where
"""

import pytest
import torch
import sys

sys.path.insert(0, "python")

from executorch.exir import to_edge
from executorch_ggml import GgmlPartitioner
from executorch.extension.pybindings.portable_lib import (
    _load_for_executorch_from_buffer,
)


def run_model_test(model, inputs, rtol=1e-4, atol=1e-4):
    """Helper to test a model through the ggml backend."""
    model.eval()

    # Get reference output
    with torch.no_grad():
        ref_output = model(*inputs)

    # Export
    with torch.no_grad():
        ep = torch.export.export(model, inputs)

    # Lower to ggml
    edge = to_edge(ep)
    edge_lowered = edge.to_backend(GgmlPartitioner())
    et_program = edge_lowered.to_executorch()

    # Load and run
    pte_model = _load_for_executorch_from_buffer(et_program.buffer)
    ggml_output = pte_model.forward(inputs)[0]

    # Compare
    if isinstance(ref_output, torch.Tensor):
        assert torch.allclose(ref_output, ggml_output, rtol=rtol, atol=atol), \
            f"Max diff: {(ref_output - ggml_output).abs().max().item()}"
    return ggml_output


class TestBmmSoftmax:
    """Test BMM and Softmax ops used in attention."""

    def test_bmm_basic(self):
        class BmmModel(torch.nn.Module):
            def forward(self, a, b):
                return torch.bmm(a, b)

        model = BmmModel()
        a = torch.randn(2, 4, 8)
        b = torch.randn(2, 8, 6)
        run_model_test(model, (a, b))

    def test_softmax_dim_minus1(self):
        class SoftmaxModel(torch.nn.Module):
            def forward(self, x):
                return torch.softmax(x, dim=-1)

        model = SoftmaxModel()
        x = torch.randn(2, 4, 8)
        run_model_test(model, (x,))

    def test_bmm_softmax_attention(self):
        class AttentionModel(torch.nn.Module):
            def forward(self, q, k, v):
                scores = torch.bmm(q, k.transpose(1, 2))
                attn = torch.softmax(scores, dim=-1)
                return torch.bmm(attn, v)

        model = AttentionModel()
        q = torch.randn(1, 4, 16)
        k = torch.randn(1, 4, 16)
        v = torch.randn(1, 4, 16)
        run_model_test(model, (q, k, v))


class TestTrigOps:
    """Test cos/sin ops used in RoPE."""

    def test_cos(self):
        class CosModel(torch.nn.Module):
            def forward(self, x):
                return torch.cos(x)

        model = CosModel()
        x = torch.randn(2, 4, 8)
        run_model_test(model, (x,))

    def test_sin(self):
        class SinModel(torch.nn.Module):
            def forward(self, x):
                return torch.sin(x)

        model = SinModel()
        x = torch.randn(2, 4, 8)
        run_model_test(model, (x,))


class TestPowOps:
    """Test pow ops used in RMSNorm."""

    def test_pow_square(self):
        class PowModel(torch.nn.Module):
            def forward(self, x):
                return torch.pow(x, 2)

        model = PowModel()
        x = torch.randn(2, 4, 8)
        run_model_test(model, (x,))


class TestWhereOp:
    """Test where op used in attention masking."""

    def test_where_basic(self):
        class WhereModel(torch.nn.Module):
            def forward(self, cond, x, y):
                return torch.where(cond, x, y)

        model = WhereModel()
        cond = torch.rand(2, 4) > 0.5
        x = torch.randn(2, 4)
        y = torch.randn(2, 4)
        run_model_test(model, (cond, x, y))


class TestSigmoidOp:
    """Test sigmoid activation."""

    def test_sigmoid(self):
        class SigmoidModel(torch.nn.Module):
            def forward(self, x):
                return torch.sigmoid(x)

        model = SigmoidModel()
        x = torch.randn(2, 4, 8)
        run_model_test(model, (x,))


class TestSubMulScalar:
    """Test sub and mul_scalar ops."""

    def test_sub(self):
        class SubModel(torch.nn.Module):
            def forward(self, a, b):
                return a - b

        model = SubModel()
        a = torch.randn(2, 4, 8)
        b = torch.randn(2, 4, 8)
        run_model_test(model, (a, b))

    def test_mul_scalar(self):
        class MulScalarModel(torch.nn.Module):
            def forward(self, x):
                return x * 0.5

        model = MulScalarModel()
        x = torch.randn(2, 4, 8)
        run_model_test(model, (x,))


class TestPartitionerSupport:
    """Test that partitioner correctly identifies supported ops."""

    def test_mask_ops_in_supported_list(self):
        from executorch_ggml.ggml_partitioner import _SUPPORTED_OP_NAMES

        mask_ops = [
            "aten.arange.start_step",
            "aten.full.default",
            "aten.cumsum.default",
            "aten.eq.Scalar",
            "aten.eq.Tensor",
            "aten.ne.Scalar",
            "aten.le.Tensor",
            "aten.lt.Tensor",
            "aten.gt.Tensor",
            "aten.ge.Tensor",
            "aten.bitwise_and.Tensor",
            "aten.bitwise_or.Tensor",
            "aten.logical_not.default",
            "aten.any.dim",
        ]

        for op in mask_ops:
            assert op in _SUPPORTED_OP_NAMES, f"{op} not in supported ops"

    def test_attention_ops_in_supported_list(self):
        from executorch_ggml.ggml_partitioner import _SUPPORTED_OP_NAMES

        attention_ops = [
            "aten.bmm.default",
            "aten._softmax.default",
            "aten.scaled_dot_product_attention.default",
        ]

        for op in attention_ops:
            assert op in _SUPPORTED_OP_NAMES, f"{op} not in supported ops"


class TestSerializeOpCodes:
    """Test that serialize module has correct op codes."""

    def test_mask_op_codes_present(self):
        from executorch_ggml import serialize

        op_codes = [
            ("OP_ARANGE", 51),
            ("OP_FULL", 52),
            ("OP_CUMSUM", 53),
            ("OP_EQ", 54),
            ("OP_NE", 55),
            ("OP_LE", 56),
            ("OP_LT", 57),
            ("OP_GT", 58),
            ("OP_GE", 59),
            ("OP_BITWISE_AND", 70),
            ("OP_BITWISE_OR", 71),
            ("OP_LOGICAL_NOT", 72),
            ("OP_ANY", 73),
        ]

        for name, expected_val in op_codes:
            assert hasattr(serialize, name), f"{name} not in serialize module"
            assert getattr(serialize, name) == expected_val, \
                f"{name} has wrong value: {getattr(serialize, name)} != {expected_val}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
