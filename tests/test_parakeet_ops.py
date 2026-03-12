"""
End-to-end tests for Parakeet (FastConformer ASR) ops through the ggml backend.

Each test exports a small PyTorch model, lowers it to the ggml backend,
and verifies the lowering succeeds. Runtime tests require the native extension.

Usage:
    FLATC_EXECUTABLE=$(python -c "import executorch; print(executorch.__path__[0])")/data/bin/flatc \
    GGML_BACKEND_DEVICE=cpu pytest tests/test_parakeet_ops.py -v -s
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.export import export

from executorch_ggml import GgmlPartitioner
from executorch_ggml.edge_pipeline import to_edge_rewrite_and_lower
from executorch_ggml.passes.replace_copy_ops_pass import ReplaceCopyOpsPass

_NATIVE_AVAILABLE = False
try:
    from executorch.extension.pybindings.portable_lib import (
        _load_for_executorch_from_buffer,
    )

    _NATIVE_AVAILABLE = True
except Exception:
    pass

requires_native = pytest.mark.skipif(
    not _NATIVE_AVAILABLE, reason="Native ggml backend extension not available"
)


def _lower(model, inp, transform_passes=None):
    """Export and lower to ggml, returning the EdgeProgramManager."""
    model.eval()
    with torch.no_grad():
        ep = export(model, (inp,))
    return to_edge_rewrite_and_lower(
        ep,
        transform_passes=transform_passes,
        partitioner=[GgmlPartitioner()],
    )


def _run_ggml(model, inp, transform_passes=None):
    """Export, lower, execute, return (eager_out, ggml_out)."""
    edge_mgr = _lower(model, inp, transform_passes)
    et = edge_mgr.to_executorch()
    pte = _load_for_executorch_from_buffer(et.buffer)
    result = pte.forward((inp,))
    ggml_out = result[0]
    with torch.no_grad():
        eager_out = model(inp)
    return eager_out, ggml_out


def _assert_close(eager, ggml, atol=1e-4, rtol=1e-4):
    diff = (eager - ggml).abs().max().item()
    assert diff < atol, f"Max abs diff {diff} >= {atol}"


# ---------------------------------------------------------------------------
# Lowering-only tests
# ---------------------------------------------------------------------------


class TestLoweringOnly:
    """Tests that export + lowering succeeds without crashing."""

    def test_relu(self):
        class ReluModel(nn.Module):
            def forward(self, x):
                return F.relu(x)

        _lower(ReluModel(), torch.randn(2, 16))

    def test_tanh(self):
        class TanhModel(nn.Module):
            def forward(self, x):
                return torch.tanh(x)

        _lower(TanhModel(), torch.randn(2, 16))

    def test_layer_norm(self):
        class LNModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.ln = nn.LayerNorm(16)

            def forward(self, x):
                return self.ln(x)

        _lower(LNModel(), torch.randn(2, 4, 16))

    def test_layer_norm_no_affine(self):
        class LNModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.ln = nn.LayerNorm(16, elementwise_affine=False)

            def forward(self, x):
                return self.ln(x)

        _lower(LNModel(), torch.randn(2, 4, 16))

    def test_conv1d(self):
        class Conv1dModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv1d(16, 32, kernel_size=3, padding=1)

            def forward(self, x):
                return self.conv(x)

        _lower(Conv1dModel(), torch.randn(1, 16, 64))

    def test_conv1d_depthwise(self):
        class DWConv1dModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv1d(16, 16, kernel_size=3, padding=1, groups=16)

            def forward(self, x):
                return self.conv(x)

        _lower(DWConv1dModel(), torch.randn(1, 16, 64))

    def test_argmax(self):
        class ArgmaxModel(nn.Module):
            def forward(self, x):
                return torch.argmax(x, dim=-1)

        _lower(ArgmaxModel(), torch.randn(2, 10))

    def test_squeeze(self):
        class SqueezeModel(nn.Module):
            def forward(self, x):
                return x.squeeze(1)

        _lower(SqueezeModel(), torch.randn(2, 1, 16), transform_passes=[ReplaceCopyOpsPass()])

    def test_split_with_sizes(self):
        class SplitModel(nn.Module):
            def forward(self, x):
                a, b = torch.split(x, [8, 8], dim=-1)
                return a * torch.sigmoid(b)  # GLU gate

        _lower(SplitModel(), torch.randn(2, 16))

    def test_conv1d_relu(self):
        class Conv1dReLU(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv1d(16, 32, kernel_size=3, padding=1)

            def forward(self, x):
                return F.relu(self.conv(x))

        _lower(Conv1dReLU(), torch.randn(1, 16, 64))

    def test_layer_norm_relu(self):
        class LNReLU(nn.Module):
            def __init__(self):
                super().__init__()
                self.ln = nn.LayerNorm(16)

            def forward(self, x):
                return F.relu(self.ln(x))

        _lower(LNReLU(), torch.randn(2, 4, 16))


# ---------------------------------------------------------------------------
# Runtime tests (require native extension)
# ---------------------------------------------------------------------------


class TestRuntime:
    """Tests that run through the native ggml backend and verify numerical output."""

    @requires_native
    def test_relu(self):
        class ReluModel(nn.Module):
            def forward(self, x):
                return F.relu(x)

        eager, ggml = _run_ggml(ReluModel(), torch.randn(2, 16))
        _assert_close(eager, ggml)

    @requires_native
    def test_tanh(self):
        class TanhModel(nn.Module):
            def forward(self, x):
                return torch.tanh(x)

        eager, ggml = _run_ggml(TanhModel(), torch.randn(2, 16))
        _assert_close(eager, ggml)

    @requires_native
    def test_layer_norm(self):
        class LNModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.ln = nn.LayerNorm(16)

            def forward(self, x):
                return self.ln(x)

        torch.manual_seed(42)
        eager, ggml = _run_ggml(LNModel(), torch.randn(2, 4, 16))
        _assert_close(eager, ggml, atol=1e-3)

    @requires_native
    def test_conv1d(self):
        class Conv1dModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv1d(16, 32, kernel_size=3, padding=1)

            def forward(self, x):
                return self.conv(x)

        torch.manual_seed(42)
        eager, ggml = _run_ggml(Conv1dModel(), torch.randn(1, 16, 64))
        _assert_close(eager, ggml, atol=0.01)  # F16 weight downcast adds ~1e-3 error

    @requires_native
    def test_argmax(self):
        class ArgmaxModel(nn.Module):
            def forward(self, x):
                return torch.argmax(x, dim=-1)

        torch.manual_seed(42)
        inp = torch.randn(2, 10)
        eager, ggml = _run_ggml(ArgmaxModel(), inp)
        # argmax returns int64 (eager) vs int32 (ggml) — compare values
        assert (eager.int() == ggml.int()).all(), f"Argmax mismatch: {eager} vs {ggml}"

    @requires_native
    def test_squeeze(self):
        class SqueezeModel(nn.Module):
            def forward(self, x):
                return x.squeeze(1)

        eager, ggml = _run_ggml(
            SqueezeModel(),
            torch.randn(2, 1, 16),
            transform_passes=[ReplaceCopyOpsPass()],
        )
        _assert_close(eager, ggml)

    @requires_native
    def test_split_glu(self):
        class GLUModel(nn.Module):
            def forward(self, x):
                a, b = torch.split(x, [8, 8], dim=-1)
                return a * torch.sigmoid(b)

        torch.manual_seed(42)
        eager, ggml = _run_ggml(GLUModel(), torch.randn(2, 16))
        _assert_close(eager, ggml)

    @requires_native
    def test_conv1d_relu_combined(self):
        class Conv1dReLU(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv1d(16, 32, kernel_size=3, padding=1)

            def forward(self, x):
                return F.relu(self.conv(x))

        torch.manual_seed(42)
        eager, ggml = _run_ggml(Conv1dReLU(), torch.randn(1, 16, 64))
        _assert_close(eager, ggml, atol=0.01)  # F16 weight downcast

    @requires_native
    def test_layer_norm_tanh_combined(self):
        class LNTanh(nn.Module):
            def __init__(self):
                super().__init__()
                self.ln = nn.LayerNorm(16)

            def forward(self, x):
                return torch.tanh(self.ln(x))

        torch.manual_seed(42)
        eager, ggml = _run_ggml(LNTanh(), torch.randn(2, 4, 16))
        _assert_close(eager, ggml, atol=1e-3)
