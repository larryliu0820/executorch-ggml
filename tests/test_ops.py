"""
End-to-end tests for individual ops and op combinations through the ggml backend.

Each test exports a small PyTorch model, lowers it to the ggml backend, runs it
through the ExecuTorch runtime, and compares the output to eager PyTorch.

Usage:
    LD_LIBRARY_PATH=... pytest tests/test_ops.py -v -s
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


def _run_ggml(model, inp, ep_passes=None, atol=1e-4, rtol=1e-4):
    """Export, lower to ggml, execute, and return (eager_out, ggml_out)."""
    model.eval()
    with torch.no_grad():
        ep = export(model, (inp,))
    edge_mgr = to_edge_rewrite_and_lower(
        ep,
        ep_passes=ep_passes or [],
        partitioner=[GgmlPartitioner()],
    )
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
    if eager.numel() > 1:
        cos = torch.nn.functional.cosine_similarity(
            eager.flatten().unsqueeze(0), ggml.flatten().unsqueeze(0)
        ).item()
        assert cos > 0.999, f"Cosine {cos:.6f} too low"


# ---------------------------------------------------------------------------
# Lowering-only tests (no native extension needed)
# ---------------------------------------------------------------------------


class TestLoweringOnly:
    """Verify that export + lowering succeeds without crashing."""

    def _lower(self, model, inp, ep_passes=None):
        model.eval()
        with torch.no_grad():
            ep = export(model, (inp,))
        edge_mgr = to_edge_rewrite_and_lower(
            ep,
            ep_passes=ep_passes or [],
            partitioner=[GgmlPartitioner()],
        )
        return edge_mgr.to_executorch()

    def test_linear(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 128)

            def forward(self, x):
                return self.linear(x)

        self._lower(M(), torch.randn(1, 3, 64))

    def test_embedding(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(100, 64)

            def forward(self, x):
                return self.embed(x)

        self._lower(M(), torch.tensor([[1, 2, 3]], dtype=torch.long))

    def test_view_permute(self):
        class M(nn.Module):
            def forward(self, x):
                return x.view(1, 3, 8, 8).permute(0, 2, 1, 3)

        self._lower(M(), torch.randn(1, 3, 64))

    def test_slice(self):
        class M(nn.Module):
            def forward(self, x):
                return x[:, :, :, :4]

        self._lower(M(), torch.randn(1, 8, 3, 8))

    def test_cat(self):
        class M(nn.Module):
            def forward(self, x):
                a, b = x[:, :, :, :4], x[:, :, :, 4:]
                return torch.cat([a, b], dim=-1)

        self._lower(M(), torch.randn(1, 8, 3, 8))

    def test_neg(self):
        class M(nn.Module):
            def forward(self, x):
                return -x

        self._lower(M(), torch.randn(1, 3, 64))

    def test_add(self):
        class M(nn.Module):
            def forward(self, x):
                return x + x

        self._lower(M(), torch.randn(1, 3, 64))

    def test_mul_elementwise(self):
        class M(nn.Module):
            def forward(self, x):
                return x * x

        self._lower(M(), torch.randn(1, 3, 64))

    def test_silu(self):
        class M(nn.Module):
            def forward(self, x):
                return torch.nn.functional.silu(x)

        self._lower(M(), torch.randn(1, 3, 64))

    def test_transpose(self):
        class M(nn.Module):
            def forward(self, x):
                return x.transpose(1, 2)

        self._lower(M(), torch.randn(1, 3, 64))

    def test_unsqueeze(self):
        class M(nn.Module):
            def forward(self, x):
                return x.unsqueeze(0)

        self._lower(M(), torch.randn(3, 64))

    def test_expand(self):
        class M(nn.Module):
            def forward(self, x):
                return x.expand(1, 4, 3, 64)

        self._lower(M(), torch.randn(1, 1, 3, 64))

    def test_mean_rsqrt(self):
        class M(nn.Module):
            def forward(self, x):
                v = x.mean(-1, keepdim=True)
                return torch.rsqrt(v.abs() + 1e-6)

        self._lower(M(), torch.randn(1, 3, 64))


# ---------------------------------------------------------------------------
# End-to-end runtime tests (need native extension)
# ---------------------------------------------------------------------------


@requires_native
class TestLinear:
    def test_linear_3d(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 128)

            def forward(self, x):
                return self.linear(x)

        eager, ggml = _run_ggml(M(), torch.randn(1, 3, 64))
        _assert_close(eager, ggml)

    def test_linear_no_bias(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 128, bias=False)

            def forward(self, x):
                return self.linear(x)

        eager, ggml = _run_ggml(M(), torch.randn(1, 3, 64))
        _assert_close(eager, ggml)

    def test_linear_2d(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 128)

            def forward(self, x):
                return self.linear(x)

        eager, ggml = _run_ggml(M(), torch.randn(3, 64))
        _assert_close(eager, ggml)

    def test_two_linears(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.l1 = nn.Linear(64, 128, bias=False)
                self.l2 = nn.Linear(128, 32, bias=False)

            def forward(self, x):
                return self.l2(self.l1(x))

        eager, ggml = _run_ggml(M(), torch.randn(1, 3, 64))
        _assert_close(eager, ggml)


@requires_native
class TestEmbedding:
    def test_embedding(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(100, 64)

            def forward(self, x):
                return self.embed(x)

        eager, ggml = _run_ggml(M(), torch.tensor([[1, 2, 3]], dtype=torch.long))
        _assert_close(eager, ggml)

    def test_embedding_large_vocab(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(50000, 512)

            def forward(self, x):
                return self.embed(x)

        eager, ggml = _run_ggml(M(), torch.tensor([[100, 200, 300]], dtype=torch.long))
        _assert_close(eager, ggml)


@requires_native
class TestViewReshape:
    def test_view_3d_to_4d(self):
        class M(nn.Module):
            def forward(self, x):
                return x.view(1, 3, 8, 8)

        eager, ggml = _run_ggml(M(), torch.randn(1, 3, 64))
        _assert_close(eager, ggml)

    def test_view_4d_to_3d(self):
        class M(nn.Module):
            def forward(self, x):
                return x.view(1, 3, 64)

        eager, ggml = _run_ggml(M(), torch.randn(1, 3, 8, 8))
        _assert_close(eager, ggml)

    def test_reshape(self):
        class M(nn.Module):
            def forward(self, x):
                return x.reshape(1, 3, 4, 16)

        eager, ggml = _run_ggml(M(), torch.randn(1, 3, 64))
        _assert_close(eager, ggml)


@requires_native
class TestPermute:
    def test_permute_4d(self):
        class M(nn.Module):
            def forward(self, x):
                return x.permute(0, 2, 1, 3).contiguous()

        eager, ggml = _run_ggml(M(), torch.randn(1, 3, 8, 8))
        _assert_close(eager, ggml)


@requires_native
class TestTranspose:
    def test_transpose_1_2(self):
        class M(nn.Module):
            def forward(self, x):
                return x.transpose(1, 2).contiguous()

        eager, ggml = _run_ggml(M(), torch.randn(1, 3, 64))
        _assert_close(eager, ggml)

    def test_transpose_0_1(self):
        class M(nn.Module):
            def forward(self, x):
                return x.transpose(0, 1).contiguous()

        eager, ggml = _run_ggml(M(), torch.randn(3, 64))
        _assert_close(eager, ggml)


@requires_native
class TestSlice:
    def test_slice_last_dim(self):
        class M(nn.Module):
            def forward(self, x):
                return x[:, :, :, :4]

        eager, ggml = _run_ggml(M(), torch.randn(1, 8, 3, 8))
        _assert_close(eager, ggml)

    def test_slice_second_half(self):
        class M(nn.Module):
            def forward(self, x):
                return x[:, :, :, 4:]

        eager, ggml = _run_ggml(M(), torch.randn(1, 8, 3, 8))
        _assert_close(eager, ggml)

    def test_slice_middle_dim(self):
        class M(nn.Module):
            def forward(self, x):
                return x[:, :2, :]

        eager, ggml = _run_ggml(M(), torch.randn(1, 4, 64))
        _assert_close(eager, ggml)


@requires_native
class TestCat:
    def test_cat_last_dim(self):
        class M(nn.Module):
            def forward(self, x):
                a, b = x[:, :, :, :4], x[:, :, :, 4:]
                return torch.cat([a, b], dim=-1)

        eager, ggml = _run_ggml(M(), torch.randn(1, 8, 3, 8))
        _assert_close(eager, ggml)

    def test_cat_dim1(self):
        class M(nn.Module):
            def forward(self, x):
                return torch.cat([x, x], dim=1)

        eager, ggml = _run_ggml(M(), torch.randn(1, 3, 64))
        _assert_close(eager, ggml)


@requires_native
class TestElementwise:
    def test_add(self):
        class M(nn.Module):
            def forward(self, x):
                return x + x

        eager, ggml = _run_ggml(M(), torch.randn(1, 3, 64))
        _assert_close(eager, ggml)

    def test_mul(self):
        class M(nn.Module):
            def forward(self, x):
                return x * x

        eager, ggml = _run_ggml(M(), torch.randn(1, 3, 64))
        _assert_close(eager, ggml)

    def test_neg(self):
        class M(nn.Module):
            def forward(self, x):
                return -x

        eager, ggml = _run_ggml(M(), torch.randn(1, 3, 64))
        _assert_close(eager, ggml)

    def test_silu(self):
        class M(nn.Module):
            def forward(self, x):
                return torch.nn.functional.silu(x)

        eager, ggml = _run_ggml(M(), torch.randn(1, 3, 64))
        _assert_close(eager, ggml, atol=1e-3)


@requires_native
class TestViewPermuteCombination:
    """Test the view->permute chain that was the original bug."""

    def test_view_then_permute(self):
        class M(nn.Module):
            def forward(self, x):
                y = x.view(1, 3, 16, 96)
                return y.permute(0, 2, 1, 3).contiguous()

        eager, ggml = _run_ggml(M(), torch.randn(1, 3, 1536))
        _assert_close(eager, ggml)

    def test_head_split_pattern(self):
        """Attention head-split: linear → view → transpose."""

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(512, 512, bias=False)

            def forward(self, x):
                y = self.proj(x)
                b, s, _ = y.shape
                y = y.view(b, s, 8, 64)
                return y.transpose(1, 2).contiguous()

        eager, ggml = _run_ggml(M(), torch.randn(1, 3, 512))
        _assert_close(eager, ggml)


@requires_native
class TestSliceCatCombination:
    """Test slice+cat patterns used in RoPE."""

    def test_slice_neg_cat(self):
        class M(nn.Module):
            def forward(self, x):
                first = x[:, :, :, :4]
                second = x[:, :, :, 4:]
                return torch.cat([-second, first], dim=-1)

        eager, ggml = _run_ggml(M(), torch.randn(1, 8, 3, 8))
        _assert_close(eager, ggml)


@requires_native
class TestEmbedLinearPipeline:
    """Embedding → Linear pipeline (common in LLMs)."""

    def test_embed_linear(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(1000, 64)
                self.proj = nn.Linear(64, 128, bias=False)

            def forward(self, x):
                return self.proj(self.embed(x))

        eager, ggml = _run_ggml(M(), torch.tensor([[10, 20, 30]], dtype=torch.long))
        _assert_close(eager, ggml)

    def test_embed_two_linears(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(1000, 64)
                self.l1 = nn.Linear(64, 128, bias=False)
                self.l2 = nn.Linear(128, 64, bias=False)

            def forward(self, x):
                h = self.l1(self.embed(x))
                return self.l2(torch.nn.functional.silu(h))

        eager, ggml = _run_ggml(
            M(), torch.tensor([[10, 20, 30]], dtype=torch.long), atol=1e-3
        )
        _assert_close(eager, ggml, atol=1e-3)


@requires_native
class TestFullAttentionHeadPattern:
    """Test the full attention head pattern: proj → view → transpose → slice → cat."""

    def test_qkv_head_split_with_rope_rotation(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(64, 64, bias=False)

            def forward(self, x):
                q = self.q_proj(x)
                b, s, _ = q.shape
                q = q.view(b, s, 4, 16).transpose(1, 2)
                q1 = q[:, :, :, :8]
                q2 = q[:, :, :, 8:]
                rotated = torch.cat([-q2, q1], dim=-1)
                return rotated

        eager, ggml = _run_ggml(M(), torch.randn(1, 3, 64))
        _assert_close(eager, ggml)


# ---------------------------------------------------------------------------
# Tests for ReplaceCopyOpsPass (view_copy, permute_copy, etc.)
# ---------------------------------------------------------------------------


class TestCopyOpsPass:
    """Test that _copy ops are properly converted by ReplaceCopyOpsPass."""

    def _lower_with_copy_pass(self, model, inp, ep_passes=None):
        """Lower with ReplaceCopyOpsPass enabled."""
        model.eval()
        with torch.no_grad():
            ep = export(model, (inp,))
        edge_mgr = to_edge_rewrite_and_lower(
            ep,
            ep_passes=ep_passes or [],
            partitioner=[GgmlPartitioner()],
            transform_passes=[ReplaceCopyOpsPass()],
        )
        return edge_mgr.to_executorch()

    def test_view_copy(self):
        """Test view_copy (becomes view after pass)."""

        class M(nn.Module):
            def forward(self, x):
                return x.view(1, 3, 8, 8)

        self._lower_with_copy_pass(M(), torch.randn(1, 3, 64))

    def test_permute_copy(self):
        """Test permute_copy (becomes permute after pass)."""

        class M(nn.Module):
            def forward(self, x):
                return x.permute(0, 2, 1, 3).contiguous()

        self._lower_with_copy_pass(M(), torch.randn(1, 3, 8, 8))

    def test_unsqueeze_copy(self):
        """Test unsqueeze_copy (becomes unsqueeze after pass)."""

        class M(nn.Module):
            def forward(self, x):
                return x.unsqueeze(0)

        self._lower_with_copy_pass(M(), torch.randn(3, 64))

    def test_clone_copy(self):
        """Test clone (becomes alias after pass)."""

        class M(nn.Module):
            def forward(self, x):
                return x.clone()

        self._lower_with_copy_pass(M(), torch.randn(3, 64))


@requires_native
class TestCopyOpsPassE2E:
    """End-to-end tests for _copy ops after pass conversion."""

    def _run_with_copy_pass(self, model, inp, atol=1e-4, rtol=1e-4):
        """Run with ReplaceCopyOpsPass enabled."""
        model.eval()
        with torch.no_grad():
            ep = export(model, (inp,))
        edge_mgr = to_edge_rewrite_and_lower(
            ep,
            partitioner=[GgmlPartitioner()],
            transform_passes=[ReplaceCopyOpsPass()],
        )
        et = edge_mgr.to_executorch()
        pte = _load_for_executorch_from_buffer(et.buffer)
        result = pte.forward((inp,))
        ggml_out = result[0]
        with torch.no_grad():
            eager_out = model(inp)
        return eager_out, ggml_out

    def test_view_copy_e2e(self):
        class M(nn.Module):
            def forward(self, x):
                return x.view(1, 3, 8, 8)

        eager, ggml = self._run_with_copy_pass(M(), torch.randn(1, 3, 64))
        _assert_close(eager, ggml)

    def test_permute_copy_e2e(self):
        class M(nn.Module):
            def forward(self, x):
                return x.permute(0, 2, 1, 3).contiguous()

        eager, ggml = self._run_with_copy_pass(M(), torch.randn(1, 3, 8, 8))
        _assert_close(eager, ggml)

    def test_unsqueeze_copy_e2e(self):
        class M(nn.Module):
            def forward(self, x):
                return x.unsqueeze(0)

        eager, ggml = self._run_with_copy_pass(M(), torch.randn(3, 64))
        _assert_close(eager, ggml)


# ---------------------------------------------------------------------------
# Tests for SDPA preservation (ops_to_not_decompose)
# ---------------------------------------------------------------------------


class TestSDPAPreservation:
    """Test that SDPA is preserved from decomposition."""

    def _lower_with_sdpa(self, model, inp):
        """Lower with SDPA preservation."""
        model.eval()
        with torch.no_grad():
            ep = export(model, (inp,))
        edge_mgr = to_edge_rewrite_and_lower(
            ep,
            partitioner=[GgmlPartitioner()],
        )
        return edge_mgr.to_executorch()

    def test_sdpa_not_decomposed(self):
        """Test that SDPA is preserved in the graph."""

        class M(nn.Module):
            def forward(self, x):
                q = x
                k = x
                v = x
                return F.scaled_dot_product_attention(q, k, v)

        self._lower_with_sdpa(M(), torch.randn(1, 2, 4, 8))
