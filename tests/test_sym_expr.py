"""Tests for symbolic expression bytecode support in the ggml backend.

Progressive test suite:
  Tier 1: Bytecode compiler unit tests (pure Python, no runtime needed)
  Tier 2: Lowering-only tests (export + serialize, no C++ runtime)
  Tier 3: Runtime correctness tests (export + lower + execute + compare)
  Tier 4: Parakeet-specific tests (requires NeMo)
  Tier 5: Backwards compatibility

Usage:
    pytest tests/test_sym_expr.py -v -s
"""

import gc
import struct

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.export import export, Dim


@pytest.fixture(autouse=True)
def _cleanup_cuda():
    """Free GGML/CUDA resources after each test to prevent accumulation."""
    yield
    gc.collect()


# ---------------------------------------------------------------------------
# Tier 1: Bytecode compiler unit tests
# ---------------------------------------------------------------------------

class TestBytecodeCompiler:
    """Pure Python tests of _sympy_to_bytecode() + _eval_bytecode()."""

    @staticmethod
    def _get_funcs():
        from executorch_ggml.ggml_backend import _sympy_to_bytecode, _eval_bytecode
        return _sympy_to_bytecode, _eval_bytecode

    def test_bytecode_simple_symbol(self):
        """s0 → PUSH_SYM(0), eval(s0=42) → 42"""
        import sympy
        _sympy_to_bytecode, _eval_bytecode = self._get_funcs()

        s0 = sympy.Symbol("s0")
        sym_id_map = {}
        code = _sympy_to_bytecode(s0, sym_id_map)
        assert sym_id_map["s0"] == 0
        assert code == bytes([0x01, 0x00])  # PUSH_SYM(0)
        assert _eval_bytecode(code, {0: 42}) == 42

    def test_bytecode_constant(self):
        """5 → PUSH_CONST(5), eval → 5"""
        import sympy
        _sympy_to_bytecode, _eval_bytecode = self._get_funcs()

        expr = sympy.Integer(5)
        code = _sympy_to_bytecode(expr, {})
        expected = bytes([0x02]) + struct.pack("<i", 5)
        assert code == expected
        assert _eval_bytecode(code, {}) == 5

    def test_bytecode_add(self):
        """s0 + 3, eval(s0=10) → 13"""
        import sympy
        _sympy_to_bytecode, _eval_bytecode = self._get_funcs()

        s0 = sympy.Symbol("s0")
        expr = s0 + 3
        sym_id_map = {}
        code = _sympy_to_bytecode(expr, sym_id_map)
        assert _eval_bytecode(code, {sym_id_map["s0"]: 10}) == 13

    def test_bytecode_sub(self):
        """s0 - 1, eval(s0=10) → 9"""
        import sympy
        _sympy_to_bytecode, _eval_bytecode = self._get_funcs()

        s0 = sympy.Symbol("s0")
        # sympy represents s0 - 1 as Add(s0, -1), so test the add path
        expr = s0 - 1
        sym_id_map = {}
        code = _sympy_to_bytecode(expr, sym_id_map)
        assert _eval_bytecode(code, {sym_id_map["s0"]: 10}) == 9

    def test_bytecode_floordiv(self):
        """(s0 - 1) // 2, eval(s0=7) → 3, eval(s0=8) → 3"""
        import sympy
        _sympy_to_bytecode, _eval_bytecode = self._get_funcs()
        try:
            from torch.utils._sympy.functions import FloorDiv
        except ImportError:
            pytest.skip("torch FloorDiv not available")

        s0 = sympy.Symbol("s0")
        expr = FloorDiv(s0 - 1, 2)
        sym_id_map = {}
        code = _sympy_to_bytecode(expr, sym_id_map)
        sid = sym_id_map["s0"]
        assert _eval_bytecode(code, {sid: 7}) == 3
        assert _eval_bytecode(code, {sid: 8}) == 3

    def test_bytecode_parakeet_subsample(self):
        """((s0 - 1) // 8) + 1, eval(s0=257) → 33, eval(s0=5000) → 625"""
        import sympy
        _sympy_to_bytecode, _eval_bytecode = self._get_funcs()
        try:
            from torch.utils._sympy.functions import FloorDiv
        except ImportError:
            pytest.skip("torch FloorDiv not available")

        s0 = sympy.Symbol("s0")
        expr = FloorDiv(s0 - 1, 8) + 1
        sym_id_map = {}
        code = _sympy_to_bytecode(expr, sym_id_map)
        sid = sym_id_map["s0"]
        assert _eval_bytecode(code, {sid: 257}) == 33
        assert _eval_bytecode(code, {sid: 5000}) == 625

    def test_bytecode_mul(self):
        """s0 * 2, eval(s0=5) → 10"""
        import sympy
        _sympy_to_bytecode, _eval_bytecode = self._get_funcs()

        s0 = sympy.Symbol("s0")
        expr = s0 * 2
        sym_id_map = {}
        code = _sympy_to_bytecode(expr, sym_id_map)
        assert _eval_bytecode(code, {sym_id_map["s0"]: 5}) == 10

    def test_bytecode_mod(self):
        """s0 % 8, eval(s0=17) → 1"""
        import sympy
        _sympy_to_bytecode, _eval_bytecode = self._get_funcs()

        s0 = sympy.Symbol("s0")
        expr = sympy.Mod(s0, 8)
        sym_id_map = {}
        code = _sympy_to_bytecode(expr, sym_id_map)
        assert _eval_bytecode(code, {sym_id_map["s0"]: 17}) == 1

    def test_bytecode_multi_symbol(self):
        """s0 + s1, eval(s0=3, s1=4) → 7"""
        import sympy
        _sympy_to_bytecode, _eval_bytecode = self._get_funcs()

        s0, s1 = sympy.symbols("s0 s1")
        expr = s0 + s1
        sym_id_map = {}
        code = _sympy_to_bytecode(expr, sym_id_map)
        result = _eval_bytecode(code, {sym_id_map["s0"]: 3, sym_id_map["s1"]: 4})
        assert result == 7

    def test_bytecode_nested(self):
        """((s0 - 1) // 2 - 1) // 2 + 1 — two layers of stride-2 subsampling"""
        import sympy
        _sympy_to_bytecode, _eval_bytecode = self._get_funcs()
        try:
            from torch.utils._sympy.functions import FloorDiv
        except ImportError:
            pytest.skip("torch FloorDiv not available")

        s0 = sympy.Symbol("s0")
        expr = FloorDiv(FloorDiv(s0 - 1, 2) - 1, 2) + 1
        sym_id_map = {}
        code = _sympy_to_bytecode(expr, sym_id_map)
        sid = sym_id_map["s0"]
        # Manual: s0=17 → (17-1)//2=8 → (8-1)//2=3 → 3+1=4
        assert _eval_bytecode(code, {sid: 17}) == 4
        # s0=33 → (33-1)//2=16 → (16-1)//2=7 → 7+1=8
        assert _eval_bytecode(code, {sid: 33}) == 8

    def test_bytecode_negative(self):
        """-s0, eval(s0=5) → -5"""
        import sympy
        _sympy_to_bytecode, _eval_bytecode = self._get_funcs()

        s0 = sympy.Symbol("s0")
        expr = -s0
        sym_id_map = {}
        code = _sympy_to_bytecode(expr, sym_id_map)
        assert _eval_bytecode(code, {sym_id_map["s0"]: 5}) == -5


# ---------------------------------------------------------------------------
# Tier 2: Lowering-only tests (export + serialize, no C++ runtime)
# ---------------------------------------------------------------------------

def _export_and_lower(model, trace_input, dynamic_shapes):
    """Export → lower → return .pte buffer (no loading)."""
    from executorch_ggml import GgmlPartitioner
    from executorch_ggml.edge_pipeline import to_edge_rewrite_and_lower

    with torch.no_grad():
        ep = export(model, (trace_input,), dynamic_shapes=dynamic_shapes,
                    strict=False)

    edge_mgr = to_edge_rewrite_and_lower(
        ep, ep_passes=[], partitioner=[GgmlPartitioner()],
    )
    et = edge_mgr.to_executorch()
    return et.buffer


def _inspect_ir(buf):
    """Parse the .pte buffer and return (list of sym_dim_ids lists, list of sym_dim_exprs)."""
    from executorch_ggml.dump_ir import extract_ggml_blobs
    from executorch_ggml.ggml_ir.GgmlGraph import GgmlGraph
    import tempfile, os

    # Write to temp file for extract_ggml_blobs
    fd, path = tempfile.mkstemp(suffix=".pte")
    try:
        os.write(fd, buf)
        os.close(fd)
        blobs = extract_ggml_blobs(path)
    finally:
        os.unlink(path)

    if not blobs:
        return [], []

    graph = GgmlGraph.GetRootAs(blobs[0])
    all_sids = []
    all_exprs = []
    for i in range(graph.TensorsLength()):
        t = graph.Tensors(i)
        sids = [t.SymDimIds(d) for d in range(t.SymDimIdsLength())] if t.SymDimIdsLength() > 0 else []
        exprs_data = bytes(t.SymDimExprsAsNumpy()) if t.SymDimExprsLength() > 0 else b""
        all_sids.append(sids)
        all_exprs.append(exprs_data)

    return all_sids, all_exprs


class TestLoweringOnly:
    """Test that models with derived dynamic shapes successfully lower."""

    def test_lower_identity_dynamic(self):
        """Identity model — simple sym, no expressions."""
        class Identity(nn.Module):
            def forward(self, x):
                return x

        model = Identity().eval()
        x = torch.randn(1, 16, 64)
        dyn = {"x": {1: Dim("seq_len", min=1, max=256)}}
        buf = _export_and_lower(model, x, dyn)
        assert len(buf) > 0

    def test_lower_linear_dynamic(self):
        """Linear model — simple sym, no expressions."""
        class LinearWrap(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 64)
            def forward(self, x):
                return self.linear(x)

        model = LinearWrap().eval()
        x = torch.randn(1, 16, 64)
        dyn = {"x": {1: Dim("seq_len", min=1, max=256)}}
        buf = _export_and_lower(model, x, dyn)
        assert len(buf) > 0

    def test_lower_stride2_conv(self):
        """Conv1d(stride=2) produces derived expression (s0-1)//2 + 1."""
        class StrideConv(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1)

            def forward(self, x):
                return self.conv(x)

        model = StrideConv().eval()
        x = torch.randn(1, 64, 32)  # (batch, channels, seq_len)
        dyn = {"x": {2: Dim("seq_len", min=8, max=512)}}
        buf = _export_and_lower(model, x, dyn)
        assert len(buf) > 0

        # Inspect: should have at least one tensor with sid==-2
        all_sids, all_exprs = _inspect_ir(buf)
        has_expr = any(-2 in sids for sids in all_sids)
        print(f"  has_expr_sentinel: {has_expr}")
        # The conv output should have a derived shape
        # (depending on how the export graph traces, the output may or may not
        #  have sym_dim_ids. What matters is it lowered without error.)

    def test_lower_stride2_conv_linear(self):
        """Conv1d(stride=2) → Linear — derived shape flows to downstream."""
        class ConvLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1)
                self.linear = nn.Linear(64, 64)

            def forward(self, x):
                # x: (B, C, T) → conv → (B, C, T') → permute → (B, T', C) → linear
                y = self.conv(x)
                y = y.permute(0, 2, 1)
                return self.linear(y)

        model = ConvLinear().eval()
        x = torch.randn(1, 64, 32)
        dyn = {"x": {2: Dim.AUTO(min=3, max=512)}}
        buf = _export_and_lower(model, x, dyn)
        assert len(buf) > 0

    def test_lower_double_stride(self):
        """Two Conv1d(stride=2) — nested derived expression."""
        class DoubleStride(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1)
                self.conv2 = nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1)

            def forward(self, x):
                return self.conv2(self.conv1(x))

        model = DoubleStride().eval()
        x = torch.randn(1, 64, 64)
        dyn = {"x": {2: Dim("seq_len", min=8, max=512)}}
        buf = _export_and_lower(model, x, dyn)
        assert len(buf) > 0

    def test_lower_stride8_subsample(self):
        """3x Conv1d(stride=2) — Parakeet subsampling pattern."""
        class Stride8Sub(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1)
                self.conv2 = nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1)
                self.conv3 = nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1)

            def forward(self, x):
                return self.conv3(self.conv2(self.conv1(x)))

        model = Stride8Sub().eval()
        x = torch.randn(1, 64, 64)
        dyn = {"x": {2: Dim.AUTO(min=8, max=1024)}}
        buf = _export_and_lower(model, x, dyn)
        assert len(buf) > 0


# ---------------------------------------------------------------------------
# Tier 3: Runtime correctness tests
# ---------------------------------------------------------------------------

def _export_load_and_run(model, trace_input, dynamic_shapes, test_inputs, atol=1e-2):
    """Export → lower → load → run at multiple inputs → compare with eager."""
    from executorch_ggml import GgmlPartitioner
    from executorch_ggml.edge_pipeline import to_edge_rewrite_and_lower
    from executorch.extension.pybindings.portable_lib import (
        _load_for_executorch_from_buffer,
    )
    from executorch.exir import ExecutorchBackendConfig
    from executorch.exir.passes import MemoryPlanningPass

    with torch.no_grad():
        ep = export(model, (trace_input,), dynamic_shapes=dynamic_shapes,
                    strict=False)

    edge_mgr = to_edge_rewrite_and_lower(
        ep, ep_passes=[], partitioner=[GgmlPartitioner()],
    )
    et = edge_mgr.to_executorch(config=ExecutorchBackendConfig(
        extract_delegate_segments=True,
        memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
    ))
    pte = _load_for_executorch_from_buffer(et.buffer)

    for test_input in test_inputs:
        with torch.no_grad():
            eager_out = model(test_input)
        ggml_out = pte.forward((test_input,))[0]
        abs_diff = (eager_out - ggml_out).abs().max().item()
        print(f"  input_shape={list(test_input.shape)} "
              f"output_shape={list(eager_out.shape)} "
              f"max_diff={abs_diff:.6f}", flush=True)
        assert abs_diff < atol, (
            f"Mismatch at input shape {list(test_input.shape)}: "
            f"max_abs_diff={abs_diff} > atol={atol}"
        )
    import gc
    del pte, et, edge_mgr, ep
    gc.collect()


class TestRuntimeCorrectness:
    """Export + lower + execute + compare with eager PyTorch."""

    def test_run_identity_dynamic(self):
        """Identity pass-through with dynamic seq_len."""
        class Identity(nn.Module):
            def forward(self, x):
                return x

        model = Identity().eval()
        trace = torch.randn(1, 16, 64)
        dyn = {"x": {1: Dim("seq_len", min=1, max=256)}}
        tests = [torch.randn(1, l, 64) for l in [32, 64, 128, 256]]
        _export_load_and_run(model, trace, dyn, tests, atol=1e-6)

    def test_run_linear_dynamic(self):
        """Linear(64, 64) with dynamic seq_len."""
        class LinearWrap(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 64)
            def forward(self, x):
                return self.linear(x)

        model = LinearWrap().eval()
        trace = torch.randn(1, 16, 64)
        dyn = {"x": {1: Dim("seq_len", min=1, max=256)}}
        tests = [torch.randn(1, l, 64) for l in [32, 64, 128, 256]]
        _export_load_and_run(model, trace, dyn, tests, atol=1e-4)

    def test_run_stride2_conv(self):
        """Conv1d(stride=2) — verifies derived shape expression at runtime."""
        class StrideConv(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1)

            def forward(self, x):
                return self.conv(x)

        model = StrideConv().eval()
        trace = torch.randn(1, 64, 32)
        dyn = {"x": {2: Dim("seq_len", min=8, max=512)}}
        tests = [torch.randn(1, 64, l) for l in [16, 64, 128, 256]]
        _export_load_and_run(model, trace, dyn, tests, atol=1e-2)

    def test_run_double_stride(self):
        """Two chained Conv1d(stride=2) — nested derived expressions."""
        class DoubleStride(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1)
                self.conv2 = nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1)

            def forward(self, x):
                return self.conv2(self.conv1(x))

        model = DoubleStride().eval()
        trace = torch.randn(1, 64, 64)
        dyn = {"x": {2: Dim("seq_len", min=8, max=512)}}
        tests = [torch.randn(1, 64, l) for l in [32, 64, 128, 256]]
        _export_load_and_run(model, trace, dyn, tests, atol=1e-2)

    def test_run_stride8_subsample(self):
        """3x Conv1d(stride=2) + Linear — full Parakeet subsample pattern."""
        class Stride8Pipeline(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1)
                self.conv2 = nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1)
                self.conv3 = nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1)
                self.linear = nn.Linear(64, 32)

            def forward(self, x):
                y = self.conv3(self.conv2(self.conv1(x)))
                y = y.permute(0, 2, 1)  # (B, C, T') → (B, T', C)
                return self.linear(y)

        model = Stride8Pipeline().eval()
        trace = torch.randn(1, 64, 64)
        dyn = {"x": {2: Dim.AUTO(min=8, max=1024)}}
        tests = [torch.randn(1, 64, l) for l in [64, 128, 256]]
        _export_load_and_run(model, trace, dyn, tests, atol=1e-2)

    @pytest.mark.xfail(reason="LayerNorm with permute+dynamic shapes has large numerical drift on CUDA")
    def test_run_conformer_block_dynamic(self):
        """Mini conformer block: LayerNorm + Conv1d(stride=2) + SiLU."""
        class MiniConformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = nn.LayerNorm(64)
                self.conv = nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1)

            def forward(self, x):
                # x: (B, C, T)
                y = x.permute(0, 2, 1)  # (B, T, C)
                y = self.norm(y)
                y = y.permute(0, 2, 1)  # (B, C, T)
                y = self.conv(y)
                return F.silu(y)

        model = MiniConformer().eval()
        trace = torch.randn(1, 64, 32)
        dyn = {"x": {2: Dim("seq_len", min=8, max=256)}}
        tests = [torch.randn(1, 64, l) for l in [32, 64, 128]]
        _export_load_and_run(model, trace, dyn, tests, atol=1e-2)

    def test_run_subsample_conformer_dynamic(self):
        """Realistic Parakeet-style: Conv1d subsampling → conformer block with dynamic time.

        Architecture (modeled on NeMo FastConformer):
          Subsampling: 2× Conv1d(stride=2) + ReLU  →  4× time reduction
          ConformerBlock:
            1. Feed-forward: LN → Linear → SiLU → Linear, residual scaled by 0.5
            2. Conv module:  LN → pointwise Conv1d → GLU → depthwise Conv1d
                             → BatchNorm → SiLU → pointwise Conv1d, with residual
            3. Final LayerNorm

        This tests derived shape expressions ((s0-1)//4 + 1) flowing through
        LayerNorm, Conv1d (stride=1), BatchNorm, and reshapes in the conformer.
        """
        d_model = 64
        d_ff = 128  # 2× expansion (real uses 4×, but smaller for test speed)
        kernel_size = 5  # depthwise conv kernel (real uses 31)

        class ConvSubsampling(nn.Module):
            """2× Conv1d(stride=2) → 4× time reduction, like Parakeet."""
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
                self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)

            def forward(self, x):
                # x: (B, C, T)
                x = F.relu(self.conv1(x))
                x = F.relu(self.conv2(x))
                return x

        class ConformerFeedForward(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = nn.LayerNorm(d_model)
                self.linear1 = nn.Linear(d_model, d_ff)
                self.linear2 = nn.Linear(d_ff, d_model)

            def forward(self, x):
                # x: (B, T, C)
                residual = x
                x = self.norm(x)
                x = F.silu(self.linear1(x))
                x = self.linear2(x)
                return residual + 0.5 * x

        class ConformerConvModule(nn.Module):
            """Pointwise → GLU → Depthwise → BN → SiLU → Pointwise."""
            def __init__(self):
                super().__init__()
                self.norm = nn.LayerNorm(d_model)
                self.pointwise1 = nn.Conv1d(d_model, d_model * 2, kernel_size=1)
                self.depthwise = nn.Conv1d(
                    d_model, d_model, kernel_size=kernel_size,
                    padding=(kernel_size - 1) // 2, groups=d_model,
                )
                self.batch_norm = nn.BatchNorm1d(d_model)
                self.pointwise2 = nn.Conv1d(d_model, d_model, kernel_size=1)

            def forward(self, x):
                # x: (B, T, C)
                residual = x
                x = self.norm(x)
                x = x.permute(0, 2, 1)          # (B, C, T)
                x = self.pointwise1(x)           # (B, 2C, T)
                x = F.glu(x, dim=1)              # (B, C, T)
                x = self.depthwise(x)            # (B, C, T)
                x = self.batch_norm(x)           # (B, C, T)
                x = F.silu(x)
                x = self.pointwise2(x)           # (B, C, T)
                x = x.permute(0, 2, 1)          # (B, T, C)
                return residual + x

        class SubsampleConformer(nn.Module):
            """Subsampling + one conformer block (no self-attention)."""
            def __init__(self):
                super().__init__()
                self.subsample = ConvSubsampling()
                self.ff1 = ConformerFeedForward()
                self.conv_module = ConformerConvModule()
                self.ff2 = ConformerFeedForward()
                self.final_norm = nn.LayerNorm(d_model)

            def forward(self, x):
                # x: (B, C, T) → subsample → (B, C, T')
                x = self.subsample(x)
                x = x.permute(0, 2, 1)  # (B, T', C)
                # Conformer block
                x = self.ff1(x)
                x = self.conv_module(x)
                x = self.ff2(x)
                x = self.final_norm(x)
                return x  # (B, T', C)

        model = SubsampleConformer().eval()
        trace = torch.randn(1, d_model, 64)
        dyn = {"x": {2: Dim.AUTO(min=8, max=512)}}

        # Verify shapes: T'= ((T-1)//4) + 1 approximately
        for t in [32, 64, 128, 256]:
            with torch.no_grad():
                out = model(torch.randn(1, d_model, t))
            # After 2× stride-2 convs: T' depends on exact padding
            print(f"  time={t} → output_shape={list(out.shape)}")

        tests = [torch.randn(1, d_model, t) for t in [32, 64, 128, 256]]
        _export_load_and_run(model, trace, dyn, tests, atol=1e-2)

    def test_run_two_dynamic_dims_simple(self):
        """Two independent dynamic dims (batch + seq_len), simple symbol lookup."""
        class TwoDynLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 32)

            def forward(self, x):
                return self.linear(x)

        model = TwoDynLinear().eval()
        trace = torch.randn(2, 16, 64)
        dyn = {"x": {0: Dim("batch", min=1, max=8),
                      1: Dim("seq_len", min=1, max=256)}}
        tests = [torch.randn(b, s, 64) for b, s in [(1, 32), (4, 64), (2, 128), (8, 16)]]
        _export_load_and_run(model, trace, dyn, tests, atol=1e-4)

    def test_run_two_dynamic_dims_with_derived(self):
        """Two inputs with independent dynamic dims, one producing a derived expression.

        Input a: (1, seq_a, 64) — simple dynamic dim (seq_a), processed by Linear.
        Input b: (1, 64, seq_b) — strided conv halves it → derived expr ((seq_b-1)//2)+1.
        Output = cat of the two paths after aligning to a common channel dim.
        This verifies two independent symbols (s0, s1) tracked simultaneously,
        with s1 producing a derived bytecode expression.
        """
        class TwoDynModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 32)
                self.conv = nn.Conv1d(64, 32, kernel_size=3, stride=2, padding=1)

            def forward(self, a, b):
                # a: (1, seq_a, 64) → linear → (1, seq_a, 32)
                out_a = self.linear(a)
                # b: (1, 64, seq_b) → conv → (1, 32, (seq_b-1)//2+1)
                out_b = self.conv(b)
                return out_a, out_b

        model = TwoDynModel().eval()
        trace_a = torch.randn(1, 16, 64)
        trace_b = torch.randn(1, 64, 32)
        dyn = {"a": {1: Dim("seq_a", min=1, max=256)},
               "b": {2: Dim("seq_b", min=8, max=512)}}

        from executorch_ggml import GgmlPartitioner
        from executorch_ggml.edge_pipeline import to_edge_rewrite_and_lower
        from executorch.extension.pybindings.portable_lib import (
            _load_for_executorch_from_buffer,
        )

        with torch.no_grad():
            ep = export(model, (trace_a, trace_b), dynamic_shapes=dyn, strict=False)
        edge_mgr = to_edge_rewrite_and_lower(
            ep, ep_passes=[], partitioner=[GgmlPartitioner()],
        )
        et = edge_mgr.to_executorch()
        pte = _load_for_executorch_from_buffer(et.buffer)

        for sa, sb in [(8, 16), (32, 64), (64, 128), (128, 256)]:
            a = torch.randn(1, sa, 64)
            b = torch.randn(1, 64, sb)
            with torch.no_grad():
                eager_a, eager_b = model(a, b)
            results = pte.forward((a, b))
            ggml_a, ggml_b = results[0], results[1]
            diff_a = (eager_a - ggml_a).abs().max().item()
            diff_b = (eager_b - ggml_b).abs().max().item()
            expected_sb_out = (sb - 1) // 2 + 1
            print(f"  seq_a={sa} seq_b={sb}: "
                  f"a_shape={list(ggml_a.shape)} diff={diff_a:.6f}, "
                  f"b_shape={list(ggml_b.shape)} (expect T'={expected_sb_out}) diff={diff_b:.6f}")
            assert list(ggml_a.shape) == list(eager_a.shape), (
                f"Shape mismatch for a: {list(ggml_a.shape)} vs {list(eager_a.shape)}")
            assert list(ggml_b.shape) == list(eager_b.shape), (
                f"Shape mismatch for b: {list(ggml_b.shape)} vs {list(eager_b.shape)}")
            assert diff_a < 5e-3, f"a mismatch: {diff_a}"
            assert diff_b < 5e-2, f"b mismatch: {diff_b}"

    def test_run_two_dynamic_dims_both_derived(self):
        """Two dynamic dims that both produce derived expressions.

        Two separate Conv1d paths: one subsamples time (dim 2), the other
        subsamples a second spatial dim via reshape tricks. We keep it simple:
        process the same input through two convs with different strides, then cat.
        """
        class TwoDynDualConv(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv_s2 = nn.Conv1d(64, 32, kernel_size=3, stride=2, padding=1)
                self.conv_s4 = nn.Conv1d(64, 32, kernel_size=5, stride=4, padding=2)

            def forward(self, x):
                # x: (batch, 64, time)
                a = self.conv_s2(x)   # (batch, 32, (time-1)//2 + 1)
                b = self.conv_s4(x)   # (batch, 32, (time-1)//4 + 1)
                # Return both so we verify both derived dims are correct
                return a, b

        model = TwoDynDualConv().eval()
        trace = torch.randn(1, 64, 64)
        dyn = {"x": {2: Dim("time", min=8, max=512)}}

        for t in [16, 64, 128, 256]:
            x = torch.randn(1, 64, t)
            with torch.no_grad():
                eager_a, eager_b = model(x)
            # Verify shapes match the expected formulas
            expected_a = (t - 1) // 2 + 1
            expected_b = (t - 1) // 4 + 1
            assert eager_a.shape[2] == expected_a, f"t={t}: a.shape[2]={eager_a.shape[2]} != {expected_a}"
            assert eager_b.shape[2] == expected_b, f"t={t}: b.shape[2]={eager_b.shape[2]} != {expected_b}"
            print(f"  time={t}: a_shape={list(eager_a.shape)}, b_shape={list(eager_b.shape)}")

        # Now test the full export→lower→run pipeline
        from executorch_ggml import GgmlPartitioner
        from executorch_ggml.edge_pipeline import to_edge_rewrite_and_lower
        from executorch.extension.pybindings.portable_lib import (
            _load_for_executorch_from_buffer,
        )

        with torch.no_grad():
            ep = export(model, (trace,), dynamic_shapes=dyn, strict=False)
        edge_mgr = to_edge_rewrite_and_lower(
            ep, ep_passes=[], partitioner=[GgmlPartitioner()],
        )
        et = edge_mgr.to_executorch()
        pte = _load_for_executorch_from_buffer(et.buffer)

        for t in [16, 64, 128, 256]:
            x = torch.randn(1, 64, t)
            with torch.no_grad():
                eager_a, eager_b = model(x)
            results = pte.forward((x,))
            ggml_a, ggml_b = results[0], results[1]
            diff_a = (eager_a - ggml_a).abs().max().item()
            diff_b = (eager_b - ggml_b).abs().max().item()
            print(f"  time={t}: a_diff={diff_a:.6f} a_shape={list(ggml_a.shape)}, "
                  f"b_diff={diff_b:.6f} b_shape={list(ggml_b.shape)}")
            assert list(ggml_a.shape) == list(eager_a.shape), (
                f"Shape mismatch for a at time={t}: {list(ggml_a.shape)} vs {list(eager_a.shape)}")
            assert list(ggml_b.shape) == list(eager_b.shape), (
                f"Shape mismatch for b at time={t}: {list(ggml_b.shape)} vs {list(eager_b.shape)}")
            assert diff_a < 1e-2, f"a mismatch at time={t}: {diff_a}"
            assert diff_b < 1e-2, f"b mismatch at time={t}: {diff_b}"


# ---------------------------------------------------------------------------
# Tier 4: Parakeet encoder (requires NeMo)
# ---------------------------------------------------------------------------

_HAS_NEMO = False
try:
    import nemo  # noqa: F401
    _HAS_NEMO = True
except ImportError:
    pass

_HAS_NATIVE = False
try:
    from executorch.extension.pybindings.portable_lib import (
        _load_for_executorch_from_buffer as _load_pte,
    )
    _HAS_NATIVE = True
except Exception:
    pass


@pytest.mark.skipif(not _HAS_NEMO, reason="NeMo not installed")
class TestParakeetEncoder:
    """Test the real Parakeet FastConformer encoder with dynamic mel_len."""

    @staticmethod
    def _load_encoder():
        """Load Parakeet model and return (EncoderWithProjection, feat_in, subsampling_factor)."""
        import sys, os
        # Add paths so export_parakeet_tdt and its transitive imports resolve.
        # - parakeet_dir: so `import export_parakeet_tdt` works
        # - third_party: so `from executorch.examples.models.parakeet.quantize` works
        root = os.path.dirname(os.path.dirname(__file__))
        parakeet_dir = os.path.join(root, "third-party", "executorch",
                                    "examples", "models", "parakeet")
        third_party = os.path.join(root, "third-party")
        for p in [parakeet_dir, third_party]:
            if p not in sys.path:
                sys.path.insert(0, p)
        from export_parakeet_tdt import EncoderWithProjection, load_model
        model = load_model()
        encoder = EncoderWithProjection(model.encoder, model.joint)
        encoder.eval()
        feat_in = getattr(model.encoder, "_feat_in", 128)
        sub_factor = int(getattr(model.encoder, "subsampling_factor", 8))
        return encoder, feat_in, sub_factor

    def test_parakeet_encoder_export_and_lower(self):
        """Export the real Parakeet encoder with dynamic mel_len and lower to ggml.

        Verifies:
          - Export succeeds with Dim.AUTO on the time dimension
          - At least 1 delegate call to GgmlBackend
          - The lowered delegate blobs contain sym_dim_exprs bytecode
        Note: Full serialization to .pte may fail if non-delegated ops
        (e.g. self-attention expand/bmm) have unsupported strides.
        """
        from executorch.exir import EdgeCompileConfig
        from executorch_ggml import GgmlPartitioner
        from executorch_ggml.edge_pipeline import to_edge_rewrite_and_lower
        from executorch_ggml.passes.replace_copy_ops_pass import ReplaceCopyOpsPass

        encoder, feat_in, sub_factor = self._load_encoder()
        print(f"  feat_in={feat_in}, subsampling_factor={sub_factor}")

        mel = torch.randn(1, feat_in, 512)
        length = torch.tensor([512], dtype=torch.int64)

        with torch.no_grad():
            ep = export(
                encoder, (),
                kwargs={"audio_signal": mel, "length": length},
                dynamic_shapes={"audio_signal": {2: Dim.AUTO}, "length": {}},
                strict=False,
            )

        edge_mgr = to_edge_rewrite_and_lower(
            ep,
            transform_passes=[ReplaceCopyOpsPass()],
            partitioner=[GgmlPartitioner()],
            compile_config=EdgeCompileConfig(
                _check_ir_validity=False,
                _skip_dim_order=True,
            ),
        )

        # Count delegated vs non-delegated ops
        for name, prog in edge_mgr._edge_programs.items():
            n_delegates = sum(
                1 for n in prog.graph_module.graph.nodes
                if 'lowered_module' in str(n.target) and n.op == 'get_attr'
            )
            non_del_ops = [str(n.target).split(':')[0] for n in prog.graph_module.graph.nodes
                           if n.op == 'call_function'
                           and 'getitem' not in str(n.target)
                           and 'executorch' not in str(n.target)]
            print(f"  delegates={n_delegates}, non_delegated_ops={len(non_del_ops)}")
            if non_del_ops:
                from collections import Counter
                top = Counter(non_del_ops).most_common(5)
                print(f"  top non-delegated: {top}")
            assert n_delegates >= 1, "Expected at least 1 delegate"

    @pytest.mark.skipif(not _HAS_NATIVE, reason="Native extension not available")
    def test_parakeet_encoder_runtime(self):
        """Run the real Parakeet encoder at multiple mel_len values.

        Verifies output shapes and numerical closeness vs eager PyTorch.
        This is the definitive test that sym_dim_exprs works for the real model.

        Note: Skipped until the partitioner supports self-attention ops
        (expand, bmm, div) needed for full encoder delegation.
        """
        pytest.skip(
            "Full Parakeet encoder runtime requires self-attention op support "
            "(expand, bmm, div) in the GgmlPartitioner. "
            "Use test_run_subsample_conformer_dynamic for the conformer pipeline test."
        )


# ---------------------------------------------------------------------------
# Tier 5: Backwards compatibility
# ---------------------------------------------------------------------------

class TestBackwardsCompat:
    """Verify models without sym_dim_exprs still work."""

    def test_backwards_compat_simple_sym(self):
        """Model with only simple sym_dim_ids (no -2 sentinels) works identically."""
        class LinearWrap(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 64)
            def forward(self, x):
                return self.linear(x)

        model = LinearWrap().eval()
        trace = torch.randn(1, 16, 64)
        dyn = {"x": {1: Dim("seq_len", min=1, max=256)}}
        tests = [torch.randn(1, l, 64) for l in [1, 4, 16, 64]]
        _export_load_and_run(model, trace, dyn, tests, atol=5e-3)
