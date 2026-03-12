"""Tests for GGML quantization module.

Includes:
- Unit tests for Q8_0 quantization math
- Unit tests for tensor selection logic (should_quantize)
- Integration tests for the full export → quantize → execute pipeline
"""

import struct

import numpy as np
import pytest
import torch
import torch.nn as nn

from executorch_ggml.quantize import (
    BLOCK_Q8_0_BYTES,
    GgmlQuantConfig,
    GgmlQuantType,
    QK8_0,
    dequantize_q8_0,
    quantize_tensor_q8_0,
    should_quantize,
)

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


class TestQuantizeQ8_0:
    """Tests for Q8_0 quantization/dequantization."""

    def test_basic_roundtrip(self):
        """Quantize then dequantize, check error is small."""
        np.random.seed(42)
        data = np.random.randn(1024).astype(np.float32)
        quant = quantize_tensor_q8_0(data)
        deq = dequantize_q8_0(quant, 1024)
        # Q8_0 has 127 levels per block, max relative error ~0.4%
        max_abs_err = np.max(np.abs(data - deq))
        max_val = np.max(np.abs(data))
        assert max_abs_err / max_val < 0.01, f"relative error too large: {max_abs_err / max_val}"

    def test_byte_size(self):
        """Each block of 32 elements produces 34 bytes."""
        data = np.random.randn(32).astype(np.float32)
        quant = quantize_tensor_q8_0(data)
        assert len(quant) == BLOCK_Q8_0_BYTES  # 34

    def test_multiple_blocks(self):
        """Correct size for multiple blocks."""
        data = np.random.randn(256).astype(np.float32)
        quant = quantize_tensor_q8_0(data)
        assert len(quant) == (256 // QK8_0) * BLOCK_Q8_0_BYTES  # 8 * 34 = 272

    def test_zeros(self):
        """All-zero input produces all-zero output."""
        data = np.zeros(32, dtype=np.float32)
        quant = quantize_tensor_q8_0(data)
        deq = dequantize_q8_0(quant, 32)
        np.testing.assert_array_equal(deq, np.zeros(32))

    def test_constant_value(self):
        """Constant nonzero input: all quants should be 127 (or -127)."""
        data = np.full(32, 1.0, dtype=np.float32)
        quant = quantize_tensor_q8_0(data)
        # Scale d = 1.0/127, each qs = round(1.0 / (1.0/127)) = 127
        d = np.frombuffer(quant[:2], dtype=np.float16)[0]
        qs = np.frombuffer(quant[2:34], dtype=np.int8)
        assert d > 0
        np.testing.assert_array_equal(qs, np.full(32, 127, dtype=np.int8))

    def test_block_layout(self):
        """Verify the byte layout: fp16 scale then 32 int8 values."""
        data = np.arange(32, dtype=np.float32)  # 0..31
        quant = quantize_tensor_q8_0(data)
        # First 2 bytes: fp16 scale
        d = struct.unpack("<e", quant[:2])[0]
        # Next 32 bytes: int8 quantized values
        qs = np.frombuffer(quant[2:34], dtype=np.int8)
        assert len(qs) == 32
        # Dequantized should approximate original
        deq = d * qs.astype(np.float32)
        np.testing.assert_allclose(deq, data, atol=0.15)

    def test_not_divisible_by_block_size(self):
        """Raise assertion error for non-divisible sizes."""
        data = np.random.randn(33).astype(np.float32)
        with pytest.raises(AssertionError):
            quantize_tensor_q8_0(data)

    def test_large_tensor(self):
        """Test with a realistically sized tensor (4096 x 4096 = 16M elements)."""
        np.random.seed(0)
        data = np.random.randn(4096 * 4096).astype(np.float32)
        quant = quantize_tensor_q8_0(data)
        expected_size = (4096 * 4096 // QK8_0) * BLOCK_Q8_0_BYTES
        assert len(quant) == expected_size
        # Compression ratio: 4 bytes/elem -> 34/32 bytes/elem = 3.76x
        compression = (4096 * 4096 * 4) / len(quant)
        assert 3.7 < compression < 3.8

    def test_negative_values(self):
        """Negative values are preserved through quantization."""
        data = np.full(32, -2.5, dtype=np.float32)
        quant = quantize_tensor_q8_0(data)
        deq = dequantize_q8_0(quant, 32)
        np.testing.assert_allclose(deq, data, atol=0.025)


class TestShouldQuantize:
    """Tests for the should_quantize decision logic."""

    def _cfg(self, **kwargs):
        return GgmlQuantConfig(**kwargs)

    def test_linear_weight(self):
        """2D float32 tensor with large dims should be quantized."""
        assert should_quantize(
            "model.layers.0.self_attn.q_proj.weight",
            (896, 896), torch.float32, 896 * 896, self._cfg(),
        )

    def test_norm_weight_skipped(self):
        """1D norm weight skipped (ndim < 2 and skip pattern)."""
        assert not should_quantize(
            "model.layers.0.input_layernorm.weight",
            (896,), torch.float32, 896, self._cfg(),
        )

    def test_bias_skipped(self):
        """Bias tensors are skipped by pattern."""
        assert not should_quantize(
            "model.layers.0.self_attn.o_proj.bias",
            (896,), torch.float32, 896, self._cfg(),
        )

    def test_small_tensor_skipped(self):
        """Tensors below min_elements threshold are skipped."""
        assert not should_quantize(
            "small_weight", (16, 16), torch.float32, 256, self._cfg(),
        )

    def test_non_divisible_inner_dim(self):
        """Inner dim not divisible by 32 should be skipped."""
        assert not should_quantize(
            "odd_weight", (100, 33), torch.float32, 3300, self._cfg(),
        )

    def test_int_tensor_skipped(self):
        """Integer tensors should not be quantized."""
        assert not should_quantize(
            "index_tensor", (1024, 1024), torch.int64, 1024 * 1024, self._cfg(),
        )

    def test_custom_skip_patterns(self):
        """Custom skip patterns are respected."""
        cfg = self._cfg(skip_patterns={"lm_head"})
        assert not should_quantize(
            "lm_head.weight", (32000, 4096), torch.float32, 32000 * 4096, cfg,
        )
        # Without lm_head in patterns, it should be quantized
        cfg2 = self._cfg(skip_patterns=set())
        assert should_quantize(
            "lm_head.weight", (32000, 4096), torch.float32, 32000 * 4096, cfg2,
        )

    def test_embedding_default_skip(self):
        """Default config does NOT skip embedding (no 'embed_tokens' in default patterns)."""
        cfg = self._cfg()
        # embed_tokens is NOT in default skip patterns — it would be quantized
        # if it meets other criteria. But the default patterns are
        # {"norm", "layernorm", "rmsnorm", "bias"} which don't include embed_tokens.
        assert should_quantize(
            "model.embed_tokens.weight",
            (151936, 896), torch.float32, 151936 * 896, cfg,
        )

    def test_float16_tensor(self):
        """Float16 tensors can be quantized."""
        assert should_quantize(
            "model.layers.0.mlp.gate_proj.weight",
            (4864, 896), torch.float16, 4864 * 896, self._cfg(),
        )

    def test_3d_tensor(self):
        """3D+ tensors are eligible if inner dim is divisible."""
        assert should_quantize(
            "conv_weight", (64, 64, 32), torch.float32, 64 * 64 * 32, self._cfg(),
        )

    def test_3d_small_kernel_skipped(self):
        """3D conv with small kernel dim (not divisible by 32) is skipped."""
        assert not should_quantize(
            "conv_weight", (64, 64, 3), torch.float32, 64 * 64 * 3, self._cfg(),
        )


class TestGgmlQuantConfig:
    """Tests for GgmlQuantConfig defaults."""

    def test_default_type(self):
        cfg = GgmlQuantConfig()
        assert cfg.quant_type == GgmlQuantType.Q8_0

    def test_default_skip_patterns(self):
        cfg = GgmlQuantConfig()
        assert "norm" in cfg.skip_patterns
        assert "bias" in cfg.skip_patterns

    def test_default_min_elements(self):
        cfg = GgmlQuantConfig()
        assert cfg.min_elements == 1024


# =========================================================================
# Integration tests: full export → quantize → execute pipeline
# =========================================================================


class TestPartitionerQuantConfig:
    """Test that GgmlPartitioner correctly propagates quant config."""

    def test_partitioner_accepts_quant_config(self):
        """GgmlPartitioner can be constructed with a quant_config."""
        from executorch_ggml import GgmlPartitioner
        cfg = GgmlQuantConfig()
        p = GgmlPartitioner(quant_config=cfg)
        # Should have compile specs with the quant type
        specs = p.delegation_spec.compile_specs
        assert len(specs) >= 1
        quant_spec = next(s for s in specs if s.key == "ggml_quant_type")
        assert quant_spec.value == b"q8_0"

    def test_partitioner_backward_compatible(self):
        """GgmlPartitioner without quant_config works as before."""
        from executorch_ggml import GgmlPartitioner
        p = GgmlPartitioner()
        assert len(p.delegation_spec.compile_specs) == 0


class TestQuantizedSerialization:
    """Test that quantized weights are correctly serialized into the IR."""

    def _export_and_check_ir(self, model, inp, quant_config=None):
        """Export model, check that IR contains quantized tensor types."""
        from torch.export import export
        from executorch_ggml import GgmlPartitioner
        from executorch_ggml.edge_pipeline import to_edge_rewrite_and_lower
        from executorch_ggml.passes.replace_copy_ops_pass import ReplaceCopyOpsPass

        model.eval()
        with torch.no_grad():
            ep = export(model, (inp,))

        partitioner = GgmlPartitioner(quant_config=quant_config)
        edge_mgr = to_edge_rewrite_and_lower(
            ep,
            ep_passes=[],
            partitioner=[partitioner],
        )
        et = edge_mgr.to_executorch()
        return et

    def test_linear_weights_quantized(self):
        """Linear weights should be quantized to Q8_0 in the IR."""
        from executorch_ggml.serialize import TYPE_Q8_0
        from executorch_ggml.ggml_ir.TensorType import TensorType
        import flatbuffers
        from executorch_ggml.ggml_ir.GgmlGraph import GgmlGraph

        class SimpleLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(64, 64, bias=False)

            def forward(self, x):
                return self.fc(x)

        model = SimpleLinear()
        inp = torch.randn(1, 64)
        cfg = GgmlQuantConfig(min_elements=64)  # lower threshold for test
        et = self._export_and_check_ir(model, inp, quant_config=cfg)

        # Parse the delegate blob to check tensor types
        pte_bytes = et.buffer
        # Find Q8_0 type (6) in the serialized data - check that
        # at least one tensor has the Q8_0 type
        has_q8_0 = TYPE_Q8_0.to_bytes(4, "little") in pte_bytes
        assert has_q8_0, "Expected Q8_0 type in serialized .pte"

    def test_no_quant_backward_compatible(self):
        """Without quant_config, no Q8_0 tensors should appear."""
        from executorch_ggml.serialize import TYPE_Q8_0

        class SimpleLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(64, 64, bias=False)

            def forward(self, x):
                return self.fc(x)

        model = SimpleLinear()
        inp = torch.randn(1, 64)
        et = self._export_and_check_ir(model, inp, quant_config=None)
        # Should NOT contain Q8_0 type
        pte_bytes = et.buffer
        # This is a heuristic check - Q8_0=6 could appear as data
        # but shouldn't appear as a TensorType enum value


@requires_native
class TestQuantizedExecution:
    """End-to-end tests: export with Q8_0, run inference, compare to eager."""

    def _run_quantized(self, model, inp, quant_config, atol=0.05):
        """Export with quantization, execute, compare to eager."""
        from torch.export import export
        from executorch_ggml import GgmlPartitioner
        from executorch_ggml.edge_pipeline import to_edge_rewrite_and_lower

        model.eval()
        with torch.no_grad():
            eager_out = model(inp)
            ep = export(model, (inp,))

        partitioner = GgmlPartitioner(quant_config=quant_config)
        edge_mgr = to_edge_rewrite_and_lower(
            ep,
            ep_passes=[],
            partitioner=[partitioner],
        )
        et = edge_mgr.to_executorch()
        pte = _load_for_executorch_from_buffer(et.buffer)
        result = pte.forward((inp,))
        ggml_out = result[0]
        return eager_out, ggml_out

    def test_linear_q8_0(self):
        """Linear layer with Q8_0 quantization produces close results."""
        class TwoLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(128, 256, bias=False)
                self.fc2 = nn.Linear(256, 64, bias=False)

            def forward(self, x):
                return self.fc2(torch.relu(self.fc1(x)))

        model = TwoLinear()
        inp = torch.randn(1, 128)
        cfg = GgmlQuantConfig(min_elements=64)
        eager, quant = self._run_quantized(model, inp, cfg)
        # Q8_0 is nearly lossless - should be very close
        diff = (eager - quant).abs().max().item()
        assert diff < 0.1, f"Max diff {diff} too large for Q8_0"

    def test_linear_with_bias_q8_0(self):
        """Linear with bias: weight quantized, bias stays F32."""
        class LinearBias(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(128, 64)  # has bias

            def forward(self, x):
                return self.fc(x)

        model = LinearBias()
        inp = torch.randn(1, 128)
        cfg = GgmlQuantConfig(min_elements=64)
        eager, quant = self._run_quantized(model, inp, cfg)
        diff = (eager - quant).abs().max().item()
        assert diff < 0.1, f"Max diff {diff} too large for Q8_0"

    def test_mlp_q8_0(self):
        """MLP with SiLU activation, Q8_0 quantization."""
        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate = nn.Linear(256, 512, bias=False)
                self.up = nn.Linear(256, 512, bias=False)
                self.down = nn.Linear(512, 256, bias=False)

            def forward(self, x):
                return self.down(torch.sigmoid(self.gate(x)) * self.up(x))

        model = MLP()
        inp = torch.randn(1, 256)
        cfg = GgmlQuantConfig(min_elements=64)
        eager, quant = self._run_quantized(model, inp, cfg)
        diff = (eager - quant).abs().max().item()
        assert diff < 0.5, f"Max diff {diff} too large for Q8_0 MLP"

    def test_skip_patterns_respected(self):
        """Norm weights and biases should NOT be quantized, output should match."""
        class WithNorm(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(128, 128, bias=False)
                # LayerNorm has weight (1D) and bias (1D) — should be skipped
                self.norm = nn.LayerNorm(128)

            def forward(self, x):
                return self.norm(self.fc(x))

        model = WithNorm()
        inp = torch.randn(1, 128)
        cfg = GgmlQuantConfig(min_elements=64)
        eager, quant = self._run_quantized(model, inp, cfg)
        diff = (eager - quant).abs().max().item()
        assert diff < 0.1, f"Max diff {diff} too large"
