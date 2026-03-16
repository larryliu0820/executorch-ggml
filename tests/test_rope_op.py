"""Tests for the fused ggml.rope op.

Verifies:
1. Eager correctness: ggml.rope matches apply_rotary_emb
2. Export preservation: op survives torch.export without decomposition
3. Lowering: op is delegated to the ggml backend partition
4. E2E (native): runtime output matches eager (if native extension is available)
"""

import sys
sys.path.insert(0, "python")

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.export import export

import executorch_ggml.rope_op  # noqa: F401 — registers torch.ops.ggml.rope
from executorch_ggml import GgmlPartitioner
from executorch_ggml.edge_pipeline import to_edge_rewrite_and_lower
from executorch_ggml.modules.rope import GgmlEncoderAttention, swap_encoder_rope

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


# Reference implementation from model.py
def apply_rotary_emb(q, k, freqs_cos, freqs_sin):
    q_r, q_i = q.float().reshape(q.shape[:-1] + (-1, 2)).unbind(-1)
    k_r, k_i = k.float().reshape(k.shape[:-1] + (-1, 2)).unbind(-1)
    fc = freqs_cos.unsqueeze(0).unsqueeze(2)
    fs = freqs_sin.unsqueeze(0).unsqueeze(2)
    q_out = torch.stack([q_r * fc - q_i * fs, q_r * fs + q_i * fc], dim=-1).flatten(-2)
    k_out = torch.stack([k_r * fc - k_i * fs, k_r * fs + k_i * fc], dim=-1).flatten(-2)
    return q_out.type_as(q), k_out.type_as(k)


def precompute_freqs_cis(head_dim, max_len, theta):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_len, dtype=torch.float)
    freqs = torch.outer(t, freqs)
    return freqs.cos(), freqs.sin()


# --- Eager correctness ---

class TestEagerCorrectness:
    def test_rope_matches_reference(self):
        """ggml.rope output matches apply_rotary_emb for Q and K."""
        B, T, H, D = 1, 16, 8, 64
        freq_base = 1_000_000.0
        q = torch.randn(B, T, H, D)
        k = torch.randn(B, T, H, D)
        freqs_cos, freqs_sin = precompute_freqs_cis(D, T, freq_base)
        positions = torch.arange(T, dtype=torch.int32)

        # Reference
        q_ref, k_ref = apply_rotary_emb(q, k, freqs_cos, freqs_sin)

        # ggml.rope
        q_fused = torch.ops.ggml.rope(q, positions, D, freq_base)
        k_fused = torch.ops.ggml.rope(k, positions, D, freq_base)

        torch.testing.assert_close(q_fused, q_ref, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(k_fused, k_ref, atol=1e-5, rtol=1e-5)

    def test_rope_preserves_dtype(self):
        """Output dtype matches input dtype."""
        for dtype in [torch.float32, torch.bfloat16]:
            x = torch.randn(1, 4, 2, 8, dtype=dtype)
            pos = torch.arange(4, dtype=torch.int32)
            out = torch.ops.ggml.rope(x, pos, 8, 10000.0)
            assert out.dtype == dtype, f"Expected {dtype}, got {out.dtype}"


# --- Export + non-decomposition ---

class SimpleRopeModel(nn.Module):
    def __init__(self, n_dims, freq_base):
        super().__init__()
        self.n_dims = n_dims
        self.freq_base = freq_base

    def forward(self, x):
        T = x.shape[1]
        pos = torch.arange(T, dtype=torch.int32, device=x.device)
        return torch.ops.ggml.rope(x, pos, self.n_dims, self.freq_base)


class TestExportPreservation:
    def test_rope_not_decomposed(self):
        """ggml.rope survives torch.export without being decomposed."""
        model = SimpleRopeModel(n_dims=16, freq_base=10000.0)
        model.eval()
        x = torch.randn(1, 8, 4, 16)

        with torch.no_grad():
            ep = export(model, (x,))

        # Check that ggml.rope is in the graph
        rope_nodes = [
            n for n in ep.graph.nodes
            if n.op == "call_function" and "ggml.rope" in str(n.target)
        ]
        assert len(rope_nodes) == 1, (
            f"Expected 1 ggml.rope node, found {len(rope_nodes)}. "
            f"Ops: {[str(n.target) for n in ep.graph.nodes if n.op == 'call_function']}"
        )

    def test_rope_delegated(self):
        """ggml.rope is delegated to ggml backend partition."""
        model = SimpleRopeModel(n_dims=16, freq_base=10000.0)
        model.eval()
        x = torch.randn(1, 8, 4, 16)

        with torch.no_grad():
            ep = export(model, (x,))
        edge_mgr = to_edge_rewrite_and_lower(
            ep, ep_passes=[], partitioner=[GgmlPartitioner()],
        )
        # After lowering, ops should be delegated (executorch_call_delegate)
        graph = edge_mgr.exported_program().graph
        delegate_nodes = [
            n for n in graph.nodes
            if n.op == "call_function" and "executorch_call_delegate" in str(n.target)
        ]
        assert len(delegate_nodes) >= 1, (
            "ggml.rope was not delegated. "
            f"Ops: {[str(n.target) for n in graph.nodes if n.op == 'call_function']}"
        )


# --- E2E with native backend ---

@requires_native
class TestE2E:
    def test_rope_e2e(self):
        """End-to-end: export -> lower -> execute -> compare with eager."""
        model = SimpleRopeModel(n_dims=16, freq_base=10000.0)
        model.eval()
        x = torch.randn(1, 8, 4, 16)

        with torch.no_grad():
            eager_out = model(x)
            ep = export(model, (x,))

        edge_mgr = to_edge_rewrite_and_lower(
            ep, ep_passes=[], partitioner=[GgmlPartitioner()],
        )
        et = edge_mgr.to_executorch()
        pte = _load_for_executorch_from_buffer(et.buffer)
        result = pte.forward((x,))
        ggml_out = result[0]

        cos_sim = F.cosine_similarity(
            eager_out.flatten().unsqueeze(0),
            ggml_out.flatten().unsqueeze(0),
        ).item()
        assert cos_sim > 0.999, f"Cosine similarity {cos_sim:.6f} too low"
        diff = (eager_out - ggml_out).abs().max().item()
        assert diff < 1e-4, f"Max abs diff {diff} >= 1e-4"


# --- Module swap test ---

class TestModuleSwap:
    def test_swap_encoder_rope(self):
        """swap_encoder_rope replaces attention modules with GgmlEncoderAttention."""
        # Minimal encoder-like structure
        from executorch.examples.models.voxtral_realtime.model import (
            CausalEncoderLayer,
            VoxtralRealtimeConfig,
            EncoderAttention,
        )
        config = VoxtralRealtimeConfig()
        layer = CausalEncoderLayer(config)
        assert isinstance(layer.attention, EncoderAttention)

        # Create a fake model with .encoder.layers
        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Module()
                self.encoder.layers = nn.ModuleList([layer])

        model = FakeModel()
        swap_encoder_rope(model, freq_base=config.enc_rope_theta)
        assert isinstance(model.encoder.layers[0].attention, GgmlEncoderAttention)

    def test_swapped_attention_matches_reference(self):
        """GgmlEncoderAttention output matches original EncoderAttention."""
        from executorch.examples.models.voxtral_realtime.model import (
            EncoderAttention,
            precompute_freqs_cis,
        )
        n_heads, head_dim, dim = 4, 16, 64
        freq_base = 10000.0
        B, T = 1, 8

        # Original
        orig = EncoderAttention(dim, n_heads, head_dim)
        orig.eval()

        # Swapped
        swapped = GgmlEncoderAttention(n_heads, head_dim, freq_base)
        swapped.wq.weight = orig.wq.weight
        swapped.wq.bias = orig.wq.bias
        swapped.wk.weight = orig.wk.weight
        swapped.wk.bias = orig.wk.bias
        swapped.wv.weight = orig.wv.weight
        swapped.wv.bias = orig.wv.bias
        swapped.wo.weight = orig.wo.weight
        swapped.wo.bias = orig.wo.bias
        swapped.eval()

        x = torch.randn(B, T, dim)
        freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, T, freq_base)

        with torch.no_grad():
            ref_out = orig(x, freqs_cos, freqs_sin)
            fused_out = swapped(x, freqs_cos, freqs_sin)

        cos_sim = F.cosine_similarity(
            ref_out.flatten().unsqueeze(0),
            fused_out.flatten().unsqueeze(0),
        ).item()
        assert cos_sim > 0.999, f"Cosine similarity {cos_sim:.6f} too low"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
