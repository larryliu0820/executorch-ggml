"""
Test KV cache support in the ggml backend.

Tests the index_put_ operation which is used for KV cache updates,
and verifies that multi-token generation works correctly.
"""

import pytest
import torch
import torch.nn as nn
import sys

sys.path.insert(0, "python")


class SimpleKVCache(nn.Module):
    """Minimal KV cache for testing."""

    def __init__(self, max_seq_len: int, n_heads: int, head_dim: int, dtype=torch.float32):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.n_heads = n_heads
        self.head_dim = head_dim
        # Shape: [batch=1, n_heads, max_seq_len, head_dim]
        cache_shape = (1, n_heads, max_seq_len, head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """Update cache at input_pos positions.

        k, v: [1, n_heads, seq_len, head_dim]
        input_pos: [seq_len] - positions to update
        """
        # This becomes aten.index_put_.default in the graph
        self.k_cache[:, :, input_pos] = k
        self.v_cache[:, :, input_pos] = v
        return self.k_cache, self.v_cache


class SimpleAttention(nn.Module):
    """Minimal attention with KV cache for testing."""

    def __init__(self, dim: int, n_heads: int, max_seq_len: int):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.max_seq_len = max_seq_len

        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

        self.kv_cache = SimpleKVCache(max_seq_len, n_heads, self.head_dim)

        # Causal mask
        mask = torch.full((max_seq_len, max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor, input_pos: torch.Tensor):
        """
        x: [1, seq_len, dim]
        input_pos: [seq_len] - positions in the sequence
        """
        bsz, seq_len, _ = x.shape

        # Project Q, K, V
        q = self.wq(x)  # [1, seq_len, dim]
        k = self.wk(x)
        v = self.wv(x)

        # Reshape to [1, n_heads, seq_len, head_dim]
        q = q.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Update KV cache
        k_cache, v_cache = self.kv_cache.update(input_pos, k, v)

        # For attention, use cached K/V up to current position
        cur_pos = input_pos[-1].item() + 1
        k_for_attn = k_cache[:, :, :cur_pos, :]
        v_for_attn = v_cache[:, :, :cur_pos, :]

        # Attention
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k_for_attn.transpose(-2, -1)) * scale

        # Apply causal mask (slice to current positions)
        mask = self.mask[input_pos][:, :cur_pos]
        attn = attn + mask.unsqueeze(0).unsqueeze(0)

        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v_for_attn)

        # Reshape back
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.dim)
        return self.wo(out)


class SimpleTransformer(nn.Module):
    """Minimal transformer with one attention layer for KV cache testing."""

    def __init__(self, vocab_size: int, dim: int, n_heads: int, max_seq_len: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.attention = SimpleAttention(dim, n_heads, max_seq_len)
        self.output = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor, input_pos: torch.Tensor):
        """
        tokens: [1, seq_len]
        input_pos: [seq_len]
        """
        x = self.embed(tokens)
        x = self.attention(x, input_pos)
        return self.output(x)


class TestKVCacheIndexPut:
    """Test index_put_ operation used in KV cache."""

    def test_index_put_basic(self):
        """Test basic index_put operation."""
        from executorch_ggml import GgmlPartitioner
        from executorch.extension.pybindings.portable_lib import (
            _load_for_executorch_from_buffer,
        )
        from executorch.exir import to_edge_transform_and_lower

        class IndexPutModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("cache", torch.zeros(1, 4, 8, 16))

            def forward(self, values: torch.Tensor, indices: torch.Tensor):
                # values: [1, 4, 1, 16], indices: [1]
                self.cache[:, :, indices] = values
                return self.cache

        model = IndexPutModel()
        model.eval()

        values = torch.randn(1, 4, 1, 16)
        indices = torch.tensor([3], dtype=torch.long)

        # Reference
        with torch.no_grad():
            model.cache.zero_()
            ref = model(values, indices).clone()

        # Export
        with torch.no_grad():
            model.cache.zero_()
            ep = torch.export.export(model, (values, indices))

        print(f"\nExported graph:")
        print(ep.graph)

        # Check if index_put is in the graph
        graph_str = str(ep.graph)
        assert "index_put" in graph_str.lower(), "Expected index_put in graph"
        print("Found index_put in exported graph")

        # Lower to ggml
        edge = to_edge_transform_and_lower(
            ep,
            partitioner=[GgmlPartitioner()],
        )
        print(f"\nDelegation summary:")
        print(edge.exported_program().graph)

        et_program = edge.to_executorch()
        pte_model = _load_for_executorch_from_buffer(et_program.buffer)

        # Reset cache and run
        model.cache.zero_()
        result = pte_model.forward((values, indices))[0]

        # Compare
        max_diff = (ref - result).abs().max().item()
        print(f"Max diff: {max_diff}")
        assert max_diff < 1e-4, f"Max diff too large: {max_diff}"


class TestKVCacheMultiToken:
    """Test KV cache with multi-token generation."""

    def test_two_token_generation(self):
        """Test generating 2 tokens with KV cache."""
        from executorch_ggml import GgmlPartitioner
        from executorch.extension.pybindings.portable_lib import (
            _load_for_executorch_from_buffer,
        )
        from executorch.exir import to_edge_transform_and_lower

        torch.manual_seed(42)

        vocab_size = 100
        dim = 64
        n_heads = 4
        max_seq_len = 32

        model = SimpleTransformer(vocab_size, dim, n_heads, max_seq_len)
        model.eval()

        # Test data - first token
        token1 = torch.tensor([[5]], dtype=torch.long)
        pos1 = torch.tensor([0], dtype=torch.long)

        # Test data - second token
        token2 = torch.tensor([[10]], dtype=torch.long)
        pos2 = torch.tensor([1], dtype=torch.long)

        # Get reference outputs using eager mode
        with torch.no_grad():
            # Reset cache
            model.attention.kv_cache.k_cache.zero_()
            model.attention.kv_cache.v_cache.zero_()

            ref1 = model(token1, pos1).clone()
            ref2 = model(token2, pos2).clone()

        print(f"Reference outputs computed")
        print(f"  Token 1 output shape: {ref1.shape}")
        print(f"  Token 2 output shape: {ref2.shape}")

        # Export the model
        with torch.no_grad():
            model.attention.kv_cache.k_cache.zero_()
            model.attention.kv_cache.v_cache.zero_()
            ep = torch.export.export(model, (token1, pos1))

        print(f"\nExported graph nodes:")
        for node in ep.graph.nodes:
            if node.op == "call_function":
                print(f"  {node.target}")

        # Lower to ggml
        edge = to_edge_transform_and_lower(
            ep,
            partitioner=[GgmlPartitioner()],
        )

        # Check delegation
        delegated_count = 0
        total_count = 0
        for node in edge.exported_program().graph.nodes:
            if node.op == "call_function":
                total_count += 1
                if "executorch_call_delegate" in str(node.target):
                    delegated_count += 1
        print(f"\nDelegation: {delegated_count}/{total_count} ops delegated")

        et_program = edge.to_executorch()
        pte_model = _load_for_executorch_from_buffer(et_program.buffer)

        # Run first token
        model.attention.kv_cache.k_cache.zero_()
        model.attention.kv_cache.v_cache.zero_()
        result1 = pte_model.forward((token1, pos1))[0]

        # Run second token
        result2 = pte_model.forward((token2, pos2))[0]

        # Compare
        diff1 = (ref1 - result1).abs().max().item()
        diff2 = (ref2 - result2).abs().max().item()

        print(f"\nResults:")
        print(f"  Token 1 max diff: {diff1:.6f}")
        print(f"  Token 2 max diff: {diff2:.6f}")

        # Verify token 2 used the cached K/V from token 1
        # by checking that its output is different from what we'd get
        # with an empty cache
        with torch.no_grad():
            model.attention.kv_cache.k_cache.zero_()
            model.attention.kv_cache.v_cache.zero_()
            # Run token 2 with fresh cache (should be different)
            ref2_fresh = model(token2, pos2).clone()

        diff_fresh = (ref2 - ref2_fresh).abs().max().item()
        print(f"  Token 2 diff (with vs without cache): {diff_fresh:.6f}")

        assert diff1 < 1e-3, f"Token 1 diff too large: {diff1}"
        assert diff2 < 1e-3, f"Token 2 diff too large: {diff2}"
        assert diff_fresh > 1e-3, "KV cache doesn't seem to be working - outputs are identical"

        print("\nKV cache test PASSED!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
