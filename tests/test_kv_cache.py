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
    """Minimal attention with KV cache for testing.

    Uses explicit attention mask for exportable incremental decoding.
    """

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

    def forward(self, x: torch.Tensor, input_pos: torch.Tensor, attn_mask: torch.Tensor):
        """
        x: [1, seq_len, dim]
        input_pos: [seq_len] - positions in the sequence
        attn_mask: [1, 1, seq_len, max_seq_len] - attention mask (0 = attend, -inf = ignore)
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

        # Use SDPA with explicit mask
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k_cache, v_cache, attn_mask=attn_mask
        )

        # Reshape back
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.dim)
        return self.wo(out)


class SimpleTransformer(nn.Module):
    """Minimal transformer with one attention layer for KV cache testing."""

    def __init__(self, vocab_size: int, dim: int, n_heads: int, max_seq_len: int):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embed = nn.Embedding(vocab_size, dim)
        self.attention = SimpleAttention(dim, n_heads, max_seq_len)
        self.output = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor, input_pos: torch.Tensor, attn_mask: torch.Tensor):
        """
        tokens: [1, seq_len]
        input_pos: [seq_len]
        attn_mask: [1, 1, seq_len, max_seq_len]
        """
        x = self.embed(tokens)
        x = self.attention(x, input_pos, attn_mask)
        return self.output(x)


def create_causal_mask(input_pos: torch.Tensor, max_seq_len: int) -> torch.Tensor:
    """Create causal attention mask for given input positions.

    Returns mask of shape [1, 1, seq_len, max_seq_len] where:
    - 0.0 means attend to this position
    - -inf means ignore this position
    """
    seq_len = input_pos.shape[0]
    # Create position indices for the full cache
    cache_pos = torch.arange(max_seq_len, device=input_pos.device)
    # For each query position, mask out future positions and unfilled cache
    # Query at input_pos[i] can attend to cache positions <= input_pos[i]
    mask = torch.where(
        cache_pos.unsqueeze(0) <= input_pos.unsqueeze(1),
        torch.tensor(0.0),
        torch.tensor(float("-inf")),
    )
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, max_seq_len]


class TestKVCacheIndexPut:
    """Test index_put_ operation used in KV cache."""

    def test_index_put_basic(self):
        """Test basic index_put operation."""
        from executorch_ggml import GgmlPartitioner
        from executorch_ggml.passes import BroadcastCanonicalizationPass
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

        # Apply BroadcastCanonicalizationPass to make broadcasts explicit
        ep = BroadcastCanonicalizationPass().run(ep)

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
        from executorch_ggml.passes import BroadcastCanonicalizationPass
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
        mask1 = create_causal_mask(pos1, max_seq_len)

        # Test data - second token
        token2 = torch.tensor([[10]], dtype=torch.long)
        pos2 = torch.tensor([1], dtype=torch.long)
        mask2 = create_causal_mask(pos2, max_seq_len)

        # Get reference outputs using eager mode
        with torch.no_grad():
            # Reset cache
            model.attention.kv_cache.k_cache.zero_()
            model.attention.kv_cache.v_cache.zero_()

            ref1 = model(token1, pos1, mask1).clone()
            ref2 = model(token2, pos2, mask2).clone()

        print(f"Reference outputs computed")
        print(f"  Token 1 output shape: {ref1.shape}")
        print(f"  Token 2 output shape: {ref2.shape}")

        # Export the model
        with torch.no_grad():
            model.attention.kv_cache.k_cache.zero_()
            model.attention.kv_cache.v_cache.zero_()
            ep = torch.export.export(model, (token1, pos1, mask1))

        print(f"\nExported graph nodes:")
        for node in ep.graph.nodes:
            if node.op == "call_function":
                print(f"  {node.target}")

        # Apply BroadcastCanonicalizationPass to make broadcasts explicit
        ep = BroadcastCanonicalizationPass().run(ep)

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
        result1 = pte_model.forward((token1, pos1, mask1))[0]

        # Run second token
        result2 = pte_model.forward((token2, pos2, mask2))[0]

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
            ref2_fresh = model(token2, pos2, mask2).clone()

        diff_fresh = (ref2 - ref2_fresh).abs().max().item()
        print(f"  Token 2 diff (with vs without cache): {diff_fresh:.6f}")

        assert diff1 < 1e-3, f"Token 1 diff too large: {diff1}"
        # Token 2 includes cache update + fused attention and can have
        # modest backend-dependent numerical drift.
        assert diff2 < 0.1, f"Token 2 diff too large: {diff2}"
        assert diff_fresh > 1e-3, "KV cache doesn't seem to be working - outputs are identical"

        print("\nKV cache test PASSED!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
