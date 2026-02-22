"""Test the llama.update_cache op support in ggml backend."""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, "python")

# Register the llama custom ops
from executorch.extension.llm.custom_ops import custom_ops  # noqa


def test_update_cache_op():
    """Test that llama.update_cache op is available and works."""
    # Check if the op is available
    print("Checking llama.update_cache op...")
    try:
        op = torch.ops.llama.update_cache
        print(f"  Op available: {op}")
    except AttributeError:
        print("  Op NOT available")
        return

    # Simple test model using update_cache
    class UpdateCacheModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("cache", torch.zeros(1, 8, 4, 16))

        def forward(self, value, start_pos):
            # value: [1, 1, 4, 16]
            torch.ops.llama.update_cache(value, self.cache, start_pos)
            return self.cache

    model = UpdateCacheModel()
    model.eval()

    value = torch.randn(1, 1, 4, 16)
    start_pos = 0

    with torch.no_grad():
        result = model(value, start_pos)
        print(f"Direct execution result shape: {result.shape}")
        nonzero = (result != 0).sum().item()
        print(f"Cache updated: {nonzero} non-zero elements")

    # Test with export
    print("\nExporting model...")
    with torch.no_grad():
        model.cache.zero_()
        ep = torch.export.export(model, (value, torch.tensor(start_pos)))

    print("Graph ops:")
    for n in ep.graph.nodes:
        if n.op == "call_function":
            print(f"  {n.target}")

    # Check if update_cache is in the graph
    has_update_cache = any(
        "update_cache" in str(n.target)
        for n in ep.graph.nodes
        if n.op == "call_function"
    )
    print(f"\nHas update_cache: {has_update_cache}")

    if has_update_cache:
        # Try to delegate with ggml
        from executorch_ggml import GgmlPartitioner
        from executorch.exir import to_edge_transform_and_lower

        print("\nLowering with GgmlPartitioner...")
        edge = to_edge_transform_and_lower(ep, partitioner=[GgmlPartitioner()])

        delegated = sum(
            1
            for n in edge.exported_program().graph.nodes
            if n.op == "call_function" and "executorch_call_delegate" in str(n.target)
        )
        total = sum(
            1 for n in edge.exported_program().graph.nodes if n.op == "call_function"
        )
        print(f"Delegation: {delegated}/{total} ops delegated")


if __name__ == "__main__":
    test_update_cache_op()
