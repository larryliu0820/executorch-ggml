"""Decompose 5D RoPE stack+flatten into 4D-compatible ops via run_decompositions.

The RoPE pattern: stack([a, b], dim=-1).flatten(-2) interleaves a,b along last dim.
We decompose stack into unsqueeze+cat which the ggml VIEW handler already supports
by collapsing >4D shapes into 4D ggml tensors.

Actually: aten.stack is not in the core ATen decomposition table, but we can provide
a custom decomposition. torch.stack([a, b], -1) on [B, T, H, D/2] tensors:
  = cat([a.unsqueeze(-1), b.unsqueeze(-1)], dim=-1)
  → [B, T, H, D/2, 2]

The subsequent flatten(-2) produces [B, T, H, D].

With `aten.stack` decomposed, the graph only has unsqueeze (4D→5D), cat (5D), and
view/flatten (5D→4D). The ggml backend's VIEW handler already handles >4D→4D shape
collapsing correctly.
"""

import torch


def rope_stack_decomposition(tensors, dim=0):
    """Decompose stack into unsqueeze+cat to expose individual ops."""
    return torch.cat([t.unsqueeze(dim) for t in tensors], dim=dim)


ROPE_DECOMP_TABLE = {
    torch.ops.aten.stack.default: rope_stack_decomposition,
}
