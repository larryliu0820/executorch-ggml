"""AOT graph pass: fuse decomposed RoPE into ggml.rope custom op.

Matches the HuggingFace rotate_half pattern:
    add(mul(x, cos), mul(cat(neg(slice(x, half:)), slice(x, :half)), sin))
and replaces it with:
    ggml.rope(x_pre_transpose, cache_position, head_dim, freq_base, mode=2)

Usage:
    from executorch_ggml.passes.fuse_rope_pass import FuseRoPEPass
    pass_ = FuseRoPEPass(head_dim=128, freq_base=1e6)
    # Use as transform_passes= in to_edge_transform_and_lower
"""

import torch
from torch.fx import Node

import executorch_ggml.rope_op  # noqa: F401 — registers torch.ops.ggml.rope


class FuseRoPEPass(torch.fx.Interpreter):
    """Not actually an Interpreter — just a pass using the class name convention."""

    pass


def _is_op(node: Node, op_name: str) -> bool:
    return (
        node.op == "call_function"
        and hasattr(node.target, "__name__")
        and node.target.__name__ == op_name
    ) or (node.op == "call_function" and str(node.target).endswith(f".{op_name}"))


def _match_target(node: Node, target) -> bool:
    return node.op == "call_function" and node.target == target


def _match_rotate_half(mul_node: Node):
    """Check if mul_node is MUL(rotate_half(x), sin).

    rotate_half(x) = cat([neg(x[..., half:]), x[..., :half]], dim=-1)

    Returns (x_node, sin_node) or None.
    """
    if not _match_target(mul_node, torch.ops.aten.mul.Tensor):
        return None

    for cat_idx in (0, 1):
        cat_node = mul_node.args[cat_idx]
        sin_node = mul_node.args[1 - cat_idx]

        if not isinstance(cat_node, Node):
            continue
        if not _match_target(cat_node, torch.ops.aten.cat.default):
            continue

        # cat args: ([neg_result, slice_result], dim)
        cat_args = cat_node.args[0]
        if not isinstance(cat_args, (list, tuple)) or len(cat_args) != 2:
            continue

        # One should be neg(slice(x)), the other slice(x)
        neg_node = None
        pos_node = None
        for ci in (0, 1):
            if isinstance(cat_args[ci], Node) and _match_target(
                cat_args[ci], torch.ops.aten.neg.default
            ):
                neg_node = cat_args[ci]
                pos_node = cat_args[1 - ci]
                break

        if neg_node is None or pos_node is None:
            continue

        # neg_node.args[0] should be slice(x, dim, half, end)
        neg_input = neg_node.args[0]
        if not isinstance(neg_input, Node) or not _match_target(
            neg_input, torch.ops.aten.slice.Tensor
        ):
            continue

        # pos_node should be slice(x, dim, 0, half)
        if not isinstance(pos_node, Node) or not _match_target(
            pos_node, torch.ops.aten.slice.Tensor
        ):
            continue

        # Both slices should share the same source tensor
        x_from_neg = neg_input.args[0]
        x_from_pos = pos_node.args[0]
        if x_from_neg != x_from_pos:
            continue

        return (x_from_neg, sin_node)

    return None


def fuse_rope_in_graph(
    gm: torch.fx.GraphModule,
    head_dim: int,
    freq_base: float,
    cache_position_name: str = "cache_position",
) -> int:
    """Walk the graph and replace decomposed RoPE with ggml.rope calls.

    Returns the number of fusions performed.
    """
    graph = gm.graph

    # Find the cache_position placeholder
    cache_pos_node = None
    for node in graph.nodes:
        if node.op == "placeholder" and node.name == cache_position_name:
            cache_pos_node = node
            break
    if cache_pos_node is None:
        return 0

    fused = 0
    nodes_to_check = list(graph.nodes)  # snapshot

    for node in nodes_to_check:
        if not _match_target(node, torch.ops.aten.add.Tensor):
            continue

        a, b = node.args[0], node.args[1]
        if not isinstance(a, Node) or not isinstance(b, Node):
            continue

        # One arg should be MUL(x, cos), the other MUL(rotate_half(x), sin)
        rot_match_a = _match_rotate_half(a)
        rot_match_b = _match_rotate_half(b)

        if rot_match_a and not rot_match_b:
            # a = rotate_half branch, b = x*cos branch
            x_rot, sin_node = rot_match_a
            cos_mul = b
        elif rot_match_b and not rot_match_a:
            # b = rotate_half branch, a = x*cos branch
            x_rot, sin_node = rot_match_b
            cos_mul = a
        else:
            continue

        # cos_mul should be MUL(x, cos) — verify x matches
        if not _match_target(cos_mul, torch.ops.aten.mul.Tensor):
            continue

        x_cos = None
        for ci in (0, 1):
            if cos_mul.args[ci] == x_rot:
                x_cos = cos_mul.args[ci]
                break
        if x_cos is None:
            continue

        # x_rot is in [B, H, T, D] layout (after transpose).
        # ggml.rope expects [B, T, H, D]. Find the pre-transpose tensor.
        # x_rot should come from transpose(view(linear(...)))
        x_pre_transpose = x_rot
        if _match_target(x_rot, torch.ops.aten.transpose.int):
            x_pre_transpose = x_rot.args[0]  # before transpose

        # Insert ggml.rope before the add node
        with graph.inserting_before(node):
            rope_node = graph.call_function(
                torch.ops.ggml.rope.default,
                args=(x_pre_transpose, cache_pos_node, head_dim, freq_base, 2),
            )
            # Copy metadata from the pre-transpose tensor for shape info
            if hasattr(x_pre_transpose, "meta") and "val" in x_pre_transpose.meta:
                rope_node.meta["val"] = x_pre_transpose.meta["val"]

            # The add result was in [B, H, T, D]. Our rope output is [B, T, H, D].
            # Re-apply the transpose to match downstream expectations.
            if x_pre_transpose != x_rot:
                # x_rot was transpose(x_pre, dim0, dim1)
                dim0 = x_rot.args[1]
                dim1 = x_rot.args[2]
                transpose_node = graph.call_function(
                    torch.ops.aten.transpose.int,
                    args=(rope_node, dim0, dim1),
                )
                if "val" in node.meta:
                    transpose_node.meta["val"] = node.meta["val"]
                node.replace_all_uses_with(transpose_node)
            else:
                node.replace_all_uses_with(rope_node)

        fused += 1

    if fused > 0:
        graph.eliminate_dead_code()
        gm.recompile()

    return fused


class FuseRoPEPass:
    """ExecuTorch-compatible transform pass for RoPE fusion."""

    def __init__(self, head_dim: int = 128, freq_base: float = 1e6):
        self.head_dim = head_dim
        self.freq_base = freq_base

    def __call__(self, gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
        from torch.fx.passes.infra.pass_base import PassResult

        n = fuse_rope_in_graph(gm, self.head_dim, self.freq_base)
        if n > 0:
            print(f"FuseRoPEPass: fused {n} RoPE instances")
        return PassResult(gm, n > 0)
