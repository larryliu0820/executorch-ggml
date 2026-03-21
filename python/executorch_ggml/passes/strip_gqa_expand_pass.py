"""AOT graph pass: strip GQA expand from SDPA K/V inputs.

PyTorch's SDPA expands K/V from num_kv_heads to num_attention_heads via
unsqueeze → expand → reshape before attention. ggml's flash_attn_ext handles
GQA natively (gqa_ratio = Q.ne[2] / K.ne[2]), so the expand is unnecessary.

This pass removes the expand chain and passes un-expanded K/V directly to SDPA.

Usage:
    from executorch_ggml.passes.strip_gqa_expand_pass import strip_gqa_expand
    n = strip_gqa_expand(ep.graph_module)
"""

import torch
from torch.fx import Node


def _trace_gqa_expand(node: Node):
    """If node is reshape(expand(unsqueeze(x))), return x. Else None."""
    if not (node.op == "call_function" and node.target == torch.ops.aten.reshape.default):
        return None

    expand = node.args[0]
    if not (
        isinstance(expand, Node)
        and expand.op == "call_function"
        and expand.target == torch.ops.aten.expand.default
    ):
        return None

    unsqueeze = expand.args[0]
    if not (
        isinstance(unsqueeze, Node)
        and unsqueeze.op == "call_function"
        and unsqueeze.target == torch.ops.aten.unsqueeze.default
    ):
        return None

    return unsqueeze.args[0]  # the un-expanded KV tensor


def strip_gqa_expand(gm: torch.fx.GraphModule) -> int:
    """Remove GQA expand from SDPA K/V args.

    Returns the number of SDPA calls modified.
    """
    graph = gm.graph
    count = 0

    for node in list(graph.nodes):
        if not (
            node.op == "call_function"
            and node.target == torch.ops.aten.scaled_dot_product_attention.default
        ):
            continue

        # SDPA args: (q, k, v, attn_mask?, ...)
        q, k, v = node.args[0], node.args[1], node.args[2]

        k_src = _trace_gqa_expand(k)
        v_src = _trace_gqa_expand(v)

        if k_src is None and v_src is None:
            continue

        # Replace K/V with un-expanded source, add enable_gqa=True
        new_args = list(node.args)
        if k_src is not None:
            new_args[1] = k_src
        if v_src is not None:
            new_args[2] = v_src
        node.args = tuple(new_args)

        # Set enable_gqa=True in kwargs so PyTorch SDPA knows about GQA
        new_kwargs = dict(node.kwargs)
        new_kwargs["enable_gqa"] = True
        node.kwargs = new_kwargs

        count += 1

    if count > 0:
        graph.eliminate_dead_code()
        gm.recompile()

    return count
