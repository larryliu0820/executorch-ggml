"""Common Subexpression Elimination for FX graphs.

Merges duplicate call_function nodes that have identical (target, args, kwargs).
This is essential after fuse_projections: the shared-weight _SlicedLinear modules
produce N identical aten.linear calls (same input, same weight placeholder) that
CSE merges into a single linear + N slices.

Usage (after torch.export, after other graph passes):
    from executorch_ggml.passes.cse_pass import eliminate_common_subexpressions
    n = eliminate_common_subexpressions(ep.graph_module)
"""

import torch


def eliminate_common_subexpressions(gm: torch.fx.GraphModule) -> int:
    """Merge duplicate call_function nodes with identical inputs.

    Two nodes are duplicates if they have the same target, args, and kwargs.
    Node args are compared by identity (same FX Node object), not by value.

    Args:
        gm: FX GraphModule to optimize in-place.

    Returns:
        Number of nodes eliminated.
    """
    graph = gm.graph
    eliminated = 0

    # Map from (target, args_key, kwargs_key) → first node with that signature
    seen: dict = {}

    for node in list(graph.nodes):
        if node.op != "call_function":
            continue

        key = _node_key(node)
        if key in seen:
            canonical = seen[key]
            node.replace_all_uses_with(canonical)
            graph.erase_node(node)
            eliminated += 1
        else:
            seen[key] = node

    if eliminated > 0:
        graph.eliminate_dead_code()
        gm.recompile()

    return eliminated


def _node_key(node: torch.fx.Node) -> tuple:
    """Create a hashable key for a call_function node.

    Args are keyed by identity (id) for Node objects, by value for literals.
    """
    return (
        node.target,
        _args_key(node.args),
        _kwargs_key(node.kwargs),
    )


def _args_key(args: tuple) -> tuple:
    """Recursively create a hashable key for args."""
    result = []
    for a in args:
        if isinstance(a, torch.fx.Node):
            result.append(("node", id(a)))
        elif isinstance(a, (list, tuple)):
            result.append(("seq", _args_key(tuple(a))))
        else:
            result.append(("val", a))
    return tuple(result)


def _kwargs_key(kwargs: dict) -> tuple:
    """Create a hashable key for kwargs."""
    items = []
    for k in sorted(kwargs.keys()):
        v = kwargs[k]
        if isinstance(v, torch.fx.Node):
            items.append((k, "node", id(v)))
        elif isinstance(v, (list, tuple)):
            items.append((k, "seq", _args_key(tuple(v))))
        else:
            items.append((k, "val", v))
    return tuple(items)
