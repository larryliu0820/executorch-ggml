"""Strip identity operations: dtype casts that don't change the type,
type_as when dtypes already match, and other no-op operations.

These are produced by models that do explicit F32 computation followed
by type_as(original) to restore the original dtype — when the model
is already in F32, these become identity ops.
"""

import torch


def strip_identity_ops(gm: torch.fx.GraphModule) -> int:
    """Remove identity cast/type_as/to ops from the graph.

    Returns the number of ops removed.
    """
    graph = gm.graph
    removed = 0

    for node in list(graph.nodes):
        if node.op != "call_function":
            continue

        target_str = str(node.target)

        # Skip non-cast ops
        if not any(x in target_str for x in ["to.", "type_as", "_to_copy"]):
            continue

        if len(node.args) < 1:
            continue

        src = node.args[0]
        if not hasattr(src, "meta") or "val" not in src.meta:
            continue

        src_fv = src.meta.get("val")
        dst_fv = node.meta.get("val")
        if src_fv is None or dst_fv is None:
            continue
        if not hasattr(src_fv, "dtype") or not hasattr(dst_fv, "dtype"):
            continue

        # Identity: same dtype in and out
        if src_fv.dtype == dst_fv.dtype:
            node.replace_all_uses_with(src)
            graph.erase_node(node)
            removed += 1

    if removed > 0:
        graph.eliminate_dead_code()
        gm.recompile()

    return removed
