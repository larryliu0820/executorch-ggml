"""Fuse MoE routing + expert dispatch into a single ggml.moe_ffn op.

Detects the pattern produced by SparseMoE.forward():
    scores = gate(x)                    # router matmul
    vals, idx = topk(scores, k)         # expert selection
    idx = idx.to(I32)
    weights = softmax(vals)             # normalize
    for k in range(top_k):
        expert_out += w[k] * down(silu(gate[idx[k]](x)) * up[idx[k]](x))

Replaces with:
    ggml.moe_ffn(x, gate_exps, up_exps, down_exps, topk_indices, topk_weights,
                 n_expert, top_k)

The C++ runtime maps ggml.moe_ffn to llama.cpp's build_moe_ffn sequence:
    argsort_top_k → softmax → mul_mat_id(gate) → mul_mat_id(up) →
    swiglu → mul_mat_id(down) → weighted sum
"""

import struct
from typing import Optional, Tuple

import torch
from torch.fx import Node


# Register the ggml.moe_ffn custom op using FRAGMENT to avoid conflict with rope_op.py
_LIB = torch.library.Library("ggml", "FRAGMENT")
_LIB.define(
    "moe_ffn(Tensor input, Tensor gate_inp, Tensor gate_exps, Tensor up_exps, "
    "Tensor down_exps, int n_expert, int top_k) -> Tensor"
)


@torch.library.impl(_LIB, "moe_ffn", "CompositeExplicitAutograd")
def _moe_ffn_impl(input, gate_inp, gate_exps, up_exps, down_exps, n_expert, top_k):
    """Fallback implementation for tracing (not used at runtime)."""
    scores = input @ gate_inp.T
    expert_weights, expert_indices = torch.topk(scores, top_k, dim=-1)
    expert_weights = expert_weights.softmax(dim=-1)
    T, D = input.shape
    out = torch.zeros_like(input)
    for k in range(top_k):
        idx = expert_indices[:, k]
        w = expert_weights[:, k]
        gate = torch.bmm(gate_exps[idx], input.unsqueeze(2)).squeeze(2)
        up = torch.bmm(up_exps[idx], input.unsqueeze(2)).squeeze(2)
        hidden = torch.silu(gate) * up
        down = torch.bmm(down_exps[idx], hidden.unsqueeze(2)).squeeze(2)
        out = out + w.unsqueeze(1) * down
    return out


@torch.library.impl_abstract("ggml::moe_ffn")
def _moe_ffn_abstract(input, gate_inp, gate_exps, up_exps, down_exps, n_expert, top_k):
    """Shape inference for torch.export."""
    return torch.empty_like(input)


def _is_topk(node: Node) -> bool:
    """Check if node is aten.topk."""
    return (node.op == "call_function" and
            "topk" in str(node.target).lower())


def _trace_gate_input(topk_node: Node) -> Optional[Tuple[Node, Node]]:
    """Trace backward from topk to find (input_flat, gate_weight).

    Pattern: topk(linear(view(x), gate_weight), k)
    Returns: (view_node, gate_weight_param) or None.
    """
    # topk.args[0] should be the router scores
    scores = topk_node.args[0]
    if not isinstance(scores, Node):
        return None

    # scores = linear(x_flat, gate_weight) or mm(x_flat, gate_weight.T)
    scores_target = str(scores.target) if hasattr(scores, 'target') else ""
    if "linear" in scores_target:
        x_flat = scores.args[0]
        gate_weight = scores.args[1]
        return (x_flat, gate_weight)
    elif "mm" in scores_target:
        x_flat = scores.args[0]
        gate_weight = scores.args[1]
        return (x_flat, gate_weight)

    return None


def _find_expert_weights(topk_node: Node) -> Optional[Tuple[Node, Node, Node]]:
    """Find gate_exps, up_exps, down_exps parameters from the expert loop.

    The loop pattern after topk:
        getitem(topk, 0) → values
        getitem(topk, 1) → indices
        ... select(indices, 1, k) → idx_k
        ... index(gate_exps, [idx_k]) → gate for expert k
        ... index(up_exps, [idx_k]) → up for expert k
        ... index(down_exps, [idx_k]) → down for expert k

    We find the 3 parameter nodes used in the first index ops.
    """
    # Find getitem nodes
    getitem_vals = None
    getitem_idxs = None
    for user in topk_node.users:
        s = str(user.target)
        if "getitem" in s:
            if user.args[1] == 0:
                getitem_vals = user
            elif user.args[1] == 1:
                getitem_idxs = user

    if getitem_idxs is None:
        return None

    # Find the I32 cast of indices (to_21 in the trace)
    idx_i32 = None
    for user in getitem_idxs.users:
        s = str(user.target)
        if "to" in s or "cast" in s or "_to_copy" in s:
            idx_i32 = user
            break
    if idx_i32 is None:
        idx_i32 = getitem_idxs  # might not have cast

    # Find the first select(idx_i32, 1, 0) → first expert index
    first_select = None
    for user in idx_i32.users:
        s = str(user.target)
        if "select" in s:
            # select(tensor, dim=1, index=0) — first expert
            if len(user.args) >= 3 and user.args[2] == 0:
                first_select = user
                break

    if first_select is None:
        return None

    # Find index ops that use the first select as index
    # Pattern: index(expert_weight, [first_select])
    gate_exps = up_exps = down_exps = None
    for user in first_select.users:
        s = str(user.target)
        if "index" in s and "index_put" not in s:
            # The first arg is the expert weight tensor
            weight_node = user.args[0]
            if not isinstance(weight_node, Node):
                continue
            weight_name = weight_node.name if hasattr(weight_node, 'name') else ""
            if "gate" in weight_name and "inp" not in weight_name:
                gate_exps = weight_node
            elif "up" in weight_name:
                up_exps = weight_node
            elif "down" in weight_name:
                down_exps = weight_node

    if gate_exps and up_exps and down_exps:
        return (gate_exps, up_exps, down_exps)
    return None


def _find_last_add_in_loop(topk_node: Node, top_k: int) -> Optional[Node]:
    """Find the final accumulator add node (the output of the expert loop).

    The loop produces a chain: zeros_like → add → add → ... → final_add
    We need the last add before the shared expert.
    """
    # Find getitem for values (index 0)
    getitem_vals = None
    for user in topk_node.users:
        if "getitem" in str(user.target) and user.args[1] == 0:
            getitem_vals = user
            break
    if getitem_vals is None:
        return None

    # The softmax of values feeds into select nodes for weights
    # Follow: getitem_vals → softmax → to → select → ... → mul → add
    # Find the last add in the chain
    softmax_node = None
    for user in getitem_vals.users:
        if "softmax" in str(user.target).lower():
            softmax_node = user
            break
    if softmax_node is None:
        return None

    # The softmax feeds through to/cast → select → unsqueeze → mul → add chain
    # Find all add nodes that are part of the accumulator
    # Strategy: find zeros_like near topk, then follow the add chain
    topk_parent = topk_node.args[0]  # the scores node
    if isinstance(topk_parent, Node):
        x_flat = topk_parent.args[0] if hasattr(topk_parent, 'args') and topk_parent.args else None
    else:
        x_flat = None

    # Find zeros_like(x_flat) — the accumulator init
    zeros_node = None
    if x_flat:
        for user in x_flat.users:
            if "zeros_like" in str(user.target):
                zeros_node = user
                break

    if zeros_node is None:
        return None

    # Follow the add chain from zeros_node
    current = zeros_node
    for _ in range(top_k):
        next_add = None
        for user in current.users:
            if "add" in str(user.target) and "add_" not in str(user.target):
                next_add = user
                break
        if next_add is None:
            break
        current = next_add

    if current != zeros_node:
        return current  # The last add is the loop output
    return None


def _collect_loop_nodes(topk_node: Node, last_add: Node) -> set:
    """Collect all nodes between topk and the last accumulator add.

    These are the nodes to remove after fusion.
    """
    nodes = set()
    # BFS forward from topk through its users, stopping at shared expert nodes
    queue = [topk_node]
    while queue:
        n = queue.pop(0)
        if n in nodes:
            continue
        nodes.add(n)
        for user in n.users:
            # Stop at the last add's users (those are shared expert or output)
            if user == last_add:
                nodes.add(user)
                continue
            # Don't follow past the loop output
            if n == last_add:
                continue
            queue.append(user)

    return nodes


def fuse_moe_in_graph(gm: torch.fx.GraphModule, num_experts: int, top_k: int) -> int:
    """Fuse MoE subgraphs in the FX graph.

    Returns the number of MoE instances fused.
    """
    graph = gm.graph
    fused = 0

    for node in list(graph.nodes):
        if not _is_topk(node):
            continue

        # Step 1: trace backward to find input + gate_weight
        gate_info = _trace_gate_input(node)
        if gate_info is None:
            continue
        x_flat, gate_weight = gate_info

        # Step 2: find expert weight parameters
        expert_info = _find_expert_weights(node)
        if expert_info is None:
            continue
        gate_exps, up_exps, down_exps = expert_info

        # Step 3: find the last accumulator add (loop output)
        last_add = _find_last_add_in_loop(node, top_k)
        if last_add is None:
            continue

        # Step 4: replace with ggml.moe_ffn
        with graph.inserting_before(last_add):
            fused_node = graph.call_function(
                torch.ops.ggml.moe_ffn,
                args=(x_flat, gate_weight, gate_exps, up_exps, down_exps,
                      num_experts, top_k),
            )
            # Copy output metadata
            if "val" in last_add.meta:
                fused_node.meta["val"] = last_add.meta["val"]
            if "tensor_meta" in last_add.meta:
                fused_node.meta["tensor_meta"] = last_add.meta["tensor_meta"]

        # Replace the loop output with the fused node
        last_add.replace_all_uses_with(fused_node)

        # Remove dead nodes (the loop body)
        graph.eliminate_dead_code()

        fused += 1
        print(f"  fuse_moe: fused MoE instance {fused} "
              f"(gate={gate_weight.name}, experts={gate_exps.name})")

    if fused > 0:
        gm.recompile()

    return fused
