"""torch.cond preprocess support.

Walks each cond branch as a fresh subgraph (with its own IR tensor ID
space sharing the parent's data_store), builds an IrSubgraph, and emits
one OP_COND IR tensor per cond output (paralleling the cond's tuple
return). Downstream `getitem(cond, k)` resolves to the k-th OP_COND
tensor via the existing getitem handler.
"""

from __future__ import annotations

import struct
from typing import List, Tuple

import torch

from executorch_ggml.ops._context import PreprocessContext
from executorch_ggml.ops._helpers import (
    _resolve_shape,
    _pytorch_shape_to_ggml_ne,
    _torch_dtype_to_ir_type,
)
from executorch_ggml.serialize import (
    IrSubgraph,
    IrTensor,
    OP_COND,
    OP_NONE,
    TYPE_F32,
)


def _is_cond_target(target) -> bool:
    """Recognise torch.ops.higher_order.cond at both export and edge level."""
    if target is torch.ops.higher_order.cond:
        return True
    return (
        getattr(target, "__name__", "") == "cond"
        and type(target).__name__ == "CondOp"
    )


def _build_subgraph(
    parent_ctx: PreprocessContext,
    sub_gm: torch.fx.GraphModule,
    cond_operand_ids: List[int],
    operand_node_meta: List[dict],
) -> Tuple[IrSubgraph, int]:
    """Build an IrSubgraph from a cond branch's GraphModule.

    Returns (subgraph, subgraph_index_in_parent_ctx). Subgraph tensors use
    the same global ID counter as the parent (one ID space across the
    whole IR — simpler than a fresh per-subgraph counter, and the runtime
    can still distinguish subgraph tensors via Subgraph.tensors[]).

    `cond_operand_ids[k]` is the parent IR tensor ID for the k-th cond
    operand; the subgraph's k-th placeholder binds to it at runtime.
    `operand_node_meta[k]` carries shape/dtype info for that placeholder
    (used to materialize the placeholder IR tensor).
    """
    # Lazy import to avoid circular dep with dispatch_op.
    from executorch_ggml.ops import dispatch_op

    # Fresh sub-context that shares the parent's edge_program reference,
    # data_store, and ID counter so nothing in the subgraph collides.
    sub_ctx = PreprocessContext(
        parent_ctx.edge_program, parent_ctx.data_store, parent_ctx.quant_config,
    )
    # Share the parent's ID counter so all tensor IDs are globally unique.
    # We hand the counter back at the end of subgraph build.
    sub_ctx._next_id = parent_ctx._next_id
    sub_ctx.sym_id_map = parent_ctx.sym_id_map

    # Walk subgraph placeholders, allocating an IR tensor per placeholder
    # whose data is "fed in" from the cond's parent operand at runtime.
    # We mark the placeholder as `is_input=True` with a pseudo input_index
    # equal to its position in the operand list — the C++ runtime resolves
    # subgraph inputs via the parent's `cond_operand_ids` rather than the
    # method-level input list, so the index is local to the cond.
    placeholder_ids: List[int] = []
    placeholder_idx = 0
    for n in sub_gm.graph.nodes:
        if n.op != "placeholder":
            continue
        meta_val = n.meta.get("val")
        if meta_val is None:
            # Placeholders without meta usually correspond to scalars or
            # symbol inputs — unsupported for now.
            raise RuntimeError(
                f"cond subgraph placeholder {n.name} has no meta['val']"
            )
        shape = _resolve_shape(meta_val)
        dtype = getattr(meta_val, "dtype", torch.float32)
        if dtype == torch.int64:
            dtype = torch.int32
        tid = sub_ctx.alloc_id()
        sub_ctx.ir_tensors.append(
            IrTensor(
                tensor_id=tid,
                tensor_type=_torch_dtype_to_ir_type(dtype),
                ne=_pytorch_shape_to_ggml_ne(shape),
                op=OP_NONE,
                is_input=True,
                input_index=placeholder_idx,
            )
        )
        sub_ctx.node_to_id[n] = tid
        placeholder_ids.append(tid)
        placeholder_idx += 1

    if len(placeholder_ids) != len(cond_operand_ids):
        raise RuntimeError(
            f"cond subgraph has {len(placeholder_ids)} placeholders but "
            f"cond was given {len(cond_operand_ids)} operands"
        )

    # Walk subgraph call_function nodes via the normal dispatch path. The
    # handlers read ctx.node_to_id (which we've populated for placeholders)
    # and append to ctx.ir_tensors.
    output_node = None
    for n in sub_gm.graph.nodes:
        if n.op == "placeholder":
            continue
        if n.op == "output":
            output_node = n
            continue
        if n.op == "get_attr":
            # Most likely a constant tensor attr (rare in cond branches).
            # Skip — if a real op needs it, it'll fail at dispatch.
            continue
        if n.op != "call_function":
            raise RuntimeError(
                f"cond subgraph: unexpected node op {n.op} ({n.name})"
            )
        target = n.target
        target_str = str(target)
        # Nested cond is rejected at the partitioner level; guard here too.
        if _is_cond_target(target):
            raise RuntimeError(
                "Nested torch.cond inside a cond branch is not supported"
            )
        if dispatch_op(sub_ctx, n, target_str):
            continue
        if "getitem" in target_str:
            src_node = n.args[0]
            idx = int(n.args[1])
            src_val = sub_ctx.node_to_id.get(src_node)
            if isinstance(src_val, list):
                sub_ctx.node_to_id[n] = src_val[idx] if idx < len(src_val) else None
            elif isinstance(src_val, int) and idx == 0:
                sub_ctx.node_to_id[n] = src_val
            continue
        raise RuntimeError(
            f"cond subgraph: unsupported op {target} ({n.name})"
        )

    if output_node is None:
        raise RuntimeError("cond subgraph has no output node")

    out_args = output_node.args[0]
    if not isinstance(out_args, (list, tuple)):
        out_args = [out_args]
    output_ids: List[int] = []
    for out_n in out_args:
        if not isinstance(out_n, torch.fx.Node):
            raise RuntimeError(
                f"cond subgraph: non-Node output {out_n!r}"
            )
        v = sub_ctx.node_to_id.get(out_n)
        if v is None:
            raise RuntimeError(
                f"cond subgraph: output node {out_n.name} unresolved"
            )
        if isinstance(v, list):
            raise RuntimeError(
                f"cond subgraph: output {out_n.name} resolved to list (tuple-yielding op)"
            )
        output_ids.append(int(v))

    # Hand the global ID counter back to the parent.
    parent_ctx._next_id = sub_ctx._next_id

    sg = IrSubgraph(
        tensors=list(sub_ctx.ir_tensors),
        input_tensor_ids=list(placeholder_ids),
        output_tensor_ids=output_ids,
    )
    sg_index = len(parent_ctx.subgraphs)
    parent_ctx.subgraphs.append(sg)
    return sg, sg_index


def handle_cond(parent_ctx: PreprocessContext, node: torch.fx.Node):
    """Lower a torch.cond call_function node into IR.

    cond.args = (predicate, true_attr, false_attr, operands_tuple).

    Emits one OP_COND IR tensor per output of the cond's tuple return.
    Each OP_COND tensor's op_params encodes (num_outputs, output_index)
    so the runtime knows which subgraph output to splice in.
    """
    if len(node.args) < 4:
        raise RuntimeError(f"cond node {node.name}: expected 4 args, got {len(node.args)}")
    predicate, true_attr, false_attr, operands = node.args

    # Resolve operand IR tensor IDs in the parent.
    if not isinstance(operands, (list, tuple)):
        raise RuntimeError(f"cond {node.name}: operands must be tuple, got {type(operands)}")
    operand_ids: List[int] = []
    operand_metas: List[dict] = []
    for op in operands:
        if not isinstance(op, torch.fx.Node):
            raise RuntimeError(f"cond {node.name}: operand {op!r} is not an FX node")
        op_id = parent_ctx.node_to_id.get(op)
        if op_id is None:
            raise RuntimeError(
                f"cond {node.name}: operand {op.name} not yet in node_to_id"
            )
        if isinstance(op_id, list):
            raise RuntimeError(
                f"cond {node.name}: operand {op.name} resolved to list — unexpected"
            )
        operand_ids.append(int(op_id))
        operand_metas.append(op.meta)

    # Resolve subgraph GraphModules from the parent's owning_module.
    parent_gm = node.graph.owning_module
    true_gm = getattr(parent_gm, true_attr.target if isinstance(true_attr, torch.fx.Node) else true_attr, None)
    false_gm = getattr(parent_gm, false_attr.target if isinstance(false_attr, torch.fx.Node) else false_attr, None)
    if not isinstance(true_gm, torch.fx.GraphModule) or not isinstance(false_gm, torch.fx.GraphModule):
        raise RuntimeError(f"cond {node.name}: branches are not GraphModule attrs")

    _, true_idx = _build_subgraph(parent_ctx, true_gm, operand_ids, operand_metas)
    _, false_idx = _build_subgraph(parent_ctx, false_gm, operand_ids, operand_metas)

    true_sg = parent_ctx.subgraphs[true_idx]
    false_sg = parent_ctx.subgraphs[false_idx]
    if len(true_sg.output_tensor_ids) != len(false_sg.output_tensor_ids):
        raise RuntimeError(
            f"cond {node.name}: branches have different output counts: "
            f"{len(true_sg.output_tensor_ids)} vs {len(false_sg.output_tensor_ids)}"
        )
    n_outputs = len(true_sg.output_tensor_ids)

    # The cond node's meta['val'] is a tuple of FakeTensors (one per output).
    # Materialize one OP_COND tensor per output, sharing the same subgraph
    # references and operand IDs. op_params = (num_outputs, output_index)
    # as int32 little-endian.
    cond_val = node.meta.get("val")
    if not isinstance(cond_val, (list, tuple)):
        # Single-output cond (rare): wrap.
        cond_val = [cond_val] if cond_val is not None else [None] * n_outputs
    if len(cond_val) != n_outputs:
        raise RuntimeError(
            f"cond {node.name}: meta['val'] has {len(cond_val)} entries, "
            f"expected {n_outputs}"
        )

    output_tids: List[int] = []
    for out_idx in range(n_outputs):
        v = cond_val[out_idx]
        shape = _resolve_shape(v) if v is not None else []
        dtype = getattr(v, "dtype", torch.float32) if v is not None else torch.float32
        if dtype == torch.int64:
            dtype = torch.int32
        tid = parent_ctx.alloc_id()
        parent_ctx.ir_tensors.append(
            IrTensor(
                tensor_id=tid,
                tensor_type=_torch_dtype_to_ir_type(dtype),
                ne=_pytorch_shape_to_ggml_ne(shape),
                op=OP_COND,
                src_ids=[operand_ids[0]] if False else [],  # predicate goes here
                op_params=struct.pack("<ii", n_outputs, out_idx),
                subgraph_ids=[true_idx, false_idx],
                cond_operand_ids=list(operand_ids),
            )
        )
        output_tids.append(tid)

    # Wire the predicate as the COND tensor's first src_id. We know the
    # predicate is a parent FX node already in node_to_id.
    if isinstance(predicate, torch.fx.Node):
        pred_id = parent_ctx.node_to_id.get(predicate)
        if pred_id is None:
            raise RuntimeError(
                f"cond {node.name}: predicate {predicate.name} not in node_to_id"
            )
        if isinstance(pred_id, list):
            raise RuntimeError(
                f"cond {node.name}: predicate {predicate.name} is a tuple"
            )
        # Patch all OP_COND tensors with the same predicate src.
        for tid in output_tids:
            for ir_t in parent_ctx.ir_tensors:
                if ir_t.id == tid:
                    ir_t.src_ids = [int(pred_id)]
                    break

    # The cond node itself yields a tuple — getitem(cond, k) reads index k.
    parent_ctx.node_to_id[node] = list(output_tids)
