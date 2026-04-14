"""Mixture-of-Experts operator handlers: topk, sort, grouped_mm (MUL_MAT_ID),
log1p.

These ops are used in MoE routing (e.g. Qwen3.5-35B-A3B):
  - topk/sort: expert selection via gating scores
  - grouped_mm: expert-indexed batched matmul
  - log1p: numerical stability in routing (log-sum-exp style)
"""

import struct

import torch

from executorch_ggml.ops._registry import register_op
from executorch_ggml.ops._helpers import (
    _resolve_shape,
    _pytorch_shape_to_ggml_ne,
    _torch_dtype_to_ir_type,
)
from executorch_ggml.ops._sym_expr import _sym_dim_info_ggml
from executorch_ggml.serialize import (
    IrTensor,
    TYPE_F32,
    TYPE_I32,
    TYPE_I64,
    OP_TOPK,
    OP_TOPK_INDICES,
    OP_SORT,
    OP_SORT_INDICES,
    OP_MUL_MAT_ID,
    OP_LOG1P,
    OP_CAST,
    OP_EXP,
    OP_SUM,
    OP_CLAMP,
    OP_SLICE_SCATTER,
    OP_FULL,
    OP_NONE,
    OP_VIEW,
    pack_cast_params,
)


# ---------------------------------------------------------------------------
# topk — returns (values, indices) tuple
# ---------------------------------------------------------------------------

@register_op("aten.topk.default")
def handle_topk(ctx, node, target_str):
    """topk(input, k, dim=-1, largest=True, sorted=True) -> (values, indices).

    Creates two IR tensors:
      - OP_TOPK for the values output
      - OP_TOPK_INDICES for the indices output
    The getitem builtin in the main loop extracts each from the list stored
    in ctx.node_to_id[node].
    """
    input_node = node.args[0]
    k = int(node.args[1])
    dim = int(node.kwargs.get("dim", node.args[2] if len(node.args) > 2 else -1))
    largest = bool(node.kwargs.get("largest", node.args[3] if len(node.args) > 3 else True))
    # sorted kwarg not needed for IR — topk is always sorted

    input_id = ctx.node_to_id[input_node]

    # node.meta['val'] is a tuple of two FakeTensors: (values, indices)
    fake_vals = node.meta.get("val")

    # Get input ndim for dim normalization
    input_fv = input_node.meta.get("val") if hasattr(input_node, "meta") else None
    input_shape = _resolve_shape(input_fv)
    ndim = len(input_shape) if input_shape else 4
    if dim < 0:
        dim = ndim + dim

    # Pack op_params: k (int32), dim (int32), largest (int32)
    op_params = struct.pack("<iii", k, dim, int(largest))

    # --- Values tensor (OP_TOPK) ---
    if isinstance(fake_vals, (list, tuple)) and len(fake_vals) >= 2:
        val_fv = fake_vals[0]
    else:
        val_fv = fake_vals
    val_shape = _resolve_shape(val_fv)
    val_dtype = getattr(val_fv, "dtype", torch.float32) if val_fv is not None else torch.float32

    _vsym_v, _vexprs_v = _sym_dim_info_ggml(val_fv, ctx.sym_id_map)
    values_tid = ctx.alloc_id()
    ctx.ir_tensors.append(
        IrTensor(
            tensor_id=values_tid,
            tensor_type=_torch_dtype_to_ir_type(val_dtype),
            ne=_pytorch_shape_to_ggml_ne(val_shape),
            op=OP_TOPK,
            src_ids=[input_id],
            op_params=op_params,
            sym_dim_ids=_vsym_v,
            sym_dim_exprs=_vexprs_v,
        )
    )

    # --- Indices tensor (OP_TOPK_INDICES) ---
    if isinstance(fake_vals, (list, tuple)) and len(fake_vals) >= 2:
        idx_fv = fake_vals[1]
    else:
        idx_fv = val_fv
    idx_shape = _resolve_shape(idx_fv)
    idx_dtype = getattr(idx_fv, "dtype", torch.int64) if idx_fv is not None else torch.int64

    _vsym_i, _vexprs_i = _sym_dim_info_ggml(idx_fv, ctx.sym_id_map)
    indices_tid = ctx.alloc_id()
    ctx.ir_tensors.append(
        IrTensor(
            tensor_id=indices_tid,
            tensor_type=_torch_dtype_to_ir_type(idx_dtype),
            ne=_pytorch_shape_to_ggml_ne(idx_shape),
            op=OP_TOPK_INDICES,
            src_ids=[input_id],
            op_params=op_params,
            sym_dim_ids=_vsym_i,
            sym_dim_exprs=_vexprs_i,
        )
    )

    # Store as list for getitem resolution: [0]=values, [1]=indices
    ctx.node_to_id[node] = [values_tid, indices_tid]


# ---------------------------------------------------------------------------
# sort — returns (sorted_values, indices) tuple
# ---------------------------------------------------------------------------

@register_op("aten.sort.default")
def handle_sort(ctx, node, target_str):
    """sort(input, dim=-1, descending=False) -> (sorted, indices).

    Creates two IR tensors:
      - OP_SORT for the sorted values
      - OP_SORT_INDICES for the indices
    """
    input_node = node.args[0]
    dim = int(node.kwargs.get("dim", node.args[1] if len(node.args) > 1 else -1))
    descending = bool(node.kwargs.get("descending", node.args[2] if len(node.args) > 2 else False))

    input_id = ctx.node_to_id[input_node]

    # node.meta['val'] is a tuple of two FakeTensors: (sorted, indices)
    fake_vals = node.meta.get("val")

    # Get input ndim for dim normalization
    input_fv = input_node.meta.get("val") if hasattr(input_node, "meta") else None
    input_shape = _resolve_shape(input_fv)
    ndim = len(input_shape) if input_shape else 4
    if dim < 0:
        dim = ndim + dim

    # Pack op_params: dim (int32), descending (int32)
    op_params = struct.pack("<ii", dim, int(descending))

    # --- Sorted values tensor (OP_SORT) ---
    if isinstance(fake_vals, (list, tuple)) and len(fake_vals) >= 2:
        val_fv = fake_vals[0]
    else:
        val_fv = fake_vals
    val_shape = _resolve_shape(val_fv)
    val_dtype = getattr(val_fv, "dtype", torch.float32) if val_fv is not None else torch.float32

    _vsym_v, _vexprs_v = _sym_dim_info_ggml(val_fv, ctx.sym_id_map)
    sorted_tid = ctx.alloc_id()
    ctx.ir_tensors.append(
        IrTensor(
            tensor_id=sorted_tid,
            tensor_type=_torch_dtype_to_ir_type(val_dtype),
            ne=_pytorch_shape_to_ggml_ne(val_shape),
            op=OP_SORT,
            src_ids=[input_id],
            op_params=op_params,
            sym_dim_ids=_vsym_v,
            sym_dim_exprs=_vexprs_v,
        )
    )

    # --- Indices tensor (OP_SORT_INDICES) ---
    if isinstance(fake_vals, (list, tuple)) and len(fake_vals) >= 2:
        idx_fv = fake_vals[1]
    else:
        idx_fv = val_fv
    idx_shape = _resolve_shape(idx_fv)
    idx_dtype = getattr(idx_fv, "dtype", torch.int64) if idx_fv is not None else torch.int64

    _vsym_i, _vexprs_i = _sym_dim_info_ggml(idx_fv, ctx.sym_id_map)
    indices_tid = ctx.alloc_id()
    ctx.ir_tensors.append(
        IrTensor(
            tensor_id=indices_tid,
            tensor_type=_torch_dtype_to_ir_type(idx_dtype),
            ne=_pytorch_shape_to_ggml_ne(idx_shape),
            op=OP_SORT_INDICES,
            src_ids=[input_id],
            op_params=op_params,
            sym_dim_ids=_vsym_i,
            sym_dim_exprs=_vexprs_i,
        )
    )

    # Store as list for getitem resolution: [0]=sorted, [1]=indices
    ctx.node_to_id[node] = [sorted_tid, indices_tid]


# ---------------------------------------------------------------------------
# grouped_mm (expert-indexed matmul) -> MUL_MAT_ID
# ---------------------------------------------------------------------------

@register_op("transformers.grouped_mm_fallback.default")
def handle_grouped_mm(ctx, node, target_str):
    """grouped_mm_fallback(input, weight, offsets) -> expert matmul.

    Maps to OP_MUL_MAT_ID: matmul where rows are routed to different expert
    weight matrices based on the offsets tensor.
    """
    input_node = node.args[0]
    weight_node = node.args[1]
    offsets_node = node.args[2]

    input_id = ctx.node_to_id[input_node]
    weight_id = ctx.node_to_id[weight_node]
    offsets_id = ctx.node_to_id[offsets_node]

    fake_val = node.meta.get("val")
    shape = _resolve_shape(fake_val)
    out_dtype = (
        getattr(fake_val, "dtype", torch.float32)
        if fake_val is not None
        else torch.float32
    )

    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, ctx.sym_id_map)
    tid = ctx.alloc_id()
    ctx.ir_tensors.append(
        IrTensor(
            tensor_id=tid,
            tensor_type=_torch_dtype_to_ir_type(out_dtype),
            ne=_pytorch_shape_to_ggml_ne(shape),
            op=OP_MUL_MAT_ID,
            src_ids=[input_id, weight_id, offsets_id],
            sym_dim_ids=_vsym,
            sym_dim_exprs=_vexprs,
        )
    )
    ctx.node_to_id[node] = tid


# ---------------------------------------------------------------------------
# log1p — log(1 + x)
# ---------------------------------------------------------------------------

@register_op("aten.log1p.default")
def handle_log1p(ctx, node, target_str):
    """log1p(x) -> log(1 + x), element-wise unary op."""
    src_node = node.args[0]
    src_id = ctx.node_to_id[src_node]

    fake_val = node.meta.get("val")
    shape = _resolve_shape(fake_val)
    out_dtype = (
        getattr(fake_val, "dtype", torch.float32)
        if fake_val is not None
        else torch.float32
    )

    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, ctx.sym_id_map)
    tid = ctx.alloc_id()
    ctx.ir_tensors.append(
        IrTensor(
            tensor_id=tid,
            tensor_type=_torch_dtype_to_ir_type(out_dtype),
            ne=_pytorch_shape_to_ggml_ne(shape),
            op=OP_LOG1P,
            src_ids=[src_id],
            sym_dim_ids=_vsym,
            sym_dim_exprs=_vexprs,
        )
    )
    ctx.node_to_id[node] = tid


# ---------------------------------------------------------------------------
# Additional ops needed by MoE models
# ---------------------------------------------------------------------------

@register_op("aten.exp.default")
def handle_exp(ctx, node, target_str):
    """exp(x) — unary exponential."""
    src_id = ctx.node_to_id[node.args[0]]
    fake_val = node.meta.get("val")
    shape = _resolve_shape(fake_val)
    out_dtype = getattr(fake_val, "dtype", torch.float32) if fake_val is not None else torch.float32
    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, ctx.sym_id_map)
    tid = ctx.alloc_id()
    ctx.ir_tensors.append(
        IrTensor(
            tensor_id=tid,
            tensor_type=_torch_dtype_to_ir_type(out_dtype),
            ne=_pytorch_shape_to_ggml_ne(shape),
            op=OP_EXP,
            src_ids=[src_id],
            sym_dim_ids=_vsym,
            sym_dim_exprs=_vexprs,
        )
    )
    ctx.node_to_id[node] = tid


@register_op("aten.sum.dim_IntList")
def handle_sum(ctx, node, target_str):
    """sum(x, dim, keepdim) — reduction sum."""
    src_id = ctx.node_to_id[node.args[0]]
    dim = node.args[1] if len(node.args) > 1 else [-1]
    keepdim = node.args[2] if len(node.args) > 2 else False
    if isinstance(dim, (list, tuple)):
        dim = dim[0] if len(dim) == 1 else dim[0]  # TODO: multi-dim
    fake_val = node.meta.get("val")
    shape = _resolve_shape(fake_val)
    out_dtype = getattr(fake_val, "dtype", torch.float32) if fake_val is not None else torch.float32
    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, ctx.sym_id_map)
    # Get input ndim for ggml axis mapping
    src_fv = node.args[0].meta.get("val") if hasattr(node.args[0], "meta") else None
    ndim = len(src_fv.shape) if src_fv is not None else 4
    tid = ctx.alloc_id()
    ctx.ir_tensors.append(
        IrTensor(
            tensor_id=tid,
            tensor_type=_torch_dtype_to_ir_type(out_dtype),
            ne=_pytorch_shape_to_ggml_ne(shape),
            op=OP_SUM,
            src_ids=[src_id],
            op_params=struct.pack("<ii", dim, ndim),
            sym_dim_ids=_vsym,
            sym_dim_exprs=_vexprs,
        )
    )
    ctx.node_to_id[node] = tid


@register_op("aten.clamp.default")
def handle_clamp(ctx, node, target_str):
    """clamp(x, min, max)."""
    src_id = ctx.node_to_id[node.args[0]]
    min_val = float(node.args[1]) if len(node.args) > 1 and node.args[1] is not None else -3.4e38
    max_val = float(node.args[2]) if len(node.args) > 2 and node.args[2] is not None else 3.4e38
    fake_val = node.meta.get("val")
    shape = _resolve_shape(fake_val)
    out_dtype = getattr(fake_val, "dtype", torch.float32) if fake_val is not None else torch.float32
    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, ctx.sym_id_map)
    tid = ctx.alloc_id()
    ctx.ir_tensors.append(
        IrTensor(
            tensor_id=tid,
            tensor_type=_torch_dtype_to_ir_type(out_dtype),
            ne=_pytorch_shape_to_ggml_ne(shape),
            op=OP_CLAMP,
            src_ids=[src_id],
            op_params=struct.pack("<ff", min_val, max_val),
            sym_dim_ids=_vsym,
            sym_dim_exprs=_vexprs,
        )
    )
    ctx.node_to_id[node] = tid


@register_op("aten.copy.default")
def handle_copy(ctx, node, target_str):
    """copy(dst, src) — just pass through src (ggml handles in-place via mutable buffers)."""
    # In the functional graph, copy returns a new tensor with src's data.
    # For our IR, just alias to the src tensor.
    src_id = ctx.node_to_id[node.args[1]]  # args: (dst, src, non_blocking)
    ctx.node_to_id[node] = src_id


@register_op("aten.slice_scatter.default")
def handle_slice_scatter(ctx, node, target_str):
    """slice_scatter(dst, src, dim, start, end, step) — scatter src into dst slice."""
    dst_id = ctx.node_to_id[node.args[0]]
    src_id = ctx.node_to_id[node.args[1]]
    dim = node.args[2] if len(node.args) > 2 else 0
    start = node.args[3] if len(node.args) > 3 and node.args[3] is not None else 0
    end = node.args[4] if len(node.args) > 4 and node.args[4] is not None else 2**31 - 1
    step = node.args[5] if len(node.args) > 5 else 1
    if hasattr(start, '__index__'):
        start = start.__index__()
    if hasattr(end, '__index__'):
        end = end.__index__()
    fake_val = node.meta.get("val")
    shape = _resolve_shape(fake_val)
    out_dtype = getattr(fake_val, "dtype", torch.float32) if fake_val is not None else torch.float32
    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, ctx.sym_id_map)
    tid = ctx.alloc_id()
    ctx.ir_tensors.append(
        IrTensor(
            tensor_id=tid,
            tensor_type=_torch_dtype_to_ir_type(out_dtype),
            ne=_pytorch_shape_to_ggml_ne(shape),
            op=OP_SLICE_SCATTER,
            src_ids=[dst_id, src_id],
            op_params=struct.pack("<iiii", dim, int(start), int(end), int(step)),
            sym_dim_ids=_vsym,
            sym_dim_exprs=_vexprs,
        )
    )
    ctx.node_to_id[node] = tid


@register_op("aten.full_like.default")
def handle_full_like(ctx, node, target_str):
    """full_like(x, fill_value) — create constant tensor with same shape as x."""
    fill_value = float(node.args[1])
    fake_val = node.meta.get("val")
    shape = _resolve_shape(fake_val)
    out_dtype = getattr(fake_val, "dtype", torch.float32) if fake_val is not None else torch.float32
    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, ctx.sym_id_map)
    tid = ctx.alloc_id()
    ctx.ir_tensors.append(
        IrTensor(
            tensor_id=tid,
            tensor_type=_torch_dtype_to_ir_type(out_dtype),
            ne=_pytorch_shape_to_ggml_ne(shape),
            op=OP_FULL,
            src_ids=[],
            op_params=struct.pack("<f", fill_value),
            sym_dim_ids=_vsym,
            sym_dim_exprs=_vexprs,
            elem_size=4,  # F32 eager constant
        )
    )
    ctx.node_to_id[node] = tid
