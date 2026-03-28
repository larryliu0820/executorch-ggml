"""Shape operator handlers: view/reshape, unsqueeze, squeeze, transpose,
permute, slice, select, split_with_sizes_copy."""

import torch

from executorch_ggml.ops._registry import register_op
from executorch_ggml.ops._helpers import (
    _concrete_int,
    _resolve_shape,
    _pytorch_shape_to_ggml_ne,
    _torch_dtype_to_ir_type,
)
from executorch_ggml.ops._sym_expr import _sym_dim_info_ggml
from executorch_ggml.serialize import (
    IrTensor,
    TYPE_F32,
    OP_VIEW,
    OP_UNSQUEEZE,
    OP_PERMUTE,
    OP_TRANSPOSE,
    OP_SLICE,
    pack_view_params,
    pack_unsqueeze_params,
    pack_permute_params,
    pack_transpose_params,
    pack_slice_params,
)


# ---------------------------------------------------------------------------
# squeeze → VIEW (remove size-1 dims is just a reshape)
# ---------------------------------------------------------------------------

@register_op("aten.squeeze.dims", "aten.squeeze_copy.dims")
def handle_squeeze(ctx, node, target_str):
    """squeeze(x, dims) -- remove size-1 dims, equivalent to reshape/VIEW."""
    src_node = node.args[0]
    src_id = ctx.node_to_id[src_node]
    fake_val = node.meta.get("val")
    shape = _resolve_shape(fake_val)
    out_dtype = (
        getattr(fake_val, "dtype", torch.float32)
        if fake_val is not None
        else torch.float32
    )

    ggml_ne = _pytorch_shape_to_ggml_ne(shape)
    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, ctx.sym_id_map)
    tid = ctx.alloc_id()
    ctx.ir_tensors.append(
        IrTensor(
            tensor_id=tid,
            tensor_type=_torch_dtype_to_ir_type(out_dtype),
            ne=ggml_ne,
            op=OP_VIEW,
            src_ids=[src_id],
            op_params=pack_view_params(ggml_ne),
            sym_dim_ids=_vsym,
            sym_dim_exprs=_vexprs,
        )
    )
    ctx.node_to_id[node] = tid


# ---------------------------------------------------------------------------
# unsqueeze
# ---------------------------------------------------------------------------

@register_op("aten.unsqueeze.default", "aten.unsqueeze_copy.default")
def handle_unsqueeze(ctx, node, target_str):
    src_node = node.args[0]
    dim = int(node.args[1])
    src_id = ctx.node_to_id[src_node]
    src_val = src_node.meta.get("val")
    src_shape = _resolve_shape(src_val)
    src_ndim = len(src_shape) if len(src_shape) > 0 else 1
    if dim < 0:
        dim += src_ndim + 1
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
            op=OP_UNSQUEEZE,
            src_ids=[src_id],
            op_params=pack_unsqueeze_params(dim, src_ndim),
            sym_dim_ids=_vsym,
            sym_dim_exprs=_vexprs,
        )
    )
    ctx.node_to_id[node] = tid


# ---------------------------------------------------------------------------
# transpose
# ---------------------------------------------------------------------------

@register_op("aten.transpose.int")
def handle_transpose(ctx, node, target_str):
    src_node = node.args[0]
    dim0 = int(node.args[1])
    dim1 = int(node.args[2])
    src_id = ctx.node_to_id[src_node]
    fake_val = node.meta.get("val")
    shape = _resolve_shape(fake_val)
    out_dtype = (
        getattr(fake_val, "dtype", torch.float32)
        if fake_val is not None
        else torch.float32
    )
    ndim = len(shape)
    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, ctx.sym_id_map)
    tid = ctx.alloc_id()
    ctx.ir_tensors.append(
        IrTensor(
            tensor_id=tid,
            tensor_type=_torch_dtype_to_ir_type(out_dtype),
            ne=_pytorch_shape_to_ggml_ne(shape),
            op=OP_TRANSPOSE,
            src_ids=[src_id],
            op_params=pack_transpose_params(dim0, dim1, ndim),
            sym_dim_ids=_vsym,
            sym_dim_exprs=_vexprs,
        )
    )
    ctx.node_to_id[node] = tid


# ---------------------------------------------------------------------------
# slice
# ---------------------------------------------------------------------------

@register_op("aten.slice.Tensor", "aten.slice_copy.Tensor")
def handle_slice(ctx, node, target_str):
    """slice.Tensor(x, dim=0, start=None, end=None, step=1)."""
    src_node = node.args[0]
    dim = int(node.args[1]) if len(node.args) > 1 else 0
    start = node.args[2] if len(node.args) > 2 else None
    end = node.args[3] if len(node.args) > 3 else None
    step = int(node.args[4]) if len(node.args) > 4 else 1

    # Normalize optional start/end.
    # Use sentinel (2**62) for non-literal start/end values
    # that may depend on dynamic dimensions -- the C++ runtime
    # will derive the correct value from the resolved output shape.
    start_i = _concrete_int(start) if start is not None else 0
    if start is not None and not isinstance(start, int):
        start_i = 2**62  # sentinel: C++ will recompute
    # If end is None, represent as a large positive bound (runtime will clamp).
    # If end is a symbolic SymInt (depends on dynamic dim), also use
    # sentinel so the C++ runtime resolves from output shape.
    end_i = _concrete_int(end) if end is not None else (2**62)
    if end is not None and not isinstance(end, int):
        end_i = 2**62  # sentinel: C++ will recompute
    src_id = ctx.node_to_id[src_node]
    fake_val = node.meta.get("val")
    shape = _resolve_shape(fake_val)
    out_dtype = (
        getattr(fake_val, "dtype", torch.float32)
        if fake_val is not None
        else torch.float32
    )
    # Get the source tensor's PyTorch rank for correct axis computation.
    src_val = (
        src_node.meta.get("val")
        if isinstance(src_node, torch.fx.Node)
        else None
    )
    src_shape = _resolve_shape(src_val)
    ndim = len(src_shape) if src_shape else len(shape)

    # If end is the sentinel (end=None -> "slice to end") and
    # start is concrete, derive end from the output shape.
    # When start is a sentinel (symbolic), leave both for the
    # C++ runtime to resolve via centering or sym_dim_exprs.
    if end_i == 2**62 and start_i != 2**62 and shape:
        d = dim if dim >= 0 else ndim + dim
        if d < len(shape) and shape[d] is not None:
            end_i = start_i + shape[d] * step

    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, ctx.sym_id_map)
    tid = ctx.alloc_id()
    ctx.ir_tensors.append(
        IrTensor(
            tensor_id=tid,
            tensor_type=_torch_dtype_to_ir_type(out_dtype),
            ne=_pytorch_shape_to_ggml_ne(shape),
            op=OP_SLICE,
            src_ids=[src_id],
            op_params=pack_slice_params(
                dim, start_i, end_i, step, ndim
            ),
            sym_dim_ids=_vsym,
            sym_dim_exprs=_vexprs,
        )
    )
    ctx.node_to_id[node] = tid


# ---------------------------------------------------------------------------
# select.int -> SLICE + VIEW
# ---------------------------------------------------------------------------

@register_op("aten.select.int", "aten.select_copy.int")
def handle_select(ctx, node, target_str):
    """select(x, dim, index) -- picks one slice along dim and squeezes it.
    e.g. [1,1,1024].select(dim=1, index=-1) -> [1,1024]
    Lower as: SLICE(dim, index, index+1, step=1) then VIEW to output shape."""
    src_node = node.args[0]
    dim = int(node.args[1])
    idx = int(node.args[2])

    src_id = ctx.node_to_id[src_node]
    src_val = src_node.meta.get("val")
    src_shape = _resolve_shape(src_val)

    # Normalize negative index
    if idx < 0 and src_shape:
        idx = src_shape[dim] + idx

    fake_val = node.meta.get("val")
    out_shape = _resolve_shape(fake_val)
    out_dtype = (
        getattr(fake_val, "dtype", torch.float32)
        if fake_val is not None
        else torch.float32
    )

    # Intermediate sliced shape: same as src but dim shrunk to 1
    sliced_shape = list(src_shape)
    sliced_shape[dim] = 1

    _vsym_slice, _vexprs_slice = _sym_dim_info_ggml(src_val, ctx.sym_id_map)
    slice_id = ctx.alloc_id()
    ctx.ir_tensors.append(
        IrTensor(
            tensor_id=slice_id,
            tensor_type=_torch_dtype_to_ir_type(out_dtype),
            ne=_pytorch_shape_to_ggml_ne(sliced_shape),
            op=OP_SLICE,
            src_ids=[src_id],
            op_params=pack_slice_params(dim, idx, idx + 1, 1, ndim=len(src_shape)),
            sym_dim_ids=_vsym_slice,
            sym_dim_exprs=_vexprs_slice,
        )
    )

    # Squeeze the dim via VIEW to out_shape
    ggml_ne = _pytorch_shape_to_ggml_ne(out_shape)
    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, ctx.sym_id_map)
    tid = ctx.alloc_id()
    ctx.ir_tensors.append(
        IrTensor(
            tensor_id=tid,
            tensor_type=_torch_dtype_to_ir_type(out_dtype),
            ne=ggml_ne,
            op=OP_VIEW,
            src_ids=[slice_id],
            op_params=pack_view_params(ggml_ne),
            sym_dim_ids=_vsym,
            sym_dim_exprs=_vexprs,
        )
    )
    ctx.node_to_id[node] = tid


# ---------------------------------------------------------------------------
# view / reshape / _unsafe_view
# ---------------------------------------------------------------------------

@register_op(
    "aten.view.default",
    "aten.view_copy.default",
    "aten._unsafe_view.default",
    "aten.reshape.default",
)
def handle_view(ctx, node, target_str):
    """view(x, new_shape) or reshape(x, new_shape)."""
    src_node = node.args[0]
    src_id = ctx.node_to_id[src_node]

    fake_val = node.meta.get("val")
    shape = _resolve_shape(fake_val)
    # Use concrete output shape from FakeTensor meta rather than
    # node.args[1] which may contain SymInt from dynamic export.
    raw_shape = shape if shape else list(node.args[1])
    new_shape = [_concrete_int(d) if _concrete_int(d) is not None else d for d in raw_shape]

    # Pack the shape in ggml ne order (reversed from PyTorch)
    # since the C++ runtime passes these directly to ggml_reshape_4d.
    ggml_ne = _pytorch_shape_to_ggml_ne(new_shape)
    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, ctx.sym_id_map)

    tid = ctx.alloc_id()
    ctx.ir_tensors.append(
        IrTensor(
            tensor_id=tid,
            tensor_type=TYPE_F32,
            ne=ggml_ne,
            op=OP_VIEW,
            src_ids=[src_id],
            op_params=pack_view_params(ggml_ne),
            sym_dim_ids=_vsym,
            sym_dim_exprs=_vexprs,
        )
    )
    ctx.node_to_id[node] = tid


# ---------------------------------------------------------------------------
# permute / permute_copy
# ---------------------------------------------------------------------------

@register_op("aten.permute.default", "aten.permute_copy.default")
def handle_permute(ctx, node, target_str):
    """permute(x, dims).

    NOTE: ``aten.permute_copy`` is a permute with explicit copy
    semantics in PyTorch export. ggml doesn't need that distinction
    for our purposes, so we lower both ``permute`` and ``permute_copy``
    to the same IR op (``OP_PERMUTE``).
    """
    src_node = node.args[0]
    perm = list(node.args[1])
    src_id = ctx.node_to_id[src_node]

    fake_val = node.meta.get("val")
    shape = _resolve_shape(fake_val)

    # Convert PyTorch permutation to ggml permutation.
    # PyTorch axes are reversed relative to ggml: pytorch_axis = (ndim-1) - ggml_axis
    # For each ggml axis j, the source axis is:
    #   ggml_perm[j] = (ndim - 1) - perm[(ndim - 1) - j]
    ndim = len(perm)
    ggml_perm = [0, 1, 2, 3]
    for j in range(4):
        if j < ndim:
            pt_result_axis = (ndim - 1) - j
            pt_source_axis = perm[pt_result_axis]
            ggml_perm[j] = (ndim - 1) - pt_source_axis
        else:
            ggml_perm[j] = j

    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, ctx.sym_id_map)
    tid = ctx.alloc_id()
    ctx.ir_tensors.append(
        IrTensor(
            tensor_id=tid,
            tensor_type=TYPE_F32,
            ne=_pytorch_shape_to_ggml_ne(shape),
            op=OP_PERMUTE,
            src_ids=[src_id],
            op_params=pack_permute_params(ggml_perm),
            sym_dim_ids=_vsym,
            sym_dim_exprs=_vexprs,
        )
    )
    ctx.node_to_id[node] = tid


# ---------------------------------------------------------------------------
# split_with_sizes_copy -> multiple SLICEs
# ---------------------------------------------------------------------------

@register_op("aten.split_with_sizes_copy.default")
def handle_split_with_sizes_copy(ctx, node, target_str):
    """split_with_sizes_copy(x, split_sizes, dim=0) -> tuple of slices.
    Decompose into multiple SLICE ops."""
    src_node = node.args[0]
    split_sizes = list(node.args[1])
    dim = int(node.args[2]) if len(node.args) > 2 else 0
    src_id = ctx.node_to_id[src_node]

    src_val = src_node.meta.get("val")
    src_shape = _resolve_shape(src_val)
    ndim = len(src_shape) if src_shape else 4

    # Normalize negative dim
    if dim < 0:
        dim = ndim + dim

    # Emit one SLICE per chunk
    chunk_ids = []
    offset = 0
    fake_vals = node.meta.get("val")  # tuple of FakeTensors
    for i, sz in enumerate(split_sizes):
        start_i = offset
        end_i = offset + sz
        offset = end_i

        # Get shape from FakeTensor tuple
        if isinstance(fake_vals, (list, tuple)) and i < len(fake_vals):
            chunk_fv = fake_vals[i]
            chunk_shape = _resolve_shape(chunk_fv)
            chunk_dtype = getattr(chunk_fv, "dtype", torch.float32)
        else:
            chunk_fv = None
            chunk_shape = list(src_shape)
            chunk_shape[dim] = sz
            chunk_dtype = torch.float32

        _vsym, _vexprs = _sym_dim_info_ggml(chunk_fv, ctx.sym_id_map)
        chunk_tid = ctx.alloc_id()
        ctx.ir_tensors.append(
            IrTensor(
                tensor_id=chunk_tid,
                tensor_type=_torch_dtype_to_ir_type(chunk_dtype),
                ne=_pytorch_shape_to_ggml_ne(chunk_shape),
                op=OP_SLICE,
                src_ids=[src_id],
                op_params=pack_slice_params(
                    dim, start_i, end_i, 1, ndim
                ),
                sym_dim_ids=_vsym,
                sym_dim_exprs=_vexprs,
            )
        )
        chunk_ids.append(chunk_tid)

    # Store as list for getitem resolution
    ctx.node_to_id[node] = chunk_ids
