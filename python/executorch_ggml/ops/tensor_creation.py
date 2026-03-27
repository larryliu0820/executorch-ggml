"""Tensor creation and dtype conversion operator handlers: cat,
repeat_interleave, repeat, full_like, arange, full, cumsum, mean,
_to_copy, type_as, clone, to.dtype."""

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
    OP_NONE,
    OP_CAT,
    OP_REPEAT_INTERLEAVE,
    OP_REPEAT,
    OP_FULL,
    OP_ARANGE,
    OP_CUMSUM,
    OP_MEAN,
    OP_CAST,
    pack_cat_params,
    pack_repeat_interleave_params,
    pack_full_params,
    pack_arange_params,
    pack_cumsum_params,
    pack_mean_params,
    pack_cast_params,
)


# ---------------------------------------------------------------------------
# cat
# ---------------------------------------------------------------------------

@register_op("aten.cat.default")
def handle_cat(ctx, node, target_str):
    """cat(tensors, dim=0)."""
    tensors = list(node.args[0])
    dim = int(node.args[1]) if len(node.args) > 1 else 0
    src_ids = [ctx.node_to_id[t] for t in tensors]
    fake_val = node.meta.get("val")
    shape = _resolve_shape(fake_val)
    out_dtype = (
        getattr(fake_val, "dtype", torch.float32)
        if fake_val is not None
        else torch.float32
    )
    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, ctx.sym_id_map)
    # Normalize negative dim and compute ggml axis directly.
    # ggml axis = (rank - 1) - dim, where rank is the full PyTorch rank.
    ndim = len(shape)
    if dim < 0:
        dim = ndim + dim
    ggml_axis = (ndim - 1) - dim
    tid = ctx.alloc_id()
    ctx.ir_tensors.append(
        IrTensor(
            tensor_id=tid,
            tensor_type=_torch_dtype_to_ir_type(out_dtype),
            ne=_pytorch_shape_to_ggml_ne(shape),
            op=OP_CAT,
            src_ids=src_ids,
            op_params=pack_cat_params(ggml_axis),
            sym_dim_ids=_vsym,
            sym_dim_exprs=_vexprs,
        )
    )
    ctx.node_to_id[node] = tid


# ---------------------------------------------------------------------------
# repeat_interleave
# ---------------------------------------------------------------------------

@register_op("aten.repeat_interleave.self_int")
def handle_repeat_interleave(ctx, node, target_str):
    """repeat_interleave(input, repeats, dim=0)."""
    src_node = node.args[0]
    repeats = int(node.args[1])
    dim = (
        int(node.args[2])
        if len(node.args) > 2 and node.args[2] is not None
        else 0
    )
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
            op=OP_REPEAT_INTERLEAVE,
            src_ids=[src_id],
            op_params=pack_repeat_interleave_params(dim, repeats),
            sym_dim_ids=_vsym,
            sym_dim_exprs=_vexprs,
        )
    )
    ctx.node_to_id[node] = tid


# ---------------------------------------------------------------------------
# repeat
# ---------------------------------------------------------------------------

@register_op("aten.repeat.default")
def handle_repeat(ctx, node, target_str):
    """repeat(input, repeats) - tile input by repeats along each dim."""
    src_node = node.args[0]
    src_id = ctx.node_to_id[src_node]
    fake_val = node.meta.get("val")
    out_shape = _resolve_shape(fake_val)
    out_dtype = (
        getattr(fake_val, "dtype", torch.float32)
        if fake_val is not None
        else torch.float32
    )
    dst_ne = _pytorch_shape_to_ggml_ne(out_shape)
    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, ctx.sym_id_map)

    # Create a shape-only "like" tensor for ggml_repeat.
    like_id = ctx.alloc_id()
    ctx.ir_tensors.append(
        IrTensor(
            tensor_id=like_id,
            tensor_type=_torch_dtype_to_ir_type(out_dtype),
            ne=dst_ne,
            op=OP_NONE,
            is_input=False,
            sym_dim_ids=_vsym,
            sym_dim_exprs=_vexprs,
        )
    )

    tid = ctx.alloc_id()
    ctx.ir_tensors.append(
        IrTensor(
            tensor_id=tid,
            tensor_type=_torch_dtype_to_ir_type(out_dtype),
            ne=dst_ne,
            op=OP_REPEAT,
            src_ids=[src_id, like_id],
            sym_dim_ids=_vsym,
            sym_dim_exprs=_vexprs,
        )
    )
    ctx.node_to_id[node] = tid


# ---------------------------------------------------------------------------
# full_like
# ---------------------------------------------------------------------------

@register_op("aten.full_like.default")
def handle_full_like(ctx, node, target_str):
    """full_like(input, fill_value, ...) - create tensor like input filled
    with fill_value.  Typically used to create constant tensors (e.g., -inf
    for masking)."""
    fill_value = float(node.args[1])

    fake_val = node.meta.get("val")
    shape = _resolve_shape(fake_val)
    out_dtype = (
        getattr(fake_val, "dtype", torch.float32)
        if fake_val is not None
        else torch.float32
    )

    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, ctx.sym_id_map)
    has_dynamic = _vsym is not None and any(s != -1 for s in _vsym)

    if has_dynamic:
        # Dynamic shape -- emit OP_FULL so C++ fills at runtime
        ir_type = _torch_dtype_to_ir_type(out_dtype)
        tid = ctx.alloc_id()
        ctx.ir_tensors.append(
            IrTensor(
                tensor_id=tid,
                tensor_type=ir_type,
                ne=_pytorch_shape_to_ggml_ne(shape),
                op=OP_FULL,
                src_ids=[],
                op_params=pack_full_params(fill_value),
                sym_dim_ids=_vsym,
                sym_dim_exprs=_vexprs,
                elem_size=0,
            )
        )
    else:
        # Static shape -- keep existing constant path (OP_NONE + baked data)
        import hashlib
        import numpy as np

        numel = 1
        for d in shape:
            numel *= d
        const_data = np.full(numel, fill_value, dtype=np.float32)
        const_key = f"_full_like_{hashlib.sha256(const_data.tobytes()).hexdigest()[:16]}"

        ctx.data_store.add_named_data(
            const_key, const_data.tobytes(), alignment=64
        )

        tid = ctx.alloc_id()
        ctx.ir_tensors.append(
            IrTensor(
                tensor_id=tid,
                tensor_type=_torch_dtype_to_ir_type(out_dtype),
                ne=_pytorch_shape_to_ggml_ne(shape),
                op=OP_NONE,  # Constant
                data_key=const_key,
            )
        )
    ctx.node_to_id[node] = tid


# ---------------------------------------------------------------------------
# arange
# NOTE: aten.arange.start_step MUST be registered before aten.arange.default
# because "aten.arange.default" is a substring that would also match
# target strings containing "arange.start_step" in an if/elif chain.
# With the registry's first-match-wins semantics, order of registration
# within this file determines priority.
# ---------------------------------------------------------------------------

@register_op("aten.arange.start_step")
def handle_arange_start_step(ctx, node, target_str):
    """arange(start, end, step, ...) - generates [start, start+step, ...]."""
    start = float(node.args[0]) if len(node.args) > 0 else 0.0
    # end = node.args[1]  # We use output shape instead
    step = float(node.args[2]) if len(node.args) > 2 else 1.0

    fake_val = node.meta.get("val")
    shape = _resolve_shape(fake_val)
    out_dtype = (
        getattr(fake_val, "dtype", torch.int64)
        if fake_val is not None
        else torch.int64
    )

    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, ctx.sym_id_map)
    ir_type = _torch_dtype_to_ir_type(out_dtype)
    tid = ctx.alloc_id()
    ctx.ir_tensors.append(
        IrTensor(
            tensor_id=tid,
            tensor_type=ir_type,
            ne=_pytorch_shape_to_ggml_ne(shape),
            op=OP_ARANGE,
            src_ids=[],
            op_params=pack_arange_params(start, step),
            sym_dim_ids=_vsym,
            sym_dim_exprs=_vexprs,
            elem_size=0,
        )
    )
    ctx.node_to_id[node] = tid


@register_op("aten.arange.default")
def handle_arange_default(ctx, node, target_str):
    """arange(end, ...) - generates [0, 1, ..., end-1]."""
    start = 0.0
    step = 1.0

    fake_val = node.meta.get("val")
    shape = _resolve_shape(fake_val)
    out_dtype = (
        getattr(fake_val, "dtype", torch.int64)
        if fake_val is not None
        else torch.int64
    )

    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, ctx.sym_id_map)
    ir_type = _torch_dtype_to_ir_type(out_dtype)
    tid = ctx.alloc_id()
    ctx.ir_tensors.append(
        IrTensor(
            tensor_id=tid,
            tensor_type=ir_type,
            ne=_pytorch_shape_to_ggml_ne(shape),
            op=OP_ARANGE,
            src_ids=[],
            op_params=pack_arange_params(start, step),
            sym_dim_ids=_vsym,
            sym_dim_exprs=_vexprs,
            elem_size=0,
        )
    )
    ctx.node_to_id[node] = tid


# ---------------------------------------------------------------------------
# full
# ---------------------------------------------------------------------------

@register_op("aten.full.default")
def handle_full(ctx, node, target_str):
    """full(size, fill_value, ...) - creates tensor filled with fill_value."""
    fill_value = float(node.args[1]) if len(node.args) > 1 else 0.0

    fake_val = node.meta.get("val")
    shape = _resolve_shape(fake_val)
    out_dtype = (
        getattr(fake_val, "dtype", torch.float32)
        if fake_val is not None
        else torch.float32
    )

    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, ctx.sym_id_map)
    ir_type = _torch_dtype_to_ir_type(out_dtype)
    tid = ctx.alloc_id()
    ctx.ir_tensors.append(
        IrTensor(
            tensor_id=tid,
            tensor_type=ir_type,
            ne=_pytorch_shape_to_ggml_ne(shape),
            op=OP_FULL,
            src_ids=[],
            op_params=pack_full_params(fill_value),
            sym_dim_ids=_vsym,
            sym_dim_exprs=_vexprs,
            elem_size=0,
        )
    )
    ctx.node_to_id[node] = tid


# ---------------------------------------------------------------------------
# cumsum
# ---------------------------------------------------------------------------

@register_op("aten.cumsum.default")
def handle_cumsum(ctx, node, target_str):
    """cumsum(x, dim) - cumulative sum along dimension."""
    src_node = node.args[0]
    dim = int(node.args[1]) if len(node.args) > 1 else 0
    src_id = ctx.node_to_id[src_node]

    fake_val = node.meta.get("val")
    shape = _resolve_shape(fake_val)
    out_dtype = (
        getattr(fake_val, "dtype", torch.int64)
        if fake_val is not None
        else torch.int64
    )
    ndim = len(shape)
    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, ctx.sym_id_map)

    ir_type = _torch_dtype_to_ir_type(out_dtype)
    tid = ctx.alloc_id()
    ctx.ir_tensors.append(
        IrTensor(
            tensor_id=tid,
            tensor_type=ir_type,
            ne=_pytorch_shape_to_ggml_ne(shape),
            op=OP_CUMSUM,
            src_ids=[src_id],
            op_params=pack_cumsum_params(dim, ndim),
            sym_dim_ids=_vsym,
            sym_dim_exprs=_vexprs,
            elem_size=0,
        )
    )
    ctx.node_to_id[node] = tid


# ---------------------------------------------------------------------------
# mean
# ---------------------------------------------------------------------------

@register_op("aten.mean.dim", "aten._mean_dim.default")
def handle_mean_dim(ctx, node, target_str):
    """mean(x, dim, keepdim)."""
    src_node = node.args[0]
    dim = node.args[1] if len(node.args) > 1 else -1
    # keepdim = bool(node.args[2]) if len(node.args) > 2 else False
    # MV2 global avg pool is exported as mean over (H, W) with keepdim=True.
    # We keep the semantics by returning a pooled tensor with singleton
    # spatial dims (H=W=1). If other keepdim=True patterns show up later,
    # we can extend the lowering.

    # ExecuTorch/Edge exports `mean.dim` dims as a tuple/list.
    # For MV2 we need global avg pool which is mean over (H, W) = dims (2, 3).
    if isinstance(dim, (list, tuple)):
        dims = [_concrete_int(d) for d in list(dim)]
    else:
        dims = [_concrete_int(dim)]

    # Canonicalize dims to positive indices when possible.
    # For NCHW 4D tensors, -1/-2 correspond to W/H.
    dims = [d + 4 if d < 0 else d for d in dims]
    src_id = ctx.node_to_id[src_node]

    fake_val = node.meta.get("val")
    shape = _resolve_shape(fake_val)

    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, ctx.sym_id_map)
    tid = ctx.alloc_id()
    ctx.ir_tensors.append(
        IrTensor(
            tensor_id=tid,
            tensor_type=TYPE_F32,
            ne=_pytorch_shape_to_ggml_ne(shape),
            op=OP_MEAN,
            src_ids=[src_id],
            op_params=pack_mean_params(dims),
            sym_dim_ids=_vsym,
            sym_dim_exprs=_vexprs,
        )
    )
    ctx.node_to_id[node] = tid


# ---------------------------------------------------------------------------
# type_as
# ---------------------------------------------------------------------------

@register_op("aten.type_as.default")
def handle_type_as(ctx, node, target_str):
    """type_as(input, other) casts input to other's dtype."""
    src_node = node.args[0]
    other_node = node.args[1]
    src_id = ctx.node_to_id[src_node]

    src_val = src_node.meta.get("val") if hasattr(src_node, "meta") else None
    other_val = other_node.meta.get("val") if hasattr(other_node, "meta") else None
    src_dtype = getattr(src_val, "dtype", torch.float32) if src_val is not None else torch.float32
    target_dtype = getattr(other_val, "dtype", torch.float32) if other_val is not None else torch.float32

    if src_dtype == target_dtype:
        # No cast needed - use identity
        ctx.node_to_id[node] = src_id
    else:
        # Need to cast
        fake_val = node.meta.get("val")
        shape = _resolve_shape(fake_val)
        _vsym, _vexprs = _sym_dim_info_ggml(fake_val, ctx.sym_id_map)
        tid = ctx.alloc_id()
        ctx.ir_tensors.append(
            IrTensor(
                tensor_id=tid,
                tensor_type=_torch_dtype_to_ir_type(target_dtype),
                ne=_pytorch_shape_to_ggml_ne(shape),
                op=OP_CAST,
                src_ids=[src_id],
                op_params=pack_cast_params(_torch_dtype_to_ir_type(target_dtype)),
                sym_dim_ids=_vsym,
                sym_dim_exprs=_vexprs,
            )
        )
        ctx.node_to_id[node] = tid


# ---------------------------------------------------------------------------
# clone (identity / no-op for ggml)
# ---------------------------------------------------------------------------

@register_op("aten.clone.default")
def handle_clone(ctx, node, target_str):
    """clone(x) - treat as identity for ggml (no explicit copy needed)."""
    src_node = node.args[0]
    ctx.node_to_id[node] = ctx.node_to_id[src_node]


# ---------------------------------------------------------------------------
# _to_copy (dtype conversion)
# ---------------------------------------------------------------------------

@register_op("aten._to_copy.default")
def handle_to_copy(ctx, node, target_str):
    """_to_copy(x, dtype=...) - dtype conversion with copy.

    When source and target dtypes match, this is lowered as a no-op
    (identity).  Otherwise emits an OP_CAST."""
    src_node = node.args[0]
    if src_node not in ctx.node_to_id:
        return  # source from auto_functionalized handled externally
    src_id = ctx.node_to_id[src_node]

    src_val = src_node.meta.get("val") if hasattr(src_node, "meta") else None
    src_dtype = getattr(src_val, "dtype", torch.float32) if src_val is not None else torch.float32

    fake_val = node.meta.get("val")
    out_dtype = fake_val.dtype if fake_val is not None else torch.float32
    out_shape = _resolve_shape(fake_val)

    if src_dtype == out_dtype:
        # No cast needed - use identity
        ctx.node_to_id[node] = src_id
    else:
        # Need to cast
        _vsym, _vexprs = _sym_dim_info_ggml(fake_val, ctx.sym_id_map)
        tid = ctx.alloc_id()
        ctx.ir_tensors.append(
            IrTensor(
                tensor_id=tid,
                tensor_type=_torch_dtype_to_ir_type(out_dtype),
                ne=_pytorch_shape_to_ggml_ne(out_shape),
                op=OP_CAST,
                src_ids=[src_id],
                op_params=pack_cast_params(_torch_dtype_to_ir_type(out_dtype)),
                sym_dim_ids=_vsym,
                sym_dim_exprs=_vexprs,
            )
        )
        ctx.node_to_id[node] = tid


# ---------------------------------------------------------------------------
# to.dtype / to.dtype_layout
# ---------------------------------------------------------------------------

@register_op("aten.to.dtype", "aten.to.dtype_layout")
def handle_to_dtype(ctx, node, target_str):
    """Type casting via to.dtype or to.dtype_layout - use CAST op."""
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
            op=OP_CAST,
            src_ids=[src_id],
            op_params=pack_cast_params(_torch_dtype_to_ir_type(out_dtype)),
            sym_dim_ids=_vsym,
            sym_dim_exprs=_vexprs,
        )
    )
    ctx.node_to_id[node] = tid


# ---------------------------------------------------------------------------
# expand / expand_copy
# ---------------------------------------------------------------------------

@register_op("aten.expand.default", "aten.expand_copy.default")
def handle_expand(ctx, node, target_str):
    """expand(x, size) - broadcast via OP_REPEAT if shapes differ."""
    src_node = node.args[0]
    src_id = ctx.node_to_id[src_node]
    fake_val = node.meta.get("val")
    shape = _resolve_shape(fake_val)
    out_dtype = (
        getattr(fake_val, "dtype", torch.float32)
        if fake_val is not None
        else torch.float32
    )

    src_fake = (
        src_node.meta.get("val") if hasattr(src_node, "meta") else None
    )
    src_shape = _resolve_shape(src_fake)

    src_ne = _pytorch_shape_to_ggml_ne(src_shape)
    dst_ne = _pytorch_shape_to_ggml_ne(shape)
    can_repeat = all(
        src_ne[i] == 1 or src_ne[i] == dst_ne[i] for i in range(4)
    )

    if can_repeat and src_ne != dst_ne:
        _vsym, _vexprs = _sym_dim_info_ggml(fake_val, ctx.sym_id_map)
        like_id = ctx.alloc_id()
        ctx.ir_tensors.append(
            IrTensor(
                tensor_id=like_id,
                tensor_type=_torch_dtype_to_ir_type(out_dtype),
                ne=dst_ne,
                op=OP_NONE,
                is_input=False,
                sym_dim_ids=_vsym,
                sym_dim_exprs=_vexprs,
            )
        )

        tid = ctx.alloc_id()
        ctx.ir_tensors.append(
            IrTensor(
                tensor_id=tid,
                tensor_type=_torch_dtype_to_ir_type(out_dtype),
                ne=dst_ne,
                op=OP_REPEAT,
                src_ids=[src_id, like_id],
                sym_dim_ids=_vsym,
                sym_dim_exprs=_vexprs,
            )
        )
        ctx.node_to_id[node] = tid
    else:
        ctx.node_to_id[node] = src_id
