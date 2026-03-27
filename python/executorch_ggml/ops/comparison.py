"""Comparison and logical operator handlers: eq, ne, le, lt, gt, ge,
bitwise_and, logical_and, bitwise_or, logical_not, bitwise_not, any,
where, argmax."""

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
    TYPE_BOOL,
    TYPE_I32,
    OP_EQ,
    OP_NE,
    OP_LE,
    OP_LT,
    OP_GT,
    OP_GE,
    OP_BITWISE_AND,
    OP_BITWISE_OR,
    OP_LOGICAL_NOT,
    OP_ANY,
    OP_WHERE,
    OP_ARGMAX,
    OP_REMAINDER,
    pack_comparison_params,
    pack_any_params,
    pack_argmax_params,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _handle_comparison_scalar(ctx, node, op_code):
    """Shared logic for scalar comparison ops: eq.Scalar, ne.Scalar, le.Scalar."""
    src_node = node.args[0]
    scalar = float(node.args[1])
    src_id = ctx.node_to_id[src_node]

    fake_val = node.meta.get("val")
    shape = _resolve_shape(fake_val)
    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, ctx.sym_id_map)

    tid = ctx.alloc_id()
    ctx.ir_tensors.append(
        IrTensor(
            tensor_id=tid,
            tensor_type=TYPE_BOOL,
            ne=_pytorch_shape_to_ggml_ne(shape),
            op=op_code,
            src_ids=[src_id],
            op_params=pack_comparison_params(scalar, is_scalar=True),
            sym_dim_ids=_vsym,
            sym_dim_exprs=_vexprs,
            elem_size=0,
        )
    )
    ctx.node_to_id[node] = tid


def _handle_comparison_tensor(ctx, node, op_code):
    """Shared logic for tensor-vs-tensor comparison ops: eq.Tensor, le.Tensor,
    lt.Tensor, gt.Tensor, ge.Tensor."""
    a_node, b_node = node.args[0], node.args[1]
    a_id = ctx.node_to_id[a_node]
    b_id = ctx.node_to_id[b_node]

    fake_val = node.meta.get("val")
    shape = _resolve_shape(fake_val)
    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, ctx.sym_id_map)

    tid = ctx.alloc_id()
    ctx.ir_tensors.append(
        IrTensor(
            tensor_id=tid,
            tensor_type=TYPE_BOOL,
            ne=_pytorch_shape_to_ggml_ne(shape),
            op=op_code,
            src_ids=[a_id, b_id],
            sym_dim_ids=_vsym,
            sym_dim_exprs=_vexprs,
        )
    )
    ctx.node_to_id[node] = tid


def _handle_binary_bitwise(ctx, node, op_code):
    """Shared logic for binary bitwise/logical ops that infer output dtype
    from the FakeTensor metadata (bitwise_and, bitwise_or)."""
    a_node, b_node = node.args[0], node.args[1]
    a_id = ctx.node_to_id[a_node]
    b_id = ctx.node_to_id[b_node]

    fake_val = node.meta.get("val")
    shape = _resolve_shape(fake_val)
    out_dtype = (
        getattr(fake_val, "dtype", torch.bool)
        if fake_val is not None
        else torch.bool
    )
    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, ctx.sym_id_map)

    tid = ctx.alloc_id()
    ctx.ir_tensors.append(
        IrTensor(
            tensor_id=tid,
            tensor_type=_torch_dtype_to_ir_type(out_dtype),
            ne=_pytorch_shape_to_ggml_ne(shape),
            op=op_code,
            src_ids=[a_id, b_id],
            sym_dim_ids=_vsym,
            sym_dim_exprs=_vexprs,
        )
    )
    ctx.node_to_id[node] = tid


def _handle_unary_logical(ctx, node, op_code):
    """Shared logic for unary logical ops (logical_not, bitwise_not)."""
    src_node = node.args[0]
    src_id = ctx.node_to_id[src_node]

    fake_val = node.meta.get("val")
    shape = _resolve_shape(fake_val)
    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, ctx.sym_id_map)

    tid = ctx.alloc_id()
    ctx.ir_tensors.append(
        IrTensor(
            tensor_id=tid,
            tensor_type=TYPE_BOOL,
            ne=_pytorch_shape_to_ggml_ne(shape),
            op=op_code,
            src_ids=[src_id],
            sym_dim_ids=_vsym,
            sym_dim_exprs=_vexprs,
        )
    )
    ctx.node_to_id[node] = tid


# ---------------------------------------------------------------------------
# Comparison: scalar variants
# ---------------------------------------------------------------------------

@register_op("aten.eq.Scalar")
def handle_eq_scalar(ctx, node, target_str):
    """eq(x, scalar) - element-wise equality with scalar."""
    _handle_comparison_scalar(ctx, node, OP_EQ)


@register_op("aten.ne.Scalar")
def handle_ne_scalar(ctx, node, target_str):
    """ne(x, scalar) - element-wise not-equal with scalar."""
    _handle_comparison_scalar(ctx, node, OP_NE)


@register_op("aten.le.Scalar")
def handle_le_scalar(ctx, node, target_str):
    """le(x, scalar) - element-wise less-than-or-equal with scalar."""
    _handle_comparison_scalar(ctx, node, OP_LE)


@register_op("aten.ge.Scalar")
def handle_ge_scalar(ctx, node, target_str):
    """ge(x, scalar) - element-wise greater-than-or-equal with scalar."""
    _handle_comparison_scalar(ctx, node, OP_GE)


@register_op("aten.lt.Scalar")
def handle_lt_scalar(ctx, node, target_str):
    """lt(x, scalar) - element-wise less-than with scalar."""
    _handle_comparison_scalar(ctx, node, OP_LT)


@register_op("aten.gt.Scalar")
def handle_gt_scalar(ctx, node, target_str):
    """gt(x, scalar) - element-wise greater-than with scalar."""
    _handle_comparison_scalar(ctx, node, OP_GT)


@register_op("aten.remainder.Scalar")
def handle_remainder_scalar(ctx, node, target_str):
    """remainder(x, scalar) - element-wise remainder (Python-style modulo)."""
    src_node = node.args[0]
    scalar = float(node.args[1])
    src_id = ctx.node_to_id[src_node]

    fake_val = node.meta.get("val")
    shape = _resolve_shape(fake_val)
    out_dtype = (
        getattr(fake_val, "dtype", torch.int64)
        if fake_val is not None
        else torch.int64
    )
    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, ctx.sym_id_map)

    tid = ctx.alloc_id()
    ctx.ir_tensors.append(
        IrTensor(
            tensor_id=tid,
            tensor_type=_torch_dtype_to_ir_type(out_dtype),
            ne=_pytorch_shape_to_ggml_ne(shape),
            op=OP_REMAINDER,
            src_ids=[src_id],
            op_params=pack_comparison_params(scalar, is_scalar=True),
            sym_dim_ids=_vsym,
            sym_dim_exprs=_vexprs,
        )
    )
    ctx.node_to_id[node] = tid


# ---------------------------------------------------------------------------
# Comparison: tensor variants
# ---------------------------------------------------------------------------

@register_op("aten.eq.Tensor")
def handle_eq_tensor(ctx, node, target_str):
    """eq(x, y) - element-wise equality."""
    a_node, b_node = node.args[0], node.args[1]
    a_id = ctx.node_to_id[a_node]
    b_id = ctx.node_to_id[b_node]

    fake_val = node.meta.get("val")
    shape = _resolve_shape(fake_val)
    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, ctx.sym_id_map)

    tid = ctx.alloc_id()
    ctx.ir_tensors.append(
        IrTensor(
            tensor_id=tid,
            tensor_type=TYPE_BOOL,
            ne=_pytorch_shape_to_ggml_ne(shape),
            op=OP_EQ,
            src_ids=[a_id, b_id],
            op_params=pack_comparison_params(0.0, is_scalar=False),
            sym_dim_ids=_vsym,
            sym_dim_exprs=_vexprs,
        )
    )
    ctx.node_to_id[node] = tid


@register_op("aten.le.Tensor")
def handle_le_tensor(ctx, node, target_str):
    """le(x, y) - element-wise less-than-or-equal."""
    _handle_comparison_tensor(ctx, node, OP_LE)


@register_op("aten.lt.Tensor")
def handle_lt_tensor(ctx, node, target_str):
    """lt(x, y) - element-wise less-than."""
    _handle_comparison_tensor(ctx, node, OP_LT)


@register_op("aten.gt.Tensor")
def handle_gt_tensor(ctx, node, target_str):
    """gt(x, y) - element-wise greater-than."""
    _handle_comparison_tensor(ctx, node, OP_GT)


@register_op("aten.ge.Tensor")
def handle_ge_tensor(ctx, node, target_str):
    """ge(x, y) - element-wise greater-than-or-equal."""
    _handle_comparison_tensor(ctx, node, OP_GE)


# ---------------------------------------------------------------------------
# Bitwise / logical ops
# ---------------------------------------------------------------------------

@register_op("aten.bitwise_and.Tensor")
def handle_bitwise_and(ctx, node, target_str):
    """bitwise_and(x, y)."""
    _handle_binary_bitwise(ctx, node, OP_BITWISE_AND)


@register_op("aten.logical_and.default")
def handle_logical_and(ctx, node, target_str):
    """logical_and(x, y) - identical to bitwise_and for bool tensors."""
    a_node, b_node = node.args[0], node.args[1]
    a_id = ctx.node_to_id[a_node]
    b_id = ctx.node_to_id[b_node]

    fake_val = node.meta.get("val")
    shape = _resolve_shape(fake_val)
    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, ctx.sym_id_map)

    tid = ctx.alloc_id()
    ctx.ir_tensors.append(
        IrTensor(
            tensor_id=tid,
            tensor_type=TYPE_BOOL,
            ne=_pytorch_shape_to_ggml_ne(shape),
            op=OP_BITWISE_AND,
            src_ids=[a_id, b_id],
            sym_dim_ids=_vsym,
            sym_dim_exprs=_vexprs,
        )
    )
    ctx.node_to_id[node] = tid


@register_op("aten.bitwise_or.Tensor")
def handle_bitwise_or(ctx, node, target_str):
    """bitwise_or(x, y)."""
    _handle_binary_bitwise(ctx, node, OP_BITWISE_OR)


@register_op("aten.logical_not.default")
def handle_logical_not(ctx, node, target_str):
    """logical_not(x)."""
    _handle_unary_logical(ctx, node, OP_LOGICAL_NOT)


@register_op("aten.bitwise_not.default")
def handle_bitwise_not(ctx, node, target_str):
    """bitwise_not(x) - identical to logical_not for bool tensors."""
    _handle_unary_logical(ctx, node, OP_LOGICAL_NOT)


# ---------------------------------------------------------------------------
# Reduction: any
# ---------------------------------------------------------------------------

@register_op("aten.any.dim")
def handle_any_dim(ctx, node, target_str):
    """any(x, dim, keepdim)."""
    src_node = node.args[0]
    dim = int(node.args[1]) if len(node.args) > 1 else 0
    src_id = ctx.node_to_id[src_node]

    fake_val = node.meta.get("val")
    shape = _resolve_shape(fake_val)
    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, ctx.sym_id_map)

    src_val = (
        src_node.meta.get("val") if hasattr(src_node, "meta") else None
    )
    src_shape = _resolve_shape(src_val) or shape
    ndim = len(src_shape)

    tid = ctx.alloc_id()
    ctx.ir_tensors.append(
        IrTensor(
            tensor_id=tid,
            tensor_type=TYPE_BOOL,
            ne=_pytorch_shape_to_ggml_ne(shape),
            op=OP_ANY,
            src_ids=[src_id],
            op_params=pack_any_params(dim, ndim),
            sym_dim_ids=_vsym,
            sym_dim_exprs=_vexprs,
            elem_size=4,  # eager F32 reduction tensor
        )
    )
    ctx.node_to_id[node] = tid


# ---------------------------------------------------------------------------
# Conditional: where
# ---------------------------------------------------------------------------

@register_op("aten.where.self")
def handle_where(ctx, node, target_str):
    """where(condition, x, y)."""
    cond_node, x_node, y_node = node.args[0], node.args[1], node.args[2]
    cond_id = ctx.node_to_id[cond_node]
    x_id = ctx.node_to_id[x_node]
    y_id = ctx.node_to_id[y_node]

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
            op=OP_WHERE,
            src_ids=[cond_id, x_id, y_id],
            sym_dim_ids=_vsym,
            sym_dim_exprs=_vexprs,
        )
    )
    ctx.node_to_id[node] = tid


# ---------------------------------------------------------------------------
# Reduction: argmax
# ---------------------------------------------------------------------------

@register_op("aten.argmax.default")
def handle_argmax(ctx, node, target_str):
    """argmax(x, dim=None, keepdim=False)."""
    src_node = node.args[0]
    dim = int(node.args[1]) if len(node.args) > 1 and node.args[1] is not None else -1
    src_id = ctx.node_to_id[src_node]

    fake_val = node.meta.get("val")
    shape = _resolve_shape(fake_val)

    src_val = src_node.meta.get("val")
    src_shape = _resolve_shape(src_val)
    ndim = len(src_shape) if src_shape else len(shape) + 1

    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, ctx.sym_id_map)
    tid = ctx.alloc_id()
    ctx.ir_tensors.append(
        IrTensor(
            tensor_id=tid,
            tensor_type=TYPE_I32,  # ggml_argmax returns I32
            ne=_pytorch_shape_to_ggml_ne(shape),
            op=OP_ARGMAX,
            src_ids=[src_id],
            op_params=pack_argmax_params(dim, ndim),
            sym_dim_ids=_vsym,
            sym_dim_exprs=_vexprs,
        )
    )
    ctx.node_to_id[node] = tid
