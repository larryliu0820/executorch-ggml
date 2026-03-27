"""Arithmetic operator handlers: add, sub, mul, div, neg, rsqrt, pow."""

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
    OP_ADD,
    OP_SUB,
    OP_MUL,
    OP_MUL_SCALAR,
    OP_NEG,
    OP_RSQRT,
    OP_DIV,
    OP_POW,
    OP_NONE,
    pack_float,
    pack_pow_params,
)


# ---------------------------------------------------------------------------
# Binary element-wise ops
# ---------------------------------------------------------------------------

@register_op("aten.mul.Tensor")
def handle_mul_tensor(ctx, node, target_str):
    """mul(a, b) -- element-wise multiply."""
    # NOTE: Broadcasting should be handled by BroadcastCanonicalizationPass
    # which inserts explicit expand_copy ops. This lowering expects inputs
    # to already have matching shapes.
    a_node, b_node = node.args[0], node.args[1]
    a_id = ctx.node_to_id[a_node]
    b_id = ctx.node_to_id[b_node]
    fake_val = node.meta.get("val")
    out_shape = _resolve_shape(fake_val)
    out_dtype = (
        getattr(fake_val, "dtype", torch.float32)
        if fake_val is not None
        else torch.float32
    )

    # Broadcasting is handled natively by the C++ ggml backend
    # (ggml_mul supports ggml_can_repeat(b, a)).
    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, ctx.sym_id_map)
    tid = ctx.alloc_id()
    ctx.ir_tensors.append(
        IrTensor(
            tensor_id=tid,
            tensor_type=_torch_dtype_to_ir_type(out_dtype),
            ne=_pytorch_shape_to_ggml_ne(out_shape),
            op=OP_MUL,
            src_ids=[a_id, b_id],
            sym_dim_ids=_vsym,
            sym_dim_exprs=_vexprs,
        )
    )
    ctx.node_to_id[node] = tid


@register_op("aten.div.Tensor_mode", "aten.div.Tensor")
def handle_div_tensor(ctx, node, target_str):
    """div(a, b) -- element-wise divide."""
    a_node, b_node = node.args[0], node.args[1]
    a_id = ctx.node_to_id[a_node]
    b_id = ctx.node_to_id[b_node]
    fake_val = node.meta.get("val")
    out_shape = _resolve_shape(fake_val)
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
            ne=_pytorch_shape_to_ggml_ne(out_shape),
            op=OP_DIV,
            src_ids=[a_id, b_id],
            sym_dim_ids=_vsym,
            sym_dim_exprs=_vexprs,
        )
    )
    ctx.node_to_id[node] = tid


@register_op("aten.add.Tensor")
def handle_add_tensor(ctx, node, target_str):
    """add(x, y, alpha=1) -- element-wise add."""
    # NOTE: Broadcasting should be handled by BroadcastCanonicalizationPass
    # which inserts explicit expand_copy ops. This lowering expects inputs
    # to already have matching shapes.
    x_node, y_node = node.args[0], node.args[1]
    # Edge graphs usually pass alpha as a kwarg.
    alpha = float(getattr(node, "kwargs", {}).get("alpha", 1))
    if alpha != 1.0:
        raise RuntimeError(
            "aten.add.Tensor with alpha != 1 not supported yet"
        )

    x_id = ctx.node_to_id[x_node]
    y_id = ctx.node_to_id[y_node]

    fake_val = node.meta.get("val")
    out_shape = _resolve_shape(fake_val)
    out_dtype = (
        getattr(fake_val, "dtype", torch.float32)
        if fake_val is not None
        else torch.float32
    )

    # Broadcasting is handled natively by the C++ ggml backend
    # (ggml_add supports broadcast via ggml_can_repeat).
    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, ctx.sym_id_map)
    tid = ctx.alloc_id()
    ctx.ir_tensors.append(
        IrTensor(
            tensor_id=tid,
            tensor_type=_torch_dtype_to_ir_type(out_dtype),
            ne=_pytorch_shape_to_ggml_ne(out_shape),
            op=OP_ADD,
            src_ids=[x_id, y_id],
            sym_dim_ids=_vsym,
            sym_dim_exprs=_vexprs,
        )
    )
    ctx.node_to_id[node] = tid


@register_op("aten.add.Scalar")
def handle_add_scalar(ctx, node, target_str):
    """add(x, scalar) -- create scalar constant tensor + ADD."""
    src_node = node.args[0]
    scalar_val = float(node.args[1])
    src_id = ctx.node_to_id[src_node]

    fake_val = node.meta.get("val")
    shape = _resolve_shape(fake_val)
    out_dtype = (
        getattr(fake_val, "dtype", torch.float32)
        if fake_val is not None
        else torch.float32
    )

    # Create a scalar constant tensor via NamedDataStore
    scalar_tid = ctx.alloc_id()
    scalar_const = torch.tensor(scalar_val, dtype=torch.float32).cpu()
    scalar_key = f"__const_add_s_{node.name}"
    ctx.data_store.add_named_data(scalar_key, scalar_const, alignment=64)
    ctx.ir_tensors.append(
        IrTensor(
            tensor_id=scalar_tid,
            tensor_type=TYPE_F32,
            ne=[1, 1, 1, 1],
            op=OP_NONE,
            data_key=scalar_key,
        )
    )

    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, ctx.sym_id_map)
    tid = ctx.alloc_id()
    ctx.ir_tensors.append(
        IrTensor(
            tensor_id=tid,
            tensor_type=_torch_dtype_to_ir_type(out_dtype),
            ne=_pytorch_shape_to_ggml_ne(shape),
            op=OP_ADD,
            src_ids=[src_id, scalar_tid],
            sym_dim_ids=_vsym,
            sym_dim_exprs=_vexprs,
        )
    )
    ctx.node_to_id[node] = tid


@register_op("aten.sub.Tensor")
def handle_sub_tensor(ctx, node, target_str):
    """sub(x, y, alpha=1) -- element-wise subtract."""
    # NOTE: Broadcasting should be handled by BroadcastCanonicalizationPass
    # which inserts explicit expand_copy ops. This lowering expects inputs
    # to already have matching shapes.
    x_node, y_node = node.args[0], node.args[1]
    x_id = ctx.node_to_id[x_node]
    y_id = ctx.node_to_id[y_node]

    fake_val = node.meta.get("val")
    out_shape = _resolve_shape(fake_val)
    out_dtype = (
        getattr(fake_val, "dtype", torch.float32)
        if fake_val is not None
        else torch.float32
    )

    # Broadcasting is handled natively by ggml_sub.
    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, ctx.sym_id_map)
    tid = ctx.alloc_id()
    ctx.ir_tensors.append(
        IrTensor(
            tensor_id=tid,
            tensor_type=_torch_dtype_to_ir_type(out_dtype),
            ne=_pytorch_shape_to_ggml_ne(out_shape),
            op=OP_SUB,
            src_ids=[x_id, y_id],
            sym_dim_ids=_vsym,
            sym_dim_exprs=_vexprs,
        )
    )
    ctx.node_to_id[node] = tid


@register_op("aten.mul.Scalar")
def handle_mul_scalar(ctx, node, target_str):
    """mul(x, scalar) -- element-wise multiply by scalar."""
    src_node = node.args[0]
    scalar = float(node.args[1])
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
            op=OP_MUL_SCALAR,
            src_ids=[src_id],
            op_params=pack_float(scalar),
            sym_dim_ids=_vsym,
            sym_dim_exprs=_vexprs,
        )
    )
    ctx.node_to_id[node] = tid


# ---------------------------------------------------------------------------
# Unary element-wise ops
# ---------------------------------------------------------------------------

@register_op("aten.neg.default")
def handle_neg(ctx, node, target_str):
    """neg(x) -- element-wise negate."""
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
            op=OP_NEG,
            src_ids=[src_id],
            sym_dim_ids=_vsym,
            sym_dim_exprs=_vexprs,
        )
    )
    ctx.node_to_id[node] = tid


@register_op("aten.rsqrt.default")
def handle_rsqrt(ctx, node, target_str):
    """rsqrt(x) -- element-wise reciprocal square root."""
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
            op=OP_RSQRT,
            src_ids=[src_id],
            sym_dim_ids=_vsym,
            sym_dim_exprs=_vexprs,
        )
    )
    ctx.node_to_id[node] = tid


@register_op("aten.pow.Tensor_Scalar")
def handle_pow(ctx, node, target_str):
    """pow(x, exponent) -- element-wise power."""
    src_node = node.args[0]
    exponent = float(node.args[1])
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
            op=OP_POW,
            src_ids=[src_id],
            op_params=pack_pow_params(exponent),
            sym_dim_ids=_vsym,
            sym_dim_exprs=_vexprs,
        )
    )
    ctx.node_to_id[node] = tid
