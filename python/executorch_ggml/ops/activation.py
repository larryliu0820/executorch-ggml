"""Activation operator handlers: silu, relu, tanh, gelu, leaky_relu,
sigmoid, softmax, hardtanh (clamp), sin, cos."""

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
    OP_SILU,
    OP_RELU,
    OP_TANH,
    OP_GELU,
    OP_LEAKY_RELU,
    OP_SIGMOID,
    OP_SOFTMAX,
    OP_HARDTANH,
    OP_SIN,
    OP_COS,
    pack_float,
    pack_hardtanh_params,
    pack_softmax_params,
)


# ---------------------------------------------------------------------------
# Simple unary activations (same pattern: single src, dtype-preserving)
# ---------------------------------------------------------------------------

def _handle_simple_unary(ctx, node, op_code):
    """Shared logic for simple unary activation ops."""
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
            op=op_code,
            src_ids=[src_id],
            sym_dim_ids=_vsym,
            sym_dim_exprs=_vexprs,
        )
    )
    ctx.node_to_id[node] = tid


@register_op("aten.silu.default")
def handle_silu(ctx, node, target_str):
    _handle_simple_unary(ctx, node, OP_SILU)


@register_op("aten.relu.default")
def handle_relu(ctx, node, target_str):
    _handle_simple_unary(ctx, node, OP_RELU)


@register_op("aten.tanh.default")
def handle_tanh(ctx, node, target_str):
    _handle_simple_unary(ctx, node, OP_TANH)


@register_op("aten.gelu.default")
def handle_gelu(ctx, node, target_str):
    _handle_simple_unary(ctx, node, OP_GELU)


@register_op("aten.sigmoid.default")
def handle_sigmoid(ctx, node, target_str):
    _handle_simple_unary(ctx, node, OP_SIGMOID)


@register_op("aten.cos.default")
def handle_cos(ctx, node, target_str):
    _handle_simple_unary(ctx, node, OP_COS)


@register_op("aten.sin.default")
def handle_sin(ctx, node, target_str):
    _handle_simple_unary(ctx, node, OP_SIN)


# ---------------------------------------------------------------------------
# Parameterized activations
# ---------------------------------------------------------------------------

@register_op("aten.leaky_relu.default")
def handle_leaky_relu(ctx, node, target_str):
    src_node = node.args[0]
    negative_slope = node.args[1] if len(node.args) > 1 else 0.01
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
            op=OP_LEAKY_RELU,
            src_ids=[src_id],
            op_params=pack_float(float(negative_slope)),
            sym_dim_ids=_vsym,
            sym_dim_exprs=_vexprs,
        )
    )
    ctx.node_to_id[node] = tid


@register_op("aten.hardtanh.default", "aten.clamp.default")
def handle_hardtanh(ctx, node, target_str):
    """hardtanh(x, min_val, max_val) or clamp(x, min, max)."""
    src_node = node.args[0]
    min_val = node.args[1] if len(node.args) > 1 else -1.0
    max_val = node.args[2] if len(node.args) > 2 else 1.0
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
            op=OP_HARDTANH,
            src_ids=[src_id],
            op_params=pack_hardtanh_params(
                float(min_val), float(max_val)
            ),
            sym_dim_ids=_vsym,
            sym_dim_exprs=_vexprs,
        )
    )
    ctx.node_to_id[node] = tid


@register_op("aten._softmax.default")
def handle_softmax(ctx, node, target_str):
    """_softmax(x, dim, half_to_float)."""
    src_node = node.args[0]
    dim = int(node.args[1])
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
            op=OP_SOFTMAX,
            src_ids=[src_id],
            op_params=pack_softmax_params(dim, ndim),
            sym_dim_ids=_vsym,
            sym_dim_exprs=_vexprs,
        )
    )
    ctx.node_to_id[node] = tid
