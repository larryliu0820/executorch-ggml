"""Normalization operator handlers: layer_norm, rms_norm, batch_norm."""

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
    OP_LAYER_NORM,
    OP_RMS_NORM,
    OP_BATCH_NORM,
    pack_layer_norm_params,
    pack_rms_norm_params,
    pack_batch_norm_params,
)


# ---------------------------------------------------------------------------
# native_layer_norm
# ---------------------------------------------------------------------------

@register_op("aten.native_layer_norm.default")
def handle_layer_norm(ctx, node, target_str):
    """native_layer_norm(input, normalized_shape, weight, bias, eps).
    Returns tuple (output, mean, rstd)."""
    input_node = node.args[0]
    # normalized_shape = node.args[1]  # not needed for IR
    weight_node = node.args[2] if len(node.args) > 2 else None
    bias_node = node.args[3] if len(node.args) > 3 else None
    eps = float(node.args[4]) if len(node.args) > 4 else 1e-5

    input_id = ctx.node_to_id[input_node]
    has_weight = weight_node is not None and not (
        isinstance(weight_node, type(None))
    )
    has_bias = bias_node is not None and not (
        isinstance(bias_node, type(None))
    )

    src_ids = [input_id]
    if has_weight:
        src_ids.append(ctx.node_to_id[weight_node])
    if has_bias:
        src_ids.append(ctx.node_to_id[bias_node])

    # Output shape comes from the first element of the tuple
    fake_val = node.meta.get("val")
    if isinstance(fake_val, (list, tuple)):
        out_fv = fake_val[0]
    else:
        out_fv = fake_val
    shape = _resolve_shape(out_fv)
    out_dtype = (
        getattr(out_fv, "dtype", torch.float32)
        if out_fv is not None
        else torch.float32
    )

    _vsym, _vexprs = _sym_dim_info_ggml(out_fv, ctx.sym_id_map)
    tid = ctx.alloc_id()
    ctx.ir_tensors.append(
        IrTensor(
            tensor_id=tid,
            tensor_type=_torch_dtype_to_ir_type(out_dtype),
            ne=_pytorch_shape_to_ggml_ne(shape),
            op=OP_LAYER_NORM,
            src_ids=src_ids,
            op_params=pack_layer_norm_params(
                eps, has_weight, has_bias
            ),
            sym_dim_ids=_vsym,
            sym_dim_exprs=_vexprs,
        )
    )
    # Store as single int -- getitem(0) will resolve to this
    ctx.node_to_id[node] = tid


# ---------------------------------------------------------------------------
# rms_norm
# ---------------------------------------------------------------------------

@register_op("aten.rms_norm.default")
def handle_rms_norm(ctx, node, target_str):
    """rms_norm(input, normalized_shape, weight, eps)."""
    input_node = node.args[0]
    # normalized_shape = node.args[1]  # not needed for IR
    weight_node = node.args[2] if len(node.args) > 2 else None
    eps = float(node.args[3]) if len(node.args) > 3 and node.args[3] is not None else 1e-5

    input_id = ctx.node_to_id[input_node]
    has_weight = weight_node is not None and not isinstance(weight_node, type(None))

    src_ids = [input_id]
    if has_weight:
        src_ids.append(ctx.node_to_id[weight_node])

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
            op=OP_RMS_NORM,
            src_ids=src_ids,
            op_params=pack_rms_norm_params(eps, has_weight),
            sym_dim_ids=_vsym,
            sym_dim_exprs=_vexprs,
        )
    )
    ctx.node_to_id[node] = tid


# ---------------------------------------------------------------------------
# batch_norm (inference-only, no training)
# ---------------------------------------------------------------------------

@register_op("aten._native_batch_norm_legit_no_training.default")
def handle_batch_norm(ctx, node, target_str):
    """_native_batch_norm_legit_no_training(input, weight, bias, running_mean, running_var, momentum, eps).
    Returns tuple (output, mean, rstd)."""
    input_node = node.args[0]
    weight_node = node.args[1] if len(node.args) > 1 else None
    bias_node = node.args[2] if len(node.args) > 2 else None
    mean_node = node.args[3] if len(node.args) > 3 else None
    var_node = node.args[4] if len(node.args) > 4 else None
    # momentum = node.args[5]  # not used
    eps = float(node.args[6]) if len(node.args) > 6 else 1e-5

    input_id = ctx.node_to_id[input_node]

    src_ids = [input_id]
    has_w = isinstance(weight_node, torch.fx.Node) and weight_node in ctx.node_to_id
    has_b = isinstance(bias_node, torch.fx.Node) and bias_node in ctx.node_to_id
    has_m = isinstance(mean_node, torch.fx.Node) and mean_node in ctx.node_to_id
    has_v = isinstance(var_node, torch.fx.Node) and var_node in ctx.node_to_id

    if has_w:
        src_ids.append(ctx.node_to_id[weight_node])
    if has_b:
        src_ids.append(ctx.node_to_id[bias_node])
    if has_m:
        src_ids.append(ctx.node_to_id[mean_node])
    if has_v:
        src_ids.append(ctx.node_to_id[var_node])

    # Output shape from tuple[0]
    fake_val = node.meta.get("val")
    if isinstance(fake_val, (list, tuple)):
        out_fv = fake_val[0]
    else:
        out_fv = fake_val
    shape = _resolve_shape(out_fv)
    out_dtype = (
        getattr(out_fv, "dtype", torch.float32)
        if out_fv is not None
        else torch.float32
    )

    _vsym, _vexprs = _sym_dim_info_ggml(out_fv, ctx.sym_id_map)
    tid = ctx.alloc_id()
    ctx.ir_tensors.append(
        IrTensor(
            tensor_id=tid,
            tensor_type=_torch_dtype_to_ir_type(out_dtype),
            ne=_pytorch_shape_to_ggml_ne(shape),
            op=OP_BATCH_NORM,
            src_ids=src_ids,
            op_params=pack_batch_norm_params(eps),
            sym_dim_ids=_vsym,
            sym_dim_exprs=_vexprs,
        )
    )
    # Store as single int -- getitem(0) will resolve to this
    ctx.node_to_id[node] = tid
