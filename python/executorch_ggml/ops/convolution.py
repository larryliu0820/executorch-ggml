"""Convolution and padding operator handlers: constant_pad_nd, convolution/conv2d."""

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
    OP_PAD,
    OP_CONV_1D,
    OP_CONV_1D_DW,
    OP_CONV_2D,
    OP_CONV_2D_DW,
    pack_pad_params,
    pack_conv1d_params,
    pack_conv2d_params,
)


# ---------------------------------------------------------------------------
# Padding
# ---------------------------------------------------------------------------

@register_op("aten.constant_pad_nd.default")
def handle_constant_pad_nd(ctx, node, target_str):
    """constant_pad_nd(input, pad_list, value=0.0)."""
    src_node = node.args[0]
    src_id = ctx.node_to_id[src_node]
    pad_list = [int(p) for p in node.args[1]]
    fill_value = float(node.args[2]) if len(node.args) > 2 else 0.0

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
            op=OP_PAD,
            src_ids=[src_id],
            op_params=pack_pad_params(pad_list, fill_value),
            sym_dim_ids=_vsym,
            sym_dim_exprs=_vexprs,
        )
    )
    ctx.node_to_id[node] = tid


# ---------------------------------------------------------------------------
# Convolution (1D and 2D, regular and depthwise)
# ---------------------------------------------------------------------------

@register_op("aten.convolution.default", "aten.conv2d.default")
def handle_convolution(ctx, node, target_str):
    """aten.convolution.default or aten.conv2d.default.

    Dispatches to OP_CONV_2D, OP_CONV_2D_DW, OP_CONV_1D, or OP_CONV_1D_DW
    based on input dimensions and groups.

    Args layout for convolution.default:
        (input, weight, bias?, stride, padding, dilation, transposed, output_padding, groups)
    Args layout for conv2d.default:
        (input, weight, bias, stride, padding, dilation, groups)
    """
    if "aten.convolution.default" in target_str:
        if len(node.args) < 9:
            raise RuntimeError(
                f"Expected 9 args for convolution, got {len(node.args)}"
            )
        input_node = node.args[0]
        weight_node = node.args[1]
        bias_node = (
            node.args[2]
            if len(node.args) > 2 and node.args[2] is not None
            else None
        )
        stride = list(node.args[3]) if len(node.args) > 3 else [1, 1]
        padding = list(node.args[4]) if len(node.args) > 4 else [0, 0]
        dilation = list(node.args[5]) if len(node.args) > 5 else [1, 1]
        transposed = node.args[6] if len(node.args) > 6 else False
        output_padding = (
            list(node.args[7]) if len(node.args) > 7 else [0, 0]
        )
        groups = node.args[8] if len(node.args) > 8 else 1
    else:  # conv2d
        if len(node.args) < 7:
            raise RuntimeError(
                f"Expected 7 args for conv2d, got {len(node.args)}"
            )
        input_node = node.args[0]
        weight_node = node.args[1]
        bias_node = (
            node.args[2]
            if len(node.args) > 2 and node.args[2] is not None
            else None
        )
        stride = list(node.args[3]) if len(node.args) > 3 else [1, 1]
        padding = list(node.args[4]) if len(node.args) > 4 else [0, 0]
        dilation = list(node.args[5]) if len(node.args) > 5 else [1, 1]
        groups = node.args[6] if len(node.args) > 6 else 1
        transposed = False
        output_padding = [0, 0]

    if transposed:
        raise RuntimeError("Transposed convolution not yet supported")

    weight_id = ctx.node_to_id[weight_node]
    if bias_node is not None:
        bias_id = ctx.node_to_id[bias_node]
    else:
        bias_id = None

    input_id = ctx.node_to_id[input_node]

    fake_val = node.meta.get("val")
    shape = _resolve_shape(fake_val)

    is_depthwise = groups > 1

    # Build src_ids: [weight, input, bias?]
    if bias_id is not None:
        src_ids = [weight_id, input_id, bias_id]
    else:
        src_ids = [weight_id, input_id]

    # Detect 1D vs 2D by stride length
    is_conv1d = len(stride) == 1

    if is_conv1d:
        op_params = pack_conv1d_params(
            stride[0], padding[0], dilation[0], groups
        )
        op_code = OP_CONV_1D_DW if is_depthwise else OP_CONV_1D
    else:
        op_params = pack_conv2d_params(stride, padding, dilation, groups)
        op_code = OP_CONV_2D_DW if is_depthwise else OP_CONV_2D

    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, ctx.sym_id_map)
    tid = ctx.alloc_id()
    ctx.ir_tensors.append(
        IrTensor(
            tensor_id=tid,
            tensor_type=TYPE_F32,
            ne=_pytorch_shape_to_ggml_ne(shape),
            op=op_code,
            src_ids=src_ids,
            op_params=op_params,
            sym_dim_ids=_vsym,
            sym_dim_exprs=_vexprs,
        )
    )
    ctx.node_to_id[node] = tid
