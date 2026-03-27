"""Linear algebra operator handlers: linear, embedding, mm, addmm, bmm, matmul."""

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
    OP_LINEAR,
    OP_EMBEDDING,
    OP_MUL_MAT,
    OP_ADD,
    OP_BMM,
    OP_CAST,
    pack_cast_params,
)


@register_op("aten.linear.default")
def handle_linear(ctx, node, target_str):
    """linear(input, weight, bias?) -- fused linear layer."""
    x_node = node.args[0]
    w_node = node.args[1]
    b_node = node.args[2] if len(node.args) > 2 else None

    x_id = ctx.node_to_id[x_node]
    w_id = ctx.node_to_id[w_node]
    src_ids = [x_id, w_id]
    if b_node is not None:
        src_ids.append(ctx.node_to_id[b_node])

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
            op=OP_LINEAR,
            src_ids=src_ids,
            sym_dim_ids=_vsym,
            sym_dim_exprs=_vexprs,
        )
    )
    ctx.node_to_id[node] = tid


@register_op("aten.embedding.default")
def handle_embedding(ctx, node, target_str):
    """embedding(weight, indices) -- table lookup."""
    weight_node = node.args[0]
    indices_node = node.args[1]
    w_id = ctx.node_to_id[weight_node]
    idx_id = ctx.node_to_id[indices_node]

    # Cast I64 indices to I32 (ggml_get_rows requires I32).
    idx_fv = indices_node.meta.get("val")
    idx_dtype = (
        getattr(idx_fv, "dtype", torch.int64)
        if idx_fv is not None
        else torch.int64
    )
    if idx_dtype == torch.int64 or idx_dtype == torch.long:
        idx_shape = _resolve_shape(idx_fv)
        _vsym_idx, _vexprs_idx = _sym_dim_info_ggml(idx_fv, ctx.sym_id_map)
        cast_tid = ctx.alloc_id()
        ctx.ir_tensors.append(
            IrTensor(
                tensor_id=cast_tid,
                tensor_type=TYPE_I32,
                ne=_pytorch_shape_to_ggml_ne(idx_shape),
                op=OP_CAST,
                src_ids=[idx_id],
                op_params=pack_cast_params(TYPE_I32),
                sym_dim_ids=_vsym_idx,
                sym_dim_exprs=_vexprs_idx,
            )
        )
        idx_id = cast_tid

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
            op=OP_EMBEDDING,
            src_ids=[w_id, idx_id],
            sym_dim_ids=_vsym,
            sym_dim_exprs=_vexprs,
        )
    )
    ctx.node_to_id[node] = tid


@register_op("aten.mm.default")
def handle_mm(ctx, node, target_str):
    """mm(input, weight_t) -- matrix multiply.

    ggml_mul_mat(a, b) computes b @ a^T, so pass the original
    (un-transposed) weight.
    """
    input_node, weight_t_node = node.args
    orig_w_node = ctx.look_through_transpose(weight_t_node)
    weight_id = ctx.node_to_id[orig_w_node]
    input_id = ctx.node_to_id[input_node]

    fake_val = node.meta.get("val")
    shape = _resolve_shape(fake_val)
    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, ctx.sym_id_map)

    tid = ctx.alloc_id()
    ctx.ir_tensors.append(
        IrTensor(
            tensor_id=tid,
            tensor_type=TYPE_F32,
            ne=_pytorch_shape_to_ggml_ne(shape),
            op=OP_MUL_MAT,
            src_ids=[weight_id, input_id],
            sym_dim_ids=_vsym,
            sym_dim_exprs=_vexprs,
        )
    )
    ctx.node_to_id[node] = tid


@register_op("aten.addmm.default")
def handle_addmm(ctx, node, target_str):
    """addmm(bias, input, weight_t) -- MUL_MAT + ADD.

    ggml_mul_mat(a, b) computes b @ a^T, so pass the original
    (un-transposed) weight.
    """
    bias_node, input_node, weight_t_node = node.args

    orig_w_node = ctx.look_through_transpose(weight_t_node)
    weight_id = ctx.node_to_id[orig_w_node]
    input_id = ctx.node_to_id[input_node]
    bias_id = ctx.node_to_id[bias_node]

    fake_val = node.meta.get("val")
    shape = _resolve_shape(fake_val)
    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, ctx.sym_id_map)

    # First: MUL_MAT
    mm_id = ctx.alloc_id()
    ctx.ir_tensors.append(
        IrTensor(
            tensor_id=mm_id,
            tensor_type=TYPE_F32,
            ne=_pytorch_shape_to_ggml_ne(shape),
            op=OP_MUL_MAT,
            src_ids=[weight_id, input_id],
            sym_dim_ids=_vsym,
            sym_dim_exprs=_vexprs,
        )
    )

    # Second: ADD
    add_id = ctx.alloc_id()
    ctx.ir_tensors.append(
        IrTensor(
            tensor_id=add_id,
            tensor_type=TYPE_F32,
            ne=_pytorch_shape_to_ggml_ne(shape),
            op=OP_ADD,
            src_ids=[mm_id, bias_id],
            sym_dim_ids=_vsym,
            sym_dim_exprs=_vexprs,
        )
    )
    ctx.node_to_id[node] = add_id


@register_op("aten.bmm.default")
def handle_bmm(ctx, node, target_str):
    """bmm(input, mat2) -- batched matrix multiply."""
    a_node, b_node = node.args[0], node.args[1]
    a_id = ctx.node_to_id[a_node]
    b_id = ctx.node_to_id[b_node]

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
            op=OP_BMM,
            src_ids=[a_id, b_id],
            sym_dim_ids=_vsym,
            sym_dim_exprs=_vexprs,
        )
    )
    ctx.node_to_id[node] = tid


@register_op("aten.matmul.default")
def handle_matmul(ctx, node, target_str):
    """matmul(a, b) -- general matrix multiplication via MUL_MAT."""
    a_node = node.args[0]
    b_node = node.args[1]
    a_id = ctx.node_to_id[a_node]
    b_id = ctx.node_to_id[b_node]

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
            op=OP_MUL_MAT,  # Use MUL_MAT for matmul
            src_ids=[a_id, b_id],
            sym_dim_ids=_vsym,
            sym_dim_exprs=_vexprs,
        )
    )
    ctx.node_to_id[node] = tid
