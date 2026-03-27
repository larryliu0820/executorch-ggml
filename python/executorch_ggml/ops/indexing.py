"""Indexing operator handlers: index (gather), index_put (scatter)."""

from typing import List

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
    TYPE_I32,
    OP_INDEX,
    OP_INDEX_MULTI,
    OP_INDEX_PUT,
    OP_CAST,
    pack_index_params,
    pack_index_multi_params,
    pack_index_put_multi_params,
    pack_cast_params,
)


# ---------------------------------------------------------------------------
# index_put MUST be registered BEFORE index.Tensor because the registry is
# first-match-wins and "aten.index.Tensor" is a substring of
# "aten.index_put.default".  By registering index_put first, it matches
# index_put targets before the broader index.Tensor pattern could.
# ---------------------------------------------------------------------------

@register_op("aten.index_put.default")
def handle_index_put(ctx, node, target_str):
    """index_put(x, indices, values, accumulate?) -- KV cache scatter.

    Lowers to OP_INDEX_PUT with a bitmask indicating which index dimensions
    are present (non-None).
    """
    x_node = node.args[0]
    indices = node.args[1]
    values_node = node.args[2]

    if not isinstance(indices, (list, tuple)):
        raise RuntimeError(
            "aten.index_put: expected indices to be a list/tuple"
        )

    # Support multi-index form: indices is a tuple of optional index tensors.
    # We'll serialize only the non-None index tensors as src_ids.
    present_mask = 0
    idx_src_ids: List[int] = []
    for i, idx in enumerate(indices):
        if idx is None:
            continue
        present_mask |= 1 << i
        idx_src_ids.append(ctx.node_to_id[idx])

    x_id = ctx.node_to_id[x_node]
    v_id = ctx.node_to_id[values_node]

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
            op=OP_INDEX_PUT,
            src_ids=[x_id] + idx_src_ids + [v_id],
            op_params=pack_index_put_multi_params(
                len(indices), present_mask
            ),
            sym_dim_ids=_vsym,
            sym_dim_exprs=_vexprs,
        )
    )
    ctx.node_to_id[node] = tid


@register_op("aten.index.Tensor")
def handle_index(ctx, node, target_str):
    """index(x, indices) -- gather rows or multi-dimensional index.

    Detects single-index (dim-0 gather via OP_INDEX / ggml_get_rows) vs.
    multi-index (all dims present, via OP_INDEX_MULTI).

    NOTE: "aten.index.Tensor" is a substring of "aten.index_put.default",
    but index_put is registered first in the registry so it matches before
    this handler for index_put targets.
    """
    src_node = node.args[0]
    indices = node.args[1]
    if not isinstance(indices, (list, tuple)):
        raise RuntimeError(
            "aten.index.Tensor: expected indices to be list/tuple"
        )

    non_none_pos = [
        i for i, idx in enumerate(indices) if idx is not None
    ]
    non_none = [indices[i] for i in non_none_pos]

    fake_val = node.meta.get("val")
    shape = _resolve_shape(fake_val)
    out_dtype = (
        getattr(fake_val, "dtype", torch.float32)
        if fake_val is not None
        else torch.float32
    )

    if (
        len(non_none) == 1
        and len(non_none_pos) == 1
        and non_none_pos[0] == 0
    ):
        # Single-index case: use ggml_get_rows (gather along dim 0).
        idx_node = non_none[0]
        src_id = ctx.node_to_id[src_node]
        idx_id = ctx.node_to_id[idx_node]

        # Cast I64 indices to I32 (ggml_get_rows requires I32).
        idx_fv = idx_node.meta.get("val")
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

        _vsym, _vexprs = _sym_dim_info_ggml(fake_val, ctx.sym_id_map)
        tid = ctx.alloc_id()
        ctx.ir_tensors.append(
            IrTensor(
                tensor_id=tid,
                tensor_type=_torch_dtype_to_ir_type(out_dtype),
                ne=_pytorch_shape_to_ggml_ne(shape),
                op=OP_INDEX,
                src_ids=[src_id, idx_id],
                op_params=pack_index_params(0),
                sym_dim_ids=_vsym,
                sym_dim_exprs=_vexprs,
            )
        )
        ctx.node_to_id[node] = tid

    elif len(non_none) > 1 and len(non_none) == len(indices):
        # Multi-index case (all indices present):
        # lower to runtime custom gather op.
        src_val = src_node.meta.get("val")
        src_shape = _resolve_shape(src_val)
        if len(src_shape) == 0 or len(src_shape) > 4:
            raise RuntimeError(
                "aten.index.Tensor multi-index: unsupported source rank "
                f"{len(src_shape)}"
            )
        if len(indices) != len(src_shape):
            raise RuntimeError(
                "aten.index.Tensor multi-index: indices rank "
                f"{len(indices)} does not match source rank {len(src_shape)}"
            )

        src_id = ctx.node_to_id[src_node]
        idx_ids = [ctx.node_to_id[i] for i in indices]
        _vsym, _vexprs = _sym_dim_info_ggml(fake_val, ctx.sym_id_map)
        tid = ctx.alloc_id()
        ctx.ir_tensors.append(
            IrTensor(
                tensor_id=tid,
                tensor_type=_torch_dtype_to_ir_type(out_dtype),
                ne=_pytorch_shape_to_ggml_ne(shape),
                op=OP_INDEX_MULTI,
                src_ids=[src_id] + idx_ids,
                op_params=pack_index_multi_params(src_shape),
                sym_dim_ids=_vsym,
                sym_dim_exprs=_vexprs,
            )
        )
        ctx.node_to_id[node] = tid

    else:
        raise RuntimeError(
            "aten.index.Tensor: unsupported indexing pattern for ggml lowering"
        )
