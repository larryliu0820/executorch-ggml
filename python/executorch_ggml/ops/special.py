"""Special operator handlers: scaled_dot_product_attention (SDPA),
ROPE (rotary position embedding), and cache update ops."""

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
    OP_LLAMA_ATTENTION,
    OP_UPDATE_CACHE,
    OP_ROPE,
    pack_sdpa_params,
    pack_update_cache_params,
    pack_rope_params,
)


# ---------------------------------------------------------------------------
# Scaled Dot-Product Attention
# ---------------------------------------------------------------------------

@register_op("aten.scaled_dot_product_attention.default")
def handle_sdpa(ctx, node, target_str):
    """Lower SDPA to fused llama.cpp attention op.

    Args: (q, k, v, attn_mask, dropout_p, is_causal, scale)
    """
    q_node, k_node, v_node = node.args[0], node.args[1], node.args[2]
    mask_node = node.args[3] if len(node.args) > 3 else None
    is_causal = node.args[5] if len(node.args) > 5 else False
    q_id = ctx.node_to_id[q_node]
    k_id = ctx.node_to_id[k_node]
    v_id = ctx.node_to_id[v_node]
    src_ids = [q_id, k_id, v_id]
    if mask_node is not None:
        src_ids.append(ctx.node_to_id[mask_node])
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
            op=OP_LLAMA_ATTENTION,
            src_ids=src_ids,
            op_params=pack_sdpa_params(is_causal=bool(is_causal)),
            sym_dim_ids=_vsym,
            sym_dim_exprs=_vexprs,
        )
    )
    ctx.node_to_id[node] = tid


# ---------------------------------------------------------------------------
# Cache Update (index_copy and llama.update_cache)
# ---------------------------------------------------------------------------

@register_op("index_copy")
def handle_index_copy(ctx, node, target_str):
    """aten.index_copy_(self, dim, index, source) -> self
    or aten.index_copy(self, dim, index, source) -> Tensor

    Used for KV cache updates: cache.index_copy_(seq_dim, positions, kv)
    """
    cache_node = node.args[0]
    seq_dim = int(node.args[1])
    index_node = node.args[2]
    value_node = node.args[3]

    cache_id = ctx.node_to_id[cache_node]
    value_id = ctx.node_to_id[value_node]
    # index_node is the position tensor (input_pos)
    start_pos_id = ctx.node_to_id.get(index_node)
    if start_pos_id is None:
        raise RuntimeError(
            f"Could not find tensor for index_copy index: {index_node}"
        )

    fake_val = node.meta.get("val")
    shape = _resolve_shape(fake_val)
    out_dtype = (
        getattr(fake_val, "dtype", torch.float32)
        if fake_val is not None
        else torch.float32
    )

    tid = ctx.alloc_id()
    ctx.ir_tensors.append(
        IrTensor(
            tensor_id=tid,
            tensor_type=_torch_dtype_to_ir_type(out_dtype),
            ne=_pytorch_shape_to_ggml_ne(shape),
            op=OP_UPDATE_CACHE,
            src_ids=[cache_id, value_id, start_pos_id],
            op_params=pack_update_cache_params(seq_dim),
        )
    )
    ctx.node_to_id[node] = tid


@register_op("llama.update_cache.default")
def handle_llama_update_cache(ctx, node, target_str):
    """llama.update_cache(value, cache, start_pos) -> cache

    Updates cache at start_pos with new values.
    Args: value (new K/V), cache (mutable buffer), start_pos (int64 scalar)
    """
    value_node = node.args[0]
    cache_node = node.args[1]

    value_id = ctx.node_to_id[value_node]
    cache_id = ctx.node_to_id[cache_node]

    # start_pos comes from: item(select(cache_position, 0, 0))
    # We need to trace back to find the original cache_position tensor
    start_pos_node = node.args[2]

    # Helper to trace back through item/select to find the tensor
    def trace_to_tensor(n):
        if n in ctx.node_to_id:
            return ctx.node_to_id[n]
        # Check if this is an item or select node
        if hasattr(n, "target"):
            target_name = str(n.target)
            if "item" in target_name or "select" in target_name:
                # Trace to input
                if n.args:
                    return trace_to_tensor(n.args[0])
        return None

    start_pos_id = trace_to_tensor(start_pos_node)
    if start_pos_id is None:
        raise RuntimeError(
            f"Could not find tensor for update_cache start_pos: {start_pos_node}"
        )

    fake_val = node.meta.get("val")
    shape = _resolve_shape(fake_val)
    out_dtype = (
        getattr(fake_val, "dtype", torch.float32)
        if fake_val is not None
        else torch.float32
    )

    # Determine sequence dimension from cache shape
    # Typical shapes: [batch, seq, n_heads, head_dim] -> seq_dim=1
    #                 [batch, n_heads, seq, head_dim] -> seq_dim=2
    cache_val = cache_node.meta.get("val") if hasattr(cache_node, "meta") else None
    cache_shape = _resolve_shape(cache_val)
    value_val = value_node.meta.get("val") if hasattr(value_node, "meta") else None
    value_shape = _resolve_shape(value_val)

    # The seq dim is where cache_shape differs from value_shape
    seq_dim = 1  # default
    if len(cache_shape) == len(value_shape) and len(cache_shape) >= 2:
        for i in range(1, len(cache_shape)):
            if cache_shape[i] != value_shape[i]:
                seq_dim = i
                break

    tid = ctx.alloc_id()
    ctx.ir_tensors.append(
        IrTensor(
            tensor_id=tid,
            tensor_type=_torch_dtype_to_ir_type(out_dtype),
            ne=_pytorch_shape_to_ggml_ne(shape),
            op=OP_UPDATE_CACHE,
            src_ids=[cache_id, value_id, start_pos_id],
            op_params=pack_update_cache_params(seq_dim),
        )
    )
    ctx.node_to_id[node] = tid


# ---------------------------------------------------------------------------
# Rotary Position Embedding (ROPE)
# ---------------------------------------------------------------------------

@register_op("ggml.rope.default")
def handle_rope(ctx, node, target_str):
    """ggml.rope(x, positions, n_dims, freq_base, mode=0) -> Tensor."""
    x_node = node.args[0]
    pos_node = node.args[1]
    n_dims = int(node.args[2])
    freq_base = float(node.args[3])
    rope_mode = int(node.args[4]) if len(node.args) > 4 else 0

    x_id = ctx.node_to_id[x_node]
    pos_id = ctx.node_to_id[pos_node]

    fake_val = node.meta.get("val")
    shape = _resolve_shape(fake_val)
    out_dtype = getattr(fake_val, "dtype", torch.float32)

    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, ctx.sym_id_map)
    tid = ctx.alloc_id()
    ctx.ir_tensors.append(
        IrTensor(
            tensor_id=tid,
            tensor_type=_torch_dtype_to_ir_type(out_dtype),
            ne=_pytorch_shape_to_ggml_ne(shape),
            op=OP_ROPE,
            src_ids=[x_id, pos_id],
            op_params=pack_rope_params(n_dims, rope_mode, freq_base),
            sym_dim_ids=_vsym,
            sym_dim_exprs=_vexprs,
        )
    )
    ctx.node_to_id[node] = tid
