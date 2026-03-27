"""Misc operator handlers: assert_tensor_metadata, scalar_tensor, t.default."""

import torch

from executorch_ggml.ops._registry import register_op
from executorch_ggml.ops._helpers import (
    _pytorch_shape_to_ggml_ne,
    _torch_dtype_to_ir_type,
)
from executorch_ggml.serialize import (
    IrTensor,
    OP_NONE,
)


@register_op("aten._assert_tensor_metadata.default")
def handle_assert_tensor_metadata(ctx, node, target_str):
    """No-op shape/dtype assertion inserted by export."""
    src_node = node.args[0]
    ctx.node_to_id[node] = ctx.node_to_id[src_node]


@register_op("aten.scalar_tensor.default")
def handle_scalar_tensor(ctx, node, target_str):
    """scalar_tensor(s, dtype?, device?) -> 0-d tensor constant."""
    fake_val = node.meta.get("val")
    out_dtype = (
        getattr(fake_val, "dtype", torch.float32)
        if fake_val is not None
        else torch.float32
    )
    value = node.args[0]

    # Dynamic scalar: value is a graph Node (from sym_size etc.)
    # For fixed-shape exports, use the concrete value from
    # the node's fake tensor metadata.
    if isinstance(value, torch.fx.Node):
        node_fake = node.meta.get("val")
        if node_fake is not None and isinstance(node_fake, torch.Tensor):
            value = node_fake.item()
        else:
            value = 0  # fallback for unknown symbolic scalars
    const = torch.tensor(value, dtype=out_dtype).cpu()

    const = torch.tensor(value, dtype=out_dtype).cpu()

    tid = ctx.alloc_id()
    # Ensure NamedDataStore keys are stable and unique across
    # multiple delegated submodules. Using just `tid` can
    # collide when merging named data stores from different
    # lowered partitions.
    key = f"__const_scalar_{node.name}_{tid}"
    ctx.data_store.add_named_data(key, const, alignment=64)
    ctx.ir_tensors.append(
        IrTensor(
            tensor_id=tid,
            tensor_type=_torch_dtype_to_ir_type(out_dtype),
            ne=_pytorch_shape_to_ggml_ne(list(const.shape)),
            op=OP_NONE,
            data_key=key,
            is_input=False,
        )
    )
    ctx.node_to_id[node] = tid


@register_op("aten.t.default")
def handle_t(ctx, node, target_str):
    """aten.t is a 2D transpose. For matmul lowering, ggml_mul_mat
    expects weight layout that already matches after shape reversal,
    so we treat it as a no-op (look-through).
    NOTE: aten.permute_copy is handled later via the full permute path."""
    src_node = node.args[0]
    ctx.node_to_id[node] = ctx.node_to_id[src_node]


@register_op("dim_order_ops._clone_dim_order.default")
def handle_clone_dim_order(ctx, node, target_str):
    """Layout materialization op used by ExecuTorch/Edge — no-op for ggml."""
    src_node = node.args[0]
    ctx.node_to_id[node] = ctx.node_to_id[src_node]


@register_op("dim_order_ops._to_dim_order_copy.default")
def handle_to_dim_order_copy(ctx, node, target_str):
    """Dim-order copy op used by ExecuTorch/Edge — no-op for ggml."""
    src_node = node.args[0]
    ctx.node_to_id[node] = ctx.node_to_id[src_node]
