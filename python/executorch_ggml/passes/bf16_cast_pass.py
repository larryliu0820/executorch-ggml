"""AOT pass to insert BF16->F32 casts around ggml ops that don't support BF16.

Several ggml CPU kernels only support F32 (or F32/F16) inputs and will
GGML_ABORT at runtime when given BF16 tensors. This pass inserts explicit
_to_copy casts *before* each unsafe op (BF16->F32) and *after* (F32->BF16) so
the graph stays in BF16 everywhere else while giving these ops F32 inputs.
"""

import operator

import torch
from executorch.exir.pass_base import ExportPass, PassResult
from torch import fx

# Op name substrings for ops whose ggml kernels don't support BF16.
# Matched via ``in target_str`` to handle both ATen and Edge op overloads.
BF16_UNSAFE_OP_NAMES = {
    "aten.gelu.default",
    "aten.silu.default",
    "aten.native_layer_norm.default",
    "aten._softmax.default",
    "aten.constant_pad_nd.default",
    "aten.pad.default",
    "aten.convolution.default",
    "aten.conv1d.default",
}

# Positional indices of tensor arguments that need casting for each op.
# Ops not listed here: only args[0] (the main input) is cast.
_TENSOR_ARG_INDICES_BY_NAME = {
    "aten.convolution.default": [0, 1],
    "aten.conv1d.default": [0, 1],
    "aten.native_layer_norm.default": [0, 1, 2],
}


def _match_unsafe_op(target) -> str | None:
    """Return the matched op name if target is a BF16-unsafe op, else None."""
    target_str = str(target)
    for name in BF16_UNSAFE_OP_NAMES:
        if name in target_str:
            return name
    return None


def _is_bf16_fake(node: fx.Node) -> bool:
    val = node.meta.get("val")
    if val is None:
        return False
    if isinstance(val, torch.Tensor):
        return val.dtype == torch.bfloat16
    return False


class BF16UnsafeOpsCastPass(ExportPass):
    """Insert BF16->F32 casts before ggml-unsafe ops and F32->BF16 casts after."""

    def call(self, graph_module: fx.GraphModule) -> PassResult:
        graph = graph_module.graph
        changed = False

        for node in list(graph.nodes):
            if node.op != "call_function":
                continue
            matched_name = _match_unsafe_op(node.target)
            if matched_name is None:
                continue

            # Determine which positional args are BF16 tensors that need casting.
            indices = _TENSOR_ARG_INDICES_BY_NAME.get(matched_name)
            args = list(node.args)
            any_cast = False

            for i, arg in enumerate(args):
                if not isinstance(arg, fx.Node):
                    continue
                if indices is not None and i not in indices:
                    continue
                if not _is_bf16_fake(arg):
                    continue

                # Insert cast BF16->F32 before this node.
                with graph.inserting_before(node):
                    cast_to_f32 = graph.call_function(
                        torch.ops.aten._to_copy.default,
                        args=(arg,),
                        kwargs={"dtype": torch.float32},
                    )
                    if "val" in arg.meta:
                        cast_to_f32.meta["val"] = arg.meta["val"].to(torch.float32)

                args[i] = cast_to_f32
                any_cast = True

            if not any_cast:
                continue

            node.args = tuple(args)
            changed = True

            # Determine the output value to see if we need to cast back.
            out_val = node.meta.get("val")

            # native_layer_norm returns a tuple -- only the main output (index 0)
            # needs casting. The other outputs (mean, rstd) are always F32.
            is_tuple_output = isinstance(out_val, (tuple, list))

            if is_tuple_output:
                if isinstance(out_val[0], torch.Tensor) and out_val[0].dtype == torch.bfloat16:
                    new_val = list(out_val)
                    new_val[0] = out_val[0].to(torch.float32)
                    node.meta["val"] = tuple(new_val)

                    # Find getitem users that extract index 0 and cast them back.
                    for user in list(node.users):
                        if (
                            user.op == "call_function"
                            and user.target == operator.getitem
                            and len(user.args) >= 2
                            and user.args[1] == 0
                        ):
                            user.meta["val"] = new_val[0]
                            self._insert_cast_back(graph, user, torch.bfloat16)
            else:
                if isinstance(out_val, torch.Tensor) and out_val.dtype == torch.bfloat16:
                    node.meta["val"] = out_val.to(torch.float32)
                    self._insert_cast_back(graph, node, torch.bfloat16)

        if changed:
            graph.lint()
            graph_module.recompile()

        return PassResult(graph_module, changed)

    @staticmethod
    def _insert_cast_back(graph: fx.Graph, node: fx.Node, target_dtype: torch.dtype):
        """Insert a cast node after ``node`` and redirect all its users."""
        with graph.inserting_after(node):
            cast_back = graph.call_function(
                torch.ops.aten._to_copy.default,
                args=(node,),
                kwargs={"dtype": target_dtype},
            )
            if "val" in node.meta:
                cast_back.meta["val"] = node.meta["val"].to(target_dtype)

        # Replace all downstream users (but not the cast node itself).
        node.replace_all_uses_with(cast_back)
        # Fix: the cast_back node's input must still point to node, not itself.
        cast_back.args = (node,)
