"""Pass to replace _copy ops with their non-copy variants for ggml backend.

This extends the AOTI ReplaceViewCopyWithViewPass to handle additional ops
that ggml can support without memory planning constraints.
"""

from typing import Dict, Iterable

import torch
from executorch.exir.dialects._ops import ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import ExportPass, PassResult
from torch import fx


_COPY_TO_ORIGINAL: Dict[
    torch._ops.OpOverload | EdgeOpOverload, torch._ops.OpOverload | EdgeOpOverload
] = {
    torch.ops.aten.slice_copy.Tensor: torch.ops.aten.slice.Tensor,
    ops.edge.aten.slice_copy.Tensor: ops.edge.aten.slice.Tensor,
    torch.ops.aten.select_copy.int: torch.ops.aten.select.int,
    ops.edge.aten.select_copy.int: ops.edge.aten.select.int,
    torch.ops.aten.view_copy.default: torch.ops.aten.view.default,
    ops.edge.aten.view_copy.default: ops.edge.aten.view.default,
    torch.ops.aten.permute_copy.default: torch.ops.aten.permute.default,
    ops.edge.aten.permute_copy.default: ops.edge.aten.permute.default,
    torch.ops.aten.unsqueeze_copy.default: torch.ops.aten.unsqueeze.default,
    ops.edge.aten.unsqueeze_copy.default: ops.edge.aten.unsqueeze.default,
    torch.ops.aten.expand_copy.default: torch.ops.aten.expand.default,
    ops.edge.aten.expand_copy.default: ops.edge.aten.expand.default,
    torch.ops.aten.alias_copy.default: torch.ops.aten.alias.default,
    ops.edge.aten.alias_copy.default: ops.edge.aten.alias.default,
    # NOTE: clone -> alias is NOT safe in general because clone ensures contiguity
    # which is required before view/reshape on expanded tensors.
    # Uncomment only if you're sure the graph doesn't have expand->clone->view patterns.
    # torch.ops.aten.clone.default: torch.ops.aten.alias.default,
    # ops.edge.aten.clone.default: ops.edge.aten.alias.default,
}


class ReplaceCopyOpsPass(ExportPass):
    """Replace _copy ops with their non-copy variants.

    This is suitable for backends like ggml that don't need ExecuTorch's
    memory planning infrastructure.
    """

    def call(self, graph_module: fx.GraphModule) -> PassResult:
        graph_changed = False

        for node in graph_module.graph.nodes:
            if node.op != "call_function" or node.target not in _COPY_TO_ORIGINAL:
                continue

            if self._has_blocking_user(node, node.users.keys()):
                continue

            node.target = _COPY_TO_ORIGINAL[node.target]
            graph_changed = True

        if graph_changed:
            graph_module.graph.lint()
            graph_module.recompile()

        return PassResult(graph_module, graph_changed)

    def _has_blocking_user(self, node: fx.Node, users: Iterable[fx.Node]) -> bool:
        for user in users:
            if self._is_mutating_user(node, user):
                return True
        return False

    def _is_mutating_user(self, node: fx.Node, user: fx.Node) -> bool:
        if user.op == "call_method":
            return isinstance(user.target, str) and user.target.endswith("_")

        if user.op != "call_function":
            return False

        target = user.target
        if not hasattr(target, "_schema"):
            return False

        schema = target._schema
        for index, arg in enumerate(user.args):
            if arg is node and self._argument_mutates(schema, index):
                return True

        for name, arg in user.kwargs.items():
            if arg is node and self._argument_mutates(schema, name):
                return True

        return False

    def _argument_mutates(
        self, schema: torch._C.FunctionSchema, key: int | str
    ) -> bool:
        arguments = schema.arguments
        if isinstance(key, int):
            if key >= len(arguments):
                return False
            argument = arguments[key]
        else:
            argument = next((arg for arg in arguments if arg.name == key), None)
            if argument is None:
                return False

        alias_info = argument.alias_info
        return bool(alias_info and alias_info.is_write)
