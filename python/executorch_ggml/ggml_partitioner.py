"""GgmlPartitioner: tags supported ATen ops for delegation to the ggml backend."""

from typing import Dict, List

import torch
from torch.export import ExportedProgram

from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)
from executorch.exir.backend.utils import tag_constant_data


# ATen ops that we support in the ggml backend.
#
# After conversion to Edge dialect, node.target becomes an EdgeOpOverload,
# not equal to torch.ops. So we match by name.
_SUPPORTED_OP_NAMES = {
    # Linear layers
    "aten.t.default",
    "aten.permute_copy.default",
    "aten.addmm.default",
    "aten.mm.default",
    "aten.leaky_relu.default",
    # MobileNetV2 ops
    "aten.convolution.default",
    "aten.conv2d.default",
    "aten.hardtanh.default",
    "aten.clamp.default",
    "aten.mean.dim",
    "aten._mean_dim.default",
    "aten.view.default",
    "aten._unsafe_view.default",
    "aten.reshape.default",
    "aten.permute.default",
    "aten.add.Tensor",
    "dim_order_ops._clone_dim_order.default",
    # BatchNorm should be removed by BatchNormFoldingRewritePass prior to partitioning.
}


def _is_supported_target(target) -> bool:
    s = str(target)
    return any(name in s for name in _SUPPORTED_OP_NAMES)

BACKEND_ID = "GgmlBackend"


class GgmlPartitioner(Partitioner):
    """Partitions an Edge-dialect ExportedProgram, delegating supported ops to ggml."""

    def __init__(self):
        super().__init__()
        self.delegation_spec = DelegationSpec(BACKEND_ID, [])

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        partition_tags: Dict[str, DelegationSpec] = {}
        graph_module = exported_program.graph_module

        # Build adjacency: for each supported node, find connected supported neighbours
        supported_nodes = []
        for node in graph_module.graph.nodes:
            if node.op == "call_function" and _is_supported_target(node.target):
                supported_nodes.append(node)

        if not supported_nodes:
            return PartitionResult(
                tagged_exported_program=exported_program,
                partition_tags={},
            )

        # Union-Find to group connected supported nodes into partitions
        parent: Dict[torch.fx.Node, torch.fx.Node] = {}

        def find(n: torch.fx.Node) -> torch.fx.Node:
            while parent[n] is not n:
                parent[n] = parent[parent[n]]
                n = parent[n]
            return n

        def union(a: torch.fx.Node, b: torch.fx.Node) -> None:
            ra, rb = find(a), find(b)
            if ra is not rb:
                parent[ra] = rb

        supported_set = set(supported_nodes)
        for node in supported_nodes:
            parent[node] = node

        # Merge nodes that are connected via data-flow edges
        for node in supported_nodes:
            for inp in node.all_input_nodes:
                if inp in supported_set:
                    union(node, inp)
            for user in node.users:
                if user in supported_set:
                    union(node, user)

        # Group by root and assign tags
        groups: Dict[torch.fx.Node, List[torch.fx.Node]] = {}
        for node in supported_nodes:
            root = find(node)
            groups.setdefault(root, []).append(node)

        for idx, (_, group_nodes) in enumerate(groups.items()):
            tag = f"ggml_partition_{idx}"
            for node in group_nodes:
                node.meta["delegation_tag"] = tag
            partition_tags[tag] = self.delegation_spec

        # Tag constant data (params, buffers) used exclusively by tagged nodes
        tag_constant_data(exported_program)

        return PartitionResult(
            tagged_exported_program=exported_program,
            partition_tags=partition_tags,
        )
