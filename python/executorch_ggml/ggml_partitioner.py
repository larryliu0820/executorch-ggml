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


# ATen ops that nn.Linear + nn.LeakyReLU decompose into after torch.export
_SUPPORTED_OPS = {
    torch.ops.aten.t.default,
    torch.ops.aten.addmm.default,
    torch.ops.aten.mm.default,
    torch.ops.aten.leaky_relu.default,
}

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
            if node.op == "call_function" and node.target in _SUPPORTED_OPS:
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
