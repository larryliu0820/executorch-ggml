"""Utilities for building Edge + lowering pipelines with executorch-ggml.

Provides a pipeline that:
1. Replaces _copy ops with non-copy variants
2. Partitions for ggml backend

For SDPA preservation, pass compile_config with preserve_ops=[torch.ops.aten.scaled_dot_product_attention.default]
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence, Union

from torch.export import ExportedProgram

from executorch.exir import to_edge
from executorch.exir.backend.partitioner import Partitioner
from executorch.exir.pass_manager import PassManager, PassType
from executorch.exir.program._program import EdgeCompileConfig, EdgeProgramManager


class ExportedProgramPassLike:
    """A small structural type for ExportedProgram-level passes.

    Any object with a `run(ep) -> ...` method is accepted. The return value can be:
    - an ExportedProgram
    - an object with an `.ep` attribute containing the updated ExportedProgram
      (e.g. our RewriteResult)
    """

    def run(self, ep: ExportedProgram):  # pragma: no cover
        raise NotImplementedError


def to_edge_rewrite_and_lower(
    programs: Union[ExportedProgram, Dict[str, ExportedProgram]],
    *,
    ep_passes: Optional[
        Union[
            Sequence[ExportedProgramPassLike],
            Dict[str, Sequence[ExportedProgramPassLike]],
        ]
    ] = None,
    partitioner: Optional[
        Union[List[Partitioner], Dict[str, List[Partitioner]]]
    ] = None,
    transform_passes: Optional[
        Union[Sequence[PassType], Dict[str, Sequence[PassType]], PassManager]
    ] = None,
    constant_methods: Optional[Dict[str, Any]] = None,
    compile_config: Optional[EdgeCompileConfig] = None,
    generate_etrecord: bool = False,
) -> EdgeProgramManager:
    """Like `executorch.exir.to_edge_transform_and_lower`, plus ExportedProgram rewrites.

    Pipeline:
      1) to_edge (ATen -> Edge)
      2) optional transform_passes (GraphModule-only; ExecuTorch-native)
      3) optional ep_passes (ExportedProgram-level; can update state_dict/signature)
      4) to_backend partitioning/lowering
    """
    edge_manager = to_edge(
        programs,
        constant_methods=constant_methods,
        compile_config=compile_config,
        generate_etrecord=generate_etrecord,
    )

    if transform_passes is not None:
        edge_manager = edge_manager.transform(transform_passes)

    # Normalize ep_passes to per-method mapping.
    if ep_passes is None:
        ep_passes = {name: [] for name in edge_manager._edge_programs.keys()}  # noqa: SLF001
    elif not isinstance(ep_passes, dict):
        ep_passes = {
            name: list(ep_passes)
            for name in edge_manager._edge_programs.keys()  # noqa: SLF001
        }

    # Apply ExportedProgram-level rewrites.
    new_programs: Dict[str, ExportedProgram] = {}
    for name, ep in edge_manager._edge_programs.items():  # noqa: SLF001 (ExecuTorch internal)
        cur = ep
        for p in ep_passes.get(name, []):
            out = p.run(cur)
            if isinstance(out, ExportedProgram):
                cur = out
            elif hasattr(out, "ep") and isinstance(out.ep, ExportedProgram):
                cur = out.ep
            else:
                raise TypeError(
                    f"ExportedProgram pass {p} returned unsupported type {type(out)}"
                )
        new_programs[name] = cur

    edge_manager = EdgeProgramManager(
        new_programs,
        copy.deepcopy(edge_manager._config_methods),  # noqa: SLF001
        edge_manager.compile_config,
    )

    # Mirror to_edge_transform_and_lower partitioner behavior.
    if not isinstance(partitioner, dict) and partitioner is not None:
        partitioner = {name: partitioner for name in new_programs.keys()}
    elif partitioner is None:
        partitioner = {name: [] for name in new_programs.keys()}

    max_num_partitioners = 0
    for partitioner_list in partitioner.values():
        max_num_partitioners = max(max_num_partitioners, len(partitioner_list))

    for i in range(max_num_partitioners):
        method_to_partitioner = {}
        for name, partitioner_list in partitioner.items():
            if i < len(partitioner_list):
                method_to_partitioner[name] = partitioner_list[i]
        edge_manager = edge_manager.to_backend(method_to_partitioner)

    return edge_manager
