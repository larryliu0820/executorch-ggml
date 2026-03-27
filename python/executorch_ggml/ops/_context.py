"""PreprocessContext: shared state passed to all operator handlers."""

from typing import Dict, List, Optional, Set

import torch
from torch.export import ExportedProgram

from executorch.exir._serialize._named_data_store import NamedDataStore

from executorch_ggml.serialize import IrTensor


class PreprocessContext:
    """Bundles all state that operator handlers need during preprocess().

    Created once at the start of preprocess() and passed to every handler.
    """

    def __init__(
        self,
        edge_program: ExportedProgram,
        data_store: NamedDataStore,
        quant_config,
    ):
        self.edge_program = edge_program
        self.data_store = data_store
        self.quant_config = quant_config

        # Maps from FX node -> IR tensor id
        self.node_to_id: Dict[torch.fx.Node, int] = {}
        # Accumulated IR tensors
        self.ir_tensors: List[IrTensor] = []
        # Monotonic ID counter
        self._next_id: int = 0
        # Maps symbolic variable name (e.g. "s0") -> unique integer ID
        self.sym_id_map: Dict[str, int] = {}

        # Build mapping from param/buffer placeholder names -> FQN -> tensor data
        sig = edge_program.graph_signature
        self.param_map: Dict[str, str] = dict(sig.inputs_to_parameters)
        self.buffer_map: Dict[str, str] = dict(sig.inputs_to_buffers)
        self.mutated_buffer_fqns: Set[str] = set(
            getattr(sig, "buffers_to_mutate", {}).values()
        )
        self.lifted_const_map: Dict[str, str] = dict(
            getattr(sig, "inputs_to_lifted_tensor_constants", {}) or {}
        )
        self.ep_constants = getattr(edge_program, "constants", {}) or {}
        self.ep_tensor_constants = getattr(edge_program, "tensor_constants", {}) or {}

    def alloc_id(self) -> int:
        """Allocate the next tensor ID."""
        tid = self._next_id
        self._next_id += 1
        return tid

    @staticmethod
    def look_through_transpose(n: torch.fx.Node) -> torch.fx.Node:
        """Look through aten.t, aten.permute_copy, aten.permute that
        transposes a 2D weight tensor. ggml_mul_mat already does an
        implicit transpose, so mm/addmm should pass the original weight."""
        if n.op != "call_function":
            return n
        t = str(n.target)
        if "aten.t" in t or "aten.permute_copy" in t or "aten.permute" in t:
            return n.args[0]
        return n
