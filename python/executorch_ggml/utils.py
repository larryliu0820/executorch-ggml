"""Utilities for multi-method ExecuTorch export with GGML backend."""

import copy
from typing import Dict
from torch.export import ExportedProgram


def namespace_lifted_constants(
    programs: Dict[str, ExportedProgram],
) -> Dict[str, ExportedProgram]:
    """Prefix lifted tensor constant FQNs with the method name to avoid
    key collisions when multiple methods are merged into a single .pte.

    PyTorch export auto-generates names like ``_lifted_tensor_constant0``
    that can collide across methods when different constants get the same
    name.  This function renames them to ``<method>/_lifted_tensor_constant0``
    so the global NamedDataStore sees unique keys.

    Only modifies programs that have lifted constants; single-method exports
    or programs without lifted constants are returned unchanged.
    """
    if len(programs) <= 1:
        return programs

    result = {}
    for method_name, ep in programs.items():
        sig = ep.graph_signature
        ltc = getattr(sig, "inputs_to_lifted_tensor_constants", None)
        if not ltc:
            result[method_name] = ep
            continue

        # Build old→new FQN mapping for lifted constants
        rename_map = {}
        for node_name, fqn in ltc.items():
            new_fqn = f"{method_name}/{fqn}"
            rename_map[fqn] = new_fqn

        if not rename_map:
            result[method_name] = ep
            continue

        # Rename in signature
        new_ltc = {k: rename_map.get(v, v) for k, v in ltc.items()}

        # Rename in constants dict
        new_constants = {}
        old_constants = getattr(ep, "constants", {}) or {}
        for k, v in old_constants.items():
            new_constants[rename_map.get(k, k)] = v

        new_tensor_constants = {}
        old_tc = getattr(ep, "tensor_constants", {}) or {}
        for k, v in old_tc.items():
            new_tensor_constants[rename_map.get(k, k)] = v

        # Create updated ExportedProgram with renamed constants
        # Use _replace on the signature to update lifted constant mapping
        new_sig = copy.copy(sig)
        new_sig.inputs_to_lifted_tensor_constants = new_ltc

        # ExportedProgram is mostly immutable; create a new one with updated fields
        new_ep = ExportedProgram(
            root=ep.graph_module,
            graph=ep.graph,
            graph_signature=new_sig,
            state_dict=ep.state_dict,
            range_constraints=ep.range_constraints,
            module_call_graph=ep.module_call_graph,
            constants=new_constants,
            verifiers=[],
        )
        result[method_name] = new_ep

    return result
