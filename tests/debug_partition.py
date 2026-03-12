#!/usr/bin/env python3
"""Debug: check why PE slice is in a separate partition."""

import os, sys
import torch
from torch.export import Dim, export

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _repo_root)
_parakeet_pkg = "executorch.examples.models.parakeet.quantize"
if _parakeet_pkg not in sys.modules:
    import types; _stub = types.ModuleType(_parakeet_pkg); _stub.quantize_model_ = lambda *a, **kw: None; sys.modules[_parakeet_pkg] = _stub
sys.path.insert(0, os.path.join(_repo_root, "third-party", "executorch", "examples", "models", "parakeet"))

from export_parakeet_ggml import export_all_ggml, load_model
from executorch_ggml import GgmlPartitioner, to_edge_rewrite_and_lower
from executorch_ggml.passes.replace_copy_ops_pass import ReplaceCopyOpsPass
from executorch.exir import EdgeCompileConfig

print("Loading model...", flush=True)
model = load_model()

print("Exporting...", flush=True)
programs, metadata = export_all_ggml(model)
ep = programs["encoder"]

# Print all nodes around the PE slice
print("\n=== PE slice and its consumers ===", flush=True)
pe_slice_node = None
for node in ep.graph.nodes:
    if node.op == "call_function":
        name = str(node.target)
        if "slice" in name:
            args_str = [str(a)[:50] if isinstance(a, torch.fx.Node) else str(a)[:50] for a in node.args]
            fv = node.meta.get("val")
            shape = list(fv.shape) if fv is not None and hasattr(fv, 'shape') else "?"
            print(f"  SLICE: {node.name} = {name}({', '.join(args_str)}) -> {shape}", flush=True)
            # Check if this is the PE slice (large output on dim 1)
            if fv is not None and hasattr(fv, 'shape') and len(fv.shape) == 3:
                if fv.shape[2] == 1024 and "pe" in str(node.args[0]):
                    pe_slice_node = node

        if "dropout" in name:
            args_str = [str(a)[:50] for a in node.args]
            fv = node.meta.get("val")
            shape = list(fv.shape) if fv is not None and hasattr(fv, 'shape') else "?"
            print(f"  DROPOUT: {node.name} = {name}({', '.join(args_str)}) -> {shape}", flush=True)

# Trace from PE slice to its consumers
if pe_slice_node:
    print(f"\nPE slice node: {pe_slice_node.name}", flush=True)
    print(f"  Users:", flush=True)
    for user in pe_slice_node.users:
        user_name = str(user.target) if user.op == "call_function" else user.op
        print(f"    {user.name}: {user_name}", flush=True)
        # One more level
        for uu in user.users:
            uu_name = str(uu.target) if uu.op == "call_function" else uu.op
            print(f"      -> {uu.name}: {uu_name}", flush=True)
else:
    # Find all slices of the PE buffer
    print("\nSearching for PE-related slices...", flush=True)
    for node in ep.graph.nodes:
        if node.op == "call_function" and "slice" in str(node.target):
            for arg in node.args:
                if isinstance(arg, torch.fx.Node) and "pe" in arg.name:
                    pe_slice_node = node
                    print(f"  Found: {node.name} slicing {arg.name}", flush=True)
                    print(f"  Users:", flush=True)
                    for user in node.users:
                        user_name = str(user.target) if user.op == "call_function" else user.op
                        print(f"    {user.name}: {user_name}", flush=True)

# Now lower and check partition structure
print("\n=== Lowering (checking partition count) ===", flush=True)
edge_mgr = to_edge_rewrite_and_lower(
    ep,
    transform_passes=[ReplaceCopyOpsPass()],
    partitioner=[GgmlPartitioner()],
    compile_config=EdgeCompileConfig(_check_ir_validity=False, _skip_dim_order=True),
)
# Print the lowered graph structure
gm = edge_mgr.exported_program().graph_module
delegate_count = 0
for node in gm.graph.nodes:
    if node.op == "get_attr" and "lowered_module" in node.name:
        delegate_count += 1
        print(f"  Delegate: {node.name}", flush=True)
    elif node.op == "call_function" and "executorch_call_delegate" in str(node.target):
        print(f"  Call delegate: {node.name} args={[str(a)[:30] for a in node.args]}", flush=True)
print(f"\nTotal delegates: {delegate_count}", flush=True)
