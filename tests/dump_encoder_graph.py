#!/usr/bin/env python3
"""Dump the encoder edge graph to see delegation structure and non-delegated ops."""
import os, sys
import torch

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

# Count ops before lowering
print(f"\n=== Exported graph stats ===", flush=True)
op_counts = {}
for node in ep.graph.nodes:
    if node.op == "call_function":
        name = str(node.target).split(".")[-1] if "." in str(node.target) else str(node.target)
        op_counts[name] = op_counts.get(name, 0) + 1
for name, count in sorted(op_counts.items(), key=lambda x: -x[1])[:20]:
    print(f"  {name}: {count}", flush=True)

print("\nLowering...", flush=True)
edge_mgr = to_edge_rewrite_and_lower(
    ep,
    transform_passes=[ReplaceCopyOpsPass()],
    partitioner=[GgmlPartitioner()],
    compile_config=EdgeCompileConfig(_check_ir_validity=False, _skip_dim_order=True),
)

gm = edge_mgr.exported_program().graph_module
print(f"\n=== Edge graph after partitioning ===", flush=True)

# Count delegates and non-delegated ops
n_delegates = 0
n_non_delegated = 0
non_delegated_ops = {}
delegate_info = []

for node in gm.graph.nodes:
    if node.op == "get_attr" and "lowered_module" in node.name:
        n_delegates += 1
    elif node.op == "call_function":
        target = str(node.target)
        if "executorch_call_delegate" in target:
            # Count inputs/outputs
            args = node.args
            n_inputs = len([a for a in args[1:] if isinstance(a, torch.fx.Node)])
            delegate_info.append((node.name, n_inputs))
        else:
            n_non_delegated += 1
            short = target.split("::")[-1] if "::" in target else target.split(".")[-1]
            fv = node.meta.get("val")
            shape = list(fv.shape) if fv is not None and hasattr(fv, "shape") else "?"
            dtype = str(fv.dtype).split(".")[-1] if fv is not None and hasattr(fv, "dtype") else "?"
            non_delegated_ops[short] = non_delegated_ops.get(short, 0) + 1
            # Print details for first few of each type
            if non_delegated_ops[short] <= 3:
                src_names = [str(a.name) if isinstance(a, torch.fx.Node) else str(a)[:30] for a in node.args[:4]]
                print(f"  NON-DELEGATED: {node.name} = {short}({', '.join(src_names)}) -> {shape} {dtype}", flush=True)

print(f"\n=== Summary ===", flush=True)
print(f"  Delegates: {n_delegates}", flush=True)
for name, n_in in delegate_info:
    print(f"    {name}: {n_in} inputs", flush=True)
print(f"  Non-delegated ops: {n_non_delegated}", flush=True)
for name, count in sorted(non_delegated_ops.items(), key=lambda x: -x[1]):
    print(f"    {name}: {count}", flush=True)

# Print the full graph structure (compact)
print(f"\n=== Full edge graph (compact) ===", flush=True)
for node in gm.graph.nodes:
    if node.op == "placeholder":
        fv = node.meta.get("val")
        shape = list(fv.shape) if fv is not None and hasattr(fv, "shape") else "?"
        print(f"  INPUT {node.name}: {shape}", flush=True)
    elif node.op == "get_attr" and "lowered_module" in node.name:
        print(f"  DELEGATE {node.name}", flush=True)
    elif node.op == "call_function":
        target = str(node.target)
        if "executorch_call_delegate" in target:
            args_str = [a.name if isinstance(a, torch.fx.Node) else "?" for a in node.args]
            fv = node.meta.get("val")
            if isinstance(fv, (list, tuple)):
                shapes = [list(v.shape) if hasattr(v, "shape") else "?" for v in fv]
            elif hasattr(fv, "shape"):
                shapes = [list(fv.shape)]
            else:
                shapes = ["?"]
            print(f"  CALL_DELEGATE {node.name}({', '.join(args_str[1:])}) -> {shapes}", flush=True)
        else:
            short = target.split("::")[-1] if "::" in target else target.split(".")[-1]
            args_str = [a.name if isinstance(a, torch.fx.Node) else str(a)[:20] for a in node.args[:4]]
            fv = node.meta.get("val")
            shape = list(fv.shape) if fv is not None and hasattr(fv, "shape") else "?"
            print(f"  {short}({', '.join(args_str)}) -> {shape}", flush=True)
    elif node.op == "output":
        print(f"  OUTPUT", flush=True)
