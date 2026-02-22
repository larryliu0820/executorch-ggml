"""BatchNorm folding *rewrite* pass for torch.export ExportedProgram.

This is rewrites the exported FX graph so BatchNorm inference ops disappear 
(and thus do not appear inside delegated subgraphs).

What it does
------------
For each detected pattern:
  conv -> (_native_)batch_norm(inference) -> getitem(0)

We:
  1) compute folded (W', b')
  2) overwrite the conv weight parameter in state_dict with W'
  3) ensure the conv has a bias tensor:
       - if it already has a bias placeholder, overwrite it with b'
       - if bias is None, we add a new *buffer* placeholder and wire it in
  4) replace uses of getitem(0) (or BN directly if no getitem) with the conv
     output and remove the BN/getitem nodes.

Notes
-----
- We intentionally do *not* prune unused BN placeholders (gamma/beta/running
  stats) from graph_signature/state_dict yet. They become dead inputs but are
  harmless; pruning can be added later.
- We only fold inference BN; training=True BN is skipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch.export import ExportedProgram
from torch.export.graph_signature import ExportGraphSignature, InputKind, InputSpec, TensorArgument

from executorch_ggml.passes.bn_folding_pass import (
    BatchNormFoldingPass,
    ConvBnPattern,
    find_conv_bn_patterns,
)


@dataclass(frozen=True)
class RewriteResult:
    ep: ExportedProgram
    num_patterns: int
    num_folded: int


def _make_unique_placeholder_name(g: torch.fx.Graph, base: str) -> str:
    existing = {n.name for n in g.nodes}
    if base not in existing:
        return base
    i = 0
    while f"{base}_{i}" in existing:
        i += 1
    return f"{base}_{i}"


def _append_buffer_input_spec(sig: ExportGraphSignature, ph_name: str, fqn: str) -> ExportGraphSignature:
    # We append a BUFFER spec; this does not change the user-facing input tree
    # spec, but does extend the lifted inputs that the ExportedProgram injects
    # from its state_dict.
    new_specs = list(sig.input_specs)
    new_specs.append(
        InputSpec(
            kind=InputKind.BUFFER,
            arg=TensorArgument(name=ph_name),
            target=fqn,
            persistent=True,
        )
    )
    return ExportGraphSignature(input_specs=new_specs, output_specs=list(sig.output_specs))


class BatchNormFoldingRewritePass:
    def __init__(self, *, bias_buffer_prefix: str = "__etggml_bn_folded_bias"):
        self.bias_buffer_prefix = bias_buffer_prefix

    def run(self, ep: ExportedProgram) -> RewriteResult:
        patterns: list[ConvBnPattern] = find_conv_bn_patterns(ep)
        if not patterns:
            return RewriteResult(ep=ep, num_patterns=0, num_folded=0)

        folded = BatchNormFoldingPass().run(ep)  # conv_node_name -> FoldedConvParams

        gm = ep.graph_module
        g = gm.graph

        # Copy state_dict so we can update tensors.
        new_state_dict = dict(ep.state_dict)

        # Need these mappings to resolve placeholder name -> FQN.
        sig = ep.graph_signature
        param_map = dict(sig.inputs_to_parameters)
        buffer_map = dict(sig.inputs_to_buffers)

        def resolve_fqn(ph_name: str) -> Optional[str]:
            return param_map.get(ph_name) or buffer_map.get(ph_name)

        num_folded = 0

        # Map old output node names (bn/getitem) -> new (conv) for signature fixups.
        out_name_remap: Dict[str, str] = {}

        # Helper: find node by name quickly.
        name_to_node: Dict[str, torch.fx.Node] = {n.name: n for n in g.nodes}

        # We will mutate the graph; use a stable list of patterns.
        for pat in patterns:
            if pat.conv_node_name not in folded:
                continue

            conv = name_to_node.get(pat.conv_node_name)
            bn = name_to_node.get(pat.bn_node_name)
            getitem = name_to_node.get(pat.getitem_node_name) if pat.getitem_node_name else None
            if conv is None or bn is None:
                continue

            # If BN's auxiliary outputs are used, we cannot erase it safely.
            # Common cases:
            # - _native_batch_norm_legit_no_training returns a tuple -> followed by getitem(0)
            # - aten.batch_norm (inference) returns a tensor directly -> no getitem
            if getitem is None:
                # Only safe to fold if BN returns a tensor (aten.batch_norm inference)
                # and none of its other tuple-like outputs exist.
                if "aten.batch_norm" not in str(bn.target):
                    continue

            f = folded[pat.conv_node_name]

            # 1) Overwrite conv weight.
            conv_w_fqn = resolve_fqn(pat.conv_weight_ph)
            if conv_w_fqn is None:
                continue
            # Preserve Parameter-ness for parameters.
            old_w = new_state_dict[conv_w_fqn]
            new_w = f.weight.detach().cpu().contiguous()
            if isinstance(old_w, torch.nn.Parameter):
                new_state_dict[conv_w_fqn] = torch.nn.Parameter(new_w, requires_grad=old_w.requires_grad)
            else:
                new_state_dict[conv_w_fqn] = new_w

            # 2) Bias handling.
            # conv args: (input, weight, bias?, ...)
            if len(conv.args) < 3:
                continue

            if pat.conv_bias_ph is not None:
                conv_b_fqn = resolve_fqn(pat.conv_bias_ph)
                if conv_b_fqn is None:
                    continue
                old_b = new_state_dict[conv_b_fqn]
                new_b = f.bias.detach().cpu().contiguous()
                if isinstance(old_b, torch.nn.Parameter):
                    new_state_dict[conv_b_fqn] = torch.nn.Parameter(new_b, requires_grad=old_b.requires_grad)
                else:
                    new_state_dict[conv_b_fqn] = new_b
            else:
                # Create a new buffer placeholder and wire it into conv.
                # Add a new placeholder at the end of the placeholder block.
                ph_base = f"{pat.conv_node_name}_folded_bias"
                ph_name = _make_unique_placeholder_name(g, ph_base)

                # Insert new placeholder after existing placeholders.
                placeholders = [n for n in g.nodes if n.op == "placeholder"]
                insert_before = None
                if placeholders:
                    # Insert *after* the last placeholder. FX Graph only supports
                    # inserting before a node, so we insert before the first
                    # non-placeholder (or output if all placeholders).
                    for n in g.nodes:
                        if n.op != "placeholder":
                            insert_before = n
                            break
                with g.inserting_before(insert_before):
                    bias_ph = g.placeholder(ph_name)

                # Attach meta info if available (helps later passes).
                bias_ph.meta["val"] = torch.empty_like(f.bias)

                # Update conv args to use new placeholder at index 2.
                new_args = list(conv.args)
                new_args[2] = bias_ph
                conv.args = tuple(new_args)

                # Add the buffer to the graph_signature + state_dict.
                bias_fqn = f"{self.bias_buffer_prefix}.{ph_name}"
                new_state_dict[bias_fqn] = f.bias.detach().cpu().contiguous()
                sig = _append_buffer_input_spec(sig, ph_name, bias_fqn)

                # Update lookup map for subsequent patterns.
                name_to_node[ph_name] = bias_ph

            # 3) Rewrite: bypass BN.
            if getitem is not None:
                # tuple-returning BN case
                out_name_remap[getitem.name] = conv.name
                out_name_remap[bn.name] = conv.name
                getitem.replace_all_uses_with(conv)
                g.erase_node(getitem)
            else:
                # tensor-returning BN case
                out_name_remap[bn.name] = conv.name
                bn.replace_all_uses_with(conv)

            # Now erase BN node.
            g.erase_node(bn)

            num_folded += 1

        # Fix up graph signature outputs if we rewired the graph output.
        if out_name_remap:
            from torch.export.graph_signature import OutputSpec, TensorArgument

            new_out_specs = []
            for os in sig.output_specs:
                if isinstance(os.arg, TensorArgument) and os.arg.name in out_name_remap:
                    new_out_specs.append(OutputSpec(kind=os.kind, arg=TensorArgument(out_name_remap[os.arg.name]), target=os.target))
                else:
                    new_out_specs.append(os)
            sig = ExportGraphSignature(input_specs=list(sig.input_specs), output_specs=new_out_specs)

        g.lint()
        gm.recompile()

        new_ep = ep._update(gm, sig, state_dict=new_state_dict)
        return RewriteResult(ep=new_ep, num_patterns=len(patterns), num_folded=num_folded)
