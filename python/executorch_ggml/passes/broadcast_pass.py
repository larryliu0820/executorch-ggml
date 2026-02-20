"""Broadcast canonicalization pass.

ExecuTorch Edge graphs often rely on PyTorch broadcasting semantics for binary
ops like `aten.add.Tensor`. ggml's broadcasting support is more limited, and our
runtime currently expects explicit shape alignment.

This pass rewrites certain broadcasted adds into:
  add(a, expand_copy(permute_copy(b), a.shape))

Currently focused on the Qwen3 failure mode where a 1D (or effectively-1D)
vector is broadcast along a non-trailing dimension.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch


@dataclass
class BroadcastCanonicalizationPass:
    """Rewrite broadcasted `aten.add.Tensor` into explicit permute+expand."""

    enabled: bool = True

    def _find_edge_target(self, ep: torch.export.ExportedProgram, opname: str):
        # Reuse existing EdgeOpOverload object when possible.
        for n in ep.graph_module.graph.nodes:
            if n.op == "call_function" and opname in str(n.target):
                return n.target
        # Fallback to ATen op; Edge conversion may fix this if run earlier.
        return getattr(torch.ops.aten, opname.split("aten.", 1)[1].split(".")[0], None)

    def _get_shape(self, n: torch.fx.Node) -> Optional[Sequence[int]]:
        v = n.meta.get("val")
        if v is None or not hasattr(v, "shape"):
            return None
        return list(v.shape)

    def run(self, ep: torch.export.ExportedProgram):
        if not self.enabled:
            return ep

        gm = ep.graph_module
        g = gm.graph

        add_name = "aten.add.Tensor"
        permute_name = "aten.permute_copy.default"
        expand_name = "aten.expand_copy.default"

        permute_tgt = None
        expand_tgt = None

        def get_perm_tgt():
            nonlocal permute_tgt
            if permute_tgt is None:
                # Edge graphs usually have permute_copy somewhere; reuse it.
                for n in g.nodes:
                    if n.op == "call_function" and permute_name in str(n.target):
                        permute_tgt = n.target
                        break
                if permute_tgt is None:
                    permute_tgt = torch.ops.aten.permute_copy.default
            return permute_tgt

        def get_expand_tgt():
            nonlocal expand_tgt
            if expand_tgt is None:
                for n in g.nodes:
                    if n.op == "call_function" and expand_name in str(n.target):
                        expand_tgt = n.target
                        break
                if expand_tgt is None:
                    expand_tgt = torch.ops.aten.expand_copy.default
            return expand_tgt

        changed = False

        for n in list(g.nodes):
            if n.op != "call_function" or add_name not in str(n.target):
                continue

            a, b = n.args[0], n.args[1]
            if not isinstance(a, torch.fx.Node) or not isinstance(b, torch.fx.Node):
                continue

            a_shape = self._get_shape(a)
            b_shape = self._get_shape(b)
            out_shape = self._get_shape(n)
            if a_shape is None or b_shape is None or out_shape is None:
                continue

            if a_shape == b_shape:
                continue

            # Only handle equal-rank tensors.
            if len(a_shape) != len(b_shape):
                continue

            # Detect "effectively 1D" b: exactly one dim > 1.
            b_non1 = [i for i, s in enumerate(b_shape) if s != 1]
            if len(b_non1) != 1:
                continue

            non1_dim = b_non1[0]
            nsize = b_shape[non1_dim]

            # Find a dim in a with the same extent.
            match_dims = [i for i, s in enumerate(a_shape) if s == nsize]
            if not match_dims:
                continue

            # If already aligned, expand is enough (handled elsewhere).
            # If not aligned, permute b to align the non-1 dim to the first matching dim.
            match_dim = match_dims[0]
            if non1_dim == match_dim:
                continue

            perm = list(range(len(b_shape)))
            perm[non1_dim], perm[match_dim] = perm[match_dim], perm[non1_dim]

            with g.inserting_before(n):
                b_perm = g.call_function(get_perm_tgt(), args=(b, perm))
                b_perm.meta["val"] = b.meta.get("val")  # best-effort

                b_exp = g.call_function(get_expand_tgt(), args=(b_perm, out_shape, False))
                b_exp.meta["val"] = n.meta.get("val")  # best-effort

                n.replace_input_with(b, b_exp)
                changed = True

        if changed:
            g.lint()
            gm.recompile()
        return ep
