"""Broadcast canonicalization pass.

ExecuTorch Edge graphs often rely on PyTorch broadcasting semantics for binary
ops like `aten.add.Tensor` and `aten.mul.Tensor`. ggml's broadcasting support
is more limited, and our runtime currently expects explicit shape alignment.

This pass rewrites broadcasted binary ops to make broadcasts explicit:
  add(a, b) -> add(expand_copy(a, out_shape), expand_copy(b, out_shape))

For tensors that need permutation before expanding (1D-like broadcast along
non-trailing dimension), it also inserts permute_copy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from executorch.exir.pass_base import PassResult


@dataclass
class BroadcastCanonicalizationPass:
    """Rewrite broadcasted binary ops into explicit expand_copy."""

    enabled: bool = True

    def __call__(self, graph_module: torch.fx.GraphModule) -> PassResult:
        """Support use as a transform pass (GraphModule-level)."""
        # For transform passes, we can't easily run the full rewrite
        # because we need access to the ExportedProgram for op lookups.
        # Return unchanged for now - the rewrite happens via run() at ep level
        return PassResult(graph_module, False)

    def _get_shape(self, n: torch.fx.Node) -> Optional[Sequence[int]]:
        v = n.meta.get("val")
        if v is None or not hasattr(v, "shape"):
            return None
        return list(v.shape)

    def _needs_broadcast(self, shape: Sequence[int], out_shape: Sequence[int]) -> bool:
        """Check if shape needs broadcasting to match out_shape."""
        if list(shape) == list(out_shape):
            return False
        return True

    def _can_simple_expand(self, shape: Sequence[int], out_shape: Sequence[int]) -> bool:
        """Check if shape can be expanded to out_shape via simple repeat.

        Handles PyTorch broadcasting rules:
        - If input has fewer dims, prepend 1s to match output rank
        - Then each dim must be 1 or equal to output dim
        """
        # Pad shape with leading 1s to match output rank
        padded_shape = [1] * (len(out_shape) - len(shape)) + list(shape)

        for s, o in zip(padded_shape, out_shape):
            if s != 1 and s != o:
                return False
        return True

    def _compute_permute_for_1d_broadcast(
        self, shape: Sequence[int], out_shape: Sequence[int]
    ) -> Optional[list[int]]:
        """For 1D-like tensors, compute permutation to align with out_shape.

        Returns permutation list or None if not applicable.
        """
        if len(shape) != len(out_shape):
            return None

        # Find dims with size > 1 in the source
        non1_dims = [i for i, s in enumerate(shape) if s != 1]
        if len(non1_dims) != 1:
            return None

        non1_dim = non1_dims[0]
        nsize = shape[non1_dim]

        # Find matching dim in output
        match_dims = [i for i, s in enumerate(out_shape) if s == nsize]
        if not match_dims:
            return None

        match_dim = match_dims[0]
        if non1_dim == match_dim:
            # Already aligned
            return None

        # Create permutation that swaps non1_dim with match_dim
        perm = list(range(len(shape)))
        perm[non1_dim], perm[match_dim] = perm[match_dim], perm[non1_dim]
        return perm

    def run(self, ep: torch.export.ExportedProgram):
        if not self.enabled:
            return ep

        gm = ep.graph_module
        g = gm.graph

        # Binary ops to handle
        binary_ops = ["aten.add.Tensor", "aten.mul.Tensor", "aten.sub.Tensor"]
        expand_name = "aten.expand_copy.default"
        view_name = "aten.view.default"
        permute_name = "aten.permute_copy.default"

        expand_tgt = None
        view_tgt = None
        permute_tgt = None

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

        def get_view_tgt():
            nonlocal view_tgt
            if view_tgt is None:
                for n in g.nodes:
                    if n.op == "call_function" and view_name in str(n.target):
                        view_tgt = n.target
                        break
                if view_tgt is None:
                    view_tgt = torch.ops.aten.view.default
            return view_tgt

        def get_perm_tgt():
            nonlocal permute_tgt
            if permute_tgt is None:
                for n in g.nodes:
                    if n.op == "call_function" and permute_name in str(n.target):
                        permute_tgt = n.target
                        break
                if permute_tgt is None:
                    permute_tgt = torch.ops.aten.permute_copy.default
            return permute_tgt

        changed = False

        for n in list(g.nodes):
            if n.op != "call_function":
                continue

            target_str = str(n.target)
            if not any(op in target_str for op in binary_ops):
                continue

            a, b = n.args[0], n.args[1]
            if not isinstance(a, torch.fx.Node) or not isinstance(b, torch.fx.Node):
                continue

            a_shape = self._get_shape(a)
            b_shape = self._get_shape(b)
            out_shape = self._get_shape(n)
            if a_shape is None or b_shape is None or out_shape is None:
                continue

            # Skip if both inputs already match output shape
            if list(a_shape) == list(out_shape) and list(b_shape) == list(out_shape):
                continue

            # Handle each input
            new_a = a
            new_b = b

            with g.inserting_before(n):
                # Handle input a
                if self._needs_broadcast(a_shape, out_shape):
                    if self._can_simple_expand(a_shape, out_shape):
                        # Need to first reshape to match output rank if ranks differ
                        if len(a_shape) != len(out_shape):
                            # Insert view to pad with leading 1s
                            padded_shape = [1] * (len(out_shape) - len(a_shape)) + list(a_shape)
                            a_view = g.call_function(get_view_tgt(), args=(a, padded_shape))
                            if a.meta.get("val") is not None:
                                a_view.meta["val"] = a.meta["val"].view(*padded_shape)
                            a_exp = g.call_function(get_expand_tgt(), args=(a_view, list(out_shape)))
                        else:
                            a_exp = g.call_function(get_expand_tgt(), args=(a, list(out_shape)))
                        if n.meta.get("val") is not None:
                            a_exp.meta["val"] = n.meta["val"].clone()
                        new_a = a_exp
                        changed = True
                    elif len(a_shape) == len(out_shape):
                        # Try permute for 1D-like broadcast
                        perm = self._compute_permute_for_1d_broadcast(a_shape, out_shape)
                        if perm is not None:
                            a_perm = g.call_function(get_perm_tgt(), args=(a, perm))
                            if a.meta.get("val") is not None:
                                a_perm.meta["val"] = a.meta["val"].permute(*perm)
                            a_exp = g.call_function(get_expand_tgt(), args=(a_perm, list(out_shape)))
                            if n.meta.get("val") is not None:
                                a_exp.meta["val"] = n.meta["val"].clone()
                            new_a = a_exp
                            changed = True

                # Handle input b
                if self._needs_broadcast(b_shape, out_shape):
                    if self._can_simple_expand(b_shape, out_shape):
                        # Need to first reshape to match output rank if ranks differ
                        if len(b_shape) != len(out_shape):
                            # Insert view to pad with leading 1s
                            padded_shape = [1] * (len(out_shape) - len(b_shape)) + list(b_shape)
                            b_view = g.call_function(get_view_tgt(), args=(b, padded_shape))
                            if b.meta.get("val") is not None:
                                b_view.meta["val"] = b.meta["val"].view(*padded_shape)
                            b_exp = g.call_function(get_expand_tgt(), args=(b_view, list(out_shape)))
                        else:
                            b_exp = g.call_function(get_expand_tgt(), args=(b, list(out_shape)))
                        if n.meta.get("val") is not None:
                            b_exp.meta["val"] = n.meta["val"].clone()
                        new_b = b_exp
                        changed = True
                    elif len(b_shape) == len(out_shape):
                        # Try permute for 1D-like broadcast
                        perm = self._compute_permute_for_1d_broadcast(b_shape, out_shape)
                        if perm is not None:
                            b_perm = g.call_function(get_perm_tgt(), args=(b, perm))
                            if b.meta.get("val") is not None:
                                b_perm.meta["val"] = b.meta["val"].permute(*perm)
                            b_exp = g.call_function(get_expand_tgt(), args=(b_perm, list(out_shape)))
                            if n.meta.get("val") is not None:
                                b_exp.meta["val"] = n.meta["val"].clone()
                            new_b = b_exp
                            changed = True

            # Replace inputs if changed
            if new_a is not a:
                n.replace_input_with(a, new_a)
            if new_b is not b:
                n.replace_input_with(b, new_b)

        if changed:
            g.lint()
            gm.recompile()
        return ep
