"""BatchNorm folding pass (analysis + parameter folding).

This module provides a *pass-like* API that can:
- Detect Conv2d -> BatchNorm(inference) -> getitem(0) patterns in an ExportedProgram
- Compute folded conv weights/bias using running stats and affine params

It does **not** rewrite the FX graph yet. Instead it returns the folded
parameters so a backend lowering pipeline can fuse BN away.

Why: graph rewriting of torch.export ExportedProgram requires updating
GraphSignature + state_dict mappings. For now, we keep folding in the backend
lowering step but expose it as a reusable pass/module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch.export import ExportedProgram

from executorch_ggml.bn_folding import fold_conv_bn_weights


@dataclass(frozen=True)
class ConvBnPattern:
    conv_node_name: str
    bn_node_name: str
    getitem_node_name: Optional[str]

    conv_weight_ph: str
    conv_bias_ph: Optional[str]

    bn_weight_ph: Optional[str]
    bn_bias_ph: Optional[str]
    bn_running_mean_ph: str
    bn_running_var_ph: str

    eps: float


def _is_conv_target(t) -> bool:
    s = str(t)
    return ("aten.convolution.default" in s) or ("aten.conv2d.default" in s)


def _is_bn_infer_target(t) -> bool:
    s = str(t)
    return (
        "aten._native_batch_norm_legit_no_training.default" in s
        or "aten.batch_norm.default" in s
    )


def _is_getitem_target(t) -> bool:
    return "<built-in function getitem>" in str(t)


def find_conv_bn_patterns(ep: ExportedProgram) -> list[ConvBnPattern]:
    """Scan ep.graph_module.graph for conv -> bn -> getitem(0) patterns."""

    g = ep.graph_module.graph
    nodes = list(g.nodes)

    patterns: list[ConvBnPattern] = []

    for i, n in enumerate(nodes):
        if n.op != "call_function" or not _is_conv_target(n.target):
            continue

        # conv args per aten.convolution schema
        # (input, weight, bias?, stride, padding, dilation, transposed, output_padding, groups)
        if len(n.args) < 3:
            continue

        conv_w = n.args[1]
        conv_b = n.args[2]
        if not isinstance(conv_w, torch.fx.Node):
            continue

        conv_bias_ph = conv_b.name if isinstance(conv_b, torch.fx.Node) else None

        # Find immediate bn user (common in exported graphs)
        bn_node = None
        for u in n.users:
            if u.op == "call_function" and _is_bn_infer_target(u.target):
                bn_node = u
                break
        if bn_node is None:
            continue

        # bn args variants:
        # - _native_batch_norm_legit_no_training(input, weight, bias, running_mean, running_var, momentum, eps)
        # - batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled)
        if "_native_batch_norm_legit_no_training" in str(bn_node.target):
            if len(bn_node.args) < 7:
                continue
            bn_w = bn_node.args[1]
            bn_b = bn_node.args[2]
            rm = bn_node.args[3]
            rv = bn_node.args[4]
            eps = float(bn_node.args[6])
        else:
            if len(bn_node.args) < 9:
                continue
            bn_w = bn_node.args[1]
            bn_b = bn_node.args[2]
            rm = bn_node.args[3]
            rv = bn_node.args[4]
            training = bool(bn_node.args[5])
            if training:
                continue  # only fold inference BN
            eps = float(bn_node.args[7])

        if not isinstance(rm, torch.fx.Node) or not isinstance(rv, torch.fx.Node):
            continue

        bn_weight_ph = bn_w.name if isinstance(bn_w, torch.fx.Node) else None
        bn_bias_ph = bn_b.name if isinstance(bn_b, torch.fx.Node) else None

        # getitem(0) is often used to extract output tensor from BN tuple
        getitem_node_name: Optional[str] = None
        for u in bn_node.users:
            if u.op == "call_function" and _is_getitem_target(u.target):
                # Ensure it's index 0
                if len(u.args) >= 2 and u.args[1] == 0:
                    getitem_node_name = u.name
                    break

        patterns.append(
            ConvBnPattern(
                conv_node_name=n.name,
                bn_node_name=bn_node.name,
                getitem_node_name=getitem_node_name,
                conv_weight_ph=conv_w.name,
                conv_bias_ph=conv_bias_ph,
                bn_weight_ph=bn_weight_ph,
                bn_bias_ph=bn_bias_ph,
                bn_running_mean_ph=rm.name,
                bn_running_var_ph=rv.name,
                eps=eps,
            )
        )

    return patterns


@dataclass(frozen=True)
class FoldedConvParams:
    weight: torch.Tensor
    bias: torch.Tensor


def fold_params_for_pattern(ep: ExportedProgram, pat: ConvBnPattern) -> FoldedConvParams:
    """Compute folded conv weights/bias for a detected pattern."""

    sig = ep.graph_signature
    param_map = dict(sig.inputs_to_parameters)
    buffer_map = dict(sig.inputs_to_buffers)

    def resolve(ph_name: str) -> torch.Tensor:
        fqn = param_map.get(ph_name) or buffer_map.get(ph_name)
        if fqn is None:
            raise KeyError(f"placeholder {ph_name} not found in graph_signature")
        return ep.state_dict[fqn]

    conv_w = resolve(pat.conv_weight_ph)
    conv_b = resolve(pat.conv_bias_ph) if pat.conv_bias_ph else None

    bn_w = resolve(pat.bn_weight_ph) if pat.bn_weight_ph else None
    bn_b = resolve(pat.bn_bias_ph) if pat.bn_bias_ph else None
    rm = resolve(pat.bn_running_mean_ph)
    rv = resolve(pat.bn_running_var_ph)

    w_fold, b_fold = fold_conv_bn_weights(
        conv_w=conv_w,
        conv_b=conv_b,
        bn_weight=bn_w,
        bn_bias=bn_b,
        running_mean=rm,
        running_var=rv,
        eps=pat.eps,
    )

    return FoldedConvParams(weight=w_fold, bias=b_fold)


class BatchNormFoldingPass:
    """Pass wrapper."""

    def __init__(self):
        pass

    def run(self, ep: ExportedProgram) -> dict[str, FoldedConvParams]:
        """Return mapping conv_node_name -> folded params."""
        out: dict[str, FoldedConvParams] = {}
        for pat in find_conv_bn_patterns(ep):
            out[pat.conv_node_name] = fold_params_for_pattern(ep, pat)
        return out
