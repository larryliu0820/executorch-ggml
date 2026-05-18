"""Phase 0 sanity probe for `torch.cond` export shape.

Builds a minimal `nn.Module` mirroring the whisper cross-attention KV
cache pattern and inspects what `torch.export` produces. Drives the
plan in needle.md / Phase 0:

  Q1. What does the top-level graph look like?
  Q2. Are subgraphs `GraphModule` attributes? What FQNs?
  Q3. Do lifted parent buffers appear as placeholders in subgraphs?
  Q4. Does edge IR preserve the cond node, or decompose it?
  Q5. Does `auto_functionalized` wrap the in-place buffer write inside
       the recompute branch?
  Q6. What dtype/shape is the predicate node?

Run:
    python -m pytest tests/test_cond_export_shape.py -s
"""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower


class _MiniWhisperCrossAttn(nn.Module):
    """Single-layer cross-attention with whisper-style cache+cond.

    Buffers:
      cache_k, cache_v: per-layer KV cache, shape [1, H, S_max, D].
      cache_initialized: scalar bool flag, False on first call.

    forward(q_input, encoder_out) returns a [1, T_q, D] tensor.
    """

    def __init__(self, dim: int = 16, n_heads: int = 2, head_dim: int = 8, s_max: int = 32):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.k_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.register_buffer("cache_k", torch.zeros(1, n_heads, s_max, head_dim))
        self.register_buffer("cache_v", torch.zeros(1, n_heads, s_max, head_dim))
        self.register_buffer("cache_initialized", torch.zeros(1, dtype=torch.bool))

    def forward(self, q_input: torch.Tensor, encoder_out: torch.Tensor) -> torch.Tensor:
        B, T_q, _ = q_input.shape
        q = self.q_proj(q_input).view(B, T_q, self.n_heads, self.head_dim).transpose(1, 2)

        def use_cache(cached_k, cached_v, enc):
            return cached_k.clone(), cached_v.clone()

        def recompute(cached_k, cached_v, enc):
            T_kv = enc.shape[1]
            k = self.k_proj(enc).view(B, T_kv, self.n_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(enc).view(B, T_kv, self.n_heads, self.head_dim).transpose(1, 2)
            # Whisper uses torch.ops.executorch.update_cross_attn_cache here;
            # for the shape probe we just emit `cached_k[..., :T_kv, :] = k`.
            new_k = cached_k.clone()
            new_v = cached_v.clone()
            new_k[:, :, :T_kv, :] = k
            new_v[:, :, :T_kv, :] = v
            return new_k, new_v

        k, v = torch.cond(
            self.cache_initialized.any(),
            use_cache,
            recompute,
            (self.cache_k, self.cache_v, encoder_out),
        )
        # Naive attention to keep the graph small.
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, T_q, self.dim)
        return out


class TestCondExportShape(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.module = _MiniWhisperCrossAttn().eval()
        cls.q_input = torch.randn(1, 4, 16)
        cls.encoder_out = torch.randn(1, 8, 16)
        cls.ep = torch.export.export(cls.module, (cls.q_input, cls.encoder_out), strict=False)

    def test_q1_top_level_has_cond_node(self):
        cond_nodes = [
            n for n in self.ep.graph_module.graph.nodes
            if n.op == "call_function" and n.target == torch.ops.higher_order.cond
        ]
        self.assertEqual(len(cond_nodes), 1, "expected one cond node")
        cond = cond_nodes[0]
        # cond.args = (predicate, true_graph, false_graph, operands)
        self.assertEqual(len(cond.args), 4)
        print(f"\n[Q1] cond node: {cond.name}, args[0]={cond.args[0]}, "
              f"args[1]={cond.args[1]}, args[2]={cond.args[2]}, "
              f"args[3] (operands): {cond.args[3]}")

    def test_q2_subgraphs_are_graphmodule_attrs(self):
        attrs = list(self.ep.graph_module.named_children())
        attr_names = [n for n, _ in attrs]
        print(f"\n[Q2] graph_module children: {attr_names}")
        # Expect at least true_graph_0 and false_graph_0
        self.assertTrue(any("true_graph" in n for n in attr_names))
        self.assertTrue(any("false_graph" in n for n in attr_names))
        for name, sub in attrs:
            self.assertIsInstance(sub, torch.fx.GraphModule)
            print(f"\n[Q2] {name} graph:")
            print(sub.graph)

    def test_q3_subgraph_placeholders_share_parent_fqns(self):
        sig = self.ep.graph_signature
        parent_buffer_targets = {
            spec.arg.name: spec.target
            for spec in sig.input_specs
            if spec.kind.name == "BUFFER"
        }
        print(f"\n[Q3] parent buffer placeholders -> FQN: {parent_buffer_targets}")

        # For each subgraph, list its placeholder names + try to match by
        # FQN to parent buffers via the cond_operand position.
        cond = next(
            n for n in self.ep.graph_module.graph.nodes
            if n.op == "call_function" and n.target == torch.ops.higher_order.cond
        )
        operand_nodes = cond.args[3]
        operand_names = [n.name if isinstance(n, torch.fx.Node) else str(n) for n in operand_nodes]
        print(f"[Q3] cond operand parent-graph names: {operand_names}")

        for name, sub in self.ep.graph_module.named_children():
            if "_graph" not in name:
                continue
            placeholders = [n.name for n in sub.graph.nodes if n.op == "placeholder"]
            print(f"[Q3] {name} placeholders: {placeholders}")
            # Subgraph placeholders bind 1:1 to operand_names by position.
            self.assertEqual(len(placeholders), len(operand_names),
                             f"{name} placeholder count mismatches operands")

    def test_q4_edge_ir_preserves_cond(self):
        edge = to_edge_transform_and_lower(
            {"forward": self.ep},
            compile_config=EdgeCompileConfig(_check_ir_validity=False, _skip_dim_order=True),
        )
        gp = list(edge._edge_programs.values())[0]
        cond_nodes = [
            n for n in gp.graph_module.graph.nodes
            if n.op == "call_function" and n.target == torch.ops.higher_order.cond
        ]
        print(f"\n[Q4] edge IR cond nodes: {len(cond_nodes)}")
        self.assertEqual(len(cond_nodes), 1, "edge IR dropped the cond node")

    def test_q5_recompute_branch_inplace_wrapping(self):
        # Find the recompute (false) branch and look for index_put / copy_
        # vs auto_functionalized wrapping.
        false_graph = self.module.cache_k  # placeholder; we'll re-find below
        for name, sub in self.ep.graph_module.named_children():
            if "false_graph" not in name:
                continue
            print(f"\n[Q5] {name} call_function targets:")
            for n in sub.graph.nodes:
                if n.op != "call_function":
                    continue
                print(f"   {n.target}  ({n.name})")
            # Look for slice_scatter or copy in the recompute branch.
            has_inplace = any(
                "scatter" in str(n.target) or "copy" in str(n.target).lower()
                for n in sub.graph.nodes if n.op == "call_function"
            )
            print(f"[Q5] recompute branch has scatter/copy: {has_inplace}")

    def test_q6_predicate_dtype_and_shape(self):
        cond = next(
            n for n in self.ep.graph_module.graph.nodes
            if n.op == "call_function" and n.target == torch.ops.higher_order.cond
        )
        pred = cond.args[0]
        if isinstance(pred, torch.fx.Node):
            v = pred.meta.get("val")
            shape = tuple(v.shape) if hasattr(v, "shape") else None
            dtype = v.dtype if hasattr(v, "dtype") else None
            print(f"\n[Q6] predicate node {pred.name}: target={pred.target} "
                  f"shape={shape} dtype={dtype}")
        else:
            print(f"\n[Q6] predicate is constant: {pred!r}")


if __name__ == "__main__":
    unittest.main()
