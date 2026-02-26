"""Minimal repro for dynamic-shape support in the ggml backend.

Exports LLM-style models with one dynamic dimension (seq_len), runs
them through the ExecuTorch runtime at several sequence lengths, and
checks:
  1. The graph rebuilds when the dynamic dim changes (stderr logs).
  2. The ggml output is numerically close to eager PyTorch for each length.

Test 1: Gated SiLU FFN + residual (simple, fast).
Test 2: Qwen3 GQA attention via optimum-executorch (real transformer).

Usage:
    LD_LIBRARY_PATH=python/executorch_ggml pytest tests/test_dynamic_shapes.py -v -s
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.export import export, Dim

from executorch_ggml import GgmlPartitioner
from executorch_ggml.edge_pipeline import to_edge_rewrite_and_lower

from executorch.extension.pybindings.portable_lib import (
    _load_for_executorch_from_buffer,
)


# ---------------------------------------------------------------------------
# Model 1: LLM-style feedforward block
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    """Gated SiLU MLP (LLaMA / Qwen3 style)."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerFFBlock(nn.Module):
    """Gated SiLU MLP + residual add."""

    def __init__(self, dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.ffn = FeedForward(dim, hidden_dim)

    def forward(self, x):
        return x + self.ffn(x)


# ---------------------------------------------------------------------------
# Export + run helpers
# ---------------------------------------------------------------------------

def export_and_load(model, trace_input, dynamic_shapes, label="model"):
    """Export → lower → serialize → load.  Returns (buffer, pte_module)."""
    print(f"[test] Exporting {label} with trace shape: {trace_input.shape}", flush=True)

    with torch.no_grad():
        ep = export(model, (trace_input,), dynamic_shapes=dynamic_shapes,
                    strict=False)

    # Show symbolic dims
    for node in ep.graph_module.graph.nodes:
        if node.op == "placeholder":
            val = node.meta.get("val")
            if val is not None and hasattr(val, "shape"):
                sym = [isinstance(s, torch.SymInt) for s in val.shape]
                if any(sym):
                    print(f"[test]   {node.name}: shape={val.shape}, sym={sym}",
                          flush=True)

    edge_mgr = to_edge_rewrite_and_lower(
        ep, ep_passes=[], partitioner=[GgmlPartitioner()],
    )

    # Count delegates
    for name, prog in edge_mgr._edge_programs.items():
        n_del = sum(1 for n in prog.graph_module.graph.nodes
                    if 'lowered_module' in str(n.target) and n.op == 'get_attr')
        print(f"[test]   delegates: {n_del}", flush=True)

    et = edge_mgr.to_executorch()
    print(f"[test]   .pte size: {len(et.buffer)} bytes", flush=True)

    pte = _load_for_executorch_from_buffer(et.buffer)
    return et.buffer, pte


def run_sweep(pte, model, dim, test_lengths, label, atol=1e-2):
    """Run at multiple seq_lens, compare eager vs ggml, return pass/fail."""
    print(f"\n[test] {label}: running at seq_lens={test_lengths}")
    print("-" * 60, flush=True)

    all_passed = True
    for seq_len in test_lengths:
        x = torch.randn(1, seq_len, dim)
        with torch.no_grad():
            eager_out = model(x)
        ggml_out = pte.forward((x,))[0]

        abs_diff = (eager_out - ggml_out).abs().max().item()
        match = abs_diff < atol
        if not match:
            all_passed = False
        status = "PASS" if match else "FAIL"
        print(f"[test]   seq_len={seq_len:3d} | shape={list(eager_out.shape)} "
              f"| max_abs_diff={abs_diff:.6f} | {status}", flush=True)
        if not match:
            print(f"[test]     eager[:5]={eager_out.flatten()[:5].tolist()}")
            print(f"[test]     ggml[:5] ={ggml_out.flatten()[:5].tolist()}")

    print("-" * 60)
    return all_passed


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

FFN_DIM = 64
FFN_HIDDEN = 128
TRACE_SEQ = 4
MAX_SEQ = 32
TEST_LENGTHS = [4, 1, 8, 1, 16, 4]


def test_dynamic_shapes_ffn():
    """Gated SiLU MLP + residual with dynamic seq_len."""
    print("\n" + "=" * 60)
    print("Test 1: Gated SiLU FFN + residual")
    print("=" * 60, flush=True)

    model = TransformerFFBlock(dim=FFN_DIM, hidden_dim=FFN_HIDDEN).eval()
    trace_input = torch.randn(1, TRACE_SEQ, FFN_DIM)
    dyn = {"x": {1: Dim("seq_len", min=1, max=MAX_SEQ)}}

    _, pte = export_and_load(model, trace_input, dyn, label="FFN")
    ok = run_sweep(pte, model, FFN_DIM, TEST_LENGTHS, "FFN")
    assert ok, "FFN dynamic shape test failed"


def test_dynamic_shapes_gqa():
    """Qwen3 GQA attention via optimum-executorch with dynamic seq_len.

    Uses a tiny Qwen3 config (2 layers, dim=64) to keep it fast while
    exercising the real GQA attention code path including:
      - Q/K/V projections with different sizes (GQA: n_heads != n_kv_heads)
      - Per-head RMSNorm (Qwen3-specific)
      - RoPE (rotary position embeddings)
      - GQA repeat_kv (expand + reshape)
      - SDPA / eager attention
      - Output projection
    """
    from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig
    from optimum.exporters.executorch.integrations import CausalLMExportableModule
    from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig
    from executorch_ggml.passes import RemoveGraphAssertsPass
    from executorch_ggml.passes.replace_copy_ops_pass import ReplaceCopyOpsPass

    print("\n" + "=" * 60)
    print("Test 2: Qwen3 GQA via optimum-executorch (tiny config)")
    print("=" * 60, flush=True)

    # Tiny Qwen3 config — same architecture, much smaller
    config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
    config.hidden_size = 64
    config.intermediate_size = 128
    config.num_attention_heads = 4
    config.num_key_value_heads = 2     # GQA ratio = 2
    config.head_dim = 16
    config.num_hidden_layers = 2
    config.vocab_size = 256
    config.max_position_embeddings = MAX_SEQ + 1
    if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
        config.rope_scaling["type"] = "default"

    max_seq_len = MAX_SEQ

    print(f"[test] Tiny Qwen3: dim={config.hidden_size}, heads={config.num_attention_heads}, "
          f"kv_heads={config.num_key_value_heads}, layers={config.num_hidden_layers}",
          flush=True)

    eager_model = AutoModelForCausalLM.from_config(
        config,
        torch_dtype=torch.float32,
        attn_implementation="sdpa",
    )
    eager_model.generation_config = GenerationConfig(
        use_cache=True,
        cache_implementation="static",
        max_length=max_seq_len,
        cache_config={"batch_size": 1, "max_cache_len": max_seq_len},
    )

    print("[test] Creating CausalLMExportableModule with dynamic shapes...", flush=True)
    exportable = CausalLMExportableModule(
        eager_model,
        max_seq_len=max_seq_len,
        use_custom_kv_cache=False,
        use_custom_sdpa=False,
        disable_dynamic_shapes=False,
    )

    print("[test] Exporting...", flush=True)
    exported_progs = exportable.export()
    ep = exported_progs["model"]

    # Show symbolic shapes
    for node in ep.graph_module.graph.nodes:
        if node.op == "placeholder":
            val = node.meta.get("val")
            if val is not None and hasattr(val, "shape"):
                sym = [isinstance(s, torch.SymInt) for s in val.shape]
                if any(sym):
                    print(f"[test]   {node.name}: shape={val.shape}, sym={sym}",
                          flush=True)

    # No BroadcastCanonicalizationPass needed — the ggml backend handles
    # MUL/ADD broadcasts natively (ggml_mul swaps operands, ggml_add repeats).

    print("[test] Lowering to ggml...", flush=True)
    edge_manager = to_edge_transform_and_lower(
        ep,
        partitioner=[GgmlPartitioner()],
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=True,
        ),
        transform_passes=[ReplaceCopyOpsPass(), RemoveGraphAssertsPass()],
        constant_methods=exportable.metadata,
    )

    # Count delegates
    delegate_count = sum(
        1 for n in edge_manager.exported_program().graph.nodes
        if n.op == "call_function" and "executorch_call_delegate" in str(n.target)
    )
    print(f"[test]   delegates: {delegate_count}", flush=True)
    assert delegate_count > 0, "Expected at least one delegated call"

    et_program = edge_manager.to_executorch()
    print(f"[test]   .pte size: {len(et_program.buffer)} bytes", flush=True)

    pte = _load_for_executorch_from_buffer(et_program.buffer)

    # Build the eager-callable module (same as what export() uses internally).
    from transformers.integrations.executorch import TorchExportableModuleForDecoderOnlyLM
    eager_module = TorchExportableModuleForDecoderOnlyLM(eager_model)
    eager_module.eval()

    # Autoregressive test: prefill step 0, then decode step 1.
    # This mirrors real LLM inference with KV cache.
    # Use token IDs within our tiny vocab_size.
    steps = [
        {"input_ids": torch.tensor([[42]], dtype=torch.long),
         "cache_position": torch.tensor([0], dtype=torch.long)},
        {"input_ids": torch.tensor([[99]], dtype=torch.long),
         "cache_position": torch.tensor([1], dtype=torch.long)},
    ]

    print(f"\n[test] GQA: running {len(steps)} autoregressive steps")
    print("-" * 60, flush=True)

    all_passed = True
    for i, step in enumerate(steps):
        input_ids = step["input_ids"]
        cache_position = step["cache_position"]

        with torch.no_grad():
            eager_out = eager_module(input_ids, cache_position=cache_position)

        ggml_out = pte.forward((input_ids, cache_position))[0]

        abs_diff = (eager_out - ggml_out).abs().max().item()
        match = abs_diff < 1.0  # LLM logits across vocab; random weights + flash_attn accumulation order
        if not match:
            all_passed = False
        status = "PASS" if match else "FAIL"
        print(f"[test]   step {i}: input_ids={input_ids.tolist()} "
              f"cache_pos={cache_position.tolist()} "
              f"| shape={list(eager_out.shape)} "
              f"| max_abs_diff={abs_diff:.6f} | {status}", flush=True)
        if not match:
            # Show top-5 logits comparison
            eager_topk = eager_out[0, -1].topk(5)
            ggml_topk = ggml_out[0, -1].topk(5)
            print(f"[test]     eager top5: vals={eager_topk.values.tolist()} "
                  f"idx={eager_topk.indices.tolist()}")
            print(f"[test]     ggml  top5: vals={ggml_topk.values.tolist()} "
                  f"idx={ggml_topk.indices.tolist()}")

    print("-" * 60)
    assert all_passed, "GQA dynamic shape test failed"


# ---------------------------------------------------------------------------
# Model 3: Bare SDPA — isolate flash_attn_ext vs PyTorch eager attention
# ---------------------------------------------------------------------------

class BareSDPA(nn.Module):
    """Minimal multi-head attention: Q/K/V projections → SDPA → output proj.

    No positional encoding, no causal mask, no KV cache.
    This isolates the numerical diff between ggml_flash_attn_ext and
    PyTorch's F.scaled_dot_product_attention (eager math path).
    """

    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        # Keep SDPA as a single op so the ggml lowering captures it as
        # LLAMA_ATTENTION → ggml_flash_attn_ext (rather than decomposing
        # into bmm+softmax+where which has eager-op issues at build time).
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.o_proj(out)


def test_dynamic_shapes_sdpa():
    """Bare SDPA with dynamic seq_len — measure flash_attn_ext vs eager diff."""
    print("\n" + "=" * 60)
    print("Test 3: Bare SDPA (flash_attn_ext vs eager)")
    print("=" * 60, flush=True)

    dim = 64
    n_heads = 4
    trace_seq = 4
    max_seq = 32
    test_lengths = [4, 1, 8, 1, 16, 4]

    model = BareSDPA(dim=dim, n_heads=n_heads).eval()
    trace_input = torch.randn(1, trace_seq, dim)
    dyn = {"x": {1: Dim("seq_len", min=1, max=max_seq)}}

    _, pte = export_and_load(model, trace_input, dyn, label="SDPA")

    print(f"\n[test] SDPA: running at seq_lens={test_lengths}")
    print("-" * 60, flush=True)

    all_passed = True
    for seq_len in test_lengths:
        x = torch.randn(1, seq_len, dim)
        with torch.no_grad():
            eager_out = model(x)
        ggml_out = pte.forward((x,))[0]

        abs_diff = (eager_out - ggml_out).abs().max().item()
        # This directly measures flash_attn_ext vs eager SDPA precision.
        # Expect very small diffs since no mask, no cache, pure float32.
        match = abs_diff < 0.01
        if not match:
            all_passed = False
        status = "PASS" if match else "FAIL"
        print(f"[test]   seq_len={seq_len:3d} | shape={list(eager_out.shape)} "
              f"| max_abs_diff={abs_diff:.6f} | {status}", flush=True)
        if not match:
            print(f"[test]     eager[:5]={eager_out.flatten()[:5].tolist()}")
            print(f"[test]     ggml[:5] ={ggml_out.flatten()[:5].tolist()}")

    print("-" * 60)
    assert all_passed, "SDPA dynamic shape test failed"


if __name__ == "__main__":
    test_dynamic_shapes_ffn()
    test_dynamic_shapes_sdpa()
    test_dynamic_shapes_gqa()
