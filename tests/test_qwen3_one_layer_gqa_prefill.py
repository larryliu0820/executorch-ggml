"""
Focused numerical test: 1-layer Qwen3 GQA with KV cache, 2-token prefill.

This is a simplified repro for multi-token prefill behavior:
  - tiny Qwen3 config (single decoder layer, GQA enabled)
  - KV cache enabled (static cache)
  - single prefill call with 2 tokens (cache_position=[0, 1])
  - compare eager vs ggml outputs on the same call
"""

import torch


def test_one_layer_gqa_kv_cache_two_token_prefill_matches_eager():
    from executorch_ggml import GgmlPartitioner
    from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig
    from optimum.exporters.executorch.integrations import CausalLMExportableModule
    from executorch_ggml.passes import RemoveGraphAssertsPass
    from executorch_ggml.passes.replace_copy_ops_pass import ReplaceCopyOpsPass
    from executorch.extension.pybindings.portable_lib import (
        _load_for_executorch_from_buffer,
    )
    from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig

    torch.manual_seed(0)

    # Tiny, single-layer Qwen3 with GQA.
    config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
    config.hidden_size = 64
    config.intermediate_size = 128
    config.num_attention_heads = 4
    config.num_key_value_heads = 2  # GQA ratio = 2
    config.head_dim = 16
    config.num_hidden_layers = 1
    config.vocab_size = 256
    max_seq_len = 32
    config.max_position_embeddings = max_seq_len + 1
    if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
        config.rope_scaling["type"] = "default"

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
    eager_model.eval()

    exportable = CausalLMExportableModule(
        eager_model,
        max_seq_len=max_seq_len,
        use_custom_kv_cache=False,
        use_custom_sdpa=False,
        disable_dynamic_shapes=False,
    )
    ep = exportable.export()["model"]

    edge_mgr = to_edge_transform_and_lower(
        ep,
        partitioner=[GgmlPartitioner()],
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=True,
        ),
        transform_passes=[ReplaceCopyOpsPass(), RemoveGraphAssertsPass()],
        constant_methods=exportable.metadata,
    )
    pte_model = _load_for_executorch_from_buffer(edge_mgr.to_executorch().buffer)

    # 2-token prefill in one call.
    tokens = torch.tensor([[42, 99]], dtype=torch.long)
    cache_position = torch.tensor([0, 1], dtype=torch.long)

    with torch.no_grad():
        eager_logits = eager_model(tokens, cache_position=cache_position).logits
    ggml_logits = pte_model.forward((tokens, cache_position))[0]

    assert eager_logits.shape == ggml_logits.shape
    assert torch.isfinite(ggml_logits).all(), "ggml logits contain non-finite values"

    full_diff = (eager_logits - ggml_logits).abs()
    full_max_diff = full_diff.max().item()
    full_mean_diff = full_diff.mean().item()

    eager_last = eager_logits[:, -1, :].reshape(-1)
    ggml_last = ggml_logits[:, -1, :].reshape(-1)
    last_cos = torch.nn.functional.cosine_similarity(
        eager_last.unsqueeze(0), ggml_last.unsqueeze(0)
    ).item()
    eager_argmax = eager_last.argmax().item()
    ggml_argmax = ggml_last.argmax().item()

    print(f"[prefill] full max |eager - ggml|: {full_max_diff:.6f}")
    print(f"[prefill] full mean |eager - ggml|: {full_mean_diff:.6f}")
    print(f"[prefill] last-token cosine: {last_cos:.6f}")
    print(
        f"[prefill] last-token argmax: eager={eager_argmax} ggml={ggml_argmax}"
    )

    # Keep this strict enough to detect divergence while allowing small
    # backend-specific accumulation differences.
    assert last_cos > 0.99, f"last-token cosine too low: {last_cos:.4f}"
    assert full_max_diff < 1.0, f"full max diff too large: {full_max_diff:.4f}"
    assert eager_argmax == ggml_argmax, (
        f"argmax mismatch: eager={eager_argmax} ggml={ggml_argmax}"
    )
