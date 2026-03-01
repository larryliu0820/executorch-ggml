"""
End-to-end Qwen3-0.6B text generation test with ggml backend.

Exports Qwen3-0.6B with SDPA preserved, delegates to ggml, and runs
multi-step greedy decode.  Verifies output tokens are coherent and
that the ggml backend produces sane text.

Usage:
    # CPU
    PATH=.venv/bin:$PATH PYTHONPATH=python:$PYTHONPATH \
      GGML_BACKEND_DEVICE=cpu pytest tests/test_qwen3_e2e.py -v -s

    # Metal
    PATH=.venv/bin:$PATH PYTHONPATH=python:$PYTHONPATH \
      GGML_BACKEND_DEVICE=metal pytest tests/test_qwen3_e2e.py -v -s
"""

import os
import tempfile

import torch


def _export_qwen3(max_seq_len: int = 128):
    """Export Qwen3-0.6B to .pte with ggml backend delegation."""
    from executorch_ggml import GgmlPartitioner
    from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig
    from optimum.exporters.executorch.integrations import CausalLMExportableModule
    from executorch_ggml.passes import RemoveGraphAssertsPass
    from executorch_ggml.passes.replace_copy_ops_pass import ReplaceCopyOpsPass
    from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig

    model_id = "Qwen/Qwen3-0.6B"
    config = AutoConfig.from_pretrained(model_id)
    if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
        config.rope_scaling["type"] = "default"

    eager_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cpu",
        torch_dtype=torch.float32,
        config=config,
        attn_implementation="sdpa",
        generation_config=GenerationConfig(
            use_cache=True,
            cache_implementation="static",
            max_length=max_seq_len,
            cache_config={"batch_size": 1, "max_cache_len": max_seq_len},
        ),
    )

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

    return edge_mgr, eager_model


def _greedy_decode_eager(model, tokenizer, prompt, max_new_tokens=10):
    """Run greedy decode on the eager PyTorch model."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    seq_len = input_ids.shape[1]

    tokens = input_ids[0].tolist()
    cache_pos = list(range(seq_len))

    with torch.no_grad():
        # Prefill
        out = model(input_ids, cache_position=torch.tensor(cache_pos, dtype=torch.long))
        next_tok = out.logits[:, -1, :].argmax(dim=-1).item()
        tokens.append(next_tok)

        # Decode
        for i in range(max_new_tokens - 1):
            pos = seq_len + i
            out = model(
                torch.tensor([[next_tok]], dtype=torch.long),
                cache_position=torch.tensor([pos], dtype=torch.long),
            )
            next_tok = out.logits[:, -1, :].argmax(dim=-1).item()
            tokens.append(next_tok)

    return tokens


def _greedy_decode_ggml(pte_model, tokenizer, prompt, max_new_tokens=10):
    """Run greedy decode on the ggml-delegated ExecuTorch model."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    seq_len = input_ids.shape[1]

    tokens = input_ids[0].tolist()
    cache_pos = torch.tensor(list(range(seq_len)), dtype=torch.long)

    # Prefill
    out = pte_model.forward((input_ids, cache_pos))
    next_tok = out[0][:, -1, :].argmax(dim=-1).item()
    tokens.append(next_tok)

    # Decode
    for i in range(max_new_tokens - 1):
        pos = seq_len + i
        out = pte_model.forward((
            torch.tensor([[next_tok]], dtype=torch.long),
            torch.tensor([pos], dtype=torch.long),
        ))
        next_tok = out[0][:, -1, :].argmax(dim=-1).item()
        tokens.append(next_tok)

    return tokens


class TestQwen3E2E:
    """End-to-end Qwen3-0.6B text generation tests."""

    def test_greedy_decode_produces_coherent_text(self):
        """Export Qwen3-0.6B, run 10-token greedy decode, verify output is sane."""
        from transformers import AutoTokenizer
        from executorch.extension.pybindings.portable_lib import _load_for_executorch

        max_seq_len = 128
        max_new_tokens = 10
        prompt = "The capital of France is"

        print(f"\nExporting Qwen3-0.6B (max_seq_len={max_seq_len})...")
        edge_mgr, eager_model = _export_qwen3(max_seq_len=max_seq_len)

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

        # Save .pte and load via ExecuTorch runtime
        with tempfile.NamedTemporaryFile(suffix=".pte", delete=False) as f:
            f.write(edge_mgr.to_executorch().buffer)
            pte_path = f.name

        try:
            pte_model = _load_for_executorch(pte_path)
            backend = os.environ.get("GGML_BACKEND_DEVICE", "cpu")
            print(f"Backend: {backend}")

            # --- Eager decode ---
            print(f"\nPrompt: {prompt!r}")
            eager_tokens = _greedy_decode_eager(
                eager_model, tokenizer, prompt, max_new_tokens
            )
            eager_text = tokenizer.decode(eager_tokens)
            print(f"Eager:  {eager_text!r}")

            # --- GGML decode ---
            ggml_tokens = _greedy_decode_ggml(
                pte_model, tokenizer, prompt, max_new_tokens
            )
            ggml_text = tokenizer.decode(ggml_tokens)
            print(f"GGML:   {ggml_text!r}")

            # --- Assertions ---
            # 1. Output must be finite and non-empty
            assert len(ggml_tokens) == len(eager_tokens), (
                f"Token count mismatch: eager={len(eager_tokens)} ggml={len(ggml_tokens)}"
            )

            # 2. Prefill token (first generated) must match
            input_len = len(tokenizer.encode(prompt))
            assert ggml_tokens[input_len] == eager_tokens[input_len], (
                f"Prefill token mismatch: eager={eager_tokens[input_len]} "
                f"({tokenizer.decode([eager_tokens[input_len]])!r}) "
                f"ggml={ggml_tokens[input_len]} "
                f"({tokenizer.decode([ggml_tokens[input_len]])!r})"
            )

            # 3. GGML text should be coherent (not garbage/NaN)
            # Check that the output contains actual words, not just special tokens
            ggml_generated = tokenizer.decode(ggml_tokens[input_len:])
            assert len(ggml_generated.strip()) > 0, "GGML generated empty text"
            # Should contain at least some alphabetic characters
            alpha_chars = sum(1 for c in ggml_generated if c.isalpha())
            assert alpha_chars > 3, (
                f"GGML output lacks alphabetic content: {ggml_generated!r}"
            )

            # 4. Log token-by-token comparison
            print("\nToken-by-token comparison:")
            for i in range(input_len, len(ggml_tokens)):
                e_tok = eager_tokens[i]
                g_tok = ggml_tokens[i]
                match = "=" if e_tok == g_tok else "X"
                print(
                    f"  [{match}] pos {i}: "
                    f"eager={e_tok} ({tokenizer.decode([e_tok])!r})  "
                    f"ggml={g_tok} ({tokenizer.decode([g_tok])!r})"
                )

            n_match = sum(
                1
                for e, g in zip(eager_tokens[input_len:], ggml_tokens[input_len:])
                if e == g
            )
            n_total = max_new_tokens
            print(f"\nToken match: {n_match}/{n_total}")

        finally:
            os.unlink(pte_path)
