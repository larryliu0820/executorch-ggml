"""
1-layer Qwen3 KV-cache decode stability test.

Loads real Qwen3-0.6B weights (first layer only), runs a prefill + 5 decode
steps, and checks that ggml logits stay within 0.01 of eager at every step.
This catches scheduler state leaks that cause logit drift on decode step 1+.
"""

import os
import pytest
import torch


@pytest.mark.xfail(
    condition=os.environ.get("GGML_NO_GRAPH_CACHE") != "1",
    reason="Graph cache reuses scheduler state; KV decode step 1+ drifts vs eager",
)
def test_one_layer_qwen3_kv_cache_decode():
    from executorch_ggml import GgmlPartitioner
    from executorch_ggml.passes import RemoveGraphAssertsPass
    from executorch_ggml.passes.replace_copy_ops_pass import ReplaceCopyOpsPass
    from executorch.exir import (
        EdgeCompileConfig,
        ExecutorchBackendConfig,
        to_edge_transform_and_lower,
    )
    from executorch.exir.passes import MemoryPlanningPass
    from executorch.extension.pybindings.portable_lib import (
        _load_for_executorch_from_buffer,
    )
    from optimum.exporters.executorch.integrations import CausalLMExportableModule
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        GenerationConfig,
    )
    from transformers.integrations.executorch import (
        TorchExportableModuleForDecoderOnlyLM,
    )

    model_id = "Qwen/Qwen3-0.6B"
    config = AutoConfig.from_pretrained(model_id)
    if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
        config.rope_scaling["type"] = "default"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cpu",
        torch_dtype=torch.float32,
        config=config,
        attn_implementation="sdpa",
        generation_config=GenerationConfig(
            use_cache=True,
            cache_implementation="static",
            max_length=128,
            cache_config={"batch_size": 1, "max_cache_len": 128},
        ),
    )
    # Keep only 1 decoder layer for speed.
    model.model.layers = model.model.layers[:1]

    exportable = CausalLMExportableModule(
        model,
        max_seq_len=128,
        use_custom_kv_cache=False,
        use_custom_sdpa=False,
        disable_dynamic_shapes=False,
    )
    ep = exportable.export()["model"]
    edge = to_edge_transform_and_lower(
        ep,
        partitioner=[GgmlPartitioner()],
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False, _skip_dim_order=True
        ),
        transform_passes=[ReplaceCopyOpsPass(), RemoveGraphAssertsPass()],
        constant_methods=exportable.metadata,
    )
    et = edge.to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
        )
    )
    pte = _load_for_executorch_from_buffer(et.buffer)
    eager = TorchExportableModuleForDecoderOnlyLM(model)

    # Prefill
    ids = tokenizer("The capital of France is", return_tensors="pt")["input_ids"]
    pos = torch.arange(ids.shape[1], dtype=torch.long)
    pte.forward((ids, pos))
    with torch.no_grad():
        eager_out = eager(ids, cache_position=pos)
    next_tok = eager_out[0, -1].argmax().item()

    # Decode 5 steps — each must match eager within 0.01
    for i in range(5):
        dec_ids = torch.tensor([[next_tok]], dtype=torch.long)
        dec_pos = torch.tensor([ids.shape[1] + i], dtype=torch.long)
        ggml_out = pte.forward((dec_ids, dec_pos))[0]
        with torch.no_grad():
            eager_out = eager(dec_ids, cache_position=dec_pos)
        diff = (eager_out - ggml_out).abs().max().item()
        g = tokenizer.decode([ggml_out[0, -1].argmax().item()])
        e = tokenizer.decode([eager_out[0, -1].argmax().item()])
        next_tok = eager_out[0, -1].argmax().item()
        print(f"Step {i}: eager={e!r:10s} ggml={g!r:10s} diff={diff:.6f}")
        assert diff < 0.01, f"decode step {i}: diff {diff:.6f} >= 0.01"
