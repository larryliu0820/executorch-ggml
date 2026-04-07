#!/usr/bin/env python3
"""Quick end-to-end test: ask Qwen3-0.6B about the capital of France."""
import os, sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.ao.quantization as _taq
if not hasattr(_taq, "quantizer"):
    import types
    _taq.quantizer = types.ModuleType("torch.ao.quantization.quantizer")
    _taq.quantizer.quantizer = types.ModuleType("torch.ao.quantization.quantizer.quantizer")
    _taq.quantizer.quantizer.Quantizer = type("Quantizer", (), {})

import executorch_ggml  # noqa: F401
from executorch_ggml import GgmlPartitioner
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from optimum.exporters.executorch.integrations import CausalLMExportableModule
from executorch_ggml.passes.replace_copy_ops_pass import ReplaceCopyOpsPass
from executorch_ggml.passes import RemoveGraphAssertsPass
from executorch.extension.pybindings.portable_lib import _load_for_executorch_from_buffer
from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig, ExecutorchBackendConfig
from executorch.exir.passes import MemoryPlanningPass

model_id = "Qwen/Qwen3-0.6B"
max_seq_len = 256

print(f"Loading {model_id}...")
config = AutoConfig.from_pretrained(model_id)
if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
    config.rope_scaling["type"] = "default"

tokenizer = AutoTokenizer.from_pretrained(model_id)
eager_model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="cpu", torch_dtype=torch.float32, config=config,
    attn_implementation="sdpa",
    generation_config=GenerationConfig(
        use_cache=True, cache_implementation="static", max_length=max_seq_len,
        cache_config={"batch_size": 1, "max_cache_len": max_seq_len},
    ),
)

question = "What is the capital of France?"
messages = [{"role": "user", "content": question}]
chat_text = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
input_ids = tokenizer.encode(chat_text, return_tensors="pt")
print(f"Prompt: {question!r}  ({input_ids.shape[1]} tokens)")

print("Exporting...")
exportable = CausalLMExportableModule(
    eager_model, max_seq_len=max_seq_len,
    use_custom_kv_cache=False, use_custom_sdpa=False,
    disable_dynamic_shapes=False,
)
ep = exportable.export()["model"]
edge_mgr = to_edge_transform_and_lower(
    ep, partitioner=[GgmlPartitioner()],
    compile_config=EdgeCompileConfig(_check_ir_validity=False, _skip_dim_order=True),
    transform_passes=[ReplaceCopyOpsPass(), RemoveGraphAssertsPass()],
    constant_methods=exportable.metadata,
)
et_module = edge_mgr.to_executorch(config=ExecutorchBackendConfig(
    extract_delegate_segments=True,
    memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
))
pte = _load_for_executorch_from_buffer(et_module.buffer)

print("Generating...")
generated = input_ids.clone()
cache_pos = torch.arange(input_ids.shape[1], dtype=torch.long)

# Prefill
output = pte.forward((input_ids, cache_pos))
logits = output[0]
next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
print(f"Prefill top: {next_token.item()} = {repr(tokenizer.decode([next_token.item()]))}")
generated = torch.cat([generated, next_token], dim=-1)

# Decode
for i in range(max_seq_len - input_ids.shape[1] - 1):
    pos = torch.tensor([input_ids.shape[1] + i], dtype=torch.long)
    output = pte.forward((next_token.to(torch.long), pos))
    logits = output[0]
    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    tok_id = next_token.item()
    print(f"Decode {i}: {tok_id} = {repr(tokenizer.decode([tok_id]))}")
    generated = torch.cat([generated, next_token], dim=-1)
    if tok_id in (tokenizer.eos_token_id, 151643, 151645):
        break

print()
text = tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True)
print(f"\nQ: {question}")
print(f"A: {text}")
