#!/usr/bin/env python3
"""Quick numerical test for the SDPA auto-slice fix.

Exports Qwen3-0.6B F32, runs prefill+decode, compares with eager.
"""
import os, sys, gc
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch

# Workaround: torchao 0.15.0 references torch.ao.quantization.quantizer
# which was removed in torch 2.11.0.dev. Patch it before imports trigger.
import torch.ao.quantization as _taq
if not hasattr(_taq, "quantizer"):
    import types
    _taq.quantizer = types.ModuleType("torch.ao.quantization.quantizer")
    _taq.quantizer.quantizer = types.ModuleType("torch.ao.quantization.quantizer.quantizer")
    _taq.quantizer.quantizer.Quantizer = type("Quantizer", (), {})

import executorch_ggml  # noqa: F401 (RTLD_GLOBAL)

from executorch_ggml import GgmlPartitioner
from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig
from optimum.exporters.executorch.integrations import CausalLMExportableModule
from executorch_ggml.passes.replace_copy_ops_pass import ReplaceCopyOpsPass
from executorch_ggml.passes import RemoveGraphAssertsPass
from executorch.extension.pybindings.portable_lib import _load_for_executorch_from_buffer
from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig
from executorch.exir import ExecutorchBackendConfig
from executorch.exir.passes import MemoryPlanningPass

model_id = "Qwen/Qwen3-0.6B"
max_seq_len = 128

print(f"[1/5] Loading {model_id}...")
config = AutoConfig.from_pretrained(model_id)
if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
    config.rope_scaling["type"] = "default"
n_layers = int(os.environ.get("N_LAYERS", "28"))
config.num_hidden_layers = n_layers
print(f"  Using {n_layers} layers")

eager_model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="cpu", torch_dtype=torch.float32, config=config,
    attn_implementation="sdpa",
    generation_config=GenerationConfig(
        use_cache=True, cache_implementation="static", max_length=max_seq_len,
        cache_config={"batch_size": 1, "max_cache_len": max_seq_len},
    ),
)

# Test: 2-token prefill + 1-token decode
steps = [
    (torch.tensor([[151644, 151645]], dtype=torch.long),
     torch.tensor([0, 1], dtype=torch.long)),
    (torch.tensor([[151646]], dtype=torch.long),
     torch.tensor([2], dtype=torch.long)),
]

print("[2/5] Eager reference (stateful with StaticCache)...")
from transformers.cache_utils import StaticCache
static_cache = StaticCache(config, batch_size=1, max_cache_len=max_seq_len, dtype=torch.float32)
with torch.no_grad():
    eager_logits = []
    for tokens, cache_position in steps:
        out = eager_model(tokens, cache_position=cache_position, past_key_values=static_cache)
        eager_logits.append(out.logits.clone())

print("[3/5] Exporting...")
exportable = CausalLMExportableModule(
    eager_model, max_seq_len=max_seq_len,
    use_custom_kv_cache=False, use_custom_sdpa=False,
    disable_dynamic_shapes=False,
)
ep = exportable.export()["model"]

print("[4/5] Lowering to ggml backend...")
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
pte_model = _load_for_executorch_from_buffer(et_module.buffer)

print("[5/5] Running ggml backend...")
all_pass = True
for step_idx, (tokens, cache_position) in enumerate(steps):
    output = pte_model.forward((tokens, cache_position))
    ggml_logits = output[0]

    eager_flat = eager_logits[step_idx][:, -1, :].detach().view(-1)
    ggml_flat = ggml_logits[:, -1, :].detach().view(-1)

    cos_sim = torch.nn.functional.cosine_similarity(
        eager_flat.unsqueeze(0), ggml_flat.unsqueeze(0)).item()
    max_diff = (eager_flat - ggml_flat).abs().max().item()
    mean_diff = (eager_flat - ggml_flat).abs().mean().item()
    eager_argmax = eager_flat.argmax().item()
    ggml_argmax = ggml_flat.argmax().item()

    status = "PASS" if (cos_sim > 0.99 and eager_argmax == ggml_argmax) else "FAIL"
    if status == "FAIL":
        all_pass = False
    print(f"  step {step_idx} [{status}]: cos_sim={cos_sim:.6f}  max_diff={max_diff:.4f}  mean_diff={mean_diff:.6f}  eager_top={eager_argmax}  ggml_top={ggml_argmax}")

if all_pass:
    print("\nRESULT: ALL CHECKS PASSED")
else:
    print("\nRESULT: FAILED")
    sys.exit(1)
