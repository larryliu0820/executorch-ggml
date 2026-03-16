# KV Cache: Qwen3 decode quality regression

**Status**: Fixed — single-buffer in-place INDEX_PUT (see below)
**Affects**: Qwen3-0.6B (via optimum-executorch), likely all HF static-cache models
**Does NOT affect**: SimpleKV test models (test_kv_cache.py passes), Parakeet

## Symptom

Prefill + first decode token are correct. Subsequent decode tokens produce wrong
output (1-step lag pattern, eventually NaN at 28 layers).

```
Prefill: eager=" Paris" ggml=" Paris" diff=0.000099  ✓
Decode 0: eager="."     ggml="."     diff=0.000017  ✓
Decode 1: eager=" The"  ggml="!"     diff=nan        ✗
```

With 1 layer (no NaN), the pattern is clearer — GGML lags by exactly 1 step:

```
Step 0: eager=" only"  ggml=" only"  diff=0.000008  ✓
Step 1: eager=" bid"   ggml=" only"  diff=25.14     ✗  (ggml = step 0's answer)
Step 2: eager="irect"  ggml=" bid"   diff=30.12     ✗  (ggml = step 1's answer)
```

## Root cause

SDPA reads from the **old cache** (mutable_buf) instead of the **index_put
output** (updated cache). The graph wiring is correct at the FX level:

```
index_put(b_key_cache, pos, new_k) → unsqueeze → expand → clone → view → SDPA
```

But in the ggml IR, the `unsqueeze → expand → clone → view` chain between
`index_put` and `LLAMA_ATTENTION` gets lowered in a way that disconnects
the SDPA from the index_put result. The SDPA ends up reading from the
original buffer tensor (mutable_buf) which has stale data from the previous
call rather than the just-updated index_put result.

## Why test_kv_cache passes

The `SimpleKVCache` test model uses a direct pattern:

```python
self.k_cache[:, :, input_pos] = k
out = sdpa(q, self.k_cache, self.v_cache, mask)
```

After export + to_edge, this becomes:

```
index_put(cache, pos, k) → sdpa(q, index_put_result, ...)
```

No intermediate view/unsqueeze/expand/clone chain. The ggml IR correctly
wires SDPA to the index_put output tensor ID.

## Why Qwen3 (optimum-executorch) fails

The HF static cache + GQA attention inserts a `repeat_kv` step between
the cache update and SDPA:

```python
k_cache[:, :, input_pos] = k
k_for_attn = repeat_kv(k_cache, n_rep)  # unsqueeze + expand + clone + view
out = sdpa(q, k_for_attn, v_for_attn, mask)
```

This creates the view chain:
```
index_put → unsqueeze → expand → clone → view → SDPA
```

The ggml preprocess serializer processes these intermediate ops and assigns
them tensor IDs. The SDPA handler correctly references the final view's
tensor ID. But somewhere in the C++ `build_graph`, the chain of
VIEW → REPEAT → UNSQUEEZE ops may resolve back to the original buffer
tensor pointer rather than following the index_put output.

## What works

| Component | Status |
|-----------|--------|
| `tag_mutated_buffer` in partitioner | ✓ Working — buffers claimed by delegate |
| BUFFER_MUTATION output filtering | ✓ Working — only USER_OUTPUT exposed |
| `is_mutable` flag on IR tensors | ✓ Working — routed to mutable_buf |
| Post-compute writeback | ✓ Working — `ggml_backend_tensor_copy` runs |
| `ggml_set_output` on writeback tensors | ✓ Working — prevents scheduler memory reuse |
| Graph cache + per-graph scheduler | ✓ Working — shape switching correct |
| SimpleKV cache pattern | ✓ Working — 0.000000 diff on both tokens |

## Investigation next steps

1. Add debug logging in C++ `build_graph` for the INDEX_PUT → SDPA chain.
   Check which ggml_tensor pointers the LLAMA_ATTENTION node's src[] point
   to — do they point to the index_put output or the original buffer leaf?

2. The issue is likely in how VIEW/UNSQUEEZE/EXPAND ops in `build_graph`
   resolve their source tensors. These ops create ggml views that may
   point to the original buffer data rather than the index_put result's data.

3. Potential fix: in the LLAMA_ATTENTION handler in preprocess, trace back
   through view/reshape ops to find the actual data source. If it's an
   index_put, wire the SDPA directly to the index_put output, skipping
   the intermediate view chain (which is just a reshape for GQA repeat_kv).

## Repro

### Quick (1-layer, ~30s)

```bash
DYLD_LIBRARY_PATH=python/executorch_ggml python3 -c "
import torch, executorch_ggml
from optimum.exporters.executorch.integrations import CausalLMExportableModule
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, AutoConfig
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower, ExecutorchBackendConfig
from executorch.exir.passes import MemoryPlanningPass
from executorch_ggml import GgmlPartitioner
from executorch_ggml.passes import RemoveGraphAssertsPass
from executorch_ggml.passes.replace_copy_ops_pass import ReplaceCopyOpsPass
from transformers.integrations.executorch import TorchExportableModuleForDecoderOnlyLM

model_id = 'Qwen/Qwen3-0.6B'
config = AutoConfig.from_pretrained(model_id)
if hasattr(config, 'rope_scaling') and config.rope_scaling is not None:
    config.rope_scaling['type'] = 'default'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map='cpu', torch_dtype=torch.float32, config=config,
    attn_implementation='sdpa',
    generation_config=GenerationConfig(
        use_cache=True, cache_implementation='static',
        max_length=128, cache_config={'batch_size': 1, 'max_cache_len': 128}))
model.model.layers = model.model.layers[:1]

exportable = CausalLMExportableModule(model, max_seq_len=128,
    use_custom_kv_cache=False, use_custom_sdpa=False, disable_dynamic_shapes=False)
ep = exportable.export()['model']
edge = to_edge_transform_and_lower(
    ep, partitioner=[GgmlPartitioner()],
    compile_config=EdgeCompileConfig(_check_ir_validity=False, _skip_dim_order=True),
    transform_passes=[ReplaceCopyOpsPass(), RemoveGraphAssertsPass()],
    constant_methods=exportable.metadata)
et = edge.to_executorch(config=ExecutorchBackendConfig(
    extract_delegate_segments=True,
    memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False)))
from executorch.extension.pybindings.portable_lib import _load_for_executorch_from_buffer
pte = _load_for_executorch_from_buffer(et.buffer)
eager = TorchExportableModuleForDecoderOnlyLM(model)

ids = tokenizer('The capital of France is', return_tensors='pt')['input_ids']
pos = torch.arange(ids.shape[1], dtype=torch.long)
ggml_out = pte.forward((ids, pos))[0]
with torch.no_grad(): eager_out = eager(ids, cache_position=pos)
next_tok = eager_out[0,-1].argmax().item()

for i in range(5):
    dec_ids = torch.tensor([[next_tok]], dtype=torch.long)
    dec_pos = torch.tensor([ids.shape[1] + i], dtype=torch.long)
    ggml_out = pte.forward((dec_ids, dec_pos))[0]
    with torch.no_grad(): eager_out = eager(dec_ids, cache_position=dec_pos)
    diff = (eager_out - ggml_out).abs().max().item()
    g = tokenizer.decode([ggml_out[0,-1].argmax().item()])
    e = tokenizer.decode([eager_out[0,-1].argmax().item()])
    next_tok = eager_out[0,-1].argmax().item()
    print(f'Step {i}: eager={e!r:10s} ggml={g!r:10s} diff={diff:.6f}')
"
```

Expected output (step 0 matches, step 1+ diverge):
```
Step 0: eager=' only'  ggml=' only'  diff=0.000008
Step 1: eager=' bid'   ggml=' only'  diff=25.143032   ← 1-step lag
Step 2: eager='irect'  ggml=' bid'   diff=30.115009
```

### Full model (Q8_0, uses C++ runner)

```bash
DYLD_LIBRARY_PATH=python/executorch_ggml python3 runner/export_qwen3_q8.py
DYLD_LIBRARY_PATH=python/executorch_ggml python3 runner/run_qwen3.py --model qwen3/qwen3_q8_0.pte
```

Produces "The capital of France is Paris." then degenerates to "in in in...".

## Gotchas

### Build

- After editing `schema/ggml_ir.fbs`, regenerate the header with:
  ```bash
  build_native/flatbuffers/flatc --cpp --scoped-enums --no-prefix -o schema/ schema/ggml_ir.fbs
  ```
  The flags `--scoped-enums --no-prefix` are required to match the existing
  `enum class OpCode` style. Without them, the generated header uses C-style
  `OpCode_NONE` names and the C++ code won't compile.

- New opcodes in `ggml_ir.fbs` must also be added to `serialize.py` (Python
  constants like `OP_FOO = N`) and imported in `ggml_backend.py`.

### Dependencies

- `optimum-executorch` pins specific `transformers` versions. If you get
  `ModuleNotFoundError: transformers.tokenization_utils_tokenizers`, run
  `pip install -U transformers`.

- `torchcodec` (used by `torchaudio.load`) requires ffmpeg libraries.
  On macOS, `brew install ffmpeg` may install a newer version than
  torchcodec expects (ffmpeg 8 vs needed ffmpeg 4/5). Workaround:
  monkey-patch `torchaudio.load` with `scipy.io.wavfile`.

### Testing

- `test_kv_cache.py` passes — it uses a simple cache pattern without the
  view chain. It does NOT catch the Qwen3 regression.

- The `IndexPutModel` test returns the cache as both BUFFER_MUTATION and
  USER_OUTPUT. The partitioner skips `tag_mutated_buffer` for this case
  (ET doesn't support dual output kinds). The model still works via the
  runtime-input fallback path.

- BF16 on Metal hits `GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32)` in
  binary ops. Use FP16 or F32 for Metal testing.

- FP16 voxtral hits `ggml_compute_forward_pad_reflect_1d` fatal error
  (pad op doesn't support FP16). Use F32 for audio encoder testing.

## Fix

Two changes fix both the stale-read bug and the dual-buffer inefficiency:

1. **Python** (`ggml_backend.py`): After `node_to_id[node] = tid` for INDEX_PUT,
   also set `node_to_id[x_node] = tid`. This forwards the buffer placeholder so
   downstream nodes (unsqueeze, expand, SDPA) resolve to the index_put output
   instead of the stale mutable_buf leaf.

2. **C++** (`ggml_backend.cpp`): When `dst->buffer == mutable_buf`, use
   `ggml_map_custom3_inplace(ctx, dst, idx, val, ...)` instead of `ggml_custom_4d`.
   The inplace op returns a view aliasing mutable_buf — the callback scatters
   values directly into the cache with no full-cache memcpy and no post-compute
   writeback. The old `ggml_custom_4d` + writeback path is kept as fallback for
   non-mutable index_put (e.g., test models).

**CUDA note**: The inplace path relies on unified memory (CPU callback directly
accesses mutable_buf). This works on Apple Silicon (Metal) and CPU backends. For
CUDA, the CPU callback cannot access device memory — either fall back to
`ggml_custom_4d` + writeback (branch on backend type), or implement the scatter
as a CUDA kernel running on-device.
