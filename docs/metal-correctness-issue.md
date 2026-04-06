# Metal Correctness Issue: PTE Output Diverges from Eager PyTorch

## Status: Open

The ggml backend produces incorrect logits on the very first forward call (prefill).
This is a pre-existing issue affecting all backends (Metal, CUDA, CPU), not specific
to any optimization pass.

## Symptoms

- **Eager PyTorch** (via `TorchExportableModuleWithStaticCache`): produces correct
  output (`<think>\nOkay, the user is asking about the capital of France...`)
- **Exported PTE** (via ggml backend): produces garbage (`ilogue`, `وار`, etc.)
- The divergence occurs at the **first token** — prefill logits are already wrong
- llama.cpp with the same GGUF weights produces correct output

## Diagnosis

```
Eager logits: mean=5.358  std=3.630  max=30.601  top=<think>
PTE   logits: mean=0.655  std=3.360  max=15.513  top=وار

Diff: max=22.556  mean=5.190  positions_with_diff>0.1: 150484/151936
```

The logits are completely uncorrelated — not a numerical precision issue but a
fundamental compute error in one or more ops.

## What has been ruled out

1. **Optimization passes** — vanilla export (no swap_rms_norm, no fold, no fusion,
   no fuse_rope, no strip_gqa) produces the same garbage
2. **GGUF weight loading** — embedded-weights PTE (762 MB, weights from HuggingFace)
   also produces garbage
3. **QKV/gate-up fusion** — same issue before and after fusion
4. **Missing constants** — fixed the `skip_weight_data` bug that zero-initialized
   lifted tensor constants; output is still wrong after fix
5. **Static cache handling** — `TorchExportableModuleWithStaticCache` works correctly
   in eager PyTorch; the cache is properly registered as mutable buffers in the export

## Likely root causes (to investigate)

The issue is somewhere in the ggml backend's op handlers. Candidates:

1. **SDPA / Flash Attention** — the ggml `FLASH_ATTN_EXT` handler might interpret
   the causal mask or KV layout differently from PyTorch's `F.scaled_dot_product_attention`
2. **Embedding lookup** — the `GET_ROWS` handler might have a shape/stride mismatch
3. **RMSNorm** — if epsilon or weight handling differs from PyTorch's `aten.rms_norm`
4. **Linear / MUL_MAT** — weight transposition mismatch between PyTorch [out, in]
   and ggml's reversed dimension order
5. **View/Reshape** — the ggml reversed dimension order might cause incorrect
   head splitting in the attention forward

## How to debug

Use `GGML_DEBUG_DUMP` to dump per-node tensor statistics, then compare with eager
PyTorch layer-by-layer:

```bash
# Dump ggml per-node stats
GGML_DEBUG_DUMP=/tmp/ggml_dump.txt ./build_native/benchmark/benchmark_llm \
    model.pte --n-decode 0 --prompt-len 5
```

```python
# Compare with eager PyTorch using forward hooks
hooks = {}
for name, mod in model.named_modules():
    def hook_fn(name):
        def fn(mod, inp, out):
            t = out if isinstance(out, torch.Tensor) else out[0]
            print(f'{name}: mean={t.float().mean():.4f} std={t.float().std():.4f}')
        return fn
    mod.register_forward_hook(hook_fn(name))
```

Binary search: find the first layer where ggml's output diverges from eager.
Then narrow to the specific op within that layer.

## Benchmark numbers are still valid

The throughput benchmarks (331 tok/s Metal, 411 tok/s CUDA) measure graph execution
speed correctly — the graph structure, node count, compute dispatches, and memory
bandwidth utilization are all accurate. The model simply computes the wrong values
due to an op-level bug.

Once the correctness issue is fixed, the same optimizations (graph cache, projection
fusion, RoPE fusion, etc.) will apply and the throughput numbers should be unchanged.
