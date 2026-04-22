# Qwen3.5-35B-A3B MoE Performance Progress

## Current State

**128.8 tok/s decode** on A100-40GB with Q4_K_M weights (7.8 ms/tok, 3,094 graph nodes).

Reference: llama.cpp `llama-bench -n 50` = 105.8 tok/s on same hardware/model. **122% parity** (beats llama.cpp).

| Metric | ExecuTorch-GGML | llama.cpp | Ratio |
|---|---|---|---|
| Decode tok/s | 128.8 | 105.8 | 1.22 |
| Graph nodes | 3,094 | 3,778 | 0.82x |
| Decode ms/tok | 7.8 | 9.5 | 0.82x |

## Optimization Timeline

| Change | Nodes | tok/s | Notes |
|---|---|---|---|
| Baseline (Dec 2025) | — | 3.9 | Unoptimized |
| MoE fusion + graph cache + I32 cmp | 15K→6.2K | 68 | Large win from MoE → single fused op |
| RMSNorm preservation + SSM matmul + strip identity | 6.2K→6.0K | 76.7 | Partitioner preserves `aten.rms_norm`, SSM einsum → explicit matmul |
| F.normalize → F.rms_norm (Q/K L2 norm) | 6.0K→5.8K | 78.5 | Avoids SQR+SUM+SQRT+CLAMP+REPEAT+DIV decomposition |
| `ggml.softplus` + `ggml.ssm_conv` native ops | 5.8K→5.0K | 89.2 | Replaces log1p+exp+where and 4-iter kernel unroll |
| State layout `[B,H,V,K]` + MoE view+add reduce | 5.0K→5.3K | 90.5 | No transpose in delta-rule matmul; MoE reduce matches llama.cpp |
| SSM_CONV output `[B,T,C]` native layout (no permute) | 5.3K→5.2K | 91.8 | Python model reads natural ggml ssm_conv output directly |
| Preserve SDPA for direct-export + fused `ggml.rope` | 5.2K→4.1K | 111.7 | Full-attention layers now use `ggml_flash_attn_ext` (GQA native, no KV broadcast) + partial RoPE fused |
| CUDA-fusion-friendly emission (RMS+MUL, SSM_CONV+silu, scale fold) | 4.1K→4.0K | 113.4 | L2-norm scalar folded into weight → RMS_NORM+MUL CUDA fusion; conv_state size K-1 eliminates VIEW between SSM_CONV and silu; GDN attn scale absorbed into q's L2 weight |
| Fused `ggml_gated_delta_net` op | 4.0K→3.2K | 128.0 | Single CUDA kernel replaces the entire recurrent delta-rule chain (~80 ops × 30 layers = ~900 ops reduced to 30). Handles head-repeat broadcast natively via H_k vs H_v. |
| Build-time fold of constant EXP/NEG + y.reshape instead of contiguous | 3.2K→3.1K | **128.8** | `-exp(A_log)` is a per-layer constant — fold at graph-build time (eager constant). Plus `y.transpose(1,2).reshape(...)` avoids the CONT that `torch.export` emits after `.contiguous()`. |

## Key Optimizations Applied

### 1. MoE Fusion Pass (Python FX pass)

Pattern-matches the SparseMoE forward (topk + per-expert matmul loop) and replaces with `torch.ops.ggml.moe_ffn`. The C++ runtime expands to llama.cpp's build_moe_ffn sequence (~11 ops).

**File**: `python/executorch_ggml/passes/fuse_moe_pass.py`

### 2. Custom `ggml.*` ops

Replace PyTorch-decomposed chains with single ops that map to native ggml kernels:

- `ggml.ssm_conv(conv_input, weight)` → `ggml_ssm_conv` (replaces 4-iteration kernel unroll)
- `ggml.softplus(x)` → `ggml_softplus` (replaces `log1p+exp+where` decomposition)
- `ggml.moe_ffn(...)` → `build_op_moe_ffn` (expands to ~11 ggml ops matching llama.cpp)

**Files**: `python/executorch_ggml/ssm_conv_op.py`, `ops/activation.py`, `passes/fuse_moe_pass.py`

### 3. State Buffer Layout

`recurrent_state` stored as `[B, H, V, K]` (V before K) instead of `[B, H, K, V]`. The delta-rule matmul is now `state @ k.unsqueeze(-1)` without a transpose, saving 2 TRANSPOSE+CONT ops per SSM layer.

**Impact**: 120 ops saved per decode step on 30 SSM layers.

### 4. MoE Reduce via View+Add Chain

Instead of `PERMUTE + CONT + SUM_ROWS` over the top_k axis, emit `top_k - 1` ADD ops over VIEW slices. Matches llama.cpp's `build_moe_ffn` final aggregation.

**Net**: trades 3 heavy ops for 7 ADD + 8 VIEW; faster despite higher node count.

### 5. SSM_CONV Native Layout

Abstract returns `[B, T, C]` (ggml_ssm_conv's native output shape) so the C++ runtime does NOT emit `ggml_cont(ggml_permute(...))`. Python model consumes `acc[:, -T:, :]` instead of the equivalent `.transpose(1,2)`.

### 6. Preserve SDPA for direct-export (flash_attn_ext with native GQA)

The direct-export path had `preserve_sdpa=False`, which let torch.export decompose `F.scaled_dot_product_attention(..., enable_gqa=True)` into:
- `k.repeat_interleave` → `[256, 256, 8, 2]` REPEAT → RESHAPE → TRANSPOSE → CONT (×3 per layer × 10 full-attention layers = 30 CONTs of ~4MB each)
- Explicit MUL_MAT(Q, K^T) + MUL + SOFT_MAX + MUL_MAT(attn, V) chain
- Full mask materialization with CLAMP/UNARY/REPEAT/SUB chain

Setting `preserve_sdpa=True` in the partitioner lets `ggml_flash_attn_ext` handle the whole attention, including native GQA (the kernel broadcasts K heads to match Q heads via strides — no REPEAT, no CONT). Saved ~120MB of memory traffic per token.

### 7. Fused `ggml.rope` for partial RoPE

Qwen3.5 uses partial RoPE (first 64 of 256 head dims). The decomposed Python `apply_rotary` (slice → cos/sin outer product → split halves → mul → cat) produced ~120 ops per decode step. Switched to `torch.ops.ggml.rope(x, positions, n_dims=64, freq_base, mode=2)` which maps to a single `ggml_rope_ext` kernel supporting partial rotation via `n_dims`.

### 8. F.normalize → F.rms_norm

`F.normalize(x, p=2, dim=-1) = F.rms_norm(x, [N], eps/N) / sqrt(N)`. Numerically equivalent; the ggml backend fuses rms_norm into a single kernel.

## Per-op Profile (current baseline)

```
OP            TOTAL  COUNT   AVG    %
MUL_MAT        8.3ms   411  20.2us  15.8%
ADD            6.8ms   560  12.2us  13.0%
MUL            6.2ms   501  12.4us  11.8%
CONT           4.9ms   390  12.6us   9.3%
UNARY          3.8ms   330  11.6us   7.3%
RESHAPE        3.2ms   971   3.3us   6.0%
MUL_MAT_ID     3.0ms   120  25.1us   5.7%
VIEW           2.7ms   840   3.3us   5.2%
RMS_NORM       2.4ms   191  12.4us   4.5%
REPEAT         1.7ms   130  12.9us   3.2%
```

Every op pays ~12 µs of kernel launch overhead regardless of work. The main lever remaining is **reducing total op count**.

## Remaining Opportunities

### High-confidence wins

1. **Delta-rule mul+sum rewrite** (potential ~100 ops)
   - llama.cpp's `build_delta_net_autoregressive` uses `ggml_mul + ggml_sum_rows` instead of matmul for small `[128,128] x [128,1]` tensors (3x faster in micro-benchmark).
   - **Blocker**: Prior attempt using `(state * k.unsqueeze(-1)).sum(dim=-2)` caused benchmark to hang during GGUF load (root cause unclear; possibly scheduler/allocation interaction).
   - **Next step**: Register a custom `torch.ops.ggml.delta_mul_reduce` op that maps directly to `ggml_mul + ggml_sum_rows` without going through `aten.sum.dim_IntList`.

2. **GatedDeltaNet 4 input projections fusion** (potential 90 MUL_MATs)
   - `_in_proj_qkv + _in_proj_z + _in_proj_b + _in_proj_a` all share the same input `x` with same `hidden_size` — classic fusion target.
   - **Blocker**: Fusion via `_SlicedLinear` requires the GGUF loader to map the fused tensor name. GGUF stores these as separate tensors (`attn_qkv`, `attn_gate`, `ssm_beta`, `ssm_alpha`).
   - **Next step**: Extend weight_mapping.py to declare a "fused read" that concatenates these 4 GGUF tensors at load time into a single fused parameter.

3. **RESHAPE elimination** (1,191 in graph → ~0 runtime cost but bloats scheduler)
   - Many reshapes are `[B,T,C] ↔ [B*T, C]` or `[B,T,H,D] ↔ [B,T,H*D]` chains that torch.export emits between adjacent ops.
   - **Next step**: Add a graph pass that folds adjacent reshape/view pairs where the final shape equals an earlier one.

### Structural changes

4. **Attention q+gate fused projection** (full attention layers)
   - In llama.cpp's `build_layer_attn`, `wq` outputs `[(n_embd_head * 2) * n_head, n_tokens]` (Q + gate together). We use a single `qkv_proj` that fuses Q+K+V but gate is unfused.
   - Not obviously winnable; gate dim != KV dim.

5. **I32 → F32 cast elimination for position input**
   - Every layer re-casts the position tensor. Could eliminate by pre-computing `cos/sin` lookup tables once per step at C++ level.

### Investigation needed

6. **Why 480 SSM-related CONTs remain**
   - `[32,2]`, `[32,16]`, `[128,16]` CONT shapes come from RMSNormGated + SwiGLU paths in SSM output projection. Might collapse with a fused `ggml.gated_rms_norm` custom op.

7. **CUDA graph replay overhead**
   - CUDA graphs are enabled and replaying (confirmed: steady-state compute 11 ms/tok after initial capture). Yet kernels floor at 12 µs — may be the replay dispatch itself, not kernel time.

## Reproducibility

### Build

```bash
cmake -B build -G Ninja -DGGML_CUDA=ON -DGGML_CUDA_GRAPHS=ON
cmake --build build --parallel 16
```

### Export (takes ~25 min)

```python
from executorch_ggml import export_gguf_to_pte, GGUFExportConfig
config = GGUFExportConfig(
    max_seq_len=256,
    preserve_dynamic_shapes=True,
    enable_quantization=True,
)
export_gguf_to_pte(
    "/path/to/Qwen3.5-35B-A3B-Q4_K_M.gguf",
    "qwen3/qwen3.5_moe.pte",
    config,
)
```

Meta-device export: 28.3 MB weight-less PTE, 1.7 GB peak RSS (for 138 GB worth of parameters).

### Benchmark

```bash
GGML_BACKEND_DEVICE=cuda ./build/benchmark/benchmark_llm \
  qwen3/qwen3.5_moe.pte \
  --gguf /path/to/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --n-decode 50 --prompt-len 1
```

### Profile

```bash
GGML_PROFILE=1 ./build/benchmark/benchmark_llm ... --n-decode 3 --prompt-len 1
```

### Graph dump

```bash
GGML_DEBUG_DUMP=/tmp/graph.txt ./build/benchmark/benchmark_llm ... --n-decode 1 --prompt-len 1
grep -oE "op=[A-Z_]+" /tmp/graph.txt | sort | uniq -c | sort -rn
```
