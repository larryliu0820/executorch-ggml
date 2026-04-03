# Plugging a Custom SDPA Kernel into the ggml Graph

## Overview

ggml's scheduler calls an **eval callback** before each graph node. Returning `false`
from the callback skips ggml's built-in kernel, letting you run your own instead.
This works with CUDA graphs — during capture your kernel gets captured alongside ggml's
other kernels, and during replay it all replays as one unit.

## Architecture

```
ggml_backend_sched_graph_compute(sched, graph)
  └─ for each node:
       1. eval_callback(node, ask=true)   ← return false to SKIP ggml's kernel
       2. [ggml's kernel runs if ask returned true]
       3. eval_callback(node, ask=false)  ← post-compute hook
```

## Implementation

### 1. Define the callback

```cpp
#include <cuda_runtime.h>

struct CustomSDPAContext {
    ggml_backend_t backend;  // the CUDA backend handle
    // Your kernel state: compiled PTX, workspace buffers, etc.
};

// Your custom SDPA kernel launcher
extern void launch_my_sdpa(
    const void* Q, const void* K, const void* V,
    const void* mask, void* output,
    int head_dim, int n_heads_q, int n_heads_kv, int seq_len_q, int seq_len_kv,
    float scale, cudaStream_t stream);

static bool custom_sdpa_callback(struct ggml_tensor* t, bool ask, void* user_data) {
    if (!ask) return true;  // post-compute: nothing to do
    if (t->op != GGML_OP_FLASH_ATTN_EXT) return true;  // not SDPA: let ggml handle it

    auto* ctx = static_cast<CustomSDPAContext*>(user_data);

    // Extract tensor pointers (already on GPU)
    struct ggml_tensor* Q = t->src[0];
    struct ggml_tensor* K = t->src[1];
    struct ggml_tensor* V = t->src[2];
    struct ggml_tensor* mask = t->src[3];  // may be nullptr

    // ggml layout: Q=[D, T_q, H_q, B], K=[D, T_kv, H_kv, B], output=[D, H_q, T_q, B]
    int head_dim   = Q->ne[0];
    int seq_len_q  = Q->ne[1];
    int n_heads_q  = Q->ne[2];
    int batch      = Q->ne[3];
    int seq_len_kv = K->ne[1];
    int n_heads_kv = K->ne[2];

    // Scale from op_params
    float scale;
    memcpy(&scale, (const char*)t->op_params + 0, sizeof(float));

    // Get CUDA stream: the backend's context holds it.
    // Access via: ((ggml_backend_cuda_context*)backend->context)->stream()
    // For simplicity, use the default stream (ggml uses stream 0 for single-GPU):
    cudaStream_t stream = nullptr;  // default stream

    // Launch your kernel
    launch_my_sdpa(
        Q->data, K->data, V->data,
        mask ? mask->data : nullptr,
        t->data,  // output tensor — write here
        head_dim, n_heads_q, n_heads_kv, seq_len_q, seq_len_kv,
        scale, stream);

    return false;  // SKIP ggml's flash_attn_ext — we handled it
}
```

### 2. Register the callback

In `ggml_backend.cpp`, in the execute function (after sched_alloc, before compute):

```cpp
CustomSDPAContext sdpa_ctx{handle->backend};

ggml_backend_sched_set_eval_callback(
    active->sched, custom_sdpa_callback, &sdpa_ctx);

// Compute — your kernel runs for FLASH_ATTN_EXT nodes
ggml_backend_sched_graph_compute(active->sched, active->graph);

// Clear callback
ggml_backend_sched_set_eval_callback(active->sched, nullptr, nullptr);
```

### 3. CUDA Graph compatibility

CUDA graphs capture the kernel launch sequence on the first run and replay it on
subsequent runs. Since the eval callback fires during capture too, your custom
kernel gets captured into the graph automatically. Requirements:

- Launch on the **same CUDA stream** as ggml (stream 0 / default)
- Use **fixed-size workspace** (no dynamic cudaMalloc during capture)
- Tensor data pointers are **stable** across calls (graph cache ensures this)

### 4. Integration with ExecuTorch

The eval callback is set per-execute call. You can enable/disable it via environment
variable:

```cpp
static int use_custom_sdpa = -1;
if (use_custom_sdpa < 0) {
    const char* env = std::getenv("GGML_CUSTOM_SDPA");
    use_custom_sdpa = (env && std::string(env) != "0") ? 1 : 0;
}

if (use_custom_sdpa) {
    ggml_backend_sched_set_eval_callback(sched, custom_sdpa_callback, &ctx);
}
```

### 5. Tensor layout reference

ggml uses reversed dimension order from PyTorch:

| | PyTorch | ggml |
|---|---|---|
| Q | `(B, H_q, T_q, D)` | `[D, T_q, H_q, B]` |
| K | `(B, H_kv, T_kv, D)` | `[D, T_kv, H_kv, B]` |
| V | `(B, H_kv, T_kv, D)` | `[D, T_kv, H_kv, B]` |
| Output | `(B, H_q, T_q, D)` | `[D, H_q, T_q, B]` |
| Mask | `(T_kv, T_q, 1, 1)` | `[T_kv, T_q, 1, 1]` (F16 additive) |

Note: output layout differs from Q layout (`H` and `T` swapped). This is how
`ggml_flash_attn_ext` returns its result.

### 6. Example: Triton kernel integration

If your fast SDPA is a Triton kernel, compile it to a CUDA binary (cubin/PTX) and
load it at runtime:

```cpp
#include <cuda.h>

// Load pre-compiled Triton kernel
static CUfunction load_triton_sdpa() {
    CUmodule module;
    cuModuleLoad(&module, "my_sdpa_kernel.cubin");
    CUfunction func;
    cuModuleGetFunction(&func, module, "my_sdpa_kernel");
    return func;
}

void launch_my_sdpa(..., cudaStream_t stream) {
    static CUfunction kernel = load_triton_sdpa();
    void* args[] = {&Q, &K, &V, &mask, &output, &head_dim, ...};
    cuLaunchKernel(kernel, grid.x, grid.y, grid.z,
                   block.x, block.y, block.z,
                   shared_mem, stream, args, nullptr);
}
```
