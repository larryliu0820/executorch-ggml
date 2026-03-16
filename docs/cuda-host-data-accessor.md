# CUDA Host Data Accessor for Eager Ops

## Problem

When the GGML backend runs on a CUDA GPU, `const_buf` (model weights) is allocated
on the device via `ggml_backend_alloc_buffer(backend, size)`. During `build_graph()`,
several "eager ops" read tensor data at graph-build time to produce constant results
(e.g. I64 to I32 casts, scalar arithmetic, batch norm parameter folding). These ops
dereference `tensor->data` directly:

```cpp
int64_t val = *(const int64_t*)src->data;  // segfault if src->data is a CUDA pointer
```

On CPU and Metal (unified memory), `tensor->data` is a host pointer and this works.
On CUDA, `tensor->data` is a device pointer that cannot be dereferenced from the host,
causing a segfault during Phase B of `build_graph()`.

## Solution: HostDataAccessor

A `HostDataAccessor` class transparently handles host vs device memory. For host
buffers it returns the pointer directly (zero-copy). For device buffers it copies the
data into a staging buffer via `ggml_backend_tensor_get()`.

```cpp
class HostDataAccessor {
public:
  const void* get(const struct ggml_tensor* t) {
    if (!t || !t->data) return nullptr;
    if (!t->buffer || ggml_backend_buffer_is_host(t->buffer))
      return t->data;                          // CPU / Metal: direct access
    staging_.resize(ggml_nbytes(t));
    ggml_backend_tensor_get(t, staging_.data(), 0, staging_.size());
    return staging_.data();                     // CUDA: copy to host staging
  }
  float   read_f32(const struct ggml_tensor* t);
  int32_t read_i32(const struct ggml_tensor* t);
  int64_t read_i64(const struct ggml_tensor* t);
private:
  std::vector<uint8_t> staging_;
};
```

The class lives inside the anonymous namespace in `runtime/ggml_backend.cpp`, before
the eager cast helpers. A single `HostDataAccessor host_acc;` instance is declared at
the top of the Phase B loop in `build_graph()` and threaded through all call sites.

### What changed

| Function / code path | Before | After |
|---|---|---|
| `eager_cast_i64_to_i32` | `(int64_t*)src->data` | `acc->get(src)` via optional param |
| `eager_cast_i32_to_i64` | `(int32_t*)src->data` | `acc->get(src)` via optional param |
| `safe_ggml_cast` | `*(int32_t*)src->data` for scalar I32->F32 | `acc->read_i32(src)` via optional param |
| `try_eager_scalar_binop` | `*(float*)a->data` | `acc.read_f32(a)` |
| I64 ADD/SUB eager path | `(int64_t*)a->data` | `host_acc.get(a)` |
| MUL scalar path | `*(float*)b->data` | `host_acc.read_f32(b)` |
| Scalar cast paths | `(int64_t*)src->data` | `host_acc.get(src)` |
| ANY reduction | `(float*)src->data` | `host_acc.get(src)` |
| BATCH_NORM | `(float*)weight->data`, etc. | `host_acc.get(weight)`, etc. |
| UPDATE_CACHE start_pos | `*(int64_t*)tensor->data` | `host_acc.read_i64(tensor)` |
| All `safe_ggml_cast` callers in Phase B | no accessor passed | `&host_acc` as 4th arg |

### Init data copy

The `init()` function copies weight data into `const_buf`. When the buffer is on a
non-host backend (CUDA), it uses `ggml_backend_tensor_set()` instead of `memcpy`:

```cpp
if (!ggml_backend_buffer_is_host(tensor->buffer)) {
  ggml_backend_tensor_set(tensor, src_data, 0, nbytes);
} else {
  memcpy(tensor->data, src_data, nbytes);
}
```

### Backend detection

`ggml_backend_buffer_is_host(buffer)` returns:
- **true** for CPU, Metal (unified memory) -- direct pointer access is safe
- **false** for CUDA -- must use `ggml_backend_tensor_get/set`

This means no behavior change on CPU or Apple Silicon. The staging copy only happens
on CUDA where it is required.

## Known issue: empty transcription on Parakeet

With the above fixes applied, the Parakeet encoder runs to completion on CUDA (Orin)
without crashing. However, the greedy decode produces **empty text** while the eager
PyTorch baseline produces the correct transcription.

Possible causes to investigate:

1. **Numerical divergence in F32 CUDA path.** The encoder output (`f_proj`) may have
   subtle differences that push the decoder toward blank tokens. Compare `f_proj`
   values between eager PyTorch and the GGML backend (cosine similarity / max abs
   diff).

2. **Eager constant data written to wrong location.** After `ggml_backend_sched_alloc_graph`
   clears tensor pointers, `execute()` restores shared leaves and eager constants.
   If an eager constant ends up on a different backend buffer than expected, its data
   may not reach the CUDA kernel.

3. **Decoder graph correctness.** The `greedy_decode_executorch` function calls the
   decoder method in a loop. Any issue with mutable KV cache updates, start_pos
   handling, or the TDT duration logic could produce all-blank output even if the
   encoder is correct.

4. **cuBLAS library mismatch.** On Jetson Orin, the pip-installed `nvidia-cublas-cu12`
   package may shadow the system cuBLAS (`/usr/local/cuda-12.6/lib64/`). The pip
   version can fail `cublasCreate_v2` or produce wrong results. Fix: uninstall
   `nvidia-cublas-cu12` so the system library is used, or preload the system
   cuBLAS before importing PyTorch.

Next step: dump the encoder output from both paths and compare numerically to isolate
whether the issue is in the encoder, decoder, or decode logic.
