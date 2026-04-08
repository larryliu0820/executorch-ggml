#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <ggml.h>
#include <ggml-backend.h>

namespace executorch_ggml {

// Constant tensor data saved during init(), keyed by IR tensor id.
struct SavedConstant {
  int ir_tensor_id;
  std::vector<uint8_t> data;
};

// Cached IR tensor — parsed once from FlatBuffer, reused on every rebuild.
// Eliminates FlatBuffer random access from Phase B (~1.7ms → ~0.3ms).
struct CachedIRTensor {
  int id;
  int op;            // ggml_ir::OpCode as int
  int ir_type;       // ggml_ir::TensorType as int
  int64_t ne[4];     // base shape from IR (before sym_dim override)
  int n_dims;
  std::vector<int> src_ids;
  bool is_input;
  bool is_output;
  int input_index;
  std::string data_key;
  std::vector<uint8_t> op_params;
  uint8_t elem_size;  // serialized element size for eager data estimate
  // Symbolic dimension info for dynamic shapes
  int32_t sym_dim_ids[4];    // -1 = static, -2 = expr, >=0 = sym_id
  bool has_sym_dims;
  // Per-dim bytecode for sym_dim_ids == -2 (expression)
  struct DimExpr { std::vector<uint8_t> bytecode; };
  DimExpr sym_dim_exprs[4];
};

// Per-graph instance -- owns context, graph, and tensor bookkeeping for one
// shape configuration (e.g. decode T_q=1, prefill T_q=N).
// Eager constant: a leaf tensor with data computed during build_graph
// (e.g. bool masks, scalar constants) that doesn't live in const_buf/mutable_buf.
// The data lives in the ggml context pool (allocated inline with no_alloc=false).
// We save the source pointer so we can restore it after scheduler allocation
// reassigns tensor data/buffer pointers.
struct EagerConstant {
  struct ggml_tensor* tensor;
  const void* ctx_data;  // pointer into ggml context pool (valid until ggml_free)
  size_t nbytes;
};

// Maps a ggml_tensor* to its shared buffer assignment.
struct SharedLeaf {
  struct ggml_tensor* tensor;
  ggml_backend_buffer_t buf;
  size_t offset;
};

// Hash a flattened input shape vector (ne[] per input, 4 dims each) into
// a single size_t key for the graph cache.
static size_t hash_shape_key(const std::vector<int64_t>& ne) {
  size_t h = 0;
  for (auto v : ne) {
    h ^= std::hash<int64_t>{}(v) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
  }
  return h;
}

// Patchable tensor for per-call updates without graph rebuild.
// Used for both K/V views (auto-slice) and dynamic masks.
struct PatchableKVView {
  struct ggml_tensor* tensor;  // K/V view or F16 mask tensor
  int64_t max_ne1;             // full cache size (for position filtering)
};

struct GraphInstance {
  struct ggml_context* ctx = nullptr;
  struct ggml_cgraph* graph = nullptr;
  ggml_backend_sched_t sched = nullptr;   // per-graph scheduler (avoids state leaks)
  std::vector<struct ggml_tensor*> inputs;
  std::vector<struct ggml_tensor*> outputs;
  std::vector<std::pair<struct ggml_tensor*, struct ggml_tensor*>> deferred_i64_to_i32;
  std::vector<EagerConstant> eager_constants;
  std::vector<SharedLeaf> shared_leaves;  // leaf tensors in const_buf / mutable_buf
  std::vector<struct ggml_tensor*> cpu_pinned;  // tensors pinned to CPU backend
  std::vector<PatchableKVView> kv_views;  // K/V views patched to kv_valid_len per-call
  ggml_backend_buffer_t eager_const_buf = nullptr;  // separate buffer for eager constants
  ggml_backend_buffer_t host_buf = nullptr;  // temporary CPU buffer for leaf data (kept alive for eager const ctx_data)
  int64_t padded_kv = 0;  // current 256-aligned KV slice size (0 = not auto-slice)
  bool is_allocated = false;  // true after first successful sched_alloc
  bool has_input_derived_eager = false;  // true if any eager constant depends on input data
};

// Per-input data blob for pre-populating input tensors during rebuild.
// When provided, eager ops (comparison, bitwise, etc.) that read from
// upstream tensors will see correct input data instead of uninitialized memory.
struct InputDataOverride {
  const void* data;
  size_t nbytes;
  int et_scalar_type;  // executorch::aten::ScalarType enum value
};

struct GgmlDelegateHandle {
  // Primary backend (GPU when available, otherwise CPU)
  ggml_backend_t backend = nullptr;
  // Dedicated CPU backend for scheduler fallback / custom ops
  ggml_backend_t backend_cpu = nullptr;
  int n_threads = 1;

  // --- Dedicated buffers -- outside scheduler's pool ---
  // Immutable weights, RoPE freqs, etc. Loaded once at init, never touched
  // by the scheduler. Shared by all graph instances.
  ggml_backend_buffer_t const_buf = nullptr;
  // KV caches. Updated by compute (ggml_set_rows / UPDATE_CACHE). Persists
  // across calls. Shared by all graph instances.
  ggml_backend_buffer_t mutable_buf = nullptr;

  // Per-leaf-tensor buffer assignments (ir_tensor_id -> {buffer, offset, nbytes}).
  // Used by build_graph() to point leaf tensors at the right region in
  // const_buf or mutable_buf instead of allocating from the scheduler pool.
  struct BufSlot {
    ggml_backend_buffer_t buf;
    size_t offset;
    size_t nbytes;
  };
  std::unordered_map<int, BufSlot> leaf_buf_map;

  // --- Fields for graph rebuild ---

  // Copy of the serialized IR FlatBuffer so build_graph() can re-parse it
  // without needing the original `processed` buffer.
  std::vector<uint8_t> ir_copy;

  // Constant data extracted from NamedDataMap during init().
  // Used to populate const_buf and mutable_buf during init().
  std::vector<SavedConstant> constant_data;

  // True if any tensor has sym_dim_ids (enables shape-change detection).
  bool has_dynamic = false;

  // Per-input sym_dim_ids (ggml ne order, padded to 4).
  // input_sym_dim_ids[i] has 4 int32_t values for input i (-1 = static).
  std::vector<std::vector<int32_t>> input_sym_dim_ids;

  // --- Shape-keyed graph cache ---
  // Maps input shape signature (flattened ne[] per input) -> GraphInstance.
  // Each unique combination of input shapes gets its own graph + scheduler,
  // avoiding stale state when switching between shapes.
  std::unordered_map<size_t, std::unique_ptr<GraphInstance>> graph_cache;
  GraphInstance* active = nullptr;
  std::vector<int64_t> init_ne;     // input shapes from the initial (init-time) graph

  // Cached IR: parsed once from FlatBuffer, reused on all subsequent builds.
  std::vector<CachedIRTensor> ir_cache;
};

} // namespace executorch_ggml
