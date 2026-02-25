/**
 * GgmlBackendInterface — ExecuTorch runtime backend delegating to ggml.
 *
 * Deserialises a FlatBuffer-encoded ggml IR graph produced by the Python
 * GgmlBackend.preprocess() and executes it using the ggml compute graph API.
 */

#include "ggml_backend.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ggml_ir_generated.h"  // flatc-generated header (checked in under schema/)

#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-alloc.h>
#include <ggml-cpu.h>
#include <ggml-backend.h>

#ifdef GGML_USE_METAL
#include <ggml-metal.h>
#endif

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>

namespace executorch_ggml {

using executorch::runtime::ArrayRef;
using executorch::runtime::BackendExecutionContext;
using executorch::runtime::BackendInitContext;
using executorch::runtime::CompileSpec;
using executorch::runtime::DelegateHandle;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::Result;
using executorch::runtime::Span;

namespace {

constexpr int64_t kInvalidIndex = std::numeric_limits<int64_t>::min();

// Read one index value from an index tensor at the broadcasted output coords.
inline int64_t read_index_value(
    const struct ggml_tensor* idx,
    const int64_t coords[4]) {
  size_t off = 0;
  for (int ax = 0; ax < 4; ++ax) {
    const int64_t ne = idx->ne[ax];
    int64_t c = coords[ax];
    if (ne == 1) {
      c = 0;
    } else if (c < 0 || c >= ne) {
      return kInvalidIndex;
    }
    off += static_cast<size_t>(c) * static_cast<size_t>(idx->nb[ax]);
  }

  const char* base = static_cast<const char*>(idx->data);
  if (base == nullptr) {
    return kInvalidIndex;
  }

  switch (idx->type) {
    case GGML_TYPE_I32:
      return static_cast<int64_t>(*reinterpret_cast<const int32_t*>(base + off));
    case GGML_TYPE_I64:
      return *reinterpret_cast<const int64_t*>(base + off);
    case GGML_TYPE_F32:
      return static_cast<int64_t>(*reinterpret_cast<const float*>(base + off));
    case GGML_TYPE_F16:
      return static_cast<int64_t>(
          ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t*>(base + off)));
    case GGML_TYPE_BF16:
      return static_cast<int64_t>(
          ggml_bf16_to_fp32(*reinterpret_cast<const ggml_bf16_t*>(base + off)));
    default:
      return kInvalidIndex;
  }
}

// Custom runtime gather for multi-index advanced indexing:
//   out = src[idx0, idx1, ...]
// where dst->src[0] is src and dst->src[1..] are index tensors.
void ggml_custom_index_multi(
    struct ggml_tensor* dst,
    int ith,
    int nth,
    void* userdata) {
  (void)userdata;

  const struct ggml_tensor* src = dst->src[0];
  if (src == nullptr || src->data == nullptr || dst->data == nullptr) {
    return;
  }

  int n_indices = 0;
  for (int s = 1; s < GGML_MAX_SRC && dst->src[s] != nullptr; ++s) {
    ++n_indices;
  }
  if (n_indices <= 0 || n_indices > 4) {
    return;
  }

  // Keep implementation conservative for now: plain element-wise copy path.
  if (ggml_is_quantized(src->type) || src->type != dst->type) {
    return;
  }

  const size_t elem_size = ggml_type_size(src->type);
  const int64_t n_out = ggml_nelements(dst);
  const int64_t dr = (n_out + nth - 1) / nth;
  const int64_t ir0 = dr * ith;
  const int64_t ir1 = std::min<int64_t>(ir0 + dr, n_out);

  const char* src_base = static_cast<const char*>(src->data);
  char* dst_base = static_cast<char*>(dst->data);

  for (int64_t i = ir0; i < ir1; ++i) {
    int64_t tmp = i;
    int64_t coords[4] = {0, 0, 0, 0};
    coords[0] = tmp % dst->ne[0];
    tmp /= dst->ne[0];
    coords[1] = tmp % dst->ne[1];
    tmp /= dst->ne[1];
    coords[2] = tmp % dst->ne[2];
    tmp /= dst->ne[2];
    coords[3] = tmp;

    size_t dst_off = 0;
    for (int ax = 0; ax < 4; ++ax) {
      dst_off += static_cast<size_t>(coords[ax]) * static_cast<size_t>(dst->nb[ax]);
    }

    size_t src_off = 0;
    bool valid = true;
    for (int k = 0; k < n_indices; ++k) {
      const struct ggml_tensor* idx = dst->src[1 + k];
      const int64_t idx_val_raw = read_index_value(idx, coords);
      if (idx_val_raw == kInvalidIndex) {
        valid = false;
        break;
      }

      const int src_ax = n_indices - 1 - k; // ggml axis for PyTorch dim k
      const int64_t dim_size = src->ne[src_ax];
      int64_t idx_val = idx_val_raw;
      if (idx_val < 0) {
        idx_val += dim_size;
      }
      if (idx_val < 0 || idx_val >= dim_size) {
        valid = false;
        break;
      }
      src_off += static_cast<size_t>(idx_val) * static_cast<size_t>(src->nb[src_ax]);
    }

    if (valid) {
      memcpy(dst_base + dst_off, src_base + src_off, elem_size);
    } else {
      memset(dst_base + dst_off, 0, elem_size);
    }
  }
}

// Eager integer casts on the CPU context.  ggml's CPU backend doesn't support
// GGML_OP_CPY for I64 or I32↔I64, so we must perform these conversions at
// graph-build time (data is available because we use no_alloc=false).
// Returns a new tensor with op=GGML_OP_NONE (treated as a constant leaf).
struct ggml_tensor* eager_cast_i64_to_i32(
    struct ggml_context* ctx,
    struct ggml_tensor* src) {
  struct ggml_tensor* dst = ggml_new_tensor(ctx, GGML_TYPE_I32, GGML_MAX_DIMS, src->ne);
  dst->op = GGML_OP_NONE;
  if (src->data && dst->data) {
    const size_t n = ggml_nelements(src);
    const int64_t* s = static_cast<const int64_t*>(src->data);
    int32_t* d = static_cast<int32_t*>(dst->data);
    for (size_t i = 0; i < n; ++i) d[i] = static_cast<int32_t>(s[i]);
  }
  return dst;
}

struct ggml_tensor* eager_cast_i32_to_i64(
    struct ggml_context* ctx,
    struct ggml_tensor* src) {
  struct ggml_tensor* dst = ggml_new_tensor(ctx, GGML_TYPE_I64, GGML_MAX_DIMS, src->ne);
  dst->op = GGML_OP_NONE;
  if (src->data && dst->data) {
    const size_t n = ggml_nelements(src);
    const int32_t* s = static_cast<const int32_t*>(src->data);
    int64_t* d = static_cast<int64_t*>(dst->data);
    for (size_t i = 0; i < n; ++i) d[i] = static_cast<int64_t>(s[i]);
  }
  return dst;
}

// Safe cast that avoids ggml_cast combos the CPU backend can't handle.
// Supported natively: F16↔{F16,BF16,F32}, BF16↔{F16,BF16,F32}, F32↔{F16,BF16,F32,I32}, I32→F32.
// Everything else routes through F32 as intermediate or uses eager helpers.
struct ggml_tensor* safe_ggml_cast(
    struct ggml_context* ctx,
    struct ggml_tensor* src,
    ggml_type target) {
  if (src->type == target) return src;
  // I64 source: eager CPU conversion
  if (src->type == GGML_TYPE_I64 && target == GGML_TYPE_I32) return eager_cast_i64_to_i32(ctx, src);
  if (src->type == GGML_TYPE_I64) {
    // I64 → F32 eager, then F32 → target via ggml_cast
    auto* i32 = eager_cast_i64_to_i32(ctx, src);
    return (target == GGML_TYPE_F32) ? safe_ggml_cast(ctx, i32, GGML_TYPE_F32)
                                     : safe_ggml_cast(ctx, safe_ggml_cast(ctx, i32, GGML_TYPE_F32), target);
  }
  // I32 source: only I32→F32 is native in ggml_cast
  if (src->type == GGML_TYPE_I32 && target == GGML_TYPE_I64) return eager_cast_i32_to_i64(ctx, src);
  if (src->type == GGML_TYPE_I32 && target != GGML_TYPE_F32) {
    return safe_ggml_cast(ctx, safe_ggml_cast(ctx, src, GGML_TYPE_F32), target);
  }
  // Everything else (F16/BF16/F32 inter-conversions) is natively supported
  return ggml_cast(ctx, src, target);
}

} // namespace

// ---------------------------------------------------------------------------
// Delegate handle — owns the ggml context, graph, and tensor bookkeeping
// ---------------------------------------------------------------------------

// Constant tensor data saved during init(), keyed by IR tensor id.
struct SavedConstant {
  int ir_tensor_id;
  std::vector<uint8_t> data;
};

struct GgmlDelegateHandle {
  struct ggml_context* ctx = nullptr;
  struct ggml_cgraph* graph = nullptr;

  // Ordered by input_index
  std::vector<struct ggml_tensor*> inputs;
  // Output tensors in the order they appear in the IR
  std::vector<struct ggml_tensor*> outputs;

  // Deferred I64→I32 casts that must run during execute() (after input copy).
  // Each pair is (src_i64_tensor, dst_i32_tensor).
  std::vector<std::pair<struct ggml_tensor*, struct ggml_tensor*>> deferred_i64_to_i32;

  // Backend compute resources (CPU, CUDA, or Metal)
  ggml_backend_t backend = nullptr;
  ggml_gallocr_t galloc = nullptr;
  int n_threads = 1;

  // --- Fields for graph rebuild ---

  // Copy of the serialized IR FlatBuffer so build_graph() can re-parse it
  // without needing the original `processed` buffer.
  std::vector<uint8_t> ir_copy;

  // Constant data extracted from NamedDataMap during init().
  // build_graph() restores these into the ggml context on each rebuild.
  std::vector<SavedConstant> constant_data;

  // True if any input has dynamic_dims (enables shape-change detection).
  bool has_dynamic = false;

  // Per-input dynamic_dims flags (ggml ne order, padded to 4).
  // input_dynamic_dims[i] has 4 bools for input i.
  std::vector<std::vector<bool>> input_dynamic_dims;

  // Last-seen input shapes (ggml ne order, 4 values per input, flattened).
  // Used to detect when a rebuild is needed.
  std::vector<int64_t> last_input_ne;
};

// ---------------------------------------------------------------------------
// build_graph() — (re)build ggml context + compute graph from IR
// ---------------------------------------------------------------------------

// Rebuild the ggml compute graph from the serialized IR in handle->ir_copy.
//
// On success, updates:
//   handle->ctx, graph, inputs, outputs, deferred_i64_to_i32, galloc
//
// Frees any previously existing ctx/galloc (safe to call on first build too,
// since those start as nullptr).
//
// input_ne_overrides: nullptr on first call (use serialized shapes).
//   In M2+ this will carry runtime shapes for dynamic dims.
// n_overrides: number of int64_t values (n_inputs × 4).
static Error build_graph(
    GgmlDelegateHandle* handle,
    const int64_t* input_ne_overrides,
    size_t n_overrides) {

  (void)n_overrides;  // used only for debug assertions

  // --- Tear down previous graph (no-op on first call) ---
  if (handle->galloc) {
    ggml_gallocr_free(handle->galloc);
    handle->galloc = nullptr;
  }
  if (handle->ctx) {
    ggml_free(handle->ctx);
    handle->ctx = nullptr;
  }
  handle->graph = nullptr;
  handle->inputs.clear();
  handle->outputs.clear();
  handle->deferred_i64_to_i32.clear();

  // --- Parse IR from the saved copy ---
  const auto* fb_graph = ggml_ir::GetGgmlGraph(handle->ir_copy.data());
  if (!fb_graph || !fb_graph->tensors()) {
    return Error::InvalidArgument;
  }
  const auto* fb_tensors = fb_graph->tensors();
  const int n_tensors = static_cast<int>(fb_tensors->size());

  // === Phase A: calculate ctx size, create ggml context ===
  // Calculate memory needed for ggml context.
  // Sum up actual constant tensor sizes from the IR so we allocate exactly
  // what is needed, plus a fixed overhead for graph bookkeeping.
  size_t constant_data_size = 0;
  for (int i = 0; i < n_tensors; ++i) {
    const auto* t = fb_tensors->Get(i);
    // Only leaf tensors with data_key hold constant data in the ggml context.
    if (static_cast<ggml_ir::OpCode>(t->op()) != ggml_ir::OpCode::NONE) continue;
    if (!t->data_key() || std::strlen(t->data_key()->c_str()) == 0) continue;

    // Compute byte size: product of ne[] dims × element size for the type.
    size_t n_elems = 1;
    if (t->ne()) {
      for (size_t d = 0; d < t->ne()->size(); ++d) {
        n_elems *= static_cast<size_t>(t->ne()->Get(d));
      }
    }
    size_t elem_size = 4; // default F32
    switch (static_cast<ggml_ir::TensorType>(t->type())) {
      case ggml_ir::TensorType::F16:  elem_size = 2; break;
      case ggml_ir::TensorType::BF16: elem_size = 2; break;
      case ggml_ir::TensorType::I32:  elem_size = 4; break;
      case ggml_ir::TensorType::I64:  elem_size = 8; break;
      case ggml_ir::TensorType::BOOL: elem_size = 4; break; // stored as I32
      default:                        elem_size = 4; break;
    }
    constant_data_size += n_elems * elem_size;
  }

  // Estimate graph size: 4× the IR tensors to account for compound ops.
  size_t est_graph_size = static_cast<size_t>(n_tensors) * 4;
  if (est_graph_size < GGML_DEFAULT_GRAPH_SIZE) {
    est_graph_size = GGML_DEFAULT_GRAPH_SIZE;
  }

  size_t ctx_size =
      static_cast<size_t>(n_tensors) * ggml_tensor_overhead() +
      constant_data_size +
      ggml_graph_overhead_custom(est_graph_size, false) +
      4ull * 1024 * 1024 * 1024;  // 4GB headroom for intermediate op tensors and graph nodes

  struct ggml_init_params params = {
      /* .mem_size   = */ ctx_size,
      /* .mem_buffer = */ nullptr,
      /* .no_alloc   = */ false,
  };

  struct ggml_context* ctx = ggml_init(params);
  if (!ctx) {
    return Error::MemoryAllocationFailed;
  }

  // === Phase B: tensor creation loop + op switch ===
  // Map from IR tensor id → ggml_tensor*
  std::vector<struct ggml_tensor*> id_to_tensor(n_tensors, nullptr);

  // Track inputs and outputs
  std::vector<std::pair<int, struct ggml_tensor*>> input_pairs;  // (index, tensor)
  std::vector<struct ggml_tensor*> output_tensors;
  // Deferred I64→I32 casts (for input tensors).
  std::vector<std::pair<struct ggml_tensor*, struct ggml_tensor*>> deferred_i64_to_i32;

  // Build dynamic size mapping: trace-time ne value → runtime ne value.
  // Used to fix output shapes of ops that bake in trace-time sizes
  // (ARANGE, FULL, comparison ops, etc.).
  std::unordered_map<int64_t, int64_t> dynamic_size_map;
  if (input_ne_overrides) {
    for (int i = 0; i < n_tensors; ++i) {
      const auto* ti = fb_tensors->Get(i);
      if (!ti->is_input() || !ti->dynamic_dims() || !ti->ne()) continue;
      int input_idx = ti->input_index();
      for (size_t d = 0; d < ti->dynamic_dims()->size() && d < 4; ++d) {
        if (ti->dynamic_dims()->Get(d)) {
          int64_t trace_val = ti->ne()->Get(d);
          int64_t runtime_val = input_ne_overrides[input_idx * 4 + d];
          if (trace_val != runtime_val) {
            dynamic_size_map[trace_val] = runtime_val;
          }
        }
      }
    }
  }

  // Walk tensors in topological order
  for (int i = 0; i < n_tensors; ++i) {
    const auto* t = fb_tensors->Get(i);
    const int tid = t->id();
    const auto op = static_cast<ggml_ir::OpCode>(t->op());

    // Read shape (ne)
    int64_t ne[4] = {1, 1, 1, 1};
    if (t->ne() && t->ne()->size() > 0) {
      for (size_t d = 0; d < t->ne()->size() && d < 4; ++d) {
        ne[d] = t->ne()->Get(d);
      }
    }

    // Override dynamic dims with runtime shapes for input tensors (M2).
    if (input_ne_overrides && t->is_input() && t->dynamic_dims()) {
      int input_idx = t->input_index();
      for (size_t d = 0; d < t->dynamic_dims()->size() && d < 4; ++d) {
        if (t->dynamic_dims()->Get(d)) {
          ne[d] = input_ne_overrides[input_idx * 4 + d];
        }
      }
    }

    // For op tensors, replace any dim matching a known dynamic trace-time
    // size with its runtime value. This handles ARANGE, FULL, comparison
    // ops, UNSQUEEZE, SLICE, etc. that bake output shapes from trace-time
    // values. For ops that auto-infer shapes from sources (ADD, MUL_MAT,
    // etc.), ne[] is not used in the op switch, so this is harmless.
    if (!dynamic_size_map.empty() && op != ggml_ir::OpCode::NONE) {
      for (int d = 0; d < 4; ++d) {
        auto it = dynamic_size_map.find(ne[d]);
        if (it != dynamic_size_map.end()) {
          ne[d] = it->second;
        }
      }
    }

    // Determine n_dims from ne (count trailing 1s)
    int n_dims = 4;
    for (int d = 3; d >= 1; --d) {
      if (ne[d] == 1) {
        n_dims = d;
      } else {
        break;
      }
    }
    if (n_dims == 0) n_dims = 1;

    if (op == ggml_ir::OpCode::NONE) {
      // Leaf tensor: input or constant
      struct ggml_tensor* gt = nullptr;

      // Respect declared tensor type.
      ggml_type gtype = GGML_TYPE_F32;
      switch (t->type()) {
        case ggml_ir::TensorType::F16:
          gtype = GGML_TYPE_F16;
          break;
        case ggml_ir::TensorType::I64:
          gtype = GGML_TYPE_I64;
          break;
        case ggml_ir::TensorType::I32:
          gtype = GGML_TYPE_I32;
          break;
        case ggml_ir::TensorType::BOOL:
          // Represent bool tensors as I32 in ggml to avoid unsupported ops on I8
          // (e.g. get_rows has limited support for I8).
          gtype = GGML_TYPE_I32;
          break;
        case ggml_ir::TensorType::BF16:
          gtype = GGML_TYPE_BF16;
          break;
        case ggml_ir::TensorType::F32:
        default:
          gtype = GGML_TYPE_F32;
          break;
      }

      switch (n_dims) {
        case 1:
          gt = ggml_new_tensor_1d(ctx, gtype, ne[0]);
          break;
        case 2:
          gt = ggml_new_tensor_2d(ctx, gtype, ne[0], ne[1]);
          break;
        case 3:
          gt = ggml_new_tensor_3d(ctx, gtype, ne[0], ne[1], ne[2]);
          break;
        default:
          gt = ggml_new_tensor_4d(ctx, gtype, ne[0], ne[1], ne[2], ne[3]);
          break;
      }

      // Load constant data from handle->constant_data (populated by init()).
      if (t->data_key() && t->data_key()->c_str() && std::strlen(t->data_key()->c_str()) > 0) {
        const size_t nbytes = ggml_nbytes(gt);
        for (const auto& sc : handle->constant_data) {
          if (sc.ir_tensor_id == tid) {
            memcpy(gt->data, sc.data.data(), nbytes);
            break;
          }
        }
      }

      if (t->is_input()) {
        input_pairs.emplace_back(t->input_index(), gt);
      }

      id_to_tensor[tid] = gt;

    } else {
      // Op tensor: build the ggml operation
      struct ggml_tensor* gt = nullptr;

      // Resolve sources
      std::vector<struct ggml_tensor*> srcs;
      if (t->src_ids()) {
        for (size_t s = 0; s < t->src_ids()->size(); ++s) {
          int src_id = t->src_ids()->Get(s);
          srcs.push_back(id_to_tensor[src_id]);
        }
      }

      switch (op) {
        case ggml_ir::OpCode::ADD: {
          struct ggml_tensor* a = srcs[0];
          struct ggml_tensor* b = srcs[1];

          if (a->type != b->type) {
            // Prefer casting to f32 if either side is f32.
            ggml_type tgt = (a->type == GGML_TYPE_F32 || b->type == GGML_TYPE_F32)
                                ? GGML_TYPE_F32
                                : a->type;
            if (a->type != tgt) a = safe_ggml_cast(ctx, a, tgt);
            if (b->type != tgt) b = safe_ggml_cast(ctx, b, tgt);
          }

          // Handle broadcast by repeating the smaller tensor to match the larger.
          if (!ggml_are_same_shape(a, b)) {
            auto try_repeat_1d_to_match = [&](struct ggml_tensor * small,
                                             struct ggml_tensor * big) -> struct ggml_tensor * {
              // If `small` is effectively 1D (exactly one dim > 1) and its length matches
              // one of big's dims, reshape+permute it so the matching dim aligns, then repeat.
              int non1 = 0;
              int non1_ax = -1;
              for (int ax = 0; ax < 4; ++ax) {
                if (small->ne[ax] != 1) {
                  non1++;
                  non1_ax = ax;
                }
              }
              if (non1 != 1) {
                return nullptr;
              }
              const int64_t n = small->ne[non1_ax];

              // Normalize to a base view where n is in axis0.
              struct ggml_tensor * base = small;
              if (non1_ax != 0) {
                // permute so that old non1_ax becomes new axis0
                int axes[4] = {0,1,2,3};
                axes[0] = non1_ax;
                int out = 1;
                for (int ax = 0; ax < 4; ++ax) {
                  if (ax == non1_ax) continue;
                  axes[out++] = ax;
                }
                base = ggml_permute(ctx, base, axes[0], axes[1], axes[2], axes[3]);
                base = ggml_cont(ctx, base);
              }
              for (int ax = 0; ax < 4; ++ax) {
                if (big->ne[ax] != n) {
                  continue;
                }
                // Start as 4D [n,1,1,1]
                struct ggml_tensor * s4 = ggml_reshape_4d(ctx, base, n, 1, 1, 1);
                // Permute so `n` ends up in axis `ax`
                int p0 = 0, p1 = 1, p2 = 2, p3 = 3;
                if (ax == 0) {
                  // already in axis0
                } else if (ax == 1) {
                  // move axis0 -> axis1
                  p0 = 1; p1 = 0; p2 = 2; p3 = 3;
                } else if (ax == 2) {
                  p0 = 1; p1 = 2; p2 = 0; p3 = 3;
                } else {
                  // ax == 3: move axis0 -> axis3
                  p0 = 1; p1 = 2; p2 = 3; p3 = 0;
                }
                s4 = ggml_permute(ctx, s4, p0, p1, p2, p3);
                // Make the view contiguous to satisfy ggml_can_repeat in some cases.
                s4 = ggml_cont(ctx, s4);
                if (ggml_can_repeat(s4, big)) {
                  return ggml_repeat(ctx, s4, big);
                }
              }
              return nullptr;
            };

            auto try_permute_to_match = [&](struct ggml_tensor * src,
                                           struct ggml_tensor * dst) -> struct ggml_tensor * {
              // If src and dst have the same multiset of extents, try a few permutes.
              int64_t aa[4] = {dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]};
              int64_t bb[4] = {src->ne[0], src->ne[1], src->ne[2], src->ne[3]};
              for (int i = 0; i < 4; ++i) {
                for (int j = i + 1; j < 4; ++j) {
                  if (aa[j] < aa[i]) { auto t = aa[i]; aa[i] = aa[j]; aa[j] = t; }
                  if (bb[j] < bb[i]) { auto t = bb[i]; bb[i] = bb[j]; bb[j] = t; }
                }
              }
              for (int i = 0; i < 4; ++i) {
                if (aa[i] != bb[i]) return nullptr;
              }

              const int perms[][4] = {
                {0,1,2,3},
                {3,1,2,0},
                {0,2,1,3},
                {1,0,2,3},
                {2,1,0,3},
                {1,2,3,0},
                {2,3,1,0},
                {3,2,1,0},
              };
              for (const auto & p : perms) {
                struct ggml_tensor * t = ggml_permute(ctx, src, p[0], p[1], p[2], p[3]);
                t = ggml_cont(ctx, t);
                if (ggml_are_same_shape(t, dst)) {
                  return t;
                }
              }
              return nullptr;
            };

            if (ggml_can_repeat(b, a)) {
              b = ggml_cont(ctx, ggml_repeat(ctx, b, a));
            } else if (ggml_can_repeat(a, b)) {
              a = ggml_cont(ctx, ggml_repeat(ctx, a, b));
            } else {
              // Try 1D→ND broadcast alignment
              if (auto * bb = try_repeat_1d_to_match(b, a)) {
                b = bb;
              } else if (auto * aa = try_repeat_1d_to_match(a, b)) {
                a = aa;
              } else if (auto * bp = try_permute_to_match(b, a)) {
                b = bp;
              } else if (auto * ap = try_permute_to_match(a, b)) {
                a = ap;
              } else {
                fprintf(stderr,
                        "[executorch-ggml] ADD shape mismatch not broadcastable: a=(%lld,%lld,%lld,%lld) b=(%lld,%lld,%lld,%lld)\n",
                        (long long) a->ne[0], (long long) a->ne[1], (long long) a->ne[2], (long long) a->ne[3],
                        (long long) b->ne[0], (long long) b->ne[1], (long long) b->ne[2], (long long) b->ne[3]);
                ggml_free(ctx);
                return Error::InvalidArgument;
              }
            }
          }

          // For int64 types, use custom element-wise add
          if (a->type == GGML_TYPE_I64) {
            gt = ggml_new_tensor_4d(ctx, GGML_TYPE_I64, a->ne[0], a->ne[1], a->ne[2], a->ne[3]);
            const size_t nelem = ggml_nelements(gt);
            const int64_t* a_data = static_cast<const int64_t*>(a->data);
            const int64_t* b_data = static_cast<const int64_t*>(b->data);
            int64_t* out_data = static_cast<int64_t*>(gt->data);
            for (size_t i = 0; i < nelem; ++i) out_data[i] = a_data[i] + b_data[i];
          } else {
            gt = ggml_add(ctx, a, b);
          }
          break;
        }

        case ggml_ir::OpCode::MUL_MAT:
          gt = ggml_mul_mat(ctx, srcs[0], srcs[1]);
          break;

        case ggml_ir::OpCode::MUL: {
          // ggml_mul(a, b) requires ggml_can_repeat(b, a) — b broadcasts to a.
          // Swap if needed so the larger tensor is first.
          struct ggml_tensor* a = srcs[0];
          struct ggml_tensor* b = srcs[1];
          if (ggml_nelements(a) < ggml_nelements(b)) {
            std::swap(a, b);
          }
          gt = ggml_mul(ctx, a, b);
          break;
        }

        case ggml_ir::OpCode::REPEAT: {
          // REPEAT(src, like): tile src to like's shape.
          // The "like" tensor (srcs[1]) may be a static shape-only tensor
          // from export time.  When ggml auto-computes upstream shapes
          // differently (e.g. after permute), the src shape may not match
          // the like shape.  Rebuild the like tensor from the src shape +
          // the expansion ratios from the IR ne.
          struct ggml_tensor* a = srcs[0];  // source data
          struct ggml_tensor* b = srcs[1];  // target shape ("like")

          // If shapes already match, this is a no-op.
          if (ggml_are_same_shape(a, b)) {
            gt = a;
            id_to_tensor[tid] = gt;
            break;
          }

          // Try to derive the correct target shape by computing the
          // expansion ratio from the IR ne and applying it to the source.
          if (!ggml_can_repeat(a, b)) {
            // Compute ratio from IR: for each dim where src is 1 and
            // target is >1, the ratio is target/1.  Use source's actual
            // shape as the base and apply the ratio.
            int64_t target_ne[4];
            for (int d = 0; d < 4; ++d) {
              if (a->ne[d] == 1 && b->ne[d] > 1) {
                // This dim was expanded from 1 — keep the target.
                target_ne[d] = b->ne[d];
              } else {
                // Use source's actual shape for this dim.
                target_ne[d] = a->ne[d];
              }
            }
            b = ggml_new_tensor_4d(ctx, a->type, target_ne[0], target_ne[1], target_ne[2], target_ne[3]);
          }

          if (!ggml_can_repeat(a, b)) {
            if (ggml_can_repeat(b, a)) {
              gt = ggml_repeat(ctx, b, a);
            } else {
              fprintf(stderr,
                      "[executorch-ggml] REPEAT shape not repeatable (tensor %d, srcs=[%d,%d]): a=(%lld,%lld,%lld,%lld) b=(%lld,%lld,%lld,%lld)\n",
                      (int)i,
                      (int)(t->src_ids() && t->src_ids()->size() > 0 ? t->src_ids()->Get(0) : -1),
                      (int)(t->src_ids() && t->src_ids()->size() > 1 ? t->src_ids()->Get(1) : -1),
                      (long long) a->ne[0], (long long) a->ne[1], (long long) a->ne[2], (long long) a->ne[3],
                      (long long) b->ne[0], (long long) b->ne[1], (long long) b->ne[2], (long long) b->ne[3]);
              ggml_free(ctx);
              return Error::InvalidArgument;
            }
          } else {
            gt = ggml_repeat(ctx, a, b);
          }
          break;
        }

        case ggml_ir::OpCode::NEG:
          gt = ggml_neg(ctx, srcs[0]);
          break;

        case ggml_ir::OpCode::RSQRT: {
          // ggml doesn't expose rsqrt directly in this version; implement as 1/sqrt(x)
          // ggml_div(a, b) requires b to be repeatable to a's shape.
          // Since we want 1/sqrt(x), we need a to have x's shape and be filled with 1.0.
          struct ggml_tensor* sx  = ggml_sqrt(ctx, srcs[0]);
          struct ggml_tensor* one = ggml_new_f32(ctx, 1.0f);
          // Repeat the scalar 1.0 to match sqrt's shape
          struct ggml_tensor* one_rep = ggml_repeat(ctx, one, sx);
          gt = ggml_div(ctx, one_rep, sx);
          break;
        }

        case ggml_ir::OpCode::SILU:
          gt = ggml_silu(ctx, srcs[0]);
          break;

        case ggml_ir::OpCode::LINEAR: {
          // src_ids: [x, w, b?] where w is [out, in] in PyTorch.
          // Our IR stores shapes in ggml order, so weight ne is [in, out].
          // Compute: y = w @ x, then add bias if present.
          struct ggml_tensor* x = srcs[0];
          struct ggml_tensor* w = srcs[1];
          struct ggml_tensor* y = ggml_mul_mat(ctx, w, x);
          if (srcs.size() > 2) {
            struct ggml_tensor* b = srcs[2];
            // Broadcast bias to y shape.
            // For PyTorch linear, bias is typically 1D [out]. In ggml order that is ne0=out.
            // Make it at least 2D so ggml_repeat can broadcast along remaining dims.
            if (ggml_n_dims(b) == 1) {
              // Prefer a 4D shape to be safely repeatable against 2D/3D/4D outputs.
              b = ggml_reshape_4d(ctx, b, b->ne[0], 1, 1, 1);
            }
            if (!ggml_can_repeat(b, y)) {
              fprintf(stderr,
                      "[executorch-ggml] LINEAR bias not repeatable: b=(%lld,%lld,%lld,%lld) y=(%lld,%lld,%lld,%lld)\n",
                      (long long) b->ne[0], (long long) b->ne[1], (long long) b->ne[2], (long long) b->ne[3],
                      (long long) y->ne[0], (long long) y->ne[1], (long long) y->ne[2], (long long) y->ne[3]);
              ggml_free(ctx);
              return Error::InvalidArgument;
            }
            struct ggml_tensor* b_rep = ggml_repeat(ctx, b, y);
            y = ggml_add(ctx, y, b_rep);
          }
          gt = y;
          break;
        }

        case ggml_ir::OpCode::EMBEDDING: {
          // src_ids: [weight, indices]
          struct ggml_tensor* w = srcs[0];
          struct ggml_tensor* idx = srcs[1];
          if (idx->type == GGML_TYPE_I64) {
            idx = eager_cast_i64_to_i32(ctx, idx);
          } else if (idx->type != GGML_TYPE_I32) {
            idx = safe_ggml_cast(ctx, idx, GGML_TYPE_I32);
          }
          gt = ggml_get_rows(ctx, w, idx);
          break;
        }

        case ggml_ir::OpCode::LEAKY_RELU: {
          float negative_slope = 0.01f;
          if (t->op_params() && t->op_params()->size() >= 4) {
            memcpy(&negative_slope, t->op_params()->data(), sizeof(float));
          }
          gt = ggml_leaky_relu(ctx, srcs[0], negative_slope, false);
          break;
        }

        case ggml_ir::OpCode::CONV_2D:
        case ggml_ir::OpCode::CONV_2D_DW: {
          // src_ids: [weight, input, bias?]
          // op_params: stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups
          if (srcs.size() < 2) {
            ggml_free(ctx);
            return Error::InvalidArgument;
          }

          struct ggml_tensor* weight = srcs[0];
          struct ggml_tensor* input = srcs[1];
          struct ggml_tensor* bias = srcs.size() > 2 ? srcs[2] : nullptr;

          // (debug logging removed)

          // Parse op_params
          int32_t stride_h = 1, stride_w = 1;
          int32_t pad_h = 0, pad_w = 0;
          int32_t dilation_h = 1, dilation_w = 1;
          int32_t groups = 1;

          if (t->op_params() && t->op_params()->size() >= 28) {
            const uint8_t* data = t->op_params()->data();
            memcpy(&stride_h, data, sizeof(int32_t));
            memcpy(&stride_w, data + 4, sizeof(int32_t));
            memcpy(&pad_h, data + 8, sizeof(int32_t));
            memcpy(&pad_w, data + 12, sizeof(int32_t));
            memcpy(&dilation_h, data + 16, sizeof(int32_t));
            memcpy(&dilation_w, data + 20, sizeof(int32_t));
            memcpy(&groups, data + 24, sizeof(int32_t));
          }

          // ggml API signature:
          // ggml_conv_2d(ctx, weight, input, s0, s1, p0, p1, d0, d1)
          // ggml_conv_2d_dw(ctx, weight, input, s0, s1, p0, p1, d0, d1)
          // Note: groups parameter not directly supported in ggml API
          // For depthwise conv (groups > 1), use ggml_conv_2d_dw
          // For regular conv (groups == 1), use ggml_conv_2d
          if (op == ggml_ir::OpCode::CONV_2D_DW || groups > 1) {
            // Depthwise convolution
            gt = ggml_conv_2d_dw(ctx, weight, input, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w);
          } else {
            // Regular convolution
            gt = ggml_conv_2d(ctx, weight, input, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w);
          }

          // Cast conv output to f32 to keep the rest of the graph in f32 and
          // avoid mixed-type binary ops (e.g. residual adds).
          if (gt && gt->type == GGML_TYPE_F16) {
            gt = safe_ggml_cast(ctx, gt, GGML_TYPE_F32);
          }

          // Add bias if present.
          // Conv bias in PyTorch is 1D [Cout]. ggml_add requires broadcastable
          // shapes, so reshape+repeat the bias to match conv output.
          if (bias && gt) {
            // conv output layout is [W, H, Cout, N]
            struct ggml_tensor* bias4 = ggml_reshape_4d(ctx, bias, 1, 1, bias->ne[0], 1);
            if (!ggml_can_repeat(bias4, gt)) {
              fprintf(stderr,
                      "[executorch-ggml] CONV bias not repeatable: b=(%lld,%lld,%lld,%lld) y=(%lld,%lld,%lld,%lld)\n",
                      (long long) bias4->ne[0], (long long) bias4->ne[1], (long long) bias4->ne[2], (long long) bias4->ne[3],
                      (long long) gt->ne[0], (long long) gt->ne[1], (long long) gt->ne[2], (long long) gt->ne[3]);
              ggml_free(ctx);
              return Error::InvalidArgument;
            }
            struct ggml_tensor* bias_rep = ggml_repeat(ctx, bias4, gt);
            if (bias_rep->type == GGML_TYPE_F16) {
              bias_rep = safe_ggml_cast(ctx, bias_rep, GGML_TYPE_F32);
            }
            gt = ggml_add(ctx, gt, bias_rep);
          }
          break;
        }

        case ggml_ir::OpCode::HARDTANH: {
          // hardtanh(x, min_val, max_val) -> clamp(x, min, max)
          float min_val = -1.0f;
          float max_val = 1.0f;
          if (t->op_params() && t->op_params()->size() >= 8) {
            const uint8_t* data = t->op_params()->data();
            memcpy(&min_val, data, sizeof(float));
            memcpy(&max_val, data + 4, sizeof(float));
          }
          gt = ggml_clamp(ctx, srcs[0], min_val, max_val);
          break;
        }

        case ggml_ir::OpCode::MEAN: {
          // mean(x, dims)
          // op_params: ndims (int32) + dims[] (int32)
          int32_t ndims = 0;
          int32_t dims[4] = {0, 0, 0, 0};
          if (t->op_params() && t->op_params()->size() >= 4) {
            const uint8_t* data = t->op_params()->data();
            memcpy(&ndims, data, sizeof(int32_t));
            if (ndims < 0 || ndims > 4) {
              ggml_free(ctx);
              return Error::InvalidArgument;
            }
            if (t->op_params()->size() < static_cast<size_t>(4 + ndims * 4)) {
              ggml_free(ctx);
              return Error::InvalidArgument;
            }
            for (int i = 0; i < ndims; ++i) {
              memcpy(&dims[i], data + 4 + i * 4, sizeof(int32_t));
            }
          }

          // Special-case global average pool for NCHW tensors: mean over (H, W)
          // which is dims (2, 3) in PyTorch order.
          if (ndims == 2 && dims[0] == 2 && dims[1] == 3) {
            // ggml tensor layout for NCHW is [W, H, C, N]
            const int k0 = static_cast<int>(srcs[0]->ne[0]);
            const int k1 = static_cast<int>(srcs[0]->ne[1]);
            gt = ggml_pool_2d(
                ctx,
                srcs[0],
                GGML_OP_POOL_AVG,
                k0,
                k1,
                k0,
                k1,
                0.0f,
                0.0f);
          } else {
            // Fallback: ggml_mean is all-dims mean.
            // TODO: implement axis-specific mean in IR/runtime.
            gt = ggml_mean(ctx, srcs[0]);
          }
          break;
        }

        case ggml_ir::OpCode::VIEW: {
          // reshape(x, new_shape)
          // Parse new shape from op_params
          if (!t->op_params() || t->op_params()->size() < 4) {
            ggml_free(ctx);
            return Error::InvalidArgument;
          }

          const uint8_t* data = t->op_params()->data();
          int32_t ndims = 0;
          memcpy(&ndims, data, sizeof(int32_t));

          if (t->op_params()->size() < 4 + ndims * 8) {
            ggml_free(ctx);
            return Error::InvalidArgument;
          }

          int64_t new_ne[4] = {1, 1, 1, 1};
          for (int32_t i = 0; i < ndims && i < 4; ++i) {
            memcpy(&new_ne[i], data + 4 + i * 8, sizeof(int64_t));
          }

          // Apply dynamic_size_map to the VIEW target shape.
          // The op_params store trace-time dims that may include dynamic
          // values (e.g. seq_len).  Replace them with runtime values.
          if (!dynamic_size_map.empty()) {
            for (int d = 0; d < 4; ++d) {
              auto it = dynamic_size_map.find(new_ne[d]);
              if (it != dynamic_size_map.end()) {
                new_ne[d] = it->second;
              }
            }
          }

          // If source numel differs from the baked-in view shape (dynamic
          // shapes changed an upstream dim), infer the one mismatched dim.
          int64_t src_numel = ggml_nelements(srcs[0]);
          int64_t view_numel = new_ne[0] * new_ne[1] * new_ne[2] * new_ne[3];
          if (src_numel != view_numel && view_numel > 0) {
            for (int d = 0; d < 4; ++d) {
              if (new_ne[d] <= 1) continue;
              int64_t others = view_numel / new_ne[d];
              if (others > 0 && src_numel % others == 0) {
                int64_t inferred = src_numel / others;
                if (inferred != new_ne[d]) {
                  new_ne[d] = inferred;
                  break;
                }
              }
            }
          }

          gt = ggml_reshape_4d(ctx, srcs[0], new_ne[0], new_ne[1], new_ne[2], new_ne[3]);
          gt = ggml_cont(ctx, gt);
          break;
        }

        case ggml_ir::OpCode::PERMUTE: {
          // permute(x, dims)
          if (!t->op_params() || t->op_params()->size() < 4) {
            ggml_free(ctx);
            return Error::InvalidArgument;
          }

          const uint8_t* data = t->op_params()->data();
          int32_t ndims = 0;
          memcpy(&ndims, data, sizeof(int32_t));

          if (t->op_params()->size() < 4 + ndims * 4) {
            ggml_free(ctx);
            return Error::InvalidArgument;
          }

          int32_t perm[4] = {0, 1, 2, 3};
          for (int32_t i = 0; i < ndims && i < 4; ++i) {
            memcpy(&perm[i], data + 4 + i * 4, sizeof(int32_t));
          }

          // ggml_permute(ctx, tensor, axis0, axis1, axis2, axis3)
          gt = ggml_permute(ctx, srcs[0], perm[0], perm[1], perm[2], perm[3]);
          // Make contiguous so that output copy works correctly.
          gt = ggml_cont(ctx, gt);
          break;
        }

        case ggml_ir::OpCode::TRANSPOSE: {
          // transpose(x, dim0, dim1) via permute
          // op_params: (dim0:int32, dim1:int32, ndim:int32)
          if (!t->op_params() || t->op_params()->size() < 12) {
            ggml_free(ctx);
            return Error::InvalidArgument;
          }
          int32_t dim0 = 0, dim1 = 1, nd = 4;
          memcpy(&dim0, t->op_params()->data(), 4);
          memcpy(&dim1, t->op_params()->data() + 4, 4);
          memcpy(&nd, t->op_params()->data() + 8, 4);

          // Map PyTorch dims -> ggml axes using the PyTorch rank (not ggml_n_dims)
          int ax0 = (nd - 1) - dim0;
          int ax1 = (nd - 1) - dim1;
          int perm[4] = {0, 1, 2, 3};
          int tmp = perm[ax0];
          perm[ax0] = perm[ax1];
          perm[ax1] = tmp;
          gt = ggml_permute(ctx, srcs[0], perm[0], perm[1], perm[2], perm[3]);
          // Make contiguous so that output copy works correctly.
          gt = ggml_cont(ctx, gt);
          break;
        }

        case ggml_ir::OpCode::UNSQUEEZE: {
          // Unsqueeze adds a dim of size 1, total elements don't change.
          // Use the source tensor's actual shape rather than the IR's ne
          // (which may have stale trace-time dims after dynamic_size_map).
          struct ggml_tensor* a = srcs[0];
          int64_t uns_ne[4] = {a->ne[0], a->ne[1], a->ne[2], a->ne[3]};
          // Read which PyTorch dim to unsqueeze from op_params.
          int32_t pt_dim = 0;
          if (t->op_params() && t->op_params()->size() >= 4) {
            memcpy(&pt_dim, t->op_params()->data(), sizeof(int32_t));
          }
          // In ggml order (reversed from PyTorch), unsqueezing pt_dim D
          // means inserting 1 at ggml axis (ndim - D).  For a 3D→4D
          // unsqueeze, shift dims above the insertion point up by one.
          // Simpler: just use IR ne but ensure numel matches source.
          int64_t src_numel = ggml_nelements(a);
          int64_t ir_numel = ne[0] * ne[1] * ne[2] * ne[3];
          if (src_numel == ir_numel) {
            gt = ggml_reshape_4d(ctx, a, ne[0], ne[1], ne[2], ne[3]);
          } else {
            // IR ne is stale — reconstruct from source shape by inserting
            // a 1 dim.  Since we don't know which ggml axis corresponds to
            // the PyTorch dim, just preserve the source shape (unsqueeze
            // only adds a trailing 1 which ggml 4D already has).
            gt = ggml_reshape_4d(ctx, a, a->ne[0], a->ne[1], a->ne[2], a->ne[3]);
          }
          break;
        }

        case ggml_ir::OpCode::SLICE: {
          // Limited slice support: step must be 1. Only supports slicing along
          // a single PyTorch dim using a view.
          // op_params: dim(i32), start(i64), end(i64), step(i64), ndim(u32)
          if (!t->op_params() || t->op_params()->size() < 28) {
            ggml_free(ctx);
            return Error::InvalidArgument;
          }
          int32_t dim = 0;
          int64_t start = 0, end = 0, step = 1;
          const uint8_t* p = t->op_params()->data();
          memcpy(&dim, p, 4);
          memcpy(&start, p + 4, 8);
          memcpy(&end, p + 12, 8);
          memcpy(&step, p + 20, 8);
          if (step != 1) {
            ggml_free(ctx);
            return Error::InvalidArgument;
          }

          // Read PyTorch rank from op_params (avoids ggml_n_dims which
          // undercounts when trailing dims are 1).
          uint32_t nd = 4;
          if (t->op_params()->size() >= 32) {
            memcpy(&nd, p + 28, 4);
          }
          struct ggml_tensor* a = srcs[0];
          int ax = (int(nd) - 1) - dim;

          // Clamp end to actual source dim (handles "slice to end" when
          // source shape changed due to dynamic dims).
          int64_t actual_dim = a->ne[ax];
          if (end > actual_dim) end = actual_dim;
          ne[ax] = end - start;

          size_t offset = static_cast<size_t>(start) * a->nb[ax];
          gt = ggml_view_4d(ctx, a, ne[0], ne[1], ne[2], ne[3],
                            a->nb[1], a->nb[2], a->nb[3], offset);
          // Slicing the innermost axis produces a non-contiguous view;
          // make contiguous so downstream ops and output copy work.
          gt = ggml_cont(ctx, gt);
          break;
        }

        case ggml_ir::OpCode::CAT: {
          // Chain ggml_concat along pre-computed ggml axis
          if (!t->op_params() || t->op_params()->size() < 4) {
            ggml_free(ctx);
            return Error::InvalidArgument;
          }
          int32_t ax = 0;
          memcpy(&ax, t->op_params()->data(), 4);
          struct ggml_tensor* cur = srcs[0];
          for (size_t si = 1; si < srcs.size(); ++si) {
            cur = ggml_concat(ctx, cur, srcs[si], ax);
          }
          gt = cur;
          break;
        }

        case ggml_ir::OpCode::REPEAT_INTERLEAVE: {
          // Support for GQA expansion: repeat_interleave(x, repeats, dim).
          // op_params: (dim:int32, repeats:int32)
          // Unlike ggml_repeat (which tiles the whole tensor), repeat_interleave
          // repeats each element `repeats` times consecutively.
          // Example: [A, B, C].repeat_interleave(2) -> [A, A, B, B, C, C]
          if (!t->op_params() || t->op_params()->size() < 8) {
            ggml_free(ctx);
            return Error::InvalidArgument;
          }
          int32_t dim = 0, reps = 1;
          memcpy(&dim, t->op_params()->data(), 4);
          memcpy(&reps, t->op_params()->data() + 4, 4);
          // Only support repeating along PyTorch dim 1 (heads), which is ggml ax=(nd-2).
          if (dim != 1 || reps < 1) {
            ggml_free(ctx);
            return Error::InvalidArgument;
          }
          struct ggml_tensor* src = srcs[0];
          // src layout in ggml (ne): [ne0=D, ne1=T, ne2=H, ne3=B]
          // We need output [ne0=D, ne1=T, ne2=H*reps, ne3=B] where output
          // head `i` equals input head `i / reps` (interleaved, not tiled).
          //
          // Build by concatenating each individual head slice reps times:
          //   concat(head0, head0, head1, head1, ...) for reps=2
          //
          // ggml axis for head dim: for a 4D src, ne[2] is the head dim.
          int64_t D  = src->ne[0];
          int64_t T  = src->ne[1];
          int64_t H  = src->ne[2];
          int64_t B  = src->ne[3];
          // byte stride for one head slice along ne[2]
          size_t  row_stride  = src->nb[1];   // stride over T (ne[1])
          size_t  head_stride = src->nb[2];   // stride over H (ne[2])
          size_t  batch_stride= src->nb[3];   // stride over B (ne[3])
          struct ggml_tensor* result = nullptr;
          for (int64_t h = 0; h < H; ++h) {
            // Create a view of one head: shape [D, T, 1, B]
            struct ggml_tensor* head_slice = ggml_view_4d(
                ctx, src,
                D, T, 1, B,
                src->nb[1], src->nb[2], src->nb[3],
                h * head_stride  // offset in bytes to head h
            );
            for (int r = 0; r < reps; ++r) {
              if (result == nullptr) {
                result = head_slice;
              } else {
                result = ggml_concat(ctx, result, head_slice, 2 /* ne[2] axis */);
              }
            }
          }
          gt = result;
          break;
        }

        case ggml_ir::OpCode::INDEX: {
          // index(x, indices) along a single dim (currently dim=0 in lowering)
          // src_ids: [x, indices]
          // op_params: int32 dim
          if (!t->op_params() || t->op_params()->size() < 4) {
            ggml_free(ctx);
            return Error::InvalidArgument;
          }
          int32_t dim = 0;
          memcpy(&dim, t->op_params()->data(), 4);
          if (dim != 0) {
            // Only dim=0 supported for now
            ggml_free(ctx);
            return Error::InvalidArgument;
          }
          struct ggml_tensor* x = srcs[0];
          struct ggml_tensor* idx = srcs[1];
          if (idx->type == GGML_TYPE_I64) {
            idx = eager_cast_i64_to_i32(ctx, idx);
          } else if (idx->type != GGML_TYPE_I32) {
            idx = safe_ggml_cast(ctx, idx, GGML_TYPE_I32);
          }
          // ggml_get_rows supports src0 types F32/I32/F16/... but not I8.
          if (x->type == GGML_TYPE_I8) {
            x = safe_ggml_cast(ctx, x, GGML_TYPE_I32);
          }
          // ggml_get_rows selects rows by indices.
          gt = ggml_get_rows(ctx, x, idx);
          break;
        }

        case ggml_ir::OpCode::INDEX_MULTI: {
          // Multi-dimensional index gather: out = x[idx0, idx1, ...]
          // Runtime implementation uses ggml custom op so gather is computed
          // per-execution (not frozen at init time).
          int32_t ndims_hint = 0;
          if (t->op_params() && t->op_params()->size() >= 4) {
            memcpy(&ndims_hint, t->op_params()->data(), 4);
          }

          if (srcs.size() < 2) {
            ggml_free(ctx);
            return Error::InvalidArgument;
          }

          const int ndims = static_cast<int>(srcs.size()) - 1; // src + indices
          if (ndims < 1 || ndims > 4) {
            ggml_free(ctx);
            return Error::InvalidArgument;
          }
          if (ndims_hint > 0 && ndims_hint != ndims) {
            ggml_free(ctx);
            return Error::InvalidArgument;
          }

          struct ggml_tensor* src_x = srcs[0];
          ggml_type out_type = GGML_TYPE_F32;
          switch (t->type()) {
            case ggml_ir::TensorType::F16:
              out_type = GGML_TYPE_F16;
              break;
            case ggml_ir::TensorType::I64:
              out_type = GGML_TYPE_I64;
              break;
            case ggml_ir::TensorType::I32:
              out_type = GGML_TYPE_I32;
              break;
            case ggml_ir::TensorType::BOOL:
              out_type = GGML_TYPE_I32;
              break;
            case ggml_ir::TensorType::F32:
            default:
              out_type = GGML_TYPE_F32;
              break;
          }

          if (src_x->type != out_type) {
            src_x = safe_ggml_cast(ctx, src_x, out_type);
          }

          std::vector<struct ggml_tensor*> custom_args;
          custom_args.reserve(1 + ndims);
          custom_args.push_back(src_x);
          for (int i = 0; i < ndims; ++i) {
            struct ggml_tensor* idx = srcs[1 + i];
            if (idx->type == GGML_TYPE_I32) {
              idx = eager_cast_i32_to_i64(ctx, idx);
            } else if (idx->type != GGML_TYPE_I64) {
              idx = eager_cast_i32_to_i64(ctx, safe_ggml_cast(ctx, idx, GGML_TYPE_I32));
            }
            custom_args.push_back(idx);
          }

          // Output shape (in ggml ne order).
          int64_t out_ne0 = t->ne() ? (*t->ne())[0] : 1;
          int64_t out_ne1 = t->ne() && t->ne()->size() > 1 ? (*t->ne())[1] : 1;
          int64_t out_ne2 = t->ne() && t->ne()->size() > 2 ? (*t->ne())[2] : 1;
          int64_t out_ne3 = t->ne() && t->ne()->size() > 3 ? (*t->ne())[3] : 1;

          gt = ggml_custom_4d(
              ctx,
              out_type,
              out_ne0,
              out_ne1,
              out_ne2,
              out_ne3,
              custom_args.data(),
              static_cast<int>(custom_args.size()),
              ggml_custom_index_multi,
              GGML_N_TASKS_MAX,
              nullptr);
          break;
        }

        case ggml_ir::OpCode::INDEX_PUT: {
          // Implement a restricted index_put via ggml_set_rows.
          // This covers the Qwen3 KV-cache update pattern, where the update is along
          // the sequence dimension.
          //
          // op_params: int32 nindices, int32 present_mask
          // src_ids: [dst, <index_tensors...>, values]
          if (!t->op_params() || t->op_params()->size() < 8) {
            ggml_free(ctx);
            return Error::InvalidArgument;
          }
          int32_t nidx = 0, pm = 0;
          memcpy(&nidx, t->op_params()->data(), 4);
          memcpy(&pm, t->op_params()->data() + 4, 4);

          if (srcs.size() < 3) {
            ggml_free(ctx);
            return Error::InvalidArgument;
          }

          struct ggml_tensor* dst = srcs.front();
          struct ggml_tensor* val = srcs.back();

          // Find the first present index tensor (we only support one index tensor for now).
          int idx_pos = -1;
          for (int i = 0; i < nidx; ++i) {
            if (pm & (1 << i)) {
              idx_pos = i;
              break;
            }
          }
          if (idx_pos < 0) {
            // no indices -> no-op
            gt = dst;
            break;
          }

          // In our serialized srcs: [dst] + present_indices + [val]
          // So the index tensor is srcs[1] when only one present index.
          struct ggml_tensor* idx = srcs[1];

          // ggml_set_rows expects indices tensor type I64.
          if (idx->type == GGML_TYPE_I32) {
            idx = eager_cast_i32_to_i64(ctx, idx);
          } else if (idx->type != GGML_TYPE_I64) {
            idx = eager_cast_i32_to_i64(ctx, safe_ggml_cast(ctx, idx, GGML_TYPE_I32));
          }

          // Ensure val has shape compatible with dst for set_rows:
          // dst: [ne0, ne1(seq), ne2(heads), ne3(batch)]
          // val: should be [ne0, n_rows, ne2, ne3] (broadcastable in ne2/ne3)
          // If needed, reshape val using the output shape stored in IR (ne).
          if (val->ne[0] != dst->ne[0]) {
            // attempt to cast/reshape; if incompatible, fail
            // (most Qwen3 K/V values already match)
          }

          // ggml_set_rows requires val (b) to be F32.
          if (val->type != GGML_TYPE_F32) {
            val = safe_ggml_cast(ctx, val, GGML_TYPE_F32);
          }
          if (dst->type != GGML_TYPE_F32) {
            dst = safe_ggml_cast(ctx, dst, GGML_TYPE_F32);
          }

          gt = ggml_set_rows(ctx, dst, val, idx);
          break;
        }

        case ggml_ir::OpCode::CAST: {
          // Type cast with eager CPU conversion (ggml_cast doesn't support all combos).
          // op_params: int32 target_type (TensorType enum)
          struct ggml_tensor* src = srcs[0];

          // Determine target ggml type from op_params.
          int32_t target_type_enum = 0;
          if (t->op_params() && t->op_params()->size() >= 4) {
            memcpy(&target_type_enum, t->op_params()->data(), 4);
          }
          ggml_type target_type = GGML_TYPE_F32;
          switch (target_type_enum) {
            case 0: target_type = GGML_TYPE_F32; break;
            case 1: target_type = GGML_TYPE_F16; break;
            case 2: target_type = GGML_TYPE_I64; break;
            case 3: target_type = GGML_TYPE_I32; break;
            default: target_type = GGML_TYPE_F32; break;
          }

          // If types match, just alias.
          if (src->type == target_type) {
            gt = src;
            break;
          }

          // ggml CPU doesn't support GGML_OP_CPY from I64 to any other type,
          // so I64-source casts must be done eagerly on the CPU.
          if (src->type == GGML_TYPE_I64) {
            // Create output tensor for eager conversion.
            gt = ggml_new_tensor(ctx, target_type, GGML_MAX_DIMS, src->ne);

            // Check if src is an input tensor (data only available at execute time).
            bool is_input_src = false;
            for (const auto& [idx, inp_tensor] : input_pairs) {
              if (inp_tensor == src) {
                is_input_src = true;
                break;
              }
            }

            if (is_input_src && target_type == GGML_TYPE_I32) {
              // Defer I64→I32 conversion to execute().
              deferred_i64_to_i32.emplace_back(src, gt);
            } else if (src->data && gt->data) {
              // Constant source: convert now.
              const size_t nelem = ggml_nelements(src);
              const int64_t* src_data = static_cast<const int64_t*>(src->data);
              if (target_type == GGML_TYPE_I32) {
                int32_t* dst_data = static_cast<int32_t*>(gt->data);
                for (size_t i = 0; i < nelem; ++i) dst_data[i] = static_cast<int32_t>(src_data[i]);
              } else if (target_type == GGML_TYPE_F32) {
                float* dst_data = static_cast<float*>(gt->data);
                for (size_t i = 0; i < nelem; ++i) dst_data[i] = static_cast<float>(src_data[i]);
              }
            }
            gt->op = GGML_OP_NONE;  // Treated as constant in compute graph.
          } else {
            // All other casts go through safe_ggml_cast which handles
            // I32→F32, I32→other (via F32), and F16/BF16/F32 conversions.
            gt = safe_ggml_cast(ctx, src, target_type);
          }
          break;
        }

        case ggml_ir::OpCode::LLAMA_ATTENTION: {
          // Fused llama.cpp attention. For initial bring-up, we rely on ggml_flash_attn_ext.
          // src_ids: [q, k, v, (optional) mask]
          if (srcs.size() < 3) {
            ggml_free(ctx);
            return Error::InvalidArgument;
          }
          struct ggml_tensor* q = srcs[0];
          struct ggml_tensor* k = srcs[1];
          struct ggml_tensor* v = srcs[2];

          // flash_attn_ext dispatches via type traits (vec_dot / to_float),
          // so it supports F16, BF16, and F32 K/V natively.  Do NOT force
          // F16 — BF16 values that exceed F16 max (~65504) would overflow to
          // Inf, producing +Inf in the dot product accumulation.

          // Optional attention mask.  ggml_flash_attn_ext requires it to be:
          //   - F16 (the CPU kernel reads it as additive logit bias in FP16)
          //   - contiguous
          // The Python lowering stores the causal mask as F16 (True→0.0, False→-inf)
          // and aten.index.Tensor selects the right row via ggml_get_rows, giving
          // shape [kv_seq_len, 1, 1, 1] in ggml order.
          struct ggml_tensor* mask = nullptr;
          if (srcs.size() > 3 && srcs[3] != nullptr) {
            mask = srcs[3];
            // Ensure F16 (bool mask should already be converted at store time, but guard).
            if (mask->type != GGML_TYPE_F16) {
              mask = safe_ggml_cast(ctx, mask, GGML_TYPE_F16);
            }
            // Ensure contiguous (get_rows result may not be contiguous).
            if (!ggml_is_contiguous(mask)) {
              mask = ggml_cont(ctx, mask);
            }
          }

          // scale = 1/sqrt(head_dim). head_dim is ne0 in ggml layout when tensors are [D, T, H, B]
          float scale = 1.0f;
          if (q->ne[0] > 0) {
            scale = 1.0f / std::sqrt((float) q->ne[0]);
          }

          gt = ggml_flash_attn_ext(ctx, q, k, v, mask, scale, 0.0f, 0.0f);
          ggml_flash_attn_ext_set_prec(gt, GGML_PREC_F32);
          break;
        }

        case ggml_ir::OpCode::SUB: {
          struct ggml_tensor* a = srcs[0];
          struct ggml_tensor* b = srcs[1];
          // For int64 types, use custom element-wise sub
          if (a->type == GGML_TYPE_I64) {
            gt = ggml_new_tensor_4d(ctx, GGML_TYPE_I64, a->ne[0], a->ne[1], a->ne[2], a->ne[3]);
            const size_t nelem = ggml_nelements(gt);
            const int64_t* a_data = static_cast<const int64_t*>(a->data);
            const int64_t* b_data = static_cast<const int64_t*>(b->data);
            int64_t* out_data = static_cast<int64_t*>(gt->data);
            for (size_t i = 0; i < nelem; ++i) out_data[i] = a_data[i] - b_data[i];
          } else {
            gt = ggml_sub(ctx, a, b);
          }
          break;
        }

        case ggml_ir::OpCode::MUL_SCALAR: {
          // mul(x, scalar)
          // op_params: float32 scalar
          float scalar = 1.0f;
          if (t->op_params() && t->op_params()->size() >= 4) {
            memcpy(&scalar, t->op_params()->data(), sizeof(float));
          }
          gt = ggml_scale(ctx, srcs[0], scalar);
          break;
        }

        case ggml_ir::OpCode::POW: {
          // pow(x, exponent) where exponent is a scalar
          // For exponent=2, use ggml_sqr. Otherwise fall back to custom op or approximation.
          // op_params: float32 exponent
          float exponent = 2.0f;
          if (t->op_params() && t->op_params()->size() >= 4) {
            memcpy(&exponent, t->op_params()->data(), sizeof(float));
          }
          if (exponent == 2.0f) {
            gt = ggml_sqr(ctx, srcs[0]);
          } else if (exponent == 0.5f) {
            gt = ggml_sqrt(ctx, srcs[0]);
          } else {
            // General power: x^n = exp(n * log(x))
            // This won't work for negative x, but for RMSNorm (x^2) we use sqr above.
            struct ggml_tensor* log_x = ggml_log(ctx, srcs[0]);
            struct ggml_tensor* scaled = ggml_scale(ctx, log_x, exponent);
            gt = ggml_exp(ctx, scaled);
          }
          break;
        }

        case ggml_ir::OpCode::COS:
          gt = ggml_cos(ctx, srcs[0]);
          break;

        case ggml_ir::OpCode::SIN:
          gt = ggml_sin(ctx, srcs[0]);
          break;

        case ggml_ir::OpCode::BMM: {
          // Batch matrix multiply: bmm(a, b)
          // PyTorch: a is [B, M, K], b is [B, K, N], result is [B, M, N]
          // ggml ne format (reversed from PyTorch):
          //   a: [K, M, B, 1], b: [N, K, B, 1], out: [N, M, B, 1]
          // ggml_mul_mat(weight, input) requires weight->ne[0] == input->ne[0]
          // But b has ne[0]=N, a has ne[0]=K, so we need to transpose b first.
          struct ggml_tensor* a = srcs[0];  // [K, M, B, 1]
          struct ggml_tensor* b = srcs[1];  // [N, K, B, 1]
          
          // Transpose b to get [K, N, B, 1] so ne[0]=K matches a->ne[0]=K
          struct ggml_tensor* b_t = ggml_cont(ctx, ggml_transpose(ctx, b));
          gt = ggml_mul_mat(ctx, b_t, a);  // result: [N, M, B, 1]
          break;
        }

        case ggml_ir::OpCode::SIGMOID:
          gt = ggml_sigmoid(ctx, srcs[0]);
          break;

        case ggml_ir::OpCode::SOFTMAX: {
          // softmax(x, dim)
          // op_params: int32 dim, int32 ndim
          // ggml_soft_max operates on the innermost dimension (ne0).
          // If dim != -1 in PyTorch, we need to permute first.
          int32_t dim = -1, ndim = 4;
          if (t->op_params() && t->op_params()->size() >= 8) {
            memcpy(&dim, t->op_params()->data(), 4);
            memcpy(&ndim, t->op_params()->data() + 4, 4);
          }

          // Normalize negative dim
          if (dim < 0) dim = ndim + dim;

          // ggml axis = (ndim - 1) - pytorch_dim
          int ggml_axis = (ndim - 1) - dim;

          struct ggml_tensor* x = srcs[0];
          if (ggml_axis == 0) {
            // Softmax on innermost dim - direct
            gt = ggml_soft_max(ctx, x);
          } else {
            // Need to permute so that the softmax dim becomes axis 0
            // Create permutation that swaps axis 0 with ggml_axis
            int perm[4] = {0, 1, 2, 3};
            perm[0] = ggml_axis;
            perm[ggml_axis] = 0;
            x = ggml_permute(ctx, x, perm[0], perm[1], perm[2], perm[3]);
            x = ggml_cont(ctx, x);
            x = ggml_soft_max(ctx, x);
            // Permute back
            x = ggml_permute(ctx, x, perm[0], perm[1], perm[2], perm[3]);
            gt = ggml_cont(ctx, x);
          }
          break;
        }

        case ggml_ir::OpCode::WHERE: {
          // where(cond, x, y) - select x where cond is true, y otherwise
          // ggml doesn't have a direct where op, so we implement it as:
          // result = cond * x + (1 - cond) * y
          // But this requires cond to be float. For now, cast cond to float.
          struct ggml_tensor* cond = srcs[0];
          struct ggml_tensor* x = srcs[1];
          struct ggml_tensor* y = srcs[2];

          // Cast condition to float if needed
          if (cond->type != GGML_TYPE_F32) {
            cond = safe_ggml_cast(ctx, cond, GGML_TYPE_F32);
          }

          // Ensure all three tensors have the same shape for the
          // element-wise arithmetic that implements WHERE.
          // Find the "largest" shape (most elements) and repeat others to it.
          struct ggml_tensor* target = x;
          if (ggml_nelements(y) > ggml_nelements(target)) target = y;
          if (ggml_nelements(cond) > ggml_nelements(target)) target = cond;

          if (!ggml_are_same_shape(cond, target) && ggml_can_repeat(cond, target)) {
            cond = ggml_repeat(ctx, cond, target);
          }
          if (!ggml_are_same_shape(x, target) && ggml_can_repeat(x, target)) {
            x = ggml_repeat(ctx, x, target);
          }
          if (!ggml_are_same_shape(y, target) && ggml_can_repeat(y, target)) {
            y = ggml_repeat(ctx, y, target);
          }

          // result = cond * x + (1 - cond) * y
          struct ggml_tensor* one = ggml_new_f32(ctx, 1.0f);
          struct ggml_tensor* one_rep = ggml_repeat(ctx, one, cond);
          struct ggml_tensor* not_cond = ggml_sub(ctx, one_rep, cond);

          // ggml_mul(a, b) requires b repeatable to a — put bigger first.
          struct ggml_tensor* x_part = ggml_mul(ctx, x, cond);
          struct ggml_tensor* y_part = ggml_mul(ctx, y, not_cond);
          gt = ggml_add(ctx, x_part, y_part);
          break;
        }

        case ggml_ir::OpCode::ARANGE: {
          // arange(start, step) - generates [start, start+step, start+2*step, ...]
          // op_params: float64 start, float64 step
          double start = 0.0, step = 1.0;
          if (t->op_params() && t->op_params()->size() >= 16) {
            memcpy(&start, t->op_params()->data(), 8);
            memcpy(&step, t->op_params()->data() + 8, 8);
          }

          // Determine output type and shape from IR
          ggml_type out_type = GGML_TYPE_I64;
          switch (t->type()) {
            case ggml_ir::TensorType::F32: out_type = GGML_TYPE_F32; break;
            case ggml_ir::TensorType::F16: out_type = GGML_TYPE_F16; break;
            case ggml_ir::TensorType::I32: out_type = GGML_TYPE_I32; break;
            case ggml_ir::TensorType::I64: out_type = GGML_TYPE_I64; break;
            default: out_type = GGML_TYPE_I64; break;
          }

          gt = ggml_new_tensor_4d(ctx, out_type, ne[0], ne[1], ne[2], ne[3]);
          const size_t nelem = ggml_nelements(gt);

          // Fill with arange values
          if (out_type == GGML_TYPE_I64) {
            int64_t* data = static_cast<int64_t*>(gt->data);
            for (size_t i = 0; i < nelem; ++i) {
              data[i] = static_cast<int64_t>(start + i * step);
            }
          } else if (out_type == GGML_TYPE_I32) {
            int32_t* data = static_cast<int32_t*>(gt->data);
            for (size_t i = 0; i < nelem; ++i) {
              data[i] = static_cast<int32_t>(start + i * step);
            }
          } else {
            float* data = static_cast<float*>(gt->data);
            for (size_t i = 0; i < nelem; ++i) {
              data[i] = static_cast<float>(start + i * step);
            }
          }
          break;
        }

        case ggml_ir::OpCode::FULL: {
          // full(fill_value) - creates tensor filled with fill_value
          // op_params: float64 fill_value
          double fill_value = 0.0;
          if (t->op_params() && t->op_params()->size() >= 8) {
            memcpy(&fill_value, t->op_params()->data(), 8);
          }

          ggml_type out_type = GGML_TYPE_F32;
          switch (t->type()) {
            case ggml_ir::TensorType::F32: out_type = GGML_TYPE_F32; break;
            case ggml_ir::TensorType::F16: out_type = GGML_TYPE_F16; break;
            case ggml_ir::TensorType::I32: out_type = GGML_TYPE_I32; break;
            case ggml_ir::TensorType::I64: out_type = GGML_TYPE_I64; break;
            case ggml_ir::TensorType::BOOL: out_type = GGML_TYPE_I32; break;
            default: out_type = GGML_TYPE_F32; break;
          }

          gt = ggml_new_tensor_4d(ctx, out_type, ne[0], ne[1], ne[2], ne[3]);
          const size_t nelem = ggml_nelements(gt);

          if (out_type == GGML_TYPE_I64) {
            int64_t* data = static_cast<int64_t*>(gt->data);
            int64_t v = static_cast<int64_t>(fill_value);
            for (size_t i = 0; i < nelem; ++i) data[i] = v;
          } else if (out_type == GGML_TYPE_I32) {
            int32_t* data = static_cast<int32_t*>(gt->data);
            int32_t v = static_cast<int32_t>(fill_value);
            for (size_t i = 0; i < nelem; ++i) data[i] = v;
          } else {
            float* data = static_cast<float*>(gt->data);
            float v = static_cast<float>(fill_value);
            for (size_t i = 0; i < nelem; ++i) data[i] = v;
          }
          break;
        }

        case ggml_ir::OpCode::CUMSUM: {
          // cumsum(x, dim) - cumulative sum along dimension
          // op_params: int32 dim, int32 ndim
          int32_t dim = 0, ndim = 4;
          if (t->op_params() && t->op_params()->size() >= 8) {
            memcpy(&dim, t->op_params()->data(), 4);
            memcpy(&ndim, t->op_params()->data() + 4, 4);
          }

          struct ggml_tensor* src = srcs[0];
          ggml_type out_type = src->type;

          gt = ggml_new_tensor_4d(ctx, out_type, ne[0], ne[1], ne[2], ne[3]);

          // Convert PyTorch dim to ggml axis
          int ggml_axis = (ndim - 1) - dim;
          const size_t nelem = ggml_nelements(gt);

          // Simple cumsum implementation for I64 type
          if (out_type == GGML_TYPE_I64) {
            const int64_t* in_data = static_cast<const int64_t*>(src->data);
            int64_t* out_data = static_cast<int64_t*>(gt->data);

            // For now, handle axis 0 (innermost in ggml) specially
            int64_t stride = 1;
            for (int ax = 0; ax < ggml_axis; ++ax) stride *= src->ne[ax];
            int64_t dim_size = src->ne[ggml_axis];
            int64_t outer_size = nelem / (dim_size * stride);

            for (int64_t outer = 0; outer < outer_size; ++outer) {
              for (int64_t inner = 0; inner < stride; ++inner) {
                int64_t cumsum = 0;
                for (int64_t d = 0; d < dim_size; ++d) {
                  int64_t idx = outer * dim_size * stride + d * stride + inner;
                  cumsum += in_data[idx];
                  out_data[idx] = cumsum;
                }
              }
            }
          } else {
            // F32 fallback
            const float* in_data = static_cast<const float*>(src->data);
            float* out_data = static_cast<float*>(gt->data);

            int64_t stride = 1;
            for (int ax = 0; ax < ggml_axis; ++ax) stride *= src->ne[ax];
            int64_t dim_size = src->ne[ggml_axis];
            int64_t outer_size = nelem / (dim_size * stride);

            for (int64_t outer = 0; outer < outer_size; ++outer) {
              for (int64_t inner = 0; inner < stride; ++inner) {
                float cumsum = 0.0f;
                for (int64_t d = 0; d < dim_size; ++d) {
                  int64_t idx = outer * dim_size * stride + d * stride + inner;
                  cumsum += in_data[idx];
                  out_data[idx] = cumsum;
                }
              }
            }
          }
          break;
        }

        case ggml_ir::OpCode::EQ: {
          // eq(a, b) or eq(a, scalar)
          // op_params: float64 scalar, int32 is_scalar
          double scalar = 0.0;
          int32_t is_scalar = 0;
          if (t->op_params() && t->op_params()->size() >= 12) {
            memcpy(&scalar, t->op_params()->data(), 8);
            memcpy(&is_scalar, t->op_params()->data() + 8, 4);
          }

          struct ggml_tensor* a = srcs[0];
          gt = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, ne[0], ne[1], ne[2], ne[3]);
          const size_t nelem = ggml_nelements(gt);
          int32_t* out_data = static_cast<int32_t*>(gt->data);

          if (is_scalar) {
            if (a->type == GGML_TYPE_I64) {
              const int64_t* in_data = static_cast<const int64_t*>(a->data);
              int64_t s = static_cast<int64_t>(scalar);
              for (size_t i = 0; i < nelem; ++i) out_data[i] = (in_data[i] == s) ? 1 : 0;
            } else if (a->type == GGML_TYPE_I32) {
              const int32_t* in_data = static_cast<const int32_t*>(a->data);
              int32_t s = static_cast<int32_t>(scalar);
              for (size_t i = 0; i < nelem; ++i) out_data[i] = (in_data[i] == s) ? 1 : 0;
            } else {
              const float* in_data = static_cast<const float*>(a->data);
              float s = static_cast<float>(scalar);
              for (size_t i = 0; i < nelem; ++i) out_data[i] = (in_data[i] == s) ? 1 : 0;
            }
          } else {
            struct ggml_tensor* b = srcs[1];
            if (a->type == GGML_TYPE_I64) {
              const int64_t* a_data = static_cast<const int64_t*>(a->data);
              const int64_t* b_data = static_cast<const int64_t*>(b->data);
              for (size_t i = 0; i < nelem; ++i) out_data[i] = (a_data[i] == b_data[i]) ? 1 : 0;
            } else if (a->type == GGML_TYPE_I32) {
              const int32_t* a_data = static_cast<const int32_t*>(a->data);
              const int32_t* b_data = static_cast<const int32_t*>(b->data);
              for (size_t i = 0; i < nelem; ++i) out_data[i] = (a_data[i] == b_data[i]) ? 1 : 0;
            } else {
              const float* a_data = static_cast<const float*>(a->data);
              const float* b_data = static_cast<const float*>(b->data);
              for (size_t i = 0; i < nelem; ++i) out_data[i] = (a_data[i] == b_data[i]) ? 1 : 0;
            }
          }
          break;
        }

        case ggml_ir::OpCode::NE: {
          // ne(a, scalar) - not equal
          double scalar = 0.0;
          int32_t is_scalar = 1;
          if (t->op_params() && t->op_params()->size() >= 12) {
            memcpy(&scalar, t->op_params()->data(), 8);
            memcpy(&is_scalar, t->op_params()->data() + 8, 4);
          }

          struct ggml_tensor* a = srcs[0];
          gt = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, ne[0], ne[1], ne[2], ne[3]);
          const size_t nelem = ggml_nelements(gt);
          int32_t* out_data = static_cast<int32_t*>(gt->data);

          if (a->type == GGML_TYPE_I64) {
            const int64_t* in_data = static_cast<const int64_t*>(a->data);
            int64_t s = static_cast<int64_t>(scalar);
            for (size_t i = 0; i < nelem; ++i) out_data[i] = (in_data[i] != s) ? 1 : 0;
          } else if (a->type == GGML_TYPE_I32) {
            const int32_t* in_data = static_cast<const int32_t*>(a->data);
            int32_t s = static_cast<int32_t>(scalar);
            for (size_t i = 0; i < nelem; ++i) out_data[i] = (in_data[i] != s) ? 1 : 0;
          } else {
            const float* in_data = static_cast<const float*>(a->data);
            float s = static_cast<float>(scalar);
            for (size_t i = 0; i < nelem; ++i) out_data[i] = (in_data[i] != s) ? 1 : 0;
          }
          break;
        }

        case ggml_ir::OpCode::LE: {
          // le(a, b) - less than or equal
          struct ggml_tensor* a = srcs[0];
          struct ggml_tensor* b = srcs[1];
          gt = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, ne[0], ne[1], ne[2], ne[3]);
          const size_t nelem = ggml_nelements(gt);
          int32_t* out_data = static_cast<int32_t*>(gt->data);

          if (a->type == GGML_TYPE_I64) {
            const int64_t* a_data = static_cast<const int64_t*>(a->data);
            const int64_t* b_data = static_cast<const int64_t*>(b->data);
            for (size_t i = 0; i < nelem; ++i) out_data[i] = (a_data[i] <= b_data[i]) ? 1 : 0;
          } else if (a->type == GGML_TYPE_I32) {
            const int32_t* a_data = static_cast<const int32_t*>(a->data);
            const int32_t* b_data = static_cast<const int32_t*>(b->data);
            for (size_t i = 0; i < nelem; ++i) out_data[i] = (a_data[i] <= b_data[i]) ? 1 : 0;
          } else {
            const float* a_data = static_cast<const float*>(a->data);
            const float* b_data = static_cast<const float*>(b->data);
            for (size_t i = 0; i < nelem; ++i) out_data[i] = (a_data[i] <= b_data[i]) ? 1 : 0;
          }
          break;
        }

        case ggml_ir::OpCode::LT: {
          // lt(a, b) - less than
          struct ggml_tensor* a = srcs[0];
          struct ggml_tensor* b = srcs[1];
          gt = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, ne[0], ne[1], ne[2], ne[3]);
          const size_t nelem = ggml_nelements(gt);
          int32_t* out_data = static_cast<int32_t*>(gt->data);

          if (a->type == GGML_TYPE_I64) {
            const int64_t* a_data = static_cast<const int64_t*>(a->data);
            const int64_t* b_data = static_cast<const int64_t*>(b->data);
            for (size_t i = 0; i < nelem; ++i) out_data[i] = (a_data[i] < b_data[i]) ? 1 : 0;
          } else if (a->type == GGML_TYPE_I32) {
            const int32_t* a_data = static_cast<const int32_t*>(a->data);
            const int32_t* b_data = static_cast<const int32_t*>(b->data);
            for (size_t i = 0; i < nelem; ++i) out_data[i] = (a_data[i] < b_data[i]) ? 1 : 0;
          } else {
            const float* a_data = static_cast<const float*>(a->data);
            const float* b_data = static_cast<const float*>(b->data);
            for (size_t i = 0; i < nelem; ++i) out_data[i] = (a_data[i] < b_data[i]) ? 1 : 0;
          }
          break;
        }

        case ggml_ir::OpCode::GT: {
          // gt(a, b) - greater than
          struct ggml_tensor* a = srcs[0];
          struct ggml_tensor* b = srcs[1];
          gt = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, ne[0], ne[1], ne[2], ne[3]);
          const size_t nelem = ggml_nelements(gt);
          int32_t* out_data = static_cast<int32_t*>(gt->data);

          if (a->type == GGML_TYPE_I64) {
            const int64_t* a_data = static_cast<const int64_t*>(a->data);
            const int64_t* b_data = static_cast<const int64_t*>(b->data);
            for (size_t i = 0; i < nelem; ++i) out_data[i] = (a_data[i] > b_data[i]) ? 1 : 0;
          } else if (a->type == GGML_TYPE_I32) {
            const int32_t* a_data = static_cast<const int32_t*>(a->data);
            const int32_t* b_data = static_cast<const int32_t*>(b->data);
            for (size_t i = 0; i < nelem; ++i) out_data[i] = (a_data[i] > b_data[i]) ? 1 : 0;
          } else {
            const float* a_data = static_cast<const float*>(a->data);
            const float* b_data = static_cast<const float*>(b->data);
            for (size_t i = 0; i < nelem; ++i) out_data[i] = (a_data[i] > b_data[i]) ? 1 : 0;
          }
          break;
        }

        case ggml_ir::OpCode::GE: {
          // ge(a, b) - greater than or equal
          struct ggml_tensor* a = srcs[0];
          struct ggml_tensor* b = srcs[1];
          gt = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, ne[0], ne[1], ne[2], ne[3]);
          const size_t nelem = ggml_nelements(gt);
          int32_t* out_data = static_cast<int32_t*>(gt->data);

          if (a->type == GGML_TYPE_I64) {
            const int64_t* a_data = static_cast<const int64_t*>(a->data);
            const int64_t* b_data = static_cast<const int64_t*>(b->data);
            for (size_t i = 0; i < nelem; ++i) out_data[i] = (a_data[i] >= b_data[i]) ? 1 : 0;
          } else if (a->type == GGML_TYPE_I32) {
            const int32_t* a_data = static_cast<const int32_t*>(a->data);
            const int32_t* b_data = static_cast<const int32_t*>(b->data);
            for (size_t i = 0; i < nelem; ++i) out_data[i] = (a_data[i] >= b_data[i]) ? 1 : 0;
          } else {
            const float* a_data = static_cast<const float*>(a->data);
            const float* b_data = static_cast<const float*>(b->data);
            for (size_t i = 0; i < nelem; ++i) out_data[i] = (a_data[i] >= b_data[i]) ? 1 : 0;
          }
          break;
        }

        case ggml_ir::OpCode::BITWISE_AND: {
          // bitwise_and(a, b)
          struct ggml_tensor* a = srcs[0];
          struct ggml_tensor* b = srcs[1];
          gt = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, ne[0], ne[1], ne[2], ne[3]);
          const size_t nelem = ggml_nelements(gt);
          int32_t* out_data = static_cast<int32_t*>(gt->data);

          const int32_t* a_data = static_cast<const int32_t*>(a->data);
          const int32_t* b_data = static_cast<const int32_t*>(b->data);
          for (size_t i = 0; i < nelem; ++i) out_data[i] = a_data[i] & b_data[i];
          break;
        }

        case ggml_ir::OpCode::BITWISE_OR: {
          // bitwise_or(a, b)
          struct ggml_tensor* a = srcs[0];
          struct ggml_tensor* b = srcs[1];
          gt = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, ne[0], ne[1], ne[2], ne[3]);
          const size_t nelem = ggml_nelements(gt);
          int32_t* out_data = static_cast<int32_t*>(gt->data);

          const int32_t* a_data = static_cast<const int32_t*>(a->data);
          const int32_t* b_data = static_cast<const int32_t*>(b->data);
          for (size_t i = 0; i < nelem; ++i) out_data[i] = a_data[i] | b_data[i];
          break;
        }

        case ggml_ir::OpCode::LOGICAL_NOT: {
          // logical_not(x)
          struct ggml_tensor* x = srcs[0];
          gt = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, ne[0], ne[1], ne[2], ne[3]);
          const size_t nelem = ggml_nelements(gt);
          int32_t* out_data = static_cast<int32_t*>(gt->data);

          const int32_t* in_data = static_cast<const int32_t*>(x->data);
          for (size_t i = 0; i < nelem; ++i) out_data[i] = (in_data[i] == 0) ? 1 : 0;
          break;
        }

        case ggml_ir::OpCode::ANY: {
          // any(x, dim) - reduce any along dimension
          // op_params: int32 dim, int32 ndim
          int32_t dim = 0, ndim = 4;
          if (t->op_params() && t->op_params()->size() >= 8) {
            memcpy(&dim, t->op_params()->data(), 4);
            memcpy(&ndim, t->op_params()->data() + 4, 4);
          }

          struct ggml_tensor* src = srcs[0];
          gt = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, ne[0], ne[1], ne[2], ne[3]);

          // Convert PyTorch dim to ggml axis
          int ggml_axis = (ndim - 1) - dim;
          const size_t nelem_out = ggml_nelements(gt);

          const int32_t* in_data = static_cast<const int32_t*>(src->data);
          int32_t* out_data = static_cast<int32_t*>(gt->data);

          // Simple reduction: for now assume output shape is source shape with dim reduced
          int64_t stride = 1;
          for (int ax = 0; ax < ggml_axis; ++ax) stride *= src->ne[ax];
          int64_t dim_size = src->ne[ggml_axis];
          int64_t outer_size = nelem_out / stride;
          if (outer_size == 0) outer_size = 1;

          for (int64_t outer = 0; outer < outer_size; ++outer) {
            for (int64_t inner = 0; inner < stride; ++inner) {
              int32_t any_true = 0;
              for (int64_t d = 0; d < dim_size; ++d) {
                int64_t in_idx = outer * dim_size * stride + d * stride + inner;
                if (in_data[in_idx] != 0) {
                  any_true = 1;
                  break;
                }
              }
              int64_t out_idx = outer * stride + inner;
              out_data[out_idx] = any_true;
            }
          }
          break;
        }

        case ggml_ir::OpCode::UPDATE_CACHE: {
          // update_cache(cache, value, start_pos)
          // Inserts value into cache starting at start_pos along the seq dimension.
          // op_params: int32 seq_dim (PyTorch dimension for sequence)
          // src_ids: [cache, value, start_pos]
          //
          // cache shape: [batch, seq_len, n_heads, head_dim] or [batch, n_heads, seq_len, head_dim]
          // value shape: [batch, seq_len_new, n_heads, head_dim] or similar
          // start_pos: scalar tensor containing the starting position

          if (srcs.size() < 3) {
            fprintf(stderr, "[executorch-ggml] UPDATE_CACHE: expected 3 sources\n");
            ggml_free(ctx);
            return Error::InvalidArgument;
          }

          struct ggml_tensor* cache = srcs[0];
          struct ggml_tensor* value = srcs[1];
          struct ggml_tensor* start_pos_tensor = srcs[2];

          // Get seq_dim from op_params
          int32_t seq_dim_pytorch = 1;  // default
          if (t->op_params() && t->op_params()->size() >= 4) {
            memcpy(&seq_dim_pytorch, t->op_params()->data(), 4);
          }

          // Convert PyTorch seq_dim to ggml axis
          // PyTorch: [batch, seq, heads, dim] -> seq_dim=1
          // ggml:    [dim, heads, seq, batch] -> ggml_axis=1
          int ndim = 4;
          int ggml_axis = (ndim - 1) - seq_dim_pytorch;

          // Get start position value
          int64_t start_pos = 0;
          if (start_pos_tensor->type == GGML_TYPE_I64) {
            start_pos = *(const int64_t*)start_pos_tensor->data;
          } else if (start_pos_tensor->type == GGML_TYPE_I32) {
            start_pos = *(const int32_t*)start_pos_tensor->data;
          }

          // Use ggml_set_rows if we have indices, otherwise use a simple copy
          // For contiguous update, we can create a view and copy
          // cache[:, start_pos:start_pos+seq_len, :, :] = value

          // Create indices tensor for ggml_set_rows
          int64_t seq_len_new = value->ne[ggml_axis];
          struct ggml_tensor* indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, seq_len_new);
          int64_t* idx_data = static_cast<int64_t*>(indices->data);
          for (int64_t i = 0; i < seq_len_new; ++i) {
            idx_data[i] = start_pos + i;
          }

          // ggml_set_rows requires value (b) and cache (a) to be F32.
          if (value->type != GGML_TYPE_F32) {
            value = safe_ggml_cast(ctx, value, GGML_TYPE_F32);
          }
          if (cache->type != GGML_TYPE_F32) {
            cache = safe_ggml_cast(ctx, cache, GGML_TYPE_F32);
          }

          // Use ggml_set_rows to scatter values into cache
          gt = ggml_set_rows(ctx, cache, value, indices);
          break;
        }

        default:
          fprintf(stderr, "[executorch-ggml] Unsupported OpCode %d\n", (int) op);
          ggml_free(ctx);
          return Error::InvalidArgument;
      }

      id_to_tensor[tid] = gt;
    }

    if (t->is_output()) {
      output_tensors.push_back(id_to_tensor[tid]);
    }
  }
  //
  // KEY CHANGE for constants:
  //   Instead of reading from NamedDataMap (context.get_named_data_map()),
  //   find matching entry in handle->constant_data by IR tensor id:
  //
  //     for (auto& sc : handle->constant_data) {
  //       if (sc.ir_tensor_id == tid) {
  //         memcpy(gt->data, sc.data.data(), nbytes);
  //         break;
  //       }
  //     }
  //
  // KEY CHANGE (M2+ only, skip for M1):
  //   For input tensors with dynamic_dims, override ne[d] from
  //   input_ne_overrides when input_ne_overrides != nullptr.

  // === Build compute graph ===
  // Build compute graph using the same estimated size from above.
  struct ggml_cgraph* graph = ggml_new_graph_custom(ctx, est_graph_size, false);
  for (auto* out : output_tensors) {
    ggml_build_forward_expand(graph, out);
  }

  // Sort inputs by input_index
  std::sort(input_pairs.begin(), input_pairs.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });

  // Collect ordered input tensor list.
  std::vector<struct ggml_tensor*> ordered_inputs;
  for (auto& [idx, tensor] : input_pairs) {
    ordered_inputs.push_back(tensor);
  }

  // === Phase C: allocate graph on backend ===
  // -------------------------------------------------------------------
  // Allocate the graph on a compute backend (CUDA if available, else CPU)
  // -------------------------------------------------------------------

  // 1. Mark input / output tensors so the graph allocator keeps them live.
  std::unordered_set<struct ggml_tensor*> input_set(
      ordered_inputs.begin(), ordered_inputs.end());
  std::unordered_set<struct ggml_tensor*> deferred_dst_set;
  for (auto& [src, dst] : deferred_i64_to_i32) {
    deferred_dst_set.insert(dst);
  }
  for (auto* inp : ordered_inputs) {
    ggml_set_input(inp);
  }
  for (auto* dst : deferred_dst_set) {
    ggml_set_input(dst);  // receives data during execute
  }
  for (auto* out : output_tensors) {
    ggml_set_output(out);
  }

  // 2. Save constant data from every tensor in the context that is NOT an
  //    input or a deferred-cast destination (those get their data at execute
  //    time).  We iterate the full context (not just id_to_tensor) because
  //    helper tensors created by ggml ops (e.g. ggml_new_f32 for scalars)
  //    also need their data preserved.
  //    Also mark them as inputs so gallocr doesn't reuse their memory
  //    (constants like the scalar 1.0 in RSQRT are read by multiple layers).
  struct SavedTensor {
    struct ggml_tensor* tensor;
    std::vector<uint8_t> data;
  };
  std::vector<SavedTensor> saved_constants;
  for (struct ggml_tensor* gt = ggml_get_first_tensor(ctx);
       gt != nullptr;
       gt = ggml_get_next_tensor(ctx, gt)) {
    if (!gt->data) continue;
    if (input_set.count(gt)) continue;
    if (deferred_dst_set.count(gt)) continue;
    // Only save leaf tensors (constants / pre-computed values).
    // Graph op nodes will be recomputed by the backend.
    if (gt->op != GGML_OP_NONE) continue;
    ggml_set_input(gt);  // prevent gallocr from reusing this memory
    size_t nbytes = ggml_nbytes(gt);
    saved_constants.push_back({gt, std::vector<uint8_t>(
        static_cast<uint8_t*>(gt->data),
        static_cast<uint8_t*>(gt->data) + nbytes)});
  }

  // 2b. Clear all tensor data pointers so gallocr allocates everything
  //     on the backend. Without this, gallocr skips tensors that already
  //     have data (from no_alloc=false) and they never get a backend buffer,
  //     causing "tensor buffer not set" asserts on CUDA.
  for (struct ggml_tensor* t = ggml_get_first_tensor(ctx);
       t != nullptr;
       t = ggml_get_next_tensor(ctx, t)) {
    t->data = nullptr;
  }

  // 4. Allocate all graph tensors on the backend.
  ggml_gallocr_t galloc =
      ggml_gallocr_new(ggml_backend_get_default_buffer_type(handle->backend));
  if (!ggml_gallocr_alloc_graph(galloc, graph)) {
    fprintf(stderr, "[ggml_backend] ERROR: graph allocation failed\n");
    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    return Error::MemoryAllocationFailed;
  }

  // 5. Restore constant data into backend buffers.
  //    Skip tensors not reachable from the graph (gallocr leaves them unallocated).
  for (auto& sc : saved_constants) {
    if (sc.tensor->buffer == nullptr) continue;
    ggml_backend_tensor_set(sc.tensor, sc.data.data(), 0, sc.data.size());
  }

  // === Update handle ===
  handle->ctx = ctx;
  handle->graph = graph;
  handle->inputs = std::move(ordered_inputs);
  handle->outputs = std::move(output_tensors);
  handle->deferred_i64_to_i32 = std::move(deferred_i64_to_i32);
  handle->galloc = galloc;

  return Error::Ok;
}

// ---------------------------------------------------------------------------
// BackendInterface implementation
// ---------------------------------------------------------------------------

bool GgmlBackendInterface::is_available() const {
  return true;
}

Result<DelegateHandle*> GgmlBackendInterface::init(
    BackendInitContext& context,
    FreeableBuffer* processed,
    ArrayRef<CompileSpec> compile_specs) const {

  // --- 1. Copy IR buffer for later rebuilds ---
  auto* handle = new GgmlDelegateHandle();
  handle->ir_copy.assign(
      static_cast<const uint8_t*>(processed->data()),
      static_cast<const uint8_t*>(processed->data()) + processed->size());

  // --- 2. Parse FlatBuffer to read metadata ---
  const auto* fb_graph = ggml_ir::GetGgmlGraph(handle->ir_copy.data());
  if (!fb_graph || !fb_graph->tensors()) {
    delete handle;
    return Error::InvalidArgument;
  }
  const auto* fb_tensors = fb_graph->tensors();
  const int n_tensors = static_cast<int>(fb_tensors->size());
  handle->n_threads = fb_graph->n_threads();

  // --- 3. Create backend (one-time) ---
  const char* device_env = std::getenv("GGML_BACKEND_DEVICE");
  bool force_cpu = device_env && std::string(device_env) == "cpu";
  if (!force_cpu) {
    ggml_backend_dev_t gpu_dev =
        ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (gpu_dev) {
      handle->backend = ggml_backend_dev_init(gpu_dev, nullptr);
      if (handle->backend) {
        fprintf(stderr, "[ggml_backend] Using GPU backend: %s\n",
                ggml_backend_name(handle->backend));
      }
    }
  }
  if (!handle->backend) {
    handle->backend = ggml_backend_cpu_init();
    int cpu_threads = handle->n_threads;
    const char* threads_env = std::getenv("GGML_CPU_THREADS");
    if (threads_env) {
      cpu_threads = std::atoi(threads_env);
    } else if (cpu_threads <= 1) {
      cpu_threads = static_cast<int>(std::thread::hardware_concurrency());
      if (cpu_threads <= 0) cpu_threads = 4;
    }
    ggml_backend_cpu_set_n_threads(handle->backend, cpu_threads);
    fprintf(stderr, "[ggml_backend] Using CPU backend (%d threads)\n",
            cpu_threads);
  }

  // --- 4. Load ALL constants from NamedDataMap → handle->constant_data ---
  // This is the ONLY place NamedDataMap is touched.
  // After this, build_graph() reads from handle->constant_data.
  for (int i = 0; i < n_tensors; ++i) {
    const auto* t = fb_tensors->Get(i);
    if (static_cast<ggml_ir::OpCode>(t->op()) != ggml_ir::OpCode::NONE) continue;
    if (!t->data_key() || std::strlen(t->data_key()->c_str()) == 0) continue;

    const auto* ndm = context.get_named_data_map();
    if (ndm == nullptr) {
      ggml_backend_free(handle->backend);
      delete handle;
      return Error::InvalidArgument;
    }

    const char* key = t->data_key()->c_str();

    // Compute byte size from IR shape + type.
    size_t n_elems = 1;
    if (t->ne()) {
      for (size_t d = 0; d < t->ne()->size(); ++d) {
        n_elems *= static_cast<size_t>(t->ne()->Get(d));
      }
    }
    size_t elem_size = 4;
    switch (static_cast<ggml_ir::TensorType>(t->type())) {
      case ggml_ir::TensorType::F16:  elem_size = 2; break;
      case ggml_ir::TensorType::BF16: elem_size = 2; break;
      case ggml_ir::TensorType::I32:  elem_size = 4; break;
      case ggml_ir::TensorType::I64:  elem_size = 8; break;
      case ggml_ir::TensorType::BOOL: elem_size = 4; break;
      default:                        elem_size = 4; break;
    }
    size_t nbytes = n_elems * elem_size;

    auto fb = ndm->get_data(key);
    if (!fb.ok()) {
      ggml_backend_free(handle->backend);
      delete handle;
      return fb.error();
    }
    auto buf = std::move(fb.get());
    if (buf.size() < nbytes) {
      buf.Free();
      ggml_backend_free(handle->backend);
      delete handle;
      return Error::InvalidExternalData;
    }

    SavedConstant sc;
    sc.ir_tensor_id = t->id();
    sc.data.assign(
        static_cast<const uint8_t*>(buf.data()),
        static_cast<const uint8_t*>(buf.data()) + nbytes);
    buf.Free();
    handle->constant_data.push_back(std::move(sc));
  }

  // --- 5. Read dynamic_dims from IR for each input ---
  for (int i = 0; i < n_tensors; ++i) {
    const auto* t = fb_tensors->Get(i);
    if (!t->is_input()) continue;

    std::vector<bool> dd(4, false);
    fprintf(stderr, "[ggml_backend] input[%d] has_dynamic_dims=%d ne=(%lld,%lld,%lld,%lld)\n",
            t->input_index(), t->dynamic_dims() != nullptr,
            t->ne() ? (long long)t->ne()->Get(0) : -1,
            t->ne() && t->ne()->size() > 1 ? (long long)t->ne()->Get(1) : -1,
            t->ne() && t->ne()->size() > 2 ? (long long)t->ne()->Get(2) : -1,
            t->ne() && t->ne()->size() > 3 ? (long long)t->ne()->Get(3) : -1);
    if (t->dynamic_dims()) {
      for (size_t d = 0; d < t->dynamic_dims()->size() && d < 4; ++d) {
        dd[d] = t->dynamic_dims()->Get(d);
      }
      fprintf(stderr, "[ggml_backend]   dynamic_dims=(%d,%d,%d,%d)\n",
              (int)dd[0], (int)dd[1], (int)dd[2], (int)dd[3]);
    }
    if (std::any_of(dd.begin(), dd.end(), [](bool v) { return v; })) {
      handle->has_dynamic = true;
    }
    // Store in input_index order (may be sparse; resize as needed).
    int idx = t->input_index();
    if (idx >= 0 && static_cast<size_t>(idx) >= handle->input_dynamic_dims.size()) {
      handle->input_dynamic_dims.resize(idx + 1);
    }
    if (idx >= 0) {
      handle->input_dynamic_dims[idx] = std::move(dd);
    }
  }

  // --- 6. Build initial graph with serialized shapes ---
  Error err = build_graph(handle, nullptr, 0);
  if (err != Error::Ok) {
    if (handle->backend) ggml_backend_free(handle->backend);
    delete handle;
    return err;
  }

  // --- 7. Initialize last_input_ne from the initial graph's input shapes ---
  if (handle->has_dynamic) {
    size_t n_inp = handle->inputs.size();
    handle->last_input_ne.resize(n_inp * 4, 1);
    for (size_t i = 0; i < n_inp; ++i) {
      for (int d = 0; d < 4; ++d) {
        handle->last_input_ne[i * 4 + d] = handle->inputs[i]->ne[d];
      }
    }
  }

  return handle;
}

Error GgmlBackendInterface::execute(
    BackendExecutionContext& context,
    DelegateHandle* handle_raw,
    Span<EValue*> args) const {

  auto* handle = reinterpret_cast<GgmlDelegateHandle*>(handle_raw);

  size_t n_inputs = handle->inputs.size();
  size_t n_outputs = handle->outputs.size();

  // --- Build mapping from ggml input index → args index ---
  // With dynamic shapes, ExecuTorch passes non-tensor args (sym_size ints)
  // alongside tensor inputs.  The args layout is:
  //   [input_0, input_1, ..., (possibly sym_size ints), output_0, output_1, ...]
  // Outputs are always the last n_outputs entries.
  //
  // Some IR "inputs" may correspond to sym_size scalars that were serialized
  // as placeholder tensors but don't have matching tensor args at runtime.
  // We build a 1:1 mapping: ggml_input_i → args[i], but only for tensor args.
  // Non-tensor args (sym_size) and IR inputs without matching tensor args
  // are skipped during data copy.
  size_t n_non_output_args = args.size() >= n_outputs ? args.size() - n_outputs : 0;


  // --- Shape-change detection: rebuild graph if dynamic dims changed ---
  if (handle->has_dynamic) {
    // Build current input shapes from ET tensors (ggml ne order).
    // Map: for each non-output arg that is a tensor, assign it to the next
    // ggml input index.  Non-tensor args (sym_size ints) are skipped.
    std::vector<int64_t> current_ne(n_inputs * 4, 1);
    size_t ggml_idx = 0;
    for (size_t a = 0; a < n_non_output_args && ggml_idx < n_inputs; ++a) {
      if (!args[a]->isTensor()) continue;
      const auto& et = args[a]->toTensor();
      int ndim = et.dim();
      for (int d = 0; d < ndim && d < 4; ++d) {
        current_ne[ggml_idx * 4 + d] = et.size(ndim - 1 - d);
      }
      ++ggml_idx;
    }

    // Compare only dynamic dims with last-seen shapes.
    bool shapes_changed = false;
    for (size_t i = 0; i < n_inputs && !shapes_changed; ++i) {
      if (i >= handle->input_dynamic_dims.size()) continue;
      const auto& dd = handle->input_dynamic_dims[i];
      for (size_t d = 0; d < dd.size() && d < 4; ++d) {
        if (dd[d] && current_ne[i * 4 + d] != handle->last_input_ne[i * 4 + d]) {
          shapes_changed = true;
          break;
        }
      }
    }

    if (shapes_changed) {
      fprintf(stderr, "[ggml_backend] Dynamic shapes changed, rebuilding graph...\n");
      for (size_t i = 0; i < n_inputs; ++i) {
        fprintf(stderr, "  input[%zu]: last=(%lld,%lld,%lld,%lld) cur=(%lld,%lld,%lld,%lld)\n",
                i,
                (long long)handle->last_input_ne[i*4], (long long)handle->last_input_ne[i*4+1],
                (long long)handle->last_input_ne[i*4+2], (long long)handle->last_input_ne[i*4+3],
                (long long)current_ne[i*4], (long long)current_ne[i*4+1],
                (long long)current_ne[i*4+2], (long long)current_ne[i*4+3]);
      }
      Error err = build_graph(handle, current_ne.data(), current_ne.size());
      if (err != Error::Ok) return err;
      handle->last_input_ne = current_ne;
      n_inputs = handle->inputs.size();
      fprintf(stderr, "[ggml_backend] Rebuild complete. %zu inputs, %zu outputs\n",
              handle->inputs.size(), handle->outputs.size());
    }
  }

  // Copy input data from ExecuTorch tensors → backend tensors.
  // Walk non-output args in order; each tensor arg maps to the next ggml input.
  // Non-tensor args (sym_size ints from dynamic shapes) are skipped.
  // IR inputs that don't have a matching tensor arg (e.g. sym_size placeholders
  // that were serialized as dummy [1,1,1,1] tensors) are also skipped.
  {
  size_t ggml_idx = 0;
  for (size_t a = 0; a < n_non_output_args && ggml_idx < n_inputs; ++a) {
    if (!args[a]->isTensor()) continue;

    struct ggml_tensor* gt = handle->inputs[ggml_idx];
    ++ggml_idx;

    // Skip inputs that aren't part of the ggml graph (gallocr didn't
    // allocate them, e.g. unused position/start_pos placeholders).
    if (gt->buffer == nullptr) continue;

    const auto& et_tensor = args[a]->toTensor();

    const size_t nelem = ggml_nelements(gt);

    if (gt->type == GGML_TYPE_F16 && et_tensor.scalar_type() == executorch::aten::ScalarType::Float) {
      // Convert fp32 → fp16 on host, then copy to backend.
      std::vector<ggml_fp16_t> tmp(nelem);
      const float* src = static_cast<const float*>(et_tensor.const_data_ptr());
      ggml_fp32_to_fp16_row(src, tmp.data(), (int64_t) nelem);
      ggml_backend_tensor_set(gt, tmp.data(), 0, nelem * sizeof(ggml_fp16_t));
    } else if (gt->type == GGML_TYPE_BF16 && et_tensor.scalar_type() == executorch::aten::ScalarType::Float) {
      // Convert fp32 → bf16 on host, then copy to backend.
      std::vector<ggml_bf16_t> tmp(nelem);
      const float* src = static_cast<const float*>(et_tensor.const_data_ptr());
      for (size_t j = 0; j < nelem; ++j) {
        tmp[j] = ggml_fp32_to_bf16(src[j]);
      }
      ggml_backend_tensor_set(gt, tmp.data(), 0, nelem * sizeof(ggml_bf16_t));
    } else if (gt->type == GGML_TYPE_BF16 && et_tensor.scalar_type() == executorch::aten::ScalarType::BFloat16) {
      // BF16 → BF16 direct copy.
      ggml_backend_tensor_set(gt, et_tensor.const_data_ptr(), 0, ggml_nbytes(gt));
    } else if (gt->type == GGML_TYPE_I32 && et_tensor.scalar_type() == executorch::aten::ScalarType::Bool) {
      std::vector<int32_t> tmp(nelem);
      const bool* src = static_cast<const bool*>(et_tensor.const_data_ptr());
      for (size_t j = 0; j < nelem; ++j) {
        tmp[j] = src[j] ? 1 : 0;
      }
      ggml_backend_tensor_set(gt, tmp.data(), 0, nelem * sizeof(int32_t));
    } else if (gt->type == GGML_TYPE_I32 && et_tensor.scalar_type() == executorch::aten::ScalarType::Long) {
      std::vector<int32_t> tmp(nelem);
      const int64_t* src = static_cast<const int64_t*>(et_tensor.const_data_ptr());
      for (size_t j = 0; j < nelem; ++j) {
        tmp[j] = (int32_t) src[j];
      }
      ggml_backend_tensor_set(gt, tmp.data(), 0, nelem * sizeof(int32_t));
    } else {
      ggml_backend_tensor_set(gt, et_tensor.const_data_ptr(), 0, ggml_nbytes(gt));
    }
  }
  }  // end input copy block

  // Run deferred I64→I32 casts (now that input data is on the backend).
  for (const auto& [src_i64, dst_i32] : handle->deferred_i64_to_i32) {
    if (src_i64->buffer == nullptr || dst_i32->buffer == nullptr) continue;
    const size_t nelem = ggml_nelements(src_i64);
    std::vector<int64_t> src_buf(nelem);
    ggml_backend_tensor_get(src_i64, src_buf.data(), 0, nelem * sizeof(int64_t));
    std::vector<int32_t> dst_buf(nelem);
    for (size_t j = 0; j < nelem; ++j) {
      dst_buf[j] = static_cast<int32_t>(src_buf[j]);
    }
    ggml_backend_tensor_set(dst_i32, dst_buf.data(), 0, nelem * sizeof(int32_t));
  }

  // Execute the graph on the backend
  ggml_backend_graph_compute(handle->backend, handle->graph);

  // Copy output data from backend tensors → ExecuTorch output tensors
  // Outputs are always the last n_outputs entries in the args span.
  for (size_t i = 0; i < n_outputs; ++i) {
    size_t out_idx = args.size() - n_outputs + i;
    if (out_idx >= (size_t)args.size() || !args[out_idx]->isTensor()) {
      fprintf(stderr, "[ggml_backend] ERROR: output args[%zu] is not a Tensor "
              "(n_outputs=%zu, args.size=%zu)\n",
              out_idx, n_outputs, (size_t)args.size());
      return Error::InvalidArgument;
    }
    auto& et_tensor = args[out_idx]->toTensor();
    struct ggml_tensor* gt = handle->outputs[i];

    // Resize ET output tensor to match actual ggml output shape (see below).

    // Resize ET output tensor to match actual ggml output shape.
    // ExecuTorch may have allocated it at the upper-bound shape for dynamic
    // dims.  Convert ggml ne order (innermost first) to PyTorch order
    // (outermost first) and resize.
    {
      int et_ndim = et_tensor.dim();
      // Build ggml shape in PyTorch order: reverse the ne array, trim to ndim.
      // ggml ne = [ne0, ne1, ne2, ne3], PyTorch = [ne_{ndim-1}, ..., ne0]
      std::vector<executorch::aten::SizesType> new_sizes(et_ndim);
      for (int d = 0; d < et_ndim; ++d) {
        new_sizes[d] = static_cast<executorch::aten::SizesType>(
            gt->ne[et_ndim - 1 - d]);
      }
      Error resize_err = executorch::ET_RUNTIME_NAMESPACE::resize_tensor(
          et_tensor, {new_sizes.data(), new_sizes.size()});
      if (resize_err != Error::Ok) {
        fprintf(stderr, "[ggml_backend] WARNING: resize_tensor failed for "
                "output %zu\n", i);
      }
    }

    const size_t nelem = ggml_nelements(gt);

    if (gt->type == GGML_TYPE_F16 && et_tensor.scalar_type() == executorch::aten::ScalarType::Float) {
      // Read fp16 from backend, convert to fp32.
      std::vector<ggml_fp16_t> tmp(nelem);
      ggml_backend_tensor_get(gt, tmp.data(), 0, nelem * sizeof(ggml_fp16_t));
      float* dst = static_cast<float*>(et_tensor.mutable_data_ptr());
      ggml_fp16_to_fp32_row(tmp.data(), dst, (int64_t) nelem);
    } else if (gt->type == GGML_TYPE_F32 && et_tensor.scalar_type() == executorch::aten::ScalarType::BFloat16) {
      // ggml output is F32 (e.g. ggml_set_rows requires F32) but ET expects BF16.
      std::vector<float> tmp(nelem);
      ggml_backend_tensor_get(gt, tmp.data(), 0, nelem * sizeof(float));
      ggml_bf16_t* dst = static_cast<ggml_bf16_t*>(et_tensor.mutable_data_ptr());
      for (size_t j = 0; j < nelem; ++j) {
        dst[j] = ggml_fp32_to_bf16(tmp[j]);
      }
    } else if (gt->type == GGML_TYPE_F32 && et_tensor.scalar_type() == executorch::aten::ScalarType::Float) {
      ggml_backend_tensor_get(gt, et_tensor.mutable_data_ptr(), 0, ggml_nbytes(gt));
    } else if (gt->type == GGML_TYPE_BF16 && et_tensor.scalar_type() == executorch::aten::ScalarType::BFloat16) {
      ggml_backend_tensor_get(gt, et_tensor.mutable_data_ptr(), 0, ggml_nbytes(gt));
    } else {
      // Fallback: copy min of both sizes to avoid buffer overflow.
      size_t copy_size = std::min(ggml_nbytes(gt), (size_t)et_tensor.nbytes());
      ggml_backend_tensor_get(gt, et_tensor.mutable_data_ptr(), 0, copy_size);
    }
  }

  return Error::Ok;
}

void GgmlBackendInterface::destroy(DelegateHandle* handle_raw) const {
  auto* handle = reinterpret_cast<GgmlDelegateHandle*>(handle_raw);
  if (handle) {
    if (handle->galloc) {
      ggml_gallocr_free(handle->galloc);
    }
    if (handle->backend) {
      ggml_backend_free(handle->backend);
    }
    if (handle->ctx) {
      ggml_free(handle->ctx);
    }
    delete handle;
  }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

namespace {
auto cls = executorch::runtime::register_backend(
    {"GgmlBackend", new GgmlBackendInterface()});
}  // namespace

}  // namespace executorch_ggml
