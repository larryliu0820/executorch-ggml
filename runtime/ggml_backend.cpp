/**
 * GgmlBackendInterface — ExecuTorch runtime backend delegating to ggml.
 *
 * Deserialises a FlatBuffer-encoded ggml IR graph produced by the Python
 * GgmlBackend.preprocess() and executes it using the ggml compute graph API.
 */

#include "ggml_backend.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <vector>

#include "ggml_ir_generated.h"  // flatc-generated header (checked in under schema/)

#include <ggml.h>
#include <ggml-cpu.h>

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>

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

} // namespace

// ---------------------------------------------------------------------------
// Delegate handle — owns the ggml context, graph, and tensor bookkeeping
// ---------------------------------------------------------------------------

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

  // Separate compute context for graph_compute (needs its own memory)
  struct ggml_context* compute_ctx = nullptr;
};

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

  // Parse FlatBuffer
  const auto* fb_graph = ggml_ir::GetGgmlGraph(processed->data());
  if (!fb_graph || !fb_graph->tensors()) {
    return Error::InvalidArgument;
  }

  const auto* fb_tensors = fb_graph->tensors();
  const int n_tensors = static_cast<int>(fb_tensors->size());
  const int n_threads = fb_graph->n_threads();

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
      case ggml_ir::TensorType::I32:  elem_size = 4; break;
      case ggml_ir::TensorType::I64:  elem_size = 8; break;
      case ggml_ir::TensorType::BOOL: elem_size = 4; break; // stored as I32
      default:                        elem_size = 4; break;
    }
    constant_data_size += n_elems * elem_size;
  }

  size_t ctx_size =
      static_cast<size_t>(n_tensors) * ggml_tensor_overhead() +
      constant_data_size +
      ggml_graph_overhead() +
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

  // Map from IR tensor id → ggml_tensor*
  std::vector<struct ggml_tensor*> id_to_tensor(n_tensors, nullptr);

  // Track inputs and outputs
  std::vector<std::pair<int, struct ggml_tensor*>> input_pairs;  // (index, tensor)
  std::vector<struct ggml_tensor*> output_tensors;
  // Deferred I64→I32 casts (for input tensors).
  std::vector<std::pair<struct ggml_tensor*, struct ggml_tensor*>> deferred_i64_to_i32;

  // Walk tensors in topological order
  for (int i = 0; i < n_tensors; ++i) {
    const auto* t = fb_tensors->Get(i);
    const int tid = t->id();

    // Read shape (ne)
    int64_t ne[4] = {1, 1, 1, 1};
    if (t->ne() && t->ne()->size() > 0) {
      for (size_t d = 0; d < t->ne()->size() && d < 4; ++d) {
        ne[d] = t->ne()->Get(d);
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

    const auto op = static_cast<ggml_ir::OpCode>(t->op());

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

      // Load constant data from NamedDataMap if present.
      if (t->data_key() && t->data_key()->c_str() && std::strlen(t->data_key()->c_str()) > 0) {
        const auto* ndm = context.get_named_data_map();
        if (ndm == nullptr) {
          ggml_free(ctx);
          return Error::InvalidArgument;
        }

        const char* key = t->data_key()->c_str();
        const size_t nbytes = ggml_nbytes(gt);

        // Prefer get_data() for maximum compatibility: some NamedDataMap
        // implementations may not implement load_data_into().
        auto fb = ndm->get_data(key);
        if (!fb.ok()) {
          ggml_free(ctx);
          return fb.error();
        }
        auto buf = std::move(fb.get());
        if (buf.size() < nbytes) {
          buf.Free();
          ggml_free(ctx);
          return Error::InvalidExternalData;
        }
        memcpy(gt->data, buf.data(), nbytes);
        buf.Free();
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
            if (a->type != tgt) a = ggml_cast(ctx, a, tgt);
            if (b->type != tgt) b = ggml_cast(ctx, b, tgt);
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

          gt = ggml_add(ctx, a, b);
          break;
        }

        case ggml_ir::OpCode::MUL_MAT:
          gt = ggml_mul_mat(ctx, srcs[0], srcs[1]);
          break;

        case ggml_ir::OpCode::MUL:
          gt = ggml_mul(ctx, srcs[0], srcs[1]);
          break;

        case ggml_ir::OpCode::REPEAT: {
          struct ggml_tensor* a = srcs[0];
          struct ggml_tensor* b = srcs[1];
          if (!ggml_can_repeat(a, b)) {
            fprintf(stderr,
                    "[executorch-ggml] REPEAT shape not repeatable: a=(%lld,%lld,%lld,%lld) b=(%lld,%lld,%lld,%lld)\n",
                    (long long) a->ne[0], (long long) a->ne[1], (long long) a->ne[2], (long long) a->ne[3],
                    (long long) b->ne[0], (long long) b->ne[1], (long long) b->ne[2], (long long) b->ne[3]);
            ggml_free(ctx);
            return Error::InvalidArgument;
          }
          gt = ggml_repeat(ctx, a, b);
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
          if (idx->type != GGML_TYPE_I32) {
            idx = ggml_cast(ctx, idx, GGML_TYPE_I32);
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
            gt = ggml_cast(ctx, gt, GGML_TYPE_F32);
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
              bias_rep = ggml_cast(ctx, bias_rep, GGML_TYPE_F32);
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

          gt = ggml_reshape_4d(ctx, srcs[0], new_ne[0], new_ne[1], new_ne[2], new_ne[3]);
          // Always make contiguous to avoid issues with nested views (e.g., view->permute).
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
          // Represent as reshape to the output shape stored in IR.
          // (op_params has dim, but shape is authoritative here)
          struct ggml_tensor* a = srcs[0];
          gt = ggml_reshape_4d(ctx, a, ne[0], ne[1], ne[2], ne[3]);
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
          if (idx->type != GGML_TYPE_I32) {
            idx = ggml_cast(ctx, idx, GGML_TYPE_I32);
          }
          // ggml_get_rows supports src0 types F32/I32/F16/... but not I8.
          if (x->type == GGML_TYPE_I8) {
            x = ggml_cast(ctx, x, GGML_TYPE_I32);
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
            src_x = ggml_cast(ctx, src_x, out_type);
          }

          std::vector<struct ggml_tensor*> custom_args;
          custom_args.reserve(1 + ndims);
          custom_args.push_back(src_x);
          for (int i = 0; i < ndims; ++i) {
            struct ggml_tensor* idx = srcs[1 + i];
            if (idx->type != GGML_TYPE_I32 && idx->type != GGML_TYPE_I64) {
              idx = ggml_cast(ctx, idx, GGML_TYPE_I64);
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
          if (idx->type != GGML_TYPE_I64) {
            idx = ggml_cast(ctx, idx, GGML_TYPE_I64);
          }

          // Ensure val has shape compatible with dst for set_rows:
          // dst: [ne0, ne1(seq), ne2(heads), ne3(batch)]
          // val: should be [ne0, n_rows, ne2, ne3] (broadcastable in ne2/ne3)
          // If needed, reshape val using the output shape stored in IR (ne).
          if (val->ne[0] != dst->ne[0]) {
            // attempt to cast/reshape; if incompatible, fail
            // (most Qwen3 K/V values already match)
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

          // Create output tensor with target type.
          gt = ggml_new_tensor(ctx, target_type, GGML_MAX_DIMS, src->ne);

          // I64 → I32 conversion: check if source is an input (defer to execute).
          if (src->type == GGML_TYPE_I64 && target_type == GGML_TYPE_I32) {
            // Check if src is an input tensor (will have data at execute time).
            bool is_input_src = false;
            for (const auto& [idx, inp_tensor] : input_pairs) {
              if (inp_tensor == src) {
                is_input_src = true;
                break;
              }
            }

            if (is_input_src) {
              // Defer conversion to execute() - record the pair.
              deferred_i64_to_i32.emplace_back(src, gt);
            } else {
              // Constant source: convert now.
              const size_t nelem = ggml_nelements(src);
              const int64_t* src_data = static_cast<const int64_t*>(src->data);
              int32_t* dst_data = static_cast<int32_t*>(gt->data);
              for (size_t i = 0; i < nelem; ++i) {
                dst_data[i] = static_cast<int32_t>(src_data[i]);
              }
            }
            gt->op = GGML_OP_NONE;  // Treated as constant in compute graph.
          } else {
            // For other conversions, try ggml_cast (may fail at compute time
            // if unsupported, but many combos like F32→F16 work).
            gt = ggml_cast(ctx, src, target_type);
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

          // Note: flash_attn expects k/v in f16 for the fast path.
          if (k->type == GGML_TYPE_F32) k = ggml_cast(ctx, k, GGML_TYPE_F16);
          if (v->type == GGML_TYPE_F32) v = ggml_cast(ctx, v, GGML_TYPE_F16);

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
              mask = ggml_cast(ctx, mask, GGML_TYPE_F16);
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

        case ggml_ir::OpCode::SUB:
          gt = ggml_sub(ctx, srcs[0], srcs[1]);
          break;

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
          // Batch matrix multiply: bmm(a, b) where a is [B, M, K] and b is [B, K, N]
          // Result is [B, M, N].
          // In ggml order: a is [K, M, B], b is [N, K, B], result is [N, M, B].
          // Use ggml_mul_mat which supports batch dimensions.
          struct ggml_tensor* a = srcs[0];
          struct ggml_tensor* b = srcs[1];
          gt = ggml_mul_mat(ctx, b, a);
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
            cond = ggml_cast(ctx, cond, GGML_TYPE_F32);
          }

          // Ensure cond is broadcastable to x/y shape
          if (!ggml_can_repeat(cond, x)) {
            cond = ggml_repeat(ctx, cond, x);
          }

          // result = cond * x + (1 - cond) * y
          struct ggml_tensor* one = ggml_new_f32(ctx, 1.0f);
          struct ggml_tensor* one_rep = ggml_repeat(ctx, one, cond);
          struct ggml_tensor* not_cond = ggml_sub(ctx, one_rep, cond);

          struct ggml_tensor* x_part = ggml_mul(ctx, cond, x);
          struct ggml_tensor* y_part = ggml_mul(ctx, not_cond, y);
          gt = ggml_add(ctx, x_part, y_part);
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

  // Build compute graph
  struct ggml_cgraph* graph = ggml_new_graph(ctx);
  for (auto* out : output_tensors) {
    ggml_build_forward_expand(graph, out);
  }

  // Sort inputs by input_index
  std::sort(input_pairs.begin(), input_pairs.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });

  // Create handle
  auto* handle = new GgmlDelegateHandle();
  handle->ctx = ctx;
  handle->graph = graph;
  for (auto& [idx, tensor] : input_pairs) {
    handle->inputs.push_back(tensor);
  }
  handle->outputs = std::move(output_tensors);
  handle->deferred_i64_to_i32 = std::move(deferred_i64_to_i32);

  // Allocate a separate context for compute
  size_t compute_size = ggml_graph_overhead() + 1024 * 1024;
  struct ggml_init_params compute_params = {
      /* .mem_size   = */ compute_size,
      /* .mem_buffer = */ nullptr,
      /* .no_alloc   = */ true,
  };
  handle->compute_ctx = ggml_init(compute_params);

  return handle;
}

Error GgmlBackendInterface::execute(
    BackendExecutionContext& context,
    DelegateHandle* handle_raw,
    Span<EValue*> args) const {

  auto* handle = reinterpret_cast<GgmlDelegateHandle*>(handle_raw);

  // Copy input data from ExecuTorch tensors → ggml tensors
  size_t n_inputs = handle->inputs.size();
  for (size_t i = 0; i < n_inputs; ++i) {
    const auto& et_tensor = args[i]->toTensor();
    struct ggml_tensor* gt = handle->inputs[i];

    const size_t nelem = ggml_nelements(gt);

    if (gt->type == GGML_TYPE_F16) {
      // If a model ends up with fp16 runtime inputs, convert from fp32.
      const float* src = static_cast<const float*>(et_tensor.const_data_ptr());
      ggml_fp16_t* dst = static_cast<ggml_fp16_t*>(gt->data);
      ggml_fp32_to_fp16_row(src, dst, (int64_t) nelem);
    } else if (gt->type == GGML_TYPE_I32 && et_tensor.scalar_type() == executorch::aten::ScalarType::Bool) {
      // Avoid over-reading bool buffers when our ggml tensor stores masks as I32.
      const bool* src = static_cast<const bool*>(et_tensor.const_data_ptr());
      int32_t* dst = static_cast<int32_t*>(gt->data);
      for (size_t j = 0; j < nelem; ++j) {
        dst[j] = src[j] ? 1 : 0;
      }
    } else if (gt->type == GGML_TYPE_I32 && et_tensor.scalar_type() == executorch::aten::ScalarType::Long) {
      // Common for start_pos etc.
      const int64_t* src = static_cast<const int64_t*>(et_tensor.const_data_ptr());
      int32_t* dst = static_cast<int32_t*>(gt->data);
      for (size_t j = 0; j < nelem; ++j) {
        dst[j] = (int32_t) src[j];
      }
    } else {
      // Default memcpy when storage types match.
      size_t nbytes = ggml_nbytes(gt);
      memcpy(gt->data, et_tensor.const_data_ptr(), nbytes);
    }
  }

  // Run deferred I64→I32 casts (now that input data is available).
  for (const auto& [src_i64, dst_i32] : handle->deferred_i64_to_i32) {
    const size_t nelem = ggml_nelements(src_i64);
    const int64_t* src_data = static_cast<const int64_t*>(src_i64->data);
    int32_t* dst_data = static_cast<int32_t*>(dst_i32->data);
    for (size_t i = 0; i < nelem; ++i) {
      dst_data[i] = static_cast<int32_t>(src_data[i]);
    }
  }

  // Execute the graph
  int n_threads = 1;
  if (handle->graph) {
    ggml_graph_compute_with_ctx(handle->ctx, handle->graph, n_threads);
  }

  // Copy output data from ggml tensors → ExecuTorch output tensors
  size_t n_outputs = handle->outputs.size();
  for (size_t i = 0; i < n_outputs; ++i) {
    size_t out_idx = n_inputs + i;
    auto& et_tensor = args[out_idx]->toTensor();
    struct ggml_tensor* gt = handle->outputs[i];

    const size_t nelem = ggml_nelements(gt);

    if (gt->type == GGML_TYPE_F16) {
      // Convert fp16 output to fp32.
      const ggml_fp16_t* src = static_cast<const ggml_fp16_t*>(gt->data);
      float* dst = static_cast<float*>(et_tensor.mutable_data_ptr());
      ggml_fp16_to_fp32_row(src, dst, (int64_t) nelem);
    } else {
      size_t nbytes = ggml_nbytes(gt);
      memcpy(et_tensor.mutable_data_ptr(), gt->data, nbytes);
    }
  }

  return Error::Ok;
}

void GgmlBackendInterface::destroy(DelegateHandle* handle_raw) const {
  auto* handle = reinterpret_cast<GgmlDelegateHandle*>(handle_raw);
  if (handle) {
    if (handle->compute_ctx) {
      ggml_free(handle->compute_ctx);
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
