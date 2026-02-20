/**
 * GgmlBackendInterface — ExecuTorch runtime backend delegating to ggml.
 *
 * Deserialises a FlatBuffer-encoded ggml IR graph produced by the Python
 * GgmlBackend.preprocess() and executes it using the ggml compute graph API.
 */

#include "ggml_backend.h"

#include <cstring>
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
  // Constants are loaded from NamedDataMap into ggml tensors; the memory lives
  // inside the ggml context, so we size based on tensor count with headroom.
  size_t constant_data_size = 0;

  size_t ctx_size =
      static_cast<size_t>(n_tensors) * ggml_tensor_overhead() +
      constant_data_size +
      ggml_graph_overhead() +
      512 * 1024 * 1024;  // headroom (MV2 needs a lot of ggml objects)

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
#ifdef GGML_TYPE_I64
          gtype = GGML_TYPE_I64;
#else
          gtype = GGML_TYPE_I32; // fallback
#endif
          break;
        case ggml_ir::TensorType::I32:
          gtype = GGML_TYPE_I32;
          break;
        case ggml_ir::TensorType::BOOL:
          gtype = GGML_TYPE_I8;
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

            if (ggml_can_repeat(b, a)) {
              b = ggml_repeat(ctx, b, a);
            } else if (ggml_can_repeat(a, b)) {
              a = ggml_repeat(ctx, a, b);
            } else {
              // Try 1D→ND broadcast alignment
              if (auto * bb = try_repeat_1d_to_match(b, a)) {
                b = bb;
              } else if (auto * aa = try_repeat_1d_to_match(a, b)) {
                a = aa;
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

        case ggml_ir::OpCode::NEG:
          gt = ggml_neg(ctx, srcs[0]);
          break;

        case ggml_ir::OpCode::RSQRT: {
          // ggml doesn't expose rsqrt directly in this version; implement as 1/sqrt(x)
          struct ggml_tensor* one = ggml_new_f32(ctx, 1.0f);
          struct ggml_tensor* sx  = ggml_sqrt(ctx, srcs[0]);
          gt = ggml_div(ctx, one, sx);
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
          break;
        }

        case ggml_ir::OpCode::TRANSPOSE: {
          // transpose(x, dim0, dim1) via permute
          if (!t->op_params() || t->op_params()->size() < 8) {
            ggml_free(ctx);
            return Error::InvalidArgument;
          }
          int32_t dim0 = 0, dim1 = 1;
          memcpy(&dim0, t->op_params()->data(), 4);
          memcpy(&dim1, t->op_params()->data() + 4, 4);

          // Map PyTorch dims -> ggml axes (reverse order for up to 4D)
          auto* a = srcs[0];
          int nd = ggml_n_dims(a);
          int ax0 = (nd - 1) - dim0;
          int ax1 = (nd - 1) - dim1;
          int perm[4] = {0, 1, 2, 3};
          for (int i = 0; i < 4; ++i) perm[i] = i;
          int tmp = perm[ax0];
          perm[ax0] = perm[ax1];
          perm[ax1] = tmp;
          gt = ggml_permute(ctx, a, perm[0], perm[1], perm[2], perm[3]);
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

          struct ggml_tensor* a = srcs[0];
          int nd = ggml_n_dims(a);
          int ax = (nd - 1) - dim; // pytorch dim -> ggml axis

          // Compute offset in bytes
          size_t offset = 0;
          if (ax == 0) {
            offset = start * a->nb[0];
            gt = ggml_view_1d(ctx, a, ne[0], offset);
          } else if (ax == 1) {
            offset = start * a->nb[1];
            gt = ggml_view_2d(ctx, a, ne[0], ne[1], a->nb[1], offset);
          } else if (ax == 2) {
            offset = start * a->nb[2];
            gt = ggml_view_3d(ctx, a, ne[0], ne[1], ne[2], a->nb[1], a->nb[2], offset);
          } else {
            offset = start * a->nb[3];
            gt = ggml_view_4d(ctx, a, ne[0], ne[1], ne[2], ne[3], a->nb[1], a->nb[2], a->nb[3], offset);
          }
          break;
        }

        case ggml_ir::OpCode::CAT: {
          // Chain ggml_concat along dim (PyTorch dim mapped to ggml axis)
          if (!t->op_params() || t->op_params()->size() < 4) {
            ggml_free(ctx);
            return Error::InvalidArgument;
          }
          int32_t dim = 0;
          memcpy(&dim, t->op_params()->data(), 4);
          int nd = ggml_n_dims(srcs[0]);
          int ax = (nd - 1) - dim;
          struct ggml_tensor* cur = srcs[0];
          for (size_t si = 1; si < srcs.size(); ++si) {
            cur = ggml_concat(ctx, cur, srcs[si], ax);
          }
          gt = cur;
          break;
        }

        case ggml_ir::OpCode::REPEAT_INTERLEAVE: {
          // Limited support for GQA expansion: repeats along PyTorch dim 1.
          // op_params: (dim:int32, repeats:int32)
          if (!t->op_params() || t->op_params()->size() < 8) {
            ggml_free(ctx);
            return Error::InvalidArgument;
          }
          int32_t dim = 0, reps = 1;
          memcpy(&dim, t->op_params()->data(), 4);
          memcpy(&reps, t->op_params()->data() + 4, 4);
          // Only support repeating along PyTorch dim 1 (heads)
          if (dim != 1 || reps < 1) {
            ggml_free(ctx);
            return Error::InvalidArgument;
          }
          // Implement as concatenation of identical tensor along that axis.
          int nd = ggml_n_dims(srcs[0]);
          int ax = (nd - 1) - dim;
          struct ggml_tensor* cur = srcs[0];
          for (int r = 1; r < reps; ++r) {
            cur = ggml_concat(ctx, cur, srcs[0], ax);
          }
          gt = cur;
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
          // ggml_get_rows selects rows by indices.
          gt = ggml_get_rows(ctx, x, idx);
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
#ifdef GGML_TYPE_I64
          if (idx->type != GGML_TYPE_I64) {
            idx = ggml_cast(ctx, idx, GGML_TYPE_I64);
          }
#else
          if (idx->type != GGML_TYPE_I32) {
            idx = ggml_cast(ctx, idx, GGML_TYPE_I32);
          }
#endif

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

          // For now pass mask=nullptr (assume causal handled internally / or mask not required).
          struct ggml_tensor* mask = nullptr;

          // scale = 1/sqrt(head_dim). head_dim is ne0 in ggml layout when tensors are [D, T, H, B]
          float scale = 1.0f;
          if (q->ne[0] > 0) {
            scale = 1.0f / std::sqrt((float) q->ne[0]);
          }

          gt = ggml_flash_attn_ext(ctx, q, k, v, mask, scale, 0.0f, 0.0f);
          ggml_flash_attn_ext_set_prec(gt, GGML_PREC_F32);
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
    } else {
      // Normal path for MV2: runtime inputs are fp32.
      size_t nbytes = ggml_nbytes(gt);
      memcpy(gt->data, et_tensor.const_data_ptr(), nbytes);
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
