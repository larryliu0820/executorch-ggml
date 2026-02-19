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
      16 * 1024 * 1024;  // extra headroom (weights live in ctx)

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
      switch (n_dims) {
        case 1:
          gt = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne[0]);
          break;
        case 2:
          gt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne[0], ne[1]);
          break;
        case 3:
          gt = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne[0], ne[1], ne[2]);
          break;
        default:
          gt = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne[0], ne[1], ne[2], ne[3]);
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
        // Load directly into ggml tensor memory.
        const size_t nbytes = ggml_nbytes(gt);
        auto err = ndm->load_data_into(key, gt->data, nbytes);
        if (err != Error::Ok) {
          ggml_free(ctx);
          return err;
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
        case ggml_ir::OpCode::ADD:
          gt = ggml_add(ctx, srcs[0], srcs[1]);
          break;

        case ggml_ir::OpCode::MUL_MAT:
          gt = ggml_mul_mat(ctx, srcs[0], srcs[1]);
          break;

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

          // Add bias if present
          if (bias && gt) {
            gt = ggml_add(ctx, gt, bias);
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
          // mean(x, dim)
          int32_t dim = -1;
          if (t->op_params() && t->op_params()->size() >= 4) {
            memcpy(&dim, t->op_params()->data(), sizeof(int32_t));
          }
          // ggml_mean computes mean over all dimensions
          // For single-dim mean, we need ggml_mean over a specific axis
          // Note: ggml may not have axis-specific mean yet
          // Workaround: use ggml_sum / ggml_repeat to normalize
          // This is a potential blocking issue
          gt = ggml_mean(ctx, srcs[0]);
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

        default:
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
    size_t nbytes = ggml_nbytes(gt);
    memcpy(gt->data, et_tensor.const_data_ptr(), nbytes);
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
    size_t nbytes = ggml_nbytes(gt);
    memcpy(et_tensor.mutable_data_ptr(), gt->data, nbytes);
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
