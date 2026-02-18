/**
 * GgmlBackendInterface — ExecuTorch runtime backend delegating to ggml.
 *
 * Deserialises a FlatBuffer-encoded ggml IR graph produced by the Python
 * GgmlBackend.preprocess() and executes it using the ggml compute graph API.
 */

#include "ggml_backend.h"

#include <cstring>
#include <vector>

#include "ggml_ir_generated.h"  // flatc-generated header

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

struct GgmlDelegateHandle : public DelegateHandle {
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

  // Calculate memory needed for ggml context
  // Each tensor needs overhead + data for constants
  size_t constant_data_size = 0;
  for (int i = 0; i < n_tensors; ++i) {
    const auto* t = fb_tensors->Get(i);
    if (t->data() && t->data()->size() > 0) {
      constant_data_size += t->data()->size();
      // Alignment padding
      constant_data_size += 64;
    }
  }

  size_t ctx_size =
      static_cast<size_t>(n_tensors) * ggml_tensor_overhead() +
      constant_data_size +
      ggml_graph_overhead() +
      1024 * 1024;  // extra headroom

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

    if (op == ggml_ir::OpCode_NONE) {
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

      // Copy constant data if present
      if (t->data() && t->data()->size() > 0) {
        memcpy(gt->data, t->data()->data(), t->data()->size());
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
        case ggml_ir::OpCode_ADD:
          gt = ggml_add(ctx, srcs[0], srcs[1]);
          break;

        case ggml_ir::OpCode_MUL_MAT:
          gt = ggml_mul_mat(ctx, srcs[0], srcs[1]);
          break;

        case ggml_ir::OpCode_LEAKY_RELU: {
          float negative_slope = 0.01f;
          if (t->op_params() && t->op_params()->size() >= 4) {
            memcpy(&negative_slope, t->op_params()->data(), sizeof(float));
          }
          gt = ggml_leaky_relu(ctx, srcs[0], negative_slope, false);
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

  auto* handle = static_cast<GgmlDelegateHandle*>(handle_raw);

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
  auto* handle = static_cast<GgmlDelegateHandle*>(handle_raw);
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
