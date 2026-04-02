/**
 * GgmlBackendInterface — ExecuTorch runtime backend delegating to ggml.
 *
 * Deserialises a FlatBuffer-encoded ggml IR graph produced by the Python
 * GgmlBackend.preprocess() and executes it using the ggml compute graph API.
 */

#include "ggml_backend.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <deque>
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
// ggml_cgraph struct definition for graph compaction (stripping view nodes).
// Only the struct layout is needed — no ggml-internal functions are called.
#include "../third-party/llama.cpp/ggml/src/ggml-impl.h"
#include <ggml-cpu.h>
#include <ggml-backend.h>

#ifdef GGML_USE_METAL
#include <ggml-metal.h>
#endif

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>

// --- Extracted headers (infrastructure, helpers, op dispatch) ---
// These are included BEFORE the namespace block because each header
// opens/closes its own `namespace executorch_ggml { }` scope.
#include "ops/sym_expr.h"
#include "ops/custom_ops.h"
#include "ops/host_data_accessor.h"
#include "ops/data_structures.h"
#include "ops/profiling.h"
#include "ops/helpers.h"
#include "ops/build_context.h"
#include "ops/ops_activation.h"
#include "ops/ops_arithmetic.h"
#include "ops/ops_linalg.h"
#include "ops/ops_shape.h"
#include "ops/ops_indexing.h"
#include "ops/ops_normalization.h"
#include "ops/ops_convolution.h"
#include "ops/ops_creation.h"
#include "ops/ops_comparison.h"
#include "ops/ops_special.h"

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

// input_ne_overrides: nullptr on first call (use serialized shapes).
//   In M2+ this will carry runtime shapes for dynamic dims.
// n_overrides: number of int64_t values (n_inputs × 4).
// input_data: optional per-input data for pre-populating input tensors.
//   When non-null, eager ops get correct values instead of uninitialized memory.
static Error build_graph(
    GgmlDelegateHandle* handle,
    GraphInstance* gi,
    const int64_t* input_ne_overrides,
    size_t n_overrides,
    const std::vector<InputDataOverride>* input_data = nullptr,
    bool verbose = true) {

  (void)n_overrides;

  // Metal binary ops require F32 inputs for ADD/SUB/MUL/DIV.
  // Metal binary ops require F32 inputs.  CUDA binary broadcast supports
  // F32 and F16 but NOT BF16 — we handle BF16 casts inline below.
#ifdef GGML_USE_METAL
  const bool metal_f32_binops = ggml_backend_is_metal(handle->backend);
#else
  const bool metal_f32_binops = false;
#endif
  const bool cuda_bf16_cast = (handle->backend != handle->backend_cpu) && !metal_f32_binops;
  // On CUDA, don't cast activations back to BF16 after ops that had to convert
  // to F32. Keeping activations in F32 avoids wasteful F32→BF16→F32 chains
  // (each is a separate kernel launch). ggml_mul_mat handles mixed types
  // natively (BF16 weights × F32 activations), and the output copy path
  // already handles F32→BF16 conversion.
  const bool skip_bf16_castback = cuda_bf16_cast;

  // --- Tear down previous context (no-op on first call) ---
  // Preserve the scheduler — it will be reset+realloc'd in execute().
  if (gi->ctx) {
    ggml_free(gi->ctx);
    gi->ctx = nullptr;
  }
  gi->graph = nullptr;
  gi->inputs.clear();
  gi->outputs.clear();
  gi->deferred_i64_to_i32.clear();

  // --- Parse IR from the saved copy ---
  auto t_bg_start = std::chrono::high_resolution_clock::now();
  const auto* fb_graph = ggml_ir::GetGgmlGraph(handle->ir_copy.data());
  if (!fb_graph || !fb_graph->tensors()) {
    return Error::InvalidArgument;
  }
  const auto* fb_tensors = fb_graph->tensors();
  const int n_tensors = static_cast<int>(fb_tensors->size());

  // === Phase A: calculate ctx size, create ggml context ===
  // Context only needs tensor metadata + graph structure.
  // Actual tensor data is allocated by gallocr (backend scheduler) or by
  // a temporary CPU host buffer for leaf tensors (constants / inputs).

  // Estimate graph size: 4× the IR tensors to account for compound ops.
  size_t est_graph_size = static_cast<size_t>(n_tensors) * 4;
  if (est_graph_size < GGML_DEFAULT_GRAPH_SIZE) {
    est_graph_size = GGML_DEFAULT_GRAPH_SIZE;
  }

  // Build sym_dim_values early: needed for shape-aware eager size estimation.
  // Maps symbolic variable ID → runtime concrete value from input tensors.
  std::unordered_map<int32_t, int64_t> sym_dim_values;
  if (input_ne_overrides) {
    for (int i = 0; i < n_tensors; ++i) {
      const auto* ti = fb_tensors->Get(i);
      if (!ti->is_input() || !ti->sym_dim_ids() || !ti->ne()) continue;
      int input_idx = ti->input_index();
      for (size_t d = 0; d < ti->sym_dim_ids()->size() && d < 4; ++d) {
        int32_t sid = ti->sym_dim_ids()->Get(d);
        if (sid >= 0) {
          sym_dim_values[sid] = input_ne_overrides[input_idx * 4 + d];
        }
      }
    }
    if (verbose) {
      fprintf(stderr, "[ggml_backend] sym_dim_values: ");
      for (auto& [k, v] : sym_dim_values) {
        fprintf(stderr, "s%d=%lld ", k, (long long)v);
      }
      fprintf(stderr, "\n");
    }
  }

  // Eager ops create data-filled leaf tensors during op dispatch, allocated
  // from the context pool. Use serialized elem_size metadata to compute
  // accurate data sizes from resolved runtime shapes.
  size_t eager_data_estimate = 0;
  std::unordered_set<uint64_t> seen_causal_masks;
  for (int i = 0; i < n_tensors; ++i) {
    const auto* t = fb_tensors->Get(i);
    uint8_t elem_size = t->elem_size();

    if (elem_size > 0) {
      // Resolve shape using tensor's own ne + sym_dim_ids
      int64_t ne[4]; int nd;
      resolve_ir_shape(t, input_ne_overrides, sym_dim_values, ne, nd);
      eager_data_estimate += (size_t)ne[0] * ne[1] * ne[2] * ne[3] * elem_size;
      continue;
    }

    // Runtime special case: LLAMA_ATTENTION causal mask [T_kv, T_q] × F16.
    // Masks are cached per unique (T_kv, T_q) shape, so only count once.
    if (static_cast<ggml_ir::OpCode>(t->op()) == ggml_ir::OpCode::LLAMA_ATTENTION) {
      bool is_causal = false;
      if (t->op_params() && t->op_params()->size() >= 4) {
        int32_t ic; memcpy(&ic, t->op_params()->data(), sizeof(int32_t));
        is_causal = (ic != 0);
      }
      if (is_causal && t->src_ids() && t->src_ids()->size() >= 2) {
        int64_t q_ne[4], k_ne[4]; int qnd, knd;
        resolve_ir_shape(fb_tensors->Get(t->src_ids()->Get(0)),
                         input_ne_overrides, sym_dim_values, q_ne, qnd);
        resolve_ir_shape(fb_tensors->Get(t->src_ids()->Get(1)),
                         input_ne_overrides, sym_dim_values, k_ne, knd);
        uint64_t mask_key = ((uint64_t)k_ne[1] << 32) | (uint64_t)q_ne[1];
        if (seen_causal_masks.insert(mask_key).second) {
          eager_data_estimate += (size_t)k_ne[1] * q_ne[1] * sizeof(ggml_fp16_t);
        }
      }
    }
  }
  // Safety margin: 2x + 4 MB for misc small allocs (make_f32_scalar, etc.)
  eager_data_estimate = eager_data_estimate * 2 + 4 * 1024 * 1024;

  size_t ctx_size =
      static_cast<size_t>(n_tensors) * 8 * ggml_tensor_overhead() +
      ggml_graph_overhead_custom(est_graph_size, false) +
      eager_data_estimate;
  if (verbose) {
    fprintf(stderr, "[ggml_backend] ctx_size=%zu MB (eager=%zu MB, tensor_oh=%zu MB, graph_oh=%zu MB), n_tensors=%d\n",
            ctx_size / (1024*1024), eager_data_estimate / (1024*1024),
            (static_cast<size_t>(n_tensors) * 8 * ggml_tensor_overhead()) / (1024*1024),
            ggml_graph_overhead_custom(est_graph_size, false) / (1024*1024), n_tensors);
  }

  struct ggml_init_params params = {
      /* .mem_size   = */ ctx_size,
      /* .mem_buffer = */ nullptr,
      /* .no_alloc   = */ true,
  };

  struct ggml_context* ctx = ggml_init(params);
  if (!ctx) {
    return Error::MemoryAllocationFailed;
  }
  auto t_bg_phaseA = std::chrono::high_resolution_clock::now();

  // --- Create per-graph scheduler (first build only) ---
  // Preserved across rebuilds for gallocr buffer reuse. Only created once
  // per GraphInstance (first build for this shape key).
  if (!gi->sched) {
    std::vector<ggml_backend_t> sched_backends;
    sched_backends.push_back(handle->backend);
    if (handle->backend_cpu && handle->backend_cpu != handle->backend) {
      sched_backends.push_back(handle->backend_cpu);
    }
    const size_t max_nodes = std::max<size_t>(
        GGML_DEFAULT_GRAPH_SIZE, static_cast<size_t>(n_tensors) * 4);
    gi->sched = ggml_backend_sched_new(
        sched_backends.data(),
        /*bufts=*/nullptr,
        static_cast<int>(sched_backends.size()),
        max_nodes,
        /*parallel=*/false,
        /*op_offload=*/true);
    if (!gi->sched) {
      ggml_free(ctx);
      return Error::MemoryAllocationFailed;
    }
  }

  // === Phase B: tensor creation loop + op switch ===
  // Map from IR tensor id → ggml_tensor*
  std::vector<struct ggml_tensor*> id_to_tensor(n_tensors, nullptr);

  // Temporary CPU host buffer for leaf tensor data.
  // Freed after gallocr copies constants to backend buffers (Phase C step 6).
  ggml_backend_buffer_t host_buf = nullptr;

  // RAII guard: ensure host_buf is freed on early return (error paths).
  struct HostBufGuard {
    ggml_backend_buffer_t& buf;
    ~HostBufGuard() { if (buf) { ggml_backend_buffer_free(buf); buf = nullptr; } }
  } host_buf_guard{host_buf};

  // Track inputs and outputs
  std::vector<std::pair<int, struct ggml_tensor*>> input_pairs;  // (index, tensor)
  std::vector<std::pair<int, struct ggml_tensor*>> output_pairs; // (index, tensor)
  // Deferred I64→I32 casts (for input tensors).
  std::vector<std::pair<struct ggml_tensor*, struct ggml_tensor*>> deferred_i64_to_i32;
  // Leaf tensors placed in shared buffers (const_buf / mutable_buf).
  std::vector<SharedLeaf> shared_leaves;
  // Tensors pinned to CPU (for custom ops on GPU backends).
  std::vector<struct ggml_tensor*> cpu_pinned;
  // Track tensors derived from graph inputs (transitive closure).
  std::unordered_set<struct ggml_tensor*> input_derived;
  // sym_dim_values already built above (Phase A, before ggml_init).

  // --- Shape resolution helper (shared by leaf and op passes) ---
  auto resolve_shape = [&](const ggml_ir::Tensor* t, int64_t ne[4], int& n_dims) {
    resolve_ir_shape(t, input_ne_overrides, sym_dim_values, ne, n_dims);
  };

  // --- IR type → ggml type helper ---
  auto resolve_gtype = [](const ggml_ir::Tensor* t) -> ggml_type {
    switch (t->type()) {
      case ggml_ir::TensorType::F16:  return GGML_TYPE_F16;
      case ggml_ir::TensorType::I64:  return GGML_TYPE_I64;
      case ggml_ir::TensorType::I32:  return GGML_TYPE_I32;
      case ggml_ir::TensorType::BOOL: return GGML_TYPE_I32;  // stored as I32
      case ggml_ir::TensorType::BF16: return GGML_TYPE_BF16;
      case ggml_ir::TensorType::Q8_0: return GGML_TYPE_Q8_0;
      case ggml_ir::TensorType::Q6_K: return GGML_TYPE_Q6_K;
      case ggml_ir::TensorType::Q4_0: return GGML_TYPE_Q4_0;
      default:                        return GGML_TYPE_F32;
    }
  };

  // === Pre-calculate total leaf data size for tallocr ===
  // Only count leaves NOT in leaf_buf_map (those use shared buffers instead).
  {
    size_t total_leaf_data = 0;
    const size_t alignment = ggml_backend_buft_get_alignment(ggml_backend_cpu_buffer_type());
    for (int i = 0; i < n_tensors; ++i) {
      const auto* t = fb_tensors->Get(i);
      if (static_cast<ggml_ir::OpCode>(t->op()) != ggml_ir::OpCode::NONE) continue;
      if (handle->leaf_buf_map.count(t->id())) continue;  // shared buffer
      int64_t leaf_ne[4] = {1,1,1,1}; int leaf_ndims;
      resolve_shape(t, leaf_ne, leaf_ndims);
      ggml_type gtype = resolve_gtype(t);
      size_t tensor_bytes = ggml_row_size(gtype, leaf_ne[0]) * leaf_ne[1] * leaf_ne[2] * leaf_ne[3];
      // Match ggml_tallocr_alloc: GGML_PAD(alloc_size, alignment)
      size_t padded = ((tensor_bytes + alignment - 1) / alignment) * alignment;
      total_leaf_data += padded;
    }
    // Safety margin: 1% + 1 MB for alignment rounding and initial offset.
    total_leaf_data += total_leaf_data / 100 + 1024 * 1024;
    if (verbose) fprintf(stderr, "[ggml_backend] total_leaf_data=%zu MB\n", total_leaf_data / (1024*1024));
    host_buf = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), total_leaf_data);
    if (!host_buf) {
      fprintf(stderr, "[ggml_backend] ERROR: failed to allocate host buffer of %zu bytes\n", total_leaf_data);
      ggml_free(ctx);
      return Error::MemoryAllocationFailed;
    }
    // Zero-initialize host buffer so leaf tensors without explicit data
    // (e.g. KV cache index tensors) start at 0 instead of uninitialized memory.
    ggml_backend_buffer_clear(host_buf, 0);
  }
  struct ggml_tallocr tallocr = ggml_tallocr_new(host_buf);

  // === Phase B: single-pass tensor creation loop ===
  if (verbose) { fprintf(stderr, "[ggml_backend] Phase B: creating tensors (single pass)...\n"); fflush(stderr); }
  static int s_last_processed = -1;
  static int s_n_tensors = 0;
  s_last_processed = -1;
  s_n_tensors = n_tensors;
  // Always use native ggml decomposition for comparison/bitwise/logical ops.
  // The custom CPU callbacks read values as int32_t, which gives wrong results
  // when inputs are F32 (reinterprets float bit patterns as integers).
  // Native decompositions cast inputs to F32 first, avoiding this bug.
  bool use_native_cmp_ops = true;
  const char* cmp_env = std::getenv("GGML_NATIVE_CMP_OPS");
  if (cmp_env) use_native_cmp_ops = (std::string(cmp_env) != "0");
  // Host-side accessor for reading tensor data that may live on a device buffer.
  HostDataAccessor host_acc;
  host_acc.input_derived = &input_derived;
  // Cache causal attention masks so identical (T_kv, T_q) shapes reuse one allocation.
  std::unordered_map<uint64_t, struct ggml_tensor*> causal_mask_cache;
  int last_processed = -1;
  for (int i = 0; i < n_tensors; ++i) {
    const auto* t = fb_tensors->Get(i);
    const int tid = t->id();
    const auto op = static_cast<ggml_ir::OpCode>(t->op());
    s_last_processed = i;
    // (debug removed)
    int64_t ne[4] = {1,1,1,1}; int n_dims = 4;
    resolve_shape(t, ne, n_dims);

    if (op == ggml_ir::OpCode::NONE) {
      // --- Leaf tensor: create, allocate from host buffer, load data ---
      ggml_type gtype = resolve_gtype(t);
      struct ggml_tensor* gt = nullptr;
      switch (n_dims) {
        case 1:  gt = ggml_new_tensor_1d(ctx, gtype, ne[0]); break;
        case 2:  gt = ggml_new_tensor_2d(ctx, gtype, ne[0], ne[1]); break;
        case 3:  gt = ggml_new_tensor_3d(ctx, gtype, ne[0], ne[1], ne[2]); break;
        default: gt = ggml_new_tensor_4d(ctx, gtype, ne[0], ne[1], ne[2], ne[3]); break;
      }
      // Check if this leaf has a pre-allocated slot in const_buf or mutable_buf.
      auto it = handle->leaf_buf_map.find(tid);
      if (it != handle->leaf_buf_map.end()) {
        auto& slot = it->second;
        // Both const and mutable leaves point directly at their shared buffer.
        // mutable_buf is allocated on CPU, so custom inplace ops write directly
        // into it without scheduler copy-buffer aliasing issues.
        gt->data = static_cast<char*>(ggml_backend_buffer_get_base(slot.buf)) + slot.offset;
        gt->buffer = slot.buf;
        ggml_set_input(gt);  // tell scheduler to skip this tensor
        shared_leaves.push_back({gt, slot.buf, slot.offset});
      } else {
        // Input tensors and any other leaves without data_key: allocate from host buffer.
        ggml_tallocr_alloc(&tallocr, gt);

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
      }

      // Pre-populate input tensor data so eager ops downstream
      // (comparisons, bitwise, etc.) read correct values during build.
      if (t->is_input() && input_data && t->input_index() >= 0 &&
          static_cast<size_t>(t->input_index()) < input_data->size()) {
        const auto& ido = (*input_data)[t->input_index()];
        if (ido.data) {
          const size_t nelem = ggml_nelements(gt);
          if (gt->type == GGML_TYPE_I32 && ido.et_scalar_type == 4 /* Long */) {
            const int64_t* src = static_cast<const int64_t*>(ido.data);
            int32_t* dst = static_cast<int32_t*>(gt->data);
            for (size_t j = 0; j < nelem; ++j) dst[j] = static_cast<int32_t>(src[j]);
          } else if (gt->type == GGML_TYPE_F32 && ido.et_scalar_type == 6 /* Float */) {
            memcpy(gt->data, ido.data, std::min(ido.nbytes, ggml_nbytes(gt)));
          } else if (gt->type == GGML_TYPE_I32 && ido.et_scalar_type == 11 /* Bool */) {
            const bool* src = static_cast<const bool*>(ido.data);
            int32_t* dst = static_cast<int32_t*>(gt->data);
            for (size_t j = 0; j < nelem; ++j) dst[j] = src[j] ? 1 : 0;
          } else {
            memcpy(gt->data, ido.data, std::min(ido.nbytes, ggml_nbytes(gt)));
          }
        }
      }

      if (t->is_input()) {
        ggml_set_input(gt);  // Mark as input early so try_eager_scalar_binop skips it
        input_pairs.emplace_back(t->input_index(), gt);
        input_derived.insert(gt);
      }
      if (t->is_output()) {
        int out_idx = t->input_index();
        output_pairs.emplace_back(out_idx >= 0 ? out_idx : (int)output_pairs.size(), gt);
      }
      id_to_tensor[tid] = gt;
      continue;
    }

    // --- Op tensor: build the ggml operation ---
    s_last_processed = -(i + 10000); // encode: op tensor entering dispatch
    struct ggml_tensor* gt = nullptr;

    // Resolve sources
    std::vector<struct ggml_tensor*> srcs;
    if (t->src_ids()) {
      for (size_t s = 0; s < t->src_ids()->size(); ++s) {
        int src_id = t->src_ids()->Get(s);
        srcs.push_back(id_to_tensor[src_id]);
      }
    }

    // Construct BuildContext for this tensor
    BuildContext bc{ctx, t, srcs, host_acc, input_derived, handle, gi,
                    cpu_pinned, id_to_tensor, sym_dim_values, causal_mask_cache,
                    deferred_i64_to_i32, input_pairs,
                    {ne[0], ne[1], ne[2], ne[3]},
                    metal_f32_binops, cuda_bf16_cast, skip_bf16_castback,
                    use_native_cmp_ops, verbose};

    switch (op) {
        // --- Arithmetic ---
        case ggml_ir::OpCode::ADD:           gt = build_op_add(bc); break;
        case ggml_ir::OpCode::SUB:           gt = build_op_sub(bc); break;
        case ggml_ir::OpCode::MUL:           gt = build_op_mul(bc); break;
        case ggml_ir::OpCode::MUL_SCALAR:    gt = build_op_mul_scalar(bc); break;
        case ggml_ir::OpCode::DIV:           gt = build_op_div(bc); break;
        case ggml_ir::OpCode::NEG:           gt = build_op_neg(bc); break;
        case ggml_ir::OpCode::RSQRT:         gt = build_op_rsqrt(bc); break;
        case ggml_ir::OpCode::POW:           gt = build_op_pow(bc); break;
        case ggml_ir::OpCode::MEAN:          gt = build_op_mean(bc); break;

        // --- Linear algebra ---
        case ggml_ir::OpCode::MUL_MAT:       gt = build_op_mul_mat(bc); break;
        case ggml_ir::OpCode::BMM:           gt = build_op_bmm(bc); break;
        case ggml_ir::OpCode::LINEAR:        gt = build_op_linear(bc); break;
        case ggml_ir::OpCode::EMBEDDING:     gt = build_op_embedding(bc); break;

        // --- Activation ---
        case ggml_ir::OpCode::SILU:          gt = build_op_silu(bc); break;
        case ggml_ir::OpCode::RELU:          gt = build_op_relu(bc); break;
        case ggml_ir::OpCode::TANH:          gt = build_op_tanh(bc); break;
        case ggml_ir::OpCode::GELU:          gt = build_op_gelu(bc); break;
        case ggml_ir::OpCode::LEAKY_RELU:    gt = build_op_leaky_relu(bc); break;
        case ggml_ir::OpCode::SIGMOID:       gt = build_op_sigmoid(bc); break;
        case ggml_ir::OpCode::SOFTMAX:       gt = build_op_softmax(bc); break;
        case ggml_ir::OpCode::HARDTANH:      gt = build_op_hardtanh(bc); break;
        case ggml_ir::OpCode::COS:           gt = build_op_cos(bc); break;
        case ggml_ir::OpCode::SIN:           gt = build_op_sin(bc); break;

        // --- Shape ---
        case ggml_ir::OpCode::VIEW:          gt = build_op_view(bc); break;
        case ggml_ir::OpCode::PERMUTE:       gt = build_op_permute(bc); break;
        case ggml_ir::OpCode::TRANSPOSE:     gt = build_op_transpose(bc); break;
        case ggml_ir::OpCode::UNSQUEEZE:     gt = build_op_unsqueeze(bc); break;
        case ggml_ir::OpCode::SLICE:         gt = build_op_slice(bc); break;

        // --- Indexing ---
        case ggml_ir::OpCode::CAT:           gt = build_op_cat(bc); break;
        case ggml_ir::OpCode::REPEAT:        gt = build_op_repeat(bc); break;
        case ggml_ir::OpCode::REPEAT_INTERLEAVE: gt = build_op_repeat_interleave(bc); break;
        case ggml_ir::OpCode::INDEX:         gt = build_op_index(bc); break;
        case ggml_ir::OpCode::INDEX_MULTI:   gt = build_op_index_multi(bc); break;
        case ggml_ir::OpCode::INDEX_PUT:     gt = build_op_index_put(bc); break;

        // --- Normalization ---
        case ggml_ir::OpCode::LAYER_NORM:    gt = build_op_layer_norm(bc); break;
        case ggml_ir::OpCode::RMS_NORM:      gt = build_op_rms_norm(bc); break;
        case ggml_ir::OpCode::BATCH_NORM:    gt = build_op_batch_norm(bc); break;

        // --- Convolution ---
        case ggml_ir::OpCode::CONV_2D:       gt = build_op_conv_2d(bc, /*is_dw=*/false); break;
        case ggml_ir::OpCode::CONV_2D_DW:    gt = build_op_conv_2d(bc, /*is_dw=*/true); break;
        case ggml_ir::OpCode::CONV_1D:       gt = build_op_conv_1d(bc, /*is_dw=*/false); break;
        case ggml_ir::OpCode::CONV_1D_DW:    gt = build_op_conv_1d(bc, /*is_dw=*/true); break;
        case ggml_ir::OpCode::PAD:           gt = build_op_pad(bc); break;

        // --- Creation / cast ---
        case ggml_ir::OpCode::ARANGE:        gt = build_op_arange(bc); break;
        case ggml_ir::OpCode::FULL:          gt = build_op_full(bc); break;
        case ggml_ir::OpCode::CUMSUM:        gt = build_op_cumsum(bc); break;
        case ggml_ir::OpCode::ARGMAX:        gt = build_op_argmax(bc); break;
        case ggml_ir::OpCode::CAST:          gt = build_op_cast(bc); break;
        case ggml_ir::OpCode::WHERE:         gt = build_op_where(bc); break;

        // --- Comparison / logical ---
        case ggml_ir::OpCode::EQ:            gt = build_op_eq(bc); break;
        case ggml_ir::OpCode::NE:            gt = build_op_ne(bc); break;
        case ggml_ir::OpCode::LE:            gt = build_op_le(bc); break;
        case ggml_ir::OpCode::LT:            gt = build_op_lt(bc); break;
        case ggml_ir::OpCode::GT:            gt = build_op_gt(bc); break;
        case ggml_ir::OpCode::GE:            gt = build_op_ge(bc); break;
        case ggml_ir::OpCode::BITWISE_AND:   gt = build_op_bitwise_and(bc); break;
        case ggml_ir::OpCode::BITWISE_OR:    gt = build_op_bitwise_or(bc); break;
        case ggml_ir::OpCode::LOGICAL_NOT:   gt = build_op_logical_not(bc); break;
        case ggml_ir::OpCode::ANY:           gt = build_op_any(bc); break;

        // --- Special / attention ---
        case ggml_ir::OpCode::LLAMA_ATTENTION: gt = build_op_llama_attention(bc); break;
        case ggml_ir::OpCode::UPDATE_CACHE:  gt = build_op_update_cache(bc); break;
        case ggml_ir::OpCode::ROPE:          gt = build_op_rope(bc); break;
        case ggml_ir::OpCode::REMAINDER:   gt = build_op_remainder(bc); break;

        default:
          fprintf(stderr, "[executorch-ggml] Unsupported OpCode %d\n", (int) op);
          ggml_free(ctx);
          return Error::InvalidArgument;
      }

    // Propagate input-derived tracking: if any source is input-derived,
    // the result is too.
    if (gt) {
      for (auto* s : srcs) {
        if (input_derived.count(s)) {
          input_derived.insert(gt);
          break;
        }
      }
    }

    id_to_tensor[tid] = gt;
    last_processed = i;
    if (t->is_output()) {
      int out_idx = t->input_index();
      output_pairs.emplace_back(out_idx >= 0 ? out_idx : (int)output_pairs.size(), gt);
    }
  }
  auto t_bg_phaseB = std::chrono::high_resolution_clock::now();
  if (verbose) { fprintf(stderr, "[ggml_backend] Phase B complete, last_processed=%d/%d\n", last_processed, n_tensors); fflush(stderr); }

  // Sort outputs by output_index (stored in input_index field for outputs)
  // to match the order ExecuTorch expects in the output args span.
  std::sort(output_pairs.begin(), output_pairs.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
  std::vector<struct ggml_tensor*> output_tensors;
  for (auto& [idx, tensor] : output_pairs) {
    // Output tensors must be contiguous for backend_tensor_get in execute().
    if (!ggml_is_contiguous(tensor)) {
      tensor = ggml_cont(ctx, tensor);
    }
    output_tensors.push_back(tensor);
  }

  // === Build compute graph ===
  auto t_bg_graph_start = std::chrono::high_resolution_clock::now();
  struct ggml_cgraph* graph = ggml_new_graph_custom(ctx, est_graph_size, false);
  // Add cache write (ggml_cpy) nodes FIRST so they appear before SDPA
  // reads in the graph's topological order. These are side-effect writes
  // to mutable KV cache buffers (INDEX_PUT with scatter_axis != 1).
  for (struct ggml_tensor* t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
    if (t->op == GGML_OP_CPY && (t->flags & GGML_TENSOR_FLAG_OUTPUT)) {
      ggml_build_forward_expand(graph, t);
    }
  }
  for (auto* out : output_tensors) {
    ggml_build_forward_expand(graph, out);
  }
  // Strip RESHAPE nodes from the graph. RESHAPE is pure metadata (same
  // data pointer, different ne[]) with no compute kernel. Stripping
  // reduces per-node dispatch overhead and CUDA graph capture time.
  // The gallocr handles stripped tensors via src[] chain traversal.
  // Safety: keep RESHAPE nodes that ARE graph outputs.
  {
    std::unordered_set<struct ggml_tensor*> output_set(
        output_tensors.begin(), output_tensors.end());
    int w = 0;
    for (int r = 0; r < graph->n_nodes; r++) {
      if (graph->nodes[r]->op == GGML_OP_RESHAPE &&
          !output_set.count(graph->nodes[r])) {
        continue;
      }
      graph->nodes[w++] = graph->nodes[r];
    }
    graph->n_nodes = w;
  }
  auto t_bg_graph_end = std::chrono::high_resolution_clock::now();
  if (verbose) fprintf(stderr, "[ggml_backend] graph built: %d nodes, %d leafs (max %zu)\n",
          ggml_graph_n_nodes(graph), (int)0, est_graph_size);
  // Sort inputs by input_index
  std::sort(input_pairs.begin(), input_pairs.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });

  // Replace I64 inputs with their I32 deferred-cast destinations.
  // The I64 tensor is NOT part of the ggml compute graph (the cast is
  // deferred to the host), so gallocr won't allocate it.  By swapping
  // the I32 destination into the input list, execute() can copy ET int64
  // data directly into the I32 tensor via the existing I32+Long path.
  for (auto& [idx, tensor] : input_pairs) {
    for (auto& [src_i64, dst_i32] : deferred_i64_to_i32) {
      if (tensor == src_i64) {
        tensor = dst_i32;
        break;
      }
    }
  }
  deferred_i64_to_i32.clear();  // no longer needed

  // Collect ordered input tensor list.
  std::vector<struct ggml_tensor*> ordered_inputs;
  for (auto& [idx, tensor] : input_pairs) {
    ordered_inputs.push_back(tensor);
  }

  // === Phase C: prepare graph for scheduler allocation ===
  auto t_bg_c1 = std::chrono::high_resolution_clock::now();
  // Leaf tensors with data_key already have data/buffer pointers set from
  // const_buf/mutable_buf and are marked as inputs so the scheduler skips them.

  // 1. Save eager constants (leaf tensors with data that aren't in shared
  //    buffers and aren't runtime inputs). These include bool masks, scalar
  //    constants, and other values computed during build_graph.
  std::unordered_set<struct ggml_tensor*> input_set(
      ordered_inputs.begin(), ordered_inputs.end());
  std::unordered_set<struct ggml_tensor*> deferred_dst_set;
  for (auto& [src, dst] : deferred_i64_to_i32) {
    deferred_dst_set.insert(dst);
  }
  std::vector<EagerConstant> eager_constants;
  bool has_input_derived_eager = false;
  for (struct ggml_tensor* gt = ggml_get_first_tensor(ctx);
       gt != nullptr;
       gt = ggml_get_next_tensor(ctx, gt)) {
    if (!gt->data) continue;
    // Skip tensors in shared buffers (const_buf / mutable_buf).
    // Only check when the buffer is actually allocated (non-null).
    if (handle->const_buf && gt->buffer == handle->const_buf) continue;
    if (handle->mutable_buf && gt->buffer == handle->mutable_buf) continue;
    if (input_set.count(gt)) continue;
    if (deferred_dst_set.count(gt)) continue;
    if (gt->op != GGML_OP_NONE) continue;
    ggml_set_input(gt);  // prevent scheduler from reusing this memory
    eager_constants.push_back({gt, gt->data, ggml_nbytes(gt)});
    if (input_derived.count(gt)) has_input_derived_eager = true;
  }

  auto t_bg_c2 = std::chrono::high_resolution_clock::now();
  {
    size_t ec_total_bytes = 0;
    size_t ec_i64_count = 0, ec_i32_count = 0, ec_f16_count = 0, ec_f32_count = 0, ec_other = 0;
    for (auto& ec : eager_constants) {
      ec_total_bytes += ec.nbytes;
      switch (ec.tensor->type) {
        case GGML_TYPE_I64: ec_i64_count++; break;
        case GGML_TYPE_I32: ec_i32_count++; break;
        case GGML_TYPE_F16: ec_f16_count++; break;
        case GGML_TYPE_F32: ec_f32_count++; break;
        default: ec_other++; break;
      }
    }
    if (should_perf_log()) {
      size_t ec_large = 0, ec_large_bytes = 0;
      for (auto& ec : eager_constants) {
        if (ec.nbytes > 1024) {
          ec_large++; ec_large_bytes += ec.nbytes;
          if (ec_large <= 3) {
            fprintf(stderr, "[perf]   large eager: ne=[%lld,%lld,%lld,%lld] type=%d bytes=%zu\n",
                    (long long)ec.tensor->ne[0], (long long)ec.tensor->ne[1],
                    (long long)ec.tensor->ne[2], (long long)ec.tensor->ne[3],
                    (int)ec.tensor->type, ec.nbytes);
          }
        }
      }
      fprintf(stderr, "[perf] eager_constants: count=%zu bytes=%zu (I64=%zu I32=%zu F16=%zu F32=%zu) large(>1KB)=%zu large_bytes=%zu input_derived=%s\n",
              eager_constants.size(), ec_total_bytes,
              ec_i64_count, ec_i32_count, ec_f16_count, ec_f32_count,
              ec_large, ec_large_bytes,
              has_input_derived_eager ? "yes" : "no");
    }
  }
  // 2. Clear data/buffer on non-shared tensors so scheduler can allocate them.
  for (struct ggml_tensor* t = ggml_get_first_tensor(ctx);
       t != nullptr;
       t = ggml_get_next_tensor(ctx, t)) {
    if (handle->const_buf && t->buffer == handle->const_buf) continue;
    if (handle->mutable_buf && t->buffer == handle->mutable_buf) continue;
    t->data   = nullptr;
    t->buffer = nullptr;
  }

  // 3. Mark inputs/outputs for the scheduler.
  for (auto* inp : ordered_inputs) {
    ggml_set_input(inp);
  }
  for (auto* dst : deferred_dst_set) {
    ggml_set_input(dst);
  }
  for (auto* out : output_tensors) {
    ggml_set_output(out);
  }

  auto t_bg_c4 = std::chrono::high_resolution_clock::now();
  // 4. Transfer host_buf ownership to GraphInstance — kept alive so that
  //    eager constant ctx_data pointers (which may point into host_buf)
  //    remain valid until execute() copies them to the GPU buffer.
  if (gi->host_buf) {
    ggml_backend_buffer_free(gi->host_buf);
  }
  gi->host_buf = host_buf;
  host_buf = nullptr;  // prevent RAII guard from freeing it

  auto t_bg_phaseC = std::chrono::high_resolution_clock::now();

  // === Update graph instance ===
  gi->ctx = ctx;
  gi->graph = graph;
  gi->inputs = std::move(ordered_inputs);
  gi->outputs = std::move(output_tensors);
  gi->deferred_i64_to_i32 = std::move(deferred_i64_to_i32);
  gi->eager_constants = std::move(eager_constants);
  gi->has_input_derived_eager = has_input_derived_eager;
  gi->shared_leaves = std::move(shared_leaves);
  gi->cpu_pinned = std::move(cpu_pinned);

  if (should_perf_log()) {
    auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    fprintf(stderr, "[perf] build_graph: phaseA=%.2fms phaseB=%.2fms graph_expand=%.2fms phaseC=%.2fms total=%.2fms tensors=%d nodes=%d\n",
            ms(t_bg_start, t_bg_phaseA), ms(t_bg_phaseA, t_bg_phaseB),
            ms(t_bg_graph_start, t_bg_graph_end),
            ms(t_bg_graph_end, t_bg_phaseC), ms(t_bg_start, t_bg_phaseC), n_tensors,
            ggml_graph_n_nodes(graph));
    fprintf(stderr, "[perf] build_graph phaseC detail: eager_collect=%.2fms clear=%.2fms mark+free=%.2fms\n",
            ms(t_bg_c1, t_bg_c2), ms(t_bg_c2, t_bg_c4), ms(t_bg_c4, t_bg_phaseC));
  }

  return Error::Ok;
}

}  // namespace

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

  auto t_init_start = std::chrono::high_resolution_clock::now();

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

  // --- 3. Create backends + scheduler (one-time) ---
  int cpu_threads = handle->n_threads;
  const char* threads_env = std::getenv("GGML_CPU_THREADS");
  if (threads_env) {
    cpu_threads = std::atoi(threads_env);
  } else if (cpu_threads <= 1) {
    cpu_threads = static_cast<int>(std::thread::hardware_concurrency());
    if (cpu_threads <= 0) cpu_threads = 4;
  }

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

  // Always have a CPU backend for scheduler fallback / custom ops.
  if (handle->backend) {
    handle->backend_cpu = ggml_backend_cpu_init();
    if (!handle->backend_cpu) {
      ggml_backend_free(handle->backend);
      delete handle;
      return Error::MemoryAllocationFailed;
    }
    ggml_backend_cpu_set_n_threads(handle->backend_cpu, cpu_threads);
  } else {
    handle->backend = ggml_backend_cpu_init();
    if (!handle->backend) {
      delete handle;
      return Error::MemoryAllocationFailed;
    }
    handle->backend_cpu = handle->backend;
    ggml_backend_cpu_set_n_threads(handle->backend_cpu, cpu_threads);
    fprintf(stderr, "[ggml_backend] Using CPU backend (%d threads)\n",
            cpu_threads);
  }

  // Schedulers are created per-graph in build_graph() to isolate internal
  // gallocr / split state between graph instances of different shapes.
  auto cleanup_backends = [&]() {
    for (auto& [k, gi] : handle->graph_cache) {
      if (gi && gi->eager_const_buf) {
        ggml_backend_buffer_free(gi->eager_const_buf);
        gi->eager_const_buf = nullptr;
      }
      if (gi && gi->host_buf) {
        ggml_backend_buffer_free(gi->host_buf);
        gi->host_buf = nullptr;
      }
      if (gi && gi->sched) {
        ggml_backend_sched_free(gi->sched);
        gi->sched = nullptr;
      }
      if (gi && gi->ctx) {
        ggml_free(gi->ctx);
        gi->ctx = nullptr;
      }
    }
    handle->graph_cache.clear();
    if (handle->backend_cpu && handle->backend_cpu != handle->backend) {
      ggml_backend_free(handle->backend_cpu);
      handle->backend_cpu = nullptr;
    }
    if (handle->backend) {
      ggml_backend_free(handle->backend);
      handle->backend = nullptr;
    }
  };

  auto t_backends_done = std::chrono::high_resolution_clock::now();

  // --- 4. Load ALL constants from NamedDataMap → handle->constant_data ---
  // This is the ONLY place NamedDataMap is touched.
  // After this, build_graph() reads from handle->constant_data.
  for (int i = 0; i < n_tensors; ++i) {
    const auto* t = fb_tensors->Get(i);
    if (static_cast<ggml_ir::OpCode>(t->op()) != ggml_ir::OpCode::NONE) continue;
    if (!t->data_key() || std::strlen(t->data_key()->c_str()) == 0) continue;

    const auto* ndm = context.get_named_data_map();
    if (ndm == nullptr) {
      cleanup_backends();
      delete handle;
      return Error::InvalidArgument;
    }

    const char* key = t->data_key()->c_str();

    // Compute byte size from IR shape + type.
    // Use ggml_row_size() to correctly handle block-based quantized types
    // (e.g. Q8_0: 34 bytes per block of 32 elements).
    int64_t ne[4] = {1, 1, 1, 1};
    if (t->ne()) {
      for (size_t d = 0; d < t->ne()->size() && d < 4; ++d) {
        ne[d] = t->ne()->Get(d);
      }
    }
    ggml_type load_gtype;
    switch (static_cast<ggml_ir::TensorType>(t->type())) {
      case ggml_ir::TensorType::F16:  load_gtype = GGML_TYPE_F16;  break;
      case ggml_ir::TensorType::BF16: load_gtype = GGML_TYPE_BF16; break;
      case ggml_ir::TensorType::I32:  load_gtype = GGML_TYPE_I32;  break;
      case ggml_ir::TensorType::I64:  load_gtype = GGML_TYPE_I64;  break;
      case ggml_ir::TensorType::BOOL: load_gtype = GGML_TYPE_I32;  break;
      case ggml_ir::TensorType::Q8_0: load_gtype = GGML_TYPE_Q8_0; break;
      case ggml_ir::TensorType::Q6_K: load_gtype = GGML_TYPE_Q6_K; break;
      case ggml_ir::TensorType::Q4_0: load_gtype = GGML_TYPE_Q4_0; break;
      default:                        load_gtype = GGML_TYPE_F32;   break;
    }
    size_t nbytes = ggml_row_size(load_gtype, ne[0]) * ne[1] * ne[2] * ne[3];

    auto fb = ndm->get_data(key);
    if (!fb.ok()) {
      // Not in PTE or GGUF — zero-init.
      SavedConstant sc;
      sc.ir_tensor_id = t->id();
      sc.data.resize(nbytes, 0);
      handle->constant_data.push_back(std::move(sc));
      continue;
    }
    auto buf = std::move(fb.get());
    if (buf.size() < nbytes) {
      fprintf(stderr, "[ggml_backend] ERROR: weight '%s' size mismatch: got %zu, need %zu\n",
              key, buf.size(), nbytes);
      buf.Free();
      cleanup_backends();
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

  auto t_constants_done = std::chrono::high_resolution_clock::now();

  // --- 5. Allocate shared const_buf and mutable_buf ---
  // Scan IR for leaf tensors with data_key. Compute total sizes,
  // record per-tensor {buf, offset, nbytes} in leaf_buf_map.
  {
    const size_t alignment = 64;  // safe alignment for all backends
    size_t const_total = 0, mutable_total = 0;

    // First pass: compute sizes
    for (int i = 0; i < n_tensors; ++i) {
      const auto* t = fb_tensors->Get(i);
      if (static_cast<ggml_ir::OpCode>(t->op()) != ggml_ir::OpCode::NONE) continue;
      if (!t->data_key() || std::strlen(t->data_key()->c_str()) == 0) continue;

      int64_t ne[4] = {1, 1, 1, 1};
      if (t->ne()) {
        for (size_t d = 0; d < t->ne()->size() && d < 4; ++d) {
          ne[d] = t->ne()->Get(d);
        }
      }
      ggml_type gtype;
      switch (static_cast<ggml_ir::TensorType>(t->type())) {
        case ggml_ir::TensorType::F16:  gtype = GGML_TYPE_F16;  break;
        case ggml_ir::TensorType::BF16: gtype = GGML_TYPE_BF16; break;
        case ggml_ir::TensorType::I32:  gtype = GGML_TYPE_I32;  break;
        case ggml_ir::TensorType::I64:  gtype = GGML_TYPE_I64;  break;
        case ggml_ir::TensorType::BOOL: gtype = GGML_TYPE_I32;  break;
        case ggml_ir::TensorType::Q8_0: gtype = GGML_TYPE_Q8_0; break;
        case ggml_ir::TensorType::Q6_K: gtype = GGML_TYPE_Q6_K; break;
        case ggml_ir::TensorType::Q4_0: gtype = GGML_TYPE_Q4_0; break;
        default:                        gtype = GGML_TYPE_F32;   break;
      }
      size_t nbytes = ggml_row_size(gtype, ne[0]) * ne[1] * ne[2] * ne[3];
      size_t padded = ((nbytes + alignment - 1) / alignment) * alignment;

      if (t->is_mutable()) {
        mutable_total += padded;
      } else {
        const_total += padded;
      }
    }

    // Add safety margin
    if (const_total > 0) const_total += alignment;
    if (mutable_total > 0) mutable_total += alignment;

    // Allocate backend buffers
    if (const_total > 0) {
      handle->const_buf = ggml_backend_alloc_buffer(handle->backend, const_total);
      if (!handle->const_buf) {
        fprintf(stderr, "[ggml_backend] ERROR: failed to allocate const_buf (%zu bytes)\n", const_total);
        cleanup_backends();
        delete handle;
        return Error::MemoryAllocationFailed;
      }
      fprintf(stderr, "[ggml_backend] const_buf allocated: %zu MB\n", const_total / (1024*1024));
    }
    if (mutable_total > 0) {
      // Allocate KV cache on the primary backend (GPU) so ggml_set_rows
      // updates run on-device without graph splits.
      // On Metal/Apple Silicon the CPU buffer type IS the unified GPU
      // buffer, so using the primary backend is correct everywhere.
      handle->mutable_buf = ggml_backend_alloc_buffer(handle->backend, mutable_total);
      if (!handle->mutable_buf) {
        fprintf(stderr, "[ggml_backend] ERROR: failed to allocate mutable_buf (%zu bytes)\n", mutable_total);
        if (handle->const_buf) ggml_backend_buffer_free(handle->const_buf);
        cleanup_backends();
        delete handle;
        return Error::MemoryAllocationFailed;
      }
      fprintf(stderr, "[ggml_backend] mutable_buf allocated: %zu MB\n", mutable_total / (1024*1024));

      // Zero-initialize the entire mutable buffer once upfront.
      // This ensures all KV cache indices and other mutable tensors start
      // at 0 regardless of whether they have explicit constant_data.
      ggml_backend_buffer_clear(handle->mutable_buf, 0);
    }

    // Second pass: record offsets and copy data
    size_t const_offset = 0, mutable_offset = 0;
    // Build ir_tensor_id → constant_data index map for efficient lookup
    std::unordered_map<int, size_t> id_to_cd;
    for (size_t ci = 0; ci < handle->constant_data.size(); ++ci) {
      id_to_cd[handle->constant_data[ci].ir_tensor_id] = ci;
    }

    bool const_is_host = !handle->const_buf || ggml_backend_buffer_is_host(handle->const_buf);
    bool mutable_is_host = !handle->mutable_buf || ggml_backend_buffer_is_host(handle->mutable_buf);

    // For non-host buffers (e.g. CUDA), create a temporary ggml_tensor for
    // ggml_backend_tensor_set() which dispatches host→device copies.
    struct ggml_init_params tmp_params = { ggml_tensor_overhead() + 64, nullptr, true };
    struct ggml_context* tmp_ctx = (!const_is_host || !mutable_is_host) ? ggml_init(tmp_params) : nullptr;
    struct ggml_tensor* tmp_tensor = tmp_ctx ? ggml_new_tensor_1d(tmp_ctx, GGML_TYPE_I8, 1) : nullptr;

    for (int i = 0; i < n_tensors; ++i) {
      const auto* t = fb_tensors->Get(i);
      if (static_cast<ggml_ir::OpCode>(t->op()) != ggml_ir::OpCode::NONE) continue;
      if (!t->data_key() || std::strlen(t->data_key()->c_str()) == 0) continue;

      int64_t ne[4] = {1, 1, 1, 1};
      if (t->ne()) {
        for (size_t d = 0; d < t->ne()->size() && d < 4; ++d) {
          ne[d] = t->ne()->Get(d);
        }
      }
      ggml_type gtype;
      switch (static_cast<ggml_ir::TensorType>(t->type())) {
        case ggml_ir::TensorType::F16:  gtype = GGML_TYPE_F16;  break;
        case ggml_ir::TensorType::BF16: gtype = GGML_TYPE_BF16; break;
        case ggml_ir::TensorType::I32:  gtype = GGML_TYPE_I32;  break;
        case ggml_ir::TensorType::I64:  gtype = GGML_TYPE_I64;  break;
        case ggml_ir::TensorType::BOOL: gtype = GGML_TYPE_I32;  break;
        case ggml_ir::TensorType::Q8_0: gtype = GGML_TYPE_Q8_0; break;
        case ggml_ir::TensorType::Q6_K: gtype = GGML_TYPE_Q6_K; break;
        case ggml_ir::TensorType::Q4_0: gtype = GGML_TYPE_Q4_0; break;
        default:                        gtype = GGML_TYPE_F32;   break;
      }
      size_t nbytes = ggml_row_size(gtype, ne[0]) * ne[1] * ne[2] * ne[3];
      size_t padded = ((nbytes + alignment - 1) / alignment) * alignment;

      bool is_mut = t->is_mutable();
      ggml_backend_buffer_t buf = is_mut ? handle->mutable_buf : handle->const_buf;
      size_t& offset = is_mut ? mutable_offset : const_offset;
      bool is_host = is_mut ? mutable_is_host : const_is_host;

      handle->leaf_buf_map[t->id()] = {buf, offset, nbytes};

      // Copy data into the buffer
      auto cd_it = id_to_cd.find(t->id());
      if (cd_it != id_to_cd.end()) {
        const auto& cd = handle->constant_data[cd_it->second];
        size_t copy_size = std::min(nbytes, cd.data.size());
        if (is_host) {
          char* dst = static_cast<char*>(ggml_backend_buffer_get_base(buf)) + offset;
          memcpy(dst, cd.data.data(), copy_size);
        } else {
          tmp_tensor->ne[0] = nbytes;
          tmp_tensor->nb[1] = nbytes;
          tmp_tensor->buffer = buf;
          tmp_tensor->data = static_cast<char*>(ggml_backend_buffer_get_base(buf)) + offset;
          ggml_backend_tensor_set(tmp_tensor, cd.data.data(), 0, copy_size);
        }
      } else if (is_mut) {
        // Mutable tensors without constant data: already zero-initialized
        // by the upfront ggml_backend_buffer_clear() call above.
        // For host buffers, zero just this region (buffer may not have been cleared).
        if (is_host) {
          char* dst = static_cast<char*>(ggml_backend_buffer_get_base(buf)) + offset;
          memset(dst, 0, nbytes);
        }
      }

      offset += padded;
    }
    if (tmp_ctx) ggml_free(tmp_ctx);

    fprintf(stderr, "[ggml_backend] leaf_buf_map: %zu entries (const=%zu MB, mutable=%zu MB)\n",
            handle->leaf_buf_map.size(), const_offset / (1024*1024), mutable_offset / (1024*1024));
  }

  auto t_buffers_done = std::chrono::high_resolution_clock::now();

  // --- 6. Read sym_dim_ids from IR ---
  // Scan ALL tensors (not just inputs) for sym_dim_ids to detect dynamic models.
  int sym_tensor_count = 0;
  for (int i = 0; i < n_tensors; ++i) {
    const auto* t = fb_tensors->Get(i);
    if (t->sym_dim_ids()) {
      for (size_t d = 0; d < t->sym_dim_ids()->size() && d < 4; ++d) {
        int32_t sid = t->sym_dim_ids()->Get(d);
        if (sid >= 0 || sid == -2) {
          handle->has_dynamic = true;
          sym_tensor_count++;
          break;
        }
      }
    }
    if (!t->is_input()) continue;

    std::vector<int32_t> sids(4, -1);
    if (t->sym_dim_ids()) {
      for (size_t d = 0; d < t->sym_dim_ids()->size() && d < 4; ++d) {
        sids[d] = t->sym_dim_ids()->Get(d);
      }
    }
    // Store in input_index order (may be sparse; resize as needed).
    int idx = t->input_index();
    if (idx >= 0 && static_cast<size_t>(idx) >= handle->input_sym_dim_ids.size()) {
      handle->input_sym_dim_ids.resize(idx + 1);
    }
    if (idx >= 0) {
      handle->input_sym_dim_ids[idx] = std::move(sids);
    }
  }

  fprintf(stderr, "[ggml_backend] has_dynamic=%d, sym_tensor_count=%d, n_tensors=%d\n",
          handle->has_dynamic, sym_tensor_count, n_tensors);

  // --- 7. Build initial graph with serialized shapes ---
  {
    auto init_gi = std::make_unique<GraphInstance>();
    Error err = build_graph(handle, init_gi.get(), nullptr, 0);
    if (err != Error::Ok) {
      cleanup_backends();
      delete handle;
      return err;
    }
    // Record initial input shapes for graph cache keying.
    if (handle->has_dynamic) {
      size_t n_inp = init_gi->inputs.size();
      handle->init_ne.resize(n_inp * 4, 1);
      for (size_t i = 0; i < n_inp; ++i) {
        for (int d = 0; d < 4; ++d) {
          handle->init_ne[i * 4 + d] = init_gi->inputs[i]->ne[d];
        }
      }
    }
    size_t key = hash_shape_key(handle->init_ne);
    handle->active = init_gi.get();
    handle->graph_cache[key] = std::move(init_gi);
  }

  {
    auto t_init_end = std::chrono::high_resolution_clock::now();
    auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    fprintf(stderr, "[perf] init: backends=%.1fms constants=%.1fms buffers=%.1fms graph=%.1fms total=%.1fms\n",
            ms(t_init_start, t_backends_done),
            ms(t_backends_done, t_constants_done),
            ms(t_constants_done, t_buffers_done),
            ms(t_buffers_done, t_init_end),
            ms(t_init_start, t_init_end));
  }

  return handle;
}

Error GgmlBackendInterface::execute(
    BackendExecutionContext& context,
    DelegateHandle* handle_raw,
    Span<EValue*> args) const {

  auto* handle = reinterpret_cast<GgmlDelegateHandle*>(handle_raw);
  GraphInstance* active = handle->active;

  size_t n_inputs = active->inputs.size();
  size_t n_outputs = active->outputs.size();
  size_t n_non_output_args = args.size() >= n_outputs ? args.size() - n_outputs : 0;

  // --- Gather input shapes + data for build_graph ---
  std::vector<int64_t> current_ne;
  std::vector<InputDataOverride> input_data_vec(n_inputs);
  if (handle->has_dynamic) {
    current_ne.resize(n_inputs * 4, 1);
  }
  {
    size_t ggml_idx = 0;
    for (size_t a = 0; a < n_non_output_args && ggml_idx < n_inputs; ++a) {
      if (!args[a]->isTensor()) continue;
      const auto& et = args[a]->toTensor();
      if (handle->has_dynamic) {
        int ndim = et.dim();
        for (int d = 0; d < ndim && d < 4; ++d) {
          current_ne[ggml_idx * 4 + d] = et.size(ndim - 1 - d);
        }
      }
      input_data_vec[ggml_idx] = {
        et.const_data_ptr(),
        static_cast<size_t>(et.nbytes()),
        static_cast<int>(et.scalar_type())
      };
      ++ggml_idx;
    }
  }

  // --- Graph instance selection (dynamic shapes) ---
  if (handle->has_dynamic) {
    size_t shape_key = hash_shape_key(current_ne);
    GraphInstance* prev_active = handle->active;
    auto it = handle->graph_cache.find(shape_key);
    if (it != handle->graph_cache.end()) {
      active = it->second.get();
    } else {
      // New shape — create a GraphInstance (sched will be created by build_graph).
      auto new_gi = std::make_unique<GraphInstance>();
      active = new_gi.get();
      handle->graph_cache[shape_key] = std::move(new_gi);
    }
    (void)prev_active;
    handle->active = active;
  }

  // --- Build or reuse graph ---
  auto t_start = std::chrono::high_resolution_clock::now();
  bool no_cache = is_graph_cache_disabled();
  // Force rebuild if this graph has input-derived eager constants (their
  // values change per call and can't be refreshed without rebuilding).
  bool need_build = (active->ctx == nullptr) || no_cache
                    || active->has_input_derived_eager;

  if (need_build) {
    // First time for this shape: full IR parse + tensor creation.
    const int64_t* ne_ptr = handle->has_dynamic ? current_ne.data() : nullptr;
    size_t ne_count = handle->has_dynamic ? current_ne.size() : 0;
    Error err = build_graph(handle, active, ne_ptr, ne_count, &input_data_vec,
                            /*verbose=*/false);
    if (err != Error::Ok) return err;
  }
  auto t_build = std::chrono::high_resolution_clock::now();
  // Update counts after build (or use existing from cache).
  n_inputs = active->inputs.size();
  n_outputs = active->outputs.size();
  n_non_output_args = args.size() >= n_outputs ? args.size() - n_outputs : 0;

  // --- Scheduler setup ---
  // MISS path: build_graph cleared tensor data → full reset + restore + alloc.
  // HIT (first time): scheduler not yet allocated → same full cycle.
  // HIT (subsequent): scheduler already allocated, tensor buffers valid from
  //   previous alloc → skip everything (like llama.cpp's graph reuse path).
  auto t_pre_alloc = t_build;
  auto t_alloc = t_build;
  bool need_alloc = need_build || !active->is_allocated;
  if (need_alloc) {
    if (!active->sched) {
      return Error::InvalidState;
    }

    ggml_backend_sched_reset(active->sched);

    // Re-apply CPU pin assignments for custom ops (reset clears them).
    for (auto* t : active->cpu_pinned) {
      if (handle->backend_cpu && handle->backend_cpu != handle->backend) {
        ggml_backend_sched_set_tensor_backend(active->sched, t, handle->backend_cpu);
      }
    }

    // Restore shared-buffer leaf tensors (const_buf / mutable_buf) BEFORE
    // sched_alloc so the scheduler sees them as pre-allocated and skips them.
    for (auto& sl : active->shared_leaves) {
      sl.tensor->data = static_cast<char*>(ggml_backend_buffer_get_base(sl.buf)) + sl.offset;
      sl.tensor->buffer = sl.buf;
    }

    // Restore eager constants into a dedicated buffer BEFORE sched_alloc.
    if (!active->eager_constants.empty()) {
      const size_t alignment = 64;
      size_t total = 0;
      for (auto& ec : active->eager_constants) {
        total += ((ec.nbytes + alignment - 1) / alignment) * alignment;
      }
      if (!active->eager_const_buf) {
        active->eager_const_buf = ggml_backend_alloc_buffer(handle->backend, total);
      }
      if (active->eager_const_buf) {
        char* base = static_cast<char*>(ggml_backend_buffer_get_base(active->eager_const_buf));
        size_t offset = 0;
        for (auto& ec : active->eager_constants) {
          size_t nbytes = ec.nbytes;
          ec.tensor->data = base + offset;
          ec.tensor->buffer = active->eager_const_buf;
          offset += ((nbytes + alignment - 1) / alignment) * alignment;
        }
      }
    }

    // Force GPU backend for the first few ops that touch graph inputs.
    // Without this, the scheduler places them on CPU (since inputs start on
    // CPU host memory), creating a graph split that prevents CUDA graphs.
    if (handle->backend && handle->backend != handle->backend_cpu) {
      std::unordered_set<struct ggml_tensor*> pinned_set(
          active->cpu_pinned.begin(), active->cpu_pinned.end());
      for (int i = 0; i < ggml_graph_n_nodes(active->graph); i++) {
        struct ggml_tensor* node = active->graph->nodes[i];
        // Skip nodes that build_graph explicitly pinned to CPU.
        if (pinned_set.count(node)) continue;
        // Check if any source is an input leaf (on CPU / no buffer).
        bool has_input_leaf = false;
        for (int j = 0; j < GGML_MAX_SRC; j++) {
          auto* s = node->src[j];
          if (s && (s->flags & GGML_TENSOR_FLAG_INPUT) && !ggml_is_view(node)) {
            has_input_leaf = true;
            break;
          }
        }
        if (has_input_leaf) {
          ggml_backend_sched_set_tensor_backend(active->sched, node, handle->backend);
        }
      }
    }

    t_pre_alloc = std::chrono::high_resolution_clock::now();
    if (!ggml_backend_sched_alloc_graph(active->sched, active->graph)) {
      fprintf(stderr, "[ggml_backend] ERROR: scheduler alloc failed\n");
      return Error::MemoryAllocationFailed;
    }
    t_alloc = std::chrono::high_resolution_clock::now();

    // Copy eager constant data after sched_alloc (buffer is already assigned).
    // On the first alloc, upload everything. On subsequent calls, the
    // eager_const_buf already has data from the previous call. Skip
    // constants whose data hasn't changed (shape-dependent masks stay
    // the same; data-dependent scalars/indices get re-uploaded).
    {
      int ec_skipped = 0, ec_uploaded = 0;
      size_t ec_uploaded_bytes = 0;
      for (auto& ec : active->eager_constants) {
        if (active->is_allocated && ec.tensor->buffer == active->eager_const_buf
            && ec.nbytes > 64) {
          // Large constant: check 64-byte prefix to skip if unchanged.
          char sample[64];
          ggml_backend_tensor_get(ec.tensor, sample, 0, 64);
          if (memcmp(sample, ec.ctx_data, 64) == 0) { ec_skipped++; continue; }
        } else if (active->is_allocated && ec.nbytes <= 4) {
          // Small constant: check exact match to skip if unchanged
          char old_val[4] = {0};
          ggml_backend_tensor_get(ec.tensor, old_val, 0, ec.nbytes);
          if (memcmp(old_val, ec.ctx_data, ec.nbytes) == 0) { ec_skipped++; continue; }
        }
        if (ec.tensor->buffer) {
          ggml_backend_tensor_set(ec.tensor, ec.ctx_data, 0, ec.nbytes);
        } else if (ec.tensor->data) {
          memcpy(ec.tensor->data, ec.ctx_data, ec.nbytes);
        }
        ec_uploaded++;
        ec_uploaded_bytes += ec.nbytes;
      }
      if (should_perf_log()) {
        fprintf(stderr, "[perf] eager upload: skipped=%d uploaded=%d (%zu bytes)\n",
                ec_skipped, ec_uploaded, ec_uploaded_bytes);
      }
    }

    active->is_allocated = true;
  }

  // --- Copy input data from ExecuTorch tensors → backend tensors ---
  {
    size_t ggml_idx = 0;
    for (size_t a = 0; a < n_non_output_args && ggml_idx < n_inputs; ++a) {
      if (!args[a]->isTensor()) continue;

      struct ggml_tensor* gt = active->inputs[ggml_idx];
      ++ggml_idx;

      if (gt->buffer == nullptr) continue;

      const auto& et_tensor = args[a]->toTensor();
      const size_t nelem = ggml_nelements(gt);

      if (gt->type == GGML_TYPE_F16 && et_tensor.scalar_type() == executorch::aten::ScalarType::Float) {
        std::vector<ggml_fp16_t> tmp(nelem);
        const float* src = static_cast<const float*>(et_tensor.const_data_ptr());
        ggml_fp32_to_fp16_row(src, tmp.data(), (int64_t) nelem);
        ggml_backend_tensor_set(gt, tmp.data(), 0, nelem * sizeof(ggml_fp16_t));
      } else if (gt->type == GGML_TYPE_BF16 && et_tensor.scalar_type() == executorch::aten::ScalarType::Float) {
        std::vector<ggml_bf16_t> tmp(nelem);
        const float* src = static_cast<const float*>(et_tensor.const_data_ptr());
        for (size_t j = 0; j < nelem; ++j) {
          tmp[j] = ggml_fp32_to_bf16(src[j]);
        }
        ggml_backend_tensor_set(gt, tmp.data(), 0, nelem * sizeof(ggml_bf16_t));
      } else if (gt->type == GGML_TYPE_BF16 && et_tensor.scalar_type() == executorch::aten::ScalarType::BFloat16) {
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
  }
  auto t_input_copy = std::chrono::high_resolution_clock::now();

  // --- Deferred I64→I32 casts ---
  for (const auto& [src_i64, dst_i32] : active->deferred_i64_to_i32) {
    if (src_i64->buffer == nullptr || dst_i32->buffer == nullptr) continue;
    const size_t nelem = ggml_nelements(src_i64);
    std::vector<int64_t> src_buf(nelem);
    ggml_backend_tensor_get(src_i64, src_buf.data(), 0, nelem * sizeof(int64_t));
    std::vector<int32_t> dst_buf(nelem);

    // Debug: Log I64→I32 conversion details
    fprintf(stderr, "[I64_TO_I32_DEBUG] Converting %zu elements\n", nelem);
    for (size_t j = 0; j < nelem; ++j) {
      int64_t src_val = src_buf[j];
      int32_t dst_val = static_cast<int32_t>(src_val);
      dst_buf[j] = dst_val;

      // Log suspicious conversions
      if (src_val != (int64_t)dst_val || src_val < INT32_MIN || src_val > INT32_MAX) {
        fprintf(stderr, "[I64_TO_I32_DEBUG] SUSPICIOUS: src_i64=%ld -> dst_i32=%d (element %zu)\n",
                src_val, dst_val, j);
      } else if (j < 4) {  // Log first few for normal cases
        fprintf(stderr, "[I64_TO_I32_DEBUG] Normal: src_i64=%ld -> dst_i32=%d (element %zu)\n",
                src_val, dst_val, j);
      }
    }
    ggml_backend_tensor_set(dst_i32, dst_buf.data(), 0, nelem * sizeof(int32_t));
  }

  // Per-op profiling (GGML_PROFILE=1)
  static int profile_mode = -1;
  if (profile_mode < 0) {
    const char* env = std::getenv("GGML_PROFILE");
    profile_mode = (env && std::string(env) != "0") ? 1 : 0;
  }
  ProfileContext prof_ctx;
  if (profile_mode) {
    ggml_backend_sched_set_eval_callback(active->sched, profile_eval_callback, &prof_ctx);
  }

  // Note: SiLU-gate fusion is now done AOT in build_graph (Phase B MUL handler)
  // via ggml_map_custom2, not via eval callback. No runtime setup needed.

  // Debug tensor dump (GGML_DEBUG_DUMP=<path>)
  DebugDumpContext debug_ctx;
  {
    const char* dump_path = std::getenv("GGML_DEBUG_DUMP");
    if (dump_path && std::strlen(dump_path) > 0) {
      debug_ctx.fp = fopen(dump_path, "w");
      if (debug_ctx.fp) {
        fprintf(debug_ctx.fp, "# nodes=%d\n", ggml_graph_n_nodes(active->graph));
        ggml_backend_sched_set_eval_callback(active->sched, debug_dump_eval_callback, &debug_ctx);
      }
    }
  }

  auto t_pre_compute = std::chrono::high_resolution_clock::now();
  // Use async compute + explicit sync. This avoids the double-sync that
  // graph_compute does (sync inside compute + sync for output copy).
  // The output copy (ggml_backend_tensor_get) will sync implicitly.
  enum ggml_status status = ggml_backend_sched_graph_compute_async(active->sched, active->graph);
  auto t_compute = std::chrono::high_resolution_clock::now();

  if (debug_ctx.fp) {
    fclose(debug_ctx.fp);
    debug_ctx.fp = nullptr;
    ggml_backend_sched_set_eval_callback(active->sched, nullptr, nullptr);
  }
  if (profile_mode) {
    ggml_backend_sched_set_eval_callback(active->sched, nullptr, nullptr);
    print_profile(prof_ctx);
  }

  if (status != GGML_STATUS_SUCCESS) {
    fprintf(stderr, "[ggml_backend] ERROR: scheduler graph compute failed: %s (%d)\n",
            ggml_status_to_string(status), (int)status);
    return Error::InvalidState;
  }

  // GGML_SKIP_OUTPUT_COPY: skip GPU→CPU copy for output tensors.
  // Instead, point the ET tensor's data pointer directly at the GPU buffer.
  // The caller must handle GPU data (e.g., use cuda_argmax_f32).
  // Off by default — generic callers (Python, etc.) expect CPU-accessible data.
  // The C++ benchmark enables this automatically via its own flag.
  static int skip_output_copy = -1;
  if (skip_output_copy < 0) {
    const char* env = std::getenv("GGML_SKIP_OUTPUT_COPY");
    skip_output_copy = (env && std::string(env) != "0") ? 1 : 0;
  }

  // Sync before output access (async compute may not have finished).
  // CUDA with skip_copy: skip sync — cuda_argmax_f32 syncs via cudaMemcpy.
  // Metal/CPU: always sync — no implicit sync from unified memory reads.
#ifdef GGML_FUSED_KERNELS
  if (!skip_output_copy) {
    ggml_backend_sched_synchronize(active->sched);
  }
#else
  ggml_backend_sched_synchronize(active->sched);
#endif

  if (skip_output_copy) {
    for (size_t i = 0; i < n_outputs; ++i) {
      size_t out_idx = args.size() - n_outputs + i;
      if (out_idx >= (size_t)args.size() || !args[out_idx]->isTensor()) continue;
      auto& et_tensor = args[out_idx]->toTensor();
      struct ggml_tensor* gt = active->outputs[i];
      // Resize ET tensor to match ggml shape
      int et_ndim = et_tensor.dim();
      std::vector<executorch::aten::SizesType> new_sizes(et_ndim);
      for (int d = 0; d < et_ndim; ++d) {
        new_sizes[d] = static_cast<executorch::aten::SizesType>(gt->ne[et_ndim - 1 - d]);
      }
      (void)executorch::ET_RUNTIME_NAMESPACE::resize_tensor(
          et_tensor, {new_sizes.data(), new_sizes.size()});
      // Point ET tensor data directly at GPU buffer (zero-copy).
      // WARNING: this data lives on GPU. The caller must not dereference
      // it on CPU without copying first.
      et_tensor.unsafeGetTensorImpl()->set_data(gt->data);
    }
    auto t_output_copy = std::chrono::high_resolution_clock::now();
    // Skip to timing/return section
    goto execute_done;
  }

  // Copy output data from backend tensors → ExecuTorch output tensors.
  for (size_t i = 0; i < n_outputs; ++i) {
    size_t out_idx = args.size() - n_outputs + i;
    if (out_idx >= (size_t)args.size() || !args[out_idx]->isTensor()) {
      fprintf(stderr, "[ggml_backend] ERROR: output args[%zu] is not a Tensor "
              "(n_outputs=%zu, args.size=%zu)\n",
              out_idx, n_outputs, (size_t)args.size());
      return Error::InvalidArgument;
    }
    auto& et_tensor = args[out_idx]->toTensor();
    struct ggml_tensor* gt = active->outputs[i];

    // Resize ET output tensor to match actual ggml output shape.
    {
      int et_ndim = et_tensor.dim();
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
      std::vector<ggml_fp16_t> tmp(nelem);
      ggml_backend_tensor_get(gt, tmp.data(), 0, nelem * sizeof(ggml_fp16_t));
      float* dst = static_cast<float*>(et_tensor.mutable_data_ptr());
      ggml_fp16_to_fp32_row(tmp.data(), dst, (int64_t) nelem);
    } else if (gt->type == GGML_TYPE_F32 && et_tensor.scalar_type() == executorch::aten::ScalarType::BFloat16) {
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
    } else if (gt->type == GGML_TYPE_I32 && et_tensor.scalar_type() == executorch::aten::ScalarType::Long) {
      std::vector<int32_t> tmp(nelem);
      ggml_backend_tensor_get(gt, tmp.data(), 0, nelem * sizeof(int32_t));
      int64_t* dst = static_cast<int64_t*>(et_tensor.mutable_data_ptr());
      for (size_t j = 0; j < nelem; ++j) {
        dst[j] = static_cast<int64_t>(tmp[j]);
      }
    } else if (gt->type == GGML_TYPE_I64 && et_tensor.scalar_type() == executorch::aten::ScalarType::Long) {
      ggml_backend_tensor_get(gt, et_tensor.mutable_data_ptr(), 0, ggml_nbytes(gt));
    } else if (gt->type == GGML_TYPE_F32 && et_tensor.scalar_type() == executorch::aten::ScalarType::Long) {
      std::vector<float> tmp(nelem);
      ggml_backend_tensor_get(gt, tmp.data(), 0, nelem * sizeof(float));
      int64_t* dst = static_cast<int64_t*>(et_tensor.mutable_data_ptr());
      for (size_t j = 0; j < nelem; ++j) {
        dst[j] = static_cast<int64_t>(tmp[j]);
      }
    } else {
      size_t copy_size = std::min(ggml_nbytes(gt), (size_t)et_tensor.nbytes());
      ggml_backend_tensor_get(gt, et_tensor.mutable_data_ptr(), 0, copy_size);
    }
  }
execute_done:
  auto t_output_copy = std::chrono::high_resolution_clock::now();
  // --- Performance logging ---
  {
    auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    static int call_count = 0;
    bool perf = should_perf_log();
    // GGML_PERF_LOG: first 5 calls + every 100th; legacy: calls 1-3
    bool should_print = perf
        ? (call_count < 5 || call_count % 100 == 0 || need_alloc)
        : (call_count > 0 && call_count <= 3);
    if (should_print) {
      int n_splits = ggml_backend_sched_get_n_splits(active->sched);
      if (perf) {
        fprintf(stderr, "[perf] execute #%d: build=%.2fms alloc=%.2fms input=%.2fms compute=%.2fms output=%.2fms total=%.2fms | %s nodes=%d splits=%d\n",
                call_count,
                ms(t_start, t_build),
                ms(t_pre_alloc, t_alloc),
                ms(t_build, t_input_copy) - ms(t_pre_alloc, t_alloc),
                ms(t_pre_compute, t_compute),
                ms(t_compute, t_output_copy),
                ms(t_start, t_output_copy),
                need_alloc ? (need_build ? "MISS(build+alloc)" : "MISS(alloc)") : "HIT",
                ggml_graph_n_nodes(active->graph), n_splits);
      } else {
        fprintf(stderr, "[timing] build=%.1fms alloc=%.1fms compute=%.1fms total=%.1fms splits=%d nodes=%d\n",
                ms(t_start, t_build), ms(t_pre_alloc, t_alloc),
                ms(t_pre_compute, t_compute), ms(t_start, t_compute),
                n_splits, ggml_graph_n_nodes(active->graph));
      }
    }
    call_count++;
  }

  return Error::Ok;
}

void GgmlBackendInterface::destroy(DelegateHandle* handle_raw) const {
  auto* handle = reinterpret_cast<GgmlDelegateHandle*>(handle_raw);
  if (handle) {
    // Free all cached graph instances (schedulers + contexts) before backends.
    for (auto& [k, gi] : handle->graph_cache) {
      if (gi && gi->eager_const_buf) {
        ggml_backend_buffer_free(gi->eager_const_buf);
      }
      if (gi && gi->host_buf) {
        ggml_backend_buffer_free(gi->host_buf);
      }
      if (gi && gi->sched) {
        ggml_backend_sched_free(gi->sched);
      }
      if (gi && gi->ctx) {
        ggml_free(gi->ctx);
      }
    }
    handle->graph_cache.clear();
    // Free shared buffers before backends (buffers may reference backend).
    if (handle->const_buf) {
      ggml_backend_buffer_free(handle->const_buf);
    }
    if (handle->mutable_buf) {
      ggml_backend_buffer_free(handle->mutable_buf);
    }
    if (handle->backend_cpu && handle->backend_cpu != handle->backend) {
      ggml_backend_free(handle->backend_cpu);
    }
    if (handle->backend) {
      ggml_backend_free(handle->backend);
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
