#pragma once

#include "build_context.h"

namespace executorch_ggml {

// ---------------------------------------------------------------------------
// TOPK (values) — returns the top-K values along a dimension
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_topk(BuildContext& bc) {
  struct ggml_tensor* src = bc.srcs[0];
  int32_t k = 1, dim = -1, largest = 1;
  if (bc.ir_tensor->op_params() && bc.ir_tensor->op_params()->size() >= 12) {
    const uint8_t* p = bc.ir_tensor->op_params()->data();
    memcpy(&k, p, 4);
    memcpy(&dim, p + 4, 4);
    memcpy(&largest, p + 8, 4);
  }
  // ggml_top_k returns top-k values per row (ne[0] dimension)
  // If dim != last dim, we'd need transpose. For MoE routing, dim=-1 is typical.
  auto* top_vals = ggml_top_k(bc.ctx, src, k);
  return top_vals;
}

// ---------------------------------------------------------------------------
// TOPK_INDICES — returns the top-K indices along a dimension
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_topk_indices(BuildContext& bc) {
  struct ggml_tensor* src = bc.srcs[0];
  int32_t k = 1, dim = -1, largest = 1;
  if (bc.ir_tensor->op_params() && bc.ir_tensor->op_params()->size() >= 12) {
    const uint8_t* p = bc.ir_tensor->op_params()->data();
    memcpy(&k, p, 4);
    memcpy(&dim, p + 4, 4);
    memcpy(&largest, p + 8, 4);
  }
  // ggml_argsort_top_k returns indices of top-k elements
  auto* indices = ggml_argsort_top_k(bc.ctx, src, k);
  return indices;
}

// ---------------------------------------------------------------------------
// SORT (values) — returns sorted values
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_sort(BuildContext& bc) {
  struct ggml_tensor* src = bc.srcs[0];
  int32_t dim = -1, descending = 0;
  if (bc.ir_tensor->op_params() && bc.ir_tensor->op_params()->size() >= 8) {
    const uint8_t* p = bc.ir_tensor->op_params()->data();
    memcpy(&dim, p, 4);
    memcpy(&descending, p + 4, 4);
  }
  enum ggml_sort_order order = descending ? GGML_SORT_ORDER_DESC : GGML_SORT_ORDER_ASC;
  // argsort + get_rows to produce sorted values
  auto* indices = ggml_argsort(bc.ctx, src, order);
  // Gather sorted values using indices
  auto* sorted = ggml_get_rows(bc.ctx, src, indices);
  return sorted;
}

// ---------------------------------------------------------------------------
// SORT_INDICES — returns sort permutation indices
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_sort_indices(BuildContext& bc) {
  struct ggml_tensor* src = bc.srcs[0];
  int32_t dim = -1, descending = 0;
  if (bc.ir_tensor->op_params() && bc.ir_tensor->op_params()->size() >= 8) {
    const uint8_t* p = bc.ir_tensor->op_params()->data();
    memcpy(&dim, p, 4);
    memcpy(&descending, p + 4, 4);
  }
  enum ggml_sort_order order = descending ? GGML_SORT_ORDER_DESC : GGML_SORT_ORDER_ASC;
  return ggml_argsort(bc.ctx, src, order);
}

// ---------------------------------------------------------------------------
// MUL_MAT_ID — expert-indexed matrix multiplication (MoE)
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_mul_mat_id(BuildContext& bc) {
  // src_ids: [input, weight, indices]
  // input: [hidden, n_tokens]  (or batched)
  // weight: [out_dim, hidden, n_experts] (stacked expert weights)
  // indices: [n_experts_per_tok, n_tokens] (which experts to use)
  struct ggml_tensor* input = bc.srcs[0];
  struct ggml_tensor* weight = bc.srcs[1];
  struct ggml_tensor* indices = bc.srcs[2];

  // ggml_mul_mat_id(experts, input, ids)
  // experts: [out_dim, hidden, n_experts]
  // input: [hidden, n_tokens]
  // ids: [n_experts_per_tok, n_tokens] (I32)
  if (indices->type != GGML_TYPE_I32) {
    indices = safe_ggml_cast(bc.ctx, indices, GGML_TYPE_I32, &bc.host_acc);
  }
  return ggml_mul_mat_id(bc.ctx, weight, input, indices);
}

// ---------------------------------------------------------------------------
// LOG1P — log(1 + x)
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_log1p(BuildContext& bc) {
  struct ggml_tensor* src = bc.srcs[0];
  if (src->type != GGML_TYPE_F32) {
    src = safe_ggml_cast(bc.ctx, src, GGML_TYPE_F32, &bc.host_acc);
  }
  // log(1 + x) = log(add(x, 1.0))
  auto* one_plus_x = ggml_add(bc.ctx, src, make_f32_scalar(bc.ctx, 1.0f));
  return ggml_log(bc.ctx, one_plus_x);
}

} // namespace executorch_ggml
