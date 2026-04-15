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

// ---------------------------------------------------------------------------
// EXP — element-wise exponential
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_exp(BuildContext& bc) {
  struct ggml_tensor* src = bc.srcs[0];
  if (src->type != GGML_TYPE_F32) {
    src = safe_ggml_cast(bc.ctx, src, GGML_TYPE_F32, &bc.host_acc);
  }
  return ggml_exp(bc.ctx, src);
}

// ---------------------------------------------------------------------------
// SUM — reduce sum along dimension
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_sum(BuildContext& bc) {
  struct ggml_tensor* src = bc.srcs[0];
  int32_t dim = 0, ndim = 4;
  if (bc.ir_tensor->op_params() && bc.ir_tensor->op_params()->size() >= 8) {
    const uint8_t* p = bc.ir_tensor->op_params()->data();
    memcpy(&dim, p, 4);
    memcpy(&ndim, p + 4, 4);
  }
  // Convert PyTorch dim to ggml axis
  int ggml_axis = (ndim - 1) - dim;
  if (ggml_axis < 0) ggml_axis = 0;
  // ggml_sum_rows sums along ne[0] (axis 0)
  // For other axes, permute to put target axis at ne[0], sum, permute back
  if (ggml_axis == 0) {
    return ggml_sum_rows(bc.ctx, src);
  }
  // For simplicity, use ggml_repeat + ggml_sum for non-0 axes
  // TODO: implement proper axis reduction
  return ggml_sum_rows(bc.ctx, src);
}

// ---------------------------------------------------------------------------
// CLAMP — element-wise clamp
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_clamp(BuildContext& bc) {
  struct ggml_tensor* src = bc.srcs[0];
  float min_val = -3.4e38f, max_val = 3.4e38f;
  if (bc.ir_tensor->op_params() && bc.ir_tensor->op_params()->size() >= 8) {
    const uint8_t* p = bc.ir_tensor->op_params()->data();
    memcpy(&min_val, p, 4);
    memcpy(&max_val, p + 4, 4);
  }
  return ggml_clamp(bc.ctx, src, min_val, max_val);
}

// ---------------------------------------------------------------------------
// SLICE_SCATTER — scatter src into a slice of dst
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_slice_scatter(BuildContext& bc) {
  // For now, return dst (placeholder — full implementation needs ggml_set)
  return bc.srcs[0];
}

// ---------------------------------------------------------------------------
// MOE_FFN — fused Mixture-of-Experts FFN (matches llama.cpp's build_moe_ffn)
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_moe_ffn(BuildContext& bc) {
  // src_ids: [input, gate_inp, gate_exps, up_exps, down_exps]
  // op_params: int32 n_expert, int32 top_k
  struct ggml_tensor* input = bc.srcs[0];      // [n_embd, n_tokens]
  struct ggml_tensor* gate_inp = bc.srcs[1];    // [n_expert, n_embd] (router weight)
  struct ggml_tensor* gate_exps = bc.srcs[2];   // [n_ff, n_embd, n_expert]
  struct ggml_tensor* up_exps = bc.srcs[3];     // [n_ff, n_embd, n_expert]
  struct ggml_tensor* down_exps = bc.srcs[4];   // [n_embd, n_ff, n_expert]

  int32_t n_expert = 0, top_k = 0;
  if (bc.ir_tensor->op_params() && bc.ir_tensor->op_params()->size() >= 8) {
    const uint8_t* p = bc.ir_tensor->op_params()->data();
    memcpy(&n_expert, p, 4);
    memcpy(&top_k, p + 4, 4);
  }

  int64_t n_embd = input->ne[0];
  int64_t n_tokens = input->ne[1];

  // 1. Router: logits = gate_inp @ input → [n_expert, n_tokens]
  auto* logits = ggml_mul_mat(bc.ctx, gate_inp, input);

  // 2. Softmax gating → [n_expert, n_tokens]
  auto* probs = ggml_soft_max(bc.ctx, logits);

  // 3. Top-K expert selection → [top_k, n_tokens] I32
  auto* selected = ggml_argsort_top_k(bc.ctx, probs, top_k);

  // 4. Gather selected expert weights → [1, top_k, n_tokens]
  auto* probs_3d = ggml_reshape_3d(bc.ctx, probs, 1, n_expert, n_tokens);
  auto* weights = ggml_get_rows(bc.ctx, probs_3d, selected);  // [1, top_k, n_tokens]

  // 5. Normalize weights (softmax over selected experts)
  auto* weights_2d = ggml_reshape_2d(bc.ctx, weights, top_k, n_tokens);
  weights = ggml_soft_max(bc.ctx, weights_2d);
  weights = ggml_reshape_3d(bc.ctx, weights, 1, top_k, n_tokens);

  // 6. Expert computation via mul_mat_id
  auto* cur = ggml_reshape_3d(bc.ctx, input, n_embd, 1, n_tokens);

  // gate = gate_exps[selected] @ cur → [n_ff, top_k, n_tokens]
  auto* gate = ggml_mul_mat_id(bc.ctx, gate_exps, cur, selected);
  // up = up_exps[selected] @ cur → [n_ff, top_k, n_tokens]
  auto* up = ggml_mul_mat_id(bc.ctx, up_exps, cur, selected);

  // 7. SwiGLU activation: silu(gate) * up
  auto* act = ggml_swiglu_split(bc.ctx, gate, up);

  // 8. Down projection: down_exps[selected] @ act → [n_embd, top_k, n_tokens]
  auto* experts = ggml_mul_mat_id(bc.ctx, down_exps, act, selected);

  // 9. Weight and reduce: sum(experts * weights, dim=top_k)
  experts = ggml_mul(bc.ctx, experts, weights);

  // Sum over top_k dimension: reshape to [n_embd * top_k, n_tokens] is wrong.
  // Instead: experts is [n_embd, top_k, n_tokens], sum over dim 1 (top_k).
  // ggml doesn't have a general reduce-sum. Use view + sum_rows trick:
  // Transpose to [top_k, n_embd, n_tokens], then sum_rows → [1, n_embd, n_tokens]
  // Actually, in ggml convention ne[0] is the "row" dimension.
  // experts: ne = [n_embd, top_k, n_tokens, 1]
  // We want to sum over ne[1] (top_k). Use permute + sum_rows + reshape.
  auto* exp_perm = ggml_cont(bc.ctx, ggml_permute(bc.ctx, experts, 1, 0, 2, 3));
  // exp_perm: ne = [top_k, n_embd, n_tokens, 1]
  auto* summed = ggml_sum_rows(bc.ctx, exp_perm);
  // summed: ne = [1, n_embd, n_tokens, 1]
  auto* result = ggml_reshape_2d(bc.ctx, summed, n_embd, n_tokens);

  return result;
}

} // namespace executorch_ggml
