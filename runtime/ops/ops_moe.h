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
  // Eager-fold when src is a constant (NONE op with host data, not input-derived).
  // This removes e.g. exp(A_log) from the runtime graph in GatedDeltaNet.
  if (src->op == GGML_OP_NONE && src->data && !bc.input_derived.count(src)) {
    const int64_t n = ggml_nelements(src);
    ggml_set_no_alloc(bc.ctx, false);
    struct ggml_tensor* out = ggml_new_tensor_4d(bc.ctx, GGML_TYPE_F32,
                                                 src->ne[0], src->ne[1], src->ne[2], src->ne[3]);
    ggml_set_no_alloc(bc.ctx, true);
    out->op = GGML_OP_NONE;
    const float* sd = (const float*)bc.host_acc.get(src);
    float* od = (float*)out->data;
    for (int64_t i = 0; i < n; i++) od[i] = expf(sd[i]);
    return out;
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
  if (dim < 0) dim += ndim;
  int ggml_axis = (ndim - 1) - dim;
  if (ggml_axis < 0) ggml_axis = 0;
  if (ggml_axis > 3) ggml_axis = 3;

  // ggml_sum_rows sums along ne[0]. For other axes, permute target axis to ne[0],
  // sum_rows, permute back. This mirrors the ANY op pattern.
  if (ggml_axis == 0) {
    return ggml_sum_rows(bc.ctx, src);
  }
  int axes[4] = {0, 1, 2, 3};
  axes[0] = ggml_axis;
  axes[ggml_axis] = 0;
  struct ggml_tensor* x = ggml_cont(bc.ctx,
      safe_ggml_permute(bc.ctx, src, axes[0], axes[1], axes[2], axes[3], "SUM_pre"));
  x = ggml_sum_rows(bc.ctx, x);
  x = safe_ggml_permute(bc.ctx, x, axes[0], axes[1], axes[2], axes[3], "SUM_post");
  return x;
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

  // 3. Reshape to 3D for get_rows compatibility. Placing this BEFORE the argsort
  //    matches llama.cpp's build_moe_ffn pattern and enables the CUDA topk_moe
  //    fusion: { SOFT_MAX, RESHAPE, ARGSORT, VIEW, GET_ROWS, RESHAPE, SOFT_MAX,
  //    RESHAPE } (ggml-cuda.cu:3733).
  auto* probs_3d = ggml_reshape_3d(bc.ctx, probs, 1, n_expert, n_tokens);

  // 4. Top-K expert selection → [top_k, n_tokens] I32
  auto* selected = ggml_argsort_top_k(bc.ctx, probs, top_k);

  // 5. Gather selected expert weights → [1, top_k, n_tokens]
  auto* weights = ggml_get_rows(bc.ctx, probs_3d, selected);  // [1, top_k, n_tokens]

  // 6. Normalize weights (softmax over selected experts)
  auto* weights_2d = ggml_reshape_2d(bc.ctx, weights, top_k, n_tokens);
  weights = ggml_soft_max(bc.ctx, weights_2d);
  weights = ggml_reshape_3d(bc.ctx, weights, 1, top_k, n_tokens);

  // 7. Expert computation via mul_mat_id
  auto* cur = ggml_reshape_3d(bc.ctx, input, n_embd, 1, n_tokens);

  // gate = gate_exps[selected] @ cur → [n_ff, top_k, n_tokens]
  auto* gate = ggml_mul_mat_id(bc.ctx, gate_exps, cur, selected);
  // up = up_exps[selected] @ cur → [n_ff, top_k, n_tokens]
  auto* up = ggml_mul_mat_id(bc.ctx, up_exps, cur, selected);

  // 7. SwiGLU activation: silu(gate) * up
  auto* act = ggml_swiglu_split(bc.ctx, gate, up);

  // 8. Down projection: down_exps[selected] @ act → [n_embd, top_k, n_tokens]
  auto* experts = ggml_mul_mat_id(bc.ctx, down_exps, act, selected);

  // 9. Weight: experts *= weights, then reduce over top_k via view+add chain.
  // llama.cpp pattern: avoids PERMUTE+CONT+SUM_ROWS setup by treating each
  // top_k slice as a view and summing with ggml_add.
  // experts: ne = [n_embd, top_k, n_tokens, 1]
  experts = ggml_mul(bc.ctx, experts, weights);

  struct ggml_tensor* moe_out = ggml_view_2d(bc.ctx, experts,
      n_embd, n_tokens, experts->nb[2], 0);
  for (int32_t i = 1; i < top_k; ++i) {
    struct ggml_tensor* slice = ggml_view_2d(bc.ctx, experts,
        n_embd, n_tokens, experts->nb[2], i * experts->nb[1]);
    moe_out = ggml_add(bc.ctx, moe_out, slice);
  }
  if (top_k == 1) {
    // avoid non-contiguous view output
    moe_out = ggml_cont(bc.ctx, moe_out);
  }
  return moe_out;
}

// ---------------------------------------------------------------------------
// GATED_DELTA_NET — fused recurrent delta-rule step (llama.cpp's
// ggml_gated_delta_net). Replaces ~80 ops per GDN layer with a single CUDA
// kernel.
//
// Emits TWO outputs (output, new_state). Both IR tensors share the same 6
// sources but have distinct op_params[output_index]. The first call builds
// ggml_gated_delta_net; the second reuses the cached result and returns the
// other view.
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_gated_delta_net(BuildContext& bc) {
  if (bc.srcs.size() < 6) {
    fprintf(stderr, "[executorch-ggml] GATED_DELTA_NET: expected 6 sources, got %zu\n", bc.srcs.size());
    return nullptr;
  }

  int32_t output_index = 0;
  if (bc.ir_tensor->op_params() && bc.ir_tensor->op_params()->size() >= 4) {
    memcpy(&output_index, bc.ir_tensor->op_params()->data(), 4);
  }

  struct ggml_tensor* q     = bc.srcs[0];
  struct ggml_tensor* k     = bc.srcs[1];
  struct ggml_tensor* v     = bc.srcs[2];
  struct ggml_tensor* g     = bc.srcs[3];
  struct ggml_tensor* beta  = bc.srcs[4];
  struct ggml_tensor* state = bc.srcs[5];

  // Cache the ggml_gated_delta_net tensor per (q, k, v, g, beta, state) tuple
  // so the two output_index IR tensors share the same underlying op.
  uint64_t cache_key =
      reinterpret_cast<uint64_t>(q) ^
      (reinterpret_cast<uint64_t>(k)     << 3) ^
      (reinterpret_cast<uint64_t>(v)     << 7) ^
      (reinterpret_cast<uint64_t>(g)     << 11) ^
      (reinterpret_cast<uint64_t>(beta)  << 13) ^
      (reinterpret_cast<uint64_t>(state) << 17);
  auto it = bc.gdn_cache.find(cache_key);
  struct ggml_tensor* result = nullptr;
  if (it != bc.gdn_cache.end()) {
    result = it->second;
  } else {
    // ggml_gated_delta_net requires F32, contiguous.
    if (q->type != GGML_TYPE_F32)     q     = safe_ggml_cast(bc.ctx, q,     GGML_TYPE_F32, &bc.host_acc);
    if (k->type != GGML_TYPE_F32)     k     = safe_ggml_cast(bc.ctx, k,     GGML_TYPE_F32, &bc.host_acc);
    if (v->type != GGML_TYPE_F32)     v     = safe_ggml_cast(bc.ctx, v,     GGML_TYPE_F32, &bc.host_acc);
    if (g->type != GGML_TYPE_F32)     g     = safe_ggml_cast(bc.ctx, g,     GGML_TYPE_F32, &bc.host_acc);
    if (beta->type != GGML_TYPE_F32)  beta  = safe_ggml_cast(bc.ctx, beta,  GGML_TYPE_F32, &bc.host_acc);
    if (state->type != GGML_TYPE_F32) state = safe_ggml_cast(bc.ctx, state, GGML_TYPE_F32, &bc.host_acc);

    q     = ensure_cont(bc.ctx, q);
    k     = ensure_cont(bc.ctx, k);
    v     = ensure_cont(bc.ctx, v);
    g     = ensure_cont(bc.ctx, g);
    beta  = ensure_cont(bc.ctx, beta);
    state = ensure_cont(bc.ctx, state);

    // Reshape g and beta to expected 4D shape [1|S_v, H_v, T, B].
    // Python emits g/beta with shape [B, T, H_v]; ggml order is
    // [H_v, T, B, 1]. ggml_gated_delta_net wants [1, H_v, T, B].
    // Condition: first (innermost) dim is H_v, i.e. != 1 and not already S_v.
    // Also require ne[3] == 1 so we're operating on a rank-3 logical tensor.
    const int64_t S_v_chk = v->ne[0];
    auto maybe_promote = [&](struct ggml_tensor* x) {
      if (x->ne[3] == 1 && x->ne[0] != 1 && x->ne[0] != S_v_chk) {
        return ggml_reshape_4d(bc.ctx, x, 1, x->ne[0], x->ne[1], x->ne[2]);
      }
      return x;
    };
    g = maybe_promote(g);
    beta = maybe_promote(beta);

    result = ggml_gated_delta_net(bc.ctx, q, k, v, g, beta, state);
    bc.gdn_cache[cache_key] = result;
  }

  // result is a 2D concat tensor: [S_v*H, n_tokens*n_seqs + S_v*n_seqs].
  // Output lives at offset 0, state lives at offset S_v*H*n_tokens*n_seqs.
  const int64_t S_v      = v->ne[0];
  const int64_t H_v      = v->ne[1];
  const int64_t n_tokens = v->ne[2];
  const int64_t n_seqs   = v->ne[3];

  if (output_index == 0) {
    // Output: [S_v, H_v, n_tokens, n_seqs]
    return ggml_view_4d(bc.ctx, result, S_v, H_v, n_tokens, n_seqs,
                        result->nb[0] * S_v,
                        result->nb[0] * S_v * H_v,
                        result->nb[0] * S_v * H_v * n_tokens,
                        0);
  } else {
    // New state: [S_v, S_v, H_v, n_seqs] at offset S_v*H_v*n_tokens*n_seqs.
    const size_t state_offset = (size_t) result->nb[0] * S_v * H_v * n_tokens * n_seqs;
    return ggml_view_4d(bc.ctx, result, S_v, S_v, H_v, n_seqs,
                        result->nb[0] * S_v,
                        result->nb[0] * S_v * S_v,
                        result->nb[0] * S_v * S_v * H_v,
                        state_offset);
  }
}

// ---------------------------------------------------------------------------
// SSM_CONV — depthwise conv1d with state (llama.cpp's ggml_ssm_conv).
//
// PyTorch input:  conv_input [B, C, L],  weight [C, K]
// Abstract output: [B, T, C] — matches ggml_ssm_conv native layout so no
// permute is needed. Python model code slices as acc[:, -T:, :] instead
// of the equivalent acc[:, :, -T:].transpose(1,2).
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_ssm_conv(BuildContext& bc) {
  struct ggml_tensor* conv_input = bc.srcs[0];
  struct ggml_tensor* weight = bc.srcs[1];
  if (conv_input->type != GGML_TYPE_F32) {
    conv_input = safe_ggml_cast(bc.ctx, conv_input, GGML_TYPE_F32, &bc.host_acc);
  }
  if (weight->type != GGML_TYPE_F32) {
    weight = safe_ggml_cast(bc.ctx, weight, GGML_TYPE_F32, &bc.host_acc);
  }
  return ggml_ssm_conv(bc.ctx, conv_input, weight);
}

} // namespace executorch_ggml
