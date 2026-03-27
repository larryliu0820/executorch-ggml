#pragma once

#include "build_context.h"

namespace executorch_ggml {

// ---------------------------------------------------------------------------
// MUL_MAT
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_mul_mat(BuildContext& bc) {
  struct ggml_tensor* mm_w = bc.srcs[0];
  struct ggml_tensor* mm_x = bc.srcs[1];
  // CUDA cuBLAS BF16 path only supports F32/F16/BF16 activations.
  if (mm_w->type == GGML_TYPE_BF16 &&
      mm_x->type != GGML_TYPE_F32 &&
      mm_x->type != GGML_TYPE_F16 &&
      mm_x->type != GGML_TYPE_BF16) {
    mm_x = safe_ggml_cast(bc.ctx, mm_x, GGML_TYPE_F32, &bc.host_acc);
  }
  return ggml_mul_mat(bc.ctx, mm_w, mm_x);
}

// ---------------------------------------------------------------------------
// BMM
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_bmm(BuildContext& bc) {
  struct ggml_tensor* a = bc.srcs[0];  // [K, M, B, 1]
  struct ggml_tensor* b = bc.srcs[1];  // [N, K, B, 1]

  // Transpose b to get [K, N, B, 1] so ne[0]=K matches a->ne[0]=K
  struct ggml_tensor* b_t = ggml_cont(bc.ctx, ggml_transpose(bc.ctx, b));
  return ggml_mul_mat(bc.ctx, b_t, a);  // result: [N, M, B, 1]
}

// ---------------------------------------------------------------------------
// LINEAR
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_linear(BuildContext& bc) {
  struct ggml_tensor* x = bc.srcs[0];
  struct ggml_tensor* w = bc.srcs[1];
  // CUDA cuBLAS BF16 path only supports F32/F16/BF16 activations.
  if (w->type == GGML_TYPE_BF16 &&
      x->type != GGML_TYPE_F32 &&
      x->type != GGML_TYPE_F16 &&
      x->type != GGML_TYPE_BF16) {
    x = safe_ggml_cast(bc.ctx, x, GGML_TYPE_F32, &bc.host_acc);
  }
  struct ggml_tensor* y = ggml_mul_mat(bc.ctx, w, x);
  if (bc.srcs.size() > 2) {
    struct ggml_tensor* b = bc.srcs[2];
    if (ggml_n_dims(b) == 1) {
      b = ggml_reshape_4d(bc.ctx, b, b->ne[0], 1, 1, 1);
    }
    if (!ggml_can_repeat(b, y)) {
      fprintf(stderr,
              "[executorch-ggml] LINEAR bias not repeatable: b=(%lld,%lld,%lld,%lld) y=(%lld,%lld,%lld,%lld)\n",
              (long long) b->ne[0], (long long) b->ne[1], (long long) b->ne[2], (long long) b->ne[3],
              (long long) y->ne[0], (long long) y->ne[1], (long long) y->ne[2], (long long) y->ne[3]);
      return nullptr;
    }
    if (bc.metal_f32_binops) b = ensure_f32(bc.ctx, b);
    if (bc.cuda_bf16_cast && b->type == GGML_TYPE_BF16) b = ggml_cast(bc.ctx, b, GGML_TYPE_F32);
    y = ggml_add(bc.ctx, y, b);
  }
  return y;
}

// ---------------------------------------------------------------------------
// EMBEDDING
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_embedding(BuildContext& bc) {
  struct ggml_tensor* w = bc.srcs[0];
  struct ggml_tensor* idx = bc.srcs[1];
  if (idx->type == GGML_TYPE_I64) {
    idx = eager_cast_i64_to_i32(bc.ctx, idx, &bc.host_acc);
  } else if (idx->type != GGML_TYPE_I32) {
    idx = safe_ggml_cast(bc.ctx, idx, GGML_TYPE_I32, &bc.host_acc);
  }
  return ggml_get_rows(bc.ctx, w, idx);
}

} // namespace executorch_ggml
