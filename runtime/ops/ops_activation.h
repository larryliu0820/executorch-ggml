#pragma once

#include "build_context.h"

namespace executorch_ggml {

// ---------------------------------------------------------------------------
// SILU
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_silu(BuildContext& bc) {
  struct ggml_tensor* x = bc.srcs[0];
  ggml_type orig = x->type;
  if (x->type == GGML_TYPE_BF16) x = ggml_cast(bc.ctx, x, GGML_TYPE_F32);
  auto* gt = ggml_silu(bc.ctx, x);
  if (orig == GGML_TYPE_BF16 && !bc.skip_bf16_castback) gt = ggml_cast(bc.ctx, gt, GGML_TYPE_BF16);
  return gt;
}

// ---------------------------------------------------------------------------
// RELU
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_relu(BuildContext& bc) {
  return ggml_relu(bc.ctx, bc.srcs[0]);
}

// ---------------------------------------------------------------------------
// SOFTPLUS — log(1 + exp(x)) with numerically stable handling for large x.
// Maps directly to ggml_softplus (native op — llama.cpp uses this for
// Qwen3.5 MoE SSM dt bias + A_log gating).
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_softplus(BuildContext& bc) {
  struct ggml_tensor* x = bc.srcs[0];
  ggml_type orig = x->type;
  if (x->type == GGML_TYPE_BF16) x = ggml_cast(bc.ctx, x, GGML_TYPE_F32);
  auto* gt = ggml_softplus(bc.ctx, x);
  if (orig == GGML_TYPE_BF16 && !bc.skip_bf16_castback) gt = ggml_cast(bc.ctx, gt, GGML_TYPE_BF16);
  return gt;
}

// ---------------------------------------------------------------------------
// TANH
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_tanh(BuildContext& bc) {
  return ggml_tanh(bc.ctx, bc.srcs[0]);
}

// ---------------------------------------------------------------------------
// GELU
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_gelu(BuildContext& bc) {
  struct ggml_tensor* x = bc.srcs[0];
  ggml_type orig = x->type;
  if (x->type == GGML_TYPE_BF16) x = ggml_cast(bc.ctx, x, GGML_TYPE_F32);
  auto* gt = ggml_gelu(bc.ctx, x);
  if (orig == GGML_TYPE_BF16 && !bc.skip_bf16_castback) gt = ggml_cast(bc.ctx, gt, GGML_TYPE_BF16);
  return gt;
}

// ---------------------------------------------------------------------------
// LEAKY_RELU
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_leaky_relu(BuildContext& bc) {
  float negative_slope = 0.01f;
  if (bc.ir_tensor->op_params() && bc.ir_tensor->op_params()->size() >= 4) {
    memcpy(&negative_slope, bc.ir_tensor->op_params()->data(), sizeof(float));
  }
  return ggml_leaky_relu(bc.ctx, bc.srcs[0], negative_slope, false);
}

// ---------------------------------------------------------------------------
// SIGMOID
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_sigmoid(BuildContext& bc) {
  return ggml_sigmoid(bc.ctx, bc.srcs[0]);
}

// ---------------------------------------------------------------------------
// SOFTMAX
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_softmax(BuildContext& bc) {
  const auto* t = bc.ir_tensor;
  int32_t dim = -1, ndim = 4;
  if (t->op_params() && t->op_params()->size() >= 8) {
    memcpy(&dim, t->op_params()->data(), 4);
    memcpy(&ndim, t->op_params()->data() + 4, 4);
  }

  if (dim < 0) dim = ndim + dim;
  int ggml_axis = (ndim - 1) - dim;

  struct ggml_tensor* x = bc.srcs[0];
  ggml_type orig_type = x->type;
  // CUDA softmax requires F32 input.
  if (x->type != GGML_TYPE_F32) {
    x = safe_ggml_cast(bc.ctx, x, GGML_TYPE_F32, &bc.host_acc);
  }
  struct ggml_tensor* gt;
  if (ggml_axis == 0) {
    gt = ggml_soft_max(bc.ctx, x);
  } else {
    int perm[4] = {0, 1, 2, 3};
    perm[0] = ggml_axis;
    perm[ggml_axis] = 0;
    x = safe_ggml_permute(bc.ctx, x, perm[0], perm[1], perm[2], perm[3], "SOFTMAX_pre");
    x = ensure_cont(bc.ctx, x);
    x = ggml_soft_max(bc.ctx, x);
    gt = safe_ggml_permute(bc.ctx, x, perm[0], perm[1], perm[2], perm[3], "SOFTMAX_post");
  }
  if (orig_type == GGML_TYPE_BF16 && !bc.skip_bf16_castback) {
    gt = ggml_cast(bc.ctx, gt, GGML_TYPE_BF16);
  }
  return gt;
}

// ---------------------------------------------------------------------------
// HARDTANH
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_hardtanh(BuildContext& bc) {
  float min_val = -1.0f;
  float max_val = 1.0f;
  if (bc.ir_tensor->op_params() && bc.ir_tensor->op_params()->size() >= 8) {
    const uint8_t* data = bc.ir_tensor->op_params()->data();
    memcpy(&min_val, data, sizeof(float));
    memcpy(&max_val, data + 4, sizeof(float));
  }
  return ggml_clamp(bc.ctx, bc.srcs[0], min_val, max_val);
}

// ---------------------------------------------------------------------------
// COS
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_cos(BuildContext& bc) {
  return ggml_cos(bc.ctx, bc.srcs[0]);
}

// ---------------------------------------------------------------------------
// SIN
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_sin(BuildContext& bc) {
  return ggml_sin(bc.ctx, bc.srcs[0]);
}

} // namespace executorch_ggml
