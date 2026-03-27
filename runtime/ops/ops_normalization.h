#pragma once

#include "build_context.h"

namespace executorch_ggml {

// ---------------------------------------------------------------------------
// LAYER_NORM
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_layer_norm(BuildContext& bc) {
  const auto* t = bc.ir_tensor;
  float eps = 1e-5f;
  int32_t has_weight = 0, has_bias = 0;
  if (t->op_params() && t->op_params()->size() >= 12) {
    const uint8_t* data = t->op_params()->data();
    memcpy(&eps, data, sizeof(float));
    memcpy(&has_weight, data + 4, sizeof(int32_t));
    memcpy(&has_bias, data + 8, sizeof(int32_t));
  }

  auto* gt = ggml_norm(bc.ctx, ensure_cont(bc.ctx, bc.srcs[0]), eps);

  if (has_weight && bc.srcs.size() > 1) {
    struct ggml_tensor* w = bc.srcs[1];
    if (bc.metal_f32_binops) w = ensure_f32(bc.ctx, w);
    if (bc.cuda_bf16_cast && w->type == GGML_TYPE_BF16) w = ggml_cast(bc.ctx, w, GGML_TYPE_F32);
    gt = ggml_mul(bc.ctx, gt, w);
  }
  if (has_bias) {
    int bias_idx = has_weight ? 2 : 1;
    if ((int)bc.srcs.size() > bias_idx) {
      struct ggml_tensor* b = bc.srcs[bias_idx];
      if (bc.metal_f32_binops) b = ensure_f32(bc.ctx, b);
      if (bc.cuda_bf16_cast && b->type == GGML_TYPE_BF16) b = ggml_cast(bc.ctx, b, GGML_TYPE_F32);
      gt = ggml_add(bc.ctx, gt, b);
    }
  }
  return gt;
}

// ---------------------------------------------------------------------------
// RMS_NORM
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_rms_norm(BuildContext& bc) {
  const auto* t = bc.ir_tensor;
  float eps = 1e-5f;
  int32_t has_weight = 0;
  if (t->op_params() && t->op_params()->size() >= 8) {
    const uint8_t* data = t->op_params()->data();
    memcpy(&eps, data, sizeof(float));
    memcpy(&has_weight, data + 4, sizeof(int32_t));
  }

  struct ggml_tensor* rms_in = ensure_cont(bc.ctx, bc.srcs[0]);
  bool rms_casted = false;
  if (rms_in->type == GGML_TYPE_BF16 || rms_in->type == GGML_TYPE_F16) {
    rms_in = ggml_cast(bc.ctx, rms_in, GGML_TYPE_F32);
    rms_casted = true;
  }
  auto* gt = ggml_rms_norm(bc.ctx, rms_in, eps);

  if (has_weight && bc.srcs.size() > 1) {
    struct ggml_tensor* w = bc.srcs[1];
    if (bc.metal_f32_binops) w = ensure_f32(bc.ctx, w);
    if (w->type == GGML_TYPE_BF16) w = ggml_cast(bc.ctx, w, GGML_TYPE_F32);
    gt = ggml_mul(bc.ctx, gt, w);
  }
  if (rms_casted && bc.srcs[0]->type == GGML_TYPE_BF16 && !bc.skip_bf16_castback) {
    gt = ggml_cast(bc.ctx, gt, GGML_TYPE_BF16);
  }
  return gt;
}

// ---------------------------------------------------------------------------
// BATCH_NORM
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_batch_norm(BuildContext& bc) {
  if (bc.srcs.size() < 5) {
    fprintf(stderr, "[executorch-ggml] BATCH_NORM: expected 5 sources, got %zu\n", bc.srcs.size());
    return nullptr;
  }

  float eps = 1e-5f;
  if (bc.ir_tensor->op_params() && bc.ir_tensor->op_params()->size() >= 4) {
    memcpy(&eps, bc.ir_tensor->op_params()->data(), sizeof(float));
  }

  struct ggml_tensor* x      = bc.srcs[0];
  struct ggml_tensor* weight  = bc.srcs[1];
  struct ggml_tensor* bias    = bc.srcs[2];
  struct ggml_tensor* mean    = bc.srcs[3];
  struct ggml_tensor* var     = bc.srcs[4];

  if (weight->data && bias->data && mean->data && var->data &&
      weight->type == GGML_TYPE_F32 && bias->type == GGML_TYPE_F32 &&
      mean->type == GGML_TYPE_F32 && var->type == GGML_TYPE_F32) {
    const int64_t C = weight->ne[0];
    const float* w_data = (const float*)bc.host_acc.get(weight);
    const float* b_data = (const float*)bc.host_acc.get(bias);
    const float* m_data = (const float*)bc.host_acc.get(mean);
    const float* v_data = (const float*)bc.host_acc.get(var);

    ggml_set_no_alloc(bc.ctx, false);
    struct ggml_tensor* scale4 = ggml_new_tensor_4d(bc.ctx, GGML_TYPE_F32, 1, C, 1, 1);
    struct ggml_tensor* shift4 = ggml_new_tensor_4d(bc.ctx, GGML_TYPE_F32, 1, C, 1, 1);
    ggml_set_no_alloc(bc.ctx, true);

    float* s_data = (float*)scale4->data;
    float* sh_data = (float*)shift4->data;
    for (int64_t c = 0; c < C; ++c) {
      float sc = w_data[c] / sqrtf(v_data[c] + eps);
      s_data[c] = sc;
      sh_data[c] = b_data[c] - m_data[c] * sc;
    }

    return ggml_add(bc.ctx, ggml_mul(bc.ctx, x, scale4), shift4);
  } else {
    struct ggml_tensor* var_eps = ggml_add1(bc.ctx, var, make_f32_scalar(bc.ctx, eps));
    struct ggml_tensor* inv_std = ggml_sqrt(bc.ctx, var_eps);
    struct ggml_tensor* one = make_f32_scalar(bc.ctx, 1.0f);
    struct ggml_tensor* one_rep = ggml_repeat(bc.ctx, one, inv_std);
    struct ggml_tensor* recip_std = ggml_div(bc.ctx, one_rep, inv_std);

    struct ggml_tensor* scale = ggml_mul(bc.ctx, weight, recip_std);
    struct ggml_tensor* mean_scaled = ggml_mul(bc.ctx, mean, scale);
    struct ggml_tensor* shift = ggml_sub(bc.ctx, bias, mean_scaled);

    int64_t C = scale->ne[0];
    struct ggml_tensor* scale4 = ggml_reshape_4d(bc.ctx, scale, 1, C, 1, 1);
    struct ggml_tensor* shift4 = ggml_reshape_4d(bc.ctx, shift, 1, C, 1, 1);

    auto* gt = ggml_mul(bc.ctx, x, ggml_repeat(bc.ctx, scale4, x));
    gt = ggml_add(bc.ctx, gt, ggml_repeat(bc.ctx, shift4, gt));
    return gt;
  }
}

} // namespace executorch_ggml
