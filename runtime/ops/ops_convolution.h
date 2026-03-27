#pragma once

#include "build_context.h"

namespace executorch_ggml {

// ---------------------------------------------------------------------------
// CONV_2D / CONV_2D_DW
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_conv_2d(BuildContext& bc, bool is_dw) {
  const auto* t = bc.ir_tensor;
  if (bc.srcs.size() < 2) {
    return nullptr;
  }

  struct ggml_tensor* weight = bc.srcs[0];
  struct ggml_tensor* input = bc.srcs[1];
  struct ggml_tensor* bias = bc.srcs.size() > 2 ? bc.srcs[2] : nullptr;

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

  struct ggml_tensor* gt = nullptr;

  int64_t kh = weight->ne[1];
  int64_t kw = weight->ne[0];
  bool is_pointwise_2d = (kh == 1 && kw == 1
                          && stride_h == 1 && stride_w == 1
                          && pad_h == 0 && pad_w == 0
                          && dilation_h == 1 && dilation_w == 1
                          && groups == 1);

  if (is_pointwise_2d) {
    struct ggml_tensor* w2d = ggml_reshape_2d(bc.ctx, weight,
                                              weight->ne[2], weight->ne[3]);
    struct ggml_tensor* inp3d = ggml_reshape_3d(bc.ctx, ensure_cont(bc.ctx, input),
                                                input->ne[0] * input->ne[1],
                                                input->ne[2],
                                                input->ne[3]);
    struct ggml_tensor* inp_t = ggml_cont(bc.ctx, ggml_transpose(bc.ctx, inp3d));
    struct ggml_tensor* mm = ggml_mul_mat(bc.ctx, w2d, inp_t);
    struct ggml_tensor* mm_t = ggml_cont(bc.ctx, ggml_transpose(bc.ctx, mm));
    gt = ggml_reshape_4d(bc.ctx, mm_t,
                         input->ne[0], input->ne[1],
                         weight->ne[3], input->ne[3]);
  } else if (is_dw || groups > 1) {
    gt = ggml_conv_2d_dw_direct(bc.ctx, weight, input, stride_w, stride_h, pad_w, pad_h, dilation_w, dilation_h);
  } else {
    if (weight->type == GGML_TYPE_BF16) {
      weight = ggml_cast(bc.ctx, weight, GGML_TYPE_F32);
    }
    if (input->type == GGML_TYPE_BF16) {
      input = ggml_cast(bc.ctx, input, GGML_TYPE_F32);
    }
    gt = ggml_conv_2d(bc.ctx, weight, input, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w);
  }

  if (gt && gt->type == GGML_TYPE_F16) {
    gt = safe_ggml_cast(bc.ctx, gt, GGML_TYPE_F32, &bc.host_acc);
  }

  if (bias && gt) {
    struct ggml_tensor* bias4 = ggml_reshape_4d(bc.ctx, bias, 1, 1, bias->ne[0], 1);
    if (!ggml_can_repeat(bias4, gt)) {
      fprintf(stderr,
              "[executorch-ggml] CONV bias not repeatable: b=(%lld,%lld,%lld,%lld) y=(%lld,%lld,%lld,%lld)\n",
              (long long) bias4->ne[0], (long long) bias4->ne[1], (long long) bias4->ne[2], (long long) bias4->ne[3],
              (long long) gt->ne[0], (long long) gt->ne[1], (long long) gt->ne[2], (long long) gt->ne[3]);
      return nullptr;
    }
    if (bias4->type == GGML_TYPE_F16) {
      bias4 = safe_ggml_cast(bc.ctx, bias4, GGML_TYPE_F32, &bc.host_acc);
    }
    gt = ggml_add(bc.ctx, gt, bias4);
  }
  return gt;
}

// ---------------------------------------------------------------------------
// CONV_1D / CONV_1D_DW
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_conv_1d(BuildContext& bc, bool is_dw) {
  const auto* t = bc.ir_tensor;
  if (bc.srcs.size() < 2) {
    return nullptr;
  }

  struct ggml_tensor* weight = bc.srcs[0];
  struct ggml_tensor* input = bc.srcs[1];
  struct ggml_tensor* bias = bc.srcs.size() > 2 ? bc.srcs[2] : nullptr;

  int32_t stride = 1, pad = 0, dilation = 1, groups = 1;
  if (t->op_params() && t->op_params()->size() >= 16) {
    const uint8_t* data = t->op_params()->data();
    memcpy(&stride, data, sizeof(int32_t));
    memcpy(&pad, data + 4, sizeof(int32_t));
    memcpy(&dilation, data + 8, sizeof(int32_t));
    memcpy(&groups, data + 12, sizeof(int32_t));
  }

  struct ggml_tensor* gt = nullptr;

  int64_t kernel_size = weight->ne[0];
  bool is_pointwise = (kernel_size == 1 && stride == 1 && pad == 0
                       && dilation == 1 && groups == 1);

  if (is_pointwise) {
    struct ggml_tensor* w2d = ggml_reshape_2d(bc.ctx, weight,
                                              weight->ne[1], weight->ne[2]);
    struct ggml_tensor* inp_t = ggml_cont(bc.ctx, ggml_transpose(bc.ctx, input));
    struct ggml_tensor* mm = ggml_mul_mat(bc.ctx, w2d, inp_t);
    gt = ggml_cont(bc.ctx, ggml_transpose(bc.ctx, mm));
  } else if (is_dw || groups > 1) {
    struct ggml_tensor* w2d = ggml_reshape_4d(bc.ctx, weight,
        weight->ne[0], 1, 1, weight->ne[2]);
    struct ggml_tensor* inp2d = ggml_reshape_4d(bc.ctx,
        ensure_cont(bc.ctx, input),
        input->ne[0], 1, input->ne[1], input->ne[2]);
    struct ggml_tensor* conv_out = ggml_conv_2d_dw_direct(bc.ctx,
        w2d, inp2d, stride, 1, pad, 0, dilation, 1);
    gt = ggml_reshape_4d(bc.ctx, conv_out,
        conv_out->ne[0], conv_out->ne[2], conv_out->ne[3], 1);
  } else {
    gt = ggml_conv_1d(bc.ctx, weight, input, stride, pad, dilation);
  }

  if (gt && gt->type == GGML_TYPE_F16) {
    gt = safe_ggml_cast(bc.ctx, gt, GGML_TYPE_F32, &bc.host_acc);
  }

  if (bias && gt) {
    struct ggml_tensor* bias4 = ggml_reshape_4d(bc.ctx, bias, 1, bias->ne[0], 1, 1);
    if (!ggml_can_repeat(bias4, gt)) {
      fprintf(stderr,
              "[executorch-ggml] CONV_1D bias not repeatable: b=(%lld,%lld,%lld,%lld) y=(%lld,%lld,%lld,%lld)\n",
              (long long) bias4->ne[0], (long long) bias4->ne[1], (long long) bias4->ne[2], (long long) bias4->ne[3],
              (long long) gt->ne[0], (long long) gt->ne[1], (long long) gt->ne[2], (long long) gt->ne[3]);
      return nullptr;
    }
    if (bias4->type == GGML_TYPE_F16 ||
        (bc.metal_f32_binops && bias4->type == GGML_TYPE_BF16)) {
      bias4 = safe_ggml_cast(bc.ctx, bias4, GGML_TYPE_F32, &bc.host_acc);
    }
    gt = ggml_add(bc.ctx, gt, bias4);
  }
  return gt;
}

// ---------------------------------------------------------------------------
// PAD
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_pad(BuildContext& bc) {
  const auto* t = bc.ir_tensor;
  int32_t ndim_pairs = 0;
  int32_t left[4] = {0, 0, 0, 0};
  int32_t right[4] = {0, 0, 0, 0};

  if (t->op_params() && t->op_params()->size() >= 4) {
    const uint8_t* data = t->op_params()->data();
    size_t off = 0;
    memcpy(&ndim_pairs, data + off, sizeof(int32_t)); off += 4;

    for (int32_t i = 0; i < ndim_pairs && i < 4; i++) {
      memcpy(&left[i], data + off, sizeof(int32_t)); off += 4;
      memcpy(&right[i], data + off, sizeof(int32_t)); off += 4;
    }
  }

  return ggml_pad_ext(bc.ctx, ensure_cont(bc.ctx, bc.srcs[0]),
                      left[0], right[0],
                      left[1], right[1],
                      left[2], right[2],
                      left[3], right[3]);
}

} // namespace executorch_ggml
