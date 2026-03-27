#pragma once

#include "build_context.h"
#include "custom_ops.h"

namespace executorch_ggml {

// ---------------------------------------------------------------------------
// ARANGE
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_arange(BuildContext& bc) {
  const auto* t = bc.ir_tensor;
  double start = 0.0, step = 1.0;
  if (t->op_params() && t->op_params()->size() >= 16) {
    memcpy(&start, t->op_params()->data(), 8);
    memcpy(&step, t->op_params()->data() + 8, 8);
  }

  ggml_type out_type = GGML_TYPE_I64;
  switch (t->type()) {
    case ggml_ir::TensorType::F32: out_type = GGML_TYPE_F32; break;
    case ggml_ir::TensorType::F16: out_type = GGML_TYPE_F16; break;
    case ggml_ir::TensorType::I32: out_type = GGML_TYPE_I32; break;
    case ggml_ir::TensorType::I64: out_type = GGML_TYPE_I64; break;
    default: out_type = GGML_TYPE_I64; break;
  }

  const int64_t nelem = bc.ne[0] * bc.ne[1] * bc.ne[2] * bc.ne[3];
  float stop = (float)(start + nelem * step);
  auto* gt = ggml_arange(bc.ctx, (float)start, stop, (float)step);

  if (out_type == GGML_TYPE_F16) {
    gt = safe_ggml_cast(bc.ctx, gt, GGML_TYPE_F16, &bc.host_acc);
  }
  return gt;
}

// ---------------------------------------------------------------------------
// FULL
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_full(BuildContext& bc) {
  const auto* t = bc.ir_tensor;
  double fill_value = 0.0;
  if (t->op_params() && t->op_params()->size() >= 8) {
    memcpy(&fill_value, t->op_params()->data(), 8);
  }

  ggml_type out_type = GGML_TYPE_F32;
  switch (t->type()) {
    case ggml_ir::TensorType::F32:  out_type = GGML_TYPE_F32; break;
    case ggml_ir::TensorType::F16:  out_type = GGML_TYPE_F16; break;
    case ggml_ir::TensorType::I32:  out_type = GGML_TYPE_I32; break;
    case ggml_ir::TensorType::I64:  out_type = GGML_TYPE_I64; break;
    case ggml_ir::TensorType::BOOL: out_type = GGML_TYPE_I32; break;
    default: out_type = GGML_TYPE_F32; break;
  }

  auto* gt = ggml_repeat_4d(bc.ctx, make_f32_scalar(bc.ctx, (float)fill_value),
                             bc.ne[0], bc.ne[1], bc.ne[2], bc.ne[3]);

  if (out_type == GGML_TYPE_F16) {
    gt = safe_ggml_cast(bc.ctx, gt, GGML_TYPE_F16, &bc.host_acc);
  }
  return gt;
}

// ---------------------------------------------------------------------------
// CUMSUM
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_cumsum(BuildContext& bc) {
  const auto* t = bc.ir_tensor;
  int32_t dim = 0, ndim = 4;
  if (t->op_params() && t->op_params()->size() >= 8) {
    memcpy(&dim, t->op_params()->data(), 4);
    memcpy(&ndim, t->op_params()->data() + 4, 4);
  }

  struct ggml_tensor* src = bc.srcs[0];
  ggml_type out_type = src->type;

  int32_t ggml_axis = (ndim - 1) - dim;

  if (out_type == GGML_TYPE_I64) {
    src = safe_ggml_cast(bc.ctx, src, GGML_TYPE_I32, &bc.host_acc);
    out_type = GGML_TYPE_I32;
  }

  struct ggml_tensor* args[1] = {src};
  auto* gt = ggml_custom_4d(bc.ctx, out_type,
      src->ne[0], src->ne[1], src->ne[2], src->ne[3],
      args, 1, ggml_custom_cumsum, 1, nullptr);
  memcpy(gt->op_params, &ggml_axis, sizeof(int32_t));
  pin_to_cpu(bc, gt);
  return gt;
}

// ---------------------------------------------------------------------------
// ARGMAX
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_argmax(BuildContext& bc) {
  const auto* t = bc.ir_tensor;
  int32_t dim = -1, ndim = 1;
  if (t->op_params() && t->op_params()->size() >= 8) {
    const uint8_t* data = t->op_params()->data();
    memcpy(&dim, data, sizeof(int32_t));
    memcpy(&ndim, data + 4, sizeof(int32_t));
  }

  if (dim < 0) dim += ndim;
  int ggml_axis = (ndim - 1) - dim;

  struct ggml_tensor* x = bc.srcs[0];

  if (ggml_axis == 0) {
    return ggml_argmax(bc.ctx, x);
  } else {
    int perm[4] = {0, 1, 2, 3};
    perm[0] = ggml_axis;
    perm[ggml_axis] = 0;
    struct ggml_tensor* xp = safe_ggml_permute(bc.ctx, x, perm[0], perm[1], perm[2], perm[3], "ARGMAX");
    xp = ggml_cont(bc.ctx, xp);
    return ggml_argmax(bc.ctx, xp);
  }
}

// ---------------------------------------------------------------------------
// CAST
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_cast(BuildContext& bc) {
  const auto* t = bc.ir_tensor;
  struct ggml_tensor* src = bc.srcs[0];

  int32_t target_type_enum = 0;
  if (t->op_params() && t->op_params()->size() >= 4) {
    memcpy(&target_type_enum, t->op_params()->data(), 4);
  }
  ggml_type target_type = GGML_TYPE_F32;
  switch (target_type_enum) {
    case 0: target_type = GGML_TYPE_F32; break;
    case 1: target_type = GGML_TYPE_F16; break;
    case 2: target_type = GGML_TYPE_I64; break;
    case 3: target_type = GGML_TYPE_I32; break;
    default: target_type = GGML_TYPE_F32; break;
  }

  if (src->type == target_type) {
    return src;
  }

  // Eager scalar cast
  if (ggml_nelements(src) == 1 && src->data
      && !bc.input_derived.count(src)) {
    const void* src_host = bc.host_acc.get(src);
    ggml_set_no_alloc(bc.ctx, false);
    auto* gt = ggml_new_tensor_4d(bc.ctx, target_type, 1, 1, 1, 1);
    ggml_set_no_alloc(bc.ctx, true);
    gt->op = GGML_OP_NONE;
    double val = read_scalar_f64_ptr(static_cast<const char*>(src_host), src->type);
    write_scalar_f64_ptr(static_cast<char*>(gt->data), target_type, val);
    return gt;
  }

  if (src->type == GGML_TYPE_I64) {
    // I64 scalar -> F32: graph op path via safe_ggml_cast
    if (target_type == GGML_TYPE_F32 && ggml_nelements(src) == 1) {
      return safe_ggml_cast(bc.ctx, src, GGML_TYPE_F32, &bc.host_acc);
    }

    ggml_set_no_alloc(bc.ctx, false);
    auto* gt = ggml_new_tensor(bc.ctx, target_type, GGML_MAX_DIMS, src->ne);
    ggml_set_no_alloc(bc.ctx, true);

    bool is_input_src = false;
    for (const auto& [idx, inp_tensor] : bc.input_pairs) {
      if (inp_tensor == src) {
        is_input_src = true;
        break;
      }
    }

    if (is_input_src && target_type == GGML_TYPE_I32) {
      bc.deferred_i64_to_i32.emplace_back(src, gt);
    } else if (src->data && gt->data) {
      const size_t nelem = ggml_nelements(src);
      const int64_t* src_data = static_cast<const int64_t*>(bc.host_acc.get(src));
      if (target_type == GGML_TYPE_I32) {
        int32_t* dst_data = static_cast<int32_t*>(gt->data);
        for (size_t i = 0; i < nelem; ++i) dst_data[i] = static_cast<int32_t>(src_data[i]);
      } else if (target_type == GGML_TYPE_F32) {
        float* dst_data = static_cast<float*>(gt->data);
        for (size_t i = 0; i < nelem; ++i) dst_data[i] = static_cast<float>(src_data[i]);
      }
    }
    gt->op = GGML_OP_NONE;
    return gt;
  } else {
    return safe_ggml_cast(bc.ctx, src, target_type, &bc.host_acc);
  }
}

// ---------------------------------------------------------------------------
// WHERE
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_where(BuildContext& bc) {
  struct ggml_tensor* cond = bc.srcs[0];
  struct ggml_tensor* x = bc.srcs[1];
  struct ggml_tensor* y = bc.srcs[2];

  if (cond->type != GGML_TYPE_F32) {
    cond = safe_ggml_cast(bc.ctx, cond, GGML_TYPE_F32, &bc.host_acc);
  }
  cond = ggml_clamp(bc.ctx, cond, 0.0f, 1.0f);

  struct ggml_tensor* target = x;
  if (ggml_nelements(y) > ggml_nelements(target)) target = y;
  if (ggml_nelements(cond) > ggml_nelements(target)) target = cond;

  if (!ggml_are_same_shape(cond, target) && ggml_can_repeat(cond, target)) {
    cond = ggml_repeat(bc.ctx, cond, target);
  }
  if (!ggml_are_same_shape(x, target) && ggml_can_repeat(x, target)) {
    x = ggml_repeat(bc.ctx, x, target);
  }
  if (!ggml_are_same_shape(y, target) && ggml_can_repeat(y, target)) {
    y = ggml_repeat(bc.ctx, y, target);
  }

  struct ggml_tensor* one = make_f32_scalar(bc.ctx, 1.0f);
  struct ggml_tensor* one_rep = ggml_repeat(bc.ctx, one, cond);
  struct ggml_tensor* not_cond = ggml_sub(bc.ctx, one_rep, cond);

  struct ggml_tensor* x_part = ggml_mul(bc.ctx, x, cond);
  struct ggml_tensor* y_part = ggml_mul(bc.ctx, y, not_cond);
  return ggml_add(bc.ctx, x_part, y_part);
}

} // namespace executorch_ggml
