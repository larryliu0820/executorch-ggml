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

  // Eager host computation for I64 ARANGE (ggml_arange only supports F32)
  if (out_type == GGML_TYPE_I64 || out_type == GGML_TYPE_I32) {
    ggml_type eager_type = (out_type == GGML_TYPE_I64) ? GGML_TYPE_I64 : GGML_TYPE_I32;
    ggml_set_no_alloc(bc.ctx, false);
    auto* gt = ggml_new_tensor_4d(bc.ctx, eager_type,
        bc.ne[0], bc.ne[1], bc.ne[2], bc.ne[3]);
    ggml_set_no_alloc(bc.ctx, true);
    gt->op = GGML_OP_NONE;
    if (eager_type == GGML_TYPE_I64) {
      int64_t* d = static_cast<int64_t*>(gt->data);
      for (int64_t i = 0; i < nelem; i++) d[i] = (int64_t)(start + i * step);
    } else {
      int32_t* d = static_cast<int32_t*>(gt->data);
      for (int64_t i = 0; i < nelem; i++) d[i] = (int32_t)(start + i * step);
    }
    ggml_set_output(gt);
    return gt;
  }

  float stop = (float)(start + nelem * step);
  auto* gt = ggml_arange(bc.ctx, (float)start, stop, (float)step);
  ggml_set_output(gt);  // prevent scheduler buffer aliasing

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
  if (x->type != GGML_TYPE_F32) {
    x = safe_ggml_cast(bc.ctx, x, GGML_TYPE_F32, &bc.host_acc);
  }
  if (y->type != GGML_TYPE_F32) {
    y = safe_ggml_cast(bc.ctx, y, GGML_TYPE_F32, &bc.host_acc);
  }

  // Eager WHERE when all inputs have host data (avoids 0*inf=NaN in arithmetic fallback)
  const float* cd = static_cast<const float*>(bc.host_acc.get(cond));
  const float* xd = static_cast<const float*>(bc.host_acc.get(x));
  const float* yd = static_cast<const float*>(bc.host_acc.get(y));
  if (cd && xd && yd) {
    int64_t out_ne[4];
    for (int d = 0; d < 4; d++) {
      out_ne[d] = std::max({cond->ne[d], x->ne[d], y->ne[d]});
    }
    int64_t n = out_ne[0] * out_ne[1] * out_ne[2] * out_ne[3];
    if (n <= 10000000) {
      ggml_set_no_alloc(bc.ctx, false);
      auto* gt = ggml_new_tensor_4d(bc.ctx, GGML_TYPE_F32,
          out_ne[0], out_ne[1], out_ne[2], out_ne[3]);
      ggml_set_no_alloc(bc.ctx, true);
      gt->op = GGML_OP_NONE;
      float* od = static_cast<float*>(gt->data);
      for (int64_t i = 0; i < n; i++) {
        int64_t i0 = i % out_ne[0], rem = i / out_ne[0];
        int64_t i1 = rem % out_ne[1]; rem /= out_ne[1];
        int64_t i2 = rem % out_ne[2], i3 = rem / out_ne[2];
        auto bcast = [&](const struct ggml_tensor* t, int64_t i0_, int64_t i1_, int64_t i2_, int64_t i3_) {
          return (i0_ % t->ne[0]) + (i1_ % t->ne[1]) * t->ne[0]
               + (i2_ % t->ne[2]) * t->ne[0] * t->ne[1]
               + (i3_ % t->ne[3]) * t->ne[0] * t->ne[1] * t->ne[2];
        };
        float cv = cd[bcast(cond, i0, i1, i2, i3)];
        float xv = xd[bcast(x, i0, i1, i2, i3)];
        float yv = yd[bcast(y, i0, i1, i2, i3)];
        // Clamp -inf to -65504 to avoid NaN in downstream arithmetic
        if (xv < -65504.0f) xv = -65504.0f;
        if (yv < -65504.0f) yv = -65504.0f;
        od[i] = (cv != 0.0f) ? xv : yv;
      }
      return gt;
    }
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

  // Clamp x and y to avoid 0*inf=NaN in cond*x + (1-cond)*y
  x = ggml_clamp(bc.ctx, x, -65504.0f, 65504.0f);
  y = ggml_clamp(bc.ctx, y, -65504.0f, 65504.0f);

  struct ggml_tensor* one = make_f32_scalar(bc.ctx, 1.0f);
  struct ggml_tensor* one_rep = ggml_repeat(bc.ctx, one, cond);
  struct ggml_tensor* not_cond = ggml_sub(bc.ctx, one_rep, cond);

  struct ggml_tensor* x_part = ggml_mul(bc.ctx, x, cond);
  struct ggml_tensor* y_part = ggml_mul(bc.ctx, y, not_cond);
  return ggml_add(bc.ctx, x_part, y_part);
}

// ---------------------------------------------------------------------------
// REMAINDER
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_remainder(BuildContext& bc) {
  struct ggml_tensor* a = bc.srcs[0];
  struct ggml_tensor* b = (bc.srcs.size() > 1) ? bc.srcs[1] : nullptr;

  // Read scalar value from op_params if present (remainder.Scalar)
  double scalar = 0.0;
  int32_t is_scalar = 0;
  if (bc.ir_tensor->op_params() && bc.ir_tensor->op_params()->size() >= 12) {
    memcpy(&scalar, bc.ir_tensor->op_params()->data(), 8);
    memcpy(&is_scalar, bc.ir_tensor->op_params()->data() + 8, 4);
  }

  // Eager I64 remainder
  if (a->type == GGML_TYPE_I64) {
    const void* a_host = bc.host_acc.get(a);
    if (a_host) {
      int64_t n = ggml_nelements(a);
      int64_t divisor = is_scalar ? (int64_t)scalar : 0;
      if (!is_scalar && b->type == GGML_TYPE_I64) {
        const void* b_host = bc.host_acc.get(b);
        if (b_host) divisor = static_cast<const int64_t*>(b_host)[0];
      }
      if (divisor != 0) {
        ggml_set_no_alloc(bc.ctx, false);
        auto* gt = ggml_new_tensor(bc.ctx, GGML_TYPE_I64, GGML_MAX_DIMS, a->ne);
        ggml_set_no_alloc(bc.ctx, true);
        gt->op = GGML_OP_NONE;
        const int64_t* ad = static_cast<const int64_t*>(a_host);
        int64_t* od = static_cast<int64_t*>(gt->data);
        for (int64_t i = 0; i < n; i++) {
          // Python-style modulo: result has same sign as divisor
          int64_t r = ad[i] % divisor;
          if (r != 0 && ((r ^ divisor) < 0)) r += divisor;
          od[i] = r;
        }
        return gt;
      }
    }
  }

  // Eager I32 remainder
  if (a->type == GGML_TYPE_I32) {
    const void* a_host = bc.host_acc.get(a);
    if (a_host) {
      int64_t n = ggml_nelements(a);
      int32_t divisor = is_scalar ? (int32_t)scalar : 0;
      if (!is_scalar && b->type == GGML_TYPE_I32) {
        const void* b_host = bc.host_acc.get(b);
        if (b_host) divisor = static_cast<const int32_t*>(b_host)[0];
      }
      if (divisor != 0) {
        ggml_set_no_alloc(bc.ctx, false);
        auto* gt = ggml_new_tensor(bc.ctx, GGML_TYPE_I32, GGML_MAX_DIMS, a->ne);
        ggml_set_no_alloc(bc.ctx, true);
        gt->op = GGML_OP_NONE;
        const int32_t* ad = static_cast<const int32_t*>(a_host);
        int32_t* od = static_cast<int32_t*>(gt->data);
        for (int64_t i = 0; i < n; i++) {
          int32_t r = ad[i] % divisor;
          if (r != 0 && ((r ^ divisor) < 0)) r += divisor;
          od[i] = r;
        }
        return gt;
      }
    }
  }

  // F32 fallback: a - floor(a/b) * b
  if (a->type != GGML_TYPE_F32) a = safe_ggml_cast(bc.ctx, a, GGML_TYPE_F32, &bc.host_acc);
  struct ggml_tensor* b_f32;
  if (is_scalar) {
    b_f32 = ggml_repeat_4d(bc.ctx, make_f32_scalar(bc.ctx, (float)scalar),
                            a->ne[0], a->ne[1], a->ne[2], a->ne[3]);
  } else {
    b_f32 = b;
    if (b_f32->type != GGML_TYPE_F32) b_f32 = safe_ggml_cast(bc.ctx, b_f32, GGML_TYPE_F32, &bc.host_acc);
  }
  auto* quotient = ggml_floor(bc.ctx, ggml_div(bc.ctx, a, b_f32));
  return ggml_sub(bc.ctx, a, ggml_mul(bc.ctx, quotient, b_f32));
}

} // namespace executorch_ggml
