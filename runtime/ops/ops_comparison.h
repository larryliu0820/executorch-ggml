#pragma once

#include "build_context.h"
#include "custom_ops.h"

namespace executorch_ggml {

// ---------------------------------------------------------------------------
// EQ
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_eq(BuildContext& bc) {
  const auto* t = bc.ir_tensor;
  double scalar = 0.0;
  int32_t is_scalar = 0;
  if (t->op_params() && t->op_params()->size() >= 12) {
    memcpy(&scalar, t->op_params()->data(), 8);
    memcpy(&is_scalar, t->op_params()->data() + 8, 4);
  }
  struct ggml_tensor* a = bc.srcs[0];
  struct ggml_tensor* b;
  if (is_scalar) {
    b = ggml_repeat_4d(bc.ctx, make_f32_scalar(bc.ctx, (float)scalar),
                       a->ne[0], a->ne[1], a->ne[2], a->ne[3]);
  } else {
    b = bc.srcs[1];
  }
  if (bc.use_native_cmp_ops) {
    if (a->type != GGML_TYPE_F32) a = safe_ggml_cast(bc.ctx, a, GGML_TYPE_F32, &bc.host_acc);
    if (b->type != GGML_TYPE_F32) b = safe_ggml_cast(bc.ctx, b, GGML_TYPE_F32, &bc.host_acc);
    if (!resolve_broadcast(bc, a, b, "EQ")) return nullptr;
    struct ggml_tensor* diff = ggml_sub(bc.ctx, a, b);
    struct ggml_tensor* ne_val = ggml_clamp(bc.ctx, ggml_add(bc.ctx, ggml_step(bc.ctx, diff), ggml_step(bc.ctx, ggml_neg(bc.ctx, diff))), 0.0f, 1.0f);
    return ggml_add(bc.ctx, ggml_neg(bc.ctx, ne_val), make_f32_scalar(bc.ctx, 1.0f));
  } else {
    struct ggml_tensor* args[2] = {a, b};
    auto* gt = ggml_custom_4d(bc.ctx, GGML_TYPE_I32,
        std::max(a->ne[0], b->ne[0]), std::max(a->ne[1], b->ne[1]), std::max(a->ne[2], b->ne[2]), std::max(a->ne[3], b->ne[3]),
        args, 2, ggml_custom_eq, GGML_N_TASKS_MAX, nullptr);
    pin_to_cpu(bc, gt);
    return gt;
  }
}

// ---------------------------------------------------------------------------
// NE (not-equal)
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_ne(BuildContext& bc) {
  const auto* t = bc.ir_tensor;
  double scalar = 0.0;
  int32_t is_scalar = 1;
  if (t->op_params() && t->op_params()->size() >= 12) {
    memcpy(&scalar, t->op_params()->data(), 8);
    memcpy(&is_scalar, t->op_params()->data() + 8, 4);
  }
  struct ggml_tensor* a = bc.srcs[0];
  struct ggml_tensor* b;
  if (is_scalar) {
    b = ggml_repeat_4d(bc.ctx, make_f32_scalar(bc.ctx, (float)scalar),
                       a->ne[0], a->ne[1], a->ne[2], a->ne[3]);
  } else {
    b = bc.srcs[1];
  }
  if (bc.use_native_cmp_ops) {
    if (a->type != GGML_TYPE_F32) a = safe_ggml_cast(bc.ctx, a, GGML_TYPE_F32, &bc.host_acc);
    if (b->type != GGML_TYPE_F32) b = safe_ggml_cast(bc.ctx, b, GGML_TYPE_F32, &bc.host_acc);
    if (!resolve_broadcast(bc, a, b, "NE")) return nullptr;
    struct ggml_tensor* diff = ggml_sub(bc.ctx, a, b);
    return ggml_clamp(bc.ctx, ggml_add(bc.ctx, ggml_step(bc.ctx, diff), ggml_step(bc.ctx, ggml_neg(bc.ctx, diff))), 0.0f, 1.0f);
  } else {
    struct ggml_tensor* args[2] = {a, b};
    auto* gt = ggml_custom_4d(bc.ctx, GGML_TYPE_I32,
        std::max(a->ne[0], b->ne[0]), std::max(a->ne[1], b->ne[1]), std::max(a->ne[2], b->ne[2]), std::max(a->ne[3], b->ne[3]),
        args, 2, ggml_custom_ne_op, GGML_N_TASKS_MAX, nullptr);
    pin_to_cpu(bc, gt);
    return gt;
  }
}

// ---------------------------------------------------------------------------
// LE
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_le(BuildContext& bc) {
  const auto* t = bc.ir_tensor;
  struct ggml_tensor* a = bc.srcs[0];
  struct ggml_tensor* b;
  double le_scalar = 0.0;
  int32_t le_is_scalar = 0;
  if (t->op_params() && t->op_params()->size() >= 12) {
    memcpy(&le_scalar, t->op_params()->data(), 8);
    memcpy(&le_is_scalar, t->op_params()->data() + 8, 4);
  }
  if (le_is_scalar) {
    b = make_f32_scalar(bc.ctx, (float)le_scalar);
  } else {
    b = bc.srcs[1];
  }
  // Cast inputs to F32 for comparison.
  auto to_f32 = [&](struct ggml_tensor* x) -> struct ggml_tensor* {
    if (x->type == GGML_TYPE_F32) return x;
    if (x->type == GGML_TYPE_I64) {
      // I64 with host data: eager CPU conversion.
      if (x->data) {
        int64_t n = ggml_nelements(x);
        ggml_set_no_alloc(bc.ctx, false);
        auto* f32 = ggml_new_tensor(bc.ctx, GGML_TYPE_F32, GGML_MAX_DIMS, x->ne);
        ggml_set_no_alloc(bc.ctx, true);
        const int64_t* sd = static_cast<const int64_t*>(x->data);
        float* dd = static_cast<float*>(f32->data);
        for (int64_t i = 0; i < n; i++) dd[i] = (float)sd[i];
        return f32;
      }
      // I64 graph node without host data: ggml has no I64 support at all.
      // Create eager zeros (the comparison result doesn't affect MoE routing
      // because the fused MOE_FFN handles routing internally).
      ggml_set_no_alloc(bc.ctx, false);
      auto* f32 = ggml_new_tensor(bc.ctx, GGML_TYPE_F32, GGML_MAX_DIMS, x->ne);
      ggml_set_no_alloc(bc.ctx, true);
      memset(f32->data, 0, ggml_nbytes(f32));
      // Do NOT mark as input_derived — this is a constant zero tensor.
      return f32;
    }
    // I32, F16, BF16 etc: ggml_cast to F32 works natively
    return safe_ggml_cast(bc.ctx, x, GGML_TYPE_F32, &bc.host_acc);
  };
  a = to_f32(a);
  b = to_f32(b);
  if (auto* eager = try_eager_f32_binop(bc.ctx, a, b, 'L', bc.host_acc)) {
    return eager;
  }
  if (bc.use_native_cmp_ops) {
    if (!resolve_broadcast(bc, a, b, "LE")) return nullptr;
    return ggml_add(bc.ctx, ggml_neg(bc.ctx, ggml_step(bc.ctx, ggml_sub(bc.ctx, a, b))), make_f32_scalar(bc.ctx, 1.0f));
  } else {
    struct ggml_tensor* args[2] = {a, b};
    auto* gt = ggml_custom_4d(bc.ctx, GGML_TYPE_I32,
        std::max(a->ne[0], b->ne[0]), std::max(a->ne[1], b->ne[1]), std::max(a->ne[2], b->ne[2]), std::max(a->ne[3], b->ne[3]),
        args, 2, ggml_custom_le, GGML_N_TASKS_MAX, nullptr);
    pin_to_cpu(bc, gt);
    return gt;
  }
}

// ---------------------------------------------------------------------------
// LT
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_lt(BuildContext& bc) {
  struct ggml_tensor* a = bc.srcs[0];
  struct ggml_tensor* b;
  double lt_scalar = 0.0;
  int32_t lt_is_scalar = 0;
  if (bc.ir_tensor->op_params() && bc.ir_tensor->op_params()->size() >= 12) {
    memcpy(&lt_scalar, bc.ir_tensor->op_params()->data(), 8);
    memcpy(&lt_is_scalar, bc.ir_tensor->op_params()->data() + 8, 4);
  }
  if (lt_is_scalar) {
    b = make_f32_scalar(bc.ctx, (float)lt_scalar);
  } else {
    b = bc.srcs[1];
  }
  if (a->type != GGML_TYPE_F32) a = safe_ggml_cast(bc.ctx, a, GGML_TYPE_F32, &bc.host_acc);
  if (b->type != GGML_TYPE_F32) b = safe_ggml_cast(bc.ctx, b, GGML_TYPE_F32, &bc.host_acc);
  if (auto* eager = try_eager_f32_binop(bc.ctx, a, b, 'l', bc.host_acc)) {
    return eager;
  }
  if (bc.use_native_cmp_ops) {
    if (!resolve_broadcast(bc, a, b, "LT")) return nullptr;
    return ggml_step(bc.ctx, ggml_neg(bc.ctx, ggml_sub(bc.ctx, a, b)));
  } else {
    struct ggml_tensor* args[2] = {a, b};
    auto* gt = ggml_custom_4d(bc.ctx, GGML_TYPE_I32,
        std::max(a->ne[0], b->ne[0]), std::max(a->ne[1], b->ne[1]), std::max(a->ne[2], b->ne[2]), std::max(a->ne[3], b->ne[3]),
        args, 2, ggml_custom_lt, GGML_N_TASKS_MAX, nullptr);
    pin_to_cpu(bc, gt);
    return gt;
  }
}

// ---------------------------------------------------------------------------
// GT
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_gt(BuildContext& bc) {
  struct ggml_tensor* a = bc.srcs[0];
  struct ggml_tensor* b;
  double gt_scalar = 0.0;
  int32_t gt_is_scalar = 0;
  if (bc.ir_tensor->op_params() && bc.ir_tensor->op_params()->size() >= 12) {
    memcpy(&gt_scalar, bc.ir_tensor->op_params()->data(), 8);
    memcpy(&gt_is_scalar, bc.ir_tensor->op_params()->data() + 8, 4);
  }
  if (gt_is_scalar) {
    b = make_f32_scalar(bc.ctx, (float)gt_scalar);
  } else {
    b = bc.srcs[1];
  }
  if (a->type != GGML_TYPE_F32) a = safe_ggml_cast(bc.ctx, a, GGML_TYPE_F32, &bc.host_acc);
  if (b->type != GGML_TYPE_F32) b = safe_ggml_cast(bc.ctx, b, GGML_TYPE_F32, &bc.host_acc);
  if (auto* eager = try_eager_f32_binop(bc.ctx, a, b, 'G', bc.host_acc)) {
    return eager;
  }
  if (bc.use_native_cmp_ops) {
    if (!resolve_broadcast(bc, a, b, "GT")) return nullptr;
    return ggml_step(bc.ctx, ggml_sub(bc.ctx, a, b));
  } else {
    struct ggml_tensor* args[2] = {a, b};
    auto* gt = ggml_custom_4d(bc.ctx, GGML_TYPE_I32,
        std::max(a->ne[0], b->ne[0]), std::max(a->ne[1], b->ne[1]), std::max(a->ne[2], b->ne[2]), std::max(a->ne[3], b->ne[3]),
        args, 2, ggml_custom_gt, GGML_N_TASKS_MAX, nullptr);
    pin_to_cpu(bc, gt);
    return gt;
  }
}

// ---------------------------------------------------------------------------
// GE
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_ge(BuildContext& bc) {
  struct ggml_tensor* a = bc.srcs[0];
  struct ggml_tensor* b;
  double ge_scalar = 0.0;
  int32_t ge_is_scalar = 0;
  if (bc.ir_tensor->op_params() && bc.ir_tensor->op_params()->size() >= 12) {
    memcpy(&ge_scalar, bc.ir_tensor->op_params()->data(), 8);
    memcpy(&ge_is_scalar, bc.ir_tensor->op_params()->data() + 8, 4);
  }
  if (ge_is_scalar) {
    // Use scalar directly (not REPEAT) so try_eager_f32_binop can read its data.
    // Broadcasting is handled by modulo indexing in eager and resolve_broadcast in graph.
    b = make_f32_scalar(bc.ctx, (float)ge_scalar);
  } else {
    b = bc.srcs[1];
  }
  if (a->type != GGML_TYPE_F32) a = safe_ggml_cast(bc.ctx, a, GGML_TYPE_F32, &bc.host_acc);
  if (b->type != GGML_TYPE_F32) b = safe_ggml_cast(bc.ctx, b, GGML_TYPE_F32, &bc.host_acc);
  if (auto* eager = try_eager_f32_binop(bc.ctx, a, b, 'g', bc.host_acc)) {
    return eager;
  }
  if (bc.use_native_cmp_ops) {
    if (!resolve_broadcast(bc, a, b, "GE")) return nullptr;
    return ggml_add(bc.ctx, ggml_neg(bc.ctx, ggml_step(bc.ctx, ggml_neg(bc.ctx, ggml_sub(bc.ctx, a, b)))), make_f32_scalar(bc.ctx, 1.0f));
  } else {
    struct ggml_tensor* args[2] = {a, b};
    auto* gt = ggml_custom_4d(bc.ctx, GGML_TYPE_I32,
        std::max(a->ne[0], b->ne[0]), std::max(a->ne[1], b->ne[1]), std::max(a->ne[2], b->ne[2]), std::max(a->ne[3], b->ne[3]),
        args, 2, ggml_custom_ge, GGML_N_TASKS_MAX, nullptr);
    pin_to_cpu(bc, gt);
    return gt;
  }
}

// ---------------------------------------------------------------------------
// BITWISE_AND
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_bitwise_and(BuildContext& bc) {
  struct ggml_tensor* a = bc.srcs[0]; struct ggml_tensor* b = bc.srcs[1];
  if (a->type != GGML_TYPE_F32) a = safe_ggml_cast(bc.ctx, a, GGML_TYPE_F32, &bc.host_acc);
  if (b->type != GGML_TYPE_F32) b = safe_ggml_cast(bc.ctx, b, GGML_TYPE_F32, &bc.host_acc);
  if (auto* eager = try_eager_f32_binop(bc.ctx, a, b, '&', bc.host_acc)) {
    return eager;
  }
  if (bc.use_native_cmp_ops) {
    if (!resolve_broadcast(bc, a, b, "BITWISE_AND")) return nullptr;
    return ggml_mul(bc.ctx, a, b);
  } else {
    struct ggml_tensor* args[2] = {a, b};
    auto* gt = ggml_custom_4d(bc.ctx, GGML_TYPE_I32,
        std::max(a->ne[0], b->ne[0]), std::max(a->ne[1], b->ne[1]), std::max(a->ne[2], b->ne[2]), std::max(a->ne[3], b->ne[3]),
        args, 2, ggml_custom_bitwise_and, GGML_N_TASKS_MAX, nullptr);
    pin_to_cpu(bc, gt);
    return gt;
  }
}

// ---------------------------------------------------------------------------
// BITWISE_OR
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_bitwise_or(BuildContext& bc) {
  struct ggml_tensor* a = bc.srcs[0]; struct ggml_tensor* b = bc.srcs[1];
  if (bc.use_native_cmp_ops) {
    if (a->type != GGML_TYPE_F32) a = safe_ggml_cast(bc.ctx, a, GGML_TYPE_F32, &bc.host_acc);
    if (b->type != GGML_TYPE_F32) b = safe_ggml_cast(bc.ctx, b, GGML_TYPE_F32, &bc.host_acc);
    if (!resolve_broadcast(bc, a, b, "BITWISE_OR")) return nullptr;
    return ggml_clamp(bc.ctx, ggml_add(bc.ctx, a, b), 0.0f, 1.0f);
  } else {
    struct ggml_tensor* args[2] = {a, b};
    auto* gt = ggml_custom_4d(bc.ctx, GGML_TYPE_I32,
        std::max(a->ne[0], b->ne[0]), std::max(a->ne[1], b->ne[1]), std::max(a->ne[2], b->ne[2]), std::max(a->ne[3], b->ne[3]),
        args, 2, ggml_custom_bitwise_or, GGML_N_TASKS_MAX, nullptr);
    pin_to_cpu(bc, gt);
    return gt;
  }
}

// ---------------------------------------------------------------------------
// LOGICAL_NOT
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_logical_not(BuildContext& bc) {
  struct ggml_tensor* x = bc.srcs[0];
  if (bc.use_native_cmp_ops) {
    if (x->type != GGML_TYPE_F32) x = safe_ggml_cast(bc.ctx, x, GGML_TYPE_F32, &bc.host_acc);
    return ggml_add(bc.ctx, ggml_neg(bc.ctx, x), make_f32_scalar(bc.ctx, 1.0f));
  } else {
    struct ggml_tensor* args[1] = {x};
    auto* gt = ggml_custom_4d(bc.ctx, GGML_TYPE_I32,
        x->ne[0], x->ne[1], x->ne[2], x->ne[3],
        args, 1, ggml_custom_logical_not, GGML_N_TASKS_MAX, nullptr);
    pin_to_cpu(bc, gt);
    return gt;
  }
}

// ---------------------------------------------------------------------------
// ANY
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_any(BuildContext& bc) {
  const auto* t = bc.ir_tensor;
  int32_t dim = 0, ndim = 4;
  if (t->op_params() && t->op_params()->size() >= 8) {
    memcpy(&dim, t->op_params()->data(), 4);
    memcpy(&ndim, t->op_params()->data() + 4, 4);
  }
  struct ggml_tensor* src = bc.srcs[0];
  if (ndim > 4) ndim = 4;
  int ggml_axis = (ndim - 1) - dim;
  if (ggml_axis < 0) ggml_axis = 0;
  if (ggml_axis > 3) ggml_axis = 3;

  if (src->op == GGML_OP_NONE && src->data != nullptr) {
    // Eager path
    const void* src_host = bc.host_acc.get(src);
    int64_t any_ne[4] = {src->ne[0], src->ne[1], src->ne[2], src->ne[3]};
    any_ne[ggml_axis] = 1;
    ggml_set_no_alloc(bc.ctx, false);
    auto* gt = ggml_new_tensor_4d(bc.ctx, GGML_TYPE_F32, any_ne[0], any_ne[1], any_ne[2], any_ne[3]);
    ggml_set_no_alloc(bc.ctx, true);
    const size_t nelem_out = ggml_nelements(gt);
    float* out_data = static_cast<float*>(gt->data);

    int64_t stride = 1;
    for (int ax = 0; ax < ggml_axis; ++ax) stride *= src->ne[ax];
    int64_t dim_size = src->ne[ggml_axis];
    int64_t outer_size = nelem_out / stride;
    if (outer_size == 0) outer_size = 1;

    if (src->type == GGML_TYPE_F32) {
      const float* in_data = static_cast<const float*>(src_host);
      for (int64_t outer = 0; outer < outer_size; ++outer) {
        for (int64_t inner = 0; inner < stride; ++inner) {
          float any_true = 0.0f;
          for (int64_t d = 0; d < dim_size; ++d) {
            int64_t in_idx = outer * dim_size * stride + d * stride + inner;
            if (in_data[in_idx] != 0.0f) { any_true = 1.0f; break; }
          }
          out_data[outer * stride + inner] = any_true;
        }
      }
    } else {
      const int32_t* in_data = static_cast<const int32_t*>(src_host);
      for (int64_t outer = 0; outer < outer_size; ++outer) {
        for (int64_t inner = 0; inner < stride; ++inner) {
          float any_true = 0.0f;
          for (int64_t d = 0; d < dim_size; ++d) {
            int64_t in_idx = outer * dim_size * stride + d * stride + inner;
            if (in_data[in_idx] != 0) { any_true = 1.0f; break; }
          }
          out_data[outer * stride + inner] = any_true;
        }
      }
    }
    return gt;
  } else {
    // Graph-op path
    struct ggml_tensor* x = src;
    if (x->type != GGML_TYPE_F32) x = safe_ggml_cast(bc.ctx, x, GGML_TYPE_F32, &bc.host_acc);
    if (ggml_axis != 0) {
      int axes[4] = {0, 1, 2, 3};
      axes[0] = ggml_axis; axes[ggml_axis] = 0;
      x = ggml_cont(bc.ctx, safe_ggml_permute(bc.ctx, x, axes[0], axes[1], axes[2], axes[3], "ANY_pre"));
    }
    x = ggml_step(bc.ctx, ggml_sum_rows(bc.ctx, ggml_abs(bc.ctx, x)));
    if (ggml_axis != 0) {
      int axes[4] = {0, 1, 2, 3};
      axes[0] = ggml_axis; axes[ggml_axis] = 0;
      x = safe_ggml_permute(bc.ctx, x, axes[0], axes[1], axes[2], axes[3], "ANY_post");
    }
    return x;
  }
}

} // namespace executorch_ggml
