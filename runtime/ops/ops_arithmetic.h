#pragma once

#include "build_context.h"

namespace executorch_ggml {

// ---------------------------------------------------------------------------
// ADD
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_add(BuildContext& bc) {
  struct ggml_tensor* a = bc.srcs[0];
  struct ggml_tensor* b = bc.srcs[1];

  // Eager scalar path for enc_len-style computations.
  if (auto* eager = try_eager_scalar_binop(bc.ctx, a, b, '+', bc.host_acc)) {
    return eager;
  }
  // Eager F32 binop for streaming encoder ring buffer mask chain.
  if (auto* eager = try_eager_f32_binop(bc.ctx, a, b, '+', bc.host_acc)) {
    return eager;
  }

  // For int64 compile-time constants, use eager element-wise add.
  if (a->type == GGML_TYPE_I64 && b->type == GGML_TYPE_I64
      && a->op == GGML_OP_NONE && a->data != nullptr
      && b->op == GGML_OP_NONE && b->data != nullptr) {
    int64_t out_ne[4];
    for (int d = 0; d < 4; ++d) {
      out_ne[d] = std::max(a->ne[d], b->ne[d]);
    }
    const int64_t* a_data = static_cast<const int64_t*>(bc.host_acc.get(a));
    const int64_t* b_data = static_cast<const int64_t*>(bc.host_acc.get(b));
    ggml_set_no_alloc(bc.ctx, false);
    auto* gt = ggml_new_tensor_4d(bc.ctx, GGML_TYPE_I64, out_ne[0], out_ne[1], out_ne[2], out_ne[3]);
    ggml_set_no_alloc(bc.ctx, true);
    int64_t* out_data = static_cast<int64_t*>(gt->data);
    for (int64_t d3 = 0; d3 < out_ne[3]; ++d3) {
      for (int64_t d2 = 0; d2 < out_ne[2]; ++d2) {
        for (int64_t d1 = 0; d1 < out_ne[1]; ++d1) {
          for (int64_t d0 = 0; d0 < out_ne[0]; ++d0) {
            int64_t ai = (d0 % a->ne[0]) + (d1 % a->ne[1]) * a->ne[0]
                       + (d2 % a->ne[2]) * a->ne[0] * a->ne[1]
                       + (d3 % a->ne[3]) * a->ne[0] * a->ne[1] * a->ne[2];
            int64_t bi = (d0 % b->ne[0]) + (d1 % b->ne[1]) * b->ne[0]
                       + (d2 % b->ne[2]) * b->ne[0] * b->ne[1]
                       + (d3 % b->ne[3]) * b->ne[0] * b->ne[1] * b->ne[2];
            int64_t oi = d0 + d1 * out_ne[0] + d2 * out_ne[0] * out_ne[1]
                       + d3 * out_ne[0] * out_ne[1] * out_ne[2];
            out_data[oi] = a_data[ai] + b_data[bi];
          }
        }
      }
    }
    return gt;
  }
  // Non-constant I64/I32: cast to F32 (CUDA binary ops only support F32/F16/BF16).
  if (a->type == GGML_TYPE_I64) a = safe_ggml_cast(bc.ctx, a, GGML_TYPE_F32, &bc.host_acc);
  if (b->type == GGML_TYPE_I64) b = safe_ggml_cast(bc.ctx, b, GGML_TYPE_F32, &bc.host_acc);
  if (a->type == GGML_TYPE_I32) a = safe_ggml_cast(bc.ctx, a, GGML_TYPE_F32, &bc.host_acc);
  if (b->type == GGML_TYPE_I32) b = safe_ggml_cast(bc.ctx, b, GGML_TYPE_F32, &bc.host_acc);
  // Retry eager after I32/I64→F32 casts (inputs may now have F32 host data)
  if (auto* eager = try_eager_f32_binop(bc.ctx, a, b, '+', bc.host_acc)) {
    return eager;
  }

  if (bc.metal_f32_binops) { a = ensure_f32(bc.ctx, a); b = ensure_f32(bc.ctx, b); }
  if (bc.cuda_bf16_cast) {
    if (a->type == GGML_TYPE_BF16) a = ggml_cast(bc.ctx, a, GGML_TYPE_F32);
    if (b->type == GGML_TYPE_BF16) b = ggml_cast(bc.ctx, b, GGML_TYPE_F32);
  }
  a = ensure_cont(bc.ctx, a); b = ensure_cont(bc.ctx, b);
  a = fix_bf16_strides(bc.ctx, a); b = fix_bf16_strides(bc.ctx, b);

  if (a->type != b->type) {
    ggml_type tgt = (a->type == GGML_TYPE_F32 || b->type == GGML_TYPE_F32)
                        ? GGML_TYPE_F32
                        : a->type;
    if (a->type != tgt) a = safe_ggml_cast(bc.ctx, a, tgt, &bc.host_acc);
    if (b->type != tgt) b = safe_ggml_cast(bc.ctx, b, tgt, &bc.host_acc);
  }

  if (!resolve_broadcast(bc, a, b, "ADD")) {
    return nullptr;
  }

  return ggml_add(bc.ctx, a, b);
}

// ---------------------------------------------------------------------------
// SUB
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_sub(BuildContext& bc) {
  struct ggml_tensor* a = bc.srcs[0];
  struct ggml_tensor* b = bc.srcs[1];
  if (auto* eager = try_eager_scalar_binop(bc.ctx, a, b, '-', bc.host_acc)) {
    return eager;
  }
  if (auto* eager = try_eager_f32_binop(bc.ctx, a, b, '-', bc.host_acc)) {
    return eager;
  }
  // For int64 compile-time constants, use eager element-wise sub.
  if (a->type == GGML_TYPE_I64 && b->type == GGML_TYPE_I64
      && a->op == GGML_OP_NONE && a->data != nullptr
      && b->op == GGML_OP_NONE && b->data != nullptr) {
    int64_t out_ne[4];
    for (int d = 0; d < 4; ++d) {
      out_ne[d] = std::max(a->ne[d], b->ne[d]);
    }
    const int64_t* a_data = static_cast<const int64_t*>(bc.host_acc.get(a));
    const int64_t* b_data = static_cast<const int64_t*>(bc.host_acc.get(b));
    ggml_set_no_alloc(bc.ctx, false);
    auto* gt = ggml_new_tensor_4d(bc.ctx, GGML_TYPE_I64, out_ne[0], out_ne[1], out_ne[2], out_ne[3]);
    ggml_set_no_alloc(bc.ctx, true);
    int64_t* out_data = static_cast<int64_t*>(gt->data);
    for (int64_t d3 = 0; d3 < out_ne[3]; ++d3) {
      for (int64_t d2 = 0; d2 < out_ne[2]; ++d2) {
        for (int64_t d1 = 0; d1 < out_ne[1]; ++d1) {
          for (int64_t d0 = 0; d0 < out_ne[0]; ++d0) {
            int64_t ai = (d0 % a->ne[0]) + (d1 % a->ne[1]) * a->ne[0]
                       + (d2 % a->ne[2]) * a->ne[0] * a->ne[1]
                       + (d3 % a->ne[3]) * a->ne[0] * a->ne[1] * a->ne[2];
            int64_t bi = (d0 % b->ne[0]) + (d1 % b->ne[1]) * b->ne[0]
                       + (d2 % b->ne[2]) * b->ne[0] * b->ne[1]
                       + (d3 % b->ne[3]) * b->ne[0] * b->ne[1] * b->ne[2];
            int64_t oi = d0 + d1 * out_ne[0] + d2 * out_ne[0] * out_ne[1]
                       + d3 * out_ne[0] * out_ne[1] * out_ne[2];
            out_data[oi] = a_data[ai] - b_data[bi];
          }
        }
      }
    }
    return gt;
  } else {
    // Non-constant I64/I32: cast to F32.
    if (a->type == GGML_TYPE_I64) a = safe_ggml_cast(bc.ctx, a, GGML_TYPE_F32, &bc.host_acc);
    if (b->type == GGML_TYPE_I64) b = safe_ggml_cast(bc.ctx, b, GGML_TYPE_F32, &bc.host_acc);
    if (a->type == GGML_TYPE_I32) a = safe_ggml_cast(bc.ctx, a, GGML_TYPE_F32, &bc.host_acc);
    if (b->type == GGML_TYPE_I32) b = safe_ggml_cast(bc.ctx, b, GGML_TYPE_F32, &bc.host_acc);
    // Retry eager after I32/I64→F32 casts
    if (auto* eager = try_eager_f32_binop(bc.ctx, a, b, '-', bc.host_acc)) {
      return eager;
    }
    if (bc.metal_f32_binops) { a = ensure_f32(bc.ctx, a); b = ensure_f32(bc.ctx, b); }
    if (bc.cuda_bf16_cast) {
      if (a->type == GGML_TYPE_BF16) a = ggml_cast(bc.ctx, a, GGML_TYPE_F32);
      if (b->type == GGML_TYPE_BF16) b = ggml_cast(bc.ctx, b, GGML_TYPE_F32);
    }
    a = ensure_cont(bc.ctx, a); b = ensure_cont(bc.ctx, b);
    a = fix_bf16_strides(bc.ctx, a); b = fix_bf16_strides(bc.ctx, b);

    if (!resolve_broadcast(bc, a, b, "SUB")) {
      return nullptr;
    }
    return ggml_sub(bc.ctx, a, b);
  }
}

// ---------------------------------------------------------------------------
// MUL (element-wise, with SiLU-gate fusion)
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_mul(BuildContext& bc) {
  struct ggml_tensor* a = bc.srcs[0];
  struct ggml_tensor* b = bc.srcs[1];
  // SiLU-gate AOT fusion: MUL(SILU(gate), up) -> swiglu_split(gate, up)
  {
    struct ggml_tensor* gate = nullptr;
    struct ggml_tensor* up = nullptr;
    for (int si = 0; si < 2; si++) {
      struct ggml_tensor* s = (si == 0) ? a : b;
      if (s->op == GGML_OP_MUL && s->src[0] && s->src[1]) {
        struct ggml_tensor* sigmoid_t = nullptr;
        struct ggml_tensor* x_t = nullptr;
        for (int j = 0; j < 2; j++) {
          if (s->src[j]->op == GGML_OP_UNARY &&
              ggml_get_unary_op(s->src[j]) == GGML_UNARY_OP_SIGMOID) {
            sigmoid_t = s->src[j];
            x_t = s->src[1-j];
            break;
          }
        }
        if (sigmoid_t && x_t && sigmoid_t->src[0] == x_t) {
          gate = x_t;
          up = (si == 0) ? b : a;
          break;
        }
      }
    }
    if (gate && up) {
      if (bc.verbose) fprintf(stderr, "[ggml_backend] SiLU-gate fusion (swiglu_split): gate ne=[%lld,%lld] up ne=[%lld,%lld]\n",
          (long long)gate->ne[0], (long long)gate->ne[1],
          (long long)up->ne[0], (long long)up->ne[1]);
      struct ggml_tensor* g = ensure_cont(bc.ctx, gate);
      struct ggml_tensor* u = ensure_cont(bc.ctx, up);
      if (bc.metal_f32_binops) { g = ensure_f32(bc.ctx, g); u = ensure_f32(bc.ctx, u); }
      if (bc.cuda_bf16_cast) {
        if (g->type == GGML_TYPE_BF16) g = ggml_cast(bc.ctx, g, GGML_TYPE_F32);
        if (u->type == GGML_TYPE_BF16) u = ggml_cast(bc.ctx, u, GGML_TYPE_F32);
      }
      return ggml_swiglu_split(bc.ctx, g, u);
    }
  }
  if (auto* eager = try_eager_scalar_binop(bc.ctx, a, b, '*', bc.host_acc)) {
    return eager;
  }
  if (auto* eager = try_eager_f32_binop(bc.ctx, a, b, '*', bc.host_acc)) {
    return eager;
  }
  // I64 eager MUL when both have host data
  if (a->type == GGML_TYPE_I64 && b->type == GGML_TYPE_I64) {
    const void* a_host = bc.host_acc.get(a);
    const void* b_host = bc.host_acc.get(b);
    if (a_host && b_host) {
      int64_t out_ne[4];
      for (int d = 0; d < 4; d++) out_ne[d] = std::max(a->ne[d], b->ne[d]);
      const int64_t* ad = static_cast<const int64_t*>(a_host);
      const int64_t* bd = static_cast<const int64_t*>(b_host);
      int64_t na = ggml_nelements(a), nb = ggml_nelements(b);
      ggml_set_no_alloc(bc.ctx, false);
      auto* gt = ggml_new_tensor_4d(bc.ctx, GGML_TYPE_I64, out_ne[0], out_ne[1], out_ne[2], out_ne[3]);
      ggml_set_no_alloc(bc.ctx, true);
      gt->op = GGML_OP_NONE;
      int64_t* od = static_cast<int64_t*>(gt->data);
      int64_t n = ggml_nelements(gt);
      for (int64_t i = 0; i < n; i++) od[i] = ad[i % na] * bd[i % nb];
      return gt;
    }
  }
  // I32/I64 -> F32 casts BEFORE ggml_scale path (CUDA requires F32 for scale)
  if (a->type == GGML_TYPE_I32) a = safe_ggml_cast(bc.ctx, a, GGML_TYPE_F32, &bc.host_acc);
  if (b->type == GGML_TYPE_I32) b = safe_ggml_cast(bc.ctx, b, GGML_TYPE_F32, &bc.host_acc);
  if (a->type == GGML_TYPE_I64) a = safe_ggml_cast(bc.ctx, a, GGML_TYPE_F32, &bc.host_acc);
  if (b->type == GGML_TYPE_I64) b = safe_ggml_cast(bc.ctx, b, GGML_TYPE_F32, &bc.host_acc);
  // Retry eager after I32/I64→F32 casts
  if (auto* eager = try_eager_f32_binop(bc.ctx, a, b, '*', bc.host_acc)) {
    return eager;
  }
  // ggml_scale for non-input-derived scalars
  if (ggml_nelements(b) == 1 && b->data && b->type == GGML_TYPE_F32
      && !bc.input_derived.count(b)) {
    return ggml_scale(bc.ctx, ggml_cont(bc.ctx, a), bc.host_acc.read_f32(b));
  }
  if (ggml_nelements(a) == 1 && a->data && a->type == GGML_TYPE_F32
      && !bc.input_derived.count(a)) {
    return ggml_scale(bc.ctx, ggml_cont(bc.ctx, b), bc.host_acc.read_f32(a));
  }
  if (ggml_nelements(a) < ggml_nelements(b)) {
    std::swap(a, b);
  }
  if (bc.metal_f32_binops) { a = ensure_f32(bc.ctx, a); b = ensure_f32(bc.ctx, b); }
  if (bc.cuda_bf16_cast) {
    if (a->type == GGML_TYPE_BF16) a = ggml_cast(bc.ctx, a, GGML_TYPE_F32);
    if (b->type == GGML_TYPE_BF16) b = ggml_cast(bc.ctx, b, GGML_TYPE_F32);
  }
  a = ensure_cont(bc.ctx, a); b = ensure_cont(bc.ctx, b);
  a = fix_bf16_strides(bc.ctx, a); b = fix_bf16_strides(bc.ctx, b);
  if (!resolve_broadcast(bc, a, b, "MUL")) {
    return nullptr;
  }
  return ggml_mul(bc.ctx, a, b);
}

// ---------------------------------------------------------------------------
// MUL_SCALAR
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_mul_scalar(BuildContext& bc) {
  float scalar = 1.0f;
  if (bc.ir_tensor->op_params() && bc.ir_tensor->op_params()->size() >= 4) {
    memcpy(&scalar, bc.ir_tensor->op_params()->data(), sizeof(float));
  }
  return ggml_scale(bc.ctx, ggml_cont(bc.ctx, bc.srcs[0]), scalar);
}

// ---------------------------------------------------------------------------
// DIV
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_div(BuildContext& bc) {
  struct ggml_tensor* a = bc.srcs[0];
  struct ggml_tensor* b = bc.srcs[1];

  // Read rounding_mode from op_params: 0=trunc (default), 1=floor
  int32_t rounding_mode = 0;
  if (bc.ir_tensor->op_params() && bc.ir_tensor->op_params()->size() >= 4) {
    memcpy(&rounding_mode, bc.ir_tensor->op_params()->data(), sizeof(int32_t));
  }

  if (auto* eager = try_eager_scalar_binop(bc.ctx, a, b, '/', bc.host_acc)) {
    return eager;
  }
  if (auto* eager = try_eager_f32_binop(bc.ctx, a, b, '/', bc.host_acc)) {
    // Apply floor if needed
    if (rounding_mode == 1 && eager) {
      float* d = static_cast<float*>(eager->data);
      int64_t n = ggml_nelements(eager);
      for (int64_t i = 0; i < n; i++) d[i] = std::floor(d[i]);
    }
    return eager;
  }
  // I64 eager DIV with floor-div support
  if (a->type == GGML_TYPE_I64 && b->type == GGML_TYPE_I64) {
    const void* a_host = bc.host_acc.get(a);
    const void* b_host = bc.host_acc.get(b);
    if (a_host && b_host) {
      int64_t out_ne[4];
      for (int d = 0; d < 4; d++) out_ne[d] = std::max(a->ne[d], b->ne[d]);
      const int64_t* ad = static_cast<const int64_t*>(a_host);
      const int64_t* bd = static_cast<const int64_t*>(b_host);
      int64_t na = ggml_nelements(a), nb = ggml_nelements(b);
      ggml_set_no_alloc(bc.ctx, false);
      auto* gt = ggml_new_tensor_4d(bc.ctx, GGML_TYPE_I64, out_ne[0], out_ne[1], out_ne[2], out_ne[3]);
      ggml_set_no_alloc(bc.ctx, true);
      gt->op = GGML_OP_NONE;
      int64_t* od = static_cast<int64_t*>(gt->data);
      int64_t n = ggml_nelements(gt);
      for (int64_t i = 0; i < n; i++) {
        int64_t dv = bd[i % nb];
        if (dv == 0) { od[i] = 0; continue; }
        int64_t q = ad[i % na] / dv;
        if (rounding_mode == 1) {  // Python floor division
          int64_t r = ad[i % na] % dv;
          if (r != 0 && ((r ^ dv) < 0)) q--;
        }
        od[i] = q;
      }
      return gt;
    }
  }
  // I32/I64 -> F32 casts
  if (a->type == GGML_TYPE_I32) a = safe_ggml_cast(bc.ctx, a, GGML_TYPE_F32, &bc.host_acc);
  if (b->type == GGML_TYPE_I32) b = safe_ggml_cast(bc.ctx, b, GGML_TYPE_F32, &bc.host_acc);
  if (a->type == GGML_TYPE_I64) a = safe_ggml_cast(bc.ctx, a, GGML_TYPE_F32, &bc.host_acc);
  if (b->type == GGML_TYPE_I64) b = safe_ggml_cast(bc.ctx, b, GGML_TYPE_F32, &bc.host_acc);
  // Retry eager after I32/I64→F32 casts
  if (auto* eager = try_eager_f32_binop(bc.ctx, a, b, '/', bc.host_acc)) {
    if (rounding_mode == 1 && eager) {
      float* d = static_cast<float*>(eager->data);
      int64_t n = ggml_nelements(eager);
      for (int64_t i = 0; i < n; i++) d[i] = std::floor(d[i]);
    }
    return eager;
  }
  if (bc.metal_f32_binops) { a = ensure_f32(bc.ctx, a); b = ensure_f32(bc.ctx, b); }
  if (bc.cuda_bf16_cast) {
    if (a->type == GGML_TYPE_BF16) a = ggml_cast(bc.ctx, a, GGML_TYPE_F32);
    if (b->type == GGML_TYPE_BF16) b = ggml_cast(bc.ctx, b, GGML_TYPE_F32);
  }
  a = ensure_cont(bc.ctx, a); b = ensure_cont(bc.ctx, b);
  a = fix_bf16_strides(bc.ctx, a); b = fix_bf16_strides(bc.ctx, b);

  if (!resolve_broadcast(bc, a, b, "DIV")) {
    return nullptr;
  }
  auto* result = ggml_div(bc.ctx, a, b);
  // Apply floor rounding if needed (aten.div.Tensor_mode with mode='floor')
  if (rounding_mode == 1) {
    result = ggml_floor(bc.ctx, result);
  }
  return result;
}

// ---------------------------------------------------------------------------
// NEG
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_neg(BuildContext& bc) {
  return ggml_neg(bc.ctx, bc.srcs[0]);
}

// ---------------------------------------------------------------------------
// RSQRT
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_rsqrt(BuildContext& bc) {
  struct ggml_tensor* sx  = ggml_sqrt(bc.ctx, bc.srcs[0]);
  struct ggml_tensor* one = make_f32_scalar(bc.ctx, 1.0f);
  struct ggml_tensor* one_rep = ggml_repeat(bc.ctx, one, sx);
  return ggml_div(bc.ctx, one_rep, sx);
}

// ---------------------------------------------------------------------------
// POW
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_pow(BuildContext& bc) {
  float exponent = 2.0f;
  if (bc.ir_tensor->op_params() && bc.ir_tensor->op_params()->size() >= 4) {
    memcpy(&exponent, bc.ir_tensor->op_params()->data(), sizeof(float));
  }
  if (exponent == 2.0f) {
    return ggml_sqr(bc.ctx, ensure_cont(bc.ctx, bc.srcs[0]));
  } else if (exponent == 0.5f) {
    return ggml_sqrt(bc.ctx, ensure_cont(bc.ctx, bc.srcs[0]));
  } else {
    struct ggml_tensor* log_x = ggml_log(bc.ctx, bc.srcs[0]);
    struct ggml_tensor* scaled = ggml_scale(bc.ctx, ggml_cont(bc.ctx, log_x), exponent);
    return ggml_exp(bc.ctx, scaled);
  }
}

// ---------------------------------------------------------------------------
// MEAN
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_mean(BuildContext& bc) {
  const auto* t = bc.ir_tensor;
  int32_t ndims = 0;
  int32_t dims[4] = {0, 0, 0, 0};
  if (t->op_params() && t->op_params()->size() >= 4) {
    const uint8_t* data = t->op_params()->data();
    memcpy(&ndims, data, sizeof(int32_t));
    if (ndims < 0 || ndims > 4) {
      return nullptr;
    }
    if (t->op_params()->size() < static_cast<size_t>(4 + ndims * 4)) {
      return nullptr;
    }
    for (int i = 0; i < ndims; ++i) {
      memcpy(&dims[i], data + 4 + i * 4, sizeof(int32_t));
    }
  }

  if (ndims == 2 && dims[0] == 2 && dims[1] == 3) {
    const int k0 = static_cast<int>(bc.srcs[0]->ne[0]);
    const int k1 = static_cast<int>(bc.srcs[0]->ne[1]);
    return ggml_pool_2d(
        bc.ctx,
        bc.srcs[0],
        GGML_OP_POOL_AVG,
        k0, k1, k0, k1,
        0.0f, 0.0f);
  } else {
    return ggml_mean(bc.ctx, bc.srcs[0]);
  }
}

} // namespace executorch_ggml
