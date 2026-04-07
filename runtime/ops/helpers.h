#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>

#include <ggml.h>
#include <ggml-backend.h>

#include "host_data_accessor.h"
#include "sym_expr.h"

// Include the flatc-generated IR header for ggml_ir types (Tensor, etc.).
// This provides the full definitions needed by resolve_ir_shape and other helpers.
#include "ggml_ir_generated.h"

namespace executorch_ggml {

// Ensure tensor is contiguous. No-op if already contiguous.
static inline struct ggml_tensor* ensure_cont(struct ggml_context* ctx, struct ggml_tensor* t) {
  return ggml_is_contiguous(t) ? t : ggml_cont(ctx, t);
}

// Fix BF16 tensor stride alignment issues. GGML expects nb values to be aligned to element size.
static inline struct ggml_tensor* fix_bf16_strides(struct ggml_context* ctx, struct ggml_tensor* t) {
  if (t->type != GGML_TYPE_BF16) return t;

  const size_t elem_size = sizeof(ggml_bf16_t); // 2 bytes

  // Check if any stride is misaligned
  bool needs_fix = false;
  for (int i = 0; i < GGML_MAX_DIMS; i++) {
    if (t->nb[i] % elem_size != 0) {
      needs_fix = true;
      break;
    }
  }

  if (!needs_fix) return t;

  // Force contiguous layout to fix stride alignment
  return ggml_cont(ctx, t);
}

// Skip RESHAPE/VIEW/CONT/PERMUTE wrappers to find the "real" op underneath.
static struct ggml_tensor* unwrap_views(struct ggml_tensor* t) {
  while (t && (t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW
               || t->op == GGML_OP_CONT || t->op == GGML_OP_TRANSPOSE
               || t->op == GGML_OP_PERMUTE)) {
    t = t->src[0];
  }
  return t;
}

// Safe wrapper for ggml_view_4d that adds bounds checking to prevent assertion failures.
// Returns the view tensor on success, or nullptr if the view would exceed bounds.
static struct ggml_tensor* safe_ggml_view_4d(
    struct ggml_context* ctx,
    struct ggml_tensor* src,
    int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
    size_t nb1, size_t nb2, size_t nb3,
    size_t offset) {

  // Calculate the data size needed for this view
  size_t data_size = ggml_row_size(src->type, ne0);
  data_size *= ne1 * ne2 * ne3;

  // Check bounds: data_size + offset <= total source tensor size
  size_t src_total_bytes = ggml_nbytes(src);
  if (offset > src_total_bytes || data_size > src_total_bytes - offset) {
    fprintf(stderr, "[GGML_VIEW_4D_ERROR] View bounds check failed:\n");
    fprintf(stderr, "  Source tensor: %ld bytes\n", src_total_bytes);
    fprintf(stderr, "  View offset: %ld bytes\n", offset);
    fprintf(stderr, "  View data size: %ld bytes\n", data_size);
    fprintf(stderr, "  Required total: %ld bytes (exceeds source by %ld)\n",
            offset + data_size, (offset + data_size) - src_total_bytes);
    fprintf(stderr, "  View shape: [%ld, %ld, %ld, %ld]\n", ne0, ne1, ne2, ne3);
    fprintf(stderr, "  Source shape: [%ld, %ld, %ld, %ld]\n",
            src->ne[0], src->ne[1], src->ne[2], src->ne[3]);
    return nullptr;
  }

  return ggml_view_4d(ctx, src, ne0, ne1, ne2, ne3, nb1, nb2, nb3, offset);
}

// Check if tensor t is rotate_half(x): CONCAT(NEG(x_first_half), x_second_half)
// Returns the original x (before the views), or nullptr if pattern doesn't match.
static struct ggml_tensor* match_rotate_half(struct ggml_tensor* t) {
  auto* core = unwrap_views(t);
  if (!core || core->op != GGML_OP_CONCAT) return nullptr;

  // CONCAT has two branches. One should contain UNARY(NEG), the other should not.
  auto* branch0 = unwrap_views(core->src[0]);
  auto* branch1 = unwrap_views(core->src[1]);
  if (!branch0 || !branch1) return nullptr;

  static int rope_debug = -1;
  if (rope_debug < 0) { const char* e = std::getenv("GGML_ROPE_DEBUG"); rope_debug = (e && *e != '0') ? 1 : 0; }
  if (rope_debug) {
    fprintf(stderr, "[rope_fuse]   CONCAT branch0=%d branch1=%d\n", branch0->op, branch1->op);
  }

  // Find which branch has the NEG (UNARY op)
  struct ggml_tensor* neg_branch = nullptr;
  struct ggml_tensor* pos_branch = nullptr;
  if (branch0->op == GGML_OP_UNARY) {
    neg_branch = branch0; pos_branch = branch1;
  } else if (branch1->op == GGML_OP_UNARY) {
    neg_branch = branch1; pos_branch = branch0;
  } else {
    return nullptr;
  }

  // NEG branch: UNARY(NEG) -> its src[0] traces back to a VIEW of x
  auto* neg_src = neg_branch->src[0];
  auto* neg_inner = unwrap_views(neg_src);

  // POS branch: traces back to a VIEW of the same x
  auto* pos_inner = pos_branch;
  if (pos_inner->op != GGML_OP_VIEW)
    pos_inner = unwrap_views(pos_inner);

  if (rope_debug) {
    fprintf(stderr, "[rope_fuse]   neg_inner=%d pos_inner=%d\n",
            neg_inner ? neg_inner->op : -1, pos_inner ? pos_inner->op : -1);
  }

  // Both should be VIEWs of the same tensor. But they might be VIEWs or
  // the original tensor itself (if no VIEW was needed).
  // Trace both back to the root non-view tensor.
  auto* x_from_neg = neg_inner;
  while (x_from_neg && (x_from_neg->op == GGML_OP_VIEW || x_from_neg->op == GGML_OP_RESHAPE
         || x_from_neg->op == GGML_OP_CONT || x_from_neg->op == GGML_OP_TRANSPOSE
         || x_from_neg->op == GGML_OP_PERMUTE))
    x_from_neg = x_from_neg->src[0];
  auto* x_from_pos = pos_inner;
  while (x_from_pos && (x_from_pos->op == GGML_OP_VIEW || x_from_pos->op == GGML_OP_RESHAPE
         || x_from_pos->op == GGML_OP_CONT || x_from_pos->op == GGML_OP_TRANSPOSE
         || x_from_pos->op == GGML_OP_PERMUTE))
    x_from_pos = x_from_pos->src[0];

  if (rope_debug) {
    fprintf(stderr, "[rope_fuse]   x_from_neg=%p (op=%d) x_from_pos=%p (op=%d)\n",
            (void*)x_from_neg, x_from_neg ? x_from_neg->op : -1,
            (void*)x_from_pos, x_from_pos ? x_from_pos->op : -1);
  }
  if (!x_from_neg || x_from_neg != x_from_pos) return nullptr;

  return x_from_neg;
}

// Trace back from a COS/SIN tensor to find the position tensor and inv_freq,
// then recover freq_base. Returns true on success.
// Pattern: COS/SIN <- (CONT <- SCALE <-)? CONCAT <- PERMUTE <- MUL_MAT(pos_scalar, inv_freq)
static bool extract_rope_params(
    struct ggml_tensor* cos_or_sin,
    struct ggml_tensor** out_pos,
    float* out_freq_base,
    int* out_n_dims,
    HostDataAccessor& acc) {
  // Skip through SCALE/CONT/RESHAPE wrappers after COS/SIN
  auto* t = cos_or_sin;
  if (!t) return false;

  // t should be COS or SIN -- its src[0] is the angle tensor
  if (t->op != GGML_OP_COS && t->op != GGML_OP_SIN) {
    t = unwrap_views(t);
    if (!t || (t->op != GGML_OP_COS && t->op != GGML_OP_SIN)) return false;
  }

  // angle tensor: CONCAT of two halves, or direct from MUL_MAT
  auto* angles = unwrap_views(t->src[0]);
  if (!angles) return false;

  static int rope_debug = -1;
  if (rope_debug < 0) { const char* e = std::getenv("GGML_ROPE_DEBUG"); rope_debug = (e && *e != '0') ? 1 : 0; }
  if (rope_debug) fprintf(stderr, "[rope_params] angles op=%d\n", angles->op);

  // May be CONCAT (doubling for NeoX style) -- unwrap to find MUL_MAT
  if (angles->op == GGML_OP_CONCAT) {
    angles = unwrap_views(angles->src[0]);  // first half
    if (!angles) return false;
    if (rope_debug) fprintf(stderr, "[rope_params] after CONCAT unwrap: op=%d\n", angles->op);
  }

  // Trace through remaining ops to find MUL_MAT
  auto* mul_mat = angles;
  int depth = 0;
  while (mul_mat && mul_mat->op != GGML_OP_MUL_MAT && depth < 20) {
    if (rope_debug) fprintf(stderr, "[rope_params]   trace[%d] op=%d\n", depth, mul_mat->op);
    mul_mat = mul_mat->src[0];
    depth++;
  }
  if (!mul_mat || mul_mat->op != GGML_OP_MUL_MAT) {
    if (rope_debug) fprintf(stderr, "[rope_params] FAIL: MUL_MAT not found after %d hops\n", depth);
    return false;
  }
  if (rope_debug) fprintf(stderr, "[rope_params] MUL_MAT found at depth %d\n", depth);

  // MUL_MAT: one src is inv_freq (constant with data), the other is position
  struct ggml_tensor* inv_freq = mul_mat->src[1];
  struct ggml_tensor* pos_tensor = mul_mat->src[0];
  // Swap if needed: inv_freq is the one with data (constant leaf)
  if ((!inv_freq || !unwrap_views(inv_freq) || !unwrap_views(inv_freq)->data)
      && unwrap_views(pos_tensor) && unwrap_views(pos_tensor)->data) {
    std::swap(inv_freq, pos_tensor);
  }

  if (rope_debug) {
    fprintf(stderr, "[rope_params] MUL_MAT src0 op=%d data=%p ne=[%lld,%lld] src1 op=%d data=%p ne=[%lld,%lld]\n",
            inv_freq?inv_freq->op:-1, inv_freq?inv_freq->data:nullptr,
            inv_freq?(long long)inv_freq->ne[0]:0, inv_freq?(long long)inv_freq->ne[1]:0,
            pos_tensor?pos_tensor->op:-1, pos_tensor?pos_tensor->data:nullptr,
            pos_tensor?(long long)pos_tensor->ne[0]:0, pos_tensor?(long long)pos_tensor->ne[1]:0);
  }

  // inv_freq should be a constant leaf with data
  auto* inv_core = unwrap_views(inv_freq);
  if (!inv_core || !inv_core->data) {
    if (rope_debug) fprintf(stderr, "[rope_params] FAIL: inv_freq has no data (inv_core=%p op=%d)\n",
        (void*)inv_core, inv_core?inv_core->op:-1);
    return false;
  }

  // Read inv_freq values to extract freq_base
  int64_t n_freq = ggml_nelements(inv_core);
  if (n_freq < 2) return false;

  const float* freq_data = static_cast<const float*>(acc.get(inv_core));
  if (!freq_data) return false;

  // inv_freq[i] = 1 / (freq_base ^ (2i / n_dims))
  // inv_freq[0] = 1.0 (always), inv_freq[1] = 1/freq_base^(2/n_dims)
  // freq_base = (1/inv_freq[1]) ^ (n_dims/2)
  int n_dims = (int)(n_freq * 2);  // n_freq = n_dims/2
  if (freq_data[1] > 0.0f && freq_data[1] < 1.0f) {
    double ratio = 1.0 / (double)freq_data[1];
    *out_freq_base = (float)std::pow(ratio, (double)n_dims / 2.0);
  } else {
    *out_freq_base = 10000.0f;  // fallback
  }
  *out_n_dims = n_dims;

  // Trace pos_tensor back to the original position input
  // It may be wrapped in CONT/TRANSPOSE/RESHAPE/CPY
  auto* pos = pos_tensor;
  while (pos && pos->op != GGML_OP_NONE) {
    if (pos->op == GGML_OP_CPY) {
      pos = pos->src[1];  // CPY copies src[0] into src[1]'s shape
    } else {
      pos = pos->src[0];
    }
  }
  if (!pos) return false;
  *out_pos = pos;
  return true;
}

// Try to fuse ADD(MUL(x, cos), MUL(rotate_half(x), sin)) into ggml_rope_ext.
// Called from the ADD handler in Phase B. Returns fused tensor or nullptr.
static struct ggml_tensor* try_fuse_rope(
    struct ggml_context* ctx,
    struct ggml_tensor* a,
    struct ggml_tensor* b,
    HostDataAccessor& acc) {
  // Both sides should be MUL
  auto* a_core = unwrap_views(a);
  auto* b_core = unwrap_views(b);
  if (!a_core || !b_core) return nullptr;
  if (a_core->op != GGML_OP_MUL || b_core->op != GGML_OP_MUL) return nullptr;
  static int rope_debug = -1;
  if (rope_debug < 0) { const char* e = std::getenv("GGML_ROPE_DEBUG"); rope_debug = (e && *e != '0') ? 1 : 0; }
  if (rope_debug) fprintf(stderr, "[rope_fuse] ADD(MUL, MUL) candidate found\n");

  // One MUL is x*cos, the other is rotate_half(x)*sin.
  // Identify which is which by checking for rotate_half pattern.
  struct ggml_tensor* x_cos_mul = a_core;
  struct ggml_tensor* rot_sin_mul = b_core;

  // Check if rot_sin_mul's src has rotate_half
  struct ggml_tensor* x_from_rot = nullptr;
  for (int attempt = 0; attempt < 2; ++attempt) {
    // Try src[0] and src[1] of rot_sin_mul for rotate_half
    for (int si = 0; si < 2; ++si) {
      x_from_rot = match_rotate_half(rot_sin_mul->src[si]);
      if (x_from_rot) break;
    }
    if (x_from_rot) break;
    // Swap and retry
    std::swap(x_cos_mul, rot_sin_mul);
  }
  if (!x_from_rot) {
    if (rope_debug) {
      fprintf(stderr, "[rope_fuse]   rotate_half not found. MUL_a src0=%d src1=%d, MUL_b src0=%d src1=%d\n",
              a_core->src[0] ? a_core->src[0]->op : -1, a_core->src[1] ? a_core->src[1]->op : -1,
              b_core->src[0] ? b_core->src[0]->op : -1, b_core->src[1] ? b_core->src[1]->op : -1);
    }
    return nullptr;
  }
  if (rope_debug) fprintf(stderr, "[rope_fuse]   rotate_half matched, x=%p\n", (void*)x_from_rot);

  // x_cos_mul = MUL(x, cos) -- verify x matches x_from_rot
  struct ggml_tensor* x_direct = nullptr;
  struct ggml_tensor* cos_tensor = nullptr;
  for (int si = 0; si < 2; ++si) {
    auto* candidate = unwrap_views(x_cos_mul->src[si]);
    if (candidate == x_from_rot) {
      x_direct = x_cos_mul->src[si];
      cos_tensor = x_cos_mul->src[1 - si];
      break;
    }
  }
  if (!x_direct) {
    auto* s0 = unwrap_views(x_cos_mul->src[0]);
    auto* s1 = unwrap_views(x_cos_mul->src[1]);
    // Also try deep unwrap (through PERMUTE etc.)
    auto* s0d = s0; while (s0d && s0d->op != GGML_OP_NONE && s0d->op != GGML_OP_MUL_MAT && s0d->op != GGML_OP_MUL) s0d = s0d->src[0];
    auto* s1d = s1; while (s1d && s1d->op != GGML_OP_NONE && s1d->op != GGML_OP_MUL_MAT && s1d->op != GGML_OP_MUL) s1d = s1d->src[0];
    if (rope_debug) fprintf(stderr, "[rope_fuse]   FAIL: x not found. cos_mul s0=%p(op=%d) s1=%p(op=%d) s0d=%p(op=%d) s1d=%p(op=%d) x_from_rot=%p(op=%d)\n",
        (void*)s0, s0?s0->op:-1, (void*)s1, s1?s1->op:-1,
        (void*)s0d, s0d?s0d->op:-1, (void*)s1d, s1d?s1d->op:-1,
        (void*)x_from_rot, x_from_rot?x_from_rot->op:-1);
    return nullptr;
  }
  if (rope_debug) fprintf(stderr, "[rope_fuse]   cos_tensor found (op=%d)\n", cos_tensor ? cos_tensor->op : -1);

  // Find sin tensor from rot_sin_mul
  struct ggml_tensor* sin_tensor = nullptr;
  for (int si = 0; si < 2; ++si) {
    if (!match_rotate_half(rot_sin_mul->src[si])) {
      sin_tensor = rot_sin_mul->src[si];
      break;
    }
  }
  if (!sin_tensor) {
    if (rope_debug) fprintf(stderr, "[rope_fuse]   FAIL: sin not found\n");
    return nullptr;
  }

  // Trace back from cos to find position tensor and freq_base
  auto* cos_core = unwrap_views(cos_tensor);
  auto* t = cos_core;
  while (t && t->op != GGML_OP_COS) {
    t = t->src[0];
  }
  if (!t) {
    if (rope_debug) fprintf(stderr, "[rope_fuse]   FAIL: COS op not found tracing from cos_tensor (op=%d)\n",
        cos_core ? cos_core->op : -1);
    return nullptr;
  }

  struct ggml_tensor* pos = nullptr;
  float freq_base = 10000.0f;
  int n_dims = 0;
  if (!extract_rope_params(t, &pos, &freq_base, &n_dims, acc)) {
    if (rope_debug) fprintf(stderr, "[rope_fuse]   FAIL: extract_rope_params failed\n");
    return nullptr;
  }
  if (rope_debug) fprintf(stderr, "[rope_fuse]   SUCCESS: freq_base=%.0f n_dims=%d\n", freq_base, n_dims);

  // Build ggml_rope_ext: mode=2 (NeoX half-rotation for HF models)
  auto* x_cont = ensure_cont(ctx, x_direct);
  auto* pos_i32 = (pos->type == GGML_TYPE_I32) ? pos : safe_ggml_cast(ctx, pos, GGML_TYPE_I32);

  auto* result = ggml_rope_ext(ctx, x_cont, pos_i32, NULL,
      n_dims, /*mode=*/2, 0, freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
  return result;
}

// Cast BF16/F16 to F32 for Metal compatibility. No-op if already F32 or integer.
// Only needed when GPU backend is active (Metal binary ops require F32).
static inline struct ggml_tensor* ensure_f32(struct ggml_context* ctx, struct ggml_tensor* t) {
  if (t->type == GGML_TYPE_BF16 || t->type == GGML_TYPE_F16)
    return safe_ggml_cast(ctx, t, GGML_TYPE_F32);
  return t;
}

// ggml_new_f32() asserts no_alloc==false.  This wrapper temporarily disables
// no_alloc so the scalar's 4 bytes are allocated inline from the context pool.
static struct ggml_tensor* make_f32_scalar(struct ggml_context* ctx, float value) {
  ggml_set_no_alloc(ctx, false);
  struct ggml_tensor* t = ggml_new_f32(ctx, value);
  ggml_set_no_alloc(ctx, true);
  return t;
}

// ---------------------------------------------------------------------------
// Eagerly compute a binary f32 scalar op during build_graph.
// Returns a new GGML_OP_NONE tensor with the result, or nullptr if not applicable.
static struct ggml_tensor* try_eager_scalar_binop(
    struct ggml_context* ctx,
    struct ggml_tensor* a, struct ggml_tensor* b,
    char op_char, HostDataAccessor& acc) {
  // Only for scalar f32 tensors where both have data available
  // AND neither is marked as an input (input data is only valid at execute
  // time, not at build time -- eagerly computing from it would break graph
  // reuse across calls with different input values).
  if (ggml_nelements(a) != 1 || ggml_nelements(b) != 1) return nullptr;
  if (a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32) return nullptr;
  if (!a->data || !b->data) return nullptr;
  float va = acc.read_f32(a);
  float vb = acc.read_f32(b);
  float result;
  switch (op_char) {
    case '+': result = va + vb; break;
    case '-': result = va - vb; break;
    case '*': result = va * vb; break;
    case '/': result = (vb != 0.0f) ? va / vb : 0.0f; break;
    default: return nullptr;
  }
  // Temporarily enable allocation for the result scalar.
  ggml_set_no_alloc(ctx, false);
  struct ggml_tensor* gt = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, 1, 1);
  ggml_set_no_alloc(ctx, true);
  gt->op = GGML_OP_NONE;
  *(float*)gt->data = result;
  return gt;
}

// ---------------------------------------------------------------------------
// Eagerly compute a binary f32 op of ANY size during build_graph.
// Used for the streaming encoder's ring buffer mask chain where the entire
// computation is input-derived and must stay on the host to avoid CUDA
// buffer aliasing issues.
// Returns a new GGML_OP_NONE tensor with the result, or nullptr if not applicable.
static struct ggml_tensor* try_eager_f32_binop(
    struct ggml_context* ctx,
    struct ggml_tensor* a, struct ggml_tensor* b,
    char op_char, HostDataAccessor& acc) {
  if (a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32) return nullptr;
  const float* ad = static_cast<const float*>(acc.get(a));
  const float* bd = static_cast<const float*>(acc.get(b));
  if (!ad || !bd) return nullptr;
  int64_t out_ne[4];
  for (int d = 0; d < 4; d++) out_ne[d] = std::max(a->ne[d], b->ne[d]);
  int64_t n = out_ne[0] * out_ne[1] * out_ne[2] * out_ne[3];
  if (n > 10000000) return nullptr;  // safety limit
  ggml_set_no_alloc(ctx, false);
  struct ggml_tensor* gt = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
      out_ne[0], out_ne[1], out_ne[2], out_ne[3]);
  ggml_set_no_alloc(ctx, true);
  gt->op = GGML_OP_NONE;
  float* od = static_cast<float*>(gt->data);
  for (int64_t i = 0; i < n; i++) {
    int64_t i0 = i % out_ne[0], rem = i / out_ne[0];
    int64_t i1 = rem % out_ne[1]; rem /= out_ne[1];
    int64_t i2 = rem % out_ne[2], i3 = rem / out_ne[2];
    int64_t ai = (i0 % a->ne[0]) + (i1 % a->ne[1]) * a->ne[0]
               + (i2 % a->ne[2]) * a->ne[0] * a->ne[1]
               + (i3 % a->ne[3]) * a->ne[0] * a->ne[1] * a->ne[2];
    int64_t bi = (i0 % b->ne[0]) + (i1 % b->ne[1]) * b->ne[0]
               + (i2 % b->ne[2]) * b->ne[0] * b->ne[1]
               + (i3 % b->ne[3]) * b->ne[0] * b->ne[1] * b->ne[2];
    float va = ad[ai], vb = bd[bi];
    switch (op_char) {
      case '+': od[i] = va + vb; break;
      case '-': od[i] = va - vb; break;
      case '*': od[i] = va * vb; break;
      case '/': od[i] = (vb != 0.0f) ? va / vb : 0.0f; break;
      case 'g': od[i] = (va >= vb) ? 1.0f : 0.0f; break;  // GE
      case 'l': od[i] = (va < vb) ? 1.0f : 0.0f; break;   // LT
      case 'L': od[i] = (va <= vb) ? 1.0f : 0.0f; break;   // LE
      case 'G': od[i] = (va > vb) ? 1.0f : 0.0f; break;    // GT
      case '&': od[i] = (va != 0.0f && vb != 0.0f) ? 1.0f : 0.0f; break;
      default: od[i] = 0.0f; break;
    }
  }
  return gt;
}

static struct ggml_tensor* safe_ggml_permute(struct ggml_context* ctx, struct ggml_tensor* a,
                                              int axis0, int axis1, int axis2, int axis3,
                                              const char* caller) {
  if (axis0 < 0 || axis0 >= GGML_MAX_DIMS || axis1 < 0 || axis1 >= GGML_MAX_DIMS ||
      axis2 < 0 || axis2 >= GGML_MAX_DIMS || axis3 < 0 || axis3 >= GGML_MAX_DIMS) {
    fprintf(stderr, "[ggml_backend] %s: invalid permute axes [%d,%d,%d,%d], src ne=[%lld,%lld,%lld,%lld]\n",
            caller, axis0, axis1, axis2, axis3,
            (long long)a->ne[0], (long long)a->ne[1], (long long)a->ne[2], (long long)a->ne[3]);
    return a;
  }
  // Identity permute -> no-op
  if (axis0 == 0 && axis1 == 1 && axis2 == 2 && axis3 == 3) return a;
  // Size-1 dim swap: when permuting dims where at least one has ne==1,
  // the data layout is unchanged. Replace with RESHAPE (no CUDA kernel).
  // This fires for decode (T=1) where H<->T swaps are data no-ops.
  if (ggml_is_contiguous(a)) {
    int64_t pne[4] = {a->ne[axis0], a->ne[axis1], a->ne[axis2], a->ne[axis3]};
    bool is_size1_swap = (pne[0] != a->ne[0] || pne[1] != a->ne[1] ||
                          pne[2] != a->ne[2] || pne[3] != a->ne[3]);
    if (is_size1_swap) {
      bool all_swaps_trivial = true;
      for (int i = 0; i < 4; i++) {
        if (pne[i] != a->ne[i] && pne[i] != 1 && a->ne[i] != 1) {
          all_swaps_trivial = false;
          break;
        }
      }
      if (all_swaps_trivial) {
        // Peel through source RESHAPEs to compose into one
        struct ggml_tensor* base = a;
        while (base->op == GGML_OP_RESHAPE && base->src[0]) base = base->src[0];
        // Identity check after reshape composition
        if (base->ne[0] == pne[0] && base->ne[1] == pne[1] &&
            base->ne[2] == pne[2] && base->ne[3] == pne[3])
          return base;
        return ggml_reshape_4d(ctx, base, pne[0], pne[1], pne[2], pne[3]);
      }
    }
  }
  // Compose consecutive permutes: PERMUTE(PERMUTE(x, p1), p2) = PERMUTE(x, p1 o p2)
  if (a->op == GGML_OP_PERMUTE) {
    int p1[4];
    memcpy(p1, a->op_params, 4 * sizeof(int));
    int p2[4] = {axis0, axis1, axis2, axis3};
    int composed[4] = {p1[p2[0]], p1[p2[1]], p1[p2[2]], p1[p2[3]]};
    // Check if composed is identity
    if (composed[0] == 0 && composed[1] == 1 && composed[2] == 2 && composed[3] == 3)
      return a->src[0];
    return ggml_permute(ctx, a->src[0], composed[0], composed[1], composed[2], composed[3]);
  }
  return ggml_permute(ctx, a, axis0, axis1, axis2, axis3);
}

// Resolve the shape of an IR tensor, applying symbolic overrides when available.
// Shared by ctx_size estimation (before ggml_init) and tensor creation (Phase B).
static void resolve_ir_shape(
    const ggml_ir::Tensor* t,
    const int64_t* input_ne_overrides,
    const std::unordered_map<int32_t, int64_t>& sym_dim_values,
    int64_t ne[4],
    int& n_dims) {
  ne[0] = ne[1] = ne[2] = ne[3] = 1;
  if (t->ne() && t->ne()->size() > 0) {
    for (size_t d = 0; d < t->ne()->size() && d < 4; ++d) {
      ne[d] = t->ne()->Get(d);
    }
  }
  if (input_ne_overrides && t->is_input() && t->sym_dim_ids()) {
    int input_idx = t->input_index();
    for (size_t d = 0; d < t->sym_dim_ids()->size() && d < 4; ++d) {
      int32_t sid = t->sym_dim_ids()->Get(d);
      if (sid >= 0) {
        ne[d] = input_ne_overrides[input_idx * 4 + d];
      }
    }
  }
  if (!t->is_input() && t->sym_dim_ids() && !sym_dim_values.empty()) {
    for (size_t d = 0; d < t->sym_dim_ids()->size() && d < 4; ++d) {
      int32_t sid = t->sym_dim_ids()->Get(d);
      if (sid == -2) {
        const uint8_t* code = nullptr;
        size_t code_len = 0;
        if (get_dim_expr_bytecode(t->sym_dim_exprs(), d, code, code_len)) {
          ne[d] = eval_sym_expr(code, code_len, sym_dim_values);
        }
      } else if (sid >= 0) {
        auto it = sym_dim_values.find(sid);
        if (it != sym_dim_values.end()) ne[d] = it->second;
      }
    }
  }
  n_dims = 4;
  for (int d = 3; d >= 1; --d) {
    if (ne[d] == 1) { n_dims = d; } else { break; }
  }
  if (n_dims == 0) n_dims = 1;
}

// Resolve shape from cached IR tensor (no FlatBuffer access).
static void resolve_cached_shape(
    const CachedIRTensor& ct,
    const int64_t* input_ne_overrides,
    const std::unordered_map<int32_t, int64_t>& sym_dim_values,
    int64_t ne[4],
    int& n_dims) {
  ne[0] = ct.ne[0]; ne[1] = ct.ne[1]; ne[2] = ct.ne[2]; ne[3] = ct.ne[3];
  if (ct.has_sym_dims) {
    for (int d = 0; d < 4; ++d) {
      int32_t sid = ct.sym_dim_ids[d];
      if (sid == -1) continue;
      if (ct.is_input && input_ne_overrides && sid >= 0) {
        ne[d] = input_ne_overrides[ct.input_index * 4 + d];
      } else if (!ct.is_input && sid == -2 && !ct.sym_dim_exprs[d].bytecode.empty()) {
        ne[d] = eval_sym_expr(ct.sym_dim_exprs[d].bytecode.data(),
                              ct.sym_dim_exprs[d].bytecode.size(), sym_dim_values);
      } else if (!ct.is_input && sid >= 0) {
        auto it = sym_dim_values.find(sid);
        if (it != sym_dim_values.end()) ne[d] = it->second;
      }
    }
  }
  n_dims = 4;
  for (int d = 3; d >= 1; --d) {
    if (ne[d] == 1) { n_dims = d; } else { break; }
  }
  if (n_dims == 0) n_dims = 1;
}

} // namespace executorch_ggml
