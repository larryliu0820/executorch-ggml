#pragma once

#include "build_context.h"
#include "sym_expr.h"

namespace executorch_ggml {

// ---------------------------------------------------------------------------
// VIEW (reshape)
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_view(BuildContext& bc) {
  const auto* t = bc.ir_tensor;
  if (!t->op_params() || t->op_params()->size() < 4) {
    return nullptr;
  }

  const uint8_t* data = t->op_params()->data();
  int32_t ndims = 0;
  memcpy(&ndims, data, sizeof(int32_t));

  if (t->op_params()->size() < 4 + ndims * 8) {
    return nullptr;
  }

  int64_t new_ne[4] = {1, 1, 1, 1};
  for (int32_t i = 0; i < ndims && i < 4; ++i) {
    memcpy(&new_ne[i], data + 4 + i * 8, sizeof(int64_t));
  }

  // Resolve sym_dim_ids for VIEW target shape.
  if (t->sym_dim_ids() && !bc.sym_dim_values.empty()) {
    for (size_t d = 0; d < t->sym_dim_ids()->size() && d < 4; ++d) {
      int32_t sid = t->sym_dim_ids()->Get(d);
      if (sid == -2) {
        const uint8_t* code = nullptr;
        size_t code_len = 0;
        if (get_dim_expr_bytecode(t->sym_dim_exprs(), d, code, code_len)) {
          new_ne[d] = eval_sym_expr(code, code_len, bc.sym_dim_values);
        }
      } else if (sid >= 0) {
        auto it = bc.sym_dim_values.find(sid);
        if (it != bc.sym_dim_values.end()) new_ne[d] = it->second;
      }
    }
  }
  // Numel-inference fallback as safety net.
  int64_t src_numel = ggml_nelements(bc.srcs[0]);
  int64_t view_numel = new_ne[0] * new_ne[1] * new_ne[2] * new_ne[3];
  if (src_numel != view_numel && view_numel > 0) {
    for (int d = 3; d >= 0; --d) {
      if (new_ne[d] <= 1) continue;
      int64_t others = view_numel / new_ne[d];
      if (others > 0 && src_numel % others == 0) {
        int64_t inferred = src_numel / others;
        if (inferred != new_ne[d]) {
          new_ne[d] = inferred;
          view_numel = new_ne[0] * new_ne[1] * new_ne[2] * new_ne[3];
          if (view_numel == src_numel) break;
        }
      }
    }
  }

  // Collapse consecutive RESHAPEs
  struct ggml_tensor* view_src = bc.srcs[0];
  while (view_src->op == GGML_OP_RESHAPE && view_src->src[0]) {
    view_src = view_src->src[0];
  }
  // Identity reshape -> no-op
  if (view_src->ne[0] == new_ne[0] && view_src->ne[1] == new_ne[1] &&
      view_src->ne[2] == new_ne[2] && view_src->ne[3] == new_ne[3] &&
      ggml_is_contiguous(view_src)) {
    return view_src;
  }
  return ggml_reshape_4d(bc.ctx, ensure_cont(bc.ctx, view_src),
                          new_ne[0], new_ne[1], new_ne[2], new_ne[3]);
}

// ---------------------------------------------------------------------------
// PERMUTE
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_permute(BuildContext& bc) {
  const auto* t = bc.ir_tensor;
  if (!t->op_params() || t->op_params()->size() < 4) {
    return nullptr;
  }

  const uint8_t* data = t->op_params()->data();
  int32_t ndims = 0;
  memcpy(&ndims, data, sizeof(int32_t));

  if (t->op_params()->size() < 4 + ndims * 4) {
    return nullptr;
  }

  int32_t perm[4] = {0, 1, 2, 3};
  for (int32_t i = 0; i < ndims && i < 4; ++i) {
    memcpy(&perm[i], data + 4 + i * 4, sizeof(int32_t));
  }

  for (int32_t i = 0; i < 4; ++i) {
    if (perm[i] < 0 || perm[i] >= 4) {
      fprintf(stderr, "[ggml_backend] PERMUTE: invalid axis perm[%d]=%d (ndims=%d, perm=[%d,%d,%d,%d])\n",
              i, perm[i], ndims, perm[0], perm[1], perm[2], perm[3]);
      return nullptr;
    }
  }
  return safe_ggml_permute(bc.ctx, bc.srcs[0], perm[0], perm[1], perm[2], perm[3], "PERMUTE");
}

// ---------------------------------------------------------------------------
// TRANSPOSE
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_transpose(BuildContext& bc) {
  const auto* t = bc.ir_tensor;
  if (!t->op_params() || t->op_params()->size() < 12) {
    return nullptr;
  }
  int32_t dim0 = 0, dim1 = 1, nd = 4;
  memcpy(&dim0, t->op_params()->data(), 4);
  memcpy(&dim1, t->op_params()->data() + 4, 4);
  memcpy(&nd, t->op_params()->data() + 8, 4);

  if (nd > 4) nd = 4;
  int ax0 = (nd - 1) - dim0;
  int ax1 = (nd - 1) - dim1;
  int perm[4] = {0, 1, 2, 3};
  if (ax0 >= 0 && ax0 < 4 && ax1 >= 0 && ax1 < 4) {
    int tmp = perm[ax0];
    perm[ax0] = perm[ax1];
    perm[ax1] = tmp;
  } else {
    fprintf(stderr, "[ggml_backend] TRANSPOSE: bad axes ax0=%d ax1=%d (dim0=%d dim1=%d nd=%d)\n",
            ax0, ax1, dim0, dim1, nd);
  }
  fprintf(stderr, "[ggml_backend] TRANSPOSE: perm=[%d,%d,%d,%d] src_ne=[%lld,%lld,%lld,%lld]\n",
          perm[0], perm[1], perm[2], perm[3],
          (long long)bc.srcs[0]->ne[0], (long long)bc.srcs[0]->ne[1],
          (long long)bc.srcs[0]->ne[2], (long long)bc.srcs[0]->ne[3]);
  return safe_ggml_permute(bc.ctx, bc.srcs[0], perm[0], perm[1], perm[2], perm[3], "TRANSPOSE");
}

// ---------------------------------------------------------------------------
// UNSQUEEZE
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_unsqueeze(BuildContext& bc) {
  const auto* t = bc.ir_tensor;
  struct ggml_tensor* a = bc.srcs[0];
  int32_t pt_dim = 0;
  int32_t pt_ndim = -1;
  if (t->op_params() && t->op_params()->size() >= 4) {
    memcpy(&pt_dim, t->op_params()->data(), sizeof(int32_t));
  }
  if (t->op_params() && t->op_params()->size() >= 8) {
    memcpy(&pt_ndim, t->op_params()->data() + 4, sizeof(int32_t));
  }

  if (pt_ndim <= 0) {
    pt_ndim = std::max(1, ggml_n_dims(a));
  }

  const int32_t out_ndim = pt_ndim + 1;
  if (pt_dim < 0) {
    pt_dim += out_ndim;
  }
  if (pt_dim < 0 || pt_dim > pt_ndim) {
    return nullptr;
  }

  // For 4D->5D unsqueeze patterns
  if (pt_ndim >= 4 || out_ndim > 4) {
    int64_t src_numel = ggml_nelements(a);
    int64_t ir_numel = bc.ne[0] * bc.ne[1] * bc.ne[2] * bc.ne[3];
    if (src_numel != ir_numel) {
      return nullptr;
    }
    return ggml_reshape_4d(bc.ctx, ensure_cont(bc.ctx, a), bc.ne[0], bc.ne[1], bc.ne[2], bc.ne[3]);
  }

  // Build PyTorch-order runtime input shape from ggml ne.
  int64_t pt_in[4] = {1, 1, 1, 1};
  for (int d = 0; d < pt_ndim; ++d) {
    pt_in[d] = a->ne[pt_ndim - 1 - d];
  }

  // Insert size-1 dim at pt_dim.
  int64_t pt_out[4] = {1, 1, 1, 1};
  for (int d = 0; d < out_ndim; ++d) {
    if (d < pt_dim) {
      pt_out[d] = pt_in[d];
    } else if (d == pt_dim) {
      pt_out[d] = 1;
    } else {
      pt_out[d] = pt_in[d - 1];
    }
  }

  // Convert back to ggml-order ne.
  int64_t out_ne[4] = {1, 1, 1, 1};
  for (int d = 0; d < out_ndim; ++d) {
    out_ne[d] = pt_out[out_ndim - 1 - d];
  }
  return ggml_reshape_4d(bc.ctx, ensure_cont(bc.ctx, a), out_ne[0], out_ne[1], out_ne[2], out_ne[3]);
}

// ---------------------------------------------------------------------------
// SLICE
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_slice(BuildContext& bc) {
  const auto* t = bc.ir_tensor;
  if (!t->op_params() || t->op_params()->size() < 28) {
    return nullptr;
  }
  int32_t dim = 0;
  int64_t start = 0, end = 0, step = 1;
  const uint8_t* p = t->op_params()->data();
  memcpy(&dim, p, 4);
  memcpy(&start, p + 4, 8);
  memcpy(&end, p + 12, 8);
  memcpy(&step, p + 20, 8);
  if (step != 1) {
    return nullptr;
  }

  uint32_t nd = 4;
  if (t->op_params()->size() >= 32) {
    memcpy(&nd, p + 28, 4);
  }
  struct ggml_tensor* a = bc.srcs[0];
  int ax = (int(nd) - 1) - dim;

  int64_t ne[4];
  for (int d = 0; d < 4; ++d) ne[d] = bc.ne[d];

  // Resolve dynamic output shape from sym_dim_ids/sym_dim_exprs.
  int64_t resolved_slice_ne = ne[ax];
  if (!bc.sym_dim_values.empty()) {
    for (int d = 0; d < 4; ++d) {
      int32_t sid = (t->sym_dim_ids() && d < (int)t->sym_dim_ids()->size())
                    ? t->sym_dim_ids()->Get(d) : -1;
      if (sid == -2) {
        const uint8_t* code = nullptr;
        size_t code_len = 0;
        if (get_dim_expr_bytecode(t->sym_dim_exprs(), d, code, code_len)) {
          ne[d] = eval_sym_expr(code, code_len, bc.sym_dim_values);
        }
      } else if (sid >= 0) {
        auto it = bc.sym_dim_values.find(sid);
        if (it != bc.sym_dim_values.end()) ne[d] = it->second;
      }
    }
    resolved_slice_ne = ne[ax];
  }

  // Use source tensor shape for non-sliced dims
  for (int d = 0; d < 4; ++d) ne[d] = a->ne[d];

  constexpr int64_t SENTINEL = static_cast<int64_t>(1) << 62;
  int64_t actual_dim = a->ne[ax];

  // Resolve Python-style negative indices (e.g. mel[:, :, -2:])
  if (start != SENTINEL && start < 0) start += actual_dim;
  if (end != SENTINEL && end < 0) end += actual_dim;

  bool start_is_sentinel = (start == SENTINEL);
  bool end_is_sentinel = (end == SENTINEL);

  if (start_is_sentinel && resolved_slice_ne > 0 &&
      resolved_slice_ne <= actual_dim) {
    start = (actual_dim - resolved_slice_ne) / 2;
    end = start + resolved_slice_ne;
    ne[ax] = resolved_slice_ne;
  } else {
    if (start_is_sentinel) start = 0;
    if (end_is_sentinel) {
      end = start + resolved_slice_ne * step;
    }
    if (end > actual_dim) end = actual_dim;
    ne[ax] = end - start;

    if (t->sym_dim_ids() && ax < (int)t->sym_dim_ids()->size()) {
      int32_t sid_ax = t->sym_dim_ids()->Get(ax);
      if ((sid_ax == -2 || sid_ax >= 0) && resolved_slice_ne > 0 &&
          resolved_slice_ne != ne[ax]) {
        ne[ax] = resolved_slice_ne;
        end = start + resolved_slice_ne;
        if (end > actual_dim) {
          end = actual_dim;
          ne[ax] = end - start;
        }
      }
    }
  }

  size_t offset = static_cast<size_t>(start) * a->nb[ax];
  auto* gt = safe_ggml_view_4d(bc.ctx, a, ne[0], ne[1], ne[2], ne[3],
                        a->nb[1], a->nb[2], a->nb[3], offset);
  if (!gt) {
    fprintf(stderr, "Slice operation: view creation failed\n");
    return nullptr;
  }
  if (!ggml_is_contiguous(gt)) {
    gt = ggml_cont(bc.ctx, gt);
  }
  return gt;
}

} // namespace executorch_ggml
