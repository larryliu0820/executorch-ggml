#pragma once

#include "build_context.h"
#include "custom_ops.h"

namespace executorch_ggml {

// ---------------------------------------------------------------------------
// INDEX (single-dim gather, dim=0)
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_index(BuildContext& bc) {
  const auto* t = bc.ir_tensor;
  if (!t->op_params() || t->op_params()->size() < 4) {
    return nullptr;
  }
  int32_t dim = 0;
  memcpy(&dim, t->op_params()->data(), 4);
  if (dim != 0) {
    return nullptr;
  }
  struct ggml_tensor* x = bc.srcs[0];
  struct ggml_tensor* idx = bc.srcs[1];
  if (idx->type == GGML_TYPE_I64) {
    idx = eager_cast_i64_to_i32(bc.ctx, idx, &bc.host_acc);
  } else if (idx->type != GGML_TYPE_I32) {
    idx = safe_ggml_cast(bc.ctx, idx, GGML_TYPE_I32, &bc.host_acc);
  }
  if (x->type == GGML_TYPE_I8) {
    x = safe_ggml_cast(bc.ctx, x, GGML_TYPE_I32, &bc.host_acc);
  }
  return ggml_get_rows(bc.ctx, x, idx);
}

// ---------------------------------------------------------------------------
// INDEX_MULTI
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_index_multi(BuildContext& bc) {
  const auto* t = bc.ir_tensor;
  int32_t ndims_hint = 0;
  if (t->op_params() && t->op_params()->size() >= 4) {
    memcpy(&ndims_hint, t->op_params()->data(), 4);
  }

  if (bc.srcs.size() < 2) {
    return nullptr;
  }

  const int ndims = static_cast<int>(bc.srcs.size()) - 1;
  if (ndims < 1 || ndims > 4) {
    return nullptr;
  }
  if (ndims_hint > 0 && ndims_hint != ndims) {
    return nullptr;
  }

  struct ggml_tensor* src_x = bc.srcs[0];
  ggml_type out_type = GGML_TYPE_F32;
  switch (t->type()) {
    case ggml_ir::TensorType::F16:  out_type = GGML_TYPE_F16;  break;
    case ggml_ir::TensorType::I64:  out_type = GGML_TYPE_I64;  break;
    case ggml_ir::TensorType::I32:  out_type = GGML_TYPE_I32;  break;
    case ggml_ir::TensorType::BOOL: out_type = GGML_TYPE_I32;  break;
    case ggml_ir::TensorType::F32:
    default:                         out_type = GGML_TYPE_F32;  break;
  }

  std::vector<struct ggml_tensor*> custom_args;
  custom_args.reserve(1 + ndims);
  custom_args.push_back(src_x);
  for (int i = 0; i < ndims; ++i) {
    struct ggml_tensor* idx = bc.srcs[1 + i];
    if (idx->type != GGML_TYPE_I32 && idx->type != GGML_TYPE_I64) {
      idx = safe_ggml_cast(bc.ctx, idx, GGML_TYPE_I32, &bc.host_acc);
    }
    custom_args.push_back(idx);
  }

  int64_t out_ne0 = t->ne() ? (*t->ne())[0] : 1;
  int64_t out_ne1 = t->ne() && t->ne()->size() > 1 ? (*t->ne())[1] : 1;
  int64_t out_ne2 = t->ne() && t->ne()->size() > 2 ? (*t->ne())[2] : 1;
  int64_t out_ne3 = t->ne() && t->ne()->size() > 3 ? (*t->ne())[3] : 1;

  auto* gt = ggml_custom_4d(
      bc.ctx,
      out_type,
      out_ne0, out_ne1, out_ne2, out_ne3,
      custom_args.data(),
      static_cast<int>(custom_args.size()),
      ggml_custom_index_multi,
      GGML_N_TASKS_MAX,
      nullptr);
  pin_to_cpu(bc, gt);
  return gt;
}

// ---------------------------------------------------------------------------
// INDEX_PUT
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_index_put(BuildContext& bc) {
  const auto* t = bc.ir_tensor;
  if (!t->op_params() || t->op_params()->size() < 8) {
    return nullptr;
  }
  int32_t nidx = 0, pm = 0;
  memcpy(&nidx, t->op_params()->data(), 4);
  memcpy(&pm, t->op_params()->data() + 4, 4);

  if (bc.srcs.size() < 3) {
    return nullptr;
  }

  struct ggml_tensor* dst = bc.srcs.front();
  struct ggml_tensor* val = bc.srcs.back();

  int idx_pos = -1;
  for (int i = 0; i < nidx; ++i) {
    if (pm & (1 << i)) {
      idx_pos = i;
      break;
    }
  }
  if (idx_pos < 0) {
    return dst;
  }

  struct ggml_tensor* idx = bc.srcs[1];

  if (idx->type != GGML_TYPE_I32 && idx->type != GGML_TYPE_I64) {
    idx = safe_ggml_cast(bc.ctx, idx, GGML_TYPE_I32, &bc.host_acc);
  }

  if (val->type != dst->type) {
    val = safe_ggml_cast(bc.ctx, val, dst->type, &bc.host_acc);
  }

  // Check if dst lives in mutable_buf (KV cache).
  bool is_mutable_dst = false;
  {
    int dst_tid = t->src_ids()->Get(0);
    auto mb_it = bc.handle->leaf_buf_map.find(dst_tid);
    if (mb_it != bc.handle->leaf_buf_map.end() &&
        bc.handle->mutable_buf && mb_it->second.buf == bc.handle->mutable_buf) {
      is_mutable_dst = true;
    }
  }

  struct ggml_tensor* gt = nullptr;
  if (is_mutable_dst) {
    if (val->type != GGML_TYPE_F32) {
      val = safe_ggml_cast(bc.ctx, val, GGML_TYPE_F32, &bc.host_acc);
    }

    int ndim = 4;
    int ggml_scatter_axis = (ndim - 1) - idx_pos;
    if (ggml_scatter_axis == 1) {
      gt = ggml_set_rows(bc.ctx, dst, val, idx);
    } else {
      fprintf(stderr, "[KV_CACHE_DEBUG] idx tensor: type=%s, ne=[%ld,%ld,%ld,%ld], nbytes=%zu\n",
              ggml_type_name(idx->type), idx->ne[0], idx->ne[1], idx->ne[2], idx->ne[3], ggml_nbytes(idx));

      int64_t start_pos;
      if (idx->type == GGML_TYPE_I64) {
        start_pos = bc.host_acc.read_i64(idx);
        fprintf(stderr, "[KV_CACHE_DEBUG] read as I64: start_pos=%ld\n", start_pos);
      } else if (idx->type == GGML_TYPE_I32) {
        int32_t start_pos_i32 = bc.host_acc.read_i32(idx);
        start_pos = static_cast<int64_t>(start_pos_i32);
        fprintf(stderr, "[KV_CACHE_DEBUG] read as I32: start_pos_i32=%d -> start_pos=%ld\n", start_pos_i32, start_pos);
      } else {
        fprintf(stderr, "[KV_CACHE_DEBUG] ERROR: Unsupported index tensor type: %s\n", ggml_type_name(idx->type));
        start_pos = 0;
      }

      const void* idx_data = bc.host_acc.get(idx);
      if (idx_data && ggml_nbytes(idx) >= 4) {
        const uint8_t* bytes = static_cast<const uint8_t*>(idx_data);
        fprintf(stderr, "[KV_CACHE_DEBUG] idx raw bytes: %02x %02x %02x %02x\n",
                bytes[0], bytes[1], bytes[2], bytes[3]);
      }

      int64_t seq_len_new = val->ne[ggml_scatter_axis];

      fprintf(stderr, "[KV_CACHE_DEBUG] start_pos=%ld, seq_len_new=%ld, scatter_axis=%d\n",
              start_pos, seq_len_new, ggml_scatter_axis);
      fprintf(stderr, "[KV_CACHE_DEBUG] dst->nb[%d]=%zu, dst->ne[%d]=%ld\n",
              ggml_scatter_axis, dst->nb[ggml_scatter_axis], ggml_scatter_axis, dst->ne[ggml_scatter_axis]);

      if (start_pos < 0 || start_pos >= dst->ne[ggml_scatter_axis]) {
        fprintf(stderr, "[KV_CACHE_ERROR] Invalid start_pos=%ld (must be in range [0, %ld))\n",
                start_pos, dst->ne[ggml_scatter_axis]);
        fprintf(stderr, "[KV_CACHE_ERROR] This suggests corrupted index tensor data\n");
        return nullptr;
      }

      size_t offset_bytes = start_pos * dst->nb[ggml_scatter_axis];

      fprintf(stderr, "[KV_CACHE_DEBUG] offset_bytes=%zu (0x%lx)\n", offset_bytes, offset_bytes);
      int64_t view_ne[4];
      for (int d = 0; d < 4; d++) view_ne[d] = dst->ne[d];
      view_ne[ggml_scatter_axis] = seq_len_new;

      auto* cache_slice = safe_ggml_view_4d(bc.ctx, dst,
          view_ne[0], view_ne[1], view_ne[2], view_ne[3],
          dst->nb[1], dst->nb[2], dst->nb[3], offset_bytes);
      if (!cache_slice) {
        fprintf(stderr, "KV cache scatter: view creation failed\n");
        return nullptr;
      }

      auto* val_c = ensure_cont(bc.ctx, val);
      gt = ggml_cpy(bc.ctx, val_c, cache_slice);
      gt = dst;  // downstream will see updated cache
    }
    ggml_set_output(gt);
  } else {
    struct ggml_tensor* args[3] = {dst, idx, val};
    gt = ggml_custom_4d(
        bc.ctx, dst->type,
        dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
        args, 3,
        ggml_custom_index_put_rows,
        GGML_N_TASKS_MAX,
        nullptr);
    pin_to_cpu(bc, gt);
  }
  return gt;
}

// ---------------------------------------------------------------------------
// CAT
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_cat(BuildContext& bc) {
  const auto* t = bc.ir_tensor;
  if (!t->op_params() || t->op_params()->size() < 4) {
    return nullptr;
  }
  int32_t ax = 0;
  memcpy(&ax, t->op_params()->data(), 4);
  struct ggml_tensor* cur = ensure_cont(bc.ctx, bc.srcs[0]);
  for (size_t si = 1; si < bc.srcs.size(); ++si) {
    cur = ggml_concat(bc.ctx, cur, ensure_cont(bc.ctx, bc.srcs[si]), ax);
  }
  return cur;
}

// ---------------------------------------------------------------------------
// REPEAT
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_repeat(BuildContext& bc) {
  const auto* t = bc.ir_tensor;
  struct ggml_tensor* a = bc.srcs[0];
  struct ggml_tensor* b = bc.srcs[1];

  if (ggml_are_same_shape(a, b)) {
    return a;
  }

  if (!ggml_can_repeat(a, b)) {
    int64_t target_ne[4];
    for (int d = 0; d < 4; ++d) {
      if (a->ne[d] == 1 && b->ne[d] > 1) {
        target_ne[d] = b->ne[d];
      } else {
        target_ne[d] = a->ne[d];
      }
    }
    b = ggml_new_tensor_4d(bc.ctx, a->type, target_ne[0], target_ne[1], target_ne[2], target_ne[3]);
  }

  if (!ggml_can_repeat(a, b)) {
    if (ggml_can_repeat(b, a)) {
      return ggml_repeat(bc.ctx, b, a);
    } else {
      fprintf(stderr,
              "[executorch-ggml] REPEAT shape not repeatable (srcs=[%d,%d]): a=(%lld,%lld,%lld,%lld) b=(%lld,%lld,%lld,%lld)\n",
              (int)(t->src_ids() && t->src_ids()->size() > 0 ? t->src_ids()->Get(0) : -1),
              (int)(t->src_ids() && t->src_ids()->size() > 1 ? t->src_ids()->Get(1) : -1),
              (long long) a->ne[0], (long long) a->ne[1], (long long) a->ne[2], (long long) a->ne[3],
              (long long) b->ne[0], (long long) b->ne[1], (long long) b->ne[2], (long long) b->ne[3]);
      return nullptr;
    }
  } else {
    return ggml_repeat(bc.ctx, a, b);
  }
}

// ---------------------------------------------------------------------------
// REPEAT_INTERLEAVE
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_repeat_interleave(BuildContext& bc) {
  const auto* t = bc.ir_tensor;
  if (!t->op_params() || t->op_params()->size() < 8) {
    return nullptr;
  }
  int32_t dim = 0, reps = 1;
  memcpy(&dim, t->op_params()->data(), 4);
  memcpy(&reps, t->op_params()->data() + 4, 4);
  if (dim != 1 || reps < 1) {
    return nullptr;
  }
  struct ggml_tensor* src = bc.srcs[0];
  int64_t D  = src->ne[0];
  int64_t T  = src->ne[1];
  int64_t H  = src->ne[2];
  int64_t B  = src->ne[3];
  size_t  head_stride = src->nb[2];
  struct ggml_tensor* result = nullptr;
  for (int64_t h = 0; h < H; ++h) {
    struct ggml_tensor* head_slice = safe_ggml_view_4d(
        bc.ctx, src,
        D, T, 1, B,
        src->nb[1], src->nb[2], src->nb[3],
        h * head_stride);
    if (!head_slice) {
      fprintf(stderr, "Head slice creation failed for head %ld\n", h);
      return nullptr;
    }
    for (int r = 0; r < reps; ++r) {
      if (result == nullptr) {
        result = head_slice;
      } else {
        result = ggml_concat(bc.ctx, result, head_slice, 2);
      }
    }
  }
  return result;
}

} // namespace executorch_ggml
