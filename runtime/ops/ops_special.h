#pragma once

#include "build_context.h"

namespace executorch_ggml {

// ---------------------------------------------------------------------------
// LLAMA_ATTENTION (SDPA via ggml_flash_attn_ext)
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_llama_attention(BuildContext& bc) {
  if (bc.srcs.size() < 3) {
    return nullptr;
  }
  struct ggml_tensor* q = bc.srcs[0];
  struct ggml_tensor* k = bc.srcs[1];
  struct ggml_tensor* v = bc.srcs[2];
  ggml_type sdpa_orig_type = q->type;

  // CUDA flash_attn_ext only supports F16 and F32, NOT BF16.
  if (q->type == GGML_TYPE_BF16) q = ggml_cast(bc.ctx, q, GGML_TYPE_F16);
  if (k->type == GGML_TYPE_BF16) k = ggml_cast(bc.ctx, k, GGML_TYPE_F16);
  if (v->type == GGML_TYPE_BF16) v = ggml_cast(bc.ctx, v, GGML_TYPE_F16);

  bool is_causal = false;
  if (bc.ir_tensor->op_params() && bc.ir_tensor->op_params()->size() >= 4) {
    int32_t ic;
    memcpy(&ic, bc.ir_tensor->op_params()->data(), sizeof(int32_t));
    is_causal = (ic != 0);
  }

  // Optional attention mask.
  struct ggml_tensor* mask = nullptr;
  if (bc.srcs.size() > 3 && bc.srcs[3] != nullptr) {
    mask = bc.srcs[3];
    auto cm_src = bc.causal_mask_cache.find(reinterpret_cast<uint64_t>(mask));
    if (cm_src != bc.causal_mask_cache.end()) {
      mask = cm_src->second;
    } else {
      struct ggml_tensor* orig_mask = mask;
      // Track whether the original mask was boolean (I32/I64) vs additive (F32).
      // Boolean masks need scale+offset to convert 0/1 -> 0/-65504.
      // Additive masks (e.g. from WHERE with values {0, -65504}) just need F16 cast.
      bool was_boolean = (mask->type == GGML_TYPE_I32 || mask->type == GGML_TYPE_I64);
      if (mask->type == GGML_TYPE_I32 || mask->type == GGML_TYPE_I64) {
        mask = safe_ggml_cast(bc.ctx, mask, GGML_TYPE_F32, &bc.host_acc);
      }
      if (mask->type == GGML_TYPE_F32) {
        if (was_boolean) {
          // Boolean 0/1 mask: scale to {0, -65504} for F16
          struct ggml_tensor* scaled = ggml_scale(bc.ctx, ggml_cont(bc.ctx, mask), 65504.0f);
          mask = safe_ggml_cast(bc.ctx,
              ggml_add(bc.ctx, scaled, make_f32_scalar(bc.ctx, -65504.0f)),
              GGML_TYPE_F16, &bc.host_acc);
        } else {
          // Additive F32 mask (e.g. from WHERE): just cast to F16
          if (!ggml_is_contiguous(mask)) {
            // For 2D masks, use F32 cont to avoid CUDA ggml_cont F16 bug
            mask = ggml_cont(bc.ctx, mask);
          }
          mask = safe_ggml_cast(bc.ctx, mask, GGML_TYPE_F16, &bc.host_acc);
        }
      } else if (mask->type != GGML_TYPE_F16) {
        mask = safe_ggml_cast(bc.ctx, mask, GGML_TYPE_F16, &bc.host_acc);
      }
      if (!ggml_is_contiguous(mask)) {
        mask = ggml_cont(bc.ctx, mask);
      }
      bc.causal_mask_cache[reinterpret_cast<uint64_t>(orig_mask)] = mask;
    }
  }
  // Build causal mask at runtime when is_causal=true and no explicit mask.
  if (is_causal && mask == nullptr) {
    int64_t T_q  = q->ne[1];
    int64_t T_kv = k->ne[1];
    uint64_t mask_key = ((uint64_t)T_kv << 32) | (uint64_t)T_q;
    auto cm_it = bc.causal_mask_cache.find(mask_key);
    if (cm_it != bc.causal_mask_cache.end() && cm_it->second != nullptr) {
      mask = cm_it->second;
    } else {
      ggml_set_no_alloc(bc.ctx, false);
      struct ggml_tensor* causal = ggml_new_tensor_4d(bc.ctx, GGML_TYPE_F16, T_kv, T_q, 1, 1);
      ggml_set_no_alloc(bc.ctx, true);
      const ggml_fp16_t zero_f16 = ggml_fp32_to_fp16(0.0f);
      const ggml_fp16_t neg_inf_f16 = ggml_fp32_to_fp16(-65504.0f);
      ggml_fp16_t* d = (ggml_fp16_t*)causal->data;
      for (int64_t row = 0; row < T_q; row++) {
        for (int64_t col = 0; col < T_kv; col++) {
          d[row * T_kv + col] = (col <= row + (T_kv - T_q)) ? zero_f16 : neg_inf_f16;
        }
      }
      bc.causal_mask_cache[mask_key] = causal;
      mask = causal;
    }
  }

  float scale = 1.0f;
  if (q->ne[0] > 0) {
    scale = 1.0f / std::sqrt((float) q->ne[0]);
  }

  struct ggml_tensor* attn = ggml_flash_attn_ext(
      bc.ctx, ensure_cont(bc.ctx, q), k, v, mask, scale, 0.0f, 0.0f);
  ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
  if (bc.verbose) fprintf(stderr, "[ggml_backend] SDPA output: ne=[%lld,%lld,%lld,%lld] type=%d\n",
          (long long)attn->ne[0], (long long)attn->ne[1],
          (long long)attn->ne[2], (long long)attn->ne[3], attn->type);
  auto* gt = safe_ggml_permute(bc.ctx, attn, 0, 2, 1, 3, "LLAMA_ATTN");
  if (sdpa_orig_type == GGML_TYPE_BF16 && !bc.skip_bf16_castback) {
    gt = ggml_cast(bc.ctx, gt, GGML_TYPE_BF16);
  }
  return gt;
}

// ---------------------------------------------------------------------------
// UPDATE_CACHE
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_update_cache(BuildContext& bc) {
  const auto* t = bc.ir_tensor;
  if (bc.srcs.size() < 3) {
    fprintf(stderr, "[executorch-ggml] UPDATE_CACHE: expected 3 sources\n");
    return nullptr;
  }

  struct ggml_tensor* cache = bc.srcs[0];
  struct ggml_tensor* value = bc.srcs[1];
  struct ggml_tensor* start_pos_tensor = bc.srcs[2];

  int32_t seq_dim_pytorch = 1;
  if (t->op_params() && t->op_params()->size() >= 4) {
    memcpy(&seq_dim_pytorch, t->op_params()->data(), 4);
  }

  int ndim = 4;
  int ggml_axis = (ndim - 1) - seq_dim_pytorch;

  fprintf(stderr, "[UPDATE_CACHE_DEBUG] start_pos tensor: type=%s, ne=[%ld,%ld,%ld,%ld], nbytes=%zu\n",
          ggml_type_name(start_pos_tensor->type), start_pos_tensor->ne[0], start_pos_tensor->ne[1],
          start_pos_tensor->ne[2], start_pos_tensor->ne[3], ggml_nbytes(start_pos_tensor));

  int64_t start_pos = 0;
  if (start_pos_tensor->type == GGML_TYPE_I64) {
    start_pos = bc.host_acc.read_i64(start_pos_tensor);
    fprintf(stderr, "[UPDATE_CACHE_DEBUG] read as I64: start_pos=%ld\n", start_pos);
  } else if (start_pos_tensor->type == GGML_TYPE_I32) {
    int32_t start_pos_i32 = bc.host_acc.read_i32(start_pos_tensor);
    start_pos = static_cast<int64_t>(start_pos_i32);
    fprintf(stderr, "[UPDATE_CACHE_DEBUG] read as I32: start_pos_i32=%d -> start_pos=%ld\n", start_pos_i32, start_pos);
  }

  const void* start_pos_data = bc.host_acc.get(start_pos_tensor);
  if (start_pos_data && ggml_nbytes(start_pos_tensor) >= 4) {
    const uint8_t* bytes = static_cast<const uint8_t*>(start_pos_data);
    fprintf(stderr, "[UPDATE_CACHE_DEBUG] start_pos raw bytes: %02x %02x %02x %02x\n",
            bytes[0], bytes[1], bytes[2], bytes[3]);
  }

  int64_t seq_len_new = value->ne[ggml_axis];
  struct ggml_tensor* indices = ggml_arange(bc.ctx,
      (float)start_pos, (float)(start_pos + seq_len_new), 1.0f);
  indices = safe_ggml_cast(bc.ctx, indices, GGML_TYPE_I32, &bc.host_acc);

  if (value->type != GGML_TYPE_F32) {
    value = safe_ggml_cast(bc.ctx, value, GGML_TYPE_F32, &bc.host_acc);
  }

  struct ggml_tensor* gt = nullptr;
  if (ggml_axis == 1) {
    gt = ggml_set_rows(bc.ctx, cache, value, indices);
  } else {
    size_t offset_bytes = start_pos * cache->nb[ggml_axis];
    int64_t view_ne[4];
    for (int d = 0; d < 4; d++) view_ne[d] = cache->ne[d];
    view_ne[ggml_axis] = seq_len_new;

    auto* cache_slice = safe_ggml_view_4d(bc.ctx, cache,
        view_ne[0], view_ne[1], view_ne[2], view_ne[3],
        cache->nb[1], cache->nb[2], cache->nb[3], offset_bytes);
    if (!cache_slice) {
      fprintf(stderr, "KV cache update: view creation failed\n");
      return nullptr;
    }

    auto* val_c = ensure_cont(bc.ctx, value);
    gt = ggml_cpy(bc.ctx, val_c, cache_slice);
    gt = cache;  // return full cache for downstream reads
  }
  return gt;
}

// ---------------------------------------------------------------------------
// ROPE
// ---------------------------------------------------------------------------
static inline struct ggml_tensor* build_op_rope(BuildContext& bc) {
  const auto* t = bc.ir_tensor;
  int32_t n_dims = 0;
  int32_t mode = 0;
  float freq_base = 10000.0f;
  if (t->op_params() && t->op_params()->size() >= 12) {
    const uint8_t* data = t->op_params()->data();
    memcpy(&n_dims, data, sizeof(int32_t));
    memcpy(&mode, data + 4, sizeof(int32_t));
    memcpy(&freq_base, data + 8, sizeof(float));
  }

  struct ggml_tensor* x = ensure_cont(bc.ctx, bc.srcs[0]);
  struct ggml_tensor* pos = bc.srcs[1];
  if (pos->type != GGML_TYPE_I32)
    pos = safe_ggml_cast(bc.ctx, pos, GGML_TYPE_I32);

  return ggml_rope_ext(bc.ctx, x, pos, NULL,
      n_dims, mode, 0, freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
}

} // namespace executorch_ggml
