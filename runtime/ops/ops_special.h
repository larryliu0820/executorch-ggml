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
  // For decoder SDPA with KV cache (T_q < T_kv): the KV cache is already
  // sliced to valid positions [0, start_pos+seq_len), so causality is enforced
  // by construction. Disable causal masking to avoid numerical divergence
  // between PyTorch's is_causal=True GQA SDPA and ggml_flash_attn_ext.
  // Keep causal masking for encoder self-attention (T_q == T_kv) where it's needed.
  if (is_causal && q->ne[1] < k->ne[1]) {
    is_causal = false;
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

  // Auto-slice K/V to valid positions when using KV cache.
  // The Python model slices cache to [0, start_pos+seq_len), but this slice
  // is data-dependent and lost during export. Detect decoder-style SDPA:
  // K lives in mutable_buf (KV cache) and has many more positions than Q.
  // Don't trigger for encoder self-attention (K from const_buf or graph ops).
  bool k_is_mutable = (k->buffer && bc.handle->mutable_buf && k->buffer == bc.handle->mutable_buf);
  // Also check through views and ops that pass through the cache buffer:
  // unwrap VIEW/PERMUTE/RESHAPE/TRANSPOSE and SET_ROWS/CPY (from
  // UPDATE_CACHE) to find the underlying leaf buffer. For SET_ROWS,
  // the cache (destination) tensor is the result shape — check all
  // sources since ggml may store cache as src[0] or src[1].
  if (!k_is_mutable && bc.handle->mutable_buf) {
    auto* kb = k;
    while (kb) {
      if (kb->op == GGML_OP_VIEW || kb->op == GGML_OP_PERMUTE ||
          kb->op == GGML_OP_RESHAPE || kb->op == GGML_OP_TRANSPOSE ||
          kb->op == GGML_OP_CONT || kb->op == GGML_OP_REPEAT) {
        kb = kb->src[0];
      } else if (kb->op == GGML_OP_SET_ROWS || kb->op == GGML_OP_CPY) {
        // Check all sources for the mutable buffer.
        for (int s = 0; s < GGML_MAX_SRC && kb->src[s]; s++) {
          auto* leaf = kb->src[s];
          while (leaf && (leaf->op == GGML_OP_VIEW || leaf->op == GGML_OP_RESHAPE))
            leaf = leaf->src[0];
          if (leaf && leaf->buffer == bc.handle->mutable_buf) {
            k_is_mutable = true;
            break;
          }
        }
        break;
      } else {
        break;
      }
    }
    // Also check the terminal node if we walked past all views
    if (!k_is_mutable && kb && kb->buffer == bc.handle->mutable_buf)
      k_is_mutable = true;
  }
  if (k_is_mutable && k->ne[1] > q->ne[1] * 2) {
    // K has much more positions than Q — this is a KV cache read.
    // Find kv_valid_len from the input pairs (cache_position input).
    // Multiple I32 inputs may have the same element count as q->ne[1]
    // (e.g. both token_ids and cache_position have seq_len elements).
    // Filter by max_value < cache_size to distinguish cache_position
    // (values 0..N) from token_ids (values in vocab range, >> cache_size).
    int64_t kv_valid_len = q->ne[1];  // default: same as Q
    for (auto& [idx, inp] : bc.input_pairs) {
      if (inp->type == GGML_TYPE_I32 && ggml_nelements(inp) == q->ne[1]) {
        const int32_t* pos_data = static_cast<const int32_t*>(bc.host_acc.get(inp));
        if (pos_data) {
          int32_t max_pos = 0;
          for (int64_t i = 0; i < ggml_nelements(inp); i++) {
            if (pos_data[i] > max_pos) max_pos = pos_data[i];
          }
          // Skip inputs whose values exceed the cache size — those are
          // token IDs, not position indices.
          if (max_pos >= k->ne[1]) continue;
          kv_valid_len = max_pos + 1;
          break;
        }
      }
    }
    if (kv_valid_len > 0 && kv_valid_len < k->ne[1]) {
      // Build a combined causal + valid-position mask over the full cache.
      // We cannot auto-slice K/V because INDEX_PUT returns the cache leaf
      // (no graph edge from ggml_cpy to SDPA), so the scheduler may
      // reorder reads before writes across layers. The full-cache mask
      // approach works because CPY nodes are dispatched first in the graph
      // and Metal executes sequentially within a command buffer.
      int64_t T_kv = k->ne[1];
      int64_t T_q  = q->ne[1];
      ggml_set_no_alloc(bc.ctx, false);
      struct ggml_tensor* new_mask = ggml_new_tensor_4d(bc.ctx, GGML_TYPE_F16, T_kv, T_q, 1, 1);
      ggml_set_no_alloc(bc.ctx, true);
      const ggml_fp16_t zero_f16 = ggml_fp32_to_fp16(0.0f);
      const ggml_fp16_t neg_inf_f16 = ggml_fp32_to_fp16(-65504.0f);
      ggml_fp16_t* md = (ggml_fp16_t*)new_mask->data;
      // Find cache_position values for per-query masking.
      std::vector<int32_t> positions(T_q, 0);
      for (auto& [idx2, inp2] : bc.input_pairs) {
        if (inp2->type == GGML_TYPE_I32 && ggml_nelements(inp2) == T_q) {
          const int32_t* pd = static_cast<const int32_t*>(bc.host_acc.get(inp2));
          if (pd) {
            int32_t mx = 0;
            for (int64_t i = 0; i < T_q; i++) mx = std::max(mx, pd[i]);
            if (mx < (int32_t)T_kv) {
              for (int64_t i = 0; i < T_q; i++) positions[i] = pd[i];
              break;
            }
          }
        }
      }
      // mask[col, row] = 0 if col <= positions[row], -inf otherwise
      for (int64_t row = 0; row < T_q; row++) {
        for (int64_t col = 0; col < T_kv; col++) {
          md[row * T_kv + col] = (col <= positions[row]) ? zero_f16 : neg_inf_f16;
        }
      }
      mask = new_mask;
      is_causal = false;
      // Mark mask as input-derived so the graph gets rebuilt when
      // positions change (each decode step has different valid positions).
      bc.input_derived.insert(new_mask);
    }
  }

  // Expand broadcast mask: ggml_flash_attn_ext requires mask->ne[1] >= Q->ne[1].
  // Some models produce masks with ne[1]=1 (broadcast along n_queries).
  // Repeat-expand the mask to match the actual number of query tokens.
  if (mask && mask->ne[1] < q->ne[1]) {
    mask = ggml_cont(bc.ctx, ggml_repeat(bc.ctx, mask,
        ggml_new_tensor_4d(bc.ctx, mask->type, mask->ne[0], q->ne[1], mask->ne[2], mask->ne[3])));
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

  int64_t start_pos = 0;
  if (start_pos_tensor->type == GGML_TYPE_I64) {
    start_pos = bc.host_acc.read_i64(start_pos_tensor);
  } else if (start_pos_tensor->type == GGML_TYPE_I32) {
    start_pos = static_cast<int64_t>(bc.host_acc.read_i32(start_pos_tensor));
  }

  int64_t seq_len_new = value->ne[ggml_axis];

  // Write new values into the cache leaf in-place via ggml_cpy.
  // Using ggml_cpy (not ggml_set_rows) preserves all prior positions in
  // mutable_buf so downstream SDPA reads the full cache history.
  size_t offset_bytes = start_pos * cache->nb[ggml_axis];
  int64_t view_ne[4];
  for (int d = 0; d < 4; d++) view_ne[d] = cache->ne[d];
  view_ne[ggml_axis] = seq_len_new;
  auto* cache_slice = safe_ggml_view_4d(bc.ctx, cache,
      view_ne[0], view_ne[1], view_ne[2], view_ne[3],
      cache->nb[1], cache->nb[2], cache->nb[3], offset_bytes);
  if (!cache_slice) return nullptr;
  auto* cpy = ggml_cpy(bc.ctx, ensure_cont(bc.ctx, value), cache_slice);
  ggml_set_output(cpy);
  return cache;
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
