#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#include <ggml.h>
#include <ggml-backend.h>

namespace executorch_ggml {

// ---------------------------------------------------------------------------
// Shared scalar read/write helpers (needed by custom ops and other code)
// ---------------------------------------------------------------------------

static constexpr int64_t kInvalidIndex = std::numeric_limits<int64_t>::min();

// Read one index value from an index tensor at the broadcasted output coords.
static inline int64_t read_index_value(
    const struct ggml_tensor* idx,
    const int64_t coords[4]) {
  size_t off = 0;
  for (int ax = 0; ax < 4; ++ax) {
    const int64_t ne = idx->ne[ax];
    int64_t c = coords[ax];
    if (ne == 1) {
      c = 0;
    } else if (c < 0 || c >= ne) {
      return kInvalidIndex;
    }
    off += static_cast<size_t>(c) * static_cast<size_t>(idx->nb[ax]);
  }

  const char* base = static_cast<const char*>(idx->data);
  if (base == nullptr) {
    return kInvalidIndex;
  }

  switch (idx->type) {
    case GGML_TYPE_I32:
      return static_cast<int64_t>(*reinterpret_cast<const int32_t*>(base + off));
    case GGML_TYPE_I64:
      return *reinterpret_cast<const int64_t*>(base + off);
    case GGML_TYPE_F32:
      return static_cast<int64_t>(*reinterpret_cast<const float*>(base + off));
    case GGML_TYPE_F16:
      return static_cast<int64_t>(
          ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t*>(base + off)));
    case GGML_TYPE_BF16:
      return static_cast<int64_t>(
          ggml_bf16_to_fp32(*reinterpret_cast<const ggml_bf16_t*>(base + off)));
    default:
      return kInvalidIndex;
  }
}

static inline double read_scalar_f64_ptr(const char* p, ggml_type ty) {
  switch (ty) {
    case GGML_TYPE_I32:  return static_cast<double>(*reinterpret_cast<const int32_t*>(p));
    case GGML_TYPE_I64:  return static_cast<double>(*reinterpret_cast<const int64_t*>(p));
    case GGML_TYPE_F32:  return static_cast<double>(*reinterpret_cast<const float*>(p));
    case GGML_TYPE_F16:  return static_cast<double>(ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t*>(p)));
    case GGML_TYPE_BF16: return static_cast<double>(ggml_bf16_to_fp32(*reinterpret_cast<const ggml_bf16_t*>(p)));
    default:             return 0.0;
  }
}

static inline void write_scalar_f64_ptr(char* p, ggml_type ty, double v) {
  switch (ty) {
    case GGML_TYPE_I32:
      *reinterpret_cast<int32_t*>(p) = static_cast<int32_t>(v);
      break;
    case GGML_TYPE_I64:
      *reinterpret_cast<int64_t*>(p) = static_cast<int64_t>(v);
      break;
    case GGML_TYPE_F32:
      *reinterpret_cast<float*>(p) = static_cast<float>(v);
      break;
    case GGML_TYPE_F16:
      *reinterpret_cast<ggml_fp16_t*>(p) = ggml_fp32_to_fp16(static_cast<float>(v));
      break;
    case GGML_TYPE_BF16:
      *reinterpret_cast<ggml_bf16_t*>(p) = ggml_fp32_to_bf16(static_cast<float>(v));
      break;
    default:
      break;
  }
}

static inline size_t broadcast_offset_bytes(
    const struct ggml_tensor* t,
    int64_t d0,
    int64_t d1,
    int64_t d2,
    int64_t d3) {
  const int64_t c0 = (t->ne[0] <= 1) ? 0 : (d0 % t->ne[0]);
  const int64_t c1 = (t->ne[1] <= 1) ? 0 : (d1 % t->ne[1]);
  const int64_t c2 = (t->ne[2] <= 1) ? 0 : (d2 % t->ne[2]);
  const int64_t c3 = (t->ne[3] <= 1) ? 0 : (d3 % t->ne[3]);
  return
      static_cast<size_t>(c0) * static_cast<size_t>(t->nb[0]) +
      static_cast<size_t>(c1) * static_cast<size_t>(t->nb[1]) +
      static_cast<size_t>(c2) * static_cast<size_t>(t->nb[2]) +
      static_cast<size_t>(c3) * static_cast<size_t>(t->nb[3]);
}

// ---------------------------------------------------------------------------
// Custom op callbacks for comparison / bitwise / logical ops.
//
// These run at graph-compute time (not build time) because their source
// tensors may be ggml graph ops whose data is only available after upstream
// ops have executed.  ggml custom ops always run on CPU -- data is
// automatically transferred by the backend scheduler.
// ---------------------------------------------------------------------------

// Helper: read a contiguous float64 value from an I32/I64/F32 tensor.
static inline double _read_f64(const void* data, ggml_type type, size_t i) {
  switch (type) {
    case GGML_TYPE_I32: return (double)((const int32_t*)data)[i];
    case GGML_TYPE_I64: return (double)((const int64_t*)data)[i];
    case GGML_TYPE_F32: return (double)((const float*)data)[i];
    default: return 0.0;
  }
}

#define DEFINE_CMP_CUSTOM_OP(name, op_expr)                                  \
  static void ggml_custom_##name(                                            \
      struct ggml_tensor* dst, int ith, int nth, void* /*ud*/) {             \
    const struct ggml_tensor* a = dst->src[0];                               \
    const struct ggml_tensor* b = dst->src[1];                               \
    if (!a || !b || !a->data || !b->data || !dst->data) return;              \
    const char* a_base = static_cast<const char*>(a->data);                  \
    const char* b_base = static_cast<const char*>(b->data);                  \
    char* out_base = static_cast<char*>(dst->data);                          \
    const int64_t n = ggml_nelements(dst);                                   \
    const int64_t dr = (n + nth - 1) / nth;                                  \
    const int64_t i0 = dr * ith;                                             \
    const int64_t i1 = std::min<int64_t>(i0 + dr, n);                       \
    for (int64_t i = i0; i < i1; ++i) {                                      \
      int64_t d0 = i % dst->ne[0];                                          \
      int64_t d1 = (i / dst->ne[0]) % dst->ne[1];                           \
      int64_t d2 = (i / (dst->ne[0]*dst->ne[1])) % dst->ne[2];             \
      int64_t d3 = i / (dst->ne[0]*dst->ne[1]*dst->ne[2]);                  \
      const size_t ao = broadcast_offset_bytes(a, d0, d1, d2, d3);          \
      const size_t bo = broadcast_offset_bytes(b, d0, d1, d2, d3);          \
      const size_t oo = broadcast_offset_bytes(dst, d0, d1, d2, d3);        \
      const double va = read_scalar_f64_ptr(a_base + ao, a->type);          \
      const double vb = read_scalar_f64_ptr(b_base + bo, b->type);          \
      *reinterpret_cast<int32_t*>(out_base + oo) = (op_expr) ? 1 : 0;       \
    }                                                                        \
  }

DEFINE_CMP_CUSTOM_OP(le, va <= vb)
DEFINE_CMP_CUSTOM_OP(lt, va <  vb)
DEFINE_CMP_CUSTOM_OP(gt, va >  vb)
DEFINE_CMP_CUSTOM_OP(ge, va >= vb)
DEFINE_CMP_CUSTOM_OP(eq, va == vb)
DEFINE_CMP_CUSTOM_OP(ne_op, va != vb)

#undef DEFINE_CMP_CUSTOM_OP

// Cumulative sum along a ggml axis.
// op_params[0] = int32 ggml_axis.
static void ggml_custom_cumsum(
    struct ggml_tensor* dst, int ith, int nth, void* /*ud*/) {
  if (ith != 0) return;  // single-threaded for simplicity
  const struct ggml_tensor* src = dst->src[0];
  if (!src || !src->data || !dst->data) return;

  int32_t ggml_axis = 0;
  memcpy(&ggml_axis, dst->op_params, sizeof(int32_t));

  const int64_t nelem = ggml_nelements(dst);
  int64_t stride = 1;
  for (int ax = 0; ax < ggml_axis; ++ax) stride *= src->ne[ax];
  int64_t dim_size = src->ne[ggml_axis];
  int64_t outer_size = nelem / (dim_size * stride);

  if (src->type == GGML_TYPE_I32) {
    const int32_t* in_data = static_cast<const int32_t*>(src->data);
    int32_t* out_data = static_cast<int32_t*>(dst->data);
    for (int64_t outer = 0; outer < outer_size; ++outer) {
      for (int64_t inner = 0; inner < stride; ++inner) {
        int32_t cumsum = 0;
        for (int64_t d = 0; d < dim_size; ++d) {
          int64_t idx = outer * dim_size * stride + d * stride + inner;
          cumsum += in_data[idx];
          out_data[idx] = cumsum;
        }
      }
    }
  } else {
    const float* in_data = static_cast<const float*>(src->data);
    float* out_data = static_cast<float*>(dst->data);
    for (int64_t outer = 0; outer < outer_size; ++outer) {
      for (int64_t inner = 0; inner < stride; ++inner) {
        float cumsum = 0.0f;
        for (int64_t d = 0; d < dim_size; ++d) {
          int64_t idx = outer * dim_size * stride + d * stride + inner;
          cumsum += in_data[idx];
          out_data[idx] = cumsum;
        }
      }
    }
  }
}

// Convert I32 boolean mask (1=attend, 0=don't attend) to F16 additive mask
// (0.0=attend, -inf=don't attend) as required by ggml_flash_attn_ext.
static void ggml_custom_bool_to_additive_mask(
    struct ggml_tensor* dst, int ith, int nth, void* /*ud*/) {
  const struct ggml_tensor* src = dst->src[0];
  if (!src || !src->data || !dst->data) return;
  const char* src_base = static_cast<const char*>(src->data);
  char* out_base = static_cast<char*>(dst->data);
  const int64_t n = ggml_nelements(dst);
  const int64_t dr = (n + nth - 1) / nth;
  const int64_t i0 = dr * ith;
  const int64_t i1 = std::min<int64_t>(i0 + dr, n);
  const ggml_fp16_t zero_f16 = ggml_fp32_to_fp16(0.0f);
  const ggml_fp16_t neg_inf_f16 = ggml_fp32_to_fp16(-65504.0f);
  for (int64_t i = i0; i < i1; ++i) {
    int64_t d0 = i % dst->ne[0];
    int64_t d1 = (i / dst->ne[0]) % dst->ne[1];
    int64_t d2 = (i / (dst->ne[0] * dst->ne[1])) % dst->ne[2];
    int64_t d3 = i / (dst->ne[0] * dst->ne[1] * dst->ne[2]);
    const size_t so = broadcast_offset_bytes(src, d0, d1, d2, d3);
    const size_t oo = broadcast_offset_bytes(dst, d0, d1, d2, d3);
    const double v = read_scalar_f64_ptr(src_base + so, src->type);
    *reinterpret_cast<ggml_fp16_t*>(out_base + oo) =
        (v != 0.0) ? zero_f16 : neg_inf_f16;
  }
}

static void ggml_custom_bitwise_and(
    struct ggml_tensor* dst, int ith, int nth, void* /*ud*/) {
  const struct ggml_tensor* a = dst->src[0];
  const struct ggml_tensor* b = dst->src[1];
  if (!a || !b || !a->data || !b->data || !dst->data) return;
  const char* a_base = static_cast<const char*>(a->data);
  const char* b_base = static_cast<const char*>(b->data);
  char* out_base = static_cast<char*>(dst->data);
  const int64_t n = ggml_nelements(dst);
  const int64_t dr = (n + nth - 1) / nth;
  const int64_t i0 = dr * ith, i1 = std::min<int64_t>(i0 + dr, n);
  for (int64_t i = i0; i < i1; ++i) {
    int64_t d0 = i % dst->ne[0];
    int64_t d1 = (i / dst->ne[0]) % dst->ne[1];
    int64_t d2 = (i / (dst->ne[0] * dst->ne[1])) % dst->ne[2];
    int64_t d3 = i / (dst->ne[0] * dst->ne[1] * dst->ne[2]);
    const size_t ao = broadcast_offset_bytes(a, d0, d1, d2, d3);
    const size_t bo = broadcast_offset_bytes(b, d0, d1, d2, d3);
    const size_t oo = broadcast_offset_bytes(dst, d0, d1, d2, d3);
    const int32_t av = *reinterpret_cast<const int32_t*>(a_base + ao);
    const int32_t bv = *reinterpret_cast<const int32_t*>(b_base + bo);
    *reinterpret_cast<int32_t*>(out_base + oo) = av & bv;
  }
}

static void ggml_custom_bitwise_or(
    struct ggml_tensor* dst, int ith, int nth, void* /*ud*/) {
  const struct ggml_tensor* a = dst->src[0];
  const struct ggml_tensor* b = dst->src[1];
  if (!a || !b || !a->data || !b->data || !dst->data) return;
  const char* a_base = static_cast<const char*>(a->data);
  const char* b_base = static_cast<const char*>(b->data);
  char* out_base = static_cast<char*>(dst->data);
  const int64_t n = ggml_nelements(dst);
  const int64_t dr = (n + nth - 1) / nth;
  const int64_t i0 = dr * ith, i1 = std::min<int64_t>(i0 + dr, n);
  for (int64_t i = i0; i < i1; ++i) {
    int64_t d0 = i % dst->ne[0];
    int64_t d1 = (i / dst->ne[0]) % dst->ne[1];
    int64_t d2 = (i / (dst->ne[0] * dst->ne[1])) % dst->ne[2];
    int64_t d3 = i / (dst->ne[0] * dst->ne[1] * dst->ne[2]);
    const size_t ao = broadcast_offset_bytes(a, d0, d1, d2, d3);
    const size_t bo = broadcast_offset_bytes(b, d0, d1, d2, d3);
    const size_t oo = broadcast_offset_bytes(dst, d0, d1, d2, d3);
    const int32_t av = *reinterpret_cast<const int32_t*>(a_base + ao);
    const int32_t bv = *reinterpret_cast<const int32_t*>(b_base + bo);
    *reinterpret_cast<int32_t*>(out_base + oo) = av | bv;
  }
}

static void ggml_custom_logical_not(
    struct ggml_tensor* dst, int ith, int nth, void* /*ud*/) {
  const struct ggml_tensor* a = dst->src[0];
  if (!a || !a->data || !dst->data) return;
  const char* a_base = static_cast<const char*>(a->data);
  char* out_base = static_cast<char*>(dst->data);
  const int64_t n = ggml_nelements(dst);
  const int64_t dr = (n + nth - 1) / nth;
  const int64_t i0 = dr * ith, i1 = std::min<int64_t>(i0 + dr, n);
  for (int64_t i = i0; i < i1; ++i) {
    int64_t d0 = i % dst->ne[0];
    int64_t d1 = (i / dst->ne[0]) % dst->ne[1];
    int64_t d2 = (i / (dst->ne[0] * dst->ne[1])) % dst->ne[2];
    int64_t d3 = i / (dst->ne[0] * dst->ne[1] * dst->ne[2]);
    const size_t ao = broadcast_offset_bytes(a, d0, d1, d2, d3);
    const size_t oo = broadcast_offset_bytes(dst, d0, d1, d2, d3);
    const int32_t av = *reinterpret_cast<const int32_t*>(a_base + ao);
    *reinterpret_cast<int32_t*>(out_base + oo) = (av == 0) ? 1 : 0;
  }
}

// F32 depthwise conv1d: avoids ggml's im2col F16 path for better precision.
// src[0] = weight [K, 1, C, 1]  (ggml layout for [C, 1, K] PyTorch)
// src[1] = input  [L, C, B, 1]  (ggml layout for [B, C, L] PyTorch)
// userdata = Conv1dDwParams* with stride, pad, dilation.
struct Conv1dDwParams { int stride, pad, dilation; };

static void conv_1d_dw_kernel(
    const float* w_data, const float* x_data, float* y_data,
    int64_t K, int64_t C, int64_t L_in, int64_t B, int64_t L_out,
    const size_t* w_nb, const size_t* x_nb, const size_t* y_nb,
    int s, int pad, int d,
    int ith, int nth) {
  const int64_t total = B * C;
  const int64_t work  = (total + nth - 1) / nth;
  const int64_t i0 = work * ith, i1 = std::min<int64_t>(i0 + work, total);

  for (int64_t bc = i0; bc < i1; ++bc) {
    const int64_t c = bc % C;
    const int64_t b = bc / C;
    const float* w = (const float*)((const char*)w_data + c * w_nb[2]);
    const float* x = (const float*)((const char*)x_data + c * x_nb[1] + b * x_nb[2]);
    float* y = (float*)((char*)y_data + c * y_nb[1] + b * y_nb[2]);
    for (int64_t o = 0; o < L_out; ++o) {
      float sum = 0.0f;
      for (int64_t k = 0; k < K; ++k) {
        int64_t in_idx = o * s - pad + k * d;
        if (in_idx >= 0 && in_idx < L_in) {
          sum += w[k] * x[in_idx];
        }
      }
      y[o] = sum;
    }
  }
}

static void ggml_custom_conv_1d_dw_f32(
    struct ggml_tensor* dst, int ith, int nth, void* userdata) {
  const auto* p = static_cast<const Conv1dDwParams*>(userdata);
  const struct ggml_tensor* weight = dst->src[0];
  const struct ggml_tensor* input  = dst->src[1];
  if (!weight || !input) return;

  const int64_t K     = weight->ne[0];
  const int64_t C     = weight->ne[2];
  const int64_t L_in  = input->ne[0];
  const int64_t B     = input->ne[2];
  const int64_t L_out = dst->ne[0];

  // Check if tensors are on host memory
  bool w_host = !weight->buffer || ggml_backend_buffer_is_host(weight->buffer);
  bool x_host = !input->buffer  || ggml_backend_buffer_is_host(input->buffer);
  bool y_host = !dst->buffer    || ggml_backend_buffer_is_host(dst->buffer);

  if (w_host && x_host && y_host) {
    // Host path: scheduler provided host buffers for the custom op.
    if (!weight->data || !input->data || !dst->data) return;
    conv_1d_dw_kernel(
        (const float*)weight->data, (const float*)input->data, (float*)dst->data,
        K, C, L_in, B, L_out,
        weight->nb, input->nb, dst->nb,
        p->stride, p->pad, p->dilation, ith, nth);
  } else if (ith == 0) {
    // Device path: copy data to host, compute, copy back (single-threaded)
    fprintf(stderr, "[conv_dw] device path: w_host=%d x_host=%d y_host=%d\n",
            w_host, x_host, y_host);
    std::vector<float> w_buf(ggml_nelements(weight));
    std::vector<float> x_buf(ggml_nelements(input));
    std::vector<float> y_buf(ggml_nelements(dst), 0.0f);
    ggml_backend_tensor_get(weight, w_buf.data(), 0, ggml_nbytes(weight));
    ggml_backend_tensor_get(input, x_buf.data(), 0, ggml_nbytes(input));

    // Compute strides for contiguous layout
    size_t w_nb[4] = {sizeof(float), (size_t)(K*sizeof(float)),
                       (size_t)(K*1*sizeof(float)), (size_t)(K*1*C*sizeof(float))};
    size_t x_nb[4] = {sizeof(float), (size_t)(L_in*sizeof(float)),
                       (size_t)(L_in*C*sizeof(float)), (size_t)(L_in*C*B*sizeof(float))};
    size_t y_nb[4] = {sizeof(float), (size_t)(L_out*sizeof(float)),
                       (size_t)(L_out*C*sizeof(float)), (size_t)(L_out*C*B*sizeof(float))};

    conv_1d_dw_kernel(
        w_buf.data(), x_buf.data(), y_buf.data(),
        K, C, L_in, B, L_out,
        w_nb, x_nb, y_nb,
        p->stride, p->pad, p->dilation, 0, 1);

    ggml_backend_tensor_set(dst, y_buf.data(), 0, ggml_nbytes(dst));
  }
}

// Custom runtime gather for multi-index advanced indexing:
//   out = src[idx0, idx1, ...]
// where dst->src[0] is src and dst->src[1..] are index tensors.
static void ggml_custom_index_multi(
    struct ggml_tensor* dst,
    int ith,
    int nth,
    void* userdata) {
  (void)userdata;

  const struct ggml_tensor* src = dst->src[0];
  if (src == nullptr || src->data == nullptr || dst->data == nullptr) {
    return;
  }

  int n_indices = 0;
  for (int s = 1; s < GGML_MAX_SRC && dst->src[s] != nullptr; ++s) {
    ++n_indices;
  }
  if (n_indices <= 0 || n_indices > 4) {
    return;
  }

  // Keep implementation conservative for now: plain element-wise path on
  // non-quantized scalars.
  if (ggml_is_quantized(src->type) || ggml_is_quantized(dst->type)) {
    return;
  }

  const size_t dst_elem_size = ggml_type_size(dst->type);
  const int64_t n_out = ggml_nelements(dst);
  const int64_t dr = (n_out + nth - 1) / nth;
  const int64_t ir0 = dr * ith;
  const int64_t ir1 = std::min<int64_t>(ir0 + dr, n_out);

  const char* src_base = static_cast<const char*>(src->data);
  char* dst_base = static_cast<char*>(dst->data);

  for (int64_t i = ir0; i < ir1; ++i) {
    int64_t tmp = i;
    int64_t coords[4] = {0, 0, 0, 0};
    coords[0] = tmp % dst->ne[0];
    tmp /= dst->ne[0];
    coords[1] = tmp % dst->ne[1];
    tmp /= dst->ne[1];
    coords[2] = tmp % dst->ne[2];
    tmp /= dst->ne[2];
    coords[3] = tmp;

    size_t dst_off = 0;
    for (int ax = 0; ax < 4; ++ax) {
      dst_off += static_cast<size_t>(coords[ax]) * static_cast<size_t>(dst->nb[ax]);
    }

    size_t src_off = 0;
    bool valid = true;
    for (int k = 0; k < n_indices; ++k) {
      const struct ggml_tensor* idx = dst->src[1 + k];
      const int64_t idx_val_raw = read_index_value(idx, coords);
      if (idx_val_raw == kInvalidIndex) {
        valid = false;
        break;
      }

      const int src_ax = n_indices - 1 - k; // ggml axis for PyTorch dim k
      const int64_t dim_size = src->ne[src_ax];
      int64_t idx_val = idx_val_raw;
      if (idx_val < 0) {
        idx_val += dim_size;
      }
      if (idx_val < 0 || idx_val >= dim_size) {
        valid = false;
        break;
      }
      src_off += static_cast<size_t>(idx_val) * static_cast<size_t>(src->nb[src_ax]);
    }

    if (valid) {
      char* dst_ptr = dst_base + dst_off;
      const char* src_ptr = src_base + src_off;
      if (src->type == dst->type) {
        memcpy(dst_ptr, src_ptr, dst_elem_size);
      } else {
        const double v = read_scalar_f64_ptr(src_ptr, src->type);
        write_scalar_f64_ptr(dst_ptr, dst->type, v);
      }
    } else {
      memset(dst_base + dst_off, 0, dst_elem_size);
    }
  }
}

// Runtime scatter update for index_put-style cache writes:
//   out = cache; out[..., idx, ...] = val
// src[0] = cache, src[1] = idx (I32/I64), src[2] = val
static void ggml_custom_index_put_rows(
    struct ggml_tensor* dst,
    int ith,
    int nth,
    void* userdata) {
  (void)nth;
  (void)userdata;

  // Single-threaded update for correctness (small tensor in test path).
  if (ith != 0) {
    return;
  }

  const struct ggml_tensor* cache = dst->src[0];
  const struct ggml_tensor* idx = dst->src[1];
  const struct ggml_tensor* val = dst->src[2];
  if (!cache || !idx || !val || !cache->data || !idx->data || !val->data || !dst->data) {
    return;
  }
  if (ggml_is_quantized(cache->type) || ggml_is_quantized(val->type) || ggml_is_quantized(dst->type)) {
    return;
  }

  // Start from the previous cache contents.
  if (cache->data != dst->data) {
    memcpy(dst->data, cache->data, ggml_nbytes(dst));
  }

  const int64_t ne0 = dst->ne[0];
  const int64_t n_rows = val->ne[1];

  const char* val_base = static_cast<const char*>(val->data);
  char* dst_base = static_cast<char*>(dst->data);

  for (int64_t i03 = 0; i03 < val->ne[3]; ++i03) {
    for (int64_t i02 = 0; i02 < val->ne[2]; ++i02) {
      for (int64_t i = 0; i < n_rows; ++i) {
        int64_t idx_coords[4] = {
            i,
            idx->ne[1] > 1 ? (i02 % idx->ne[1]) : 0,
            idx->ne[2] > 1 ? (i03 % idx->ne[2]) : 0,
            0,
        };
        int64_t row = read_index_value(idx, idx_coords);
        if (row == kInvalidIndex) {
          continue;
        }
        if (row < 0) {
          row += dst->ne[1];
        }
        if (row < 0 || row >= dst->ne[1]) {
          continue;
        }

        const size_t src_row_off =
            static_cast<size_t>(i) * static_cast<size_t>(val->nb[1]) +
            static_cast<size_t>(i02) * static_cast<size_t>(val->nb[2]) +
            static_cast<size_t>(i03) * static_cast<size_t>(val->nb[3]);
        const size_t dst_row_off =
            static_cast<size_t>(row) * static_cast<size_t>(dst->nb[1]) +
            static_cast<size_t>(i02) * static_cast<size_t>(dst->nb[2]) +
            static_cast<size_t>(i03) * static_cast<size_t>(dst->nb[3]);

        if (val->type == dst->type) {
          for (int64_t d0 = 0; d0 < ne0; ++d0) {
            const char* src_ptr = val_base + src_row_off + static_cast<size_t>(d0) * static_cast<size_t>(val->nb[0]);
            char* dst_ptr = dst_base + dst_row_off + static_cast<size_t>(d0) * static_cast<size_t>(dst->nb[0]);
            memcpy(dst_ptr, src_ptr, ggml_type_size(dst->type));
          }
        } else {
          for (int64_t d0 = 0; d0 < ne0; ++d0) {
            const char* src_ptr = val_base + src_row_off + static_cast<size_t>(d0) * static_cast<size_t>(val->nb[0]);
            char* dst_ptr = dst_base + dst_row_off + static_cast<size_t>(d0) * static_cast<size_t>(dst->nb[0]);
            const double v = read_scalar_f64_ptr(src_ptr, val->type);
            write_scalar_f64_ptr(dst_ptr, dst->type, v);
          }
        }
      }
    }
  }
}

// Scatter-only inplace callback for mutable INDEX_PUT.
// dst is an inplace view of the mutable_buf cache tensor (dst->data == src[0]->data).
// src[1] = idx (I32/I64), src[2] = val.
// Only scatters val rows into dst at idx positions -- no full-cache memcpy needed
// because dst already IS the cache.
static void ggml_custom_index_put_inplace(
    struct ggml_tensor* dst,
    int ith,
    int nth,
    void* userdata) {
  (void)nth;
  (void)userdata;

  if (ith != 0) return;

  const struct ggml_tensor* idx = dst->src[1];
  const struct ggml_tensor* val = dst->src[2];
  if (!idx || !val || !idx->data || !val->data || !dst->data) {
    return;
  }
  if (ggml_is_quantized(dst->type) || ggml_is_quantized(val->type)) return;

  const int64_t ne0 = dst->ne[0];
  const int64_t n_rows = val->ne[1];

  const char* val_base = static_cast<const char*>(val->data);
  char* dst_base = static_cast<char*>(dst->data);

  for (int64_t i03 = 0; i03 < val->ne[3]; ++i03) {
    for (int64_t i02 = 0; i02 < val->ne[2]; ++i02) {
      for (int64_t i = 0; i < n_rows; ++i) {
        int64_t idx_coords[4] = {
            i,
            idx->ne[1] > 1 ? (i02 % idx->ne[1]) : 0,
            idx->ne[2] > 1 ? (i03 % idx->ne[2]) : 0,
            0,
        };
        int64_t row = read_index_value(idx, idx_coords);
        if (row == kInvalidIndex) continue;
        if (row < 0) row += dst->ne[1];
        if (row < 0 || row >= dst->ne[1]) continue;

        const size_t src_row_off =
            static_cast<size_t>(i) * static_cast<size_t>(val->nb[1]) +
            static_cast<size_t>(i02) * static_cast<size_t>(val->nb[2]) +
            static_cast<size_t>(i03) * static_cast<size_t>(val->nb[3]);
        const size_t dst_row_off =
            static_cast<size_t>(row) * static_cast<size_t>(dst->nb[1]) +
            static_cast<size_t>(i02) * static_cast<size_t>(dst->nb[2]) +
            static_cast<size_t>(i03) * static_cast<size_t>(dst->nb[3]);

        if (val->type == dst->type) {
          for (int64_t d0 = 0; d0 < ne0; ++d0) {
            const char* src_ptr = val_base + src_row_off + static_cast<size_t>(d0) * static_cast<size_t>(val->nb[0]);
            char* dst_ptr = dst_base + dst_row_off + static_cast<size_t>(d0) * static_cast<size_t>(dst->nb[0]);
            memcpy(dst_ptr, src_ptr, ggml_type_size(dst->type));
          }
        } else {
          for (int64_t d0 = 0; d0 < ne0; ++d0) {
            const char* src_ptr = val_base + src_row_off + static_cast<size_t>(d0) * static_cast<size_t>(val->nb[0]);
            char* dst_ptr = dst_base + dst_row_off + static_cast<size_t>(d0) * static_cast<size_t>(dst->nb[0]);
            const double v = read_scalar_f64_ptr(src_ptr, val->type);
            write_scalar_f64_ptr(dst_ptr, dst->type, v);
          }
        }
      }
    }
  }
}

} // namespace executorch_ggml
