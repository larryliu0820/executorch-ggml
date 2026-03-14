/**
 * GgmlBackendInterface — ExecuTorch runtime backend delegating to ggml.
 *
 * Deserialises a FlatBuffer-encoded ggml IR graph produced by the Python
 * GgmlBackend.preprocess() and executes it using the ggml compute graph API.
 */

#include "ggml_backend.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <limits>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ggml_ir_generated.h"  // flatc-generated header (checked in under schema/)

#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-alloc.h>
#include <ggml-cpu.h>
#include <ggml-backend.h>

#ifdef GGML_USE_METAL
#include <ggml-metal.h>
#endif

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>

namespace executorch_ggml {

using executorch::runtime::ArrayRef;
using executorch::runtime::BackendExecutionContext;
using executorch::runtime::BackendInitContext;
using executorch::runtime::CompileSpec;
using executorch::runtime::DelegateHandle;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::Result;
using executorch::runtime::Span;

namespace {

constexpr int64_t kInvalidIndex = std::numeric_limits<int64_t>::min();

// Read one index value from an index tensor at the broadcasted output coords.
inline int64_t read_index_value(
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

inline double read_scalar_f64_ptr(const char* p, ggml_type ty) {
  switch (ty) {
    case GGML_TYPE_I32:  return static_cast<double>(*reinterpret_cast<const int32_t*>(p));
    case GGML_TYPE_I64:  return static_cast<double>(*reinterpret_cast<const int64_t*>(p));
    case GGML_TYPE_F32:  return static_cast<double>(*reinterpret_cast<const float*>(p));
    case GGML_TYPE_F16:  return static_cast<double>(ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t*>(p)));
    case GGML_TYPE_BF16: return static_cast<double>(ggml_bf16_to_fp32(*reinterpret_cast<const ggml_bf16_t*>(p)));
    default:             return 0.0;
  }
}

inline void write_scalar_f64_ptr(char* p, ggml_type ty, double v) {
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

// ---------------------------------------------------------------------------
// Symbolic expression bytecode evaluator
// ---------------------------------------------------------------------------
// Opcodes match the Python SYM_OP_* constants in ggml_backend.py.

enum SymExprOp : uint8_t {
  SYM_PUSH_SYM   = 0x01,  // 1-byte operand: sym_id
  SYM_PUSH_CONST = 0x02,  // 4-byte operand: int32 LE
  SYM_ADD        = 0x10,
  SYM_SUB        = 0x11,
  SYM_MUL        = 0x12,
  SYM_FLOORDIV   = 0x13,
  SYM_MOD        = 0x14,
  SYM_NEG        = 0x15,
};

// Evaluate postfix bytecode with given symbol values.
// Returns the computed int64_t value, or 0 on error.
static int64_t eval_sym_expr(
    const uint8_t* code, size_t code_len,
    const std::unordered_map<int32_t, int64_t>& sym_values) {
  int64_t stack[16];
  int sp = 0;
  size_t pc = 0;
  while (pc < code_len) {
    uint8_t op = code[pc++];
    switch (op) {
      case SYM_PUSH_SYM: {
        if (pc >= code_len) return 0;
        uint8_t sid = code[pc++];
        auto it = sym_values.find(static_cast<int32_t>(sid));
        stack[sp++] = (it != sym_values.end()) ? it->second : 0;
        break;
      }
      case SYM_PUSH_CONST: {
        if (pc + 4 > code_len) return 0;
        int32_t val;
        memcpy(&val, code + pc, 4);
        pc += 4;
        stack[sp++] = static_cast<int64_t>(val);
        break;
      }
      case SYM_ADD: {
        if (sp < 2) return 0;
        int64_t b = stack[--sp], a = stack[--sp];
        stack[sp++] = a + b;
        break;
      }
      case SYM_SUB: {
        if (sp < 2) return 0;
        int64_t b = stack[--sp], a = stack[--sp];
        stack[sp++] = a - b;
        break;
      }
      case SYM_MUL: {
        if (sp < 2) return 0;
        int64_t b = stack[--sp], a = stack[--sp];
        stack[sp++] = a * b;
        break;
      }
      case SYM_FLOORDIV: {
        if (sp < 2) return 0;
        int64_t b = stack[--sp], a = stack[--sp];
        if (b == 0) return 0;
        // C++ integer division truncates; Python floor-divides toward -inf.
        // For positive divisors (our use case), they're equivalent when a>=0.
        // Handle general case correctly:
        int64_t q = a / b;
        int64_t r = a % b;
        if (r != 0 && ((r ^ b) < 0)) q--;
        stack[sp++] = q;
        break;
      }
      case SYM_MOD: {
        if (sp < 2) return 0;
        int64_t b = stack[--sp], a = stack[--sp];
        if (b == 0) return 0;
        int64_t r = a % b;
        if (r != 0 && ((r ^ b) < 0)) r += b;
        stack[sp++] = r;
        break;
      }
      case SYM_NEG: {
        if (sp < 1) return 0;
        stack[sp - 1] = -stack[sp - 1];
        break;
      }
      default:
        return 0;  // Unknown opcode
    }
    if (sp > 15) return 0;  // Stack overflow
  }
  return (sp == 1) ? stack[0] : 0;
}

// Unpack per-dim bytecode from the packed sym_dim_exprs vector.
// Layout: 4 entries, each prefixed by uint16 length.
// Returns true if bytecode was found for the given dim.
static bool get_dim_expr_bytecode(
    const ::flatbuffers::Vector<uint8_t>* exprs,
    size_t dim,
    const uint8_t*& out_code,
    size_t& out_len) {
  if (!exprs || dim >= 4) return false;
  const uint8_t* data = exprs->data();
  size_t total = exprs->size();
  size_t offset = 0;
  for (size_t d = 0; d < 4 && offset + 2 <= total; ++d) {
    uint16_t len;
    memcpy(&len, data + offset, 2);
    offset += 2;
    if (d == dim) {
      if (len == 0 || offset + len > total) return false;
      out_code = data + offset;
      out_len = len;
      return true;
    }
    offset += len;
  }
  return false;
}

inline size_t broadcast_offset_bytes(
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
// ops have executed.  ggml custom ops always run on CPU — data is
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
  void ggml_custom_##name(                                                   \
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

// Convert I32 boolean mask (1=attend, 0=don't attend) to F16 additive mask
// (0.0=attend, -inf=don't attend) as required by ggml_flash_attn_ext.
void ggml_custom_bool_to_additive_mask(
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
  const ggml_fp16_t neg_inf_f16 = ggml_fp32_to_fp16(-INFINITY);
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

void ggml_custom_bitwise_and(
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

void ggml_custom_bitwise_or(
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

void ggml_custom_logical_not(
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

void ggml_custom_conv_1d_dw_f32(
    struct ggml_tensor* dst, int ith, int nth, void* userdata) {
  const auto* p = static_cast<const Conv1dDwParams*>(userdata);
  const struct ggml_tensor* weight = dst->src[0];
  const struct ggml_tensor* input  = dst->src[1];
  if (!weight || !input || !weight->data || !input->data || !dst->data) return;

  const int64_t K     = weight->ne[0];
  const int64_t C     = weight->ne[2];
  const int64_t L_in  = input->ne[0];
  const int64_t B     = input->ne[2];
  const int64_t L_out = dst->ne[0];
  const int s = p->stride, pad = p->pad, d = p->dilation;

  // Parallelise over (batch, channel) pairs.
  const int64_t total = B * C;
  const int64_t work  = (total + nth - 1) / nth;
  const int64_t i0 = work * ith, i1 = std::min<int64_t>(i0 + work, total);

  for (int64_t bc = i0; bc < i1; ++bc) {
    const int64_t c = bc % C;
    const int64_t b = bc / C;
    const float* w = (const float*)((const char*)weight->data + c * weight->nb[2]);
    const float* x = (const float*)((const char*)input->data
                     + c * input->nb[1] + b * input->nb[2]);
    float* y = (float*)((char*)dst->data + c * dst->nb[1] + b * dst->nb[2]);
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

// Custom runtime gather for multi-index advanced indexing:
//   out = src[idx0, idx1, ...]
// where dst->src[0] is src and dst->src[1..] are index tensors.
void ggml_custom_index_multi(
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
void ggml_custom_index_put_rows(
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

// Eager integer casts on the CPU context.  ggml's CPU backend doesn't support
// GGML_OP_CPY for I64 or I32↔I64, so we must perform these conversions at
// graph-build time (data is available because we use no_alloc=false).
// Returns a new tensor with op=GGML_OP_NONE (treated as a constant leaf).
struct ggml_tensor* eager_cast_i64_to_i32(
    struct ggml_context* ctx,
    struct ggml_tensor* src) {
  ggml_set_no_alloc(ctx, false);
  struct ggml_tensor* dst = ggml_new_tensor(ctx, GGML_TYPE_I32, GGML_MAX_DIMS, src->ne);
  ggml_set_no_alloc(ctx, true);
  dst->op = GGML_OP_NONE;
  if (src->data && dst->data) {
    const size_t n = ggml_nelements(src);
    const int64_t* s = static_cast<const int64_t*>(src->data);
    int32_t* d = static_cast<int32_t*>(dst->data);
    for (size_t i = 0; i < n; ++i) d[i] = static_cast<int32_t>(s[i]);
  }
  return dst;
}

struct ggml_tensor* eager_cast_i32_to_i64(
    struct ggml_context* ctx,
    struct ggml_tensor* src) {
  ggml_set_no_alloc(ctx, false);
  struct ggml_tensor* dst = ggml_new_tensor(ctx, GGML_TYPE_I64, GGML_MAX_DIMS, src->ne);
  ggml_set_no_alloc(ctx, true);
  dst->op = GGML_OP_NONE;
  if (src->data && dst->data) {
    const size_t n = ggml_nelements(src);
    const int32_t* s = static_cast<const int32_t*>(src->data);
    int64_t* d = static_cast<int64_t*>(dst->data);
    for (size_t i = 0; i < n; ++i) d[i] = static_cast<int64_t>(s[i]);
  }
  return dst;
}

// Safe cast that avoids ggml_cast combos the CPU backend can't handle.
// Supported natively: F16↔{F16,BF16,F32}, BF16↔{F16,BF16,F32}, F32↔{F16,BF16,F32,I32}, I32→F32.
// Everything else routes through F32 as intermediate or uses eager helpers.
struct ggml_tensor* safe_ggml_cast(
    struct ggml_context* ctx,
    struct ggml_tensor* src,
    ggml_type target) {
  if (src->type == target) return src;
  // I64 source: eager CPU conversion
  if (src->type == GGML_TYPE_I64 && target == GGML_TYPE_I32) return eager_cast_i64_to_i32(ctx, src);
  if (src->type == GGML_TYPE_I64) {
    // I64 → F32 eager, then F32 → target via ggml_cast
    auto* i32 = eager_cast_i64_to_i32(ctx, src);
    return (target == GGML_TYPE_F32) ? safe_ggml_cast(ctx, i32, GGML_TYPE_F32)
                                     : safe_ggml_cast(ctx, safe_ggml_cast(ctx, i32, GGML_TYPE_F32), target);
  }
  // I32 source: only I32→F32 is native in ggml_cast
  if (src->type == GGML_TYPE_I32 && target == GGML_TYPE_I64) return eager_cast_i32_to_i64(ctx, src);
  if (src->type == GGML_TYPE_I32 && target != GGML_TYPE_F32) {
    return safe_ggml_cast(ctx, safe_ggml_cast(ctx, src, GGML_TYPE_F32), target);
  }
  // F32/F16/BF16 → I64: go through I32 first, then eager I32→I64
  if (target == GGML_TYPE_I64) {
    auto* i32 = safe_ggml_cast(ctx, src, GGML_TYPE_I32);
    return eager_cast_i32_to_i64(ctx, i32);
  }
  // Eager scalar cast for I32→F32 (enc_len computation chains).
  if (src->type == GGML_TYPE_I32 && target == GGML_TYPE_F32 &&
      ggml_nelements(src) == 1 && src->data) {
    int32_t ival = *(const int32_t*)src->data;
    ggml_set_no_alloc(ctx, false);
    struct ggml_tensor* dst = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, 1, 1);
    ggml_set_no_alloc(ctx, true);
    dst->op = GGML_OP_NONE;
    *(float*)dst->data = static_cast<float>(ival);
    return dst;
  }
  // Everything else (F16/BF16/F32 inter-conversions) is natively supported
  return ggml_cast(ctx, src, target);
}

} // namespace

// ---------------------------------------------------------------------------
// Delegate handle — owns the ggml context, graph, and tensor bookkeeping
// ---------------------------------------------------------------------------

// Constant tensor data saved during init(), keyed by IR tensor id.
struct SavedConstant {
  int ir_tensor_id;
  std::vector<uint8_t> data;
};

// Saved constant tensor data for re-copy after sched_reset.
struct SavedGraphConstant {
  struct ggml_tensor* tensor;
  std::vector<uint8_t> data;
};

// ---------------------------------------------------------------------------
// Per-op profiling (enabled by GGML_PROFILE=1 environment variable)
// ---------------------------------------------------------------------------
// Uses the scheduler's eval callback to time each node. Forces per-node
// synchronization, so measured times include sync overhead and don't reflect
// pipelined throughput. Useful for identifying expensive ops.

struct OpProfile {
  int64_t total_us = 0;
  int count = 0;
};

struct ProfileContext {
  std::unordered_map<int, OpProfile> by_op;  // ggml_op -> timing
  int64_t t_start = 0;
};

static bool profile_eval_callback(struct ggml_tensor* t, bool ask, void* user_data) {
  auto* ctx = static_cast<ProfileContext*>(user_data);
  if (ask) {
    // Before compute: record start time
    ctx->t_start = ggml_time_us();
    return true;  // yes, we want to observe this node
  }
  // After compute + sync: record elapsed
  int64_t elapsed = ggml_time_us() - ctx->t_start;
  auto& p = ctx->by_op[t->op];
  p.total_us += elapsed;
  p.count++;
  return true;
}

static void print_profile(const ProfileContext& ctx) {
  // Sort by total time descending
  std::vector<std::pair<int, OpProfile>> sorted(ctx.by_op.begin(), ctx.by_op.end());
  std::sort(sorted.begin(), sorted.end(),
            [](const auto& a, const auto& b) { return a.second.total_us > b.second.total_us; });

  int64_t total = 0;
  for (auto& [op, p] : sorted) total += p.total_us;

  fprintf(stderr, "\n[ggml_profile] Per-op timing (total %.1f ms):\n", total / 1000.0);
  fprintf(stderr, "  %-25s %8s %6s %8s %5s\n", "OP", "TOTAL", "COUNT", "AVG", "%");
  fprintf(stderr, "  %-25s %8s %6s %8s %5s\n", "-------------------------", "--------", "------", "--------", "-----");
  for (auto& [op, p] : sorted) {
    if (p.total_us < 10) continue;  // skip noise
    double pct = total > 0 ? 100.0 * p.total_us / total : 0;
    fprintf(stderr, "  %-25s %7.1fms %5d  %6.1fus %4.1f%%\n",
            ggml_op_name((enum ggml_op)op),
            p.total_us / 1000.0, p.count,
            (double)p.total_us / p.count, pct);
  }
  fprintf(stderr, "\n");
}

// Ensure tensor is contiguous. No-op if already contiguous.
static inline struct ggml_tensor* ensure_cont(struct ggml_context* ctx, struct ggml_tensor* t) {
  return ggml_is_contiguous(t) ? t : ggml_cont(ctx, t);
}

static uint64_t fnv1a_64(const uint8_t* data, size_t len) {
  uint64_t h = 0xcbf29ce484222325ULL;
  for (size_t i = 0; i < len; i++) {
    h ^= data[i];
    h *= 0x100000001b3ULL;
  }
  return h;
}

struct GgmlDelegateHandle {
  struct ggml_context* ctx = nullptr;
  struct ggml_cgraph* graph = nullptr;

  // Ordered by input_index
  std::vector<struct ggml_tensor*> inputs;
  // Output tensors in the order they appear in the IR
  std::vector<struct ggml_tensor*> outputs;

  // Deferred I64→I32 casts that must run during execute() (after input copy).
  // Each pair is (src_i64_tensor, dst_i32_tensor).
  std::vector<std::pair<struct ggml_tensor*, struct ggml_tensor*>> deferred_i64_to_i32;

  // Constant data saved during build_graph, restored after sched_reset.
  std::vector<SavedGraphConstant> saved_constants;

  // Constant integrity detection across execute() calls.
  // After the first compute, some constants may be corrupted by gallocr
  // buffer sharing (an intermediate tensor's buffer overlaps a constant's).
  // On the second call we probe ALL constants to find which are corrupted,
  // then only re-copy those on subsequent calls.
  static constexpr size_t CONST_SIG_SIZE = 64;
  int execute_count = 0;                    // 0=never, 1=once, 2+=probed
  std::vector<bool> const_needs_restore;    // per-constant: true if corrupted

  // Primary backend (GPU when available, otherwise CPU)
  ggml_backend_t backend = nullptr;
  // Dedicated CPU backend for scheduler fallback / custom ops
  ggml_backend_t backend_cpu = nullptr;
  // Multi-backend scheduler (GPU+CPU or CPU-only)
  ggml_backend_sched_t sched = nullptr;
  int n_threads = 1;

  // --- Fields for graph rebuild ---

  // Copy of the serialized IR FlatBuffer so build_graph() can re-parse it
  // without needing the original `processed` buffer.
  std::vector<uint8_t> ir_copy;

  // Constant data extracted from NamedDataMap during init().
  // build_graph() restores these into the ggml context on each rebuild.
  std::vector<SavedConstant> constant_data;

  // True if any tensor has sym_dim_ids (enables shape-change detection).
  bool has_dynamic = false;

  // Per-input sym_dim_ids (ggml ne order, padded to 4).
  // input_sym_dim_ids[i] has 4 int32_t values for input i (-1 = static).
  std::vector<std::vector<int32_t>> input_sym_dim_ids;

  // Last-seen input shapes (ggml ne order, 4 values per input, flattened).
  // Used to detect when a rebuild is needed.
  std::vector<int64_t> last_input_ne;
};

// ---------------------------------------------------------------------------
// build_graph() — (re)build ggml context + compute graph from IR
// ---------------------------------------------------------------------------

// Per-input data blob for pre-populating input tensors during rebuild.
// When provided, eager ops (comparison, bitwise, etc.) that read from
// upstream tensors will see correct input data instead of uninitialized memory.
struct InputDataOverride {
  const void* data;
  size_t nbytes;
  int et_scalar_type;  // executorch::aten::ScalarType enum value
};

// Rebuild the ggml compute graph from the serialized IR in handle->ir_copy.
//
// On success, updates:
//   handle->ctx, graph, inputs, outputs, deferred_i64_to_i32
//
// Frees any previously existing ctx (safe to call on first build too,
// since it starts as nullptr).
//
// ggml_new_f32() asserts no_alloc==false.  This wrapper temporarily disables
// no_alloc so the scalar's 4 bytes are allocated inline from the context pool.
static struct ggml_tensor* make_f32_scalar(struct ggml_context* ctx, float value) {
  ggml_set_no_alloc(ctx, false);
  struct ggml_tensor* t = ggml_new_f32(ctx, value);
  ggml_set_no_alloc(ctx, true);
  return t;
}

// Eagerly compute a binary f32 scalar op during build_graph.
// Returns a new GGML_OP_NONE tensor with the result, or nullptr if not applicable.
static struct ggml_tensor* try_eager_scalar_binop(
    struct ggml_context* ctx,
    struct ggml_tensor* a, struct ggml_tensor* b,
    char op_char) {
  // Only for scalar f32 tensors where both have data available.
  if (ggml_nelements(a) != 1 || ggml_nelements(b) != 1) return nullptr;
  if (a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32) return nullptr;
  if (!a->data || !b->data) return nullptr;
  float va = *(const float*)a->data;
  float vb = *(const float*)b->data;
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

// input_ne_overrides: nullptr on first call (use serialized shapes).
//   In M2+ this will carry runtime shapes for dynamic dims.
// n_overrides: number of int64_t values (n_inputs × 4).
// input_data: optional per-input data for pre-populating input tensors.
//   When non-null, eager ops get correct values instead of uninitialized memory.
static Error build_graph(
    GgmlDelegateHandle* handle,
    const int64_t* input_ne_overrides,
    size_t n_overrides,
    const std::vector<InputDataOverride>* input_data = nullptr) {

  (void)n_overrides;

  // --- Tear down previous graph (no-op on first call) ---
  if (handle->ctx) {
    ggml_free(handle->ctx);
    handle->ctx = nullptr;
  }
  handle->graph = nullptr;
  handle->inputs.clear();
  handle->outputs.clear();
  handle->deferred_i64_to_i32.clear();

  // --- Parse IR from the saved copy ---
  const auto* fb_graph = ggml_ir::GetGgmlGraph(handle->ir_copy.data());
  if (!fb_graph || !fb_graph->tensors()) {
    return Error::InvalidArgument;
  }
  const auto* fb_tensors = fb_graph->tensors();
  const int n_tensors = static_cast<int>(fb_tensors->size());

  // === Phase A: calculate ctx size, create ggml context ===
  // Context only needs tensor metadata + graph structure.
  // Actual tensor data is allocated by gallocr (backend scheduler) or by
  // a temporary CPU host buffer for leaf tensors (constants / inputs).

  // Estimate graph size: 4× the IR tensors to account for compound ops.
  size_t est_graph_size = static_cast<size_t>(n_tensors) * 4;
  if (est_graph_size < GGML_DEFAULT_GRAPH_SIZE) {
    est_graph_size = GGML_DEFAULT_GRAPH_SIZE;
  }

  // Eager ops (ARANGE, FULL, CUMSUM, EQ, NE, CAST, etc.) create
  // data-filled leaf tensors during op dispatch.  Their data is allocated
  // inline from the context pool (no_alloc temporarily false).
  // Estimate the headroom needed for these eager allocations.
  size_t eager_data_estimate = 0;
  for (int i = 0; i < n_tensors; ++i) {
    const auto* t = fb_tensors->Get(i);
    auto op = static_cast<ggml_ir::OpCode>(t->op());
    if (op == ggml_ir::OpCode::NONE) continue;
    // Only count ops known to create data-filled tensors during build.
    bool is_eager = false;
    // Only count ops that ALWAYS create data-filled tensors during build.
    // ADD/SUB are only eager for I64 (rare, small). CAST is only eager for
    // I64 sources or scalars (also small). Skip those to avoid inflating
    // the estimate with large max-shape tensors.
    switch (op) {
      case ggml_ir::OpCode::ARANGE:
      case ggml_ir::OpCode::FULL:
      case ggml_ir::OpCode::CUMSUM:
      case ggml_ir::OpCode::EQ:
      case ggml_ir::OpCode::NE:
      case ggml_ir::OpCode::ANY:
      case ggml_ir::OpCode::INDEX_PUT:
        is_eager = true;
        break;
      default:
        break;
    }
    if (!is_eager) continue;
    size_t n_elems = 1;
    if (t->ne()) {
      for (size_t d = 0; d < t->ne()->size(); ++d) {
        n_elems *= static_cast<size_t>(t->ne()->Get(d));
      }
    }
    // Worst-case element size (I64 = 8 bytes) + per-tensor overhead.
    eager_data_estimate += n_elems * 8 + ggml_tensor_overhead() + 64;
  }
  // Safety multiplier: compound dispatch may create sub-tensors.
  eager_data_estimate = std::max(eager_data_estimate * 2, (size_t)(4 * 1024 * 1024));

  size_t ctx_size =
      static_cast<size_t>(n_tensors) * 4 * ggml_tensor_overhead() +
      ggml_graph_overhead_custom(est_graph_size, false) +
      eager_data_estimate;
  fprintf(stderr, "[ggml_backend] ctx_size=%zu MB, tensor_overhead=%zu, n_tensors=%d\n",
          ctx_size / (1024*1024), ggml_tensor_overhead(), n_tensors);

  struct ggml_init_params params = {
      /* .mem_size   = */ ctx_size,
      /* .mem_buffer = */ nullptr,
      /* .no_alloc   = */ true,
  };

  fprintf(stderr, "[ggml_backend] calling ggml_init with %zu bytes\n", ctx_size); fflush(stderr);
  struct ggml_context* ctx = ggml_init(params);
  fprintf(stderr, "[ggml_backend] ggml_init returned ctx=%p\n", (void*)ctx); fflush(stderr);
  if (!ctx) {
    return Error::MemoryAllocationFailed;
  }

  // === Phase B: tensor creation loop + op switch ===
  // Map from IR tensor id → ggml_tensor*
  std::vector<struct ggml_tensor*> id_to_tensor(n_tensors, nullptr);

  // Temporary CPU host buffer for leaf tensor data.
  // Freed after gallocr copies constants to backend buffers (Phase C step 6).
  ggml_backend_buffer_t host_buf = nullptr;

  // RAII guard: ensure host_buf is freed on early return (error paths).
  struct HostBufGuard {
    ggml_backend_buffer_t& buf;
    ~HostBufGuard() { if (buf) { ggml_backend_buffer_free(buf); buf = nullptr; } }
  } host_buf_guard{host_buf};

  // Track inputs and outputs
  std::vector<std::pair<int, struct ggml_tensor*>> input_pairs;  // (index, tensor)
  std::vector<std::pair<int, struct ggml_tensor*>> output_pairs; // (index, tensor)
  // Deferred I64→I32 casts (for input tensors).
  std::vector<std::pair<struct ggml_tensor*, struct ggml_tensor*>> deferred_i64_to_i32;

  // Build sym_dim_values: symbolic variable ID → runtime concrete value.
  // Populated from input tensors that have sym_dim_ids annotations.
  std::unordered_map<int32_t, int64_t> sym_dim_values;
  if (input_ne_overrides) {
    for (int i = 0; i < n_tensors; ++i) {
      const auto* ti = fb_tensors->Get(i);
      if (!ti->is_input() || !ti->sym_dim_ids() || !ti->ne()) continue;
      int input_idx = ti->input_index();
      for (size_t d = 0; d < ti->sym_dim_ids()->size() && d < 4; ++d) {
        int32_t sid = ti->sym_dim_ids()->Get(d);
        if (sid >= 0) {
          sym_dim_values[sid] = input_ne_overrides[input_idx * 4 + d];
        }
      }
    }
  }

  // --- Shape resolution helper (shared by leaf and op passes) ---
  auto resolve_shape = [&](const ggml_ir::Tensor* t, int64_t ne[4], int& n_dims) {
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
  };

  // --- IR type → ggml type helper ---
  auto resolve_gtype = [](const ggml_ir::Tensor* t) -> ggml_type {
    switch (t->type()) {
      case ggml_ir::TensorType::F16:  return GGML_TYPE_F16;
      case ggml_ir::TensorType::I64:  return GGML_TYPE_I64;
      case ggml_ir::TensorType::I32:  return GGML_TYPE_I32;
      case ggml_ir::TensorType::BOOL: return GGML_TYPE_I32;  // stored as I32
      case ggml_ir::TensorType::BF16: return GGML_TYPE_BF16;
      case ggml_ir::TensorType::Q8_0: return GGML_TYPE_Q8_0;
      case ggml_ir::TensorType::Q6_K: return GGML_TYPE_Q6_K;
      case ggml_ir::TensorType::Q4_0: return GGML_TYPE_Q4_0;
      default:                        return GGML_TYPE_F32;
    }
  };

  // === Pre-calculate total leaf data size for tallocr ===
  {
    size_t total_leaf_data = 0;
    const size_t alignment = ggml_backend_buft_get_alignment(ggml_backend_cpu_buffer_type());
    for (int i = 0; i < n_tensors; ++i) {
      const auto* t = fb_tensors->Get(i);
      if (static_cast<ggml_ir::OpCode>(t->op()) != ggml_ir::OpCode::NONE) continue;
      int64_t leaf_ne[4] = {1,1,1,1}; int leaf_ndims;
      resolve_shape(t, leaf_ne, leaf_ndims);
      ggml_type gtype = resolve_gtype(t);
      size_t tensor_bytes = ggml_row_size(gtype, leaf_ne[0]) * leaf_ne[1] * leaf_ne[2] * leaf_ne[3];
      // Match ggml_tallocr_alloc: GGML_PAD(alloc_size, alignment)
      size_t padded = ((tensor_bytes + alignment - 1) / alignment) * alignment;
      total_leaf_data += padded;
    }
    // Safety margin: 1% + 1 MB for alignment rounding and initial offset.
    total_leaf_data += total_leaf_data / 100 + 1024 * 1024;
    fprintf(stderr, "[ggml_backend] total_leaf_data=%zu MB\n", total_leaf_data / (1024*1024));
    host_buf = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), total_leaf_data);
    if (!host_buf) {
      fprintf(stderr, "[ggml_backend] ERROR: failed to allocate host buffer of %zu bytes\n", total_leaf_data);
      ggml_free(ctx);
      return Error::MemoryAllocationFailed;
    }
  }
  struct ggml_tallocr tallocr = ggml_tallocr_new(host_buf);

  // === Phase B: single-pass tensor creation loop ===
  fprintf(stderr, "[ggml_backend] Phase B: creating tensors (single pass)...\n"); fflush(stderr);
  static int s_last_processed = -1;
  static int s_n_tensors = 0;
  s_last_processed = -1;
  s_n_tensors = n_tensors;
  int last_processed = -1;
  for (int i = 0; i < n_tensors; ++i) {
    const auto* t = fb_tensors->Get(i);
    const int tid = t->id();
    const auto op = static_cast<ggml_ir::OpCode>(t->op());
    int64_t ne[4] = {1,1,1,1}; int n_dims = 4;
    resolve_shape(t, ne, n_dims);

    if (op == ggml_ir::OpCode::NONE) {
      // --- Leaf tensor: create, allocate from host buffer, load data ---
      ggml_type gtype = resolve_gtype(t);
      struct ggml_tensor* gt = nullptr;
      switch (n_dims) {
        case 1:  gt = ggml_new_tensor_1d(ctx, gtype, ne[0]); break;
        case 2:  gt = ggml_new_tensor_2d(ctx, gtype, ne[0], ne[1]); break;
        case 3:  gt = ggml_new_tensor_3d(ctx, gtype, ne[0], ne[1], ne[2]); break;
        default: gt = ggml_new_tensor_4d(ctx, gtype, ne[0], ne[1], ne[2], ne[3]); break;
      }
      s_last_processed = -(i + 100); // encode: NONE tensor entering tallocr
      if (i == 1371) {
        fprintf(stderr, "[ggml_backend] NONE tensor %d: gtype=%d ne=(%lld,%lld,%lld,%lld) nbytes=%zu\n",
            i, gt->type, (long long)gt->ne[0], (long long)gt->ne[1], (long long)gt->ne[2], (long long)gt->ne[3],
            ggml_nbytes(gt)); fflush(stderr);
      }
      if (i >= 1370 && i <= 1375) {
        size_t alloc_sz = ggml_backend_buffer_get_alloc_size(host_buf, gt);
        size_t padded = ((alloc_sz + tallocr.alignment - 1) / tallocr.alignment) * tallocr.alignment;
        fprintf(stderr, "[ggml_backend] tensor %d: nbytes=%zu alloc_sz=%zu padded=%zu offset=%zu / buf=%zu data=%p buf_ptr=%p\n",
            i, ggml_nbytes(gt), alloc_sz, padded, tallocr.offset,
            ggml_backend_buffer_get_size(host_buf), gt->data, (void*)gt->buffer); fflush(stderr);
      }
      ggml_tallocr_alloc(&tallocr, gt);

      // Load constant data from handle->constant_data (populated by init()).
      if (t->data_key() && t->data_key()->c_str() && std::strlen(t->data_key()->c_str()) > 0) {
        const size_t nbytes = ggml_nbytes(gt);
        for (const auto& sc : handle->constant_data) {
          if (sc.ir_tensor_id == tid) {
            memcpy(gt->data, sc.data.data(), nbytes);
            break;
          }
        }
      }

      // Pre-populate input tensor data so eager ops downstream
      // (comparisons, bitwise, etc.) read correct values during build.
      if (t->is_input() && input_data && t->input_index() >= 0 &&
          static_cast<size_t>(t->input_index()) < input_data->size()) {
        const auto& ido = (*input_data)[t->input_index()];
        if (ido.data) {
          const size_t nelem = ggml_nelements(gt);
          if (gt->type == GGML_TYPE_I32 && ido.et_scalar_type == 4 /* Long */) {
            const int64_t* src = static_cast<const int64_t*>(ido.data);
            int32_t* dst = static_cast<int32_t*>(gt->data);
            for (size_t j = 0; j < nelem; ++j) dst[j] = static_cast<int32_t>(src[j]);
          } else if (gt->type == GGML_TYPE_F32 && ido.et_scalar_type == 6 /* Float */) {
            memcpy(gt->data, ido.data, std::min(ido.nbytes, ggml_nbytes(gt)));
          } else if (gt->type == GGML_TYPE_I32 && ido.et_scalar_type == 11 /* Bool */) {
            const bool* src = static_cast<const bool*>(ido.data);
            int32_t* dst = static_cast<int32_t*>(gt->data);
            for (size_t j = 0; j < nelem; ++j) dst[j] = src[j] ? 1 : 0;
          } else {
            memcpy(gt->data, ido.data, std::min(ido.nbytes, ggml_nbytes(gt)));
          }
        }
      }

      if (t->is_input()) {
        input_pairs.emplace_back(t->input_index(), gt);
      }
      if (t->is_output()) {
        int out_idx = t->input_index();
        output_pairs.emplace_back(out_idx >= 0 ? out_idx : (int)output_pairs.size(), gt);
      }
      id_to_tensor[tid] = gt;
      continue;
    }

    // --- Op tensor: build the ggml operation ---
    s_last_processed = -(i + 10000); // encode: op tensor entering dispatch
    struct ggml_tensor* gt = nullptr;

    // Resolve sources
    std::vector<struct ggml_tensor*> srcs;
    if (t->src_ids()) {
      for (size_t s = 0; s < t->src_ids()->size(); ++s) {
        int src_id = t->src_ids()->Get(s);
        srcs.push_back(id_to_tensor[src_id]);
      }
    }
    // ggml custom ops are CPU callbacks; force them to CPU when mixed
    // backends are active so GPU backends never try to encode GGML_OP_CUSTOM.
    auto pin_to_cpu = [&](struct ggml_tensor* x) {
      if (x && handle->sched && handle->backend_cpu) {
        ggml_backend_sched_set_tensor_backend(handle->sched, x, handle->backend_cpu);
      }
    };

    // ── Shared broadcast helpers for binary ops (ADD, MUL, DIV, SUB) ──
    auto try_repeat_1d_to_match = [&](struct ggml_tensor * small,
                                     struct ggml_tensor * big) -> struct ggml_tensor * {
      // If `small` is effectively 1D (exactly one dim > 1) and its length matches
      // one of big's dims, reshape+permute it so the matching dim aligns, then repeat.
      int non1 = 0;
      int non1_ax = -1;
      for (int ax = 0; ax < 4; ++ax) {
        if (small->ne[ax] != 1) {
          non1++;
          non1_ax = ax;
        }
      }
      if (non1 != 1) {
        return nullptr;
      }
      const int64_t n = small->ne[non1_ax];

      // Normalize to a base view where n is in axis0.
      struct ggml_tensor * base = small;
      if (non1_ax != 0) {
        // permute so that old non1_ax becomes new axis0
        int axes[4] = {0,1,2,3};
        axes[0] = non1_ax;
        int out = 1;
        for (int ax = 0; ax < 4; ++ax) {
          if (ax == non1_ax) continue;
          axes[out++] = ax;
        }
        base = ggml_permute(ctx, base, axes[0], axes[1], axes[2], axes[3]);
        base = ggml_cont(ctx, base);
      }
      for (int ax = 0; ax < 4; ++ax) {
        if (big->ne[ax] != n) {
          continue;
        }
        // Start as 4D [n,1,1,1]
        struct ggml_tensor * s4 = ggml_reshape_4d(ctx, base, n, 1, 1, 1);
        // Permute so `n` ends up in axis `ax`
        int p0 = 0, p1 = 1, p2 = 2, p3 = 3;
        if (ax == 0) {
          // already in axis0
        } else if (ax == 1) {
          // move axis0 -> axis1
          p0 = 1; p1 = 0; p2 = 2; p3 = 3;
        } else if (ax == 2) {
          p0 = 1; p1 = 2; p2 = 0; p3 = 3;
        } else {
          // ax == 3: move axis0 -> axis3
          p0 = 1; p1 = 2; p2 = 3; p3 = 0;
        }
        s4 = ggml_permute(ctx, s4, p0, p1, p2, p3);
        // Make the view contiguous to satisfy ggml_can_repeat in some cases.
        s4 = ggml_cont(ctx, s4);
        if (ggml_can_repeat(s4, big)) {
          return ggml_repeat(ctx, s4, big);
        }
      }
      return nullptr;
    };

    auto try_permute_to_match = [&](struct ggml_tensor * src,
                                   struct ggml_tensor * dst) -> struct ggml_tensor * {
      // If src and dst have the same multiset of extents, try a few permutes.
      int64_t aa[4] = {dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]};
      int64_t bb[4] = {src->ne[0], src->ne[1], src->ne[2], src->ne[3]};
      for (int i = 0; i < 4; ++i) {
        for (int j = i + 1; j < 4; ++j) {
          if (aa[j] < aa[i]) { auto t = aa[i]; aa[i] = aa[j]; aa[j] = t; }
          if (bb[j] < bb[i]) { auto t = bb[i]; bb[i] = bb[j]; bb[j] = t; }
        }
      }
      for (int i = 0; i < 4; ++i) {
        if (aa[i] != bb[i]) return nullptr;
      }

      const int perms[][4] = {
        {0,1,2,3},
        {3,1,2,0},
        {0,2,1,3},
        {1,0,2,3},
        {2,1,0,3},
        {1,2,3,0},
        {2,3,1,0},
        {3,2,1,0},
      };
      for (const auto & p : perms) {
        struct ggml_tensor * t = ggml_permute(ctx, src, p[0], p[1], p[2], p[3]);
        if (ggml_are_same_shape(t, dst)) {
          return ggml_cont(ctx, t);
        }
      }
      return nullptr;
    };

    // General PyTorch-style broadcast: reshape `small` by placing its non-1
    // dims into axes of `big` where they divide evenly, then ggml_repeat.
    // Handles cases like mask (13,13,1,1) → attention (16,13,256,1).
    auto try_reshape_broadcast = [&](struct ggml_tensor * small,
                                    struct ggml_tensor * big) -> struct ggml_tensor * {
      // Collect non-1 dims of small.
      struct { int ax; int64_t n; } sdims[4];
      int ns = 0;
      for (int d = 0; d < 4; ++d) {
        if (small->ne[d] != 1) {
          sdims[ns++] = {d, small->ne[d]};
        }
      }
      if (ns == 0 || ns > 4) return nullptr;

      // For each non-1 dim of small, find candidate axes in big where it divides.
      // We brute-force all assignments (at most 4^4 = 256 combos).
      // Each small dim must map to a distinct big axis.
      int candidates[4][4];
      int ncand[4];
      for (int si = 0; si < ns; ++si) {
        ncand[si] = 0;
        for (int bi = 0; bi < 4; ++bi) {
          if (big->ne[bi] % sdims[si].n == 0) {
            candidates[si][ncand[si]++] = bi;
          }
        }
        if (ncand[si] == 0) return nullptr;
      }

      // Try all assignments via recursive enumeration (iterative for up to 4).
      int assign[4] = {0, 0, 0, 0};
      bool used[4];

      // Helper: check current assignment combo
      auto try_assignment = [&]() -> struct ggml_tensor * {
        // Check uniqueness of big-axis assignments.
        memset(used, 0, sizeof(used));
        for (int si = 0; si < ns; ++si) {
          int bi = candidates[si][assign[si]];
          if (used[bi]) return nullptr;
          used[bi] = true;
        }
        // Build reshaped dims: start with all 1s, place small dims.
        int64_t new_ne[4] = {1, 1, 1, 1};
        for (int si = 0; si < ns; ++si) {
          new_ne[candidates[si][assign[si]]] = sdims[si].n;
        }
        // We need to reshape `small` from its current layout to new_ne.
        // First flatten small to 1D, then reshape to new_ne.
        int64_t nel = ggml_nelements(small);
        struct ggml_tensor * flat = ggml_cont(ctx, ggml_reshape_1d(ctx, small, nel));
        struct ggml_tensor * reshaped = ggml_reshape_4d(ctx, flat, new_ne[0], new_ne[1], new_ne[2], new_ne[3]);
        if (ggml_can_repeat(reshaped, big)) {
          return ggml_repeat(ctx, reshaped, big);
        }
        return nullptr;
      };

      // Enumerate all assignment combinations.
      if (ns == 1) {
        for (assign[0] = 0; assign[0] < ncand[0]; ++assign[0]) {
          if (auto * r = try_assignment()) return r;
        }
      } else if (ns == 2) {
        for (assign[0] = 0; assign[0] < ncand[0]; ++assign[0])
          for (assign[1] = 0; assign[1] < ncand[1]; ++assign[1]) {
            if (auto * r = try_assignment()) return r;
          }
      } else if (ns == 3) {
        for (assign[0] = 0; assign[0] < ncand[0]; ++assign[0])
          for (assign[1] = 0; assign[1] < ncand[1]; ++assign[1])
            for (assign[2] = 0; assign[2] < ncand[2]; ++assign[2]) {
              if (auto * r = try_assignment()) return r;
            }
      } else {
        for (assign[0] = 0; assign[0] < ncand[0]; ++assign[0])
          for (assign[1] = 0; assign[1] < ncand[1]; ++assign[1])
            for (assign[2] = 0; assign[2] < ncand[2]; ++assign[2])
              for (assign[3] = 0; assign[3] < ncand[3]; ++assign[3]) {
                if (auto * r = try_assignment()) return r;
              }
      }
      return nullptr;
    };

    // Slice a larger (max-shape) tensor down to match a smaller (runtime-shape)
    // tensor. Used when a pre-computed constant was exported at max_seq_len but
    // the runtime sequence is shorter.
    auto try_slice_to_match = [&](struct ggml_tensor * big,
                                 struct ggml_tensor * small) -> struct ggml_tensor * {
      bool any_diff = false;
      for (int d = 0; d < 4; ++d) {
        if (big->ne[d] < small->ne[d]) return nullptr;
        if (big->ne[d] != small->ne[d]) any_diff = true;
      }
      if (!any_diff) return nullptr;
      auto * v = ggml_view_4d(ctx, big,
          small->ne[0], small->ne[1], small->ne[2], small->ne[3],
          big->nb[1], big->nb[2], big->nb[3], 0);
      return ensure_cont(ctx, v);
    };

    // Shared broadcast resolution for binary ops. Returns true on success,
    // updating a and b in place. Returns false on failure.
    auto resolve_broadcast = [&](struct ggml_tensor *& a, struct ggml_tensor *& b,
                                const char * op_name) -> bool {
      if (ggml_are_same_shape(a, b)) return true;
      // ggml binary ops natively broadcast b over a — no explicit repeat needed.
      if (ggml_can_repeat(b, a)) return true;
      if (ggml_can_repeat(a, b)) {
        // a is smaller — must expand since ggml only broadcasts b over a.
        // ggml_repeat output is already contiguous, no ggml_cont needed.
        a = ggml_repeat(ctx, a, b);
      } else if (auto * bb = try_repeat_1d_to_match(b, a)) {
        b = bb;
      } else if (auto * aa = try_repeat_1d_to_match(a, b)) {
        a = aa;
      } else if (auto * bp = try_permute_to_match(b, a)) {
        b = bp;
      } else if (auto * ap = try_permute_to_match(a, b)) {
        a = ap;
      } else if (auto * bb2 = try_reshape_broadcast(b, a)) {
        b = bb2;
      } else if (auto * aa2 = try_reshape_broadcast(a, b)) {
        a = aa2;
      } else if (auto * as = try_slice_to_match(a, b)) {
        a = as;
      } else if (auto * bs = try_slice_to_match(b, a)) {
        b = bs;
      } else {
        fprintf(stderr,
                "[executorch-ggml] %s shape mismatch not broadcastable: "
                "a=(%lld,%lld,%lld,%lld) b=(%lld,%lld,%lld,%lld)\n",
                op_name,
                (long long) a->ne[0], (long long) a->ne[1], (long long) a->ne[2], (long long) a->ne[3],
                (long long) b->ne[0], (long long) b->ne[1], (long long) b->ne[2], (long long) b->ne[3]);
        return false;
      }
      return true;
    };

    switch (op) {
        case ggml_ir::OpCode::ADD: {
          struct ggml_tensor* a = srcs[0];
          struct ggml_tensor* b = srcs[1];

          // Eager scalar path for enc_len-style computations.
          if (auto* eager = try_eager_scalar_binop(ctx, a, b, '+')) {
            gt = eager;
            break;
          }

          // For int64 types with data available, use eager element-wise add
          // with manual broadcasting. Must be done BEFORE the broadcast
          // handling below, which creates ggml ops with null data.
          if (a->type == GGML_TYPE_I64 && b->type == GGML_TYPE_I64
              && a->data && b->data) {
            // Compute output shape (element-wise max of dims).
            int64_t out_ne[4];
            for (int d = 0; d < 4; ++d) {
              out_ne[d] = std::max(a->ne[d], b->ne[d]);
            }
            ggml_set_no_alloc(ctx, false);
            gt = ggml_new_tensor_4d(ctx, GGML_TYPE_I64, out_ne[0], out_ne[1], out_ne[2], out_ne[3]);
            ggml_set_no_alloc(ctx, true);
            int64_t* out_data = static_cast<int64_t*>(gt->data);
            const int64_t* a_data = static_cast<const int64_t*>(a->data);
            const int64_t* b_data = static_cast<const int64_t*>(b->data);
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
            break;
          }

          if (a->type != b->type) {
            // Prefer casting to f32 if either side is f32.
            ggml_type tgt = (a->type == GGML_TYPE_F32 || b->type == GGML_TYPE_F32)
                                ? GGML_TYPE_F32
                                : a->type;
            if (a->type != tgt) a = safe_ggml_cast(ctx, a, tgt);
            if (b->type != tgt) b = safe_ggml_cast(ctx, b, tgt);
          }

          if (!resolve_broadcast(a, b, "ADD")) {
            ggml_free(ctx);
            return Error::InvalidArgument;
          }

          gt = ggml_add(ctx, a, b);
          break;
        }

        case ggml_ir::OpCode::MUL_MAT:
          gt = ggml_mul_mat(ctx, srcs[0], srcs[1]);
          break;

        case ggml_ir::OpCode::MUL: {
          // ggml_mul(a, b) requires ggml_can_repeat(b, a) — b broadcasts to a.
          // Swap if needed so the larger tensor is first.
          struct ggml_tensor* a = srcs[0];
          struct ggml_tensor* b = srcs[1];
          if (auto* eager = try_eager_scalar_binop(ctx, a, b, '*')) {
            gt = eager;
            break;
          }
          if (ggml_nelements(a) < ggml_nelements(b)) {
            std::swap(a, b);
          }
          if (!resolve_broadcast(a, b, "MUL")) {
            ggml_free(ctx);
            return Error::InvalidArgument;
          }
          gt = ggml_mul(ctx, a, b);
          break;
        }

        case ggml_ir::OpCode::DIV: {
          // ggml_div(a, b) requires ggml_can_repeat(b, a) — b broadcasts to a.
          struct ggml_tensor* a = srcs[0];
          struct ggml_tensor* b = srcs[1];
          if (auto* eager = try_eager_scalar_binop(ctx, a, b, '/')) {
            gt = eager;
            break;
          }
          if (!resolve_broadcast(a, b, "DIV")) {
            ggml_free(ctx);
            return Error::InvalidArgument;
          }
          gt = ggml_div(ctx, a, b);
          break;
        }

        case ggml_ir::OpCode::REPEAT: {
          // REPEAT(src, like): tile src to like's shape.
          // The "like" tensor (srcs[1]) may be a static shape-only tensor
          // from export time.  When ggml auto-computes upstream shapes
          // differently (e.g. after permute), the src shape may not match
          // the like shape.  Rebuild the like tensor from the src shape +
          // the expansion ratios from the IR ne.
          struct ggml_tensor* a = srcs[0];  // source data
          struct ggml_tensor* b = srcs[1];  // target shape ("like")

          // If shapes already match, this is a no-op.
          if (ggml_are_same_shape(a, b)) {
            gt = a;
            id_to_tensor[tid] = gt;
            break;
          }

          // Try to derive the correct target shape by computing the
          // expansion ratio from the IR ne and applying it to the source.
          if (!ggml_can_repeat(a, b)) {
            // Compute ratio from IR: for each dim where src is 1 and
            // target is >1, the ratio is target/1.  Use source's actual
            // shape as the base and apply the ratio.
            int64_t target_ne[4];
            for (int d = 0; d < 4; ++d) {
              if (a->ne[d] == 1 && b->ne[d] > 1) {
                // This dim was expanded from 1 — keep the target.
                target_ne[d] = b->ne[d];
              } else {
                // Use source's actual shape for this dim.
                target_ne[d] = a->ne[d];
              }
            }
            b = ggml_new_tensor_4d(ctx, a->type, target_ne[0], target_ne[1], target_ne[2], target_ne[3]);
          }

          if (!ggml_can_repeat(a, b)) {
            if (ggml_can_repeat(b, a)) {
              gt = ggml_repeat(ctx, b, a);
            } else {
              fprintf(stderr,
                      "[executorch-ggml] REPEAT shape not repeatable (tensor %d, srcs=[%d,%d]): a=(%lld,%lld,%lld,%lld) b=(%lld,%lld,%lld,%lld)\n",
                      (int)i,
                      (int)(t->src_ids() && t->src_ids()->size() > 0 ? t->src_ids()->Get(0) : -1),
                      (int)(t->src_ids() && t->src_ids()->size() > 1 ? t->src_ids()->Get(1) : -1),
                      (long long) a->ne[0], (long long) a->ne[1], (long long) a->ne[2], (long long) a->ne[3],
                      (long long) b->ne[0], (long long) b->ne[1], (long long) b->ne[2], (long long) b->ne[3]);
              ggml_free(ctx);
              return Error::InvalidArgument;
            }
          } else {
            gt = ggml_repeat(ctx, a, b);
          }
          break;
        }

        case ggml_ir::OpCode::NEG:
          gt = ggml_neg(ctx, srcs[0]);
          break;

        case ggml_ir::OpCode::RSQRT: {
          // ggml doesn't expose rsqrt directly in this version; implement as 1/sqrt(x)
          // ggml_div(a, b) requires b to be repeatable to a's shape.
          // Since we want 1/sqrt(x), we need a to have x's shape and be filled with 1.0.
          struct ggml_tensor* sx  = ggml_sqrt(ctx, srcs[0]);
          struct ggml_tensor* one = make_f32_scalar(ctx, 1.0f);
          // Repeat the scalar 1.0 to match sqrt's shape
          struct ggml_tensor* one_rep = ggml_repeat(ctx, one, sx);
          gt = ggml_div(ctx, one_rep, sx);
          break;
        }

        case ggml_ir::OpCode::SILU:
          gt = ggml_silu(ctx, srcs[0]);
          break;

        case ggml_ir::OpCode::RELU:
          gt = ggml_relu(ctx, srcs[0]);
          break;

        case ggml_ir::OpCode::TANH:
          gt = ggml_tanh(ctx, srcs[0]);
          break;

        case ggml_ir::OpCode::LINEAR: {
          // src_ids: [x, w, b?] where w is [out, in] in PyTorch.
          // Our IR stores shapes in ggml order, so weight ne is [in, out].
          // Compute: y = w @ x, then add bias if present.
          struct ggml_tensor* x = srcs[0];
          struct ggml_tensor* w = srcs[1];
          struct ggml_tensor* y = ggml_mul_mat(ctx, w, x);
          if (srcs.size() > 2) {
            struct ggml_tensor* b = srcs[2];
            // Broadcast bias to y shape.
            // For PyTorch linear, bias is typically 1D [out]. In ggml order that is ne0=out.
            // Make it at least 2D so ggml_repeat can broadcast along remaining dims.
            if (ggml_n_dims(b) == 1) {
              // Prefer a 4D shape to be safely repeatable against 2D/3D/4D outputs.
              b = ggml_reshape_4d(ctx, b, b->ne[0], 1, 1, 1);
            }
            if (!ggml_can_repeat(b, y)) {
              fprintf(stderr,
                      "[executorch-ggml] LINEAR bias not repeatable: b=(%lld,%lld,%lld,%lld) y=(%lld,%lld,%lld,%lld)\n",
                      (long long) b->ne[0], (long long) b->ne[1], (long long) b->ne[2], (long long) b->ne[3],
                      (long long) y->ne[0], (long long) y->ne[1], (long long) y->ne[2], (long long) y->ne[3]);
              ggml_free(ctx);
              return Error::InvalidArgument;
            }
            struct ggml_tensor* b_rep = ggml_repeat(ctx, b, y);
            y = ggml_add(ctx, y, b_rep);
          }
          gt = y;
          break;
        }

        case ggml_ir::OpCode::EMBEDDING: {
          // src_ids: [weight, indices]
          struct ggml_tensor* w = srcs[0];
          struct ggml_tensor* idx = srcs[1];
          if (idx->type == GGML_TYPE_I64) {
            idx = eager_cast_i64_to_i32(ctx, idx);
          } else if (idx->type != GGML_TYPE_I32) {
            idx = safe_ggml_cast(ctx, idx, GGML_TYPE_I32);
          }
          gt = ggml_get_rows(ctx, w, idx);
          break;
        }

        case ggml_ir::OpCode::LEAKY_RELU: {
          float negative_slope = 0.01f;
          if (t->op_params() && t->op_params()->size() >= 4) {
            memcpy(&negative_slope, t->op_params()->data(), sizeof(float));
          }
          gt = ggml_leaky_relu(ctx, srcs[0], negative_slope, false);
          break;
        }

        case ggml_ir::OpCode::CONV_2D:
        case ggml_ir::OpCode::CONV_2D_DW: {
          // src_ids: [weight, input, bias?]
          // op_params: stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups
          if (srcs.size() < 2) {
            ggml_free(ctx);
            return Error::InvalidArgument;
          }

          struct ggml_tensor* weight = srcs[0];
          struct ggml_tensor* input = srcs[1];
          struct ggml_tensor* bias = srcs.size() > 2 ? srcs[2] : nullptr;

          // Parse op_params
          int32_t stride_h = 1, stride_w = 1;
          int32_t pad_h = 0, pad_w = 0;
          int32_t dilation_h = 1, dilation_w = 1;
          int32_t groups = 1;

          if (t->op_params() && t->op_params()->size() >= 28) {
            const uint8_t* data = t->op_params()->data();
            memcpy(&stride_h, data, sizeof(int32_t));
            memcpy(&stride_w, data + 4, sizeof(int32_t));
            memcpy(&pad_h, data + 8, sizeof(int32_t));
            memcpy(&pad_w, data + 12, sizeof(int32_t));
            memcpy(&dilation_h, data + 16, sizeof(int32_t));
            memcpy(&dilation_w, data + 20, sizeof(int32_t));
            memcpy(&groups, data + 24, sizeof(int32_t));
          }

          // ggml API signature:
          // ggml_conv_2d(ctx, weight, input, s0, s1, p0, p1, d0, d1)
          // ggml_conv_2d_dw(ctx, weight, input, s0, s1, p0, p1, d0, d1)
          // Note: groups parameter not directly supported in ggml API
          // For depthwise conv (groups > 1), use ggml_conv_2d_dw
          // For regular conv (groups == 1), use ggml_conv_2d
          // Pointwise 2D conv (k=1x1, s=1x1, p=0, d=1x1, groups=1) is a matmul.
          // Bypass ggml_conv_2d (F16 im2col) to stay in F32.
          int64_t kh = weight->ne[1]; // ggml layout: [kW, kH, C_in, C_out]
          int64_t kw = weight->ne[0];
          bool is_pointwise_2d = (kh == 1 && kw == 1
                                  && stride_h == 1 && stride_w == 1
                                  && pad_h == 0 && pad_w == 0
                                  && dilation_h == 1 && dilation_w == 1
                                  && groups == 1);

          if (is_pointwise_2d) {
            // weight ggml shape: [kW=1, kH=1, C_in, C_out] → [C_in, C_out]
            struct ggml_tensor* w2d = ggml_reshape_2d(ctx, weight,
                                                      weight->ne[2], weight->ne[3]);
            // input ggml shape: [W, H, C_in, N] → collapse spatial to [W*H, C_in, N, 1]
            struct ggml_tensor* inp3d = ggml_reshape_3d(ctx, ensure_cont(ctx, input),
                                                        input->ne[0] * input->ne[1],
                                                        input->ne[2],
                                                        input->ne[3]);
            // ggml_mul_mat contracts on ne[0], but inp3d has W*H in ne[0].
            // Transpose so C_in is in ne[0], do matmul, transpose back.
            struct ggml_tensor* inp_t = ggml_cont(ctx, ggml_transpose(ctx, inp3d));
            // inp_t: [C_in, W*H, N, 1]
            struct ggml_tensor* mm = ggml_mul_mat(ctx, w2d, inp_t);
            // mm: [C_out, W*H, N, 1]
            struct ggml_tensor* mm_t = ggml_cont(ctx, ggml_transpose(ctx, mm));
            // mm_t: [W*H, C_out, N, 1]
            // Reshape back to [W, H, C_out, N]
            gt = ggml_reshape_4d(ctx, mm_t,
                                 input->ne[0], input->ne[1],
                                 weight->ne[3], input->ne[3]);
          } else if (op == ggml_ir::OpCode::CONV_2D_DW || groups > 1) {
            // Use ggml_conv_2d_dw_direct which avoids im2col and works in
            // the input's native type (F32), preserving precision.
            gt = ggml_conv_2d_dw_direct(ctx, weight, input, stride_w, stride_h, pad_w, pad_h, dilation_w, dilation_h);
          } else {
            // Regular convolution — uses im2col in the kernel's type.
            gt = ggml_conv_2d(ctx, weight, input, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w);
          }

          // Cast conv output to f32 to keep the rest of the graph in f32 and
          // avoid mixed-type binary ops (e.g. residual adds).
          if (gt && gt->type == GGML_TYPE_F16) {
            gt = safe_ggml_cast(ctx, gt, GGML_TYPE_F32);
          }

          // Add bias if present.
          // Conv bias in PyTorch is 1D [Cout]. ggml_add requires broadcastable
          // shapes, so reshape+repeat the bias to match conv output.
          if (bias && gt) {
            // conv output layout is [W, H, Cout, N]
            struct ggml_tensor* bias4 = ggml_reshape_4d(ctx, bias, 1, 1, bias->ne[0], 1);
            if (!ggml_can_repeat(bias4, gt)) {
              fprintf(stderr,
                      "[executorch-ggml] CONV bias not repeatable: b=(%lld,%lld,%lld,%lld) y=(%lld,%lld,%lld,%lld)\n",
                      (long long) bias4->ne[0], (long long) bias4->ne[1], (long long) bias4->ne[2], (long long) bias4->ne[3],
                      (long long) gt->ne[0], (long long) gt->ne[1], (long long) gt->ne[2], (long long) gt->ne[3]);
              ggml_free(ctx);
              return Error::InvalidArgument;
            }
            struct ggml_tensor* bias_rep = ggml_repeat(ctx, bias4, gt);
            if (bias_rep->type == GGML_TYPE_F16) {
              bias_rep = safe_ggml_cast(ctx, bias_rep, GGML_TYPE_F32);
            }
            gt = ggml_add(ctx, gt, bias_rep);
          }
          break;
        }

        case ggml_ir::OpCode::HARDTANH: {
          // hardtanh(x, min_val, max_val) -> clamp(x, min, max)
          float min_val = -1.0f;
          float max_val = 1.0f;
          if (t->op_params() && t->op_params()->size() >= 8) {
            const uint8_t* data = t->op_params()->data();
            memcpy(&min_val, data, sizeof(float));
            memcpy(&max_val, data + 4, sizeof(float));
          }
          gt = ggml_clamp(ctx, srcs[0], min_val, max_val);
          break;
        }

        case ggml_ir::OpCode::MEAN: {
          // mean(x, dims)
          // op_params: ndims (int32) + dims[] (int32)
          int32_t ndims = 0;
          int32_t dims[4] = {0, 0, 0, 0};
          if (t->op_params() && t->op_params()->size() >= 4) {
            const uint8_t* data = t->op_params()->data();
            memcpy(&ndims, data, sizeof(int32_t));
            if (ndims < 0 || ndims > 4) {
              ggml_free(ctx);
              return Error::InvalidArgument;
            }
            if (t->op_params()->size() < static_cast<size_t>(4 + ndims * 4)) {
              ggml_free(ctx);
              return Error::InvalidArgument;
            }
            for (int i = 0; i < ndims; ++i) {
              memcpy(&dims[i], data + 4 + i * 4, sizeof(int32_t));
            }
          }

          // Special-case global average pool for NCHW tensors: mean over (H, W)
          // which is dims (2, 3) in PyTorch order.
          if (ndims == 2 && dims[0] == 2 && dims[1] == 3) {
            // ggml tensor layout for NCHW is [W, H, C, N]
            const int k0 = static_cast<int>(srcs[0]->ne[0]);
            const int k1 = static_cast<int>(srcs[0]->ne[1]);
            gt = ggml_pool_2d(
                ctx,
                srcs[0],
                GGML_OP_POOL_AVG,
                k0,
                k1,
                k0,
                k1,
                0.0f,
                0.0f);
          } else {
            // Fallback: ggml_mean is all-dims mean.
            // TODO: implement axis-specific mean in IR/runtime.
            gt = ggml_mean(ctx, srcs[0]);
          }
          break;
        }

        case ggml_ir::OpCode::VIEW: {
          // reshape(x, new_shape)
          // Parse new shape from op_params
          if (!t->op_params() || t->op_params()->size() < 4) {
            ggml_free(ctx);
            return Error::InvalidArgument;
          }

          const uint8_t* data = t->op_params()->data();
          int32_t ndims = 0;
          memcpy(&ndims, data, sizeof(int32_t));

          if (t->op_params()->size() < 4 + ndims * 8) {
            ggml_free(ctx);
            return Error::InvalidArgument;
          }

          int64_t new_ne[4] = {1, 1, 1, 1};
          for (int32_t i = 0; i < ndims && i < 4; ++i) {
            memcpy(&new_ne[i], data + 4 + i * 8, sizeof(int64_t));
          }

          // Resolve sym_dim_ids for VIEW target shape.
          if (t->sym_dim_ids() && !sym_dim_values.empty()) {
            for (size_t d = 0; d < t->sym_dim_ids()->size() && d < 4; ++d) {
              int32_t sid = t->sym_dim_ids()->Get(d);
              if (sid == -2) {
                const uint8_t* code = nullptr;
                size_t code_len = 0;
                if (get_dim_expr_bytecode(t->sym_dim_exprs(), d, code, code_len)) {
                  new_ne[d] = eval_sym_expr(code, code_len, sym_dim_values);
                }
              } else if (sid >= 0) {
                auto it = sym_dim_values.find(sid);
                if (it != sym_dim_values.end()) new_ne[d] = it->second;
              }
            }
          }
          // Numel-inference fallback as safety net.
          int64_t src_numel = ggml_nelements(srcs[0]);
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

          gt = ggml_reshape_4d(ctx, ensure_cont(ctx, srcs[0]),
                              new_ne[0], new_ne[1], new_ne[2], new_ne[3]);
          break;
        }

        case ggml_ir::OpCode::PERMUTE: {
          // permute(x, dims)
          if (!t->op_params() || t->op_params()->size() < 4) {
            ggml_free(ctx);
            return Error::InvalidArgument;
          }

          const uint8_t* data = t->op_params()->data();
          int32_t ndims = 0;
          memcpy(&ndims, data, sizeof(int32_t));

          if (t->op_params()->size() < 4 + ndims * 4) {
            ggml_free(ctx);
            return Error::InvalidArgument;
          }

          int32_t perm[4] = {0, 1, 2, 3};
          for (int32_t i = 0; i < ndims && i < 4; ++i) {
            memcpy(&perm[i], data + 4 + i * 4, sizeof(int32_t));
          }

          // ggml_permute(ctx, tensor, axis0, axis1, axis2, axis3)
          gt = ggml_permute(ctx, srcs[0], perm[0], perm[1], perm[2], perm[3]);
          break;
        }

        case ggml_ir::OpCode::TRANSPOSE: {
          // transpose(x, dim0, dim1) via permute
          // op_params: (dim0:int32, dim1:int32, ndim:int32)
          if (!t->op_params() || t->op_params()->size() < 12) {
            ggml_free(ctx);
            return Error::InvalidArgument;
          }
          int32_t dim0 = 0, dim1 = 1, nd = 4;
          memcpy(&dim0, t->op_params()->data(), 4);
          memcpy(&dim1, t->op_params()->data() + 4, 4);
          memcpy(&nd, t->op_params()->data() + 8, 4);

          // Map PyTorch dims -> ggml axes using the PyTorch rank (not ggml_n_dims)
          int ax0 = (nd - 1) - dim0;
          int ax1 = (nd - 1) - dim1;
          int perm[4] = {0, 1, 2, 3};
          int tmp = perm[ax0];
          perm[ax0] = perm[ax1];
          perm[ax1] = tmp;
          gt = ggml_permute(ctx, srcs[0], perm[0], perm[1], perm[2], perm[3]);
          break;
        }

        case ggml_ir::OpCode::UNSQUEEZE: {
          // Unsqueeze adds a dim of size 1.
          // Reconstruct output shape from the runtime source shape + PyTorch
          // dim/rank so dynamic dims remain correct.
          struct ggml_tensor* a = srcs[0];
          int32_t pt_dim = 0;
          int32_t pt_ndim = -1;
          if (t->op_params() && t->op_params()->size() >= 4) {
            memcpy(&pt_dim, t->op_params()->data(), sizeof(int32_t));
          }
          if (t->op_params() && t->op_params()->size() >= 8) {
            memcpy(&pt_ndim, t->op_params()->data() + 4, sizeof(int32_t));
          }

          // Backward compatibility with older exports (dim only).
          if (pt_ndim <= 0) {
            pt_ndim = std::max(1, ggml_n_dims(a));
          }

          const int32_t out_ndim = pt_ndim + 1;
          if (pt_dim < 0) {
            pt_dim += out_ndim;
          }
          if (pt_dim < 0 || pt_dim > pt_ndim) {
            ggml_free(ctx);
            return Error::InvalidArgument;
          }

          // For 4D->5D unsqueeze patterns, the serializer already collapses
          // ranks >4 into ggml 4D. Keep using IR shape (if compatible).
          if (pt_ndim >= 4 || out_ndim > 4) {
            int64_t src_numel = ggml_nelements(a);
            int64_t ir_numel = ne[0] * ne[1] * ne[2] * ne[3];
            if (src_numel != ir_numel) {
              ggml_free(ctx);
              return Error::InvalidArgument;
            }
            gt = ggml_reshape_4d(ctx, ensure_cont(ctx, a), ne[0], ne[1], ne[2], ne[3]);
            break;
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
          gt = ggml_reshape_4d(ctx, ensure_cont(ctx, a), out_ne[0], out_ne[1], out_ne[2], out_ne[3]);
          break;
        }

        case ggml_ir::OpCode::SLICE: {
          // Limited slice support: step must be 1. Only supports slicing along
          // a single PyTorch dim using a view.
          // op_params: dim(i32), start(i64), end(i64), step(i64), ndim(u32)
          if (!t->op_params() || t->op_params()->size() < 28) {
            ggml_free(ctx);
            return Error::InvalidArgument;
          }
          int32_t dim = 0;
          int64_t start = 0, end = 0, step = 1;
          const uint8_t* p = t->op_params()->data();
          memcpy(&dim, p, 4);
          memcpy(&start, p + 4, 8);
          memcpy(&end, p + 12, 8);
          memcpy(&step, p + 20, 8);
          if (step != 1) {
            ggml_free(ctx);
            return Error::InvalidArgument;
          }

          // Read PyTorch rank from op_params (avoids ggml_n_dims which
          // undercounts when trailing dims are 1).
          uint32_t nd = 4;
          if (t->op_params()->size() >= 32) {
            memcpy(&nd, p + 28, 4);
          }
          struct ggml_tensor* a = srcs[0];
          int ax = (int(nd) - 1) - dim;

          // Resolve dynamic output shape from sym_dim_ids/sym_dim_exprs.
          // Only attempt when we have runtime symbol values; during init
          // (no overrides) the expressions evaluate to garbage (sym=0).
          int64_t resolved_slice_ne = ne[ax];  // start with static IR value
          if (!sym_dim_values.empty()) {
            for (int d = 0; d < 4; ++d) {
              int32_t sid = (t->sym_dim_ids() && d < (int)t->sym_dim_ids()->size())
                            ? t->sym_dim_ids()->Get(d) : -1;
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
            resolved_slice_ne = ne[ax];
          }

          // Use source tensor shape for non-sliced dims (not stale IR ne[]).
          // But keep the (possibly sym-resolved) IR ne for the sliced axis.
          for (int d = 0; d < 4; ++d) ne[d] = a->ne[d];

          // Derive start/end from the resolved output shape when op_params
          // contain the 2^62 sentinel (unresolvable SymInt at export time).
          constexpr int64_t SENTINEL = static_cast<int64_t>(1) << 62;
          int64_t actual_dim = a->ne[ax];
          bool start_is_sentinel = (start == SENTINEL);
          bool end_is_sentinel = (end == SENTINEL);

          // When start is a sentinel, derive it from the resolved output shape.
          // RelPositionalEncoding slices pe[:, center-T : center+T-1] where
          // center = pe_size/2+1. Centering: start = (pe_size - slice_len) / 2.
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
            // Clamp end to actual source dim (handles "slice to end" when
            // source shape changed due to dynamic dims).
            if (end > actual_dim) end = actual_dim;
            ne[ax] = end - start;

            // When sym_dim_ids resolved a different value for the sliced axis,
            // prefer it — the baked end/start from op_params may be trace-time
            // constants that don't match the runtime source shape.
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
          gt = ggml_view_4d(ctx, a, ne[0], ne[1], ne[2], ne[3],
                            a->nb[1], a->nb[2], a->nb[3], offset);
          // Only make contiguous if the slice created a non-contiguous view.
          // Slicing the innermost dimension or taking the full outer dimension
          // produces a contiguous result and doesn't need a copy.
          if (!ggml_is_contiguous(gt)) {
            gt = ggml_cont(ctx, gt);
          }
          break;
        }

        case ggml_ir::OpCode::CAT: {
          // Chain ggml_concat along pre-computed ggml axis
          if (!t->op_params() || t->op_params()->size() < 4) {
            ggml_free(ctx);
            return Error::InvalidArgument;
          }
          int32_t ax = 0;
          memcpy(&ax, t->op_params()->data(), 4);
          struct ggml_tensor* cur = ensure_cont(ctx, srcs[0]);
          for (size_t si = 1; si < srcs.size(); ++si) {
            cur = ggml_concat(ctx, cur, ensure_cont(ctx, srcs[si]), ax);
          }
          gt = cur;
          break;
        }

        case ggml_ir::OpCode::REPEAT_INTERLEAVE: {
          // Support for GQA expansion: repeat_interleave(x, repeats, dim).
          // op_params: (dim:int32, repeats:int32)
          // Unlike ggml_repeat (which tiles the whole tensor), repeat_interleave
          // repeats each element `repeats` times consecutively.
          // Example: [A, B, C].repeat_interleave(2) -> [A, A, B, B, C, C]
          if (!t->op_params() || t->op_params()->size() < 8) {
            ggml_free(ctx);
            return Error::InvalidArgument;
          }
          int32_t dim = 0, reps = 1;
          memcpy(&dim, t->op_params()->data(), 4);
          memcpy(&reps, t->op_params()->data() + 4, 4);
          // Only support repeating along PyTorch dim 1 (heads), which is ggml ax=(nd-2).
          if (dim != 1 || reps < 1) {
            ggml_free(ctx);
            return Error::InvalidArgument;
          }
          struct ggml_tensor* src = srcs[0];
          // src layout in ggml (ne): [ne0=D, ne1=T, ne2=H, ne3=B]
          // We need output [ne0=D, ne1=T, ne2=H*reps, ne3=B] where output
          // head `i` equals input head `i / reps` (interleaved, not tiled).
          //
          // Build by concatenating each individual head slice reps times:
          //   concat(head0, head0, head1, head1, ...) for reps=2
          //
          // ggml axis for head dim: for a 4D src, ne[2] is the head dim.
          int64_t D  = src->ne[0];
          int64_t T  = src->ne[1];
          int64_t H  = src->ne[2];
          int64_t B  = src->ne[3];
          // byte stride for one head slice along ne[2]
          size_t  row_stride  = src->nb[1];   // stride over T (ne[1])
          size_t  head_stride = src->nb[2];   // stride over H (ne[2])
          size_t  batch_stride= src->nb[3];   // stride over B (ne[3])
          struct ggml_tensor* result = nullptr;
          for (int64_t h = 0; h < H; ++h) {
            // Create a view of one head: shape [D, T, 1, B]
            struct ggml_tensor* head_slice = ggml_view_4d(
                ctx, src,
                D, T, 1, B,
                src->nb[1], src->nb[2], src->nb[3],
                h * head_stride  // offset in bytes to head h
            );
            for (int r = 0; r < reps; ++r) {
              if (result == nullptr) {
                result = head_slice;
              } else {
                result = ggml_concat(ctx, result, head_slice, 2 /* ne[2] axis */);
              }
            }
          }
          gt = result;
          break;
        }

        case ggml_ir::OpCode::INDEX: {
          // index(x, indices) along a single dim (currently dim=0 in lowering)
          // src_ids: [x, indices]
          // op_params: int32 dim
          if (!t->op_params() || t->op_params()->size() < 4) {
            ggml_free(ctx);
            return Error::InvalidArgument;
          }
          int32_t dim = 0;
          memcpy(&dim, t->op_params()->data(), 4);
          if (dim != 0) {
            // Only dim=0 supported for now
            ggml_free(ctx);
            return Error::InvalidArgument;
          }
          struct ggml_tensor* x = srcs[0];
          struct ggml_tensor* idx = srcs[1];
          if (idx->type == GGML_TYPE_I64) {
            idx = eager_cast_i64_to_i32(ctx, idx);
          } else if (idx->type != GGML_TYPE_I32) {
            idx = safe_ggml_cast(ctx, idx, GGML_TYPE_I32);
          }
          // ggml_get_rows supports src0 types F32/I32/F16/... but not I8.
          if (x->type == GGML_TYPE_I8) {
            x = safe_ggml_cast(ctx, x, GGML_TYPE_I32);
          }
          // ggml_get_rows selects rows by indices.
          gt = ggml_get_rows(ctx, x, idx);
          break;
        }

        case ggml_ir::OpCode::INDEX_MULTI: {
          // Multi-dimensional index gather: out = x[idx0, idx1, ...]
          // Runtime implementation uses ggml custom op so gather is computed
          // per-execution (not frozen at init time).
          int32_t ndims_hint = 0;
          if (t->op_params() && t->op_params()->size() >= 4) {
            memcpy(&ndims_hint, t->op_params()->data(), 4);
          }

          if (srcs.size() < 2) {
            ggml_free(ctx);
            return Error::InvalidArgument;
          }

          const int ndims = static_cast<int>(srcs.size()) - 1; // src + indices
          if (ndims < 1 || ndims > 4) {
            ggml_free(ctx);
            return Error::InvalidArgument;
          }
          if (ndims_hint > 0 && ndims_hint != ndims) {
            ggml_free(ctx);
            return Error::InvalidArgument;
          }

          struct ggml_tensor* src_x = srcs[0];
          ggml_type out_type = GGML_TYPE_F32;
          switch (t->type()) {
            case ggml_ir::TensorType::F16:
              out_type = GGML_TYPE_F16;
              break;
            case ggml_ir::TensorType::I64:
              out_type = GGML_TYPE_I64;
              break;
            case ggml_ir::TensorType::I32:
              out_type = GGML_TYPE_I32;
              break;
            case ggml_ir::TensorType::BOOL:
              out_type = GGML_TYPE_I32;
              break;
            case ggml_ir::TensorType::F32:
            default:
              out_type = GGML_TYPE_F32;
              break;
          }

          // Do not pre-cast src_x here. In particular, safe_ggml_cast(I32->I64)
          // is eager and can freeze uninitialized data for runtime tensors.
          // ggml_custom_index_multi handles src->dst scalar conversion at execute().

          std::vector<struct ggml_tensor*> custom_args;
          custom_args.reserve(1 + ndims);
          custom_args.push_back(src_x);
          for (int i = 0; i < ndims; ++i) {
            struct ggml_tensor* idx = srcs[1 + i];
            // Index tensors may be runtime inputs. Avoid eager I32->I64 casts here,
            // otherwise we'd freeze uninitialized build-time input data.
            if (idx->type != GGML_TYPE_I32 && idx->type != GGML_TYPE_I64) {
              idx = safe_ggml_cast(ctx, idx, GGML_TYPE_I32);
            }
            custom_args.push_back(idx);
          }

          // Output shape (in ggml ne order).
          int64_t out_ne0 = t->ne() ? (*t->ne())[0] : 1;
          int64_t out_ne1 = t->ne() && t->ne()->size() > 1 ? (*t->ne())[1] : 1;
          int64_t out_ne2 = t->ne() && t->ne()->size() > 2 ? (*t->ne())[2] : 1;
          int64_t out_ne3 = t->ne() && t->ne()->size() > 3 ? (*t->ne())[3] : 1;

          gt = ggml_custom_4d(
              ctx,
              out_type,
              out_ne0,
              out_ne1,
              out_ne2,
              out_ne3,
              custom_args.data(),
              static_cast<int>(custom_args.size()),
              ggml_custom_index_multi,
              GGML_N_TASKS_MAX,
              nullptr);
          pin_to_cpu(gt);
          break;
        }

        case ggml_ir::OpCode::INDEX_PUT: {
          // Implement index_put via runtime custom scatter on CPU.
          // This is robust for KV-cache updates across repeated decode steps.
          //
          // op_params: int32 nindices, int32 present_mask
          // src_ids: [dst, <index_tensors...>, values]
          if (!t->op_params() || t->op_params()->size() < 8) {
            ggml_free(ctx);
            return Error::InvalidArgument;
          }
          int32_t nidx = 0, pm = 0;
          memcpy(&nidx, t->op_params()->data(), 4);
          memcpy(&pm, t->op_params()->data() + 4, 4);

          if (srcs.size() < 3) {
            ggml_free(ctx);
            return Error::InvalidArgument;
          }

          struct ggml_tensor* dst = srcs.front();
          struct ggml_tensor* val = srcs.back();

          // Find the first present index tensor (we only support one index tensor for now).
          int idx_pos = -1;
          for (int i = 0; i < nidx; ++i) {
            if (pm & (1 << i)) {
              idx_pos = i;
              break;
            }
          }
          if (idx_pos < 0) {
            // no indices -> no-op
            gt = dst;
            break;
          }

          // In our serialized srcs: [dst] + present_indices + [val]
          // So the index tensor is srcs[1] when only one present index.
          struct ggml_tensor* idx = srcs[1];

          // Keep index tensors in runtime-safe integer types.
          if (idx->type != GGML_TYPE_I32 && idx->type != GGML_TYPE_I64) {
            idx = safe_ggml_cast(ctx, idx, GGML_TYPE_I32);
          }

          // Keep value/cache aligned for row writes in the callback.
          if (val->type != dst->type) {
            val = safe_ggml_cast(ctx, val, dst->type);
          }

          struct ggml_tensor* args[3] = {dst, idx, val};
          gt = ggml_custom_4d(
              ctx,
              dst->type,
              dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
              args, 3,
              ggml_custom_index_put_rows,
              GGML_N_TASKS_MAX,
              nullptr);
          pin_to_cpu(gt);
          break;
        }

        case ggml_ir::OpCode::CAST: {
          // Type cast with eager CPU conversion (ggml_cast doesn't support all combos).
          // op_params: int32 target_type (TensorType enum)
          struct ggml_tensor* src = srcs[0];

          // Determine target ggml type from op_params.
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

          // If types match, just alias.
          if (src->type == target_type) {
            gt = src;
            break;
          }

          // Eager scalar cast: compute immediately when source has data.
          // This is needed for enc_len computation chains where F32→I64
          // casts produce the final output scalar.
          if (ggml_nelements(src) == 1 && src->data) {
            ggml_set_no_alloc(ctx, false);
            gt = ggml_new_tensor_4d(ctx, target_type, 1, 1, 1, 1);
            ggml_set_no_alloc(ctx, true);
            gt->op = GGML_OP_NONE;
            double val = read_scalar_f64_ptr(static_cast<const char*>(src->data), src->type);
            write_scalar_f64_ptr(static_cast<char*>(gt->data), target_type, val);
            break;
          }

          // ggml CPU doesn't support GGML_OP_CPY from I64 to any other type,
          // so I64-source casts must be done eagerly on the CPU.
          if (src->type == GGML_TYPE_I64) {
            // Create output tensor for eager conversion.
            ggml_set_no_alloc(ctx, false);
            gt = ggml_new_tensor(ctx, target_type, GGML_MAX_DIMS, src->ne);
            ggml_set_no_alloc(ctx, true);

            // Check if src is an input tensor (data only available at execute time).
            bool is_input_src = false;
            for (const auto& [idx, inp_tensor] : input_pairs) {
              if (inp_tensor == src) {
                is_input_src = true;
                break;
              }
            }

            if (is_input_src && target_type == GGML_TYPE_I32) {
              // Defer I64→I32 conversion to execute().
              deferred_i64_to_i32.emplace_back(src, gt);
            } else if (src->data && gt->data) {
              // Constant source: convert now.
              const size_t nelem = ggml_nelements(src);
              const int64_t* src_data = static_cast<const int64_t*>(src->data);
              if (target_type == GGML_TYPE_I32) {
                int32_t* dst_data = static_cast<int32_t*>(gt->data);
                for (size_t i = 0; i < nelem; ++i) dst_data[i] = static_cast<int32_t>(src_data[i]);
              } else if (target_type == GGML_TYPE_F32) {
                float* dst_data = static_cast<float*>(gt->data);
                for (size_t i = 0; i < nelem; ++i) dst_data[i] = static_cast<float>(src_data[i]);
              }
            }
            gt->op = GGML_OP_NONE;  // Treated as constant in compute graph.
          } else {
            // All other casts go through safe_ggml_cast which handles
            // I32→F32, I32→other (via F32), and F16/BF16/F32 conversions.
            gt = safe_ggml_cast(ctx, src, target_type);
          }
          break;
        }

        case ggml_ir::OpCode::LLAMA_ATTENTION: {
          // Fused llama.cpp attention. For initial bring-up, we rely on ggml_flash_attn_ext.
          // src_ids: [q, k, v, (optional) mask]
          if (srcs.size() < 3) {
            ggml_free(ctx);
            return Error::InvalidArgument;
          }
          struct ggml_tensor* q = srcs[0];
          struct ggml_tensor* k = srcs[1];
          struct ggml_tensor* v = srcs[2];

          // flash_attn_ext dispatches via type traits (vec_dot / to_float),
          // so it supports F16, BF16, and F32 K/V natively.  Do NOT force
          // F16 — BF16 values that exceed F16 max (~65504) would overflow to
          // Inf, producing +Inf in the dot product accumulation.

          // Optional attention mask.  ggml_flash_attn_ext requires it to be:
          //   - F16 additive logit bias (0.0=attend, -inf=don't attend)
          //   - contiguous
          // The mask may arrive as:
          //   (a) I32/I64/F32 boolean from comparison ops (1=attend, 0=don't)
          //   (b) F16 already in additive format
          struct ggml_tensor* mask = nullptr;
          if (srcs.size() > 3 && srcs[3] != nullptr) {
            mask = srcs[3];
            if ((mask->type == GGML_TYPE_I32 || mask->type == GGML_TYPE_I64 || mask->type == GGML_TYPE_F32)
                && mask->op == GGML_OP_NONE && mask->data != nullptr) {
              // Pre-computed boolean mask — convert to F16 additive at build time.
              // This avoids graph splits and CPU↔GPU copies at each attention layer.
              int64_t n = ggml_nelements(mask);
              ggml_set_no_alloc(ctx, false);
              struct ggml_tensor* f16_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16,
                  mask->ne[0], mask->ne[1], mask->ne[2], mask->ne[3]);
              ggml_set_no_alloc(ctx, true);
              const ggml_fp16_t zero_f16 = ggml_fp32_to_fp16(0.0f);
              const ggml_fp16_t neg_inf_f16 = ggml_fp32_to_fp16(-INFINITY);
              ggml_fp16_t* dst_data = (ggml_fp16_t*)f16_mask->data;
              if (mask->type == GGML_TYPE_F32) {
                const float* src_data = (const float*)mask->data;
                for (int64_t j = 0; j < n; j++) {
                  dst_data[j] = src_data[j] != 0.0f ? zero_f16 : neg_inf_f16;
                }
              } else if (mask->type == GGML_TYPE_I32) {
                const int32_t* src_data = (const int32_t*)mask->data;
                for (int64_t j = 0; j < n; j++) {
                  dst_data[j] = src_data[j] ? zero_f16 : neg_inf_f16;
                }
              } else {
                const int64_t* src_data = (const int64_t*)mask->data;
                for (int64_t j = 0; j < n; j++) {
                  dst_data[j] = src_data[j] ? zero_f16 : neg_inf_f16;
                }
              }
              mask = f16_mask;
            } else if (mask->type == GGML_TYPE_I32 || mask->type == GGML_TYPE_I64) {
              // Non-constant I32/I64 boolean mask — fall back to custom op.
              struct ggml_tensor* args[1] = {mask};
              mask = ggml_custom_4d(ctx, GGML_TYPE_F16,
                  mask->ne[0], mask->ne[1], mask->ne[2], mask->ne[3],
                  args, 1, ggml_custom_bool_to_additive_mask,
                  GGML_N_TASKS_MAX, nullptr);
              pin_to_cpu(mask);
            } else if (mask->type == GGML_TYPE_F32) {
              // Non-constant F32 boolean mask from native comparison ops.
              // log({0.0,1.0}) → {-inf, 0.0} = additive mask, then cast to F16.
              mask = safe_ggml_cast(ctx, ggml_log(ctx, mask), GGML_TYPE_F16);
            } else if (mask->type != GGML_TYPE_F16) {
              mask = safe_ggml_cast(ctx, mask, GGML_TYPE_F16);
            }
            if (!ggml_is_contiguous(mask)) {
              mask = ggml_cont(ctx, mask);
            }
          }
          // scale = 1/sqrt(head_dim). head_dim is ne0 in ggml layout when tensors are [D, T, H, B]
          float scale = 1.0f;
          if (q->ne[0] > 0) {
            scale = 1.0f / std::sqrt((float) q->ne[0]);
          }

          struct ggml_tensor* attn = ggml_flash_attn_ext(
              ctx, ensure_cont(ctx, q), k, v, mask, scale, 0.0f, 0.0f);
          ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
          // ggml_flash_attn_ext returns [D, H, T, B], but the surrounding
          // lowered graph expects [D, T, H, B] (ggml order for [B, H, T, D]).
          gt = ggml_permute(ctx, attn, 0, 2, 1, 3);
          break;
        }

        case ggml_ir::OpCode::SUB: {
          struct ggml_tensor* a = srcs[0];
          struct ggml_tensor* b = srcs[1];
          if (auto* eager = try_eager_scalar_binop(ctx, a, b, '-')) {
            gt = eager;
            break;
          }
          // For int64 types with data, use eager element-wise sub with broadcasting.
          if (a->type == GGML_TYPE_I64 && b->type == GGML_TYPE_I64
              && a->data && b->data) {
            int64_t out_ne[4];
            for (int d = 0; d < 4; ++d) {
              out_ne[d] = std::max(a->ne[d], b->ne[d]);
            }
            ggml_set_no_alloc(ctx, false);
            gt = ggml_new_tensor_4d(ctx, GGML_TYPE_I64, out_ne[0], out_ne[1], out_ne[2], out_ne[3]);
            ggml_set_no_alloc(ctx, true);
            int64_t* out_data = static_cast<int64_t*>(gt->data);
            const int64_t* a_data = static_cast<const int64_t*>(a->data);
            const int64_t* b_data = static_cast<const int64_t*>(b->data);
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
          } else {
            if (!resolve_broadcast(a, b, "SUB")) {
              ggml_free(ctx);
              return Error::InvalidArgument;
            }
            gt = ggml_sub(ctx, a, b);
          }
          break;
        }

        case ggml_ir::OpCode::MUL_SCALAR: {
          // mul(x, scalar)
          // op_params: float32 scalar
          float scalar = 1.0f;
          if (t->op_params() && t->op_params()->size() >= 4) {
            memcpy(&scalar, t->op_params()->data(), sizeof(float));
          }
          gt = ggml_scale(ctx, srcs[0], scalar);
          break;
        }

        case ggml_ir::OpCode::POW: {
          // pow(x, exponent) where exponent is a scalar
          // For exponent=2, use ggml_sqr. Otherwise fall back to custom op or approximation.
          // op_params: float32 exponent
          float exponent = 2.0f;
          if (t->op_params() && t->op_params()->size() >= 4) {
            memcpy(&exponent, t->op_params()->data(), sizeof(float));
          }
          if (exponent == 2.0f) {
            gt = ggml_sqr(ctx, srcs[0]);
          } else if (exponent == 0.5f) {
            gt = ggml_sqrt(ctx, srcs[0]);
          } else {
            // General power: x^n = exp(n * log(x))
            // This won't work for negative x, but for RMSNorm (x^2) we use sqr above.
            struct ggml_tensor* log_x = ggml_log(ctx, srcs[0]);
            struct ggml_tensor* scaled = ggml_scale(ctx, log_x, exponent);
            gt = ggml_exp(ctx, scaled);
          }
          break;
        }

        case ggml_ir::OpCode::COS:
          gt = ggml_cos(ctx, srcs[0]);
          break;

        case ggml_ir::OpCode::SIN:
          gt = ggml_sin(ctx, srcs[0]);
          break;

        case ggml_ir::OpCode::BMM: {
          // Batch matrix multiply: bmm(a, b)
          // PyTorch: a is [B, M, K], b is [B, K, N], result is [B, M, N]
          // ggml ne format (reversed from PyTorch):
          //   a: [K, M, B, 1], b: [N, K, B, 1], out: [N, M, B, 1]
          // ggml_mul_mat(weight, input) requires weight->ne[0] == input->ne[0]
          // But b has ne[0]=N, a has ne[0]=K, so we need to transpose b first.
          struct ggml_tensor* a = srcs[0];  // [K, M, B, 1]
          struct ggml_tensor* b = srcs[1];  // [N, K, B, 1]
          
          // Transpose b to get [K, N, B, 1] so ne[0]=K matches a->ne[0]=K
          struct ggml_tensor* b_t = ggml_cont(ctx, ggml_transpose(ctx, b));
          gt = ggml_mul_mat(ctx, b_t, a);  // result: [N, M, B, 1]
          break;
        }

        case ggml_ir::OpCode::SIGMOID:
          gt = ggml_sigmoid(ctx, srcs[0]);
          break;

        case ggml_ir::OpCode::SOFTMAX: {
          // softmax(x, dim)
          // op_params: int32 dim, int32 ndim
          // ggml_soft_max operates on the innermost dimension (ne0).
          // If dim != -1 in PyTorch, we need to permute first.
          int32_t dim = -1, ndim = 4;
          if (t->op_params() && t->op_params()->size() >= 8) {
            memcpy(&dim, t->op_params()->data(), 4);
            memcpy(&ndim, t->op_params()->data() + 4, 4);
          }

          // Normalize negative dim
          if (dim < 0) dim = ndim + dim;

          // ggml axis = (ndim - 1) - pytorch_dim
          int ggml_axis = (ndim - 1) - dim;

          struct ggml_tensor* x = srcs[0];
          if (ggml_axis == 0) {
            // Softmax on innermost dim - direct
            gt = ggml_soft_max(ctx, x);
          } else {
            // Need to permute so that the softmax dim becomes axis 0
            // Create permutation that swaps axis 0 with ggml_axis
            int perm[4] = {0, 1, 2, 3};
            perm[0] = ggml_axis;
            perm[ggml_axis] = 0;
            x = ggml_permute(ctx, x, perm[0], perm[1], perm[2], perm[3]);
            x = ensure_cont(ctx, x);
            x = ggml_soft_max(ctx, x);
            // Permute back
            gt = ggml_permute(ctx, x, perm[0], perm[1], perm[2], perm[3]);
          }
          break;
        }

        case ggml_ir::OpCode::WHERE: {
          // where(cond, x, y) - select x where cond is true, y otherwise
          // ggml doesn't have a direct where op, so we implement it as:
          // result = cond * x + (1 - cond) * y
          // But this requires cond to be float. For now, cast cond to float.
          struct ggml_tensor* cond = srcs[0];
          struct ggml_tensor* x = srcs[1];
          struct ggml_tensor* y = srcs[2];

          // Cast condition to float if needed.
          // Legacy .pte files may store bool masks as F16 with 0.0/-inf
          // (for flash_attn additive bias). Clamp to [0, 1] to recover
          // proper boolean semantics for the arithmetic emulation below.
          if (cond->type != GGML_TYPE_F32) {
            cond = safe_ggml_cast(ctx, cond, GGML_TYPE_F32);
          }
          cond = ggml_clamp(ctx, cond, 0.0f, 1.0f);

          // Ensure all three tensors have the same shape for the
          // element-wise arithmetic that implements WHERE.
          // Find the "largest" shape (most elements) and repeat others to it.
          struct ggml_tensor* target = x;
          if (ggml_nelements(y) > ggml_nelements(target)) target = y;
          if (ggml_nelements(cond) > ggml_nelements(target)) target = cond;

          if (!ggml_are_same_shape(cond, target) && ggml_can_repeat(cond, target)) {
            cond = ggml_repeat(ctx, cond, target);
          }
          if (!ggml_are_same_shape(x, target) && ggml_can_repeat(x, target)) {
            x = ggml_repeat(ctx, x, target);
          }
          if (!ggml_are_same_shape(y, target) && ggml_can_repeat(y, target)) {
            y = ggml_repeat(ctx, y, target);
          }

          // result = cond * x + (1 - cond) * y
          struct ggml_tensor* one = make_f32_scalar(ctx, 1.0f);
          struct ggml_tensor* one_rep = ggml_repeat(ctx, one, cond);
          struct ggml_tensor* not_cond = ggml_sub(ctx, one_rep, cond);

          // ggml_mul(a, b) requires b repeatable to a — put bigger first.
          struct ggml_tensor* x_part = ggml_mul(ctx, x, cond);
          struct ggml_tensor* y_part = ggml_mul(ctx, y, not_cond);
          gt = ggml_add(ctx, x_part, y_part);
          break;
        }

        case ggml_ir::OpCode::ARANGE: {
          // arange(start, step) - generates [start, start+step, start+2*step, ...]
          // op_params: float64 start, float64 step
          double start = 0.0, step = 1.0;
          if (t->op_params() && t->op_params()->size() >= 16) {
            memcpy(&start, t->op_params()->data(), 8);
            memcpy(&step, t->op_params()->data() + 8, 8);
          }

          // Determine output type and shape from IR
          ggml_type out_type = GGML_TYPE_I64;
          switch (t->type()) {
            case ggml_ir::TensorType::F32: out_type = GGML_TYPE_F32; break;
            case ggml_ir::TensorType::F16: out_type = GGML_TYPE_F16; break;
            case ggml_ir::TensorType::I32: out_type = GGML_TYPE_I32; break;
            case ggml_ir::TensorType::I64: out_type = GGML_TYPE_I64; break;
            default: out_type = GGML_TYPE_I64; break;
          }

          // sym_dim_ids resolution already applied to ne[] above.

          ggml_set_no_alloc(ctx, false);
          gt = ggml_new_tensor_4d(ctx, out_type, ne[0], ne[1], ne[2], ne[3]);
          ggml_set_no_alloc(ctx, true);
          const size_t nelem = ggml_nelements(gt);

          // Fill with arange values
          if (out_type == GGML_TYPE_I64) {
            int64_t* data = static_cast<int64_t*>(gt->data);
            for (size_t i = 0; i < nelem; ++i) {
              data[i] = static_cast<int64_t>(start + i * step);
            }
          } else if (out_type == GGML_TYPE_I32) {
            int32_t* data = static_cast<int32_t*>(gt->data);
            for (size_t i = 0; i < nelem; ++i) {
              data[i] = static_cast<int32_t>(start + i * step);
            }
          } else {
            float* data = static_cast<float*>(gt->data);
            for (size_t i = 0; i < nelem; ++i) {
              data[i] = static_cast<float>(start + i * step);
            }
          }
          break;
        }

        case ggml_ir::OpCode::FULL: {
          // full(fill_value) - creates tensor filled with fill_value
          // op_params: float64 fill_value
          double fill_value = 0.0;
          if (t->op_params() && t->op_params()->size() >= 8) {
            memcpy(&fill_value, t->op_params()->data(), 8);
          }

          ggml_type out_type = GGML_TYPE_F32;
          switch (t->type()) {
            case ggml_ir::TensorType::F32: out_type = GGML_TYPE_F32; break;
            case ggml_ir::TensorType::F16: out_type = GGML_TYPE_F16; break;
            case ggml_ir::TensorType::I32: out_type = GGML_TYPE_I32; break;
            case ggml_ir::TensorType::I64: out_type = GGML_TYPE_I64; break;
            case ggml_ir::TensorType::BOOL: out_type = GGML_TYPE_I32; break;
            default: out_type = GGML_TYPE_F32; break;
          }

          // sym_dim_ids resolution already applied to ne[] above.
          ggml_set_no_alloc(ctx, false);
          gt = ggml_new_tensor_4d(ctx, out_type, ne[0], ne[1], ne[2], ne[3]);
          ggml_set_no_alloc(ctx, true);
          const size_t nelem = ggml_nelements(gt);

          if (out_type == GGML_TYPE_I64) {
            int64_t* data = static_cast<int64_t*>(gt->data);
            int64_t v = static_cast<int64_t>(fill_value);
            for (size_t i = 0; i < nelem; ++i) data[i] = v;
          } else if (out_type == GGML_TYPE_I32) {
            int32_t* data = static_cast<int32_t*>(gt->data);
            int32_t v = static_cast<int32_t>(fill_value);
            for (size_t i = 0; i < nelem; ++i) data[i] = v;
          } else {
            float* data = static_cast<float*>(gt->data);
            float v = static_cast<float>(fill_value);
            for (size_t i = 0; i < nelem; ++i) data[i] = v;
          }
          break;
        }

        case ggml_ir::OpCode::CUMSUM: {
          // cumsum(x, dim) - cumulative sum along dimension
          // op_params: int32 dim, int32 ndim
          int32_t dim = 0, ndim = 4;
          if (t->op_params() && t->op_params()->size() >= 8) {
            memcpy(&dim, t->op_params()->data(), 4);
            memcpy(&ndim, t->op_params()->data() + 4, 4);
          }

          struct ggml_tensor* src = srcs[0];
          ggml_type out_type = src->type;

          // CUMSUM output shape = input shape (use input, not stale IR ne[]).
          ggml_set_no_alloc(ctx, false);
          gt = ggml_new_tensor_4d(ctx, out_type, src->ne[0], src->ne[1], src->ne[2], src->ne[3]);
          ggml_set_no_alloc(ctx, true);

          // Convert PyTorch dim to ggml axis
          int ggml_axis = (ndim - 1) - dim;
          const size_t nelem = ggml_nelements(gt);

          // Simple cumsum implementation for I64 type
          if (out_type == GGML_TYPE_I64) {
            const int64_t* in_data = static_cast<const int64_t*>(src->data);
            int64_t* out_data = static_cast<int64_t*>(gt->data);

            // For now, handle axis 0 (innermost in ggml) specially
            int64_t stride = 1;
            for (int ax = 0; ax < ggml_axis; ++ax) stride *= src->ne[ax];
            int64_t dim_size = src->ne[ggml_axis];
            int64_t outer_size = nelem / (dim_size * stride);

            for (int64_t outer = 0; outer < outer_size; ++outer) {
              for (int64_t inner = 0; inner < stride; ++inner) {
                int64_t cumsum = 0;
                for (int64_t d = 0; d < dim_size; ++d) {
                  int64_t idx = outer * dim_size * stride + d * stride + inner;
                  cumsum += in_data[idx];
                  out_data[idx] = cumsum;
                }
              }
            }
          } else {
            // F32 fallback
            const float* in_data = static_cast<const float*>(src->data);
            float* out_data = static_cast<float*>(gt->data);

            int64_t stride = 1;
            for (int ax = 0; ax < ggml_axis; ++ax) stride *= src->ne[ax];
            int64_t dim_size = src->ne[ggml_axis];
            int64_t outer_size = nelem / (dim_size * stride);

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
          break;
        }

        case ggml_ir::OpCode::EQ: {
          // eq(a, b) or eq(a, scalar) → 1 - clamp(step(a-b) + step(b-a), 0, 1)
          double scalar = 0.0;
          int32_t is_scalar = 0;
          if (t->op_params() && t->op_params()->size() >= 12) {
            memcpy(&scalar, t->op_params()->data(), 8);
            memcpy(&is_scalar, t->op_params()->data() + 8, 4);
          }
          struct ggml_tensor* a = srcs[0];
          struct ggml_tensor* b;
          if (is_scalar) {
            ggml_set_no_alloc(ctx, false);
            b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, a->ne[0], a->ne[1], a->ne[2], a->ne[3]);
            ggml_set_no_alloc(ctx, true);
            const size_t n = ggml_nelements(b);
            float* d = (float*)b->data;
            for (size_t j = 0; j < n; ++j) d[j] = (float)scalar;
          } else {
            b = srcs[1];
          }
          if (a->type != GGML_TYPE_F32) a = safe_ggml_cast(ctx, a, GGML_TYPE_F32);
          if (b->type != GGML_TYPE_F32) b = safe_ggml_cast(ctx, b, GGML_TYPE_F32);
          if (!resolve_broadcast(a, b, "EQ")) { ggml_free(ctx); return Error::InvalidArgument; }
          struct ggml_tensor* diff = ggml_sub(ctx, a, b);
          struct ggml_tensor* ne_val = ggml_clamp(ctx, ggml_add(ctx, ggml_step(ctx, diff), ggml_step(ctx, ggml_neg(ctx, diff))), 0.0f, 1.0f);
          gt = ggml_add(ctx, ggml_neg(ctx, ne_val), make_f32_scalar(ctx, 1.0f));
          break;
        }

        case ggml_ir::OpCode::NE: {
          // ne(a, b) or ne(a, scalar) → clamp(step(a-b) + step(b-a), 0, 1)
          double scalar = 0.0;
          int32_t is_scalar = 1;
          if (t->op_params() && t->op_params()->size() >= 12) {
            memcpy(&scalar, t->op_params()->data(), 8);
            memcpy(&is_scalar, t->op_params()->data() + 8, 4);
          }
          struct ggml_tensor* a = srcs[0];
          struct ggml_tensor* b;
          if (is_scalar) {
            ggml_set_no_alloc(ctx, false);
            b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, a->ne[0], a->ne[1], a->ne[2], a->ne[3]);
            ggml_set_no_alloc(ctx, true);
            const size_t n = ggml_nelements(b);
            float* d = (float*)b->data;
            for (size_t j = 0; j < n; ++j) d[j] = (float)scalar;
          } else {
            b = srcs[1];
          }
          if (a->type != GGML_TYPE_F32) a = safe_ggml_cast(ctx, a, GGML_TYPE_F32);
          if (b->type != GGML_TYPE_F32) b = safe_ggml_cast(ctx, b, GGML_TYPE_F32);
          if (!resolve_broadcast(a, b, "NE")) { ggml_free(ctx); return Error::InvalidArgument; }
          struct ggml_tensor* diff = ggml_sub(ctx, a, b);
          gt = ggml_clamp(ctx, ggml_add(ctx, ggml_step(ctx, diff), ggml_step(ctx, ggml_neg(ctx, diff))), 0.0f, 1.0f);
          break;
        }

        // ---- Comparison / bitwise / logical ops ----
        // Decomposed into native ggml ops (step/sub/neg/add/clamp/mul) so
        // the entire graph stays on GPU (Metal + CUDA).  All output F32 {0.0, 1.0}.

        case ggml_ir::OpCode::LE: {
          // le(a, b) = 1 - step(a - b)
          struct ggml_tensor* a = srcs[0]; struct ggml_tensor* b = srcs[1];
          if (a->type != GGML_TYPE_F32) a = safe_ggml_cast(ctx, a, GGML_TYPE_F32);
          if (b->type != GGML_TYPE_F32) b = safe_ggml_cast(ctx, b, GGML_TYPE_F32);
          if (!resolve_broadcast(a, b, "LE")) { ggml_free(ctx); return Error::InvalidArgument; }
          gt = ggml_add(ctx, ggml_neg(ctx, ggml_step(ctx, ggml_sub(ctx, a, b))), make_f32_scalar(ctx, 1.0f));
          break;
        }
        case ggml_ir::OpCode::LT: {
          // lt(a, b) = step(b - a) = step(-(a - b))
          struct ggml_tensor* a = srcs[0]; struct ggml_tensor* b = srcs[1];
          if (a->type != GGML_TYPE_F32) a = safe_ggml_cast(ctx, a, GGML_TYPE_F32);
          if (b->type != GGML_TYPE_F32) b = safe_ggml_cast(ctx, b, GGML_TYPE_F32);
          if (!resolve_broadcast(a, b, "LT")) { ggml_free(ctx); return Error::InvalidArgument; }
          gt = ggml_step(ctx, ggml_neg(ctx, ggml_sub(ctx, a, b)));
          break;
        }
        case ggml_ir::OpCode::GT: {
          // gt(a, b) = step(a - b)
          struct ggml_tensor* a = srcs[0]; struct ggml_tensor* b = srcs[1];
          if (a->type != GGML_TYPE_F32) a = safe_ggml_cast(ctx, a, GGML_TYPE_F32);
          if (b->type != GGML_TYPE_F32) b = safe_ggml_cast(ctx, b, GGML_TYPE_F32);
          if (!resolve_broadcast(a, b, "GT")) { ggml_free(ctx); return Error::InvalidArgument; }
          gt = ggml_step(ctx, ggml_sub(ctx, a, b));
          break;
        }
        case ggml_ir::OpCode::GE: {
          // ge(a, b) = 1 - step(b - a) = 1 - step(-(a - b))
          struct ggml_tensor* a = srcs[0]; struct ggml_tensor* b = srcs[1];
          if (a->type != GGML_TYPE_F32) a = safe_ggml_cast(ctx, a, GGML_TYPE_F32);
          if (b->type != GGML_TYPE_F32) b = safe_ggml_cast(ctx, b, GGML_TYPE_F32);
          if (!resolve_broadcast(a, b, "GE")) { ggml_free(ctx); return Error::InvalidArgument; }
          gt = ggml_add(ctx, ggml_neg(ctx, ggml_step(ctx, ggml_neg(ctx, ggml_sub(ctx, a, b)))), make_f32_scalar(ctx, 1.0f));
          break;
        }
        case ggml_ir::OpCode::BITWISE_AND: {
          // and(a, b) = a * b  (both {0.0, 1.0})
          struct ggml_tensor* a = srcs[0]; struct ggml_tensor* b = srcs[1];
          if (a->type != GGML_TYPE_F32) a = safe_ggml_cast(ctx, a, GGML_TYPE_F32);
          if (b->type != GGML_TYPE_F32) b = safe_ggml_cast(ctx, b, GGML_TYPE_F32);
          if (!resolve_broadcast(a, b, "BITWISE_AND")) { ggml_free(ctx); return Error::InvalidArgument; }
          gt = ggml_mul(ctx, a, b);
          break;
        }
        case ggml_ir::OpCode::BITWISE_OR: {
          // or(a, b) = clamp(a + b, 0, 1)
          struct ggml_tensor* a = srcs[0]; struct ggml_tensor* b = srcs[1];
          if (a->type != GGML_TYPE_F32) a = safe_ggml_cast(ctx, a, GGML_TYPE_F32);
          if (b->type != GGML_TYPE_F32) b = safe_ggml_cast(ctx, b, GGML_TYPE_F32);
          if (!resolve_broadcast(a, b, "BITWISE_OR")) { ggml_free(ctx); return Error::InvalidArgument; }
          gt = ggml_clamp(ctx, ggml_add(ctx, a, b), 0.0f, 1.0f);
          break;
        }
        case ggml_ir::OpCode::LOGICAL_NOT: {
          // not(x) = 1 - x  (x in {0.0, 1.0})
          struct ggml_tensor* x = srcs[0];
          if (x->type != GGML_TYPE_F32) x = safe_ggml_cast(ctx, x, GGML_TYPE_F32);
          gt = ggml_add(ctx, ggml_neg(ctx, x), make_f32_scalar(ctx, 1.0f));
          break;
        }

        case ggml_ir::OpCode::ANY: {
          // any(x, dim) - reduce any along dimension
          int32_t dim = 0, ndim = 4;
          if (t->op_params() && t->op_params()->size() >= 8) {
            memcpy(&dim, t->op_params()->data(), 4);
            memcpy(&ndim, t->op_params()->data() + 4, 4);
          }
          struct ggml_tensor* src = srcs[0];
          int ggml_axis = (ndim - 1) - dim;

          if (src->op == GGML_OP_NONE && src->data != nullptr) {
            // Eager path: source data available at build time.
            int64_t any_ne[4] = {src->ne[0], src->ne[1], src->ne[2], src->ne[3]};
            any_ne[ggml_axis] = 1;
            ggml_set_no_alloc(ctx, false);
            gt = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, any_ne[0], any_ne[1], any_ne[2], any_ne[3]);
            ggml_set_no_alloc(ctx, true);
            const size_t nelem_out = ggml_nelements(gt);
            float* out_data = static_cast<float*>(gt->data);

            int64_t stride = 1;
            for (int ax = 0; ax < ggml_axis; ++ax) stride *= src->ne[ax];
            int64_t dim_size = src->ne[ggml_axis];
            int64_t outer_size = nelem_out / stride;
            if (outer_size == 0) outer_size = 1;

            if (src->type == GGML_TYPE_F32) {
              const float* in_data = static_cast<const float*>(src->data);
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
              const int32_t* in_data = static_cast<const int32_t*>(src->data);
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
          } else {
            // Graph-op path: source is a graph node, use native ggml ops.
            // any(x, dim) = step(sum_rows(abs(x))) along the target axis.
            // ggml_sum_rows reduces dim 0, so permute target axis to dim 0 first.
            struct ggml_tensor* x = src;
            if (x->type != GGML_TYPE_F32) x = safe_ggml_cast(ctx, x, GGML_TYPE_F32);
            if (ggml_axis != 0) {
              // Permute to bring ggml_axis to dim 0.
              int axes[4] = {0, 1, 2, 3};
              axes[0] = ggml_axis; axes[ggml_axis] = 0;
              x = ggml_cont(ctx, ggml_permute(ctx, x, axes[0], axes[1], axes[2], axes[3]));
            }
            x = ggml_step(ctx, ggml_sum_rows(ctx, ggml_abs(ctx, x)));
            if (ggml_axis != 0) {
              // Permute back.
              int axes[4] = {0, 1, 2, 3};
              axes[0] = ggml_axis; axes[ggml_axis] = 0;
              x = ggml_permute(ctx, x, axes[0], axes[1], axes[2], axes[3]);
            }
            gt = x;
          }
          break;
        }

        case ggml_ir::OpCode::UPDATE_CACHE: {
          // update_cache(cache, value, start_pos)
          // Inserts value into cache starting at start_pos along the seq dimension.
          // op_params: int32 seq_dim (PyTorch dimension for sequence)
          // src_ids: [cache, value, start_pos]
          //
          // cache shape: [batch, seq_len, n_heads, head_dim] or [batch, n_heads, seq_len, head_dim]
          // value shape: [batch, seq_len_new, n_heads, head_dim] or similar
          // start_pos: scalar tensor containing the starting position

          if (srcs.size() < 3) {
            fprintf(stderr, "[executorch-ggml] UPDATE_CACHE: expected 3 sources\n");
            ggml_free(ctx);
            return Error::InvalidArgument;
          }

          struct ggml_tensor* cache = srcs[0];
          struct ggml_tensor* value = srcs[1];
          struct ggml_tensor* start_pos_tensor = srcs[2];

          // Get seq_dim from op_params
          int32_t seq_dim_pytorch = 1;  // default
          if (t->op_params() && t->op_params()->size() >= 4) {
            memcpy(&seq_dim_pytorch, t->op_params()->data(), 4);
          }

          // Convert PyTorch seq_dim to ggml axis
          // PyTorch: [batch, seq, heads, dim] -> seq_dim=1
          // ggml:    [dim, heads, seq, batch] -> ggml_axis=1
          int ndim = 4;
          int ggml_axis = (ndim - 1) - seq_dim_pytorch;

          // Get start position value
          int64_t start_pos = 0;
          if (start_pos_tensor->type == GGML_TYPE_I64) {
            start_pos = *(const int64_t*)start_pos_tensor->data;
          } else if (start_pos_tensor->type == GGML_TYPE_I32) {
            start_pos = *(const int32_t*)start_pos_tensor->data;
          }

          // Use ggml_set_rows if we have indices, otherwise use a simple copy
          // For contiguous update, we can create a view and copy
          // cache[:, start_pos:start_pos+seq_len, :, :] = value

          // Create indices tensor for ggml_set_rows
          int64_t seq_len_new = value->ne[ggml_axis];
          ggml_set_no_alloc(ctx, false);
          struct ggml_tensor* indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, seq_len_new);
          ggml_set_no_alloc(ctx, true);
          int64_t* idx_data = static_cast<int64_t*>(indices->data);
          for (int64_t i = 0; i < seq_len_new; ++i) {
            idx_data[i] = start_pos + i;
          }

          // ggml_set_rows requires value (b) and cache (a) to be F32.
          if (value->type != GGML_TYPE_F32) {
            value = safe_ggml_cast(ctx, value, GGML_TYPE_F32);
          }
          if (cache->type != GGML_TYPE_F32) {
            cache = safe_ggml_cast(ctx, cache, GGML_TYPE_F32);
          }

          // Use ggml_set_rows to scatter values into cache
          gt = ggml_set_rows(ctx, cache, value, indices);
          break;
        }

        case ggml_ir::OpCode::LAYER_NORM: {
          // src_ids: [x, weight?, bias?]
          // op_params: float32 eps, int32 has_weight, int32 has_bias
          float eps = 1e-5f;
          int32_t has_weight = 0, has_bias = 0;
          if (t->op_params() && t->op_params()->size() >= 12) {
            const uint8_t* data = t->op_params()->data();
            memcpy(&eps, data, sizeof(float));
            memcpy(&has_weight, data + 4, sizeof(int32_t));
            memcpy(&has_bias, data + 8, sizeof(int32_t));
          }

          gt = ggml_norm(ctx, ensure_cont(ctx, srcs[0]), eps);

          if (has_weight && srcs.size() > 1) {
            struct ggml_tensor* w = srcs[1];
            gt = ggml_mul(ctx, gt, w);
          }
          if (has_bias) {
            int bias_idx = has_weight ? 2 : 1;
            if ((int)srcs.size() > bias_idx) {
              struct ggml_tensor* b = srcs[bias_idx];
              gt = ggml_add(ctx, gt, b);
            }
          }
          break;
        }

        case ggml_ir::OpCode::BATCH_NORM: {
          // src_ids: [x, weight, bias, mean, var]
          // op_params: float32 eps
          // Compute: scale = weight / sqrt(var + eps), shift = bias - mean * scale
          // Then: y = x * scale + shift
          if (srcs.size() < 5) {
            fprintf(stderr, "[executorch-ggml] BATCH_NORM: expected 5 sources, got %zu\n", srcs.size());
            ggml_free(ctx);
            return Error::InvalidArgument;
          }

          float eps = 1e-5f;
          if (t->op_params() && t->op_params()->size() >= 4) {
            memcpy(&eps, t->op_params()->data(), sizeof(float));
          }

          struct ggml_tensor* x      = srcs[0];
          struct ggml_tensor* weight  = srcs[1];  // gamma [C]
          struct ggml_tensor* bias    = srcs[2];  // beta [C]
          struct ggml_tensor* mean    = srcs[3];  // running_mean [C]
          struct ggml_tensor* var     = srcs[4];  // running_var [C]

          // Eagerly compute scale and shift from constant running stats.
          // weight, bias, mean, var are all 1D constants [C].
          // scale = weight / sqrt(var + eps)
          // shift = bias - mean * scale
          //
          // For ggml, the BN input x has layout [innermost, ..., C, batch]
          // where C is ne[ggml_axis]. For 1D input [batch, C, L] → ggml [L, C, batch, 1]
          // and for 2D [batch, C, H, W] → ggml [W, H, C, batch].
          // The 1D scale/shift [C] → ggml [C, 1, 1, 1].
          // ggml_mul/ggml_add broadcast when the smaller tensor divides the larger.
          // We need scale as [1, C, 1, 1] to match axis 1 (C channel).
          // Reshape scale/shift to [1, C, 1, 1] for correct broadcasting.

          struct ggml_tensor* var_eps = ggml_add1(ctx, var, make_f32_scalar(ctx, eps));
          struct ggml_tensor* inv_std = ggml_sqrt(ctx, var_eps);
          // 1/inv_std
          struct ggml_tensor* one = make_f32_scalar(ctx, 1.0f);
          struct ggml_tensor* one_rep = ggml_repeat(ctx, one, inv_std);
          struct ggml_tensor* recip_std = ggml_div(ctx, one_rep, inv_std);

          struct ggml_tensor* scale = ggml_mul(ctx, weight, recip_std);
          struct ggml_tensor* mean_scaled = ggml_mul(ctx, mean, scale);
          struct ggml_tensor* shift = ggml_sub(ctx, bias, mean_scaled);

          // Reshape scale/shift for broadcasting: [C] → [1, C, 1, 1]
          int64_t C = scale->ne[0];
          struct ggml_tensor* scale4 = ggml_reshape_4d(ctx, scale, 1, C, 1, 1);
          struct ggml_tensor* shift4 = ggml_reshape_4d(ctx, shift, 1, C, 1, 1);

          gt = ggml_mul(ctx, x, ggml_repeat(ctx, scale4, x));
          gt = ggml_add(ctx, gt, ggml_repeat(ctx, shift4, gt));
          break;
        }

        case ggml_ir::OpCode::ARGMAX: {
          // src_ids: [x]
          // op_params: int32 dim, int32 ndim
          // ggml_argmax reduces along axis 0 (innermost).
          // If dim maps to ggml axis 0, use directly; otherwise permute.
          int32_t dim = -1, ndim = 1;
          if (t->op_params() && t->op_params()->size() >= 8) {
            const uint8_t* data = t->op_params()->data();
            memcpy(&dim, data, sizeof(int32_t));
            memcpy(&ndim, data + 4, sizeof(int32_t));
          }

          // Normalize negative dim
          if (dim < 0) dim += ndim;

          // Convert PyTorch dim to ggml axis: ggml_axis = (ndim-1) - dim
          int ggml_axis = (ndim - 1) - dim;

          struct ggml_tensor* x = srcs[0];

          if (ggml_axis == 0) {
            // Common case: argmax along last PyTorch dim → ggml axis 0
            gt = ggml_argmax(ctx, x);
          } else {
            // Need to permute so target axis becomes axis 0
            // Build permutation that swaps axis 0 and ggml_axis
            int perm[4] = {0, 1, 2, 3};
            perm[0] = ggml_axis;
            perm[ggml_axis] = 0;
            struct ggml_tensor* xp = ggml_permute(ctx, x, perm[0], perm[1], perm[2], perm[3]);
            xp = ggml_cont(ctx, xp);
            gt = ggml_argmax(ctx, xp);
          }
          break;
        }

        case ggml_ir::OpCode::CONV_1D:
        case ggml_ir::OpCode::CONV_1D_DW: {
          // src_ids: [weight, input, bias?]
          // op_params: int32 stride, pad, dilation, groups
          if (srcs.size() < 2) {
            ggml_free(ctx);
            return Error::InvalidArgument;
          }

          struct ggml_tensor* weight = srcs[0];
          struct ggml_tensor* input = srcs[1];
          struct ggml_tensor* bias = srcs.size() > 2 ? srcs[2] : nullptr;

          int32_t stride = 1, pad = 0, dilation = 1, groups = 1;
          if (t->op_params() && t->op_params()->size() >= 16) {
            const uint8_t* data = t->op_params()->data();
            memcpy(&stride, data, sizeof(int32_t));
            memcpy(&pad, data + 4, sizeof(int32_t));
            memcpy(&dilation, data + 8, sizeof(int32_t));
            memcpy(&groups, data + 12, sizeof(int32_t));
          }

          // Pointwise conv (k=1, s=1, p=0, d=1, groups=1) is just a matmul.
          // Bypass ggml_conv_1d (which uses F16 im2col) to stay in F32.
          int64_t kernel_size = weight->ne[0];
          bool is_pointwise = (kernel_size == 1 && stride == 1 && pad == 0
                               && dilation == 1 && groups == 1);

          if (is_pointwise) {
            // weight ggml shape: [K=1, C_in, C_out, 1] → squeeze to [C_in, C_out]
            struct ggml_tensor* w2d = ggml_reshape_2d(ctx, weight,
                                                      weight->ne[1], weight->ne[2]);
            // input ggml shape: [L, C_in, B, 1]
            // ggml_mul_mat contracts on ne[0], but input has L in ne[0].
            // Transpose so C_in is in ne[0], do matmul, transpose back.
            struct ggml_tensor* inp_t = ggml_cont(ctx, ggml_transpose(ctx, input));
            // inp_t: [C_in, L, B, 1]
            struct ggml_tensor* mm = ggml_mul_mat(ctx, w2d, inp_t);
            // mm: [C_out, L, B, 1]
            gt = ggml_cont(ctx, ggml_transpose(ctx, mm));
            // gt: [L, C_out, B, 1]
          } else if (op == ggml_ir::OpCode::CONV_1D_DW || groups > 1) {
            // Use custom F32 depthwise conv to avoid ggml's im2col F16 path.
            // weight ggml shape: [K, 1, C, 1], input ggml shape: [L, C, B, 1]
            int64_t L_out = (input->ne[0] + 2 * pad - dilation * (weight->ne[0] - 1) - 1) / stride + 1;
            auto* params = new Conv1dDwParams{stride, pad, dilation};
            struct ggml_tensor* args[2] = {weight, input};
            gt = ggml_custom_4d(ctx, GGML_TYPE_F32,
                L_out, weight->ne[2], input->ne[2], 1,
                args, 2, ggml_custom_conv_1d_dw_f32, GGML_N_TASKS_MAX, params);
            pin_to_cpu(gt);
          } else {
            gt = ggml_conv_1d(ctx, weight, input, stride, pad, dilation);
          }

          // Cast to f32 if output is f16
          if (gt && gt->type == GGML_TYPE_F16) {
            gt = safe_ggml_cast(ctx, gt, GGML_TYPE_F32);
          }

          // Add bias if present
          if (bias && gt) {
            // Conv1d output layout in ggml: [L_out, C_out, batch, 1]
            // Bias is 1D [C_out] → reshape to [1, C_out, 1, 1] for broadcast
            struct ggml_tensor* bias4 = ggml_reshape_4d(ctx, bias, 1, bias->ne[0], 1, 1);
            if (!ggml_can_repeat(bias4, gt)) {
              fprintf(stderr,
                      "[executorch-ggml] CONV_1D bias not repeatable: b=(%lld,%lld,%lld,%lld) y=(%lld,%lld,%lld,%lld)\n",
                      (long long) bias4->ne[0], (long long) bias4->ne[1], (long long) bias4->ne[2], (long long) bias4->ne[3],
                      (long long) gt->ne[0], (long long) gt->ne[1], (long long) gt->ne[2], (long long) gt->ne[3]);
              ggml_free(ctx);
              return Error::InvalidArgument;
            }
            struct ggml_tensor* bias_rep = ggml_repeat(ctx, bias4, gt);
            if (bias_rep->type == GGML_TYPE_F16) {
              bias_rep = safe_ggml_cast(ctx, bias_rep, GGML_TYPE_F32);
            }
            gt = ggml_add(ctx, gt, bias_rep);
          }
          break;
        }

        case ggml_ir::OpCode::PAD: {
          // constant_pad_nd: src_ids=[x], op_params: ndim_pairs, pairs..., fill_value
          // PyTorch pad list is innermost-first; ggml_pad_ext uses ggml axis order.
          // ggml axes: 0=innermost (PyTorch last), 1, 2, 3.
          int32_t ndim_pairs = 0;
          int32_t left[4] = {0, 0, 0, 0};
          int32_t right[4] = {0, 0, 0, 0};

          if (t->op_params() && t->op_params()->size() >= 4) {
            const uint8_t* data = t->op_params()->data();
            size_t off = 0;
            memcpy(&ndim_pairs, data + off, sizeof(int32_t)); off += 4;

            // Pairs are stored innermost-first (matching PyTorch order).
            // ggml axis 0 = innermost = first pair.
            for (int32_t i = 0; i < ndim_pairs && i < 4; i++) {
              memcpy(&left[i], data + off, sizeof(int32_t)); off += 4;
              memcpy(&right[i], data + off, sizeof(int32_t)); off += 4;
            }
          }

          gt = ggml_pad_ext(ctx, ensure_cont(ctx, srcs[0]),
                            left[0], right[0],
                            left[1], right[1],
                            left[2], right[2],
                            left[3], right[3]);
          break;
        }

        default:
          fprintf(stderr, "[executorch-ggml] Unsupported OpCode %d\n", (int) op);
          ggml_free(ctx);
          return Error::InvalidArgument;
      }

    id_to_tensor[tid] = gt;
    last_processed = i;
    if (t->is_output()) {
      int out_idx = t->input_index();
      output_pairs.emplace_back(out_idx >= 0 ? out_idx : (int)output_pairs.size(), gt);
    }
  }
  fprintf(stderr, "[ggml_backend] Phase B complete, last_processed=%d/%d\n", last_processed, n_tensors); fflush(stderr);

  // Sort outputs by output_index (stored in input_index field for outputs)
  // to match the order ExecuTorch expects in the output args span.
  std::sort(output_pairs.begin(), output_pairs.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
  std::vector<struct ggml_tensor*> output_tensors;
  for (auto& [idx, tensor] : output_pairs) {
    // Output tensors must be contiguous for backend_tensor_get in execute().
    if (!ggml_is_contiguous(tensor)) {
      tensor = ggml_cont(ctx, tensor);
    }
    output_tensors.push_back(tensor);
  }

  // === Build compute graph ===
  // Build compute graph using the same estimated size from above.
  struct ggml_cgraph* graph = ggml_new_graph_custom(ctx, est_graph_size, false);
  for (auto* out : output_tensors) {
    ggml_build_forward_expand(graph, out);
  }
  fprintf(stderr, "[ggml_backend] graph built: %d nodes, %d leafs (max %zu)\n",
          ggml_graph_n_nodes(graph), (int)0, est_graph_size);
  // Sort inputs by input_index
  std::sort(input_pairs.begin(), input_pairs.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });

  // Replace I64 inputs with their I32 deferred-cast destinations.
  // The I64 tensor is NOT part of the ggml compute graph (the cast is
  // deferred to the host), so gallocr won't allocate it.  By swapping
  // the I32 destination into the input list, execute() can copy ET int64
  // data directly into the I32 tensor via the existing I32+Long path.
  for (auto& [idx, tensor] : input_pairs) {
    for (auto& [src_i64, dst_i32] : deferred_i64_to_i32) {
      if (tensor == src_i64) {
        tensor = dst_i32;
        break;
      }
    }
  }
  deferred_i64_to_i32.clear();  // no longer needed

  // Collect ordered input tensor list.
  std::vector<struct ggml_tensor*> ordered_inputs;
  for (auto& [idx, tensor] : input_pairs) {
    ordered_inputs.push_back(tensor);
  }

  // === Phase C: allocate graph on backend ===
  // -------------------------------------------------------------------
  // Allocate the graph on a compute backend (CUDA if available, else CPU)
  // -------------------------------------------------------------------

  // 1. Mark input / output tensors so the graph allocator keeps them live.
  std::unordered_set<struct ggml_tensor*> input_set(
      ordered_inputs.begin(), ordered_inputs.end());
  std::unordered_set<struct ggml_tensor*> deferred_dst_set;
  for (auto& [src, dst] : deferred_i64_to_i32) {
    deferred_dst_set.insert(dst);
  }

  // 2. Save constant data for restore after sched_reset.
  handle->saved_constants.clear();
  for (struct ggml_tensor* gt = ggml_get_first_tensor(ctx);
       gt != nullptr;
       gt = ggml_get_next_tensor(ctx, gt)) {
    if (!gt->data) continue;
    if (input_set.count(gt)) continue;
    if (deferred_dst_set.count(gt)) continue;
    if (gt->op != GGML_OP_NONE) continue;
    ggml_set_input(gt);  // prevent gallocr from reusing this memory
    size_t nbytes = ggml_nbytes(gt);
    handle->saved_constants.push_back({gt, std::vector<uint8_t>(
        static_cast<uint8_t*>(gt->data),
        static_cast<uint8_t*>(gt->data) + nbytes)});
  }

  // 2b. Clear all tensor data/buffer pointers for scheduler allocation.
  for (struct ggml_tensor* t = ggml_get_first_tensor(ctx);
       t != nullptr;
       t = ggml_get_next_tensor(ctx, t)) {
    t->data   = nullptr;
    t->buffer = nullptr;
  }

  // 3. Mark inputs/outputs for the scheduler.
  for (auto* inp : ordered_inputs) {
    ggml_set_input(inp);
  }
  for (auto* dst : deferred_dst_set) {
    ggml_set_input(dst);
  }
  for (auto* out : output_tensors) {
    ggml_set_output(out);
  }

  // 4. Allocate graph tensors using scheduler.
  if (!handle->sched) {
    ggml_free(ctx);
    return Error::InvalidState;
  }
  ggml_backend_sched_reset(handle->sched);
  if (!ggml_backend_sched_alloc_graph(handle->sched, graph)) {
    fprintf(stderr, "[ggml_backend] ERROR: scheduler graph allocation failed\n");
    ggml_free(ctx);
    return Error::MemoryAllocationFailed;
  }

  // 5. Restore constant data into backend buffers.
  for (auto& sc : handle->saved_constants) {
    if (sc.tensor->buffer == nullptr) continue;
    ggml_backend_tensor_set(sc.tensor, sc.data.data(), 0, sc.data.size());
  }

  // 5b. Reset constant integrity tracking for the new graph.
  handle->execute_count = 0;
  handle->const_needs_restore.clear();

  // 6. Free the temporary CPU host buffer.
  if (host_buf) {
    ggml_backend_buffer_free(host_buf);
    host_buf = nullptr;
  }

  // === Update handle ===
  handle->ctx = ctx;
  handle->graph = graph;
  handle->inputs = std::move(ordered_inputs);
  handle->outputs = std::move(output_tensors);
  handle->deferred_i64_to_i32 = std::move(deferred_i64_to_i32);

  return Error::Ok;
}

// ---------------------------------------------------------------------------
// BackendInterface implementation
// ---------------------------------------------------------------------------

bool GgmlBackendInterface::is_available() const {
  return true;
}

Result<DelegateHandle*> GgmlBackendInterface::init(
    BackendInitContext& context,
    FreeableBuffer* processed,
    ArrayRef<CompileSpec> compile_specs) const {

  // --- 1. Copy IR buffer for later rebuilds ---
  auto* handle = new GgmlDelegateHandle();
  handle->ir_copy.assign(
      static_cast<const uint8_t*>(processed->data()),
      static_cast<const uint8_t*>(processed->data()) + processed->size());

  // --- 2. Parse FlatBuffer to read metadata ---
  const auto* fb_graph = ggml_ir::GetGgmlGraph(handle->ir_copy.data());
  if (!fb_graph || !fb_graph->tensors()) {
    delete handle;
    return Error::InvalidArgument;
  }
  const auto* fb_tensors = fb_graph->tensors();
  const int n_tensors = static_cast<int>(fb_tensors->size());
  handle->n_threads = fb_graph->n_threads();

  // --- 3. Create backends + scheduler (one-time) ---
  int cpu_threads = handle->n_threads;
  const char* threads_env = std::getenv("GGML_CPU_THREADS");
  if (threads_env) {
    cpu_threads = std::atoi(threads_env);
  } else if (cpu_threads <= 1) {
    cpu_threads = static_cast<int>(std::thread::hardware_concurrency());
    if (cpu_threads <= 0) cpu_threads = 4;
  }

  const char* device_env = std::getenv("GGML_BACKEND_DEVICE");
  bool force_cpu = device_env && std::string(device_env) == "cpu";
  if (!force_cpu) {
    ggml_backend_dev_t gpu_dev =
        ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (gpu_dev) {
      handle->backend = ggml_backend_dev_init(gpu_dev, nullptr);
      if (handle->backend) {
        fprintf(stderr, "[ggml_backend] Using GPU backend: %s\n",
                ggml_backend_name(handle->backend));
      }
    }
  }

  // Always have a CPU backend for scheduler fallback / custom ops.
  if (handle->backend) {
    handle->backend_cpu = ggml_backend_cpu_init();
    if (!handle->backend_cpu) {
      ggml_backend_free(handle->backend);
      delete handle;
      return Error::MemoryAllocationFailed;
    }
    ggml_backend_cpu_set_n_threads(handle->backend_cpu, cpu_threads);
  } else {
    handle->backend = ggml_backend_cpu_init();
    if (!handle->backend) {
      delete handle;
      return Error::MemoryAllocationFailed;
    }
    handle->backend_cpu = handle->backend;
    ggml_backend_cpu_set_n_threads(handle->backend_cpu, cpu_threads);
    fprintf(stderr, "[ggml_backend] Using CPU backend (%d threads)\n",
            cpu_threads);
  }

  // Build scheduler backend list (GPU+CPU fallback, or CPU-only).
  std::vector<ggml_backend_t> sched_backends;
  sched_backends.push_back(handle->backend);
  if (handle->backend_cpu && handle->backend_cpu != handle->backend) {
    sched_backends.push_back(handle->backend_cpu);
  }
  const size_t max_nodes = std::max<size_t>(GGML_DEFAULT_GRAPH_SIZE, static_cast<size_t>(n_tensors) * 4);
  handle->sched = ggml_backend_sched_new(
      sched_backends.data(),
      /*bufts=*/nullptr,
      static_cast<int>(sched_backends.size()),
      max_nodes,
      /*parallel=*/false,
      /*op_offload=*/true);
  if (!handle->sched) {
    if (handle->backend_cpu && handle->backend_cpu != handle->backend) {
      ggml_backend_free(handle->backend_cpu);
    }
    if (handle->backend) {
      ggml_backend_free(handle->backend);
    }
    delete handle;
    return Error::MemoryAllocationFailed;
  }
  auto cleanup_backends = [&]() {
    if (handle->sched) {
      ggml_backend_sched_free(handle->sched);
      handle->sched = nullptr;
    }
    if (handle->backend_cpu && handle->backend_cpu != handle->backend) {
      ggml_backend_free(handle->backend_cpu);
      handle->backend_cpu = nullptr;
    }
    if (handle->backend) {
      ggml_backend_free(handle->backend);
      handle->backend = nullptr;
    }
  };

  // --- 4. Load ALL constants from NamedDataMap → handle->constant_data ---
  // This is the ONLY place NamedDataMap is touched.
  // After this, build_graph() reads from handle->constant_data.
  for (int i = 0; i < n_tensors; ++i) {
    const auto* t = fb_tensors->Get(i);
    if (static_cast<ggml_ir::OpCode>(t->op()) != ggml_ir::OpCode::NONE) continue;
    if (!t->data_key() || std::strlen(t->data_key()->c_str()) == 0) continue;

    const auto* ndm = context.get_named_data_map();
    if (ndm == nullptr) {
      cleanup_backends();
      delete handle;
      return Error::InvalidArgument;
    }

    const char* key = t->data_key()->c_str();

    // Compute byte size from IR shape + type.
    // Use ggml_row_size() to correctly handle block-based quantized types
    // (e.g. Q8_0: 34 bytes per block of 32 elements).
    int64_t ne[4] = {1, 1, 1, 1};
    if (t->ne()) {
      for (size_t d = 0; d < t->ne()->size() && d < 4; ++d) {
        ne[d] = t->ne()->Get(d);
      }
    }
    ggml_type load_gtype;
    switch (static_cast<ggml_ir::TensorType>(t->type())) {
      case ggml_ir::TensorType::F16:  load_gtype = GGML_TYPE_F16;  break;
      case ggml_ir::TensorType::BF16: load_gtype = GGML_TYPE_BF16; break;
      case ggml_ir::TensorType::I32:  load_gtype = GGML_TYPE_I32;  break;
      case ggml_ir::TensorType::I64:  load_gtype = GGML_TYPE_I64;  break;
      case ggml_ir::TensorType::BOOL: load_gtype = GGML_TYPE_I32;  break;
      case ggml_ir::TensorType::Q8_0: load_gtype = GGML_TYPE_Q8_0; break;
      case ggml_ir::TensorType::Q6_K: load_gtype = GGML_TYPE_Q6_K; break;
      case ggml_ir::TensorType::Q4_0: load_gtype = GGML_TYPE_Q4_0; break;
      default:                        load_gtype = GGML_TYPE_F32;   break;
    }
    size_t nbytes = ggml_row_size(load_gtype, ne[0]) * ne[1] * ne[2] * ne[3];

    auto fb = ndm->get_data(key);
    if (!fb.ok()) {
      cleanup_backends();
      delete handle;
      return fb.error();
    }
    auto buf = std::move(fb.get());
    if (buf.size() < nbytes) {
      buf.Free();
      cleanup_backends();
      delete handle;
      return Error::InvalidExternalData;
    }

    SavedConstant sc;
    sc.ir_tensor_id = t->id();
    sc.data.assign(
        static_cast<const uint8_t*>(buf.data()),
        static_cast<const uint8_t*>(buf.data()) + nbytes);
    buf.Free();
    handle->constant_data.push_back(std::move(sc));
  }

  // --- 5. Read sym_dim_ids from IR ---
  // Scan ALL tensors (not just inputs) for sym_dim_ids to detect dynamic models.
  for (int i = 0; i < n_tensors; ++i) {
    const auto* t = fb_tensors->Get(i);
    if (t->sym_dim_ids()) {
      for (size_t d = 0; d < t->sym_dim_ids()->size() && d < 4; ++d) {
        int32_t sid = t->sym_dim_ids()->Get(d);
        if (sid >= 0 || sid == -2) {
          handle->has_dynamic = true;
          break;
        }
      }
    }
    if (!t->is_input()) continue;

    std::vector<int32_t> sids(4, -1);
    if (t->sym_dim_ids()) {
      for (size_t d = 0; d < t->sym_dim_ids()->size() && d < 4; ++d) {
        sids[d] = t->sym_dim_ids()->Get(d);
      }
    }
    // Store in input_index order (may be sparse; resize as needed).
    int idx = t->input_index();
    if (idx >= 0 && static_cast<size_t>(idx) >= handle->input_sym_dim_ids.size()) {
      handle->input_sym_dim_ids.resize(idx + 1);
    }
    if (idx >= 0) {
      handle->input_sym_dim_ids[idx] = std::move(sids);
    }
  }

  // --- 6. Build initial graph with serialized shapes ---
  Error err = build_graph(handle, nullptr, 0);
  if (err != Error::Ok) {
    cleanup_backends();
    delete handle;
    return err;
  }

  // --- 7. Initialize last_input_ne from the initial graph's input shapes ---
  if (handle->has_dynamic) {
    size_t n_inp = handle->inputs.size();
    handle->last_input_ne.resize(n_inp * 4, 1);
    for (size_t i = 0; i < n_inp; ++i) {
      for (int d = 0; d < 4; ++d) {
        handle->last_input_ne[i * 4 + d] = handle->inputs[i]->ne[d];
      }
    }
  }

  return handle;
}

Error GgmlBackendInterface::execute(
    BackendExecutionContext& context,
    DelegateHandle* handle_raw,
    Span<EValue*> args) const {

  auto* handle = reinterpret_cast<GgmlDelegateHandle*>(handle_raw);

  size_t n_inputs = handle->inputs.size();
  size_t n_outputs = handle->outputs.size();

  // --- Build mapping from ggml input index → args index ---
  // With dynamic shapes, ExecuTorch passes non-tensor args (sym_size ints)
  // alongside tensor inputs.  The args layout is:
  //   [input_0, input_1, ..., (possibly sym_size ints), output_0, output_1, ...]
  // Outputs are always the last n_outputs entries.
  //
  // Some IR "inputs" may correspond to sym_size scalars that were serialized
  // as placeholder tensors but don't have matching tensor args at runtime.
  // We build a 1:1 mapping: ggml_input_i → args[i], but only for tensor args.
  // Non-tensor args (sym_size) and IR inputs without matching tensor args
  // are skipped during data copy.
  size_t n_non_output_args = args.size() >= n_outputs ? args.size() - n_outputs : 0;

  // --- Always rebuild graph for dynamic models ---
  // The graph must be rebuilt on every step because eager ops (comparison,
  // bitwise, etc.) compute data at build time. Their results depend on
  // input DATA (e.g. cache_position), not just input shapes. Without
  // rebuilding, these ops produce stale values from the previous step.
  if (handle->has_dynamic) {
    // Build current input shapes from ET tensors (ggml ne order).
    std::vector<int64_t> current_ne(n_inputs * 4, 1);
    // Also collect input data pointers for pre-population.
    std::vector<InputDataOverride> input_data_vec(n_inputs);
    size_t ggml_idx = 0;
    for (size_t a = 0; a < n_non_output_args && ggml_idx < n_inputs; ++a) {
      if (!args[a]->isTensor()) continue;
      const auto& et = args[a]->toTensor();
      int ndim = et.dim();
      for (int d = 0; d < ndim && d < 4; ++d) {
        current_ne[ggml_idx * 4 + d] = et.size(ndim - 1 - d);
      }
      input_data_vec[ggml_idx] = {
        et.const_data_ptr(),
        static_cast<size_t>(et.nbytes()),
        static_cast<int>(et.scalar_type())
      };
      ++ggml_idx;
    }

    Error err = build_graph(handle, current_ne.data(), current_ne.size(), &input_data_vec);
    if (err != Error::Ok) return err;
    handle->last_input_ne = current_ne;
    n_inputs = handle->inputs.size();
  }

  // Copy input data from ExecuTorch tensors → backend tensors.
  // Walk non-output args in order; each tensor arg maps to the next ggml input.
  // Non-tensor args (sym_size ints from dynamic shapes) are skipped.
  // IR inputs that don't have a matching tensor arg (e.g. sym_size placeholders
  // that were serialized as dummy [1,1,1,1] tensors) are also skipped.
  {
  size_t ggml_idx = 0;
  for (size_t a = 0; a < n_non_output_args && ggml_idx < n_inputs; ++a) {
    if (!args[a]->isTensor()) continue;

    struct ggml_tensor* gt = handle->inputs[ggml_idx];
    ++ggml_idx;

    // Skip inputs that aren't part of the ggml graph (gallocr didn't
    // allocate them, e.g. unused position/start_pos placeholders).
    if (gt->buffer == nullptr) continue;

    const auto& et_tensor = args[a]->toTensor();

    const size_t nelem = ggml_nelements(gt);

    if (gt->type == GGML_TYPE_F16 && et_tensor.scalar_type() == executorch::aten::ScalarType::Float) {
      // Convert fp32 → fp16 on host, then copy to backend.
      std::vector<ggml_fp16_t> tmp(nelem);
      const float* src = static_cast<const float*>(et_tensor.const_data_ptr());
      ggml_fp32_to_fp16_row(src, tmp.data(), (int64_t) nelem);
      ggml_backend_tensor_set(gt, tmp.data(), 0, nelem * sizeof(ggml_fp16_t));
    } else if (gt->type == GGML_TYPE_BF16 && et_tensor.scalar_type() == executorch::aten::ScalarType::Float) {
      // Convert fp32 → bf16 on host, then copy to backend.
      std::vector<ggml_bf16_t> tmp(nelem);
      const float* src = static_cast<const float*>(et_tensor.const_data_ptr());
      for (size_t j = 0; j < nelem; ++j) {
        tmp[j] = ggml_fp32_to_bf16(src[j]);
      }
      ggml_backend_tensor_set(gt, tmp.data(), 0, nelem * sizeof(ggml_bf16_t));
    } else if (gt->type == GGML_TYPE_BF16 && et_tensor.scalar_type() == executorch::aten::ScalarType::BFloat16) {
      // BF16 → BF16 direct copy.
      ggml_backend_tensor_set(gt, et_tensor.const_data_ptr(), 0, ggml_nbytes(gt));
    } else if (gt->type == GGML_TYPE_I32 && et_tensor.scalar_type() == executorch::aten::ScalarType::Bool) {
      std::vector<int32_t> tmp(nelem);
      const bool* src = static_cast<const bool*>(et_tensor.const_data_ptr());
      for (size_t j = 0; j < nelem; ++j) {
        tmp[j] = src[j] ? 1 : 0;
      }
      ggml_backend_tensor_set(gt, tmp.data(), 0, nelem * sizeof(int32_t));
    } else if (gt->type == GGML_TYPE_I32 && et_tensor.scalar_type() == executorch::aten::ScalarType::Long) {
      std::vector<int32_t> tmp(nelem);
      const int64_t* src = static_cast<const int64_t*>(et_tensor.const_data_ptr());
      for (size_t j = 0; j < nelem; ++j) {
        tmp[j] = (int32_t) src[j];
      }
      ggml_backend_tensor_set(gt, tmp.data(), 0, nelem * sizeof(int32_t));
    } else {
      ggml_backend_tensor_set(gt, et_tensor.const_data_ptr(), 0, ggml_nbytes(gt));
    }
  }
  }  // end input copy block

  // Run deferred I64→I32 casts (now that input data is on the backend).
  for (const auto& [src_i64, dst_i32] : handle->deferred_i64_to_i32) {
    if (src_i64->buffer == nullptr || dst_i32->buffer == nullptr) continue;
    const size_t nelem = ggml_nelements(src_i64);
    std::vector<int64_t> src_buf(nelem);
    ggml_backend_tensor_get(src_i64, src_buf.data(), 0, nelem * sizeof(int64_t));
    std::vector<int32_t> dst_buf(nelem);
    for (size_t j = 0; j < nelem; ++j) {
      dst_buf[j] = static_cast<int32_t>(src_buf[j]);
    }
    ggml_backend_tensor_set(dst_i32, dst_buf.data(), 0, nelem * sizeof(int32_t));
  }

  // Execute the graph via scheduler.
  // Reset + re-alloc before every compute to clear stale cross-backend state.
  // Then check whether constants survived via a signature probe (first call
  // only). If they survive, skip the expensive bulk re-copy on all future calls.
  if (!handle->sched) {
    return Error::InvalidState;
  }
  ggml_backend_sched_reset(handle->sched);
  if (!ggml_backend_sched_alloc_graph(handle->sched, handle->graph)) {
    fprintf(stderr, "[ggml_backend] ERROR: scheduler re-alloc failed\n");
    return Error::MemoryAllocationFailed;
  }
  // Restore constants if needed.
  // execute_count 0: first call — always restore (build_graph loaded them,
  //   but sched_reset+alloc just ran and may have invalidated them).
  // execute_count 1: second call — probe whether constants survived the
  //   previous compute + this reset+alloc cycle. Cache the result.
  // execute_count 2+: use cached result. Skip restore if constants survive.
  {
    if (handle->execute_count == 0) {
      // First call: always restore all constants.
      for (const auto& sc : handle->saved_constants) {
        if (sc.tensor->buffer == nullptr) continue;
        ggml_backend_tensor_set(sc.tensor, sc.data.data(), 0, sc.data.size());
      }
    } else if (handle->execute_count == 1) {
      // Second call: probe each constant to find which were corrupted by
      // the first compute. Build a per-constant mask for future calls.
      handle->const_needs_restore.resize(handle->saved_constants.size(), false);
      for (size_t i = 0; i < handle->saved_constants.size(); i++) {
        auto& sc = handle->saved_constants[i];
        if (sc.tensor->buffer == nullptr) continue;
        size_t n = std::min<size_t>(GgmlDelegateHandle::CONST_SIG_SIZE, sc.data.size());
        uint8_t sample[GgmlDelegateHandle::CONST_SIG_SIZE] = {};
        ggml_backend_tensor_get(sc.tensor, sample, 0, n);
        if (fnv1a_64(sample, n) != fnv1a_64(sc.data.data(), n)) {
          handle->const_needs_restore[i] = true;
        }
      }
      // Restore only corrupted constants.
      for (size_t i = 0; i < handle->saved_constants.size(); i++) {
        if (!handle->const_needs_restore[i]) continue;
        auto& sc = handle->saved_constants[i];
        if (sc.tensor->buffer == nullptr) continue;
        ggml_backend_tensor_set(sc.tensor, sc.data.data(), 0, sc.data.size());
      }
    } else {
      // Steady state: restore only the constants known to be corrupted.
      for (size_t i = 0; i < handle->saved_constants.size(); i++) {
        if (i >= handle->const_needs_restore.size() || !handle->const_needs_restore[i]) continue;
        auto& sc = handle->saved_constants[i];
        if (sc.tensor->buffer == nullptr) continue;
        ggml_backend_tensor_set(sc.tensor, sc.data.data(), 0, sc.data.size());
      }
    }
    handle->execute_count = std::min(handle->execute_count + 1, 2);
  }
  // Per-op profiling (GGML_PROFILE=1)
  static int profile_mode = -1;
  if (profile_mode < 0) {
    const char* env = std::getenv("GGML_PROFILE");
    profile_mode = (env && std::string(env) != "0") ? 1 : 0;
  }
  ProfileContext prof_ctx;
  if (profile_mode) {
    ggml_backend_sched_set_eval_callback(handle->sched, profile_eval_callback, &prof_ctx);
  }

  enum ggml_status status = ggml_backend_sched_graph_compute(handle->sched, handle->graph);

  if (profile_mode) {
    ggml_backend_sched_set_eval_callback(handle->sched, nullptr, nullptr);
    print_profile(prof_ctx);
  }

  if (status != GGML_STATUS_SUCCESS) {
    fprintf(stderr, "[ggml_backend] ERROR: scheduler graph compute failed: %s (%d)\n",
            ggml_status_to_string(status), (int)status);
    return Error::InvalidState;
  }

  // Copy output data from backend tensors → ExecuTorch output tensors
  // Outputs are always the last n_outputs entries in the args span.
  for (size_t i = 0; i < n_outputs; ++i) {
    size_t out_idx = args.size() - n_outputs + i;
    if (out_idx >= (size_t)args.size() || !args[out_idx]->isTensor()) {
      fprintf(stderr, "[ggml_backend] ERROR: output args[%zu] is not a Tensor "
              "(n_outputs=%zu, args.size=%zu)\n",
              out_idx, n_outputs, (size_t)args.size());
      return Error::InvalidArgument;
    }
    auto& et_tensor = args[out_idx]->toTensor();
    struct ggml_tensor* gt = handle->outputs[i];

    // Resize ET output tensor to match actual ggml output shape (see below).

    // Resize ET output tensor to match actual ggml output shape.
    // ExecuTorch may have allocated it at the upper-bound shape for dynamic
    // dims.  Convert ggml ne order (innermost first) to PyTorch order
    // (outermost first) and resize.
    {
      int et_ndim = et_tensor.dim();
      // Build ggml shape in PyTorch order: reverse the ne array, trim to ndim.
      // ggml ne = [ne0, ne1, ne2, ne3], PyTorch = [ne_{ndim-1}, ..., ne0]
      std::vector<executorch::aten::SizesType> new_sizes(et_ndim);
      for (int d = 0; d < et_ndim; ++d) {
        new_sizes[d] = static_cast<executorch::aten::SizesType>(
            gt->ne[et_ndim - 1 - d]);
      }
      Error resize_err = executorch::ET_RUNTIME_NAMESPACE::resize_tensor(
          et_tensor, {new_sizes.data(), new_sizes.size()});
      if (resize_err != Error::Ok) {
        fprintf(stderr, "[ggml_backend] WARNING: resize_tensor failed for "
                "output %zu\n", i);
      }
    }

    const size_t nelem = ggml_nelements(gt);

    if (gt->type == GGML_TYPE_F16 && et_tensor.scalar_type() == executorch::aten::ScalarType::Float) {
      // Read fp16 from backend, convert to fp32.
      std::vector<ggml_fp16_t> tmp(nelem);
      ggml_backend_tensor_get(gt, tmp.data(), 0, nelem * sizeof(ggml_fp16_t));
      float* dst = static_cast<float*>(et_tensor.mutable_data_ptr());
      ggml_fp16_to_fp32_row(tmp.data(), dst, (int64_t) nelem);
    } else if (gt->type == GGML_TYPE_F32 && et_tensor.scalar_type() == executorch::aten::ScalarType::BFloat16) {
      // ggml output is F32 (e.g. ggml_set_rows requires F32) but ET expects BF16.
      std::vector<float> tmp(nelem);
      ggml_backend_tensor_get(gt, tmp.data(), 0, nelem * sizeof(float));
      ggml_bf16_t* dst = static_cast<ggml_bf16_t*>(et_tensor.mutable_data_ptr());
      for (size_t j = 0; j < nelem; ++j) {
        dst[j] = ggml_fp32_to_bf16(tmp[j]);
      }
    } else if (gt->type == GGML_TYPE_F32 && et_tensor.scalar_type() == executorch::aten::ScalarType::Float) {
      ggml_backend_tensor_get(gt, et_tensor.mutable_data_ptr(), 0, ggml_nbytes(gt));
    } else if (gt->type == GGML_TYPE_BF16 && et_tensor.scalar_type() == executorch::aten::ScalarType::BFloat16) {
      ggml_backend_tensor_get(gt, et_tensor.mutable_data_ptr(), 0, ggml_nbytes(gt));
    } else if (gt->type == GGML_TYPE_I32 && et_tensor.scalar_type() == executorch::aten::ScalarType::Long) {
      // ggml I32 → ET int64: widen each element.
      std::vector<int32_t> tmp(nelem);
      ggml_backend_tensor_get(gt, tmp.data(), 0, nelem * sizeof(int32_t));
      int64_t* dst = static_cast<int64_t*>(et_tensor.mutable_data_ptr());
      for (size_t j = 0; j < nelem; ++j) {
        dst[j] = static_cast<int64_t>(tmp[j]);
      }
    } else {
      // Fallback: copy min of both sizes to avoid buffer overflow.
      size_t copy_size = std::min(ggml_nbytes(gt), (size_t)et_tensor.nbytes());
      ggml_backend_tensor_get(gt, et_tensor.mutable_data_ptr(), 0, copy_size);
    }
  }

  return Error::Ok;
}

void GgmlBackendInterface::destroy(DelegateHandle* handle_raw) const {
  auto* handle = reinterpret_cast<GgmlDelegateHandle*>(handle_raw);
  if (handle) {
    if (handle->sched) {
      ggml_backend_sched_free(handle->sched);
    }
    if (handle->backend_cpu && handle->backend_cpu != handle->backend) {
      ggml_backend_free(handle->backend_cpu);
    }
    if (handle->backend) {
      ggml_backend_free(handle->backend);
    }
    if (handle->ctx) {
      ggml_free(handle->ctx);
    }
    delete handle;
  }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

namespace {
auto cls = executorch::runtime::register_backend(
    {"GgmlBackend", new GgmlBackendInterface()});
}  // namespace

}  // namespace executorch_ggml
