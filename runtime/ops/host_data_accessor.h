#pragma once

#include <cstdint>
#include <deque>
#include <unordered_set>
#include <vector>

#include <ggml.h>
#include <ggml-backend.h>

namespace executorch_ggml {

// ---------------------------------------------------------------------------
// HostDataAccessor -- safe host-side access to tensor data that may live on GPU.
//
// When const_buf is on a non-host backend (e.g. CUDA), tensor->data is a
// device pointer that cannot be dereferenced from the host.  This helper
// transparently copies the data to a host-side staging buffer when needed,
// while returning the original pointer directly for host-resident tensors.
// ---------------------------------------------------------------------------
class HostDataAccessor {
public:
  const void* get(const struct ggml_tensor* t) {
    if (!t || !t->data) return nullptr;
    if (!t->buffer || ggml_backend_buffer_is_host(t->buffer)) {
      return t->data;
    }
    // Each call gets its own staging buffer so multiple pointers remain valid.
    size_t nb = ggml_nbytes(t);
    staging_.emplace_back(nb);
    ggml_backend_tensor_get(t, staging_.back().data(), 0, nb);
    return staging_.back().data();
  }

  float read_f32(const struct ggml_tensor* t) {
    const void* p = get(t);
    return p ? *static_cast<const float*>(p) : 0.0f;
  }
  int32_t read_i32(const struct ggml_tensor* t) {
    const void* p = get(t);
    return p ? *static_cast<const int32_t*>(p) : 0;
  }
  int64_t read_i64(const struct ggml_tensor* t) {
    const void* p = get(t);
    return p ? *static_cast<const int64_t*>(p) : 0;
  }

  // Track tensors derived from graph inputs (set during build_graph).
  // Used to avoid baking in input-dependent values as eager constants.
  std::unordered_set<struct ggml_tensor*>* input_derived = nullptr;

  bool is_input_derived(const struct ggml_tensor* t) const {
    return input_derived && input_derived->count(const_cast<struct ggml_tensor*>(t));
  }
  void propagate_derived(struct ggml_tensor* dst, const struct ggml_tensor* src) {
    if (is_input_derived(src) && input_derived) input_derived->insert(dst);
  }

private:
  std::deque<std::vector<uint8_t>> staging_;
};

// Eager integer casts on the CPU context.  ggml's CPU backend doesn't support
// GGML_OP_CPY for I64 or I32<->I64, so we must perform these conversions at
// graph-build time (data is available because we use no_alloc=false).
// Returns a new tensor with op=GGML_OP_NONE (treated as a constant leaf).
static struct ggml_tensor* eager_cast_i64_to_i32(
    struct ggml_context* ctx,
    struct ggml_tensor* src,
    HostDataAccessor* acc = nullptr) {
  ggml_set_no_alloc(ctx, false);
  struct ggml_tensor* dst = ggml_new_tensor(ctx, GGML_TYPE_I32, GGML_MAX_DIMS, src->ne);
  ggml_set_no_alloc(ctx, true);
  dst->op = GGML_OP_NONE;
  if (acc) acc->propagate_derived(dst, src);
  if (src->data && dst->data) {
    const size_t n = ggml_nelements(src);
    const int64_t* s = acc ? static_cast<const int64_t*>(acc->get(src))
                           : static_cast<const int64_t*>(src->data);
    int32_t* d = static_cast<int32_t*>(dst->data);
    for (size_t i = 0; i < n; ++i) d[i] = static_cast<int32_t>(s[i]);
  }
  return dst;
}

static struct ggml_tensor* eager_cast_i32_to_i64(
    struct ggml_context* ctx,
    struct ggml_tensor* src,
    HostDataAccessor* acc = nullptr) {
  ggml_set_no_alloc(ctx, false);
  struct ggml_tensor* dst = ggml_new_tensor(ctx, GGML_TYPE_I64, GGML_MAX_DIMS, src->ne);
  ggml_set_no_alloc(ctx, true);
  dst->op = GGML_OP_NONE;
  if (acc) acc->propagate_derived(dst, src);
  if (src->data && dst->data) {
    const size_t n = ggml_nelements(src);
    const int32_t* s = acc ? static_cast<const int32_t*>(acc->get(src))
                           : static_cast<const int32_t*>(src->data);
    int64_t* d = static_cast<int64_t*>(dst->data);
    for (size_t i = 0; i < n; ++i) d[i] = static_cast<int64_t>(s[i]);
  }
  return dst;
}

// Safe cast that avoids ggml_cast combos the CPU backend can't handle.
// Supported natively: F16<->{F16,BF16,F32}, BF16<->{F16,BF16,F32}, F32<->{F16,BF16,F32,I32}, I32->F32.
// Everything else routes through F32 as intermediate or uses eager helpers.
static struct ggml_tensor* safe_ggml_cast(
    struct ggml_context* ctx,
    struct ggml_tensor* src,
    ggml_type target,
    HostDataAccessor* acc = nullptr) {
  if (src->type == target) return src;
  // I64 source: eager CPU conversion
  if (src->type == GGML_TYPE_I64 && target == GGML_TYPE_I32) {
    auto* r = eager_cast_i64_to_i32(ctx, src, acc);
    return r;
  }
  // I64 -> F32: eager host conversion (avoids CUDA buffer aliasing in CPY)
  if (src->type == GGML_TYPE_I64 && target == GGML_TYPE_F32) {
    const void* src_data = acc ? acc->get(src) : src->data;
    if (src_data) {
      int64_t n = ggml_nelements(src);
      ggml_set_no_alloc(ctx, false);
      struct ggml_tensor* f32 = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, src->ne);
      ggml_set_no_alloc(ctx, true);
      f32->op = GGML_OP_NONE;
      if (acc) acc->propagate_derived(f32, src);
      const int64_t* sd = static_cast<const int64_t*>(src_data);
      float* dd = static_cast<float*>(f32->data);
      for (int64_t i = 0; i < n; i++) dd[i] = (float)sd[i];
      return f32;
    }
  }
  if (src->type == GGML_TYPE_I64) {
    // I64 -> I32 eager, then I32 -> target via native ggml_cast
    auto* i32 = eager_cast_i64_to_i32(ctx, src, acc);
    return (target == GGML_TYPE_F32) ? safe_ggml_cast(ctx, i32, GGML_TYPE_F32, acc)
                                     : safe_ggml_cast(ctx, safe_ggml_cast(ctx, i32, GGML_TYPE_F32, acc), target, acc);
  }
  // I32 source: only I32->F32 is native in ggml_cast
  if (src->type == GGML_TYPE_I32 && target == GGML_TYPE_I64) return eager_cast_i32_to_i64(ctx, src, acc);
  if (src->type == GGML_TYPE_I32 && target != GGML_TYPE_F32) {
    return safe_ggml_cast(ctx, safe_ggml_cast(ctx, src, GGML_TYPE_F32, acc), target, acc);
  }
  // F32/F16/BF16 -> I64: go through I32 first, then eager I32->I64
  if (target == GGML_TYPE_I64) {
    auto* i32 = safe_ggml_cast(ctx, src, GGML_TYPE_I32, acc);
    if (i32->data) {
      return eager_cast_i32_to_i64(ctx, i32, acc);
    }
    // I32 is a graph node (no data yet) -- return I32 since ggml can't do
    // I32->I64 as a graph op.  The output copy in execute() handles I32->Long.
    return i32;
  }
  // Eager I32->F32 cast (avoids CUDA buffer aliasing in CPY).
  // Skip if the source is input-derived -- must stay as a graph op for reuse.
  if (src->type == GGML_TYPE_I32 && target == GGML_TYPE_F32 &&
      src->data && !(acc && acc->is_input_derived(src))) {
    const void* src_data = acc ? acc->get(src) : src->data;
    if (src_data) {
      int64_t n = ggml_nelements(src);
      ggml_set_no_alloc(ctx, false);
      struct ggml_tensor* dst = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, src->ne);
      ggml_set_no_alloc(ctx, true);
      dst->op = GGML_OP_NONE;
      if (acc) acc->propagate_derived(dst, src);
      const int32_t* sd = static_cast<const int32_t*>(src_data);
      float* dd = static_cast<float*>(dst->data);
      for (int64_t i = 0; i < n; i++) dd[i] = static_cast<float>(sd[i]);
      return dst;
    }
  }
  // Everything else (F16/BF16/F32 inter-conversions) is natively supported
  return ggml_cast(ctx, src, target);
}

} // namespace executorch_ggml
