#pragma once

#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <ggml.h>
#include <ggml-backend.h>

#include "ggml_ir_generated.h"  // flatc-generated IR types (Tensor, TensorType, OpCode, etc.)

#include "data_structures.h"
#include "helpers.h"
#include "host_data_accessor.h"

namespace executorch_ggml {

// ---------------------------------------------------------------------------
// BuildContext -- captures all state that op handlers need during build_graph.
// Passed by reference to every build_op_XXX function so handlers remain
// stateless and easily testable.
// ---------------------------------------------------------------------------
struct BuildContext {
  ggml_context* ctx;
  const ggml_ir::Tensor* ir_tensor;  // current IR tensor being processed
  std::vector<struct ggml_tensor*>& srcs;  // resolved source tensors
  HostDataAccessor& host_acc;
  std::unordered_set<struct ggml_tensor*>& input_derived;
  GgmlDelegateHandle* handle;
  GraphInstance* gi;
  std::vector<struct ggml_tensor*>& cpu_pinned;
  std::vector<struct ggml_tensor*>& id_to_tensor;
  std::unordered_map<int32_t, int64_t>& sym_dim_values;
  std::unordered_map<uint64_t, struct ggml_tensor*>& causal_mask_cache;
  std::vector<std::pair<struct ggml_tensor*, struct ggml_tensor*>>& deferred_i64_to_i32;
  std::vector<std::pair<int, struct ggml_tensor*>>& input_pairs;
  int64_t ne[4];       // resolved output shape from IR (ggml order)
  bool metal_f32_binops;
  bool cuda_bf16_cast;
  bool skip_bf16_castback;
  bool use_native_cmp_ops;
  bool verbose;
};

// ---------------------------------------------------------------------------
// pin_to_cpu -- force a tensor onto the CPU backend so custom ops run there.
// ---------------------------------------------------------------------------
static inline void pin_to_cpu(BuildContext& bc, struct ggml_tensor* x) {
  if (x && bc.gi->sched && bc.handle->backend_cpu) {
    ggml_backend_sched_set_tensor_backend(bc.gi->sched, x, bc.handle->backend_cpu);
    bc.cpu_pinned.push_back(x);
  }
}

// ---------------------------------------------------------------------------
// Broadcast helpers -- converted from the lambdas in build_graph.
// ---------------------------------------------------------------------------

// If `small` is effectively 1D (exactly one dim > 1) and its length matches
// one of big's dims, reshape+permute it so the matching dim aligns, then repeat.
static inline struct ggml_tensor* try_repeat_1d_to_match(
    ggml_context* ctx,
    struct ggml_tensor* small,
    struct ggml_tensor* big) {
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
  struct ggml_tensor* base = small;
  if (non1_ax != 0) {
    int axes[4] = {0, 1, 2, 3};
    axes[0] = non1_ax;
    int out = 1;
    for (int ax = 0; ax < 4; ++ax) {
      if (ax == non1_ax) continue;
      axes[out++] = ax;
    }
    base = safe_ggml_permute(ctx, base, axes[0], axes[1], axes[2], axes[3], "try_permute_base");
    base = ggml_cont(ctx, base);
  }
  for (int ax = 0; ax < 4; ++ax) {
    if (big->ne[ax] != n) {
      continue;
    }
    struct ggml_tensor* s4 = ggml_reshape_4d(ctx, base, n, 1, 1, 1);
    int p0 = 0, p1 = 1, p2 = 2, p3 = 3;
    if (ax == 0) {
      // already in axis0
    } else if (ax == 1) {
      p0 = 1; p1 = 0; p2 = 2; p3 = 3;
    } else if (ax == 2) {
      p0 = 1; p1 = 2; p2 = 0; p3 = 3;
    } else {
      p0 = 1; p1 = 2; p2 = 3; p3 = 0;
    }
    s4 = safe_ggml_permute(ctx, s4, p0, p1, p2, p3, "try_permute_s4");
    s4 = ggml_cont(ctx, s4);
    if (ggml_can_repeat(s4, big)) {
      return ggml_repeat(ctx, s4, big);
    }
  }
  return nullptr;
}

// If src and dst have the same multiset of extents, try a few permutes.
static inline struct ggml_tensor* try_permute_to_match(
    ggml_context* ctx,
    struct ggml_tensor* src,
    struct ggml_tensor* dst) {
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
  for (const auto& p : perms) {
    struct ggml_tensor* t = safe_ggml_permute(ctx, src, p[0], p[1], p[2], p[3], "try_permute_reshape");
    if (ggml_are_same_shape(t, dst)) {
      return ggml_cont(ctx, t);
    }
  }
  return nullptr;
}

// General PyTorch-style broadcast: reshape `small` by placing its non-1
// dims into axes of `big` where they divide evenly, then ggml_repeat.
static inline struct ggml_tensor* try_reshape_broadcast(
    ggml_context* ctx,
    struct ggml_tensor* small,
    struct ggml_tensor* big) {
  struct { int ax; int64_t n; } sdims[4];
  int ns = 0;
  for (int d = 0; d < 4; ++d) {
    if (small->ne[d] != 1) {
      sdims[ns++] = {d, small->ne[d]};
    }
  }
  if (ns == 0 || ns > 4) return nullptr;

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

  int assign[4] = {0, 0, 0, 0};
  bool used[4];

  auto try_assignment = [&]() -> struct ggml_tensor* {
    memset(used, 0, sizeof(used));
    for (int si = 0; si < ns; ++si) {
      int bi = candidates[si][assign[si]];
      if (used[bi]) return nullptr;
      used[bi] = true;
    }
    int64_t new_ne[4] = {1, 1, 1, 1};
    for (int si = 0; si < ns; ++si) {
      new_ne[candidates[si][assign[si]]] = sdims[si].n;
    }
    int64_t nel = ggml_nelements(small);
    struct ggml_tensor* flat = ggml_cont(ctx, ggml_reshape_1d(ctx, small, nel));
    struct ggml_tensor* reshaped = ggml_reshape_4d(ctx, flat, new_ne[0], new_ne[1], new_ne[2], new_ne[3]);
    if (ggml_can_repeat(reshaped, big)) {
      return ggml_repeat(ctx, reshaped, big);
    }
    return nullptr;
  };

  if (ns == 1) {
    for (assign[0] = 0; assign[0] < ncand[0]; ++assign[0]) {
      if (auto* r = try_assignment()) return r;
    }
  } else if (ns == 2) {
    for (assign[0] = 0; assign[0] < ncand[0]; ++assign[0])
      for (assign[1] = 0; assign[1] < ncand[1]; ++assign[1]) {
        if (auto* r = try_assignment()) return r;
      }
  } else if (ns == 3) {
    for (assign[0] = 0; assign[0] < ncand[0]; ++assign[0])
      for (assign[1] = 0; assign[1] < ncand[1]; ++assign[1])
        for (assign[2] = 0; assign[2] < ncand[2]; ++assign[2]) {
          if (auto* r = try_assignment()) return r;
        }
  } else {
    for (assign[0] = 0; assign[0] < ncand[0]; ++assign[0])
      for (assign[1] = 0; assign[1] < ncand[1]; ++assign[1])
        for (assign[2] = 0; assign[2] < ncand[2]; ++assign[2])
          for (assign[3] = 0; assign[3] < ncand[3]; ++assign[3]) {
            if (auto* r = try_assignment()) return r;
          }
  }
  return nullptr;
}

// Slice a larger (max-shape) tensor down to match a smaller (runtime-shape)
// tensor.
static inline struct ggml_tensor* try_slice_to_match(
    ggml_context* ctx,
    struct ggml_tensor* big,
    struct ggml_tensor* small) {
  bool any_diff = false;
  for (int d = 0; d < 4; ++d) {
    if (big->ne[d] < small->ne[d]) return nullptr;
    if (big->ne[d] != small->ne[d]) any_diff = true;
  }
  if (!any_diff) return nullptr;
  auto* v = safe_ggml_view_4d(ctx, big,
      small->ne[0], small->ne[1], small->ne[2], small->ne[3],
      big->nb[1], big->nb[2], big->nb[3], 0);
  if (!v) return nullptr;
  return ensure_cont(ctx, v);
}

// Shared broadcast resolution for binary ops. Returns true on success,
// updating a and b in place. Returns false on failure.
static inline bool resolve_broadcast(
    BuildContext& bc,
    struct ggml_tensor*& a,
    struct ggml_tensor*& b,
    const char* op_name) {
  ggml_context* ctx = bc.ctx;
  if (ggml_are_same_shape(a, b)) return true;
  if (ggml_can_repeat(b, a)) return true;
  if (ggml_can_repeat(a, b)) {
    a = ggml_repeat(ctx, a, b);
  } else if (auto* bb = try_repeat_1d_to_match(ctx, b, a)) {
    b = bb;
  } else if (auto* aa = try_repeat_1d_to_match(ctx, a, b)) {
    a = aa;
  } else if (auto* bp = try_permute_to_match(ctx, b, a)) {
    b = bp;
  } else if (auto* ap = try_permute_to_match(ctx, a, b)) {
    a = ap;
  } else if (auto* bb2 = try_reshape_broadcast(ctx, b, a)) {
    b = bb2;
  } else if (auto* aa2 = try_reshape_broadcast(ctx, a, b)) {
    a = aa2;
  } else if (auto* as = try_slice_to_match(ctx, a, b)) {
    a = as;
  } else if (auto* bs = try_slice_to_match(ctx, b, a)) {
    b = bs;
  } else {
    // Mutual broadcast: both tensors need expanding in different dims.
    int64_t tgt[4];
    bool can_broadcast = true;
    for (int d = 0; d < 4; d++) {
      if (a->ne[d] == b->ne[d]) { tgt[d] = a->ne[d]; }
      else if (a->ne[d] == 1)   { tgt[d] = b->ne[d]; }
      else if (b->ne[d] == 1)   { tgt[d] = a->ne[d]; }
      else { can_broadcast = false; break; }
    }
    if (can_broadcast) {
      struct ggml_tensor* target = ggml_new_tensor_4d(ctx, a->type, tgt[0], tgt[1], tgt[2], tgt[3]);
      if (!ggml_are_same_shape(a, target)) a = ggml_repeat(ctx, a, target);
      if (!ggml_are_same_shape(b, target)) b = ggml_repeat(ctx, b, target);
    } else {
      fprintf(stderr,
              "[executorch-ggml] %s shape mismatch not broadcastable: "
              "a=(%lld,%lld,%lld,%lld) b=(%lld,%lld,%lld,%lld)\n",
              op_name,
              (long long) a->ne[0], (long long) a->ne[1], (long long) a->ne[2], (long long) a->ne[3],
              (long long) b->ne[0], (long long) b->ne[1], (long long) b->ne[2], (long long) b->ne[3]);
      return false;
    }
  }
  return true;
}

} // namespace executorch_ggml
