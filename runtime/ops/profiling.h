#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <ggml.h>
#include <ggml-backend.h>

namespace executorch_ggml {

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

// ---------------------------------------------------------------------------
// Performance logging (enabled by GGML_PERF_LOG=1 environment variable)
// ---------------------------------------------------------------------------
static int perf_log_mode = -1;
static bool should_perf_log() {
  if (perf_log_mode < 0) {
    const char* env = std::getenv("GGML_PERF_LOG");
    perf_log_mode = (env && std::string(env) != "0") ? 1 : 0;
  }
  return perf_log_mode != 0;
}

// Graph cache: skip build_graph for repeated shapes.
// Disabled by default -- build_graph has data-dependent eager constants
// that must be recomputed each call. With zero-copy eager constants,
// build_graph is ~2ms which is acceptable overhead.
// Enable with GGML_GRAPH_CACHE=1 for stateless models only.
static int graph_cache_enabled = -1;
static bool is_graph_cache_disabled() {
  if (graph_cache_enabled < 0) {
    const char* env = std::getenv("GGML_GRAPH_CACHE");
    graph_cache_enabled = (env && std::string(env) != "0") ? 1 : 0;
  }
  return graph_cache_enabled == 0;
}

// ---------------------------------------------------------------------------
// Debug tensor dump (enabled by GGML_DEBUG_DUMP=<path> environment variable)
// ---------------------------------------------------------------------------
// Writes per-node stats (mean, std, min, max) to a file after each node
// executes. Compare two files (CPU vs CUDA) to find divergence point.

struct DebugDumpContext {
  FILE* fp = nullptr;
  int node_idx = 0;
};

static bool debug_dump_eval_callback(struct ggml_tensor* t, bool ask, void* user_data) {
  if (ask) return true;
  auto* ctx = static_cast<DebugDumpContext*>(user_data);
  if (!ctx->fp) return true;

  const size_t nelem = ggml_nelements(t);
  if (nelem == 0 || t->type == GGML_TYPE_I32 || t->type == GGML_TYPE_I64) {
    fprintf(ctx->fp, "node %4d  op=%-20s  type=%-6s  ne=[%lld,%lld,%lld,%lld]  (non-float)\n",
            ctx->node_idx++, ggml_op_name(t->op), ggml_type_name(t->type),
            (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3]);
    return true;
  }
  // Read data to host
  std::vector<float> buf(nelem);
  if (t->type == GGML_TYPE_F32) {
    ggml_backend_tensor_get(t, buf.data(), 0, nelem * sizeof(float));
  } else if (t->type == GGML_TYPE_F16) {
    std::vector<ggml_fp16_t> tmp(nelem);
    ggml_backend_tensor_get(t, tmp.data(), 0, nelem * sizeof(ggml_fp16_t));
    ggml_fp16_to_fp32_row(tmp.data(), buf.data(), (int64_t)nelem);
  } else if (t->type == GGML_TYPE_BF16) {
    std::vector<ggml_bf16_t> tmp(nelem);
    ggml_backend_tensor_get(t, tmp.data(), 0, nelem * sizeof(ggml_bf16_t));
    for (size_t i = 0; i < nelem; i++) buf[i] = ggml_bf16_to_fp32(tmp[i]);
  } else {
    fprintf(ctx->fp, "node %4d  op=%-20s  type=%-6s  (unsupported type)\n",
            ctx->node_idx++, ggml_op_name(t->op), ggml_type_name(t->type));
    return true;
  }

  double sum = 0, sum2 = 0;
  float mn = buf[0], mx = buf[0];
  for (size_t i = 0; i < nelem; i++) {
    float v = buf[i];
    sum += v; sum2 += (double)v * v;
    if (v < mn) mn = v;
    if (v > mx) mx = v;
  }
  double mean = sum / nelem;
  double var = sum2 / nelem - mean * mean;
  double std = var > 0 ? std::sqrt(var) : 0;

  fprintf(ctx->fp, "node %4d  op=%-20s  type=%-6s  ne=[%lld,%lld,%lld,%lld]  mean=%.6f  std=%.6f  min=%.6f  max=%.6f",
          ctx->node_idx, ggml_op_name(t->op), ggml_type_name(t->type),
          (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3],
          mean, std, (double)mn, (double)mx);
  // For MUL, dump src info
  if (t->op == GGML_OP_MUL && ctx->node_idx > 0) {
    for (int si = 0; si < 2; si++) {
      if (t->src[si]) {
        fprintf(ctx->fp, "  src%d=[op=%s type=%s ne=%lldx%lld buf=%p data=%p]",
                si, ggml_op_name(t->src[si]->op), ggml_type_name(t->src[si]->type),
                (long long)t->src[si]->ne[0], (long long)t->src[si]->ne[1],
                (void*)t->src[si]->buffer, t->src[si]->data);
      }
    }
  }
  fprintf(ctx->fp, "\n");
  ctx->node_idx++;
  return true;
}

} // namespace executorch_ggml
