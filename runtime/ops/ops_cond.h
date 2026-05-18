#pragma once

#include <cstdio>
#include "build_context.h"
#include "data_structures.h"

namespace executorch_ggml {

// OP_COND build-time stub.
//
// The IR emission side (ggml_partitioner.py + ops/_cond.py + serialize.py)
// is complete: cond regions are recognised, both branches lower to IR
// subgraphs, and the FlatBuffer carries OP_COND tensors with subgraph_ids.
//
// What's left for full runtime support (~150 lines, planned but not yet
// landed):
//   1. Recurse into the two CachedIRSubgraphs at this point, building
//      both subtrees into the parent ggml_context. The simplest approach
//      is "build both branches, select via ggml_where at the output."
//      Correct for pure-functional cond bodies; not yet correct for
//      cond bodies that mutate buffers (whisper's update_cross_attn_cache
//      pattern), since both branches' mutations would then run.
//   2. The harder, perf-correct approach: build each branch as its own
//      cgraph_subtree, read the predicate value at execute() time via
//      host_acc, splice the chosen branch's outputs into the parent
//      graph's downstream consumers, run only the selected subtree.
//      Requires extending GraphInstance with branch-specific cgraphs
//      and execute-time pointer swapping.
//   3. Disable CUDA graph capture when handle->has_cond is set
//      (predicate-dependent kernel sequence).
//
// Until that lands, hitting OP_COND at runtime is a hard error (we
// fail the build_graph call with a clear message). The Python lowering
// path is still useful as it shapes the IR — full runtime support can
// land separately.
static inline struct ggml_tensor* build_op_cond(BuildContext& bc) {
  fprintf(stderr,
          "[ggml_backend] ERROR: OP_COND runtime support is not yet "
          "implemented. The IR was lowered correctly but cannot be "
          "executed. See runtime/ops/ops_cond.h for the planned design "
          "and runtime/ggml_backend.cpp's `populate_ir_cache` for the "
          "subgraph parsing already in place.\n");
  (void)bc;
  return nullptr;
}

}  // namespace executorch_ggml
