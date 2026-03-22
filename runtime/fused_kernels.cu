/**
 * Fused CUDA kernels for cross-op fusion in the ggml graph.
 *
 * These are used via ggml_map_custom2() at graph build time.
 * The CUDA backend dispatches them like any other op, so they get
 * captured by CUDA graphs normally.
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <ggml.h>

// ---------------------------------------------------------------------------
// SiLU-gate fusion: silu(gate) * up = gate * sigmoid(gate) * up
// Replaces: UNARY(SiLU, gate) → MUL(silu_out, up)  (2 nodes → 1)
// Saves: 1 global memory round-trip for the intermediate silu output
// ---------------------------------------------------------------------------
__global__ void fused_silu_gate_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ output,
    int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float g = gate[i];
    float sigmoid_g = 1.0f / (1.0f + expf(-g));
    output[i] = g * sigmoid_g * up[i];
  }
}

// ggml_custom2_op_t callback — called by ggml's CUDA compute dispatch.
// a = gate_proj output, b = up_proj output, dst = fused result.
extern "C" void ggml_fused_silu_gate(
    struct ggml_tensor* dst,
    const struct ggml_tensor* a,
    const struct ggml_tensor* b,
    int ith, int nth, void* userdata) {
  (void)ith; (void)nth; (void)userdata;

  const float* gate = static_cast<const float*>(a->data);
  const float* up   = static_cast<const float*>(b->data);
  float* output     = static_cast<float*>(dst->data);
  int64_t n         = ggml_nelements(dst);

  // userdata contains the CUDA stream (passed by our forked ggml-cuda dispatch)
  cudaStream_t stream = static_cast<cudaStream_t>(userdata);

  const int block = 256;
  const int grid  = (n + block - 1) / block;
  fused_silu_gate_kernel<<<grid, block, 0, stream>>>(gate, up, output, n);
}
