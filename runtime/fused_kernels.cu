/**
 * Fused CUDA kernels for cross-op fusion in the ggml graph.
 *
 * These are launched via the eval callback to replace multi-node subgraphs
 * with single fused kernels, reducing both node count and memory traffic.
 */

#include <cuda_runtime.h>
#include <cstdint>

// SiLU-gate fusion: computes silu(gate) * up = gate * sigmoid(gate) * up
// Replaces: UNARY(SiLU, gate) -> MUL(silu_out, up)
// Saves: 1 global memory write+read (the intermediate silu output)
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

extern "C" void launch_fused_silu_gate(
    const void* gate, const void* up, void* output,
    int64_t n, void* stream_ptr) {
  cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
  const int block = 256;
  const int grid = (n + block - 1) / block;
  fused_silu_gate_kernel<<<grid, block, 0, stream>>>(
      static_cast<const float*>(gate),
      static_cast<const float*>(up),
      static_cast<float*>(output),
      n);
}
