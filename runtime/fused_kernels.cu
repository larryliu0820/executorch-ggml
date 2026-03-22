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

// ---------------------------------------------------------------------------
// RMSNorm + weight fusion: rms_norm(x, eps) * weight
// Replaces: RMS_NORM(x) → MUL(norm_out, weight)  (2 nodes → 1)
// Saves: 1 global memory round-trip for the intermediate norm output
// ---------------------------------------------------------------------------
__global__ void fused_rms_norm_weight_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int64_t ne0,       // normalized dim (head_dim)
    int64_t n_rows,    // number of rows to normalize
    float eps) {
  int64_t row = blockIdx.x;
  if (row >= n_rows) return;

  const float* x_row = x + row * ne0;
  float* out_row = output + row * ne0;

  // Compute sum of squares
  float sum_sq = 0.0f;
  for (int64_t j = threadIdx.x; j < ne0; j += blockDim.x) {
    float v = x_row[j];
    sum_sq += v * v;
  }
  // Warp reduction
  for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
  // Block reduction via shared memory
  __shared__ float shared_rms;
  __shared__ float shared_warp[32];
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;
  if (lane == 0) shared_warp[wid] = sum_sq;
  __syncthreads();
  if (wid == 0) {
    sum_sq = (lane < blockDim.x / warpSize) ? shared_warp[lane] : 0.0f;
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
      sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    if (lane == 0) shared_rms = rsqrtf(sum_sq / (float)ne0 + eps);
  }
  __syncthreads();
  float rms = shared_rms;

  // Apply normalization + weight
  for (int64_t j = threadIdx.x; j < ne0; j += blockDim.x) {
    // weight broadcasts over rows: weight[j % weight_ne0]
    out_row[j] = x_row[j] * rms * weight[j];
  }
}

// a = x (input), b = weight, dst = output
extern "C" void ggml_fused_rms_norm_weight(
    struct ggml_tensor* dst,
    const struct ggml_tensor* a,
    const struct ggml_tensor* b,
    int ith, int nth, void* userdata) {
  (void)ith; (void)nth;

  int64_t ne0 = a->ne[0];  // normalized dim
  int64_t n_rows = ggml_nelements(a) / ne0;

  // eps stored at offset 24 in dst->op_params (after map_custom2 params)
  float eps = 1e-6f;
  memcpy(&eps, (const char*)dst->op_params + 24, sizeof(float));

  cudaStream_t stream = static_cast<cudaStream_t>(userdata);
  int block = (ne0 < 256) ? 32 : 256;
  fused_rms_norm_weight_kernel<<<n_rows, block, 0, stream>>>(
      static_cast<const float*>(a->data),
      static_cast<const float*>(b->data),
      static_cast<float*>(dst->data),
      ne0, n_rows, eps);
}
