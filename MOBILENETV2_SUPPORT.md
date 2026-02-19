# MobileNetV2 Ops Support for ExecuTorch-GGML Backend

## Summary

This implementation extends the executorch-ggml backend to support MobileNetV2 operations by adding:
- **conv2d**: Standard 2D convolution
- **depthwise conv2d**: Depthwise convolution (groups > 1)
- **hardtanh/clamp**: Clipping operation (used for ReLU6)
- **mean(dim)**: Mean reduction along a dimension
- **view/reshape**: Tensor reshaping
- **permute**: Dimension permutation

The implementation also includes **BatchNorm folding** infrastructure that can fuse BatchNorm layers into preceding conv layers during preprocessing.

## Changes Made

### 1. Schema Updates (`schema/ggml_ir.fbs`)
Added new opcodes to the FlatBuffer schema:
- `CONV_2D = 4`
- `CONV_2D_DW = 5` (depthwise)
- `HARDTANH = 6`
- `MEAN = 7`
- `VIEW = 8`
- `PERMUTE = 9`

**Regenerated header**: `schema/ggml_ir_generated.h` (checked in)

### 2. Python Backend (`python/executorch_ggml/`)
- **serialize.py**: Added helper functions to pack op parameters:
  - `pack_conv2d_params(stride, padding, dilation, groups)`
  - `pack_hardtanh_params(min_val, max_val)`
  - `pack_mean_params(dim)`
  - `pack_view_params(new_shape)`
  - `pack_permute_params(perm)`

- **ggml_backend.py**: Extended `preprocess()` to handle:
  - `aten.convolution.default` / `aten.conv2d.default`
  - `aten.hardtanh.default` / `aten.clamp.default`
  - `aten.mean.dim` / `aten._mean_dim.default`
  - `aten.view.default` / `aten._unsafe_view.default` / `aten.reshape.default`
  - `aten.permute.default` / `aten.permute_copy.default`
  - BN folding fusion (treats BN and getitem as no-ops when folded params exist)

- **ggml_partitioner.py**: Added new ops to supported list

- **passes/bn_folding_pass.py**: Existing BN folding infrastructure (analysis + parameter folding)

### 3. C++ Runtime (`runtime/ggml_backend.cpp`)
Extended `init()` to build ggml graph for new ops:
- **CONV_2D**: Uses `ggml_conv_2d(ctx, weight, input, s0, s1, p0, p1, d0, d1)`
- **CONV_2D_DW**: Uses `ggml_conv_2d_dw(ctx, weight, input, s0, s1, p0, p1, d0, d1)`
- **HARDTANH**: Uses `ggml_clamp(ctx, src, min_val, max_val)`
- **MEAN**: Uses `ggml_mean(ctx, src)` (single-dim only)
- **VIEW**: Uses `ggml_reshape_4d(ctx, src, ne0, ne1, ne2, ne3)`
- **PERMUTE**: Uses `ggml_permute(ctx, src, axis0, axis1, axis2, axis3)`

### 4. Tests
Created three test suites:
- **test_conv_ops.py**: Unit tests for conv2d, depthwise conv2d, hardtanh (ReLU6)
- **test_mv2_op_inventory.py**: Verifies BN folding reduces op inventory
- **test_mv2_partial_forward.py**: Tests MobileNetV2 InvertedResidual block

## Build Instructions

### Prerequisites
- CMake 3.18+
- C++17 compiler
- Python 3.13+
- Required repos:
  - `llama.cpp` (for ggml)
  - `executorch` (for ExecuTorch runtime)

### Build Steps

```bash
cd /Volumes/larryliu/work/executorch-ggml

# 1. Create build directory (if not exists)
mkdir -p build

# 2. Configure CMake
cmake -B build \
  -DLLAMA_CPP_DIR=/path/to/llama.cpp \
  -DEXECUTORCH_DIR=/path/to/executorch \
  -DCMAKE_BUILD_TYPE=Release

# 3. Build
cd build
ninja

# This produces:
# - python/executorch_ggml/_ggml_backend.cpython-313-darwin.so (Python extension)
# - runtime/libexecutorch_ggml_runtime.a (Static library)
```

### Alternative: Build via setup.py

```bash
cd /Volumes/larryliu/work/executorch-ggml

# Install in development mode
LLAMA_CPP_DIR=/path/to/llama.cpp \
EXECUTORCH_DIR=/path/to/executorch \
pip install -e .
```

## Run Tests

```bash
cd /Volumes/larryliu/work/executorch-ggml

# Activate virtual environment
source .venv/bin/activate

# Run all tests
pytest tests/ -v

# Run specific test suites
pytest tests/test_conv_ops.py -v                    # Conv2d ops
pytest tests/test_mv2_op_inventory.py -v           # BN folding
pytest tests/test_mv2_partial_forward.py -v        # MobileNetV2 blocks
pytest tests/test_linear_leaky_relu.py -v          # Original Linear test
```

## Current Limitations

### 1. Multi-dimensional Mean
**Issue**: `ggml_mean()` computes mean over all dimensions, not along a specific axis.
**Current workaround**: Only support single-dim mean (`dim=scalar`).
**Impact**: Global average pooling (mean over [2, 3]) not supported directly.

**Suggested fallback**: Use XNNPACK for multi-dim mean operations.

### 2. BatchNorm Folding
**Issue**: Current BN folding only computes folded parameters but doesn't rewrite the graph.
**Current status**: BN nodes are treated as no-ops in the backend, but the graph structure remains.
**Impact**: Serialization may fail for graphs with BN + getitem due to spec mismatches.

**Workaround for testing**: Use conv layers without BN, or manually fold BN before export.

**Suggested fix**: Implement proper graph rewriting to remove BN and getitem nodes after folding.

### 3. Groups Parameter
**Issue**: ggml's `ggml_conv_2d` doesn't support the `groups` parameter directly.
**Current workaround**: Use `ggml_conv_2d_dw` for depthwise conv (groups > 1).
**Impact**: Only depthwise (groups == in_channels) and regular (groups == 1) convolutions supported.

**Suggestion**: For general grouped convolutions, fall back to XNNPACK.

## GGML API Verification

The following ggml functions are used and verified available in llama.cpp:
- ✅ `ggml_conv_2d(ctx, a, b, s0, s1, p0, p1, d0, d1)`
- ✅ `ggml_conv_2d_dw(ctx, a, b, s0, s1, p0, p1, d0, d1)`
- ✅ `ggml_clamp(ctx, a, min, max)`
- ✅ `ggml_mean(ctx, a)`
- ⚠️ `ggml_mean` - only supports all-dims, not axis-specific
- ✅ `ggml_reshape_4d(ctx, a, ne0, ne1, ne2, ne3)`
- ✅ `ggml_permute(ctx, a, axis0, axis1, axis2, axis3)`

## Example: MobileNetV2 Block

```python
import torch
import torch.nn as nn
from torch.export import export
from executorch.exir import to_edge_transform_and_lower
from executorch_ggml import GgmlPartitioner

class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, expand_ratio=6):
        super().__init__()
        hidden = in_ch * expand_ratio
        layers = [
            # Expansion (1x1 conv)
            nn.Conv2d(in_ch, hidden, 1, bias=False),
            nn.ReLU6(inplace=True),
            # Depthwise (3x3)
            nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False),
            nn.ReLU6(inplace=True),
            # Projection (1x1, linear)
            nn.Conv2d(hidden, out_ch, 1, bias=False),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

# Export and lower
model = InvertedResidual(16, 24, stride=1).eval()
exported = export(model, (torch.randn(1, 16, 56, 56),))
edge = to_edge_transform_and_lower(exported, partitioner=[GgmlPartitioner()])
executorch_program = edge.to_executorch()

# Save to .pte file
with open("mv2_block.pte", "wb") as f:
    f.write(executorch_program.buffer)
```

## XNNPACK Fallback Recommendation

For ops not yet supported or with limitations:
1. **Multi-dimensional mean**: Use XNNPACK's `average_pooling_2d` or `reduce_mean`
2. **General grouped convolutions**: Use XNNPACK's grouped convolution support
3. **Other ops**: Delegate unsupported ops to XNNPACK backend via ExecuTorch's multi-backend partitioning

## Future Work

1. **Complete BN folding graph rewriting**: Remove BN/getitem nodes from the graph
2. **Support multi-dimensional mean**: Implement using multiple single-dim mean or use XNNPACK
3. **Support more activation functions**: Add swish, hardsigmoid, etc.
4. **Quantization support**: Integrate with ExecuTorch quantization flow
5. **Performance optimization**: Optimize memory layout for ggml operations

## Testing Summary

All core tests pass:
- ✅ Linear + LeakyReLU (original)
- ✅ Conv2d + ReLU6 (hardtanh)
- ✅ Depthwise Conv2d + ReLU
- ✅ Mean (single-dim) + View + Permute
- ✅ MobileNetV2 InvertedResidual block (without BN)
- ✅ BN folding detection and parameter computation

Tests with BN in the graph require graph rewriting (future work).
