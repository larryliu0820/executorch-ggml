# Quick Start: Build and Test MobileNetV2 Ops

## One-Time Setup

```bash
# Navigate to repo
cd /Volumes/larryliu/work/executorch-ggml

# Ensure dependencies are available
# - llama.cpp should be at: /Volumes/larryliu/work/llama.cpp
# - executorch should be at: /Volumes/larryliu/work/executorch
```

## Build

```bash
# Configure and build (from repo root)
cmake -B build \
  -DLLAMA_CPP_DIR=/Volumes/larryliu/work/llama.cpp \
  -DEXECUTORCH_DIR=/Volumes/larryliu/work/executorch \
  -DCMAKE_BUILD_TYPE=Release

cd build && ninja

# Verify build artifacts
ls -lh python/executorch_ggml/_ggml_backend.*.so
ls -lh runtime/libexecutorch_ggml_runtime.a
```

## Run Tests

```bash
cd /Volumes/larryliu/work/executorch-ggml

# Activate venv
source .venv/bin/activate

# Run all tests
pytest tests/ -v

# Expected: 11 passed, 3 failed (BN-related tests fail as documented)

# Run only passing tests
pytest tests/test_conv_ops.py tests/test_linear_leaky_relu.py -v
pytest tests/test_mv2_op_inventory.py -v
pytest tests/test_mobilenetv2_ops.py::TestMeanViewPermute -v
pytest tests/test_mobilenetv2_ops.py::TestDepthwiseConv -v
```

## Test Results

### ✅ Passing Tests (11/14)

1. **test_bn_folding.py** - BN folding utilities
2. **test_bn_folding_pass.py** - BN folding pattern detection
3. **test_conv_ops.py::TestConvReLU6** - Conv2d + ReLU6
4. **test_conv_ops.py::TestDepthwiseConv** - Depthwise conv + ReLU
5. **test_linear_leaky_relu.py** - Linear + LeakyReLU (2 tests)
6. **test_mobilenetv2_ops.py::TestDepthwiseConv** - Depthwise conv standalone
7. **test_mobilenetv2_ops.py::TestMeanViewPermute** - Mean/view/permute
8. **test_mv2_op_inventory.py** - BN folding inventory shrink (2 tests)

### ❌ Expected Failures (3/14)

These tests fail due to BN graph structure not being rewritten (documented limitation):

1. **test_mobilenetv2_ops.py::TestConvBNReLU6::test_partition_and_lower** - BN+getitem spec mismatch
2. **test_mv2_partial_forward.py** - MobileNetV2 blocks with BN

**Workaround**: Use conv layers without BN for testing, or manually fold BN before export.

## Key Files Changed

```
schema/ggml_ir.fbs                  # Added new opcodes
schema/ggml_ir_generated.h          # Regenerated (checked in)
python/executorch_ggml/
  serialize.py                      # Added pack_*_params functions
  ggml_backend.py                   # Added conv2d/hardtanh/mean/view/permute
  ggml_partitioner.py               # Added new ops to supported list
runtime/ggml_backend.cpp            # Added ggml graph building for new ops
tests/
  test_conv_ops.py                  # NEW: Conv2d tests
  test_mv2_op_inventory.py          # NEW: BN folding tests
  test_mv2_partial_forward.py       # NEW: MobileNetV2 block tests
  test_mobilenetv2_ops.py           # NEW: Various MV2 ops
```

## Verification

```bash
# Check that the Python extension can be imported
python -c "import executorch_ggml; print('✅ Extension loaded')"

# Check that partitioner recognizes new ops
python -c "from executorch_ggml import GgmlPartitioner; print('✅ Partitioner loaded')"

# Check that conv ops are in supported list
python -c "from executorch_ggml.ggml_partitioner import _SUPPORTED_OP_NAMES; print('conv2d' in str(_SUPPORTED_OP_NAMES))"
```

## What Works

- ✅ Standard 2D convolution (stride, padding, dilation)
- ✅ Depthwise convolution (groups > 1)
- ✅ Hardtanh / clamp (ReLU6: min=-6, max=6)
- ✅ Mean along single dimension
- ✅ View / reshape
- ✅ Permute / transpose
- ✅ BN parameter folding (computation)
- ✅ Linear + LeakyReLU (original)

## What Doesn't Work Yet

- ❌ BN graph rewriting (removing BN/getitem nodes)
- ❌ Multi-dimensional mean (e.g., mean(dim=[2, 3]))
- ❌ General grouped convolutions (only depthwise supported)

## Suggested Fallback

For unsupported ops, use XNNPACK backend via multi-backend partitioning.

## Next Steps

1. Implement BN graph rewriting to remove BN/getitem nodes
2. Add support for multi-dimensional mean (or use XNNPACK)
3. Add quantization support
4. Add performance benchmarks
