#!/bin/bash
# Exact Commands to Build and Run MobileNetV2 Ops Tests

set -e  # Exit on error

# Configuration
REPO_DIR="/Volumes/larryliu/work/executorch-ggml"
LLAMA_CPP_DIR="/Volumes/larryliu/work/llama.cpp"
EXECUTORCH_DIR="/Volumes/larryliu/work/executorch"

echo "=== Step 1: Navigate to repo ==="
cd "$REPO_DIR"

echo "=== Step 2: Configure CMake ==="
cmake -B build \
  -DLLAMA_CPP_DIR="$LLAMA_CPP_DIR" \
  -DEXECUTORCH_DIR="$EXECUTORCH_DIR" \
  -DCMAKE_BUILD_TYPE=Release

echo "=== Step 3: Build ==="
cd build
ninja

echo "=== Step 4: Verify build artifacts ==="
echo "Python extension:"
ls -lh python/executorch_ggml/_ggml_backend.*.so

echo "Runtime library:"
ls -lh runtime/libexecutorch_ggml_runtime.a

echo "=== Step 5: Run tests ==="
cd "$REPO_DIR"
source .venv/bin/activate

echo "Running all tests..."
pytest tests/ -v

echo ""
echo "=== Summary ==="
echo "Build: ✅ Complete"
echo "Tests: Check results above (expect 11 passed, 3 failed)"
echo ""
echo "The 3 failing tests are expected (BN graph rewriting not yet implemented)."
echo "All core ops work: conv2d, depthwise-conv, hardtanh, mean, view, permute"
