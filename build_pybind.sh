#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

# Requires you already built ggml + generated ggml_ir_generated.h via CMake:
#   cmake -B build -G Ninja -DLLAMA_CPP_DIR=... -DEXECUTORCH_DIR=...
#   cmake --build build

source .venv/bin/activate

PYTHON=${PYTHON:-python}

EXT_SUFFIX=$($PYTHON -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
OUT="$ROOT_DIR/python/executorch_ggml/_ggml_backend${EXT_SUFFIX}"

PYBIND_INCLUDES=$($PYTHON -m pybind11 --includes)

SDK=$(xcrun --sdk macosx --show-sdk-path)

R1="$ROOT_DIR/build/ggml/src"
R2="$ROOT_DIR/build/ggml/src/ggml-blas"
R3="$ROOT_DIR/build/ggml/src/ggml-metal"

# ExecuTorch repo root containing executorch/ folder.
EXECUTORCH_REPO=${EXECUTORCH_DIR:-"/Volumes/larryliu/work/executorch"}

# torch include for c10/
TORCH_INCLUDE=$($PYTHON -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "include"))')

clang++ -std=c++17 -O3 -DNDEBUG -arch arm64 -isysroot "$SDK" \
  -shared -fPIC -undefined dynamic_lookup \
  -Wl,-headerpad_max_install_names \
  -Wl,-rpath,"$R1" -Wl,-rpath,"$R2" -Wl,-rpath,"$R3" \
  $PYBIND_INCLUDES \
  -I"$ROOT_DIR/runtime" \
  -I"$ROOT_DIR/schema" \
  -I"$ROOT_DIR/build" -I"$ROOT_DIR/build/.." \
  -I"/Volumes/larryliu/work/llama.cpp/ggml/include" \
  -I"$EXECUTORCH_REPO/.." -I"$EXECUTORCH_REPO" -I"$EXECUTORCH_REPO/include" \
  -I"$EXECUTORCH_REPO/third-party/flatbuffers/include" \
  -I"$TORCH_INCLUDE" \
  "$ROOT_DIR/runtime/ggml_backend.cpp" \
  "$ROOT_DIR/python/executorch_ggml/_ggml_backend_pybind.cpp" \
  -L"$R1" -L"$R2" \
  -lggml -lggml-base -lggml-cpu -lggml-blas \
  -o "$OUT"

echo "Built: $OUT"

# Quick import test
$PYTHON -c 'import executorch.runtime as rt; from executorch.runtime import Runtime; import executorch_ggml._ggml_backend as b; print("pybind ping", b.ping()); print("backends", Runtime.get().backend_registry.registered_backend_names)'
