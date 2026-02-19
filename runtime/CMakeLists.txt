add_library(executorch_ggml_runtime STATIC
  ggml_backend.cpp
)

add_dependencies(executorch_ggml_runtime ggml_ir_gen)

# ExecuTorch headers use <executorch/...> includes, where the repo has an
# executorch/ folder at its root. So we need EXECUTORCH_DIR/.. on include path.
#
# ExecuTorch runtime headers also depend on c10/ (from the active PyTorch install),
# and FlatBuffers headers.
# Try to locate PyTorch include dir (for c10/).
# Prefer the configured Python3 interpreter if provided, otherwise fall back.
if(DEFINED Python3_EXECUTABLE)
  set(_ET_PYTHON "${Python3_EXECUTABLE}")
else()
  set(_ET_PYTHON "python3")
endif()

execute_process(
  COMMAND ${_ET_PYTHON} -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'include'))"
  OUTPUT_VARIABLE TORCH_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)

# Fallback: common venv location in this repo
if(NOT EXISTS "${TORCH_INCLUDE_DIR}/c10")
  set(TORCH_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.venv/lib/python3.13/site-packages/torch/include")
endif()

target_include_directories(executorch_ggml_runtime PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_BINARY_DIR}/.."   # for ggml_ir_generated.h
  "${LLAMA_CPP_DIR}/ggml/include"
  "${EXECUTORCH_DIR}/.."
  "${EXECUTORCH_DIR}"
  "${EXECUTORCH_DIR}/include"
  "${EXECUTORCH_DIR}/third-party/flatbuffers/include"
  "${TORCH_INCLUDE_DIR}"
)

target_link_libraries(executorch_ggml_runtime PRIVATE
  ggml
  ggml-cpu
)

# ExecuTorch libraries (link order matters)
find_library(EXECUTORCH_LIB executorch PATHS "${EXECUTORCH_DIR}/lib" NO_DEFAULT_PATH)
if(EXECUTORCH_LIB)
  target_link_libraries(executorch_ggml_runtime PRIVATE "${EXECUTORCH_LIB}")
endif()
