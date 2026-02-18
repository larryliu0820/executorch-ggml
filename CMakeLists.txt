cmake_minimum_required(VERSION 3.18)
project(executorch_ggml LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Required external paths
# ---------------------------------------------------------------------------
set(LLAMA_CPP_DIR "" CACHE PATH "Path to llama.cpp source tree (contains ggml/)")
set(EXECUTORCH_DIR "" CACHE PATH "Path to ExecuTorch source/install tree")

if(NOT LLAMA_CPP_DIR)
  message(FATAL_ERROR "LLAMA_CPP_DIR must be set (path to llama.cpp root)")
endif()

if(NOT EXECUTORCH_DIR)
  message(FATAL_ERROR "EXECUTORCH_DIR must be set (path to ExecuTorch root)")
endif()

# ---------------------------------------------------------------------------
# FlatBuffers – generate C++ header from schema
# ---------------------------------------------------------------------------
find_program(FLATC flatc REQUIRED)

set(GGML_IR_FBS "${CMAKE_CURRENT_SOURCE_DIR}/schema/ggml_ir.fbs")
set(GGML_IR_GENERATED_H "${CMAKE_CURRENT_BINARY_DIR}/ggml_ir_generated.h")

add_custom_command(
  OUTPUT "${GGML_IR_GENERATED_H}"
  COMMAND "${FLATC}" --cpp -o "${CMAKE_CURRENT_BINARY_DIR}" "${GGML_IR_FBS}"
  DEPENDS "${GGML_IR_FBS}"
  COMMENT "Generating ggml_ir_generated.h from FlatBuffer schema"
)
add_custom_target(ggml_ir_gen DEPENDS "${GGML_IR_GENERATED_H}")

# ---------------------------------------------------------------------------
# ggml (from llama.cpp)
# ---------------------------------------------------------------------------
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory("${LLAMA_CPP_DIR}/ggml" "${CMAKE_CURRENT_BINARY_DIR}/ggml")

# ---------------------------------------------------------------------------
# Runtime library
# ---------------------------------------------------------------------------
add_subdirectory(runtime)
