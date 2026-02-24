cmake_minimum_required(VERSION 3.18)
project(executorch_ggml LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Required external paths
# ---------------------------------------------------------------------------
set(LLAMA_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third-party/llama.cpp" CACHE PATH "Path to llama.cpp source tree (contains ggml/)")
set(EXECUTORCH_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third-party/executorch" CACHE PATH "Path to ExecuTorch source/install tree")

message("LLAMA_CPP_DIR being set to ${LLAMA_CPP_DIR}")

message("EXECUTORCH_DIR being set to ${EXECUTORCH_DIR}")

# ---------------------------------------------------------------------------
# FlatBuffers – C++ header from schema
# ---------------------------------------------------------------------------
# We check in schema/ggml_ir_generated.h so builds (including pip build
# isolation) do not require a working `flatc` binary.
#
# If you want to regenerate it, run:
#   flatc --cpp -o schema schema/ggml_ir.fbs
add_custom_target(ggml_ir_gen)

# ---------------------------------------------------------------------------
# ggml (from llama.cpp)
# ---------------------------------------------------------------------------
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

# Metal backend option (defaults to ON on macOS)
if(APPLE)
  option(EXECUTORCH_GGML_BUILD_METAL "Build with Metal GPU acceleration" ON)
else()
  option(EXECUTORCH_GGML_BUILD_METAL "Build with Metal GPU acceleration" OFF)
endif()

# Enable Metal in ggml if requested
if(EXECUTORCH_GGML_BUILD_METAL AND APPLE)
  set(GGML_METAL ON CACHE BOOL "Enable Metal backend on macOS" FORCE)
  set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "Embed Metal library" FORCE)
  message(STATUS "Metal backend: ENABLED")
else()
  set(GGML_METAL OFF CACHE BOOL "Enable Metal backend on macOS" FORCE)
  message(STATUS "Metal backend: DISABLED")
endif()

# Auto-detect CUDA and enable ggml-cuda when available.
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(GGML_CUDA ON CACHE BOOL "ggml: use CUDA" FORCE)
  message(STATUS "CUDA detected (${CMAKE_CUDA_COMPILER}) – enabling ggml-cuda")
else()
  message(STATUS "CUDA not found – ggml-cuda disabled")
endif()

add_subdirectory("${LLAMA_CPP_DIR}/ggml" "${CMAKE_CURRENT_BINARY_DIR}/ggml")

# ---------------------------------------------------------------------------
# Runtime library
# ---------------------------------------------------------------------------
add_subdirectory(runtime)

# ---------------------------------------------------------------------------
# Python extension (pybind11)
# ---------------------------------------------------------------------------

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

# Prefer vendored pybind11 (ExecuTorch submodule) to avoid requiring the
# Python pybind11 package in build isolation environments.
if(EXISTS "${EXECUTORCH_DIR}/third-party/pybind11/include")
  set(PYBIND11_INCLUDE_DIR "${EXECUTORCH_DIR}/third-party/pybind11/include")
elseif(EXISTS "${EXECUTORCH_DIR}/third-party/executorch/third-party/pybind11/include")
  set(PYBIND11_INCLUDE_DIR "${EXECUTORCH_DIR}/third-party/executorch/third-party/pybind11/include")
else()
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_include())"
    OUTPUT_VARIABLE PYBIND11_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
endif()

execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '.so')"
  OUTPUT_VARIABLE PY_EXT_SUFFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Output directory for all artifacts (extension + ggml libs)
set(GGML_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/python/executorch_ggml")

# Override ggml library output directories so they go to our package
set_target_properties(ggml ggml-base ggml-cpu PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${GGML_OUTPUT_DIR}"
  RUNTIME_OUTPUT_DIRECTORY "${GGML_OUTPUT_DIR}"
)
if(TARGET ggml-blas)
  set_target_properties(ggml-blas PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${GGML_OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${GGML_OUTPUT_DIR}"
  )
endif()
if(TARGET ggml-metal)
  set_target_properties(ggml-metal PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${GGML_OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${GGML_OUTPUT_DIR}"
  )
endif()
if(TARGET ggml-cuda)
  set_target_properties(ggml-cuda PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${GGML_OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${GGML_OUTPUT_DIR}"
  )
endif()

add_library(executorch_ggml_backend_py MODULE
  runtime/ggml_backend.cpp
  python/executorch_ggml/_ggml_backend_pybind.cpp
)
add_dependencies(executorch_ggml_backend_py ggml_ir_gen)

set_target_properties(executorch_ggml_backend_py PROPERTIES
  OUTPUT_NAME "_ggml_backend"
  PREFIX ""
  SUFFIX "${PY_EXT_SUFFIX}"
  LIBRARY_OUTPUT_DIRECTORY "${GGML_OUTPUT_DIR}"
  # Use @loader_path so the extension finds ggml libs in the same directory
  BUILD_RPATH "@loader_path"
  INSTALL_RPATH "@loader_path"
)

# Include dirs (match runtime/CMakeLists.txt but local to this target)
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'include'))"
  OUTPUT_VARIABLE TORCH_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)
if(NOT EXISTS "${TORCH_INCLUDE_DIR}/c10")
  set(TORCH_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.venv/lib/python3.13/site-packages/torch/include")
endif()

target_include_directories(executorch_ggml_backend_py PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/runtime"
  "${CMAKE_CURRENT_SOURCE_DIR}/schema"     # ggml_ir_generated.h (checked in)
  "${CMAKE_CURRENT_BINARY_DIR}"            # legacy
  "${LLAMA_CPP_DIR}/ggml/include"
  "${EXECUTORCH_DIR}/.."
  "${EXECUTORCH_DIR}"
  "${EXECUTORCH_DIR}/include"
  "${EXECUTORCH_DIR}/third-party/flatbuffers/include"
  "${TORCH_INCLUDE_DIR}"
  "${PYBIND11_INCLUDE_DIR}"
  ${Python3_INCLUDE_DIRS}
)

# Link against ggml shared libs (now in same output directory)

target_link_libraries(executorch_ggml_backend_py PRIVATE
  ggml
  ggml-base
  ggml-cpu
)
if(TARGET ggml-blas)
  target_link_libraries(executorch_ggml_backend_py PRIVATE ggml-blas)
endif()
if(TARGET ggml-metal)
  target_link_libraries(executorch_ggml_backend_py PRIVATE ggml-metal)
  target_compile_definitions(executorch_ggml_backend_py PRIVATE GGML_USE_METAL=1)
endif()
if(TARGET ggml-cuda)
  target_link_libraries(executorch_ggml_backend_py PRIVATE ggml-cuda)
endif()

# ExecuTorch symbols are expected to resolve from the already-loaded
# portable runtime extension (dynamic lookup).
if(APPLE)
  target_link_options(executorch_ggml_backend_py PRIVATE "-undefined" "dynamic_lookup")
else()
  target_link_options(executorch_ggml_backend_py PRIVATE "-Wl,--allow-shlib-undefined")
endif()

# ---------------------------------------------------------------------------
# ExecuTorch (as subdirectory, for llama runner)
# ---------------------------------------------------------------------------
option(EXECUTORCH_GGML_BUILD_LLAMA_RUNNER
       "Build llm_main by compiling ExecuTorch and the llama runner from source"
       OFF)

if(EXECUTORCH_GGML_BUILD_LLAMA_RUNNER)
  if(CMAKE_VERSION VERSION_LESS "3.29")
    message(FATAL_ERROR
      "EXECUTORCH_GGML_BUILD_LLAMA_RUNNER requires CMake >= 3.29 "
      "(needed by ExecuTorch). Found ${CMAKE_VERSION}.")
  endif()

  # Use the LLM preset to enable all options needed by the llama runner:
  #   EXTENSION_LLM_RUNNER, EXTENSION_MODULE, EXTENSION_TENSOR,
  #   EXTENSION_FLAT_TENSOR, EXTENSION_DATA_LOADER, EXTENSION_NAMED_DATA_MAP,
  #   KERNELS_OPTIMIZED, KERNELS_QUANTIZED, XNNPACK, etc.
  set(EXECUTORCH_BUILD_PRESET_FILE
      "${EXECUTORCH_DIR}/tools/cmake/preset/llm.cmake")

  # Disable targets we don't need.
  set(EXECUTORCH_BUILD_EXECUTOR_RUNNER OFF)
  set(EXECUTORCH_BUILD_TESTS OFF)

  add_subdirectory("${EXECUTORCH_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/executorch")
endif()

# ---------------------------------------------------------------------------
# Runner executables
# ---------------------------------------------------------------------------
add_subdirectory(runner)
