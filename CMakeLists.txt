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

# ExecuTorch symbols are expected to resolve from the already-loaded
# portable runtime extension (dynamic lookup).
if(APPLE)
  target_link_options(executorch_ggml_backend_py PRIVATE "-undefined" "dynamic_lookup")
else()
  target_link_options(executorch_ggml_backend_py PRIVATE "-Wl,--allow-shlib-undefined")
endif()
