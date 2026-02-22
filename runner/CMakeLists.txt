# Runner executable for ExecuTorch GGML backend benchmark

add_executable(ggml_runner main.cpp)

# Include directories
target_include_directories(ggml_runner PRIVATE
  "${CMAKE_SOURCE_DIR}/runtime"
  "${CMAKE_SOURCE_DIR}/schema"
  "${LLAMA_CPP_DIR}/ggml/include"
  "${EXECUTORCH_DIR}/.."
  "${EXECUTORCH_DIR}"
  "${EXECUTORCH_DIR}/include"
  "${EXECUTORCH_DIR}/third-party/flatbuffers/include"
)

# For ExecuTorch installed via pip, we need the torch include dir
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'include'))"
  OUTPUT_VARIABLE TORCH_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)
if(EXISTS "${TORCH_INCLUDE_DIR}/c10")
  target_include_directories(ggml_runner PRIVATE "${TORCH_INCLUDE_DIR}")
endif()

# Link directories for ExecuTorch libs
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import executorch; import os; print(os.path.dirname(executorch.__file__))"
  OUTPUT_VARIABLE EXECUTORCH_PYTHON_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)

# Link against ggml and executorch
target_link_directories(ggml_runner PRIVATE
  "${CMAKE_BINARY_DIR}/ggml/src"
  "${CMAKE_BINARY_DIR}/ggml/src/ggml-blas"
  "${CMAKE_BINARY_DIR}/ggml/src/ggml-metal"
  "${EXECUTORCH_PYTHON_DIR}/extension/pybindings"
)

# Link libraries
target_link_libraries(ggml_runner PRIVATE
  ggml_backend
  ggml
  ggml-base
  ggml-cpu
)

if(TARGET ggml-blas)
  target_link_libraries(ggml_runner PRIVATE ggml-blas)
endif()
if(TARGET ggml-metal)
  target_link_libraries(ggml_runner PRIVATE ggml-metal)
endif()

# Link ExecuTorch runtime
# For pip-installed executorch, use the portable_lib
if(EXISTS "${EXECUTORCH_PYTHON_DIR}/extension/pybindings")
  target_link_directories(ggml_runner PRIVATE "${EXECUTORCH_PYTHON_DIR}/extension/pybindings")
endif()

# macOS rpath settings
if(APPLE)
  set_target_properties(ggml_runner PROPERTIES
    INSTALL_RPATH "@executable_path;${CMAKE_BINARY_DIR}/ggml/src;${CMAKE_BINARY_DIR}/ggml/src/ggml-blas;${CMAKE_BINARY_DIR}/ggml/src/ggml-metal"
    BUILD_WITH_INSTALL_RPATH TRUE
  )
endif()
