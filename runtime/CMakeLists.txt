add_library(executorch_ggml_runtime STATIC
  ggml_backend.cpp
)

add_dependencies(executorch_ggml_runtime ggml_ir_gen)

target_include_directories(executorch_ggml_runtime PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_BINARY_DIR}/.."   # for ggml_ir_generated.h
  "${LLAMA_CPP_DIR}/ggml/include"
  "${EXECUTORCH_DIR}"
  "${EXECUTORCH_DIR}/include"
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
