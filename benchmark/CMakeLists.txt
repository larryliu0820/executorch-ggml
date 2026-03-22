# Benchmark executable for executorch-ggml LLM decode speed.
# Requires ExecuTorch built from source (EXECUTORCH_GGML_BUILD_LLAMA_RUNNER=ON).

if(EXECUTORCH_GGML_BUILD_LLAMA_RUNNER)

# Build as plain C++ executable. CUDA device symbols are resolved
# through the fused_kernels static lib (linked via executorch_ggml_runtime).
add_executable(benchmark_llm benchmark_llm.cpp)

target_include_directories(benchmark_llm PRIVATE
  "${CMAKE_SOURCE_DIR}/runtime"
  "${CMAKE_SOURCE_DIR}/schema"
  "${LLAMA_CPP_DIR}/ggml/include"
  "${EXECUTORCH_DIR}/.."
  "${EXECUTORCH_DIR}"
  "${EXECUTORCH_DIR}/include"
  "${EXECUTORCH_DIR}/third-party/flatbuffers/include"
)

target_link_libraries(benchmark_llm PRIVATE
  -Wl,--whole-archive executorch_ggml_runtime -Wl,--no-whole-archive
  -Wl,--no-as-needed executorch -Wl,--as-needed
  ggml ggml-base ggml-cpu
  extension_module_static
  extension_tensor
)

if(TARGET ggml-blas)
  target_link_libraries(benchmark_llm PRIVATE ggml-blas)
endif()
if(TARGET ggml-metal)
  target_link_libraries(benchmark_llm PRIVATE ggml-metal)
endif()
if(TARGET ggml-cuda)
  target_link_libraries(benchmark_llm PRIVATE ggml-cuda)
  # CUDA runtime needed for fused kernels linked through executorch_ggml_runtime.
  # Use -Wl,--no-as-needed to force cudart even though benchmark_llm itself
  # doesn't directly reference CUDA symbols.
  target_link_libraries(benchmark_llm PRIVATE -Wl,--no-as-needed -L/usr/local/cuda/lib64 -lcudart -Wl,--as-needed)
endif()
if(TARGET portable_ops_lib)
  target_link_libraries(benchmark_llm PRIVATE -Wl,--no-as-needed portable_ops_lib portable_kernels -Wl,--as-needed)
endif()

if(APPLE)
  set_target_properties(benchmark_llm PROPERTIES
    BUILD_RPATH "${CMAKE_BINARY_DIR}/ggml/src"
    INSTALL_RPATH "@executable_path"
  )
else()
  set_target_properties(benchmark_llm PROPERTIES
    BUILD_RPATH "${CMAKE_BINARY_DIR}/ggml/src"
  )
endif()

message(STATUS "benchmark_llm target configured")
else()
  message(STATUS "Skipping benchmark_llm: requires EXECUTORCH_GGML_BUILD_LLAMA_RUNNER=ON")
endif()
