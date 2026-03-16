# Live microphone transcription demo using Parakeet TDT + GGML backend
# Requires: EXECUTORCH_GGML_BUILD_LLAMA_RUNNER=ON, libasound (ALSA)

if(EXECUTORCH_GGML_BUILD_LLAMA_RUNNER)

find_package(PkgConfig REQUIRED)
pkg_check_modules(ALSA REQUIRED alsa)

include("${EXECUTORCH_DIR}/tools/cmake/Utils.cmake")

add_executable(parakeet_demo
  main.cpp
  "${CMAKE_CURRENT_SOURCE_DIR}/../pal_init.cpp"
)

target_include_directories(parakeet_demo PRIVATE
  "${EXECUTORCH_DIR}/.."
  "${EXECUTORCH_DIR}"
  "${EXECUTORCH_DIR}/include"
  "${EXECUTORCH_DIR}/third-party/flatbuffers/include"
  "${EXECUTORCH_DIR}/extension/llm/tokenizers/include"
  "${CMAKE_SOURCE_DIR}/runtime"
  "${CMAKE_SOURCE_DIR}/schema"
  "${LLAMA_CPP_DIR}/ggml/include"
  ${ALSA_INCLUDE_DIRS}
)

# For ExecuTorch installed via pip, include torch headers
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'include'))"
  OUTPUT_VARIABLE TORCH_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)
if(EXISTS "${TORCH_INCLUDE_DIR}/c10")
  target_include_directories(parakeet_demo PRIVATE "${TORCH_INCLUDE_DIR}")
endif()

# Core link libraries
set(_parakeet_libs executorch)

# Ops — prefer optimized ops (needed for preprocessor STFT/mel ops),
# fall back to portable ops
if(TARGET optimized_native_cpu_ops_lib)
  list(APPEND _parakeet_libs optimized_native_cpu_ops_lib cpublas eigen_blas)
  executorch_target_link_options_shared_lib(optimized_native_cpu_ops_lib)
elseif(TARGET portable_ops_lib)
  list(APPEND _parakeet_libs portable_ops_lib portable_kernels)
  executorch_target_link_options_shared_lib(portable_ops_lib)
endif()

# Quantized + custom ops (needed for preprocessor on CPU-only builds)
if(TARGET quantized_ops_lib)
  list(APPEND _parakeet_libs quantized_ops_lib)
  executorch_target_link_options_shared_lib(quantized_ops_lib)
endif()
if(TARGET custom_ops)
  list(APPEND _parakeet_libs custom_ops)
  executorch_target_link_options_shared_lib(custom_ops)
endif()

executorch_target_link_options_shared_lib(executorch)

# Tokenizer + LLM runner helper (for load_tokenizer)
list(APPEND _parakeet_libs tokenizers::tokenizers extension_llm_runner)

# Extensions
list(APPEND _parakeet_libs extension_module_static extension_tensor extension_data_loader)

if(TARGET extension_threadpool)
  list(APPEND _parakeet_libs extension_threadpool pthreadpool)
endif()
if(TARGET cpuinfo)
  list(APPEND _parakeet_libs cpuinfo)
endif()

target_link_libraries(parakeet_demo PUBLIC
  -Wl,--whole-archive executorch_ggml_runtime -Wl,--no-whole-archive
  ggml ggml-base ggml-cpu
  -Wl,--no-as-needed ${_parakeet_libs} -Wl,--as-needed
  ${ALSA_LIBRARIES}
)

if(TARGET ggml-blas)
  target_link_libraries(parakeet_demo PRIVATE ggml-blas)
endif()
if(TARGET ggml-metal)
  target_link_libraries(parakeet_demo PRIVATE ggml-metal)
endif()
if(TARGET ggml-cuda)
  target_link_libraries(parakeet_demo PRIVATE ggml-cuda)
endif()

target_compile_options(parakeet_demo PRIVATE -Wno-deprecated-declarations)

if(APPLE)
  set_target_properties(parakeet_demo PROPERTIES
    BUILD_RPATH "${CMAKE_BINARY_DIR}/ggml/src"
    INSTALL_RPATH "@executable_path"
  )
else()
  set_target_properties(parakeet_demo PROPERTIES
    BUILD_RPATH "${CMAKE_BINARY_DIR}/ggml/src"
  )
endif()

message(STATUS "parakeet_demo target configured")
else()
  message(STATUS "Skipping parakeet_demo: requires EXECUTORCH_GGML_BUILD_LLAMA_RUNNER=ON")
endif()
