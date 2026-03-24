# Runner executable for ExecuTorch GGML backend benchmark
# Requires ExecuTorch built from source (EXECUTORCH_GGML_BUILD_LLAMA_RUNNER=ON)
# since it links Module, Tensor, and the ExecuTorch runtime.

if(EXECUTORCH_GGML_BUILD_LLAMA_RUNNER)

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

# Link libraries — executorch_ggml_runtime + ggml + ExecuTorch runtime
target_link_libraries(ggml_runner PRIVATE
  executorch_ggml_runtime
  ggml
  ggml-base
  ggml-cpu
  executorch
  extension_module_static
  extension_tensor
)

if(TARGET ggml-blas)
  target_link_libraries(ggml_runner PRIVATE ggml-blas)
endif()
if(TARGET ggml-metal)
  target_link_libraries(ggml_runner PRIVATE ggml-metal)
endif()
if(TARGET ggml-cuda)
  target_link_libraries(ggml_runner PRIVATE ggml-cuda)
endif()
if(TARGET portable_ops_lib)
  target_link_libraries(ggml_runner PRIVATE portable_ops_lib portable_kernels)
endif()

# macOS rpath settings
if(APPLE)
  set_target_properties(ggml_runner PROPERTIES
    INSTALL_RPATH "@executable_path;${CMAKE_BINARY_DIR}/ggml/src;${CMAKE_BINARY_DIR}/ggml/src/ggml-blas;${CMAKE_BINARY_DIR}/ggml/src/ggml-metal"
    BUILD_WITH_INSTALL_RPATH TRUE
  )
endif()

message(STATUS "ggml_runner target configured")
else()
  message(STATUS "Skipping ggml_runner: requires EXECUTORCH_GGML_BUILD_LLAMA_RUNNER=ON")
endif()

# ---------------------------------------------------------------------------
# llm_main: llama runner linked with the GGML backend
# ---------------------------------------------------------------------------
# Enable with -DEXECUTORCH_GGML_BUILD_LLAMA_RUNNER=ON in the root project.
# ExecuTorch is added as a subdirectory so all targets are built in-tree.

set(ET_LLAMA_DIR "${EXECUTORCH_DIR}/examples/models/llama")

if(EXECUTORCH_GGML_BUILD_LLAMA_RUNNER)
  include("${EXECUTORCH_DIR}/tools/cmake/Utils.cmake")

  # Build the llama_runner library from the ExecuTorch example source.
  # This subdirectory only needs EXECUTORCH_ROOT set and the core ExecuTorch
  # targets to exist (provided by add_subdirectory(executorch) in the root).
  set(EXECUTORCH_ROOT "${EXECUTORCH_DIR}")
  add_subdirectory(
    "${ET_LLAMA_DIR}/runner"
    "${CMAKE_CURRENT_BINARY_DIR}/llama_runner"
  )

  # -- Assemble link libraries (mirrors examples/models/llama/CMakeLists.txt) --
  set(_link_libs executorch gflags)

  # if(TARGET optimized_native_cpu_ops_lib)
  #   list(APPEND _link_libs optimized_native_cpu_ops_lib optimized_kernels
  #        portable_kernels cpublas eigen_blas)
  #   executorch_target_link_options_shared_lib(optimized_native_cpu_ops_lib)
  # else()
  #   list(APPEND _link_libs portable_ops_lib portable_kernels)
  #   executorch_target_link_options_shared_lib(portable_ops_lib)
  # endif()

  list(APPEND _link_libs portable_ops_lib portable_kernels)
  executorch_target_link_options_shared_lib(portable_ops_lib)
  # executorch_target_link_options_shared_lib(quantized_ops_lib)
  # list(APPEND _link_libs quantized_kernels quantized_ops_lib)

  # if(TARGET custom_ops)
  #   executorch_target_link_options_shared_lib(custom_ops)
  #   list(APPEND _link_libs custom_ops)
  # endif()

  if(TARGET extension_threadpool)
    list(APPEND _link_libs extension_threadpool pthreadpool)
  endif()
  if(TARGET cpuinfo)
    list(APPEND _link_libs cpuinfo)
  endif()

  # if(TARGET xnnpack_backend)
  #   set(_xnnpack_libs xnnpack_backend XNNPACK xnnpack-microkernels-prod)
  #   if(TARGET kleidiai)
  #     list(APPEND _xnnpack_libs kleidiai)
  #   endif()
  #   list(APPEND _link_libs ${_xnnpack_libs})
  #   executorch_target_link_options_shared_lib(xnnpack_backend)
  # endif()

  executorch_target_link_options_shared_lib(executorch)

  # -- Build llm_main executable --
  add_executable(llm_main
    "${ET_LLAMA_DIR}/main.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/pal_init.cpp"
  )

  target_include_directories(llm_main PRIVATE
    "${EXECUTORCH_DIR}/.."
    "${EXECUTORCH_DIR}/extension/llm/tokenizers/include"
    "${CMAKE_SOURCE_DIR}/runtime"
    "${CMAKE_SOURCE_DIR}/schema"
    "${LLAMA_CPP_DIR}/ggml/include"
  )

  if(APPLE)
    target_link_libraries(llm_main PUBLIC
      llama_runner
      -Wl,-force_load,$<TARGET_FILE:executorch_ggml_runtime>
      ggml ggml-base ggml-cpu
      ${_link_libs})
  else()
    target_link_libraries(llm_main PUBLIC
      llama_runner
      -Wl,--whole-archive executorch_ggml_runtime -Wl,--no-whole-archive
      ggml ggml-base ggml-cpu
      # --no-as-needed ensures the ops shared libs stay linked even though
      # llm_main has no direct symbol references (registration is via static
      # constructors).
      -Wl,--no-as-needed ${_link_libs} -Wl,--as-needed)
  endif()
  if(TARGET ggml-blas)
    target_link_libraries(llm_main PRIVATE ggml-blas)
  endif()
  if(TARGET ggml-metal)
    target_link_libraries(llm_main PRIVATE ggml-metal)
  endif()
  if(TARGET ggml-cuda)
    target_link_libraries(llm_main PRIVATE ggml-cuda)
  endif()

  # RPATH so it can find shared libs at runtime
  if(APPLE)
    set_target_properties(llm_main PROPERTIES
      BUILD_RPATH "$<TARGET_FILE_DIR:llama_runner>;${CMAKE_BINARY_DIR}/ggml/src"
      INSTALL_RPATH "@executable_path"
    )
  else()
    set_target_properties(llm_main PROPERTIES
      BUILD_RPATH "$<TARGET_FILE_DIR:llama_runner>;${CMAKE_BINARY_DIR}/ggml/src"
    )
  endif()

  target_compile_options(llm_main PRIVATE -Wno-deprecated-declarations)

  message(STATUS "llm_main target configured (llama runner + GGML backend)")

# ---------------------------------------------------------------------------
# voxtral_rt: Voxtral Realtime streaming runner linked with GGML backend
# ---------------------------------------------------------------------------
# Compiles the upstream voxtral_realtime_runner source directly — no patches.
# The GGML backend self-registers at static init; PTE files with GGML
# delegates are handled automatically.

set(ET_VOXTRAL_DIR "${EXECUTORCH_DIR}/examples/models/voxtral_realtime")

# Use optimized CPU ops (superset of portable ops, includes STFT/FFT for mel
# preprocessor). Cannot link both portable_ops_lib and optimized_native_cpu_ops_lib
# (duplicate op registration aborts at static init).
set(_voxtral_ops_lib)
if(TARGET optimized_native_cpu_ops_lib)
  set(_voxtral_ops_lib optimized_native_cpu_ops_lib)
else()
  set(_voxtral_ops_lib portable_ops_lib)
endif()

# Ensure static constructors (op/backend registration) are loaded from shared
# libs — mirrors upstream voxtral_realtime CMakeLists.txt.
executorch_target_link_options_shared_lib(executorch)
executorch_target_link_options_shared_lib(${_voxtral_ops_lib})

add_executable(voxtral_rt
  "${ET_VOXTRAL_DIR}/main.cpp"
  "${ET_VOXTRAL_DIR}/voxtral_realtime_runner.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/pal_init.cpp"
)

target_include_directories(voxtral_rt PRIVATE
  "${ET_VOXTRAL_DIR}"
  "${EXECUTORCH_DIR}/.."
  "${EXECUTORCH_DIR}/extension/llm/tokenizers/include"
  "${CMAKE_SOURCE_DIR}/runtime"
  "${CMAKE_SOURCE_DIR}/schema"
  "${LLAMA_CPP_DIR}/ggml/include"
)

set(_voxtral_link_libs
  gflags
  extension_llm_runner extension_module extension_data_loader
  extension_tensor extension_flat_tensor)

if(APPLE)
  target_link_libraries(voxtral_rt PUBLIC
    -Wl,-force_load,$<TARGET_FILE:executorch_ggml_runtime>
    -Wl,-force_load,$<TARGET_FILE:${_voxtral_ops_lib}>
    ggml ggml-base ggml-cpu executorch
    ${_voxtral_link_libs})
else()
  # --whole-archive on GGML runtime for backend registration.
  # --no-as-needed on executorch + ops lib to ensure their static
  # constructors (op/backend registration) are loaded even though
  # voxtral_rt has no direct symbol references to them.
  target_link_libraries(voxtral_rt PUBLIC
    -Wl,--whole-archive executorch_ggml_runtime -Wl,--no-whole-archive
    ggml ggml-base ggml-cpu
    -Wl,--no-as-needed executorch ${_voxtral_ops_lib} -Wl,--as-needed
    ${_voxtral_link_libs})
endif()
if(TARGET ggml-blas)
  target_link_libraries(voxtral_rt PRIVATE ggml-blas)
endif()
if(TARGET ggml-metal)
  target_link_libraries(voxtral_rt PRIVATE ggml-metal)
endif()
if(TARGET ggml-cuda)
  target_link_libraries(voxtral_rt PRIVATE ggml-cuda)
  target_link_libraries(voxtral_rt PRIVATE -Wl,--no-as-needed -L/usr/local/cuda/lib64 -lcudart -Wl,--as-needed)
endif()
# XNNPACK backend for preprocessor PTE (mel spectrogram uses XNNPACK-delegated ops)
if(TARGET xnnpack_backend)
  set(_xnnpack_libs xnnpack_backend XNNPACK xnnpack-microkernels-prod)
  if(TARGET kleidiai)
    list(APPEND _xnnpack_libs kleidiai)
  endif()
  target_link_libraries(voxtral_rt PRIVATE -Wl,--no-as-needed ${_xnnpack_libs} -Wl,--as-needed)
endif()

if(APPLE)
  set_target_properties(voxtral_rt PROPERTIES
    BUILD_RPATH "${CMAKE_BINARY_DIR}/ggml/src"
    INSTALL_RPATH "@executable_path"
  )
else()
  set_target_properties(voxtral_rt PROPERTIES
    BUILD_RPATH "${CMAKE_BINARY_DIR}/ggml/src"
  )
endif()

target_compile_options(voxtral_rt PRIVATE -Wno-deprecated-declarations)

message(STATUS "voxtral_rt target configured (streaming runner + GGML backend)")

else()
  message(STATUS "Skipping llm_main: EXECUTORCH_GGML_BUILD_LLAMA_RUNNER is OFF")
endif()
