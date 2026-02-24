NPROC ?= $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

llm-runner:
	cmake -B build-llm \
		-DCMAKE_BUILD_TYPE=Release \
		-DEXECUTORCH_GGML_BUILD_LLAMA_RUNNER=ON
	cmake --build build-llm --target llm_main -j$(NPROC)

llm-runner-debug:
	cmake -B build-llm-debug \
		-DCMAKE_BUILD_TYPE=Debug \
		-DEXECUTORCH_GGML_BUILD_LLAMA_RUNNER=ON
	cmake --build build-llm-debug --target llm_main -j$(NPROC)

.PHONY: llm-runner llm-runner-debug
