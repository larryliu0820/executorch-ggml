NPROC ?= $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
CMAKE ?= $(shell which cmake)

llm-runner:
	$(CMAKE) -B build-llm \
		-DCMAKE_BUILD_TYPE=Release \
		-DEXECUTORCH_GGML_BUILD_LLAMA_RUNNER=ON
	$(CMAKE) --build build-llm --target llm_main -j$(NPROC)

llm-runner-debug:
	$(CMAKE) -B build-llm-debug \
		-DCMAKE_BUILD_TYPE=Debug \
		-DEXECUTORCH_GGML_BUILD_LLAMA_RUNNER=ON
	$(CMAKE) --build build-llm-debug --target llm_main -j$(NPROC)

parakeet-demo:
	$(CMAKE) -B build-llm \
		-DCMAKE_BUILD_TYPE=Release \
		-DEXECUTORCH_GGML_BUILD_LLAMA_RUNNER=ON
	$(CMAKE) --build build-llm --target parakeet_demo -j$(NPROC)

.PHONY: llm-runner llm-runner-debug parakeet-demo
