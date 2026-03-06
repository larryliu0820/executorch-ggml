#!/usr/bin/env python3
"""Simple script to run Qwen3-0.6B using TextLLMRunner."""

# executorch_ggml must be imported BEFORE TextLLMRunner.  Its __init__ loads
# _portable_lib with RTLD_GLOBAL so that _ggml_backend.so can resolve
# ExecuTorch symbols and register GgmlBackend via static init.  If
# TextLLMRunner is imported first, _portable_lib gets loaded without
# RTLD_GLOBAL and the backend registration silently fails.
import executorch_ggml  # noqa: F401, I001  # isort: skip

from executorch.extension.llm.runner import GenerationConfig, TextLLMRunner

MODEL_PATH = "/data/users/larryliu/executorch-ggml/tests/qwen3_e2e.pte"
TOKENIZER_PATH = "/data/users/larryliu/models/qwen3_ggml/"

runner = TextLLMRunner(MODEL_PATH, TOKENIZER_PATH)

tokens = []
runner.generate(
    "The capital of France is",
    GenerationConfig(seq_len=128, temperature=0, echo=True),
    token_callback=lambda t: tokens.append(t),
)

print("".join(tokens))
