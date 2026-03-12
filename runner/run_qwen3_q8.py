#!/usr/bin/env python3
"""Run Qwen3-0.6B Q8_0 using TextLLMRunner and compare to F32."""

import executorch_ggml  # noqa: F401, I001  # isort: skip

from executorch.extension.llm.runner import GenerationConfig, TextLLMRunner

Q8_PATH = "/Users/larryliu/executorch-ggml/qwen3/qwen3_q8_0.pte"
F32_PATH = "/Users/larryliu/executorch-ggml/qwen3/qwen3.pte"
TOKENIZER_PATH = "/Users/larryliu/executorch-ggml/qwen3/"
PROMPT = "The capital of France is"

print(f"Prompt: {PROMPT!r}\n")

# Q8_0
print("--- Q8_0 ---")
runner_q8 = TextLLMRunner(Q8_PATH, TOKENIZER_PATH)
tokens_q8 = []
runner_q8.generate(
    PROMPT,
    GenerationConfig(seq_len=128, temperature=0, echo=True),
    token_callback=lambda t: tokens_q8.append(t),
)
print("".join(tokens_q8))

# F32
print("\n--- F32 ---")
runner_f32 = TextLLMRunner(F32_PATH, TOKENIZER_PATH)
tokens_f32 = []
runner_f32.generate(
    PROMPT,
    GenerationConfig(seq_len=128, temperature=0, echo=True),
    token_callback=lambda t: tokens_f32.append(t),
)
print("".join(tokens_f32))
