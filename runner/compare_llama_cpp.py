#!/usr/bin/env python3
"""
Compare ExecuTorch GGML backend performance with llama.cpp.

Usage:
    python runner/compare_llama_cpp.py \\
        --et-model <model.pt2> \\
        --gguf-model <model.gguf> \\
        --llama-cpp-dir <path/to/llama.cpp> \\
        [options]

The --et-model should be a .pt2 exported model file. It will be lowered to
the ggml backend in-process.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import torch


def run_llama_cpp_bench(gguf_model: str, llama_cpp_dir: str, prompt: str,
                        max_tokens: int, n_gpu_layers: int = 0):
    """Run llama.cpp benchmark using llama-bench or main."""
    llama_main = Path(llama_cpp_dir) / "build" / "bin" / "llama-cli"
    if not llama_main.exists():
        llama_main = Path(llama_cpp_dir) / "main"
    if not llama_main.exists():
        llama_main = Path(llama_cpp_dir) / "build" / "bin" / "main"

    if not llama_main.exists():
        print(f"Error: llama-cli/main not found in {llama_cpp_dir}")
        print("Please build llama.cpp first.")
        return None

    # Run llama.cpp with timing
    cmd = [
        str(llama_main),
        "-m", gguf_model,
        "-p", prompt,
        "-n", str(max_tokens),
        "-ngl", str(n_gpu_layers),
        "--temp", "0",  # Greedy sampling for consistency
        "-b", "1",  # Batch size 1
    ]

    print(f"Running: {' '.join(cmd)}")
    print()

    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end = time.perf_counter()

    if result.returncode != 0:
        print(f"Error running llama.cpp:")
        print(result.stderr)
        return None

    # Parse output for timing
    output = result.stderr + result.stdout
    print("llama.cpp output:")
    print("-" * 60)
    for line in output.split('\n'):
        if 'eval time' in line.lower() or 'sample time' in line.lower() or \
           'load time' in line.lower() or 'total time' in line.lower() or \
           'prompt eval' in line.lower() or 'tokens per second' in line.lower():
            print(line)
    print("-" * 60)

    return {'total_time': end - start}


def run_executorch_bench(pt2_model: str, prompt: str, max_tokens: int,
                          model_id: str, warmup: int):
    """Run ExecuTorch GGML benchmark."""
    from executorch.extension.pybindings.portable_lib import _load_for_executorch_from_buffer
    from executorch.exir import to_edge_transform_and_lower
    from executorch_ggml import GgmlPartitioner
    from executorch_ggml.passes import RemoveGraphAssertsPass
    from executorch_ggml.passes.replace_copy_ops_pass import ReplaceCopyOpsPass
    from transformers import AutoTokenizer

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load and lower model
    ep = torch.export.load(pt2_model)
    edge = to_edge_transform_and_lower(
        ep,
        partitioner=[GgmlPartitioner()],
        transform_passes=[ReplaceCopyOpsPass(), RemoveGraphAssertsPass()],
    )
    et_program = edge.to_executorch()
    model = _load_for_executorch_from_buffer(et_program.buffer)

    # Tokenize
    prompt_tokens = tokenizer.encode(prompt)

    # Warmup
    for _ in range(warmup):
        token = torch.tensor([[prompt_tokens[0]]], dtype=torch.long)
        pos = torch.tensor([0], dtype=torch.long)
        model.forward((token, pos))

    # Prefill
    prefill_start = time.perf_counter()
    for i, token_id in enumerate(prompt_tokens):
        token = torch.tensor([[token_id]], dtype=torch.long)
        pos = torch.tensor([i], dtype=torch.long)
        logits = model.forward((token, pos))[0]
    prefill_time = time.perf_counter() - prefill_start

    # Decode
    decode_start = time.perf_counter()
    current_pos = len(prompt_tokens)
    generated = 0
    for _ in range(max_tokens):
        next_token = logits[0, :].argmax().item()
        if next_token == tokenizer.eos_token_id:
            break
        token = torch.tensor([[next_token]], dtype=torch.long)
        pos = torch.tensor([current_pos], dtype=torch.long)
        logits = model.forward((token, pos))[0]
        current_pos += 1
        generated += 1
    decode_time = time.perf_counter() - decode_start

    return {
        'prefill_tokens': len(prompt_tokens),
        'prefill_time': prefill_time,
        'prefill_tps': len(prompt_tokens) / prefill_time,
        'decode_tokens': generated,
        'decode_time': decode_time,
        'decode_tps': generated / decode_time if decode_time > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description='Compare ExecuTorch GGML vs llama.cpp')
    parser.add_argument('--et-model', type=str,
                        default='/tmp/qwen3_0_6b_ggml_export/qwen3_0_6b.pt2',
                        help='Path to ExecuTorch .pt2 exported model')
    parser.add_argument('--gguf-model', type=str,
                        help='Path to llama.cpp .gguf model')
    parser.add_argument('--llama-cpp-dir', type=str,
                        help='Path to llama.cpp directory')
    parser.add_argument('--prompt', type=str, default='The capital of France is',
                        help='Prompt text')
    parser.add_argument('--tokens', type=int, default=32,
                        help='Number of tokens to generate')
    parser.add_argument('--warmup', type=int, default=3,
                        help='Warmup iterations')
    parser.add_argument('--model-id', type=str, default='Qwen/Qwen3-0.6B',
                        help='HuggingFace model ID for tokenizer')
    parser.add_argument('--ngl', type=int, default=0,
                        help='Number of GPU layers for llama.cpp')

    args = parser.parse_args()

    print("=" * 70)
    print("ExecuTorch GGML vs llama.cpp Performance Comparison")
    print("=" * 70)
    print(f"Prompt: \"{args.prompt}\"")
    print(f"Max tokens: {args.tokens}")
    print()

    # Run ExecuTorch benchmark
    print("Running ExecuTorch GGML benchmark...")
    print("-" * 70)
    et_results = run_executorch_bench(
        args.et_model, args.prompt, args.tokens, args.model_id, args.warmup)

    print(f"Prefill: {et_results['prefill_tokens']} tokens in "
          f"{et_results['prefill_time']*1000:.2f} ms "
          f"({et_results['prefill_tps']:.2f} t/s)")
    print(f"Decode:  {et_results['decode_tokens']} tokens in "
          f"{et_results['decode_time']*1000:.2f} ms "
          f"({et_results['decode_tps']:.2f} t/s)")
    print()

    # Run llama.cpp benchmark if provided
    if args.gguf_model and args.llama_cpp_dir:
        print("Running llama.cpp benchmark...")
        print("-" * 70)
        llama_results = run_llama_cpp_bench(
            args.gguf_model, args.llama_cpp_dir, args.prompt,
            args.tokens, args.ngl)

        if llama_results:
            print()
    else:
        print("Note: Skipping llama.cpp benchmark (--gguf-model and --llama-cpp-dir not provided)")

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"ExecuTorch GGML Decode: {et_results['decode_tps']:.2f} tokens/sec")


if __name__ == '__main__':
    main()
