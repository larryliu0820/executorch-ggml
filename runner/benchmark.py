#!/usr/bin/env python3
"""
Benchmark script for ExecuTorch GGML backend with Qwen3.

Usage:
    python runner/benchmark.py [options]

This uses the pre-exported Qwen3-0.6B model at:
    /tmp/qwen3_0_6b_ggml_export/qwen3_0_6b.pt2
"""

import argparse
import sys
import time

import torch


def benchmark(
    model_path: str = "/tmp/qwen3_0_6b_ggml_export/qwen3_0_6b.pt2",
    prompt: str = "The capital of France is",
    max_tokens: int = 16,
    warmup: int = 2,
    model_id: str = "Qwen/Qwen3-0.6B",
):
    """Run benchmark and print results."""
    from transformers import AutoTokenizer
    from executorch.extension.pybindings.portable_lib import _load_for_executorch_from_buffer
    from executorch.exir import to_edge_transform_and_lower
    from executorch_ggml import GgmlPartitioner
    from executorch_ggml.passes import RemoveGraphAssertsPass
    from executorch_ggml.passes.replace_copy_ops_pass import ReplaceCopyOpsPass

    print("=" * 60)
    print("ExecuTorch GGML Benchmark - Qwen3-0.6B")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Prompt: \"{prompt}\"")
    print(f"Max tokens: {max_tokens}")
    print(f"Warmup iterations: {warmup}")
    print()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load and lower model
    print("Loading and lowering model...")
    load_start = time.perf_counter()

    ep = torch.export.load(model_path)
    edge = to_edge_transform_and_lower(
        ep,
        partitioner=[GgmlPartitioner()],
        transform_passes=[ReplaceCopyOpsPass(), RemoveGraphAssertsPass()],
    )
    et_program = edge.to_executorch()
    model = _load_for_executorch_from_buffer(et_program.buffer)

    load_time = time.perf_counter() - load_start
    print(f"Model loaded and lowered in {load_time:.2f}s")

    # Tokenize prompt
    prompt_tokens = tokenizer.encode(prompt)
    print(f"Prompt tokens: {len(prompt_tokens)}")

    # Warmup
    if warmup > 0:
        print(f"\nWarmup ({warmup} iterations)...")
        for _ in range(warmup):
            token = torch.tensor([[prompt_tokens[0]]], dtype=torch.long)
            pos = torch.tensor([0], dtype=torch.long)
            model.forward((token, pos))
        print("Warmup complete.")

    # Prefill benchmark
    print(f"\nPrefill ({len(prompt_tokens)} tokens)...")
    prefill_start = time.perf_counter()

    for i, token_id in enumerate(prompt_tokens):
        token = torch.tensor([[token_id]], dtype=torch.long)
        pos = torch.tensor([i], dtype=torch.long)
        logits = model.forward((token, pos))[0]

    prefill_time = time.perf_counter() - prefill_start
    prefill_tps = len(prompt_tokens) / prefill_time

    # Decode benchmark
    print(f"Decode ({max_tokens} tokens)...")
    decode_start = time.perf_counter()

    generated_tokens = []
    current_pos = len(prompt_tokens)

    for i in range(max_tokens):
        # Sample next token (greedy)
        next_token = logits.view(-1).argmax().item()
        generated_tokens.append(next_token)

        # Check for EOS
        if next_token == tokenizer.eos_token_id:
            break

        # Forward pass
        token = torch.tensor([[next_token]], dtype=torch.long)
        pos = torch.tensor([current_pos], dtype=torch.long)
        logits = model.forward((token, pos))[0]
        current_pos += 1

    decode_time = time.perf_counter() - decode_start
    decode_tps = len(generated_tokens) / decode_time if decode_time > 0 else 0

    # Print results
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Model load+lower:    {load_time:10.2f} s")
    print()
    print(f"Prefill tokens:      {len(prompt_tokens):10d}")
    print(f"Prefill time:        {prefill_time*1000:10.2f} ms")
    print(f"Prefill speed:       {prefill_tps:10.2f} tokens/sec")
    print()
    print(f"Decode tokens:       {len(generated_tokens):10d}")
    print(f"Decode time:         {decode_time*1000:10.2f} ms")
    print(f"Decode speed:        {decode_tps:10.2f} tokens/sec")
    print()
    total_tokens = len(prompt_tokens) + len(generated_tokens)
    total_time = prefill_time + decode_time
    print(f"Total tokens:        {total_tokens:10d}")
    print(f"Total time:          {total_time*1000:10.2f} ms")
    print(f"Overall speed:       {total_tokens/total_time:10.2f} tokens/sec")
    print()

    # Print generated text
    output = tokenizer.decode(prompt_tokens + generated_tokens, skip_special_tokens=True)
    print("Generated text:")
    print("-" * 60)
    print(output)
    print("-" * 60)

    return {
        'prefill_tokens': len(prompt_tokens),
        'prefill_time_ms': prefill_time * 1000,
        'prefill_tps': prefill_tps,
        'decode_tokens': len(generated_tokens),
        'decode_time_ms': decode_time * 1000,
        'decode_tps': decode_tps,
    }


def main():
    parser = argparse.ArgumentParser(description='ExecuTorch GGML Benchmark')
    parser.add_argument('--model', type=str,
                        default='/tmp/qwen3_0_6b_ggml_export/qwen3_0_6b.pt2',
                        help='Path to .pt2 exported model')
    parser.add_argument('--prompt', type=str, default='The capital of France is',
                        help='Prompt text')
    parser.add_argument('--tokens', type=int, default=16,
                        help='Number of tokens to generate')
    parser.add_argument('--warmup', type=int, default=2,
                        help='Number of warmup iterations')
    parser.add_argument('--model-id', type=str, default='Qwen/Qwen3-0.6B',
                        help='HuggingFace model ID for tokenizer')

    args = parser.parse_args()

    import os
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        print("\nTo export the model, run:")
        print("  python scripts/export_qwen3.py")
        sys.exit(1)

    with torch.no_grad():
        benchmark(
            model_path=args.model,
            prompt=args.prompt,
            max_tokens=args.tokens,
            warmup=args.warmup,
            model_id=args.model_id,
        )


if __name__ == '__main__':
    main()
