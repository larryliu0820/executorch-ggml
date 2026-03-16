#!/usr/bin/env python3
"""Dump graph_signature fields for the Voxtral text_decoder.

Shows buffers_to_mutate at both the edge-program level (before partitioning)
and the delegated subgraph level (what GgmlBackend.preprocess sees).

Usage:
    python tests/dump_decoder_signature.py --model-path ~/models/Voxtral-Mini-4B-Realtime-2602
"""

import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.export import Dim, export

from executorch.examples.models.voxtral_realtime.export_voxtral_rt import TextDecoderExport
from executorch.examples.models.voxtral_realtime.model import load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--max-seq-len", type=int, default=128)
    args = parser.parse_args()

    model = load_model(
        args.model_path,
        max_seq_len=args.max_seq_len,
        dtype=torch.bfloat16,
        use_standard_attention=True,
    )

    text_decoder = TextDecoderExport(model)
    text_decoder.eval()

    seq_dim = Dim("seq_len", min=1, max=args.max_seq_len)
    sample_embeds = torch.randn(1, 4, model.config.dim, dtype=torch.bfloat16)
    sample_pos = torch.arange(4, dtype=torch.long)

    print("=" * 70)
    print("1. torch.export signature (before edge transform)")
    print("=" * 70)
    ep = export(
        text_decoder,
        (sample_embeds, sample_pos),
        dynamic_shapes={
            "input_embeds": {1: seq_dim},
            "cache_position": {0: seq_dim},
        },
        strict=True,
    )
    sig = ep.graph_signature
    print(f"\nbuffers_to_mutate: {sig.buffers_to_mutate}")
    print(f"\ninputs_to_buffers ({len(dict(sig.inputs_to_buffers))} entries):")
    for node_name, fqn in sig.inputs_to_buffers.items():
        print(f"  {node_name} → {fqn}")

    print(f"\ninputs_to_parameters: {len(dict(sig.inputs_to_parameters))} entries")
    # Just show a few
    for i, (node_name, fqn) in enumerate(sig.inputs_to_parameters.items()):
        if i < 5:
            print(f"  {node_name} → {fqn}")
        elif i == 5:
            print(f"  ... ({len(dict(sig.inputs_to_parameters)) - 5} more)")
            break

    print(f"\noutput_specs:")
    for spec in sig.output_specs:
        print(f"  kind={spec.kind} arg={spec.arg} target={spec.target}")

    # Show which placeholders are user inputs vs buffers vs params
    print(f"\nPlaceholder classification:")
    param_map = dict(sig.inputs_to_parameters)
    buffer_map = dict(sig.inputs_to_buffers)
    mutated = set(sig.buffers_to_mutate.values())
    for node in ep.graph.nodes:
        if node.op == "placeholder":
            if node.name in param_map:
                kind = "PARAM"
            elif node.name in buffer_map:
                fqn = buffer_map[node.name]
                kind = f"BUFFER (mutable)" if fqn in mutated else "BUFFER (const)"
            else:
                kind = "USER_INPUT"
            val = node.meta.get("val")
            shape = tuple(val.shape) if val is not None and hasattr(val, "shape") else "?"
            print(f"  {node.name}: {kind}  shape={shape}")


if __name__ == "__main__":
    main()
