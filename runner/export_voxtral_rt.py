#!/usr/bin/env python3
"""Export Voxtral-Mini-4B-Realtime-2602 with GGML backend and configurable quantization.

Usage:
    source .venv/bin/activate
    python runner/export_voxtral_rt.py --model-path ~/models/Voxtral-Mini-4B-Realtime-2602
    python runner/export_voxtral_rt.py --model-path ~/models/Voxtral-Mini-4B-Realtime-2602 --dtype Q8_0
    python runner/export_voxtral_rt.py --model-path ~/models/Voxtral-Mini-4B-Realtime-2602 --streaming
"""

import argparse
import os
import sys
import time

import torch

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _repo_root)

from export_voxtral_ggml import (
    export_all_ggml,
    export_streaming_ggml,
    lower_to_ggml,
)
from executorch.examples.models.voxtral_realtime.model import load_model

from executorch_ggml import GgmlQuantConfig, GgmlQuantType


def main():
    parser = argparse.ArgumentParser(
        description="Export Voxtral Realtime to ExecuTorch with GGML backend"
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Directory with params.json + consolidated.safetensors",
    )
    parser.add_argument(
        "--output-dir",
        default="./voxtral_ggml",
        help="Output directory (default: ./voxtral_ggml)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=4096,
        help="KV cache length (default: 4096)",
    )
    parser.add_argument(
        "--dtype",
        default="Q8_0",
        choices=["BF16", "FP16"] + [t.value.upper() for t in GgmlQuantType],
        help="Weight dtype / quantization format (default: Q8_0)",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Export streaming encoder (encode_audio_chunk) instead of offline encoder.",
    )
    parser.add_argument(
        "--max-enc-len",
        type=int,
        default=750,
        help="Encoder sliding window size for streaming (default: 750).",
    )
    parser.add_argument(
        "--delay-tokens",
        type=int,
        default=6,
        help="Transcription delay in tokens (default: 6 = 480ms)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.dtype == "BF16":
        quant_config = None
        model_dtype = torch.bfloat16
    elif args.dtype == "FP16":
        quant_config = None
        model_dtype = torch.float16
    else:
        quant_config = GgmlQuantConfig(quant_type=GgmlQuantType(args.dtype.lower()))
        # Quantized models use FP32 for export; GGML quantizes at lowering time.
        # BF16 causes dtype mismatches during CPU-traced export (e.g. GELU returns
        # FP32 but conv bias stays BF16).
        model_dtype = torch.float32

    print(f"Loading model from {args.model_path}...")
    model = load_model(
        args.model_path,
        max_seq_len=args.max_seq_len,
        n_delay_tokens=args.delay_tokens,
        dtype=model_dtype,
    )

    # Encoder RoPE fusion — only for offline mode. The streaming encoder
    # (StreamingAudioEncoderExport) has its own RoPE handling and references
    # the vanilla encoder layers directly.
    if not args.streaming:
        from executorch_ggml.modules.rope import swap_encoder_rope
        swap_encoder_rope(model, freq_base=model.config.enc_rope_theta)
        print("  Applied fused RoPE (swap_encoder_rope)")

    # Replace llama custom ops with standard ATen ops for GGML export
    from executorch_ggml.modules.voxtral_attention import swap_voxtral_attention
    swap_voxtral_attention(model)
    print("  Applied standard attention (swap_voxtral_attention)")

    # Fuse decoder RoPE into ggml_rope_ext (saves ~210 graph nodes)
    from executorch_ggml.modules.voxtral_decoder_rope import swap_decoder_rope
    n_rope = swap_decoder_rope(model, freq_base=model.config.rope_theta)
    print(f"  Applied fused decoder RoPE ({n_rope} layers)")

    # Fold RMS norm weights into subsequent linear projections (saves ~26 nodes)
    from executorch_ggml.passes.fold_decoder_rms_norm_weights import fold_decoder_rms_norm_weights
    n_fold = fold_decoder_rms_norm_weights(model)
    print(f"  Applied fold_decoder_rms_norm_weights ({n_fold} folded)")

    print(f"\nExporting components (dtype={args.dtype})...")
    with torch.no_grad():
        if args.streaming:
            programs, metadata = export_streaming_ggml(
                model, args.max_seq_len, args.max_enc_len, dtype=model_dtype
            )
        else:
            programs, metadata = export_all_ggml(model, args.max_seq_len, dtype=model_dtype)

        t0 = time.time()
        et = lower_to_ggml(programs, metadata=metadata, quant_config=quant_config,
                           target_dtype=args.dtype)
        t1 = time.time()
        print(f"Lowering took {t1 - t0:.1f}s")

    out_filename = f"model_{args.dtype.lower()}.pte"
    out_path = os.path.join(args.output_dir, out_filename)
    print(f"\nSaving {args.dtype} .pte to: {out_path}")
    with open(out_path, "wb") as f:
        et.write_to_file(f)
    out_size = os.path.getsize(out_path) / (1024 * 1024)
    print(f"{args.dtype} size: {out_size:.1f} MB")

    print("\nDone!")
    print(f"\nTo run inference:")
    print(f"  python runner/run_voxtral_rt.py --model {out_path} --model-path {args.model_path} --audio test_audio.wav")


if __name__ == "__main__":
    main()
