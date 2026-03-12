"""Export nvidia/parakeet-tdt-0.6b-v3 to ExecuTorch with GGML backend.

Reuses model wrappers from the upstream Parakeet export script and adds
GGML-specific lowering with BatchNormFoldingRewritePass for Conv1d+BN.

Usage:
    python export_parakeet_ggml.py [--audio test.wav] [--output-dir ./parakeet_ggml]
"""

import argparse
import os
import sys

import torch
from torch.export import Dim, export

# Import from the upstream Parakeet export script
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "third-party",
        "executorch",
        "examples",
        "models",
        "parakeet",
    ),
)
from export_parakeet_tdt import (
    DecoderStep,
    EncoderWithProjection,
    JointWithArgmax,
    PreprocessorWrapper,
    extract_tokenizer,
    greedy_decode_executorch,
    load_model,
    transcribe_eager,
)

from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig
from executorch.exir.passes import MemoryPlanningPass

from executorch_ggml import GgmlPartitioner, to_edge_rewrite_and_lower
from executorch_ggml.passes.bn_folding_pass import BatchNormFoldingPass
from executorch_ggml.passes.replace_copy_ops_pass import ReplaceCopyOpsPass


def export_all_ggml(model, dtype=torch.float):
    """Export all Parakeet components as ExportedPrograms."""
    programs = {}

    sample_rate = model.preprocessor._cfg.sample_rate
    window_stride = float(model.preprocessor._cfg.window_stride)
    encoder_max_frames = model.encoder.max_audio_length
    max_audio_sec = int(encoder_max_frames * window_stride)
    max_audio_samples = int(sample_rate * max_audio_sec)
    max_mel_frames = int(max_audio_sec / window_stride)

    # --- Preprocessor ---
    preprocessor_wrapper = PreprocessorWrapper(model.preprocessor)
    preprocessor_wrapper.eval()
    sample_audio = torch.randn(max_audio_samples, dtype=torch.float)
    sample_length = torch.tensor([sample_audio.shape[0]], dtype=torch.int64)

    old_cuda_is_available = torch.cuda.is_available
    torch.cuda.is_available = lambda: False
    programs["preprocessor"] = export(
        preprocessor_wrapper,
        (sample_audio, sample_length),
        dynamic_shapes={
            "audio": {0: Dim("audio_len", min=1600, max=max_audio_samples)},
            "length": {},
        },
        strict=False,
    )
    torch.cuda.is_available = old_cuda_is_available

    # --- Encoder ---
    feat_in = getattr(model.encoder, "_feat_in", 128)
    audio_signal = torch.randn(1, feat_in, max_mel_frames, dtype=dtype)
    length = torch.tensor([max_mel_frames], dtype=torch.int64)
    encoder_with_proj = EncoderWithProjection(model.encoder, model.joint)
    encoder_with_proj.eval()

    programs["encoder"] = export(
        encoder_with_proj,
        (),
        kwargs={"audio_signal": audio_signal, "length": length},
        dynamic_shapes={
            "audio_signal": {2: Dim.AUTO},
            "length": {},
        },
        strict=False,
    )

    # --- Decoder ---
    num_layers = model.decoder.pred_rnn_layers
    pred_hidden = model.decoder.pred_hidden
    decoder_step = DecoderStep(model.decoder, model.joint)
    decoder_step.eval()

    token = torch.tensor([[0]], dtype=torch.long)
    h = torch.zeros(num_layers, 1, pred_hidden, dtype=dtype)
    c = torch.zeros(num_layers, 1, pred_hidden, dtype=dtype)
    programs["decoder_step"] = export(
        decoder_step,
        (token, h, c),
        dynamic_shapes={"token": {}, "h": {}, "c": {}},
        strict=False,
    )

    # --- Joint ---
    joint_hidden = model.joint.joint_hidden
    num_token_classes = model.tokenizer.vocab_size + 1

    f_proj = torch.randn(1, 1, joint_hidden, dtype=dtype)
    g_proj = torch.randn(1, 1, joint_hidden, dtype=dtype)
    programs["joint"] = export(
        JointWithArgmax(model.joint, num_token_classes),
        (f_proj, g_proj),
        dynamic_shapes={"f": {}, "g": {}},
        strict=False,
    )

    encoder_subsampling_factor = int(
        getattr(model.encoder, "subsampling_factor", 8)
    )

    metadata = {
        "num_rnn_layers": num_layers,
        "pred_hidden": pred_hidden,
        "joint_hidden": joint_hidden,
        "vocab_size": model.tokenizer.vocab_size,
        "blank_id": model.tokenizer.vocab_size,
        "sample_rate": sample_rate,
        "window_stride": window_stride,
        "encoder_subsampling_factor": encoder_subsampling_factor,
    }

    # Remove dropout ops (identity in eval mode) so the partitioner
    # can form a single large delegate instead of many fragments.
    _dropout_decomp = {
        torch.ops.aten.dropout.default: lambda input, p, train: input,
    }
    for key in list(programs.keys()):
        if key != "preprocessor":
            programs[key] = programs[key].run_decompositions(_dropout_decomp)

    return programs, metadata


def lower_to_ggml(programs, metadata=None):
    """Lower exported programs to ExecuTorch with GGML backend."""
    print("\nLowering to ExecuTorch with GGML backend...")

    # Use GgmlPartitioner for encoder/decoder/joint; preprocessor stays portable
    partitioner = {}
    for key in programs.keys():
        if key == "preprocessor":
            partitioner[key] = []
        else:
            partitioner[key] = [GgmlPartitioner()]

    constant_methods = {}
    if metadata:
        for key, value in metadata.items():
            constant_methods[key] = value

    # Only apply ReplaceCopyOpsPass to GGML-partitioned methods (not preprocessor)
    ggml_transform = {
        key: [ReplaceCopyOpsPass()] if key != "preprocessor" else []
        for key in programs.keys()
    }
    et_prog = to_edge_rewrite_and_lower(
        programs,
        transform_passes=ggml_transform,
        partitioner=partitioner,
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=True,
        ),
        constant_methods=constant_methods if constant_methods else None,
    )
    return et_prog.to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
        ),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Export Parakeet TDT to ExecuTorch with GGML backend"
    )
    parser.add_argument("--output-dir", default="./parakeet_ggml")
    parser.add_argument(
        "--audio", type=str, help="Path to audio file for transcription test"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Extracting tokenizer...")
    extract_tokenizer(args.output_dir)

    print("Loading model...")
    model = load_model()

    print("\nExporting components...")
    programs, metadata = export_all_ggml(model)

    et = lower_to_ggml(programs, metadata=metadata)

    pte_path = os.path.join(args.output_dir, "model.pte")
    print(f"\nSaving ExecuTorch program to: {pte_path}")
    with open(pte_path, "wb") as f:
        et.write_to_file(f)
    print(f"Saved {os.path.getsize(pte_path) / (1024 * 1024):.1f} MB")

    if args.audio:
        print("\n" + "=" * 60)
        print("Testing transcription...")
        print("=" * 60)

        print("\n[Eager PyTorch]")
        eager_text = transcribe_eager(args.audio, model)
        print(f"  Result: {eager_text}")

        print("\n[ExecuTorch + GGML Runtime]")
        from executorch.runtime import Runtime

        runtime = Runtime.get()
        program = runtime.load_program(et.buffer)

        with torch.no_grad():
            from export_parakeet_tdt import load_audio

            sample_rate = model.preprocessor._cfg.sample_rate
            audio = load_audio(args.audio, sample_rate=sample_rate)
            preprocessor_method = program.load_method("preprocessor")
            audio_1d = audio.squeeze(0)
            audio_len = torch.tensor([audio_1d.shape[0]], dtype=torch.int64)
            proc_result = preprocessor_method.execute([audio_1d, audio_len])
            mel = proc_result[0]
            mel_len = proc_result[1].item()

            encoder_method = program.load_method("encoder")
            mel_len_tensor = torch.tensor([mel_len], dtype=torch.int64)
            enc_result = encoder_method.execute([mel, mel_len_tensor])
            f_proj = enc_result[0]
            encoded_len = enc_result[1].item()

            tokens = greedy_decode_executorch(
                f_proj,
                encoded_len,
                program,
                blank_id=model.tokenizer.vocab_size,
                num_rnn_layers=model.decoder.pred_rnn_layers,
                pred_hidden=model.decoder.pred_hidden,
            )

            et_text = model.tokenizer.ids_to_text(tokens)
            print(f"  Result: {et_text}")

            if eager_text == et_text:
                print("\nTranscriptions match!")
            else:
                print("\nTranscriptions differ!")
                print(f"  Eager: {eager_text}")
                print(f"  GGML:  {et_text}")

    print("\nDone!")


if __name__ == "__main__":
    main()
