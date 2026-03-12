#!/usr/bin/env python3
"""Export nvidia/parakeet-tdt-0.6b-v3 with Q8_0 quantization and test transcription.

Usage:
    source .venv/bin/activate
    python runner/export_parakeet_q8.py [--audio test_audio.wav]
"""

import argparse
import os
import sys
import time

import torch

# Add paths for upstream parakeet export
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _repo_root)

# Stub out the missing quantize import before importing export_parakeet_tdt
_parakeet_pkg = "executorch.examples.models.parakeet.quantize"
if _parakeet_pkg not in sys.modules:
    import types
    _stub = types.ModuleType(_parakeet_pkg)
    _stub.quantize_model_ = lambda *a, **kw: None
    sys.modules[_parakeet_pkg] = _stub

sys.path.insert(
    0,
    os.path.join(_repo_root, "third-party", "executorch", "examples", "models", "parakeet"),
)

from export_parakeet_ggml import export_all_ggml, load_model, extract_tokenizer
from export_parakeet_tdt import greedy_decode_executorch, transcribe_eager, load_audio

from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig
from executorch.exir.passes import MemoryPlanningPass

from executorch_ggml import GgmlPartitioner, GgmlQuantConfig, to_edge_rewrite_and_lower
from executorch_ggml.passes.replace_copy_ops_pass import ReplaceCopyOpsPass


def lower_to_ggml_q8(programs, metadata=None):
    """Lower exported programs to ExecuTorch with GGML backend + Q8_0."""
    print("\nLowering to ExecuTorch with GGML backend (Q8_0)...")

    quant_config = GgmlQuantConfig()

    partitioner = {}
    for key in programs.keys():
        if key == "preprocessor":
            partitioner[key] = []
        else:
            partitioner[key] = [GgmlPartitioner(quant_config=quant_config)]

    constant_methods = {}
    if metadata:
        for key, value in metadata.items():
            constant_methods[key] = value

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="./parakeet_ggml")
    parser.add_argument("--audio", type=str, default="test_audio.wav")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Extracting tokenizer...")
    extract_tokenizer(args.output_dir)

    print("Loading model...")
    model = load_model()

    print("\nExporting components...")
    programs, metadata = export_all_ggml(model)

    t0 = time.time()
    et = lower_to_ggml_q8(programs, metadata=metadata)
    t1 = time.time()
    print(f"Lowering took {t1 - t0:.1f}s")

    q8_path = os.path.join(args.output_dir, "model_q8_0.pte")
    print(f"\nSaving Q8_0 .pte to: {q8_path}")
    with open(q8_path, "wb") as f:
        et.write_to_file(f)
    q8_size = os.path.getsize(q8_path) / (1024 * 1024)
    print(f"Q8_0 size: {q8_size:.1f} MB")

    f32_path = os.path.join(args.output_dir, "model.pte")
    if os.path.exists(f32_path):
        f32_size = os.path.getsize(f32_path) / (1024 * 1024)
        print(f"F32 size:  {f32_size:.1f} MB")
        print(f"Compression: {f32_size / q8_size:.2f}x")

    if args.audio and os.path.exists(args.audio):
        print("\n" + "=" * 60)
        print("Testing transcription...")
        print("=" * 60)

        # Load audio with scipy (works reliably without torchcodec)
        import scipy.io.wavfile as wavfile
        import numpy as np
        sr_file, audio_np = wavfile.read(args.audio)
        if audio_np.dtype == np.int16:
            audio_np = audio_np.astype(np.float32) / 32768.0
        elif audio_np.dtype == np.int32:
            audio_np = audio_np.astype(np.float32) / 2147483648.0
        audio_tensor = torch.from_numpy(audio_np).float()
        sample_rate = model.preprocessor._cfg.sample_rate
        if sr_file != sample_rate:
            import torchaudio
            audio_tensor = torchaudio.functional.resample(
                audio_tensor.unsqueeze(0), sr_file, sample_rate
            ).squeeze(0)

        def _transcribe_et(program_obj, label):
            with torch.no_grad():
                preprocessor_method = program_obj.load_method("preprocessor")
                audio_1d = audio_tensor
                audio_len = torch.tensor([audio_1d.shape[0]], dtype=torch.int64)
                proc_result = preprocessor_method.execute([audio_1d, audio_len])
                mel = proc_result[0]
                mel_len = proc_result[1].item()

                encoder_method = program_obj.load_method("encoder")
                mel_len_tensor = torch.tensor([mel_len], dtype=torch.int64)
                enc_result = encoder_method.execute([mel, mel_len_tensor])
                f_proj = enc_result[0]
                encoded_len = enc_result[1].item()

                tokens = greedy_decode_executorch(
                    f_proj, encoded_len, program_obj,
                    blank_id=model.tokenizer.vocab_size,
                    num_rnn_layers=model.decoder.pred_rnn_layers,
                    pred_hidden=model.decoder.pred_hidden,
                )
                text = model.tokenizer.ids_to_text(tokens)
                print(f"\n[{label}]")
                print(f"  Result: {text}")
                return text

        from executorch.runtime import Runtime
        runtime = Runtime.get()

        q8_text = _transcribe_et(runtime.load_program(et.buffer), "Q8_0")

        if os.path.exists(f32_path):
            with open(f32_path, "rb") as f:
                f32_buf = f.read()
            f32_text = _transcribe_et(runtime.load_program(f32_buf), "F32")

            if q8_text.lower().strip() == f32_text.lower().strip():
                print("\nTranscriptions match!")
            else:
                print(f"\nTranscriptions differ.")

    print("\nDone!")


if __name__ == "__main__":
    main()
