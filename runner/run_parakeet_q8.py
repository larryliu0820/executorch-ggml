#!/usr/bin/env python3
"""Run Parakeet Q8_0 and F32 transcription comparison.

Usage:
    source .venv/bin/activate
    python runner/run_parakeet_q8.py [--audio test_audio.wav]
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torchaudio

import executorch_ggml  # noqa: F401  -- registers GgmlBackend with ET runtime

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _repo_root)

# Stub out missing quantize import
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
from export_parakeet_tdt import greedy_decode_executorch, load_model


def load_wav(path, target_sr=16000):
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
    return waveform.squeeze(0)  # return 1D for preprocessor


def transcribe(program, audio_tensor, model):
    with torch.no_grad():
        pre = program.load_method("preprocessor")
        audio_len = torch.tensor([audio_tensor.shape[0]], dtype=torch.int64)
        proc = pre.execute([audio_tensor, audio_len])
        mel, mel_len = proc[0], proc[1].item()

        enc = program.load_method("encoder")
        enc_result = enc.execute([mel, torch.tensor([mel_len], dtype=torch.int64)])
        f_proj, encoded_len = enc_result[0], enc_result[1].item()

        tokens = greedy_decode_executorch(
            f_proj, encoded_len, program,
            blank_id=model.tokenizer.vocab_size,
            num_rnn_layers=model.decoder.pred_rnn_layers,
            pred_hidden=model.decoder.pred_hidden,
        )
        return model.tokenizer.ids_to_text(tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", default="test_audio.wav")
    parser.add_argument("--model-dir", default="./parakeet_ggml")
    args = parser.parse_args()

    q8_path = os.path.join(args.model_dir, "model_q8_0.pte")
    f32_path = os.path.join(args.model_dir, "model.pte")

    if not os.path.exists(q8_path):
        print(f"Q8_0 model not found at {q8_path}. Run export first.")
        return

    print("Loading NeMo model (for tokenizer)...")
    model = load_model()

    audio = load_wav(args.audio, target_sr=model.preprocessor._cfg.sample_rate)
    print(f"Audio: {args.audio} ({audio.shape[0] / model.preprocessor._cfg.sample_rate:.1f}s)\n")

    # Eager baseline first
    print("--- Eager PyTorch ---")
    with torch.no_grad():
        waveform_2d = audio.unsqueeze(0)  # [1, samples]
        mel, mel_len = model.preprocessor(
            input_signal=waveform_2d,
            length=torch.tensor([waveform_2d.shape[1]])
        )
        encoded, encoded_len = model.encoder(audio_signal=mel, length=mel_len)
        from export_parakeet_tdt import greedy_decode_eager
        tokens = greedy_decode_eager(encoded, encoded_len, model)
        eager_text = model.tokenizer.ids_to_text(tokens)
        print(f"  {eager_text}\n")

    from executorch.runtime import Runtime
    runtime = Runtime.get()

    # Q8_0
    print(f"--- Q8_0 ({os.path.getsize(q8_path) / (1024*1024):.0f} MB) ---")
    t0 = time.time()
    prog_q8 = runtime.load_program(open(q8_path, "rb").read())
    q8_text = transcribe(prog_q8, audio, model)
    t1 = time.time()
    print(f"  {q8_text}")
    print(f"  ({t1-t0:.1f}s)")

    # F32
    if os.path.exists(f32_path):
        print(f"\n--- F32 ({os.path.getsize(f32_path) / (1024*1024):.0f} MB) ---")
        t0 = time.time()
        prog_f32 = runtime.load_program(open(f32_path, "rb").read())
        f32_text = transcribe(prog_f32, audio, model)
        t1 = time.time()
        print(f"  {f32_text}")
        print(f"  ({t1-t0:.1f}s)")

        if q8_text.strip().lower() == f32_text.strip().lower():
            print("\nTranscriptions match!")
        else:
            print("\nTranscriptions differ.")


if __name__ == "__main__":
    main()
