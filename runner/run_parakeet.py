#!/usr/bin/env python3
"""Run Parakeet transcription from a .pte model file.

Usage:
    source .venv/bin/activate
    python runner/run_parakeet.py --model parakeet_ggml/model_q8_0.pte [--audio test_audio.wav]
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
    parser.add_argument("--model", required=True, help="Path to .pte model file")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Model not found at {args.model}. Run export first.")
        return

    print("Loading NeMo model (for tokenizer)...")
    model = load_model()

    audio = load_wav(args.audio, target_sr=model.preprocessor._cfg.sample_rate)
    print(f"Audio: {args.audio} ({audio.shape[0] / model.preprocessor._cfg.sample_rate:.1f}s)\n")

    # Eager baseline
    print("--- Eager PyTorch ---")
    with torch.no_grad():
        waveform_2d = audio.unsqueeze(0)
        mel, mel_len = model.preprocessor(
            input_signal=waveform_2d,
            length=torch.tensor([waveform_2d.shape[1]])
        )
        encoded, encoded_len = model.encoder(audio_signal=mel, length=mel_len)
        from export_parakeet_tdt import greedy_decode_eager
        tokens = greedy_decode_eager(encoded, encoded_len, model)
        print(f"  {model.tokenizer.ids_to_text(tokens)}\n")

    # Free the NeMo eager model from GPU before loading the .pte program
    # to reclaim VRAM for the GGML backend.
    model.cpu()
    torch.cuda.empty_cache()

    from executorch.runtime import Runtime
    runtime = Runtime.get()

    label = os.path.splitext(os.path.basename(args.model))[0]
    size_mb = os.path.getsize(args.model) / (1024 * 1024)
    print(f"--- {label} ({size_mb:.0f} MB) ---")
    t0 = time.time()
    prog = runtime.load_program(open(args.model, "rb").read())
    text = transcribe(prog, audio, model)
    t1 = time.time()
    print(f"  {text}")
    print(f"  ({t1-t0:.1f}s)")


if __name__ == "__main__":
    main()
