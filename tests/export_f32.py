#!/usr/bin/env python3
"""Export F32 Parakeet model and test transcription."""
import os, sys, time
import torch

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _repo_root)

_parakeet_pkg = "executorch.examples.models.parakeet.quantize"
if _parakeet_pkg not in sys.modules:
    import types; _stub = types.ModuleType(_parakeet_pkg); _stub.quantize_model_ = lambda *a, **kw: None; sys.modules[_parakeet_pkg] = _stub
sys.path.insert(0, os.path.join(_repo_root, "third-party", "executorch", "examples", "models", "parakeet"))

from export_parakeet_ggml import export_all_ggml, load_model, lower_to_ggml

import executorch_ggml  # noqa

print("Loading model...", flush=True)
model = load_model()

print("Exporting...", flush=True)
programs, metadata = export_all_ggml(model)

print("Lowering...", flush=True)
t0 = time.time()
et = lower_to_ggml(programs, metadata=metadata)
print(f"Lowering took {time.time()-t0:.1f}s", flush=True)

f32_path = os.path.join(_repo_root, "parakeet_ggml", "model.pte")
print(f"Saving to {f32_path}...", flush=True)
with open(f32_path, "wb") as f:
    et.write_to_file(f)
print(f"Size: {os.path.getsize(f32_path) / (1024*1024):.1f} MB", flush=True)

# Test
audio_path = os.path.join(_repo_root, "test_audio.wav")
if os.path.exists(audio_path):
    import scipy.io.wavfile as wavfile
    import numpy as np
    from export_parakeet_tdt import greedy_decode_executorch
    from executorch.runtime import Runtime

    sr_file, audio_np = wavfile.read(audio_path)
    if audio_np.dtype == np.int16:
        audio_np = audio_np.astype(np.float32) / 32768.0
    audio_tensor = torch.from_numpy(audio_np).float()
    sample_rate = model.preprocessor._cfg.sample_rate
    if sr_file != sample_rate:
        import torchaudio
        audio_tensor = torchaudio.functional.resample(
            audio_tensor.unsqueeze(0), sr_file, sample_rate
        ).squeeze(0)

    runtime = Runtime.get()
    prog = runtime.load_program(et.buffer)
    with torch.no_grad():
        pre = prog.load_method("preprocessor")
        audio_len = torch.tensor([audio_tensor.shape[0]], dtype=torch.int64)
        proc = pre.execute([audio_tensor, audio_len])
        mel, mel_len = proc[0], proc[1].item()

        enc = prog.load_method("encoder")
        enc_result = enc.execute([mel, torch.tensor([mel_len], dtype=torch.int64)])
        f_proj, encoded_len = enc_result[0], enc_result[1].item()

        tokens = greedy_decode_executorch(
            f_proj, encoded_len, prog,
            blank_id=model.tokenizer.vocab_size,
            num_rnn_layers=model.decoder.pred_rnn_layers,
            pred_hidden=model.decoder.pred_hidden,
        )
        text = model.tokenizer.ids_to_text(tokens)
        print(f"\n[F32 GGML] {text}", flush=True)

        # Also test eager
        waveform_2d = audio_tensor.unsqueeze(0)
        mel_eager, ml = model.preprocessor(input_signal=waveform_2d, length=torch.tensor([waveform_2d.shape[1]]))
        encoded, el = model.encoder(audio_signal=mel_eager, length=ml)
        from export_parakeet_tdt import greedy_decode_eager
        tokens_eager = greedy_decode_eager(encoded, el, model)
        eager_text = model.tokenizer.ids_to_text(tokens_eager)
        print(f"[Eager]    {eager_text}", flush=True)
        if text.strip().lower() == eager_text.strip().lower():
            print("MATCH!", flush=True)
        else:
            print("MISMATCH", flush=True)
