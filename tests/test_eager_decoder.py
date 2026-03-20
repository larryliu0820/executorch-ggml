#!/usr/bin/env python3
"""Test: GGML encoder + eager decoder.

Requires NeMo model weights and parakeet_ggml/model.pte.
Run directly: python tests/test_eager_decoder.py
"""
import os, sys
import pytest
import torch
import numpy as np

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_model_path = os.path.join(_repo_root, "parakeet_ggml", "model.pte")

if not os.path.exists(_model_path):
    pytest.skip(f"Model not found: {_model_path}", allow_module_level=True)

sys.path.insert(0, _repo_root)
_parakeet_pkg = "executorch.examples.models.parakeet.quantize"
if _parakeet_pkg not in sys.modules:
    import types; _stub = types.ModuleType(_parakeet_pkg); _stub.quantize_model_ = lambda *a, **kw: None; sys.modules[_parakeet_pkg] = _stub
sys.path.insert(0, os.path.join(_repo_root, "third-party", "executorch", "examples", "models", "parakeet"))

from export_parakeet_tdt import greedy_decode_eager, load_model
import executorch_ggml  # noqa

print("Loading model...", flush=True)
model = load_model()

import scipy.io.wavfile as wavfile
sr_file, audio_np = wavfile.read(os.path.join(_repo_root, "test_audio.wav"))
if audio_np.dtype == np.int16:
    audio_np = audio_np.astype(np.float32) / 32768.0
audio_tensor = torch.from_numpy(audio_np).float()
sample_rate = model.preprocessor._cfg.sample_rate

from executorch.runtime import Runtime
runtime = Runtime.get()
f32_path = os.path.join(_repo_root, "parakeet_ggml", "model.pte")
with open(f32_path, "rb") as f:
    prog = runtime.load_program(f.read())

with torch.no_grad():
    # GGML preprocessor
    pre = prog.load_method("preprocessor")
    audio_len = torch.tensor([audio_tensor.shape[0]], dtype=torch.int64)
    proc = pre.execute([audio_tensor, audio_len])
    mel, mel_len = proc[0], proc[1].item()

    # GGML encoder
    enc = prog.load_method("encoder")
    enc_result = enc.execute([mel, torch.tensor([mel_len], dtype=torch.int64)])
    f_proj_ggml = enc_result[0]
    enc_len_ggml = enc_result[1].item()
    print(f"GGML encoder: f_proj={f_proj_ggml.shape} enc_len={enc_len_ggml}", flush=True)

    # Eager encoder for comparison
    waveform_2d = audio_tensor.unsqueeze(0)
    mel_eager, ml = model.preprocessor(input_signal=waveform_2d, length=torch.tensor([waveform_2d.shape[1]]))
    encoded_eager, el_eager = model.encoder(audio_signal=mel_eager, length=ml)
    f_proj_eager = model.joint.project_encoder(encoded_eager.transpose(1, 2))
    print(f"Eager encoder: f_proj={f_proj_eager.shape} enc_len={el_eager.item()}", flush=True)

    # Compare
    diff = (f_proj_eager - f_proj_ggml).abs()
    print(f"Encoder diff: max={diff.max().item():.6f} mean={diff.mean().item():.6f}", flush=True)

    # Eager decoder with GGML encoder output
    # Need to un-project: the eager decoder expects raw encoder output, not projected
    # Actually greedy_decode_eager expects (encoder_output, encoded_lengths, model)
    # where encoder_output is the RAW encoder output (not projected)
    # So let's use f_proj directly with the eager TDT decoder
    print(f"\n[Eager decoder + Eager encoder]", flush=True)
    tokens = greedy_decode_eager(encoded_eager, el_eager, model)
    print(f"  {model.tokenizer.ids_to_text(tokens)}", flush=True)

    # For GGML encoder -> eager decoder, we need the raw (unprojected) encoder output
    # But GGML only gives us the projected output. Let's just compare the projected outputs
    # and use the eager decoder's own greedy_decode which works on unprojected output.

    # Actually, let's do it manually with the projected output
    # The TDT decoder needs: project_encoder output (f_proj) and project_prednet output (g_proj)
    print(f"\n[Eager decoder + GGML encoder (manual TDT decode)]", flush=True)
    durations = [0, 1, 2, 3, 4]
    hypothesis = []
    blank_id = model.tokenizer.vocab_size

    # Initialize decoder
    h = torch.zeros(model.decoder.pred_rnn_layers, 1, model.decoder.pred_hidden)
    c = torch.zeros(model.decoder.pred_rnn_layers, 1, model.decoder.pred_hidden)

    # SOS
    sos_token = torch.tensor([[blank_id]], dtype=torch.long)
    g, new_state = model.decoder.predict(y=sos_token, state=[h, c], add_sos=False)
    g_proj = model.joint.project_prednet(g)
    h, c = new_state[0], new_state[1]

    t = 0
    symbols_on_frame = 0
    max_symbols_per_step = 10

    while t < enc_len_ggml:
        f_t = f_proj_ggml[:, t:t+1, :].contiguous()
        logits = model.joint.joint_after_projection(f_t, g_proj).squeeze()
        num_token_classes = model.tokenizer.vocab_size + 1
        k = logits[:num_token_classes].argmax().item()
        dur_idx = logits[num_token_classes:].argmax().item()
        dur = durations[dur_idx]

        if k == blank_id:
            t += max(dur, 1)
            symbols_on_frame = 0
        else:
            hypothesis.append(k)
            token = torch.tensor([[k]], dtype=torch.long)
            g, new_state = model.decoder.predict(y=token, state=[h, c], add_sos=False)
            g_proj = model.joint.project_prednet(g)
            h, c = new_state[0], new_state[1]
            t += dur
            if dur == 0:
                symbols_on_frame += 1
                if symbols_on_frame >= max_symbols_per_step:
                    t += 1
                    symbols_on_frame = 0
            else:
                symbols_on_frame = 0

    text = model.tokenizer.ids_to_text(hypothesis)
    print(f"  {text}", flush=True)
