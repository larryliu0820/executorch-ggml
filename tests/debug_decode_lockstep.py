#!/usr/bin/env python3
"""Lockstep decode: feed same inputs to eager and GGML decoder, find first divergence."""
import os, sys
import torch
import numpy as np

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _repo_root)
_parakeet_pkg = "executorch.examples.models.parakeet.quantize"
if _parakeet_pkg not in sys.modules:
    import types; _stub = types.ModuleType(_parakeet_pkg); _stub.quantize_model_ = lambda *a, **kw: None; sys.modules[_parakeet_pkg] = _stub
sys.path.insert(0, os.path.join(_repo_root, "third-party", "executorch", "examples", "models", "parakeet"))

from export_parakeet_tdt import load_model
import executorch_ggml  # noqa

print("Loading model...", flush=True)
model = load_model()

from executorch.runtime import Runtime
runtime = Runtime.get()
with open(os.path.join(_repo_root, "parakeet_ggml", "model.pte"), "rb") as f:
    prog = runtime.load_program(f.read())

blank_id = model.tokenizer.vocab_size
num_layers = model.decoder.pred_rnn_layers
pred_hidden = model.decoder.pred_hidden
num_token_classes = model.tokenizer.vocab_size + 1
durations = [0, 1, 2, 3, 4]

# Get encoder output
import scipy.io.wavfile as wavfile
sr_file, audio_np = wavfile.read(os.path.join(_repo_root, "test_audio.wav"))
if audio_np.dtype == np.int16:
    audio_np = audio_np.astype(np.float32) / 32768.0
audio_tensor = torch.from_numpy(audio_np).float()

with torch.no_grad():
    waveform_2d = audio_tensor.unsqueeze(0)
    mel_eager, ml = model.preprocessor(input_signal=waveform_2d, length=torch.tensor([waveform_2d.shape[1]]))
    encoded_eager, el_eager = model.encoder(audio_signal=mel_eager, length=ml)
    f_proj = model.joint.project_encoder(encoded_eager.transpose(1, 2))
    enc_len = int(el_eager.item())

    # Init both decoders
    h_e = torch.zeros(num_layers, 1, pred_hidden)
    c_e = torch.zeros(num_layers, 1, pred_hidden)
    h_g = torch.zeros(num_layers, 1, pred_hidden)
    c_g = torch.zeros(num_layers, 1, pred_hidden)

    dec_ggml = prog.load_method("decoder_step")
    joint_ggml = prog.load_method("joint")

    # SOS
    sos = torch.tensor([[blank_id]], dtype=torch.long)
    g_e, ns_e = model.decoder.predict(y=sos, state=[h_e, c_e], add_sos=False)
    g_proj_e = model.joint.project_prednet(g_e)
    h_e, c_e = ns_e[0], ns_e[1]

    r = dec_ggml.execute([sos, h_g, c_g])
    g_proj_g = r[0]
    h_g, c_g = r[1], r[2]

    print(f"After SOS: g_proj diff={( g_proj_e - g_proj_g).abs().max():.6f} "
          f"h diff={(h_e - h_g).abs().max():.6f} c diff={(c_e - c_g).abs().max():.6f}", flush=True)

    # Greedy decode - lockstep (use eager decisions for both)
    t = 0
    tokens = []
    sym = 0
    step = 0

    print(f"\n=== Lockstep decode (eager controls, both execute) ===", flush=True)
    while t < enc_len and step < 50:
        step += 1
        f_t = f_proj[:, t:t+1, :].contiguous()

        # Eager joint
        logits_e = model.joint.joint_after_projection(f_t, g_proj_e).squeeze()
        k_e = logits_e[:num_token_classes].argmax().item()
        d_e = durations[logits_e[num_token_classes:].argmax().item()]

        # GGML joint
        jr = joint_ggml.execute([f_t, g_proj_g])
        k_g = jr[0].item()
        d_g = durations[jr[1].item()]

        tok_match = "tok_OK" if k_e == k_g else "tok_DIFF"
        dur_match = "dur_OK" if d_e == d_g else "dur_DIFF"

        if k_e != k_g or d_e != d_g:
            g_diff = (g_proj_e - g_proj_g).abs()
            print(f"  step {step:2d} t={t:3d}: eager(tok={k_e:5d},dur={d_e}) "
                  f"ggml(tok={k_g:5d},dur={d_g}) {tok_match} {dur_match} "
                  f"g_proj_diff={g_diff.max():.6f}", flush=True)
            if k_e != k_g:
                # Show top-5 token logits
                topk_e = logits_e[:num_token_classes].topk(3)
                print(f"    eager top3: {list(zip(topk_e.indices.tolist(), [f'{v:.3f}' for v in topk_e.values.tolist()]))}", flush=True)
                print(f"    eager dur_logits: {[f'{v:.3f}' for v in logits_e[num_token_classes:].tolist()]}", flush=True)

        # Use EAGER decision for both paths
        if k_e == blank_id:
            t += max(d_e, 1)
            sym = 0
        else:
            tokens.append(k_e)
            token = torch.tensor([[k_e]], dtype=torch.long)

            # Advance eager
            g_e2, ns_e2 = model.decoder.predict(y=token, state=[h_e, c_e], add_sos=False)
            g_proj_e = model.joint.project_prednet(g_e2)
            h_e, c_e = ns_e2[0], ns_e2[1]

            # Advance GGML
            r2 = dec_ggml.execute([token, h_g, c_g])
            g_proj_g = r2[0]
            h_g, c_g = r2[1], r2[2]

            # Compare after this step
            g_diff = (g_proj_e - g_proj_g).abs().max().item()
            h_diff = (h_e - h_g).abs().max().item()
            c_diff = (c_e - c_g).abs().max().item()
            if g_diff > 0.001 or h_diff > 0.001 or c_diff > 0.001:
                print(f"  step {step:2d} t={t:3d}: after tok={k_e}: "
                      f"g_diff={g_diff:.6f} h_diff={h_diff:.6f} c_diff={c_diff:.6f}", flush=True)

            t += d_e
            if d_e == 0:
                sym += 1
                if sym >= 10: t += 1; sym = 0
            else:
                sym = 0

    text = model.tokenizer.ids_to_text(tokens)
    print(f"\nTokens: {tokens[:30]}...", flush=True)
    print(f"Text: {text}", flush=True)
