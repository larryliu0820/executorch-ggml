#!/usr/bin/env python3
"""Debug GGML decoder_step and joint vs eager, step by step."""
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


def compare(name, a, b):
    if a.shape != b.shape:
        print(f"  {name}: SHAPE MISMATCH {a.shape} vs {b.shape}")
        return
    diff = (a.float() - b.float()).abs()
    cos = torch.nn.functional.cosine_similarity(
        a.float().flatten().unsqueeze(0), b.float().flatten().unsqueeze(0)
    ).item()
    print(f"  {name}: max_diff={diff.max().item():.6f} mean_diff={diff.mean().item():.6f} cos={cos:.6f}")
    if torch.isnan(b).any():
        print(f"    WARNING: NaN!")


print("Loading model...", flush=True)
model = load_model()

from executorch.runtime import Runtime
runtime = Runtime.get()
f32_path = os.path.join(_repo_root, "parakeet_ggml", "model.pte")
with open(f32_path, "rb") as f:
    prog = runtime.load_program(f.read())

blank_id = model.tokenizer.vocab_size
num_layers = model.decoder.pred_rnn_layers
pred_hidden = model.decoder.pred_hidden
durations = [0, 1, 2, 3, 4]
num_token_classes = model.tokenizer.vocab_size + 1

# Get encoder output (use eager for clean input)
import scipy.io.wavfile as wavfile
sr_file, audio_np = wavfile.read(os.path.join(_repo_root, "test_audio.wav"))
if audio_np.dtype == np.int16:
    audio_np = audio_np.astype(np.float32) / 32768.0
audio_tensor = torch.from_numpy(audio_np).float()

with torch.no_grad():
    waveform_2d = audio_tensor.unsqueeze(0)
    mel_eager, ml = model.preprocessor(input_signal=waveform_2d, length=torch.tensor([waveform_2d.shape[1]]))
    encoded_eager, el_eager = model.encoder(audio_signal=mel_eager, length=ml)
    f_proj_eager = model.joint.project_encoder(encoded_eager.transpose(1, 2))
    enc_len = int(el_eager.item())
    print(f"Encoder: f_proj={f_proj_eager.shape} enc_len={enc_len}", flush=True)

# ======== Compare decoder_step ========
print("\n=== DECODER STEP COMPARISON ===", flush=True)

with torch.no_grad():
    # Initialize states
    h_eager = torch.zeros(num_layers, 1, pred_hidden)
    c_eager = torch.zeros(num_layers, 1, pred_hidden)
    h_ggml = h_eager.clone()
    c_ggml = c_eager.clone()

    dec_ggml = prog.load_method("decoder_step")
    joint_ggml = prog.load_method("joint")

    # SOS step
    sos = torch.tensor([[blank_id]], dtype=torch.long)

    # Eager decoder step
    g_eager, new_state_eager = model.decoder.predict(y=sos, state=[h_eager, c_eager], add_sos=False)
    g_proj_eager = model.joint.project_prednet(g_eager)
    h_eager, c_eager = new_state_eager[0], new_state_eager[1]

    # GGML decoder step
    sos_result = dec_ggml.execute([sos, h_ggml, c_ggml])
    g_proj_ggml = sos_result[0]
    h_ggml, c_ggml = sos_result[1], sos_result[2]

    print("After SOS:", flush=True)
    compare("g_proj", g_proj_eager, g_proj_ggml)
    compare("h", h_eager, h_ggml)
    compare("c", c_eager, c_ggml)

    # ======== Compare joint ========
    print("\n=== JOINT COMPARISON (first 5 frames) ===", flush=True)

    for t in range(min(5, enc_len)):
        f_t = f_proj_eager[:, t:t+1, :].contiguous()

        # Eager joint
        logits_eager = model.joint.joint_after_projection(f_t, g_proj_eager).squeeze()
        k_eager = logits_eager[:num_token_classes].argmax().item()
        dur_idx_eager = logits_eager[num_token_classes:].argmax().item()

        # GGML joint
        joint_result = joint_ggml.execute([f_t, g_proj_ggml])
        k_ggml = joint_result[0].item()
        dur_idx_ggml = joint_result[1].item()

        match = "OK" if k_eager == k_ggml else "MISMATCH"
        print(f"  t={t}: eager=(tok={k_eager}, dur={dur_idx_eager}) "
              f"ggml=(tok={k_ggml}, dur={dur_idx_ggml}) {match}", flush=True)

        # If there's a mismatch, compare logits
        if k_eager != k_ggml:
            print(f"    eager logits: top5={logits_eager[:num_token_classes].topk(5)}", flush=True)
            # We can't easily get raw logits from GGML joint (it returns argmax)
            # But we can check if g_proj diverged
            compare("    g_proj at mismatch", g_proj_eager, g_proj_ggml)
            break

        # Advance if non-blank
        if k_eager != blank_id:
            token = torch.tensor([[k_eager]], dtype=torch.long)

            g_e, ns_e = model.decoder.predict(y=token, state=[h_eager, c_eager], add_sos=False)
            g_proj_eager = model.joint.project_prednet(g_e)
            h_eager, c_eager = ns_e[0], ns_e[1]

            dec_result = dec_ggml.execute([token, h_ggml, c_ggml])
            g_proj_ggml = dec_result[0]
            h_ggml, c_ggml = dec_result[1], dec_result[2]

            compare(f"    g_proj after tok={k_eager}", g_proj_eager, g_proj_ggml)
            compare(f"    h after tok={k_eager}", h_eager, h_ggml)
            compare(f"    c after tok={k_eager}", c_eager, c_ggml)

    # ======== Full greedy decode comparison (step by step) ========
    print("\n=== FULL DECODE: first 20 steps ===", flush=True)

    # Reset states
    h_eager = torch.zeros(num_layers, 1, pred_hidden)
    c_eager = torch.zeros(num_layers, 1, pred_hidden)
    h_ggml = h_eager.clone()
    c_ggml = c_eager.clone()

    # SOS
    g_e, ns_e = model.decoder.predict(y=sos, state=[h_eager, c_eager], add_sos=False)
    g_proj_eager = model.joint.project_prednet(g_e)
    h_eager, c_eager = ns_e[0], ns_e[1]

    dec_ggml2 = prog.load_method("decoder_step")
    joint_ggml2 = prog.load_method("joint")
    sos_r = dec_ggml2.execute([sos, h_ggml, c_ggml])
    g_proj_ggml = sos_r[0]
    h_ggml, c_ggml = sos_r[1], sos_r[2]

    t_eager = 0
    t_ggml = 0
    tokens_eager = []
    tokens_ggml = []
    steps = 0
    sym_eager = 0
    sym_ggml = 0

    while (t_eager < enc_len or t_ggml < enc_len) and steps < 20:
        steps += 1

        # Eager step
        if t_eager < enc_len:
            f_t_e = f_proj_eager[:, t_eager:t_eager+1, :].contiguous()
            logits_e = model.joint.joint_after_projection(f_t_e, g_proj_eager).squeeze()
            k_e = logits_e[:num_token_classes].argmax().item()
            d_e = durations[logits_e[num_token_classes:].argmax().item()]
        else:
            k_e = blank_id
            d_e = 1

        # GGML step
        if t_ggml < enc_len:
            f_t_g = f_proj_eager[:, t_ggml:t_ggml+1, :].contiguous()
            jr = joint_ggml2.execute([f_t_g, g_proj_ggml])
            k_g = jr[0].item()
            d_g = durations[jr[1].item()]
        else:
            k_g = blank_id
            d_g = 1

        match = "OK" if k_e == k_g else "DIFF"
        print(f"  step {steps:2d}: t_e={t_eager:3d} t_g={t_ggml:3d} "
              f"eager=(tok={k_e:5d}, dur={d_e}) "
              f"ggml=(tok={k_g:5d}, dur={d_g}) {match}", flush=True)

        # Advance eager
        if k_e == blank_id:
            t_eager += max(d_e, 1); sym_eager = 0
        else:
            tokens_eager.append(k_e)
            token_e = torch.tensor([[k_e]], dtype=torch.long)
            g_e, ns_e = model.decoder.predict(y=token_e, state=[h_eager, c_eager], add_sos=False)
            g_proj_eager = model.joint.project_prednet(g_e)
            h_eager, c_eager = ns_e[0], ns_e[1]
            t_eager += d_e
            if d_e == 0:
                sym_eager += 1
                if sym_eager >= 10: t_eager += 1; sym_eager = 0
            else:
                sym_eager = 0

        # Advance ggml
        if k_g == blank_id:
            t_ggml += max(d_g, 1); sym_ggml = 0
        else:
            tokens_ggml.append(k_g)
            token_g = torch.tensor([[k_g]], dtype=torch.long)
            dr = dec_ggml2.execute([token_g, h_ggml, c_ggml])
            g_proj_ggml = dr[0]
            h_ggml, c_ggml = dr[1], dr[2]
            t_ggml += d_g
            if d_g == 0:
                sym_ggml += 1
                if sym_ggml >= 10: t_ggml += 1; sym_ggml = 0
            else:
                sym_ggml = 0

    print(f"\nEager tokens: {tokens_eager}", flush=True)
    print(f"GGML  tokens: {tokens_ggml}", flush=True)
    print(f"Eager text: {model.tokenizer.ids_to_text(tokens_eager)}", flush=True)
    print(f"GGML  text: {model.tokenizer.ids_to_text(tokens_ggml)}", flush=True)
