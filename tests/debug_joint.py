#!/usr/bin/env python3
"""Compare GGML joint logits vs eager at the first decoding step."""
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

# Get eager encoder output
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

    # Eager decoder SOS
    h = torch.zeros(num_layers, 1, pred_hidden)
    c = torch.zeros(num_layers, 1, pred_hidden)
    sos = torch.tensor([[blank_id]], dtype=torch.long)
    g_eager, ns = model.decoder.predict(y=sos, state=[h, c], add_sos=False)
    g_proj_eager = model.joint.project_prednet(g_eager)
    h_eager, c_eager = ns[0], ns[1]

    # GGML decoder SOS
    h_g = torch.zeros(num_layers, 1, pred_hidden)
    c_g = torch.zeros(num_layers, 1, pred_hidden)
    dec = prog.load_method("decoder_step")
    r = dec.execute([sos, h_g, c_g])
    g_proj_ggml = r[0]
    h_ggml, c_ggml = r[1], r[2]

    print("=== After SOS ===", flush=True)
    diff_g = (g_proj_eager - g_proj_ggml).abs()
    diff_h = (h_eager - h_ggml).abs()
    diff_c = (c_eager - c_ggml).abs()
    print(f"g_proj: max_diff={diff_g.max():.6f}", flush=True)
    print(f"h: max_diff={diff_h.max():.6f}", flush=True)
    print(f"c: max_diff={diff_c.max():.6f}", flush=True)

    # Joint at frame 0
    f_t = f_proj[:, 0:1, :].contiguous()

    # Eager joint (raw logits)
    logits_eager = model.joint.joint_after_projection(f_t, g_proj_eager).squeeze()
    tok_logits_eager = logits_eager[:num_token_classes]
    dur_logits_eager = logits_eager[num_token_classes:]
    print(f"\n=== Joint at t=0 (eager) ===", flush=True)
    print(f"Token argmax: {tok_logits_eager.argmax().item()} (blank={blank_id})", flush=True)
    print(f"Duration logits: {dur_logits_eager.tolist()}", flush=True)
    print(f"Duration argmax: {dur_logits_eager.argmax().item()}", flush=True)

    # GGML joint (returns argmax only)
    joint = prog.load_method("joint")
    jr = joint.execute([f_t, g_proj_eager])  # Use EAGER g_proj to isolate joint
    print(f"\n=== Joint at t=0 (GGML, with eager g_proj) ===", flush=True)
    print(f"Token: {jr[0].item()}, Duration idx: {jr[1].item()}", flush=True)

    # Also try with GGML g_proj
    jr2 = joint.execute([f_t, g_proj_ggml])
    print(f"\n=== Joint at t=0 (GGML, with ggml g_proj) ===", flush=True)
    print(f"Token: {jr2[0].item()}, Duration idx: {jr2[1].item()}", flush=True)

    # Let's look at more frames
    print(f"\n=== Duration comparison for first 10 frames ===", flush=True)
    for t in range(min(10, enc_len)):
        f_t = f_proj[:, t:t+1, :].contiguous()
        logits = model.joint.joint_after_projection(f_t, g_proj_eager).squeeze()
        dur_eager = logits[num_token_classes:].argmax().item()
        tok_eager = logits[:num_token_classes].argmax().item()

        jr = joint.execute([f_t, g_proj_eager])
        tok_ggml = jr[0].item()
        dur_ggml = jr[1].item()

        match = "OK" if (tok_eager == tok_ggml and dur_eager == dur_ggml) else "DIFF"
        print(f"  t={t}: eager(tok={tok_eager}, dur={dur_eager}) "
              f"ggml(tok={tok_ggml}, dur={dur_ggml}) {match}", flush=True)
        if dur_eager != dur_ggml:
            print(f"    eager dur_logits: {logits[num_token_classes:].tolist()}", flush=True)
