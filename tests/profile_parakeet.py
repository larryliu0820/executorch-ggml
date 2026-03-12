#!/usr/bin/env python3
"""Profile Parakeet on GGML backend — method-level + per-op timing."""
import os, sys, time
import torch
import numpy as np

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _repo_root)
_parakeet_pkg = "executorch.examples.models.parakeet.quantize"
if _parakeet_pkg not in sys.modules:
    import types; _stub = types.ModuleType(_parakeet_pkg); _stub.quantize_model_ = lambda *a, **kw: None; sys.modules[_parakeet_pkg] = _stub
sys.path.insert(0, os.path.join(_repo_root, "third-party", "executorch", "examples", "models", "parakeet"))

from export_parakeet_tdt import greedy_decode_executorch, load_model
import executorch_ggml  # noqa

print("Loading model...", flush=True)
model = load_model()

import scipy.io.wavfile as wavfile
sr_file, audio_np = wavfile.read(os.path.join(_repo_root, "test_audio.wav"))
if audio_np.dtype == np.int16:
    audio_np = audio_np.astype(np.float32) / 32768.0
audio_tensor = torch.from_numpy(audio_np).float()

from executorch.runtime import Runtime
runtime = Runtime.get()

for label, pte_path in [
    ("F32", os.path.join(_repo_root, "parakeet_ggml", "model.pte")),
    ("Q8_0", os.path.join(_repo_root, "parakeet_ggml", "model_q8_0.pte")),
]:
    if not os.path.exists(pte_path):
        continue
    size_mb = os.path.getsize(pte_path) / (1024 * 1024)
    print(f"\n{'='*60}", flush=True)
    print(f"  {label} ({size_mb:.0f} MB)", flush=True)
    print(f"{'='*60}", flush=True)

    with open(pte_path, "rb") as f:
        prog = runtime.load_program(f.read())

    # Warmup
    with torch.no_grad():
        pre = prog.load_method("preprocessor")
        audio_len = torch.tensor([audio_tensor.shape[0]], dtype=torch.int64)
        proc = pre.execute([audio_tensor, audio_len])
        mel, mel_len_val = proc[0], proc[1].item()

        enc = prog.load_method("encoder")
        mel_len_t = torch.tensor([mel_len_val], dtype=torch.int64)
        enc_result = enc.execute([mel, mel_len_t])

    # Timed runs (3 iterations)
    n_runs = 3
    pre_times, enc_times, dec_times, total_times = [], [], [], []

    for run in range(n_runs):
        t_total_start = time.perf_counter()

        with torch.no_grad():
            # Preprocessor
            pre = prog.load_method("preprocessor")
            t0 = time.perf_counter()
            proc = pre.execute([audio_tensor, audio_len])
            t1 = time.perf_counter()
            mel, mel_len_val = proc[0], proc[1].item()
            pre_times.append(t1 - t0)

            # Encoder
            enc = prog.load_method("encoder")
            mel_len_t = torch.tensor([mel_len_val], dtype=torch.int64)
            t0 = time.perf_counter()
            enc_result = enc.execute([mel, mel_len_t])
            t1 = time.perf_counter()
            f_proj = enc_result[0]
            encoded_len = enc_result[1].item()
            enc_times.append(t1 - t0)

            # Decoder (greedy TDT decode — many sequential steps)
            t0 = time.perf_counter()
            tokens = greedy_decode_executorch(
                f_proj, encoded_len, prog,
                blank_id=model.tokenizer.vocab_size,
                num_rnn_layers=model.decoder.pred_rnn_layers,
                pred_hidden=model.decoder.pred_hidden,
            )
            t1 = time.perf_counter()
            dec_times.append(t1 - t0)

        t_total_end = time.perf_counter()
        total_times.append(t_total_end - t_total_start)

    text = model.tokenizer.ids_to_text(tokens)

    # Count decoder steps
    # Re-run to count
    n_decoder_steps = 0
    with torch.no_grad():
        dec_m = prog.load_method("decoder_step")
        joint_m = prog.load_method("joint")
        h = torch.zeros(model.decoder.pred_rnn_layers, 1, model.decoder.pred_hidden)
        c = torch.zeros(model.decoder.pred_rnn_layers, 1, model.decoder.pred_hidden)
        sos = torch.tensor([[model.tokenizer.vocab_size]], dtype=torch.long)
        r = dec_m.execute([sos, h, c])
        g_proj, h, c = r[0], r[1], r[2]
        n_decoder_steps += 1

        t_pos = 0
        durations = [0, 1, 2, 3, 4]
        dec_step_times = []
        joint_step_times = []
        while t_pos < encoded_len:
            f_t = f_proj[:, t_pos:t_pos+1, :].contiguous()

            t0 = time.perf_counter()
            jr = joint_m.execute([f_t, g_proj])
            t1 = time.perf_counter()
            joint_step_times.append(t1 - t0)

            k = jr[0].item()
            dur = durations[jr[1].item()]
            n_decoder_steps += 1

            if k == model.tokenizer.vocab_size:
                t_pos += max(dur, 1)
            else:
                token = torch.tensor([[k]], dtype=torch.long)
                t0 = time.perf_counter()
                dr = dec_m.execute([token, h, c])
                t1 = time.perf_counter()
                dec_step_times.append(t1 - t0)
                g_proj, h, c = dr[0], dr[1], dr[2]
                n_decoder_steps += 1
                t_pos += dur
                if dur == 0:
                    pass  # simplified

    avg = lambda l: sum(l) / len(l) if l else 0

    print(f"\n  Text: {text}", flush=True)
    print(f"\n  Method timing (avg of {n_runs} runs):", flush=True)
    print(f"    Preprocessor:  {avg(pre_times)*1000:7.1f} ms", flush=True)
    print(f"    Encoder:       {avg(enc_times)*1000:7.1f} ms", flush=True)
    print(f"    Decoder total: {avg(dec_times)*1000:7.1f} ms", flush=True)
    print(f"    Total:         {avg(total_times)*1000:7.1f} ms", flush=True)
    print(f"\n  Decoder breakdown:", flush=True)
    print(f"    Total steps:     {n_decoder_steps}", flush=True)
    print(f"    Joint calls:     {len(joint_step_times)}", flush=True)
    print(f"    Decoder calls:   {len(dec_step_times)}", flush=True)
    print(f"    Avg joint time:  {avg(joint_step_times)*1000:.3f} ms/step", flush=True)
    print(f"    Avg decoder time:{avg(dec_step_times)*1000:.3f} ms/step", flush=True)
    print(f"    Joint total:     {sum(joint_step_times)*1000:.1f} ms", flush=True)
    print(f"    Decoder total:   {sum(dec_step_times)*1000:.1f} ms", flush=True)
