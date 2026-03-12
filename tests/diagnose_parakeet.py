#!/usr/bin/env python3
"""Diagnose Parakeet GGML backend by comparing each component vs eager."""

import os, sys
import torch
import numpy as np

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _repo_root)

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
from export_parakeet_tdt import (
    EncoderWithProjection, DecoderStep, JointWithArgmax,
    PreprocessorWrapper, greedy_decode_executorch, load_model,
)

import executorch_ggml  # noqa

def compare_tensors(name, eager_t, ggml_t):
    if eager_t.shape != ggml_t.shape:
        print(f"  {name}: SHAPE MISMATCH eager={eager_t.shape} ggml={ggml_t.shape}")
        return
    diff = (eager_t.float() - ggml_t.float()).abs()
    cos = torch.nn.functional.cosine_similarity(
        eager_t.float().flatten().unsqueeze(0),
        ggml_t.float().flatten().unsqueeze(0),
    ).item()
    print(f"  {name}: shape={eager_t.shape}")
    print(f"    max_diff={diff.max().item():.6f}  mean_diff={diff.mean().item():.6f}  cos_sim={cos:.6f}")
    print(f"    eager: mean={eager_t.float().mean().item():.6f} std={eager_t.float().std().item():.6f} min={eager_t.float().min().item():.6f} max={eager_t.float().max().item():.6f}")
    print(f"    ggml:  mean={ggml_t.float().mean().item():.6f} std={ggml_t.float().std().item():.6f} min={ggml_t.float().min().item():.6f} max={ggml_t.float().max().item():.6f}")
    # Check for NaN/Inf
    if torch.isnan(ggml_t).any():
        print(f"    WARNING: NaN detected in ggml output!")
    if torch.isinf(ggml_t).any():
        print(f"    WARNING: Inf detected in ggml output!")

def main():
    import scipy.io.wavfile as wavfile
    audio_path = os.path.join(_repo_root, "test_audio.wav")
    f32_path = os.path.join(_repo_root, "parakeet_ggml", "model.pte")

    print("Loading NeMo model...")
    model = load_model()

    # Load audio
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

    print(f"Audio: {audio_tensor.shape[0]} samples ({audio_tensor.shape[0]/sample_rate:.1f}s)")

    # ===================== EAGER =====================
    print("\n" + "="*60)
    print("EAGER BASELINE")
    print("="*60)
    with torch.no_grad():
        # Preprocessor
        pre_wrapper = PreprocessorWrapper(model.preprocessor)
        pre_wrapper.eval()
        audio_len = torch.tensor([audio_tensor.shape[0]], dtype=torch.int64)
        eager_mel, eager_mel_len = pre_wrapper(audio_tensor, audio_len)
        print(f"Preprocessor: mel={eager_mel.shape} mel_len={eager_mel_len.item()}")

        # Encoder
        enc_proj = EncoderWithProjection(model.encoder, model.joint)
        enc_proj.eval()
        eager_f_proj, eager_enc_len = enc_proj(
            audio_signal=eager_mel,
            length=eager_mel_len.unsqueeze(0) if eager_mel_len.dim() == 0 else eager_mel_len,
        )
        print(f"Encoder: f_proj={eager_f_proj.shape} enc_len={eager_enc_len.item()}")

    # ===================== GGML (F32) =====================
    print("\n" + "="*60)
    print("GGML F32")
    print("="*60)

    from executorch.runtime import Runtime
    runtime = Runtime.get()

    with open(f32_path, "rb") as f:
        prog = runtime.load_program(f.read())

    with torch.no_grad():
        # Preprocessor (not delegated)
        pre_method = prog.load_method("preprocessor")
        proc_result = pre_method.execute([audio_tensor, audio_len])
        ggml_mel = proc_result[0]
        ggml_mel_len = proc_result[1]
        print(f"Preprocessor: mel={ggml_mel.shape} mel_len={ggml_mel_len.item()}")

    print("\n--- Preprocessor comparison ---")
    compare_tensors("mel", eager_mel, ggml_mel)
    print(f"  mel_len: eager={eager_mel_len.item()} ggml={ggml_mel_len.item()}")

    with torch.no_grad():
        # Encoder with GGML preprocessor output
        enc_method = prog.load_method("encoder")
        mel_len_t = torch.tensor([ggml_mel_len.item()], dtype=torch.int64)
        enc_result = enc_method.execute([ggml_mel, mel_len_t])
        ggml_f_proj = enc_result[0]
        ggml_enc_len = enc_result[1]
        print(f"\nEncoder: f_proj={ggml_f_proj.shape} enc_len={ggml_enc_len.item()}")

    print("\n--- Encoder comparison ---")
    compare_tensors("f_proj", eager_f_proj, ggml_f_proj)
    print(f"  enc_len: eager={eager_enc_len.item()} ggml={ggml_enc_len.item()}")

    # Test decoder with EAGER encoder output (to isolate encoder vs decoder)
    print("\n" + "="*60)
    print("DECODER TEST: using EAGER encoder output with GGML decoder")
    print("="*60)
    with torch.no_grad():
        eager_tokens = greedy_decode_executorch(
            eager_f_proj, int(eager_enc_len.item()), prog,
            blank_id=model.tokenizer.vocab_size,
            num_rnn_layers=model.decoder.pred_rnn_layers,
            pred_hidden=model.decoder.pred_hidden,
        )
        eager_in_text = model.tokenizer.ids_to_text(eager_tokens)
        print(f"  Result: {eager_in_text}")

    # Test decoder with GGML encoder output
    print("\n" + "="*60)
    print("DECODER TEST: using GGML encoder output with GGML decoder")
    print("="*60)
    with torch.no_grad():
        ggml_tokens = greedy_decode_executorch(
            ggml_f_proj, int(ggml_enc_len.item()), prog,
            blank_id=model.tokenizer.vocab_size,
            num_rnn_layers=model.decoder.pred_rnn_layers,
            pred_hidden=model.decoder.pred_hidden,
        )
        ggml_in_text = model.tokenizer.ids_to_text(ggml_tokens)
        print(f"  Result: {ggml_in_text}")

    # Also test: GGML encoder -> single joint step to see raw logits
    print("\n" + "="*60)
    print("SINGLE JOINT STEP ANALYSIS")
    print("="*60)
    with torch.no_grad():
        dec_method = prog.load_method("decoder_step")
        joint_method = prog.load_method("joint")

        h = torch.zeros(model.decoder.pred_rnn_layers, 1, model.decoder.pred_hidden)
        c = torch.zeros(model.decoder.pred_rnn_layers, 1, model.decoder.pred_hidden)

        # SOS step
        sos = torch.tensor([[model.tokenizer.vocab_size]], dtype=torch.long)
        sos_result = dec_method.execute([sos, h, c])
        g_proj = sos_result[0]
        h, c = sos_result[1], sos_result[2]

        print(f"g_proj after SOS: shape={g_proj.shape}")
        print(f"  mean={g_proj.mean().item():.6f} std={g_proj.std().item():.6f}")

        # First frame with eager encoder output
        f_t_eager = eager_f_proj[:, 0:1, :].contiguous()
        joint_eager = joint_method.execute([f_t_eager, g_proj])
        print(f"\nJoint (eager enc, frame 0): token={joint_eager[0].item()} dur={joint_eager[1].item()}")

        # First frame with ggml encoder output
        f_t_ggml = ggml_f_proj[:, 0:1, :].contiguous()
        joint_ggml = joint_method.execute([f_t_ggml, g_proj])
        print(f"Joint (ggml enc, frame 0): token={joint_ggml[0].item()} dur={joint_ggml[1].item()}")


if __name__ == "__main__":
    main()
