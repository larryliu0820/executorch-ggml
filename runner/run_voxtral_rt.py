#!/usr/bin/env python3
"""Run Voxtral-Mini-4B-Realtime-2602 inference with ExecuTorch GGML backend.

Usage:
    source .venv/bin/activate
    python runner/run_voxtral_rt.py --model voxtral_ggml/model_q8_0.pte \
        --model-path ~/models/Voxtral-Mini-4B-Realtime-2602 --audio test_audio.wav
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _repo_root)

# Must import executorch_ggml first for RTLD_GLOBAL symbol resolution
import executorch_ggml  # noqa: F401, I001  # isort: skip


def load_audio(audio_path: str, sample_rate: int = 16000) -> torch.Tensor:
    """Load audio file and resample to target sample rate. Returns (1, N_samples)."""
    import scipy.io.wavfile as wavfile

    sr_file, audio_np = wavfile.read(audio_path)
    if audio_np.dtype == np.int16:
        audio_np = audio_np.astype(np.float32) / 32768.0
    elif audio_np.dtype == np.int32:
        audio_np = audio_np.astype(np.float32) / 2147483648.0
    else:
        audio_np = audio_np.astype(np.float32)

    audio = torch.from_numpy(audio_np).float()
    if sr_file != sample_rate:
        import torchaudio
        audio = torchaudio.functional.resample(
            audio.unsqueeze(0), sr_file, sample_rate
        ).squeeze(0)

    return audio.unsqueeze(0)  # (1, N_samples)


def load_tokenizer(model_path: str):
    """Load Tekken tokenizer from the model directory."""
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

    tekken_path = os.path.join(model_path, "tekken.json")
    if os.path.exists(tekken_path):
        return MistralTokenizer.from_file(tekken_path)

    # Try loading from the model directory directly
    return MistralTokenizer.from_model("open-mistral-nemo")


def decode_tokens(tokenizer, token_ids: list) -> str:
    """Decode token IDs to text using the Tekken tokenizer."""
    from mistral_common.tokens.tokenizers.tekken import Tekkenizer

    if isinstance(tokenizer, Tekkenizer):
        return tokenizer.decode(token_ids)
    # MistralTokenizer wrapper
    inner = getattr(tokenizer, "instruct_tokenizer", tokenizer)
    inner = getattr(inner, "tokenizer", inner)
    return inner.decode(token_ids)


def compute_mel(waveform: torch.Tensor) -> torch.Tensor:
    """Compute log-mel spectrogram from waveform (1, N_samples) -> (1, 128, T_mel)."""
    import torchaudio

    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=400, hop_length=160, n_mels=128, power=2.0
    )
    mel = mel_spec(waveform)
    mel = torch.clamp(mel, min=1e-10).log10()
    mel = torch.maximum(mel, mel.max() - 8.0)
    mel = (mel + 4.0) / 4.0
    return mel


def run_offline(program, waveform: torch.Tensor, tokenizer, max_new_tokens: int = 512,
                preprocessor_program=None, model_dtype=torch.bfloat16):
    """Run offline (non-streaming) inference.

    Follows the Voxtral Realtime inference protocol: at each position,
    the decoder input is the element-wise SUM of audio_embeds[pos] and
    token_embedding(prev_token).  This matches the C++ runner.
    """

    # 1. Preprocessor: waveform -> mel spectrogram
    if preprocessor_program is not None:
        print("  Running preprocessor.pte...")
        preprocessor = preprocessor_program.load_method("forward")
        mel = preprocessor.execute([waveform])[0]
    else:
        print("  Computing mel spectrogram...")
        mel = compute_mel(waveform)
    print(f"    mel shape: {list(mel.shape)}")

    # Ensure T_mel is a multiple of 8 (required by encoder conv layers)
    T_mel = mel.shape[2]
    pad_to = ((T_mel + 7) // 8) * 8
    if pad_to > T_mel:
        mel = torch.nn.functional.pad(mel, (0, pad_to - T_mel))
    # 2. Audio encoder: mel -> audio embeddings
    print("  Running audio_encoder...")
    t0 = time.time()
    encoder = program.load_method("audio_encoder")
    mel = mel.to(model_dtype).contiguous()
    audio_embeds = encoder.execute([mel])[0]
    t1 = time.time()
    T_audio = audio_embeds.shape[1]
    print(f"    audio_embeds shape: {list(audio_embeds.shape)}, took {t1-t0:.2f}s")

    # 3. Autoregressive decode with audio+token embedding summation.
    # At each position:
    #   pos < T_audio: input = audio_embeds[pos] + token_embed(prev_token)
    #   pos >= T_audio: input = token_embed(prev_token)
    print("  Decoding (audio+token sum protocol)...")
    decoder = program.load_method("text_decoder")
    tok_embed_method = program.load_method("token_embedding")

    bos_id = 1  # Tekken BOS
    eos_id = 2  # Tekken EOS
    prev_token = torch.tensor([[bos_id]], dtype=torch.long)
    max_pos = min(max_new_tokens + T_audio, 4096)
    generated_tokens = []

    t0 = time.time()
    for pos in range(max_pos):
        # a. Token embedding for previous token
        tok_embed = tok_embed_method.execute([prev_token])[0]  # (1, 1, dim)

        # b. Sum with audio embedding if in audio range
        if pos < T_audio:
            audio_frame = audio_embeds[:, pos:pos+1, :]  # (1, 1, dim)
            input_embeds = (audio_frame.float() + tok_embed.float()).to(model_dtype)
        else:
            input_embeds = tok_embed

        # c. One decoder step
        cache_pos = torch.tensor([pos], dtype=torch.long)
        logits = decoder.execute([input_embeds, cache_pos])[0]

        # d. Greedy sample
        next_token_id = logits[:, -1, :].argmax(dim=-1).item()

        # e. Decode and record
        if next_token_id == eos_id:
            break
        generated_tokens.append(next_token_id)
        prev_token = torch.tensor([[next_token_id]], dtype=torch.long)

    t1 = time.time()
    n_tokens = len(generated_tokens)
    if n_tokens > 0:
        tok_per_sec = n_tokens / (t1 - t0)
        print(f"    Generated {n_tokens} tokens in {t1-t0:.2f}s ({tok_per_sec:.1f} tok/s)")

    return decode_tokens(tokenizer, generated_tokens)


def run_streaming(program, waveform: torch.Tensor, tokenizer, max_new_tokens: int = 512):
    """Run streaming inference, processing audio in chunks."""

    # Get streaming metadata
    sample_rate = program.method_meta("sample_rate") if hasattr(program, "method_meta") else 16000
    step_samples = 1280  # 80ms at 16kHz
    chunk_mel_len = 8
    hop_length = 160
    n_fft = 400
    stft_left_overlap = ((n_fft // 2 + hop_length - 1) // hop_length) * hop_length

    preprocessor = program.load_method("preprocessor")
    encoder = program.load_method("encode_audio_chunk")
    decoder = program.load_method("text_decoder")
    tok_embed_method = program.load_method("token_embedding")

    N_samples = waveform.shape[1]
    all_tokens = []
    dec_pos = 0
    enc_chunk_idx = 0
    eos_id = 2

    print(f"  Streaming {N_samples} samples in {step_samples}-sample chunks...")

    # Process audio in step_samples chunks
    offset = 0
    while offset < N_samples:
        chunk_end = min(offset + step_samples + stft_left_overlap + n_fft // 2, N_samples)
        audio_chunk = waveform[:, :chunk_end]

        # Preprocess full audio up to this point to get mel
        mel = preprocessor.execute([audio_chunk])[0]

        # Extract the mel frames for this chunk
        mel_start = max(0, (offset // hop_length))
        mel_end = mel_start + chunk_mel_len
        if mel_end > mel.shape[2]:
            break
        mel_chunk = mel[:, :, mel_start:mel_end]

        # Encode audio chunk
        enc_pos = torch.arange(enc_chunk_idx * 4, enc_chunk_idx * 4 + 4, dtype=torch.long)
        audio_embeds = encoder.execute([mel_chunk, enc_pos])[0]  # (1, 1, 3072)
        enc_chunk_idx += 1

        # Feed through decoder
        cache_pos = torch.tensor([dec_pos], dtype=torch.long)
        logits = decoder.execute([audio_embeds, cache_pos])[0]
        dec_pos += audio_embeds.shape[1]

        # Greedy decode
        next_token = logits[:, -1:, :].argmax(dim=-1)
        token_id = next_token.item()

        if token_id != eos_id and token_id != 0:
            all_tokens.append(token_id)
            # Continue decoding text tokens
            for _ in range(max_new_tokens):
                tok_embed = tok_embed_method.execute([next_token])[0]
                cache_pos_step = torch.tensor([dec_pos], dtype=torch.long)
                logits = decoder.execute([tok_embed, cache_pos_step])[0]
                next_token = logits[:, -1:, :].argmax(dim=-1)
                token_id = next_token.item()
                dec_pos += 1
                if token_id == eos_id or token_id == 0:
                    break
                all_tokens.append(token_id)

        if token_id == eos_id:
            break

        offset += step_samples

    return decode_tokens(tokenizer, all_tokens)


def run_mic_streaming(program, tokenizer, max_new_tokens: int = 512):
    """Run live microphone streaming inference. Requires pyaudio."""
    import pyaudio

    sample_rate = 16000
    step_samples = 1280  # 80ms at 16kHz
    chunk_mel_len = 8
    hop_length = 160
    n_fft = 400
    stft_left_overlap = ((n_fft // 2 + hop_length - 1) // hop_length) * hop_length

    preprocessor = program.load_method("preprocessor")
    encoder = program.load_method("encode_audio_chunk")
    decoder = program.load_method("text_decoder")
    tok_embed_method = program.load_method("token_embedding")

    dec_pos = 0
    enc_chunk_idx = 0
    eos_id = 2
    audio_buffer = torch.zeros(1, 0, dtype=torch.float32)

    pa = pyaudio.PyAudio()
    # Read audio in step_samples-sized chunks (80ms)
    stream = pa.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=sample_rate,
        input=True,
        frames_per_buffer=step_samples,
    )

    print("Listening... (Ctrl+C to stop)\n")
    try:
        while True:
            raw = stream.read(step_samples, exception_on_overflow=False)
            chunk = torch.frombuffer(raw, dtype=torch.float32).unsqueeze(0)  # (1, step_samples)
            audio_buffer = torch.cat([audio_buffer, chunk], dim=1)

            # Need enough samples for mel + overlap
            needed = stft_left_overlap + chunk_mel_len * hop_length + n_fft // 2
            if audio_buffer.shape[1] < needed:
                continue

            # Compute mel for current buffer
            mel = preprocessor.execute([audio_buffer])[0]

            # Extract chunk
            mel_start = max(0, mel.shape[2] - chunk_mel_len)
            if mel_start + chunk_mel_len > mel.shape[2]:
                continue
            mel_chunk = mel[:, :, mel_start:mel_start + chunk_mel_len]

            # Encode
            enc_pos = torch.arange(enc_chunk_idx * 4, enc_chunk_idx * 4 + 4, dtype=torch.long)
            audio_embeds = encoder.execute([mel_chunk, enc_pos])[0]
            enc_chunk_idx += 1

            # Decode
            cache_pos = torch.tensor([dec_pos], dtype=torch.long)
            logits = decoder.execute([audio_embeds, cache_pos])[0]
            dec_pos += audio_embeds.shape[1]

            next_token = logits[:, -1:, :].argmax(dim=-1)
            token_id = next_token.item()

            if token_id != eos_id and token_id != 0:
                tokens = [token_id]
                for _ in range(max_new_tokens):
                    tok_embed = tok_embed_method.execute([next_token])[0]
                    cache_pos_step = torch.tensor([dec_pos], dtype=torch.long)
                    logits = decoder.execute([tok_embed, cache_pos_step])[0]
                    next_token = logits[:, -1:, :].argmax(dim=-1)
                    token_id = next_token.item()
                    dec_pos += 1
                    if token_id == eos_id or token_id == 0:
                        break
                    tokens.append(token_id)

                text = decode_tokens(tokenizer, tokens)
                print(text, end="", flush=True)

    except KeyboardInterrupt:
        print("\n\nStopped.")
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()


def main():
    parser = argparse.ArgumentParser(
        description="Run Voxtral Realtime inference with ExecuTorch GGML backend"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to .pte model file",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to model directory (for tokenizer)",
    )
    parser.add_argument(
        "--preprocessor",
        default=None,
        help="Path to preprocessor .pte file (optional, uses torchaudio if not provided)",
    )
    parser.add_argument(
        "--audio",
        default=None,
        help="Path to audio file (.wav)",
    )
    parser.add_argument(
        "--mic",
        action="store_true",
        help="Stream from microphone for live transcription (requires pyaudio)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512)",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming inference mode (for file-based audio)",
    )
    parser.add_argument(
        "--compare-eager",
        action="store_true",
        help="Also run eager PyTorch for comparison",
    )
    parser.add_argument(
        "--dtype",
        default="BF16",
        choices=["BF16", "FP16"],
        help="Model dtype (must match export dtype, default: BF16)",
    )
    args = parser.parse_args()

    if not args.audio and not args.mic:
        parser.error("Either --audio or --mic is required")

    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = load_tokenizer(args.model_path)

    if args.mic:
        print(f"\nLoading model from {args.model}...")
        from executorch.runtime import Runtime

        runtime = Runtime.get()
        program = runtime.load_program(args.model)

        print("\n" + "=" * 60)
        print("[ExecuTorch + GGML — Microphone Streaming]")
        print("=" * 60)
        with torch.no_grad():
            run_mic_streaming(program, tokenizer, args.max_new_tokens)
        return

    print(f"Loading audio from {args.audio}...")
    waveform = load_audio(args.audio)
    duration = waveform.shape[1] / 16000
    print(f"  Audio: {duration:.1f}s, {waveform.shape[1]} samples")

    print(f"\nLoading model from {args.model}...")
    from executorch.runtime import Runtime

    runtime = Runtime.get()
    program = runtime.load_program(args.model)

    preprocessor_program = None
    if args.preprocessor:
        print(f"Loading preprocessor from {args.preprocessor}...")
        preprocessor_program = runtime.load_program(args.preprocessor)

    if args.compare_eager:
        print("\n" + "=" * 60)
        print("[Eager PyTorch]")
        print("=" * 60)
        from executorch.examples.models.voxtral_realtime.model import load_model
        eager_model = load_model(args.model_path)
        eager_model.eval()

        with torch.no_grad():
            import torchaudio
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000, n_fft=400, hop_length=160, n_mels=128, power=2.0
            )
            mel = mel_transform(waveform)
            mel = torch.clamp(mel, min=1e-10).log10()
            mel = torch.maximum(mel, mel.max() - 8.0)
            mel = (mel + 4.0) / 4.0

            from executorch.examples.models.voxtral_realtime.export_voxtral_rt import AudioEncoderExport
            enc = AudioEncoderExport(eager_model)
            enc.eval()
            T_mel = mel.shape[2]
            pad_to = ((T_mel + 7) // 8) * 8
            if pad_to > T_mel:
                mel = torch.nn.functional.pad(mel, (0, pad_to - T_mel))
            audio_embeds = enc(mel)

            from executorch.examples.models.voxtral_realtime.export_voxtral_rt import TextDecoderExport
            dec = TextDecoderExport(eager_model)
            dec.eval()
            T_audio = audio_embeds.shape[1]
            cache_pos = torch.arange(T_audio, dtype=torch.long)
            logits = dec(audio_embeds, cache_pos)

            next_token = logits[:, -1:, :].argmax(dim=-1)
            tokens = [next_token.item()]
            pos = T_audio
            for _ in range(args.max_new_tokens):
                if tokens[-1] == 2:
                    break
                tok_emb = eager_model.decoder.tok_embeddings(next_token)
                logits = dec(tok_emb, torch.tensor([pos], dtype=torch.long))
                next_token = logits[:, -1:, :].argmax(dim=-1)
                tokens.append(next_token.item())
                pos += 1
            if tokens and tokens[-1] == 2:
                tokens = tokens[:-1]
            eager_text = decode_tokens(tokenizer, tokens)
            print(f"  Result: {eager_text}")

    print("\n" + "=" * 60)
    print("[ExecuTorch + GGML]")
    print("=" * 60)

    with torch.no_grad():
        if args.streaming:
            text = run_streaming(program, waveform, tokenizer, args.max_new_tokens)
        else:
            model_dtype = torch.float16 if args.dtype == "FP16" else torch.bfloat16
            text = run_offline(program, waveform, tokenizer, args.max_new_tokens,
                               preprocessor_program=preprocessor_program,
                               model_dtype=model_dtype)

    print(f"\n  Transcription: {text}")
    print("\nDone!")


if __name__ == "__main__":
    main()
