#!/usr/bin/env python3
"""Test mask generation and encoder sub-operations through GGML."""

import os, sys
import torch
import torch.nn as nn
from torch.export import Dim, export

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _repo_root)

_parakeet_pkg = "executorch.examples.models.parakeet.quantize"
if _parakeet_pkg not in sys.modules:
    import types
    _stub = types.ModuleType(_parakeet_pkg)
    _stub.quantize_model_ = lambda *a, **kw: None
    sys.modules[_parakeet_pkg] = _stub

sys.path.insert(0, os.path.join(_repo_root, "third-party", "executorch", "examples", "models", "parakeet"))

import executorch_ggml
from executorch_ggml import GgmlPartitioner, to_edge_rewrite_and_lower
from executorch_ggml.passes.replace_copy_ops_pass import ReplaceCopyOpsPass
from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig
from executorch.exir.passes import MemoryPlanningPass
from executorch.extension.pybindings.portable_lib import _load_for_executorch_from_buffer


def lower_and_run(ep, inputs):
    edge_mgr = to_edge_rewrite_and_lower(
        ep,
        transform_passes=[ReplaceCopyOpsPass()],
        partitioner=[GgmlPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False, _skip_dim_order=True),
    )
    et = edge_mgr.to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
        ),
    )
    pte = _load_for_executorch_from_buffer(et.buffer)
    return pte.forward(inputs)


def compare(name, eager, ggml):
    if eager.shape != ggml.shape:
        print(f"  {name}: SHAPE MISMATCH eager={eager.shape} ggml={ggml.shape}")
        return
    if eager.dtype == torch.bool:
        match = (eager == ggml).all().item()
        mismatch = (eager != ggml).sum().item()
        print(f"  {name}: shape={eager.shape} dtype=bool match={match} mismatches={mismatch}")
        if mismatch > 0:
            # Show first few mismatches
            indices = (eager != ggml).nonzero()[:5]
            for idx in indices:
                idx = tuple(idx.tolist())
                print(f"    [{idx}] eager={eager[idx].item()} ggml={ggml[idx].item()}")
    else:
        diff = (eager.float() - ggml.float()).abs()
        cos = torch.nn.functional.cosine_similarity(
            eager.float().flatten().unsqueeze(0),
            ggml.float().flatten().unsqueeze(0),
        ).item()
        print(f"  {name}: shape={eager.shape} max_diff={diff.max().item():.6f} cos_sim={cos:.6f}")
        if torch.isnan(ggml).any():
            print(f"    WARNING: NaN in ggml output!")
        if torch.isinf(ggml).any():
            print(f"    WARNING: Inf in ggml output!")


class PreEncodeAndMask(nn.Module):
    """Just pre_encode + pos_enc + mask creation."""
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, audio_signal: torch.Tensor, length: torch.Tensor):
        enc = self.encoder
        x = audio_signal.transpose(1, 2)
        x, length = enc.pre_encode(x=x, lengths=length)
        length = length.to(torch.int64)
        x, pos_emb = enc.pos_enc(x=x, cache_len=0)
        max_audio_length = x.size(1)
        pad_mask, att_mask = enc._create_masks(
            att_context_size=enc.att_context_size,
            padding_length=length,
            max_audio_length=max_audio_length,
            offset=None,
            device=x.device,
        )
        # Return x, pos_emb, and masks for comparison
        # Cast bool masks to float for comparison (GGML stores bools as I32)
        att_mask_f = att_mask.float() if att_mask is not None else torch.zeros(1)
        pad_mask_f = pad_mask.float()
        return x, pos_emb, att_mask_f, pad_mask_f, length


class PreEncodeOnly(nn.Module):
    """Just pre_encode + pos_enc, no masks."""
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, audio_signal: torch.Tensor, length: torch.Tensor):
        enc = self.encoder
        x = audio_signal.transpose(1, 2)
        x, length = enc.pre_encode(x=x, lengths=length)
        length = length.to(torch.int64)
        x, pos_emb = enc.pos_enc(x=x, cache_len=0)
        return x, pos_emb, length


class PreEncodeOneLayer(nn.Module):
    """Pre_encode + pos_enc + mask + 1 layer."""
    def __init__(self, encoder, num_layers=1):
        super().__init__()
        self.encoder = encoder
        self.num_layers = num_layers

    def forward(self, audio_signal: torch.Tensor, length: torch.Tensor):
        enc = self.encoder
        x = audio_signal.transpose(1, 2)
        x, length = enc.pre_encode(x=x, lengths=length)
        length = length.to(torch.int64)
        x, pos_emb = enc.pos_enc(x=x, cache_len=0)
        max_audio_length = x.size(1)
        pad_mask, att_mask = enc._create_masks(
            att_context_size=enc.att_context_size,
            padding_length=length,
            max_audio_length=max_audio_length,
            offset=None,
            device=x.device,
        )
        for layer in enc.layers[:self.num_layers]:
            x = layer(x=x, att_mask=att_mask, pos_emb=pos_emb, pad_mask=pad_mask)
        return x, length


def main():
    from export_parakeet_tdt import load_model
    import soundfile as sf
    import numpy as np

    print("Loading model...")
    model = load_model()
    encoder = model.encoder

    # Load mel spectrogram
    audio_path = os.path.join(_repo_root, "test_audio.wav")
    data, sr = sf.read(audio_path)
    waveform = torch.from_numpy(np.array(data, dtype=np.float32)).unsqueeze(0)
    if sr != model.preprocessor._cfg.sample_rate:
        import torchaudio
        waveform = torchaudio.transforms.Resample(sr, model.preprocessor._cfg.sample_rate)(waveform)
    with torch.no_grad():
        mel, mel_len = model.preprocessor(
            input_signal=waveform, length=torch.tensor([waveform.shape[1]])
        )
    mel_len = mel_len.to(torch.int64)
    T_mel = mel.shape[2]
    print(f"Mel: {mel.shape}, mel_len={mel_len.item()}")

    max_mel_frames = int(50 / model.preprocessor._cfg.window_stride)  # same as export
    encoder.update_max_seq_length(seq_length=max_mel_frames, device=mel.device)

    # ======== Test 1: Pre-encode only (no masks) ========
    print("\n" + "="*60)
    print("TEST 1: Pre-encode + pos_enc only (no masks)")
    print("="*60)
    wrapper = PreEncodeOnly(encoder)
    wrapper.eval()

    with torch.no_grad():
        eager_x, eager_pos, eager_len = wrapper(audio_signal=mel, length=mel_len)
        print(f"Eager: x={eager_x.shape} pos_emb={eager_pos.shape} len={eager_len.item()}")

    # Export with max shapes, run with actual shapes
    max_mel = torch.randn(1, 128, max_mel_frames)
    max_len = torch.tensor([max_mel_frames], dtype=torch.int64)

    ep = export(
        wrapper, (),
        kwargs={"audio_signal": max_mel, "length": max_len},
        dynamic_shapes={"audio_signal": {2: Dim.AUTO}, "length": {}},
        strict=False,
    )
    ggml_result = lower_and_run(ep, (mel, mel_len))
    ggml_x, ggml_pos, ggml_len = ggml_result[0], ggml_result[1], ggml_result[2]
    print(f"GGML:  x={ggml_x.shape} pos_emb={ggml_pos.shape} len={ggml_len.item()}")
    compare("x", eager_x, ggml_x)
    compare("pos_emb", eager_pos, ggml_pos)

    # ======== Test 2: Pre-encode + masks ========
    print("\n" + "="*60)
    print("TEST 2: Pre-encode + pos_enc + mask creation")
    print("="*60)
    wrapper2 = PreEncodeAndMask(encoder)
    wrapper2.eval()

    with torch.no_grad():
        eager_x2, eager_pos2, eager_att, eager_pad, eager_len2 = wrapper2(
            audio_signal=mel, length=mel_len
        )
        print(f"Eager: x={eager_x2.shape} att_mask={eager_att.shape} pad_mask={eager_pad.shape}")

    ep2 = export(
        wrapper2, (),
        kwargs={"audio_signal": max_mel, "length": max_len},
        dynamic_shapes={"audio_signal": {2: Dim.AUTO}, "length": {}},
        strict=False,
    )
    ggml_result2 = lower_and_run(ep2, (mel, mel_len))
    ggml_x2, ggml_pos2, ggml_att, ggml_pad, ggml_len2 = (
        ggml_result2[0], ggml_result2[1], ggml_result2[2], ggml_result2[3], ggml_result2[4]
    )
    print(f"GGML:  x={ggml_x2.shape} att_mask={ggml_att.shape} pad_mask={ggml_pad.shape}")
    compare("x", eager_x2, ggml_x2)
    compare("pos_emb", eager_pos2, ggml_pos2)
    compare("att_mask", eager_att, ggml_att)
    compare("pad_mask", eager_pad, ggml_pad)

    # ======== Test 3: Pre-encode + masks + 1 layer ========
    print("\n" + "="*60)
    print("TEST 3: Pre-encode + mask + 1 conformer layer")
    print("="*60)
    wrapper3 = PreEncodeOneLayer(encoder, num_layers=1)
    wrapper3.eval()

    with torch.no_grad():
        eager_out3, eager_len3 = wrapper3(audio_signal=mel, length=mel_len)
        print(f"Eager: out={eager_out3.shape}")

    ep3 = export(
        wrapper3, (),
        kwargs={"audio_signal": max_mel, "length": max_len},
        dynamic_shapes={"audio_signal": {2: Dim.AUTO}, "length": {}},
        strict=False,
    )
    ggml_result3 = lower_and_run(ep3, (mel, mel_len))
    ggml_out3 = ggml_result3[0]
    print(f"GGML:  out={ggml_out3.shape}")
    compare("layer_0_out", eager_out3, ggml_out3)


if __name__ == "__main__":
    with torch.no_grad():
        main()
