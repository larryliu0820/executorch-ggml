#!/usr/bin/env python3
"""Minimal test: just pre-encode + pos_enc through GGML."""

import os, sys
import torch
import torch.nn as nn
from torch.export import Dim, export

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _repo_root)
_parakeet_pkg = "executorch.examples.models.parakeet.quantize"
if _parakeet_pkg not in sys.modules:
    import types; _stub = types.ModuleType(_parakeet_pkg); _stub.quantize_model_ = lambda *a, **kw: None; sys.modules[_parakeet_pkg] = _stub
sys.path.insert(0, os.path.join(_repo_root, "third-party", "executorch", "examples", "models", "parakeet"))

import executorch_ggml
from executorch_ggml import GgmlPartitioner, to_edge_rewrite_and_lower
from executorch_ggml.passes.replace_copy_ops_pass import ReplaceCopyOpsPass
from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig
from executorch.exir.passes import MemoryPlanningPass
from executorch.extension.pybindings.portable_lib import _load_for_executorch_from_buffer


class PreEncodeOnly(nn.Module):
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


def main():
    from export_parakeet_tdt import load_model
    import soundfile as sf
    import numpy as np

    print("Loading model...")
    model = load_model()
    encoder = model.encoder

    # Load mel
    audio_path = os.path.join(_repo_root, "test_audio.wav")
    data, sr = sf.read(audio_path)
    waveform = torch.from_numpy(np.array(data, dtype=np.float32)).unsqueeze(0)
    with torch.no_grad():
        mel, mel_len = model.preprocessor(input_signal=waveform, length=torch.tensor([waveform.shape[1]]))
    mel_len = mel_len.to(torch.int64)
    print(f"Mel: {mel.shape}, mel_len={mel_len.item()}")

    max_mel_frames = int(50 / model.preprocessor._cfg.window_stride)
    encoder.update_max_seq_length(seq_length=max_mel_frames, device=mel.device)

    wrapper = PreEncodeOnly(encoder)
    wrapper.eval()

    with torch.no_grad():
        eager_x, eager_pos, eager_len = wrapper(audio_signal=mel, length=mel_len)
        print(f"Eager: x={eager_x.shape} pos_emb={eager_pos.shape} len={eager_len.item()}")

    max_mel = torch.randn(1, 128, max_mel_frames)
    max_len = torch.tensor([max_mel_frames], dtype=torch.int64)

    print("Exporting...")
    ep = export(wrapper, (), kwargs={"audio_signal": max_mel, "length": max_len},
                dynamic_shapes={"audio_signal": {2: Dim.AUTO}, "length": {}}, strict=False)

    # Print the slice operations in the graph to see the op_params
    print("\nSlice ops in exported graph:")
    for node in ep.graph.nodes:
        if node.op == "call_function":
            name = str(node.target)
            if "slice" in name:
                args_str = ", ".join(str(a) for a in node.args[1:])
                fv = node.meta.get("val")
                shape = list(fv.shape) if fv is not None else "?"
                print(f"  {node.name}: {name}({args_str}) -> {shape}")

    print("\nLowering...")
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

    print("Loading .pte...")
    pte = _load_for_executorch_from_buffer(et.buffer)

    print("Running...")
    result = pte.forward((mel, mel_len))
    ggml_x, ggml_pos, ggml_len = result[0], result[1], result[2]
    print(f"GGML: x={ggml_x.shape} pos_emb={ggml_pos.shape} len={ggml_len.item()}")

    # Compare
    diff_x = (eager_x.float() - ggml_x.float()).abs()
    print(f"x: max_diff={diff_x.max().item():.6f}")

    if eager_pos.shape == ggml_pos.shape:
        diff_pos = (eager_pos.float() - ggml_pos.float()).abs()
        print(f"pos_emb: max_diff={diff_pos.max().item():.6f}")
    else:
        print(f"pos_emb: SHAPE MISMATCH eager={eager_pos.shape} ggml={ggml_pos.shape}")


if __name__ == "__main__":
    with torch.no_grad():
        main()
