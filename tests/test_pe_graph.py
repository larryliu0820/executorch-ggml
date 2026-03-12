#!/usr/bin/env python3
"""Debug: print the FX graph ops for the PreEncode wrapper."""

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

from executorch_ggml import GgmlPartitioner
from executorch_ggml.passes.replace_copy_ops_pass import ReplaceCopyOpsPass
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower


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
    print("Loading model...", flush=True)
    model = load_model()
    encoder = model.encoder

    max_mel_frames = int(50 / model.preprocessor._cfg.window_stride)
    encoder.update_max_seq_length(seq_length=max_mel_frames, device="cpu")

    wrapper = PreEncodeOnly(encoder)
    wrapper.eval()

    max_mel = torch.randn(1, 128, max_mel_frames)
    max_len = torch.tensor([max_mel_frames], dtype=torch.int64)

    print("Exporting...", flush=True)
    ep = export(wrapper, (), kwargs={"audio_signal": max_mel, "length": max_len},
                dynamic_shapes={"audio_signal": {2: Dim.AUTO}, "length": {}}, strict=False)

    print("\n=== EXPORTED GRAPH (last 40 nodes) ===", flush=True)
    nodes = list(ep.graph.nodes)
    for node in nodes[-40:]:
        if node.op == "call_function":
            name = str(node.target).split(".")[-2] + "." + str(node.target).split(".")[-1] if "." in str(node.target) else str(node.target)
            fv = node.meta.get("val")
            if fv is not None and hasattr(fv, 'shape'):
                shape = list(fv.shape)
                dtype = fv.dtype
            else:
                shape = "?"
                dtype = "?"
            args_brief = []
            for a in node.args:
                if isinstance(a, torch.fx.Node):
                    args_brief.append(a.name)
                else:
                    args_brief.append(repr(a))
            print(f"  {node.name}: {name}({', '.join(args_brief[:4])}) -> {shape} {dtype}")
        elif node.op == "output":
            print(f"  {node.name}: output")
        elif node.op == "placeholder":
            fv = node.meta.get("val")
            if fv is not None and hasattr(fv, 'shape'):
                print(f"  {node.name}: placeholder -> {list(fv.shape)} {fv.dtype}")

    print("\n=== EDGE GRAPH (after to_edge) ===", flush=True)
    edge = to_edge_transform_and_lower(
        ep,
        partitioner=[GgmlPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False, _skip_dim_order=True),
    )
    print(edge.exported_program().graph_module, flush=True)


if __name__ == "__main__":
    with torch.no_grad():
        main()
