#!/usr/bin/env python3
"""Voxtral 1-layer text decoder cosine similarity test.

Loads Voxtral, truncates to 1 decoder layer, exports via GGML backend,
and compares logits against eager PyTorch for prefill + 5 decode steps.

Usage:
    DYLD_LIBRARY_PATH=python/executorch_ggml python3 tests/test_voxtral_decoder.py
"""

import os
import sys

import torch
import torch.nn.functional as F
from torch.export import Dim, export

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _repo_root)

MODEL_PATH = (
    "/Users/larryliu/.cache/huggingface/hub/"
    "models--mistralai--Voxtral-Mini-4B-Realtime-2602/"
    "snapshots/2769294da9567371363522aac9bbcfdd19447add"
)
DTYPE = torch.bfloat16
MAX_SEQ_LEN = 128


def main():
    from executorch.examples.models.voxtral_realtime.export_voxtral_rt import (
        TextDecoderExport,
    )
    from executorch.examples.models.voxtral_realtime.model import load_model
    from executorch.exir import (
        EdgeCompileConfig,
        ExecutorchBackendConfig,
        to_edge_transform_and_lower,
    )
    from executorch.exir.passes import MemoryPlanningPass
    from executorch.extension.pybindings.portable_lib import (
        _load_for_executorch_from_buffer,
    )

    from executorch_ggml import GgmlPartitioner
    from executorch_ggml.passes import BF16UnsafeOpsCastPass, RemoveGraphAssertsPass
    from executorch_ggml.passes.replace_copy_ops_pass import ReplaceCopyOpsPass

    # 1. Load model + truncate to 1 decoder layer
    print(f"Loading model from {MODEL_PATH}...")
    model = load_model(
        MODEL_PATH,
        max_seq_len=MAX_SEQ_LEN,
        dtype=DTYPE,
        use_standard_attention=True,
    )
    model.decoder.layers = model.decoder.layers[:1]
    dim = model.config.dim
    print(f"Truncated to 1 decoder layer, dim={dim}")

    # 2. Export text_decoder
    print("\nExporting text_decoder...")
    text_decoder = TextDecoderExport(model)
    text_decoder.eval()
    seq_dim = Dim("seq_len", min=1, max=MAX_SEQ_LEN)
    sample_embeds = torch.randn(1, 4, dim, dtype=DTYPE)
    sample_pos = torch.arange(4, dtype=torch.long)
    prog = export(
        text_decoder,
        (sample_embeds, sample_pos),
        dynamic_shapes={
            "input_embeds": {1: seq_dim},
            "cache_position": {0: seq_dim},
        },
        strict=True,
    )

    # 3. Lower through GGML
    print("\nLowering to GGML...")
    et = to_edge_transform_and_lower(
        {"forward": prog},
        transform_passes=[
            BF16UnsafeOpsCastPass(),
            ReplaceCopyOpsPass(),
            RemoveGraphAssertsPass(),
        ],
        partitioner=[GgmlPartitioner()],
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=True,
        ),
    ).to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
        ),
    )
    pte = _load_for_executorch_from_buffer(et.buffer)

    # 4. Prepare eager model (shares model.decoder, KV caches are still zeros)
    eager = TextDecoderExport(model)
    eager.eval()

    # 5. Run prefill + 5 decode steps, compare logits
    torch.manual_seed(42)
    prefill_len = 4
    prefill_embeds = torch.randn(1, prefill_len, dim, dtype=DTYPE)
    prefill_pos = torch.arange(prefill_len, dtype=torch.long)

    print("\n--- Prefill ---")
    pte_out = pte.forward((prefill_embeds, prefill_pos))[0]
    with torch.no_grad():
        eager_out = eager(prefill_embeds, prefill_pos)

    cos_sim = F.cosine_similarity(
        eager_out.float().flatten(), pte_out.float().flatten(), dim=0
    ).item()
    max_diff = (eager_out.float() - pte_out.float()).abs().max().item()
    print(f"Prefill: cos_sim={cos_sim:.6f}  max_diff={max_diff:.6f}")
    assert cos_sim > 0.99, f"Prefill cosine sim {cos_sim:.6f} < 0.99"
    assert max_diff < 0.5, f"Prefill max diff {max_diff:.6f} >= 0.5"

    print("\n--- Decode ---")
    for i in range(5):
        dec_embeds = torch.randn(1, 1, dim, dtype=DTYPE)
        dec_pos = torch.tensor([prefill_len + i], dtype=torch.long)

        pte_out = pte.forward((dec_embeds, dec_pos))[0]
        with torch.no_grad():
            eager_out = eager(dec_embeds, dec_pos)

        cos_sim = F.cosine_similarity(
            eager_out.float().flatten(), pte_out.float().flatten(), dim=0
        ).item()
        max_diff = (eager_out.float() - pte_out.float()).abs().max().item()
        print(f"Step {i}: cos_sim={cos_sim:.6f}  max_diff={max_diff:.6f}")
        assert cos_sim > 0.99, f"Decode step {i} cosine sim {cos_sim:.6f} < 0.99"
        assert max_diff < 0.5, f"Decode step {i} max diff {max_diff:.6f} >= 0.5"

    print("\nAll checks passed!")


if __name__ == "__main__":
    main()
