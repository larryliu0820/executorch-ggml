# executorch-ggml

An [ExecuTorch](https://github.com/pytorch/executorch) backend that delegates computation to [ggml](https://github.com/ggerganov/ggml).

## Motivation

`torch.export.export()` produces a clean, functional FX graph from any PyTorch model that follows its conventions (no data-dependent control flow, no in-place mutations on inputs). This covers a wide range of architectures: linear layers, convolutions, attention blocks, normalization layers, activation functions, and more.

ggml provides highly optimized, portable C kernels for tensor operations — quantized matmuls, fused attention, SIMD-accelerated element-wise ops — across CPU, Metal, CUDA, Vulkan, and SYCL, with no external dependencies.

**executorch-ggml bridges the two:** any model that exports cleanly through `torch.export` can be partitioned and lowered to ggml kernels at ahead-of-time compile, then executed through ggml's compute graph at runtime. This means:

- **Broad model coverage** — if your model exports, it can run on ggml. No manual ggml graph construction needed.
- **Optimized inference** — ggml's hand-tuned kernels (quantized matmul, fused softmax, etc.) replace generic ATen implementations.
- **Portable deployment** — ggml runs on x86, ARM, Apple Silicon, and GPU backends without framework-level dependencies.
- **Incremental adoption** — the partitioner only delegates ops that ggml supports. Unsupported ops fall back to ExecuTorch's default CPU executor. You can start with a few ops and expand coverage over time.

## How It Works

```
PyTorch Model
    │
    ▼
torch.export.export()      # Produces an ExportedProgram (ATen dialect)
    │
    ▼
executorch.exir.to_edge()  # Converts to Edge dialect
    │
    ▼
ExportedProgram rewrites   # (optional) e.g. BN folding
    │
    ▼
GgmlPartitioner            # Tags supported Edge ops for delegation
    │
    ▼
GgmlBackend.preprocess()   # Maps ATen ops → ggml IR, serializes to FlatBuffer
    │
    ▼
.pte file                  # ExecuTorch program with embedded ggml subgraphs
    │
    ▼
GgmlBackendInterface       # C++ runtime: deserializes IR, builds ggml_cgraph,
(init / execute / destroy)   executes via ggml_graph_compute
```

## Supported Models

| Model | Type | Status | Notes |
|-------|------|--------|-------|
| [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | LLM | Full (F32 + Q8_0) | Text generation with KV cache, fused SDPA |
| [Voxtral-Mini-4B-Realtime](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602) | ASR | Full (BF16) | Audio encoder (fused RoPE + RMS norm) + text decoder with KV cache |
| [Parakeet TDT 0.6B](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | ASR | Full (F32 + Q8_0) | Speech-to-text, FastConformer encoder + TDT decoder |
| MobileNetV2 | Vision | Full | Requires `BatchNormFoldingRewritePass` for Conv+BN |
| Linear + ReLU | MLP | Full | Basic MLP architectures |
| Custom CNNs | Vision | Partial | Conv2d, depthwise conv, pooling, activations |

Unsupported ops automatically fall back to ExecuTorch's CPU executor.

## Quick Start

### Installation

**Stable (PyPI):**
```bash
pip install -e .
```

**Nightly builds:** Choose one of these methods:

1. **With extra-index-url flag:**
   ```bash
   pip install -e . --extra-index-url https://download.pytorch.org/whl/nightly/cpu/
   ```

2. **Using requirements-nightly.txt:**
   ```bash
   pip install -r requirements-nightly.txt
   ```

3. **Set pip config globally (one-time setup):**
   ```bash
   pip config --user set global.extra-index-url https://download.pytorch.org/whl/nightly/cpu/
   # Now just run:
   pip install -e .
   ```

These methods will install the latest `executorch` nightly version from PyTorch's nightly wheel server.

### Python (ahead-of-time compilation)

**Simple model (no BatchNorm):**
```python
import torch
from torch.export import export
from executorch.exir import to_edge_transform_and_lower
from executorch_ggml import GgmlPartitioner

model = torch.nn.Sequential(
    torch.nn.Linear(4, 8),
    torch.nn.LeakyReLU(0.1),
).eval()

exported = export(model, (torch.randn(2, 4),))
edge = to_edge_transform_and_lower(exported, partitioner=[GgmlPartitioner()])
et_program = edge.to_executorch()

with open("model.pte", "wb") as f:
    f.write(et_program.buffer)
```

**MobileNetV2 (with BatchNorm folding):**
```python
import torch
from torch.export import export
from torchvision.models import mobilenet_v2
from executorch_ggml import GgmlPartitioner, to_edge_rewrite_and_lower
from executorch_ggml.passes import BatchNormFoldingRewritePass

model = mobilenet_v2(weights=None).eval()
exported = export(model, (torch.randn(1, 3, 224, 224),))
edge = to_edge_rewrite_and_lower(
    exported,
    ep_passes=[BatchNormFoldingRewritePass()],
    partitioner=[GgmlPartitioner()],
)
et_program = edge.to_executorch()

with open("mobilenet_v2.pte", "wb") as f:
    f.write(et_program.buffer)
```

### Qwen3-0.6B (Text Generation)

**Export:**
```bash
# F32
python runner/export_qwen3_q8.py

# This produces qwen3/qwen3_q8_0.pte (Q8_0 quantized)
# For F32, export with the same script but without quant_config (see code)
```

**Run:**
```bash
# Q8_0 (~370 MB)
python runner/run_qwen3.py --model qwen3/qwen3_q8_0.pte

# F32 (~1.2 GB)
python runner/run_qwen3.py --model qwen3/qwen3.pte
```

Requires `optimum-executorch` for the export wrapper:
```bash
pip install optimum[executorch]
```

### Parakeet TDT 0.6B (Speech Recognition)

**Get test audio (30s LibriSpeech clip):**
```bash
python -c "from datasets import load_dataset; import soundfile as sf; s = load_dataset('hf-internal-testing/librispeech_asr_demo', 'clean', split='validation')[0]['audio']; sf.write('test_audio.wav', s['array'][:s['sampling_rate']*30], s['sampling_rate'])"
```

**Export:**
```bash
# F32 model
python runner/export_parakeet.py --dtype F32 --audio test_audio.wav

# Q8_0 model (default)
python runner/export_parakeet.py --dtype Q8_0 --audio test_audio.wav
```

**Run:**
```bash
python runner/run_parakeet.py --model parakeet_ggml/model_q8_0.pte --audio test_audio.wav
```

This runs eager PyTorch, Q8_0, and F32 side by side and compares transcriptions.

Requires NeMo for the model:
```bash
pip install nemo_toolkit[asr]
```

To force CPU-only execution (no Metal GPU):
```bash
GGML_BACKEND_DEVICE=cpu python runner/run_parakeet.py --model parakeet_ggml/model_q8_0.pte --audio test_audio.wav
```

### Voxtral-Mini-4B-Realtime (Speech Recognition)

**Export:**
```bash
python runner/export_voxtral_rt.py \
  --model-path /path/to/Voxtral-Mini-4B-Realtime-2602 \
  --dtype BF16
```

**Get test audio (30s LibriSpeech clip):**
```bash
python -c "from datasets import load_dataset; import soundfile as sf; s = load_dataset('distil-whisper/librispeech_long', 'clean', split='validation')[0]['audio']; sf.write('test_audio.wav', s['array'][:s['sampling_rate']*30], s['sampling_rate'])"
```

**Run:**
```bash
DYLD_LIBRARY_PATH=python/executorch_ggml python runner/run_voxtral_rt.py \
  --model voxtral_ggml/model_bf16.pte \
  --model-path /path/to/Voxtral-Mini-4B-Realtime-2602 \
  --audio test_audio.wav
```

The encoder uses fused RoPE (`ggml_rope_ext`) and native RMS norm for the 32-layer causal whisper encoder. The decoder follows the Voxtral Realtime inference protocol: at each position, the input is the element-wise sum of `audio_embeds[pos]` and `token_embedding(prev_token)`.

Requires `mistral-common` for the tokenizer:
```bash
pip install mistral-common
```

### C++ (runtime)

```bash
cmake -B build \
  -DLLAMA_CPP_DIR=/path/to/llama.cpp \
  -DEXECUTORCH_DIR=/path/to/executorch
cmake --build build
```

Link `executorch_ggml_runtime` into your ExecuTorch runner. The backend registers itself automatically at static init time — any `.pte` file containing `GgmlBackend` delegates will route through ggml.

## Extending to More Ops

To add support for a new ATen op:

1. Add the op to the `OpCode` enum in `schema/ggml_ir.fbs`
2. Add the ATen op to `_SUPPORTED_OPS` in `ggml_partitioner.py`
3. Add the ATen→IR mapping in `GgmlBackend.preprocess()` in `ggml_backend.py`
4. Add the ggml builder call in `GgmlBackendInterface::init()` in `ggml_backend.cpp`
5. Regenerate FlatBuffer headers: `flatc --cpp -o build/ schema/ggml_ir.fbs`

## License

BSD License. See [LICENSE](LICENSE).
