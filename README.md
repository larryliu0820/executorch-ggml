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

| Model | Status | Notes |
|-------|--------|-------|
| MobileNetV2 | Full | Requires `BatchNormFoldingRewritePass` to fold Conv+BN patterns |
| Linear + ReLU | Full | Basic MLP architectures |
| Custom CNNs | Partial | Conv2d, depthwise conv, pooling, activations supported |

Models with BatchNorm layers require the `BatchNormFoldingRewritePass` to fold BN parameters into preceding convolutions before partitioning. Use `to_edge_rewrite_and_lower(..., ep_passes=[BatchNormFoldingRewritePass()])`.

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

These methods will install the latest `executorch` nightly version from PyTorch's nightly wheel server (currently 1.2.0.dev20260218).

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
