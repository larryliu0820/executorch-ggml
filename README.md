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

## Currently Supported Ops

| ATen Op | ggml Op | Notes |
|---------|---------|-------|
| `aten.addmm` | `MUL_MAT` + `ADD` | Linear layer with bias |
| `aten.mm` | `MUL_MAT` | Linear layer without bias |
| `aten.t` / `aten.permute_copy` | *(no-op)* | Looked through; ggml layout already matches |
| `aten.leaky_relu` | `LEAKY_RELU` | With configurable negative slope |
| `aten.convolution` / `aten.conv2d` | `CONV_2D` | regular conv |
| `aten.convolution` / `aten.conv2d` | `CONV_2D_DW` | depthwise conv (groups>1) |
| `aten.hardtanh` / `aten.clamp` | `HARDTANH` | used for ReLU6/clamp |
| `aten.add.Tensor` | `ADD` | residual connections (alpha==1) |
| `aten.mean.dim` | `MEAN` | supports MV2 global avg pool (mean over H,W) via `ggml_pool_2d(AVG, ...)` |
| `aten.view` / `aten.reshape` | `VIEW` | reshape/view |
| `aten.permute` / `aten.permute_copy` | `PERMUTE` | transpose dims |
| `dim_order_ops._clone_dim_order` | *(no-op)* | look-through layout materialization |

### MobileNetV2 status

`torchvision.models.mobilenet_v2(weights=None)` can now be exported, lowered, and executed end-to-end
on the ggml backend when you run BN folding rewrite before partitioning.

Notes:
- ggml conv (CPU im2col) has dtype contracts; we store conv weights as fp16 and keep runtime activations fp32.
- conv outputs are cast back to fp32 to avoid mixed-type residual adds.

### BatchNorm folding (Conv+BN)

MobileNetV2-style graphs often contain the pattern:

`conv -> batch_norm (inference) -> getitem(0)`

To avoid BN’s tuple-return semantics inside delegated subgraphs, we provide an
**ExportedProgram-level rewrite pass** that folds BN parameters into the upstream
conv weights/bias and removes the BN/getitem nodes:

- `executorch_ggml.passes.BatchNormFoldingRewritePass`

This pass is intended to run **after** `to_edge(...)` and **before** partitioning.
Use `executorch_ggml.to_edge_rewrite_and_lower(..., ep_passes=[...])` to compose
these ExportedProgram rewrites into the standard ExecuTorch lowering pipeline.

For MobileNetV2, you should include this pass in `ep_passes`.

More ops can be added by extending the `OpCode` enum in `schema/ggml_ir.fbs`, the ATen→IR mapping in `ggml_backend.py`, and the ggml builder call in `ggml_backend.cpp`.

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

```python
import torch
import torch.nn as nn
from torch.export import export
from executorch_ggml import GgmlPartitioner, to_edge_rewrite_and_lower
from executorch_ggml.passes import BatchNormFoldingRewritePass

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 8)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leaky_relu(self.linear(x))

model = MyModel().eval()
exported = export(model, (torch.randn(2, 4),))
edge = to_edge_rewrite_and_lower(
    exported,
    ep_passes=[BatchNormFoldingRewritePass()],
    partitioner=[GgmlPartitioner()],
)
et_program = edge.to_executorch()

with open("model.pte", "wb") as f:
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
