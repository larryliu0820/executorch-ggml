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
torch.export.export()      # Produces Edge-dialect FX graph
    │
    ▼
GgmlPartitioner            # Tags supported ATen ops for delegation
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
| `aten.t` | *(no-op)* | Looked through; ggml layout already matches |
| `aten.leaky_relu` | `LEAKY_RELU` | With configurable negative slope |

More ops can be added by extending the `OpCode` enum in `schema/ggml_ir.fbs`, the ATen→IR mapping in `ggml_backend.py`, and the ggml builder call in `ggml_backend.cpp`.

## Quick Start

### Python (ahead-of-time compilation)

```bash
pip install -e .
```

```python
import torch
import torch.nn as nn
from torch.export import export
from executorch.exir import to_edge_transform_and_lower
from executorch_ggml import GgmlPartitioner

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 8)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leaky_relu(self.linear(x))

model = MyModel().eval()
exported = export(model, (torch.randn(2, 4),))
edge = to_edge_transform_and_lower(exported, partitioner=[GgmlPartitioner()])
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
