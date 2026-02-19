"""ExecuTorch ggml backend.

Importing this package will attempt to load the native extension that registers
`GgmlBackend` with the ExecuTorch Runtime backend registry.

If the extension is not built/available, Python-side AOT lowering still works,
but executing a .pte containing delegated ggml subgraphs will fail with
"Backend GgmlBackend is not registered".
"""

from __future__ import annotations

# Best-effort native backend registration
try:
    # This module is produced by the CMake target `executorch_ggml_backend_py`.
    # Import side-effect: registers backend via static init.
    from . import _ggml_backend as _native_backend  # noqa: F401
except Exception:
    # Keep import working for pure-Python workflows.
    _native_backend = None

from executorch_ggml.ggml_partitioner import GgmlPartitioner
from executorch_ggml.ggml_backend import GgmlBackend
from executorch_ggml.edge_pipeline import to_edge_rewrite_and_lower

__all__ = [
    "GgmlPartitioner",
    "GgmlBackend",
    "to_edge_rewrite_and_lower",
]
