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
    import ctypes
    import sys

    # Preload the pip-installed libnvJitLink before _portable_lib pulls in
    # libcusparse.  Without this, the dynamic linker may find the *system*
    # libnvJitLink (e.g. CUDA 12.6 in /usr/local/cuda-*/lib64) which lacks
    # the _12_8 versioned symbols required by the pip-installed libcusparse
    # (CUDA 12.8), causing an ImportError.
    try:
        import os

        import nvidia.nvjitlink.lib  # type: ignore[import-untyped]

        _nvjitlink_dir = os.path.dirname(nvidia.nvjitlink.lib.__file__)
        _nvjitlink_so = os.path.join(_nvjitlink_dir, "libnvJitLink.so.12")
        if os.path.isfile(_nvjitlink_so):
            ctypes.CDLL(_nvjitlink_so, mode=ctypes.RTLD_GLOBAL)
    except Exception:
        pass

    # On Linux, Python imports don't make symbols globally visible by default
    # (unlike macOS's -undefined dynamic_lookup). We need RTLD_GLOBAL so that
    # _ggml_backend can resolve ExecuTorch symbols from the runtime .so.
    #
    # If an ExecuTorch .so was already loaded (e.g. _llm_runner.so from
    # TextLLMRunner, or _portable_lib.so), its symbols are process-resident
    # but not globally visible.  We re-dlopen it with RTLD_GLOBAL to promote
    # the symbols before loading _ggml_backend.so.
    import os

    _et_so = None
    for _name, _mod in list(sys.modules.items()):
        if "executorch" not in _name:
            continue
        _f = getattr(_mod, "__file__", None)
        if _f and _f.endswith(".so") and os.path.isfile(_f):
            _et_so = _f
            break

    if _et_so is not None:
        ctypes.CDLL(_et_so, mode=ctypes.RTLD_GLOBAL)
    else:
        # No ExecuTorch .so loaded yet — try to import one with RTLD_GLOBAL.
        # We attempt multiple modules because some ET wheel builds ship
        # _portable_lib or _llm_runner but not both, and either may fail to
        # load depending on how torch/ET were built.
        _old_flags = sys.getdlopenflags()
        sys.setdlopenflags(_old_flags | ctypes.RTLD_GLOBAL)
        try:
            from executorch.extension.pybindings import _portable_lib  # noqa: F401
        except Exception:
            # _portable_lib may be broken (e.g. rpath mismatch); try the
            # LLM runner which also contains the full ET runtime.
            from executorch.extension.llm.runner import _llm_runner  # noqa: F401

        sys.setdlopenflags(_old_flags)

    # This module is produced by the CMake target `executorch_ggml_backend_py`.
    # Import side-effect: registers backend via static init.
    from . import _ggml_backend as _native_backend  # noqa: F401
except Exception:
    # Keep import working for pure-Python workflows.
    _native_backend = None

from executorch_ggml.edge_pipeline import to_edge_rewrite_and_lower
from executorch_ggml.ggml_backend import GgmlBackend
from executorch_ggml.ggml_partitioner import GgmlPartitioner
from executorch_ggml.quantize import GgmlQuantConfig, GgmlQuantType

__all__ = [
    "GgmlPartitioner",
    "GgmlBackend",
    "GgmlQuantConfig",
    "GgmlQuantType",
    "to_edge_rewrite_and_lower",
]
