#include <pybind11/pybind11.h>

#include <ggml-backend.h>

#ifdef GGML_USE_METAL
#include <ggml-metal.h>
#endif

// Importing this extension should register the backend via the static
// registration block in runtime/ggml_backend.cpp.
//
// This module intentionally does not expose any ExecuTorch APIs.

namespace py = pybind11;

PYBIND11_MODULE(_ggml_backend, m) {
  m.doc() = "ExecuTorch ggml backend (pybind11 shim). Importing registers GgmlBackend.";
  m.def("ping", []() { return true; }, "Sanity check that the extension is importable.");

  // Check if Metal support was compiled in
  m.def("has_metal_support", []() {
#ifdef GGML_USE_METAL
    return true;
#else
    return false;
#endif
  }, "Returns True if Metal support was compiled into the backend.");

  // Check if Metal is available at runtime (device can be initialized)
  m.def("is_metal_available", []() {
#ifdef GGML_USE_METAL
    ggml_backend_t backend = ggml_backend_metal_init();
    if (backend) {
      ggml_backend_free(backend);
      return true;
    }
#endif
    return false;
  }, "Returns True if Metal backend can be initialized on this system.");
}
