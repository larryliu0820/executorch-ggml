#include <pybind11/pybind11.h>

// Importing this extension should register the backend via the static
// registration block in runtime/ggml_backend.cpp.
//
// This module intentionally does not expose any ExecuTorch APIs.

namespace py = pybind11;

PYBIND11_MODULE(_ggml_backend, m) {
  m.doc() = "ExecuTorch ggml backend (pybind11 shim). Importing registers GgmlBackend.";
  m.def("ping", []() { return true; }, "Sanity check that the extension is importable.");
}
