#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <ggml-backend.h>

#ifdef GGML_USE_METAL
#include <ggml-metal.h>
#endif

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/extension/memory_allocator/malloc_memory_allocator.h>
#include <executorch/runtime/core/data_loader.h>
#include <executorch/extension/data_loader/file_data_loader.h>
#include "../runtime/gguf_data_map.h"

namespace py = pybind11;

using executorch::extension::Module;
using executorch::extension::FileDataLoader;
using executorch::runtime::Error;
using executorch::runtime::Program;

/// Thin wrapper: loads a .pte program, resolves weights from a .gguf via
/// GGUFNamedDataMap, and exposes forward() to Python.
class GGUFRuntime {
 public:
  GGUFRuntime(const std::string& pte_path, const std::string& gguf_path) {
    // 1. Load GGUF as NamedDataMap.
    auto gguf_res = executorch_ggml::GGUFNamedDataMap::load(gguf_path);
    if (!gguf_res.ok()) {
      throw std::runtime_error("Failed to open GGUF: " + gguf_path);
    }
    gguf_map_ = std::move(gguf_res.get());
    fprintf(stderr, "[GGUFRuntime] GGUF: %s (%lld tensors)\n",
            gguf_path.c_str(), (long long)gguf_map_->num_tensors());

    // 2. Create Module from PTE, passing GGUF data map path is not possible
    //    directly, so we use Module but manually load the method with our map.
    module_ = std::make_unique<Module>(pte_path);
    auto err = module_->load();
    if (err != Error::Ok) {
      throw std::runtime_error("Failed to load PTE: " + pte_path +
                               " (error 0x" + std::to_string((int)err) + ")");
    }

    // 3. Load forward method — Module::load_method uses its internal
    //    merged_data_map_. We need to override that. Access the program
    //    directly and load the method ourselves.
    //    But Module::load_method doesn't accept a NamedDataMap*.
    //    So we use a small trick: Module stores the program_ as shared_ptr.
    //    We call program->load_method directly.

    // For now, use Module's standard load which works when PTE has weights.
    // When PTE is weight-less, the data_map provides from GGUF.
    // We need to patch Module to support external NamedDataMap.
    // Simplest: set the module's merged_data_map_ before load_method.
    // This requires making Module a friend or using the data_files path.

    // PRAGMATIC APPROACH: write GGUF data_keys that we know the Module needs
    // into a temp file in .ptd format... NO — user said don't convert.
    //
    // ACTUAL APPROACH: use the Module constructor that takes data_map_loader.
    // We'll create a DataLoader adapter around GGUFNamedDataMap... NO — Module
    // wraps it in FlatTensorDataMap.
    //
    // REAL SIMPLEST: directly use program_->load_method with our NamedDataMap.
    // We just need to replicate what Module::load_method does internally.
    has_forward_ = false;
  }

  // Load the forward method with GGUF as the NamedDataMap.
  void load_forward() {
    auto program = module_->program();
    if (!program) {
      throw std::runtime_error("Program not loaded");
    }

    // Get method metadata for memory planning.
    auto method_meta_res = program->method_meta("forward");
    if (!method_meta_res.ok()) {
      throw std::runtime_error("Failed to get method meta for forward");
    }
    auto& method_meta = method_meta_res.get();

    // Allocate planned memory.
    planned_buffers_.clear();
    planned_spans_.clear();
    for (size_t i = 0; i < method_meta.num_memory_planned_buffers(); ++i) {
      size_t buf_size = method_meta.memory_planned_buffer_size(i).get();
      planned_buffers_.emplace_back(buf_size);
      planned_spans_.emplace_back(planned_buffers_.back().data(), buf_size);
    }
    planned_memory_ = std::make_unique<executorch::runtime::HierarchicalAllocator>(
        executorch::runtime::Span(planned_spans_.data(), planned_spans_.size()));

    memory_allocator_ = std::make_unique<executorch::extension::MallocMemoryAllocator>();
    temp_allocator_ = std::make_unique<executorch::extension::MallocMemoryAllocator>();
    memory_manager_ = std::make_unique<executorch::runtime::MemoryManager>(
        memory_allocator_.get(), planned_memory_.get(), temp_allocator_.get());

    auto res = program->load_method(
        "forward",
        memory_manager_.get(),
        /*event_tracer=*/nullptr,
        gguf_map_.get());

    if (!res.ok()) {
      throw std::runtime_error(
          "Failed to load forward with GGUF data (error 0x" +
          std::to_string((int)res.error()) + ")");
    }
    method_ = std::make_unique<executorch::runtime::Method>(std::move(res.get()));
    has_forward_ = true;
    fprintf(stderr, "[GGUFRuntime] forward loaded with GGUF weights\n");
  }

  // Forward pass matching ExecuTorch's own pybind pattern.
  py::list forward(py::list py_inputs) {
    if (!has_forward_) {
      load_forward();
    }

    // Build EValue inputs — keep TensorPtrs alive through execute().
    std::vector<executorch::extension::TensorPtr> input_tensors;
    std::vector<executorch::runtime::EValue> cpp_inputs;
    input_tensors.reserve(py::len(py_inputs));
    cpp_inputs.reserve(py::len(py_inputs));

    for (size_t i = 0; i < py::len(py_inputs); ++i) {
      py::object t = py_inputs[i];
      auto data_ptr_val = t.attr("data_ptr")().cast<intptr_t>();
      auto shape = t.attr("shape").cast<std::vector<int64_t>>();
      bool is_long = t.attr("dtype").attr("__str__")().cast<std::string>() == "torch.int64";

      auto scalar_type = is_long ? executorch::aten::ScalarType::Long
                                 : executorch::aten::ScalarType::Float;

      std::vector<int> sizes(shape.begin(), shape.end());
      // Contiguous strides
      std::vector<int> strides(shape.size());
      int stride = 1;
      for (int d = (int)shape.size() - 1; d >= 0; --d) {
        strides[d] = stride;
        stride *= sizes[d];
      }
      // Contiguous dim order
      std::vector<uint8_t> dim_order(shape.size());
      for (size_t d = 0; d < shape.size(); ++d) dim_order[d] = d;

      auto tensor = executorch::extension::for_blob(
              reinterpret_cast<void*>(data_ptr_val), std::move(sizes), scalar_type)
          .strides(std::move(strides))
          .dim_order(std::move(dim_order))
          .dynamism(executorch::aten::TensorShapeDynamism::STATIC)
          .make_tensor_ptr();

      input_tensors.push_back(tensor);
      cpp_inputs.emplace_back(input_tensors.back());
    }

    executorch::aten::ArrayRef<executorch::runtime::EValue> input_ref(
        cpp_inputs.data(), cpp_inputs.size());
    auto err = method_->set_inputs(input_ref);
    if (err != Error::Ok) {
      throw std::runtime_error("set_inputs failed (error 0x" + std::to_string((int)err) + ")");
    }

    err = method_->execute();
    if (err != Error::Ok) {
      throw std::runtime_error("execute failed (error 0x" + std::to_string((int)err) + ")");
    }

    // Collect outputs as numpy arrays.
    py::list results;
    auto n_outputs = method_->outputs_size();
    for (size_t i = 0; i < n_outputs; ++i) {
      auto eval = method_->get_output(i);
      if (eval.isTensor()) {
        auto& et = eval.toTensor();
        std::vector<ssize_t> np_shape(et.sizes().begin(), et.sizes().end());
        size_t elem_size = 4;
        if (et.scalar_type() == executorch::aten::ScalarType::Long) elem_size = 8;
        std::vector<ssize_t> np_strides;
        ssize_t s = elem_size;
        for (int d = (int)np_shape.size() - 1; d >= 0; --d) {
          np_strides.insert(np_strides.begin(), s);
          s *= np_shape[d];
        }
        std::string fmt = (et.scalar_type() == executorch::aten::ScalarType::Long) ? "l" : "f";
        auto buf_info = py::buffer_info(
            et.mutable_data_ptr(), elem_size, fmt,
            np_shape.size(), np_shape, np_strides);
        results.append(py::array(buf_info).attr("copy")());
      }
    }
    return results;
  }

  int64_t num_gguf_tensors() const { return gguf_map_->num_tensors(); }
  const std::string& gguf_path() const { return gguf_map_->path(); }

 private:
  std::unique_ptr<Module> module_;
  std::unique_ptr<executorch_ggml::GGUFNamedDataMap> gguf_map_;
  std::unique_ptr<executorch::runtime::Method> method_;
  bool has_forward_ = false;

  // Memory management (kept alive for the lifetime of the method).
  std::vector<std::vector<uint8_t>> planned_buffers_;
  std::vector<executorch::runtime::Span<uint8_t>> planned_spans_;
  std::unique_ptr<executorch::runtime::HierarchicalAllocator> planned_memory_;
  std::unique_ptr<executorch::runtime::MemoryManager> memory_manager_;
  std::unique_ptr<executorch::extension::MallocMemoryAllocator> memory_allocator_;
  std::unique_ptr<executorch::extension::MallocMemoryAllocator> temp_allocator_;
};


PYBIND11_MODULE(_ggml_backend, m) {
  m.doc() = "ExecuTorch ggml backend (pybind11 shim). Importing registers GgmlBackend.";
  m.def("ping", []() { return true; }, "Sanity check that the extension is importable.");

  // GGUFRuntime: load .pte + .gguf and run inference.
  py::class_<GGUFRuntime>(m, "GGUFRuntime")
    .def(py::init<const std::string&, const std::string&>(),
         py::arg("pte_path"), py::arg("gguf_path"))
    .def("load_forward", &GGUFRuntime::load_forward)
    .def("forward", &GGUFRuntime::forward)
    .def("num_gguf_tensors", &GGUFRuntime::num_gguf_tensors)
    .def("gguf_path", &GGUFRuntime::gguf_path);

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
