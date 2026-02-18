# ExecuTorch ggml Backend

## Goal

Create a new repo `larryliu0820/executorch-ggml` that implements an ExecuTorch backend delegating to ggml. Start with support for `nn.Linear` + `nn.LeakyReLU`.

## Repo Structure

```
executorch-ggml/
├── CMakeLists.txt                     # Top-level CMake (links ggml + executorch)
├── setup.py                           # Python package
├── schema/
│   └── ggml_ir.fbs                    # FlatBuffer schema for ggml-ir
├── python/
│   └── executorch_ggml/
│       ├── __init__.py
│       ├── ggml_partitioner.py        # Partitioner: tags supported FX nodes
│       ├── ggml_backend.py            # BackendDetails: preprocess → serialize
│       └── serialize.py               # FlatBuffer serialization helpers
├── runtime/
│   ├── ggml_backend.h
│   ├── ggml_backend.cpp               # C++ BackendInterface: init/execute/destroy
│   └── CMakeLists.txt
└── tests/
    └── test_linear_leaky_relu.py      # End-to-end test
```

## Step 1: Create the GitHub repo

Create `larryliu0820/executorch-ggml` via `gh repo create`.

## Step 2: FlatBuffer Schema (`schema/ggml_ir.fbs`)

Defines the serialized IR format. Key tables:

- **`Tensor`**: `id`, `type` (F32/F16), `ne` (shape in ggml order, up to 4 dims), `op` (opcode), `src_ids` (input tensor IDs), `op_params` (raw bytes, e.g. negative\_slope), `data` (constant weight/bias bytes), `is_input`/`is_output` flags, `input_index`.
- **`GgmlGraph`**: `tensors` (in topological order), `n_threads`.
- **OpCode enum**: `NONE=0, ADD=1, MUL_MAT=2, LEAKY_RELU=3`.

Tensors are in topological order so deserialization can resolve `src_ids` in a single pass.

## Step 3: Python Partitioner (`ggml_partitioner.py`)

Subclass `executorch.exir.backend.partitioner.Partitioner`.

**Supported ATen ops** (what `nn.Linear + nn.LeakyReLU` decomposes to after `torch.export`):
- `aten.t.default` — weight transpose
- `aten.addmm.default` — `bias + input @ weight^T`
- `aten.mm.default` — `input @ weight` (no bias case)
- `aten.leaky_relu.default`

**Algorithm**:
1. Walk the FX graph nodes
2. For each `call_function` node whose target is in the supported set, tag it
3. Group connected supported nodes into one partition (union-find on data-flow edges)
4. Call `tag_constant_data(exported_program)` to tag params/buffers
5. Return `PartitionResult` with `DelegationSpec(backend_id="GgmlBackend", ...)`

## Step 4: Python Backend (`ggml_backend.py`)

Subclass `executorch.exir.backend.backend_details.BackendDetails`.

Implement `preprocess(edge_program, compile_specs) -> PreprocessResult`:

1. Walk the partitioned FX subgraph in topological order
2. For each `placeholder` node: determine if it's a runtime input or constant (weight/bias). For constants, extract tensor data via `edge_program.graph_signature.inputs_to_parameters` / `inputs_to_buffers` and `edge_program.state_dict`.
3. For each `call_function` node, map ATen op → ggml IR op:
   - **`aten.t.default`**: No-op — ggml\_mul\_mat already expects weight in `ne=[in, out]` layout, which matches PyTorch's `[out, in]` reversed. "Look through" the transpose.
   - **`aten.addmm(bias, input, weight_t)`**: Emit TWO IR tensors:
     1. `MUL_MAT(original_weight, input)` — resolve through `aten.t` to get original weight
     2. `ADD(mul_mat_result, bias)`
   - **`aten.mm(input, weight_t)`**: Emit `MUL_MAT(original_weight, input)`
   - **`aten.leaky_relu(x, negative_slope)`**: Emit `LEAKY_RELU(x)` with `op_params = pack("<f", negative_slope)`
4. Mark output nodes
5. Serialize tensor list to FlatBuffer → `PreprocessResult(processed_bytes=...)`

**Critical: shape mapping** — PyTorch `[d0, d1, ..., dn]` → ggml `ne = [dn, ..., d1, d0]` (reversed, padded to 4 dims with 1s).

## Step 5: C++ Runtime (`runtime/ggml_backend.cpp`)

Implement `executorch::runtime::BackendInterface`:

### `init(context, processed, compile_specs) -> DelegateHandle*`
1. Parse FlatBuffer from `processed->data()`
2. Calculate ggml context memory: `n_tensors * ggml_tensor_overhead() + graph_overhead + constant_data_size`
3. `ggml_init(params)` with `no_alloc = false`
4. Iterate tensors in topological order:
   - Leaf tensors (`op == NONE`): `ggml_new_tensor()`, copy constant data via `memcpy`
   - Op tensors: call the corresponding ggml builder function:
     - `ADD` → `ggml_add(ctx, src0, src1)`
     - `MUL_MAT` → `ggml_mul_mat(ctx, src0, src1)`
     - `LEAKY_RELU` → `ggml_leaky_relu(ctx, src0, negative_slope, false)`
5. Track input tensors (by `input_index`) and output tensors
6. Build `ggml_cgraph` via `ggml_new_graph()` + `ggml_build_forward_expand(graph, output)`
7. Return `GgmlDelegateHandle` containing ctx, graph, input/output tensor pointers

### `execute(context, handle, args) -> Error`
1. Copy input data: `memcpy(ggml_input->data, et_tensor.const_data_ptr(), nbytes)`
2. `ggml_graph_compute_with_ctx(ctx, graph, n_threads)` (from `ggml-cpu.h`)
3. Copy output data: `memcpy(et_output.mutable_data_ptr(), ggml_output->data, nbytes)`

### `destroy(handle)`
1. `ggml_free(compute_ctx)`, `ggml_free(ctx)`, `delete handle`

### Registration
```cpp
namespace {
auto cls = executorch::runtime::register_backend({"GgmlBackend", new GgmlBackendInterface()});
}
```

## Step 6: Build System

**CMakeLists.txt**: Takes `LLAMA_CPP_DIR` and `EXECUTORCH_DIR` as cache variables, `add_subdirectory(${LLAMA_CPP_DIR}/ggml)` to build ggml. Uses `flatc --cpp` custom command to generate `ggml_ir_generated.h`. Links against `ggml`, `ggml-cpu`, `executorch`.

**setup.py**: pip-installable Python package with `flatbuffers` and `torch` dependencies.

## Step 7: Test (`tests/test_linear_leaky_relu.py`)

```python
class LinearLeakyReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 8)
        self.leaky_relu = nn.LeakyReLU(0.1)
    def forward(self, x):
        return self.leaky_relu(self.linear(x))

# Export → partition → lower → serialize → verify delegation happened
exported = export(model, (torch.randn(2, 4),))
edge = to_edge_transform_and_lower(exported, partitioner=[GgmlPartitioner()])
# Assert delegated call nodes exist in the graph
# Serialize to .pte and verify non-empty
```

## Key Technical Insights

1. **`aten.t` is a no-op**: ggml's `mul_mat(W, x)` expects `W.ne[0] == x.ne[0]`. PyTorch weight `[out, in]` reversed to ggml `ne=[in, out]` already matches. The transpose in the FX graph is "looked through".

2. **`addmm` becomes 2 IR nodes**: One `MUL_MAT` + one `ADD`. The FX node maps to the ADD tensor ID.

3. **Shape reversal**: PyTorch outermost-first → ggml innermost-first (reversed + pad to 4D).

4. **Memory**: ggml context owns all memory. `no_alloc=false` means ggml allocates data buffers. Sufficient for initial scope.

## Implementation Order

1. Create repo + scaffold
2. Write FlatBuffer schema
3. Implement `ggml_partitioner.py`
4. Implement `ggml_backend.py` + `serialize.py`
5. Implement `ggml_backend.cpp` + `ggml_backend.h`
6. Write CMakeLists.txt + setup.py
7. Write test
8. Verify end-to-end
