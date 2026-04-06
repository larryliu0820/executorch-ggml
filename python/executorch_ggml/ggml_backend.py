"""GgmlBackend: ExecuTorch BackendDetails that serialises an FX subgraph to ggml IR."""

from typing import Dict, List

import torch
from torch.export import ExportedProgram

from executorch.exir.backend.backend_details import BackendDetails, PreprocessResult
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir._serialize._named_data_store import NamedDataStore

from executorch_ggml.serialize import (
    IrTensor,
    OP_NONE,
    TYPE_F32,
    TYPE_F16,
    TYPE_I64,
    TYPE_I32,
    TYPE_BOOL,
    TYPE_BF16,
    TYPE_Q8_0,
    serialize_graph,
)

# Import helpers and symbolic expression utilities from ops package.
# These are re-exported here for backward compatibility with tests that
# import from executorch_ggml.ggml_backend directly.
from executorch_ggml.ops._helpers import (
    _concrete_int,
    _resolve_shape,
    _type_elem_size,
    _pytorch_shape_to_ggml_ne,
    _torch_dtype_to_ir_type,
)
from executorch_ggml.ops._sym_expr import (
    SYM_OP_PUSH_SYM,
    SYM_OP_PUSH_CONST,
    SYM_OP_ADD,
    SYM_OP_SUB,
    SYM_OP_MUL,
    SYM_OP_FLOORDIV,
    SYM_OP_MOD,
    SYM_OP_NEG,
    _sympy_to_bytecode,
    _eval_bytecode,
    _get_sym_dim_info,
    _sym_dim_info_ggml,
)
from executorch_ggml.ops._context import PreprocessContext

# Importing ops triggers registration of all operator handlers.
from executorch_ggml.ops import dispatch_op  # noqa: F401 (side-effect import)


class GgmlBackend(BackendDetails):
    """Converts a partitioned Edge-dialect subgraph into a ggml IR FlatBuffer."""

    @staticmethod
    def preprocess(
        edge_program: ExportedProgram,
        compile_specs: List[CompileSpec],
    ) -> PreprocessResult:
        graph_module = edge_program.graph_module
        graph = graph_module.graph

        # Named data store for weights/constants.
        data_store = NamedDataStore()

        # Parse quantization config and GGUF options from compile_specs (set by GgmlPartitioner).
        quant_config = None
        gguf_weight_map = None  # Optional: PyTorch FQN -> GGUF tensor name
        skip_weight_data = False  # When True, don't store weight bytes in PTE
        for spec in compile_specs:
            if spec.key == "ggml_quant_type":
                from executorch_ggml.quantize import GgmlQuantConfig, GgmlQuantType
                qtype_str = spec.value.decode()
                quant_config = GgmlQuantConfig(quant_type=GgmlQuantType(qtype_str))
            elif spec.key == "ggml_quant_skip" and quant_config is not None:
                quant_config.skip_patterns = set(spec.value.decode().split(","))
            elif spec.key == "gguf_weight_map":
                import json
                gguf_weight_map = json.loads(spec.value.decode())
            elif spec.key == "gguf_skip_weight_data":
                skip_weight_data = spec.value.decode() == "true"

        # Create the preprocessing context used by all operator handlers.
        ctx = PreprocessContext(edge_program, data_store, quant_config)

        # Convenience aliases for the loop below.
        node_to_id = ctx.node_to_id
        ir_tensors = ctx.ir_tensors
        sym_id_map = ctx.sym_id_map
        alloc_id = ctx.alloc_id

        sig = edge_program.graph_signature

        # Debug: show buffer mutation info
        from torch.export.graph_signature import OutputKind
        n_buf_mut = sum(1 for s in getattr(sig, "output_specs", [])
                        if s.kind == OutputKind.BUFFER_MUTATION)
        n_user_out = sum(1 for s in getattr(sig, "output_specs", [])
                         if s.kind == OutputKind.USER_OUTPUT)
        if n_buf_mut > 0 or ctx.mutated_buffer_fqns:
            print(f"[ggml preprocess] buffers_to_mutate: {len(ctx.mutated_buffer_fqns)}, "
                  f"output_specs: {n_buf_mut} BUFFER_MUTATION + {n_user_out} USER_OUTPUT")

        # Track runtime input index (for inputs that are NOT params/buffers)
        runtime_input_idx = 0

        # ---- Walk graph in topological order ----
        for node in graph.nodes:
            if node.op == "placeholder":
                node_name = node.name
                fqn = (ctx.param_map.get(node_name)
                       or ctx.buffer_map.get(node_name)
                       or ctx.lifted_const_map.get(node_name))
                tensor_from_state = fqn is not None and fqn in edge_program.state_dict
                tensor_from_constants = fqn is not None and (
                    fqn in ctx.ep_constants or fqn in ctx.ep_tensor_constants
                )
                is_constant = tensor_from_state or tensor_from_constants

                if is_constant:
                    # Prefix auto-generated constant names to avoid collisions
                    # in multi-method .pte files (different subgraphs may have
                    # duplicate _lifted_tensor_constant keys).
                    data_key = fqn
                    if fqn and fqn.startswith("_lifted_tensor_constant"):
                        sg_id = id(graph_module) & 0xFFFF
                        data_key = f"__ggml_sg{sg_id}_{fqn}"

                    # If GGUF weight map is provided, remap data_key to GGUF tensor name
                    if gguf_weight_map and fqn in gguf_weight_map:
                        data_key = gguf_weight_map[fqn]

                    if tensor_from_state:
                        tensor = edge_program.state_dict[fqn]
                    elif fqn in ctx.ep_constants:
                        tensor = ctx.ep_constants[fqn]
                    else:
                        tensor = ctx.ep_tensor_constants[fqn]
                    shape = list(tensor.shape)

                    t_cpu = tensor.detach().contiguous().cpu()
                    if t_cpu.dtype == torch.bool:
                        t_cpu = t_cpu.to(torch.float32)
                    elif t_cpu.dtype == torch.float32 and t_cpu.ndim >= 3:
                        pass

                    is_mutable = (fqn in ctx.mutated_buffer_fqns)

                    ir_type = _torch_dtype_to_ir_type(t_cpu.dtype)
                    # skip_weight_data: only skip constants that the GGUF data
                    # map can provide (actual model weights with a GGUF name).
                    # Small scalars, lifted constants, and KV caches must
                    # always be embedded — they don't exist in the GGUF file.
                    has_gguf_source = (gguf_weight_map is not None and fqn in gguf_weight_map)
                    skip_this = skip_weight_data and has_gguf_source

                    if quant_config is not None:
                        from executorch_ggml.quantize import (
                            should_quantize,
                            quantize_tensor,
                            GgmlQuantType,
                        )
                        if not is_mutable and should_quantize(
                            fqn, tuple(t_cpu.shape), t_cpu.dtype,
                            t_cpu.numel(), quant_config,
                        ):
                            import numpy as np
                            f32_data = t_cpu.to(torch.float32).numpy().flatten()
                            quant_bytes = quantize_tensor(
                                f32_data, quant_config.quant_type,
                            )
                            if not skip_this:
                                data_store.add_named_data(data_key, quant_bytes, alignment=64)
                            if quant_config.quant_type == GgmlQuantType.Q8_0:
                                ir_type = TYPE_Q8_0
                            tid = alloc_id()
                            ir_tensors.append(
                                IrTensor(
                                    tensor_id=tid,
                                    tensor_type=ir_type,
                                    ne=_pytorch_shape_to_ggml_ne(shape),
                                    op=OP_NONE,
                                    data_key=data_key,
                                    is_input=False,
                                    is_mutable=is_mutable,
                                )
                            )
                            node_to_id[node] = tid
                            continue

                    if not skip_this:
                        data_store.add_named_data(data_key, t_cpu, alignment=64)

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=ir_type,
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_NONE,
                            data_key=data_key,
                            is_input=False,
                            is_mutable=is_mutable,
                        )
                    )
                    node_to_id[node] = tid
                else:
                    # Runtime input
                    fake_val = node.meta.get("val")

                    if fake_val is None or not hasattr(fake_val, "shape"):
                        if isinstance(fake_val, (tuple, list)):
                            node_to_id[node] = [None] * len(fake_val)
                        continue

                    shape = _resolve_shape(fake_val)
                    ggml_sym, ggml_exprs = _sym_dim_info_ggml(fake_val, sym_id_map)

                    tid = alloc_id()
                    in_dtype = (
                        getattr(fake_val, "dtype", torch.float32)
                        if fake_val is not None and hasattr(fake_val, "dtype")
                        else torch.float32
                    )
                    if in_dtype == torch.int64:
                        in_dtype = torch.int32
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=_torch_dtype_to_ir_type(in_dtype),
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_NONE,
                            is_input=True,
                            input_index=runtime_input_idx,
                            sym_dim_ids=ggml_sym,
                            sym_dim_exprs=ggml_exprs,
                        )
                    )
                    node_to_id[node] = tid
                    runtime_input_idx += 1

            elif node.op == "call_function":
                target = node.target
                target_str = str(target)

                # --- Dispatch to registered operator handlers ---
                if dispatch_op(ctx, node, target_str):
                    pass  # Handler matched and executed.

                elif callable(target) and "getitem" in target_str:
                    # Generic getitem handler for tuple outputs
                    # (layer_norm, batch_norm, split_with_sizes_copy)
                    src_node = node.args[0]
                    idx = int(node.args[1])
                    src_val = node_to_id.get(src_node)

                    if isinstance(src_val, list):
                        val = src_val[idx] if idx < len(src_val) else None
                        if val is not None:
                            node_to_id[node] = val
                    elif isinstance(src_val, int):
                        if idx == 0:
                            node_to_id[node] = src_val
                    elif src_val is None:
                        pass
                    else:
                        raise RuntimeError(
                            f"getitem: unexpected source type {type(src_val)} for {src_node}"
                        )

                else:
                    raise RuntimeError(
                        f"GgmlBackend.preprocess: unsupported op {target} "
                        f"(node={node.name}, op={node.op})"
                    )

            elif node.op == "output":
                from torch.export.graph_signature import OutputKind
                output_specs = getattr(sig, "output_specs", [])
                buffer_mutation_indices = set()
                for spec_idx, spec in enumerate(output_specs):
                    if spec.kind == OutputKind.BUFFER_MUTATION:
                        buffer_mutation_indices.add(spec_idx)

                output_args = node.args[0]
                if not isinstance(output_args, (list, tuple)):
                    output_args = [output_args]
                user_out_idx = 0
                for out_idx, out_node in enumerate(output_args):
                    if out_idx in buffer_mutation_indices:
                        continue
                    if out_node in node_to_id:
                        tid = node_to_id[out_node]
                        for ir_t in ir_tensors:
                            if ir_t.id == tid:
                                ir_t.is_output = True
                                ir_t.input_index = user_out_idx
                                break
                    user_out_idx += 1

        # Serialize to FlatBuffer
        fb_bytes = serialize_graph(ir_tensors)
        return PreprocessResult(
            processed_bytes=fb_bytes,
            data_store_output=data_store.get_named_data_store_output(),
        )
