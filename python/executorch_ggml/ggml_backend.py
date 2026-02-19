"""GgmlBackend: ExecuTorch BackendDetails that serialises an FX subgraph to ggml IR."""

from typing import Dict, List

import torch
from torch.export import ExportedProgram

from executorch.exir.backend.backend_details import BackendDetails, PreprocessResult
from executorch.exir.backend.compile_spec_schema import CompileSpec

from executorch_ggml.serialize import (
    IrTensor,
    OP_ADD,
    OP_LEAKY_RELU,
    OP_MUL_MAT,
    OP_NONE,
    TYPE_F32,
    TYPE_F16,
    pack_float,
    serialize_graph,
)


def _pytorch_shape_to_ggml_ne(shape: List[int]) -> List[int]:
    """PyTorch [d0, d1, ..., dn] → ggml ne [dn, ..., d1, d0], padded to 4D."""
    ne = list(reversed(shape))
    while len(ne) < 4:
        ne.append(1)
    return ne[:4]


def _torch_dtype_to_ir_type(dtype: torch.dtype) -> int:
    if dtype == torch.float16:
        return TYPE_F16
    return TYPE_F32


class GgmlBackend(BackendDetails):
    """Converts a partitioned Edge-dialect subgraph into a ggml IR FlatBuffer."""

    @staticmethod
    def preprocess(
        edge_program: ExportedProgram,
        compile_specs: List[CompileSpec],
    ) -> PreprocessResult:
        graph_module = edge_program.graph_module
        graph = graph_module.graph

        # Maps from FX node → IR tensor id
        node_to_id: Dict[torch.fx.Node, int] = {}
        ir_tensors: List[IrTensor] = []
        next_id = 0

        def alloc_id() -> int:
            nonlocal next_id
            tid = next_id
            next_id += 1
            return tid

        # Build mapping from param/buffer placeholder names → FQN → tensor data
        sig = edge_program.graph_signature
        param_map = dict(sig.inputs_to_parameters)   # node_name → param FQN
        buffer_map = dict(sig.inputs_to_buffers)      # node_name → buffer FQN

        # Track runtime input index (for inputs that are NOT params/buffers)
        runtime_input_idx = 0

        # ---- Pass 1: Walk graph in topological order ----
        for node in graph.nodes:
            if node.op == "placeholder":
                node_name = node.name
                is_constant = node_name in param_map or node_name in buffer_map

                if is_constant:
                    # Resolve the parameter/buffer FQN
                    fqn = param_map.get(node_name) or buffer_map.get(node_name)
                    tensor = edge_program.state_dict[fqn]
                    tensor_contig = tensor.contiguous().to(torch.float32)
                    data_bytes = tensor_contig.detach().cpu().numpy().tobytes()
                    shape = list(tensor.shape)

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=TYPE_F32,
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_NONE,
                            data=data_bytes,
                            is_input=False,
                        )
                    )
                    node_to_id[node] = tid
                else:
                    # Runtime input
                    fake_val = node.meta.get("val")
                    if fake_val is not None:
                        shape = list(fake_val.shape)
                    else:
                        shape = []

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=TYPE_F32,
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_NONE,
                            is_input=True,
                            input_index=runtime_input_idx,
                        )
                    )
                    node_to_id[node] = tid
                    runtime_input_idx += 1

            elif node.op == "call_function":
                target = node.target
                target_str = str(target)

                if "aten.t.default" in target_str or "aten.permute_copy.default" in target_str:
                    # Transpose is a no-op: ggml_mul_mat expects weight in
                    # ne=[in_features, out_features] which matches PyTorch's
                    # [out_features, in_features] after shape reversal.
                    # "Look through" the transpose.
                    src_node = node.args[0]
                    node_to_id[node] = node_to_id[src_node]

                elif "aten.mm.default" in target_str:
                    # mm(input, weight_t) → MUL_MAT(original_weight, input)
                    input_node, weight_t_node = node.args
                    weight_id = node_to_id[weight_t_node]
                    input_id = node_to_id[input_node]

                    fake_val = node.meta.get("val")
                    shape = list(fake_val.shape) if fake_val is not None else []

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=TYPE_F32,
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_MUL_MAT,
                            src_ids=[weight_id, input_id],
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.addmm.default" in target_str:
                    # addmm(bias, input, weight_t)
                    # → MUL_MAT(original_weight, input) then ADD(result, bias)
                    bias_node, input_node, weight_t_node = node.args

                    weight_id = node_to_id[weight_t_node]
                    input_id = node_to_id[input_node]
                    bias_id = node_to_id[bias_node]

                    fake_val = node.meta.get("val")
                    shape = list(fake_val.shape) if fake_val is not None else []

                    # First: MUL_MAT
                    mm_id = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=mm_id,
                            tensor_type=TYPE_F32,
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_MUL_MAT,
                            src_ids=[weight_id, input_id],
                        )
                    )

                    # Second: ADD
                    add_id = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=add_id,
                            tensor_type=TYPE_F32,
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_ADD,
                            src_ids=[mm_id, bias_id],
                        )
                    )
                    node_to_id[node] = add_id

                elif "aten.leaky_relu.default" in target_str:
                    src_node = node.args[0]
                    negative_slope = node.args[1] if len(node.args) > 1 else 0.01
                    src_id = node_to_id[src_node]

                    fake_val = node.meta.get("val")
                    shape = list(fake_val.shape) if fake_val is not None else []

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=TYPE_F32,
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_LEAKY_RELU,
                            src_ids=[src_id],
                            op_params=pack_float(float(negative_slope)),
                        )
                    )
                    node_to_id[node] = tid

                else:
                    raise RuntimeError(
                        f"GgmlBackend.preprocess: unsupported op {target}"
                    )

            elif node.op == "output":
                # Mark output tensors
                output_args = node.args[0]
                if not isinstance(output_args, (list, tuple)):
                    output_args = [output_args]
                for out_node in output_args:
                    if out_node in node_to_id:
                        tid = node_to_id[out_node]
                        # Find the IrTensor and mark it as output
                        for ir_t in ir_tensors:
                            if ir_t.id == tid:
                                ir_t.is_output = True
                                break

        # Serialize to FlatBuffer
        fb_bytes = serialize_graph(ir_tensors)
        return PreprocessResult(processed_bytes=fb_bytes)
