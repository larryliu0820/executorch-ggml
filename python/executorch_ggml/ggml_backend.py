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
    OP_CONV_2D,
    OP_CONV_2D_DW,
    OP_HARDTANH,
    OP_MEAN,
    OP_VIEW,
    OP_PERMUTE,
    TYPE_F32,
    TYPE_F16,
    pack_float,
    pack_conv2d_params,
    pack_hardtanh_params,
    pack_mean_params,
    pack_view_params,
    pack_permute_params,
    serialize_graph,
)

from executorch.exir._serialize._named_data_store import NamedDataStore


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

        # NOTE: BN folding is now intended to be done via a rewrite pass
        # (BatchNormFoldingRewritePass) *before* partitioning/lowering.
        # The previous "no-op BN" backend fusion approach breaks ExportedProgram
        # structured-output semantics (BN returns a tuple) during to_executorch().

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

        # Named data store for weights/constants.
        # We follow ExecuTorch convention: store blobs in NamedDataStore and refer
        # to them by key from the delegate blob. Keys are state_dict FQNs.
        data_store = NamedDataStore()

        # ---- Pass 1: Walk graph in topological order ----
        for node in graph.nodes:
            if node.op == "placeholder":
                node_name = node.name
                is_constant = node_name in param_map or node_name in buffer_map

                if is_constant:
                    # Resolve the parameter/buffer FQN
                    fqn = param_map.get(node_name) or buffer_map.get(node_name)
                    tensor = edge_program.state_dict[fqn]
                    shape = list(tensor.shape)

                    # Store the tensor bytes externally (dedup handled by NamedDataStore).
                    # Use contiguous CPU tensor for stable storage.
                    t_cpu = tensor.detach().contiguous().cpu()
                    # Store inside the .pte (no external_tag).
                    data_store.add_named_data(fqn, t_cpu, alignment=64)

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=_torch_dtype_to_ir_type(t_cpu.dtype),
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_NONE,
                            data_key=fqn,
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

                # BN folding is handled by BatchNormFoldingRewritePass before partitioning.
                # BN ops should not appear inside a delegated region.

                # -----------------------------------------------------------------
                # Existing supported ops
                # -----------------------------------------------------------------
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

                elif "aten.convolution.default" in target_str or "aten.conv2d.default" in target_str:
                    # aten.convolution.default or aten.conv2d.default
                    # Args: (input, weight, bias?, stride, padding, dilation, transposed, output_padding, groups)
                    # For conv2d: (input, weight, bias, stride, padding, dilation, groups)

                    if "aten.convolution.default" in target_str:
                        if len(node.args) < 9:
                            raise RuntimeError(f"Expected 9 args for convolution, got {len(node.args)}")
                        input_node = node.args[0]
                        weight_node = node.args[1]
                        bias_node = node.args[2] if len(node.args) > 2 and node.args[2] is not None else None
                        stride = list(node.args[3]) if len(node.args) > 3 else [1, 1]
                        padding = list(node.args[4]) if len(node.args) > 4 else [0, 0]
                        dilation = list(node.args[5]) if len(node.args) > 5 else [1, 1]
                        transposed = node.args[6] if len(node.args) > 6 else False
                        output_padding = list(node.args[7]) if len(node.args) > 7 else [0, 0]
                        groups = node.args[8] if len(node.args) > 8 else 1
                    else:  # conv2d
                        if len(node.args) < 7:
                            raise RuntimeError(f"Expected 7 args for conv2d, got {len(node.args)}")
                        input_node = node.args[0]
                        weight_node = node.args[1]
                        bias_node = node.args[2] if len(node.args) > 2 and node.args[2] is not None else None
                        stride = list(node.args[3]) if len(node.args) > 3 else [1, 1]
                        padding = list(node.args[4]) if len(node.args) > 4 else [0, 0]
                        dilation = list(node.args[5]) if len(node.args) > 5 else [1, 1]
                        groups = node.args[6] if len(node.args) > 6 else 1
                        transposed = False
                        output_padding = [0, 0]

                    if transposed:
                        raise RuntimeError("Transposed convolution not yet supported")

                    # Check if we have folded params for this conv
                    weight_id = node_to_id[weight_node]
                    if bias_node is not None:
                        bias_id = node_to_id[bias_node]
                    else:
                        bias_id = None

                    # If this conv has folded BN params, replace the weight/bias tensors
                    # Note: We already have the folded params from BatchNormFoldingPass
                    # We need to check if the weight/bias placeholders should be replaced
                    # For now, we use the weight/bias as-is (they should already be folded by the pass)

                    input_id = node_to_id[input_node]

                    fake_val = node.meta.get("val")
                    shape = list(fake_val.shape) if fake_val is not None else []

                    # Determine if this is depthwise conv
                    # Depthwise: groups > 1 and groups == in_channels
                    # For MobileNetV2, depthwise conv has groups == in_channels
                    is_depthwise = groups > 1

                    # Pack parameters
                    op_params = pack_conv2d_params(stride, padding, dilation, groups)

                    # Build src_ids: [weight, input, bias?]
                    if bias_id is not None:
                        src_ids = [weight_id, input_id, bias_id]
                    else:
                        src_ids = [weight_id, input_id]

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=TYPE_F32,
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_CONV_2D_DW if is_depthwise else OP_CONV_2D,
                            src_ids=src_ids,
                            op_params=op_params,
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.hardtanh.default" in target_str or "aten.clamp.default" in target_str:
                    # hardtanh(x, min_val, max_val) or clamp(x, min, max)
                    src_node = node.args[0]
                    min_val = node.args[1] if len(node.args) > 1 else -1.0
                    max_val = node.args[2] if len(node.args) > 2 else 1.0
                    src_id = node_to_id[src_node]

                    fake_val = node.meta.get("val")
                    shape = list(fake_val.shape) if fake_val is not None else []

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=TYPE_F32,
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_HARDTANH,
                            src_ids=[src_id],
                            op_params=pack_hardtanh_params(float(min_val), float(max_val)),
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.mean.dim" in target_str or "aten._mean_dim.default" in target_str:
                    # mean(x, dim, keepdim)
                    src_node = node.args[0]
                    dim = node.args[1] if len(node.args) > 1 else -1
                    if isinstance(dim, (list, tuple)):
                        # For now, only support single dim
                        if len(dim) != 1:
                            raise RuntimeError(f"Only single-dim mean supported, got dim={dim}")
                        dim = dim[0]
                    src_id = node_to_id[src_node]

                    fake_val = node.meta.get("val")
                    shape = list(fake_val.shape) if fake_val is not None else []

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=TYPE_F32,
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_MEAN,
                            src_ids=[src_id],
                            op_params=pack_mean_params(int(dim)),
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.view.default" in target_str or "aten._unsafe_view.default" in target_str or "aten.reshape.default" in target_str:
                    # view(x, new_shape) or reshape(x, new_shape)
                    src_node = node.args[0]
                    new_shape = list(node.args[1])
                    src_id = node_to_id[src_node]

                    fake_val = node.meta.get("val")
                    shape = list(fake_val.shape) if fake_val is not None else []

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=TYPE_F32,
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_VIEW,
                            src_ids=[src_id],
                            op_params=pack_view_params(new_shape),
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.permute.default" in target_str or "aten.permute_copy.default" in target_str:
                    # permute(x, dims)
                    src_node = node.args[0]
                    perm = list(node.args[1])
                    src_id = node_to_id[src_node]

                    fake_val = node.meta.get("val")
                    shape = list(fake_val.shape) if fake_val is not None else []

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=TYPE_F32,
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_PERMUTE,
                            src_ids=[src_id],
                            op_params=pack_permute_params(perm),
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
        return PreprocessResult(
            processed_bytes=fb_bytes,
            data_store_output=data_store.get_named_data_store_output(),
        )
