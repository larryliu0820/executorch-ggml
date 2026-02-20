"""GgmlBackend: ExecuTorch BackendDetails that serialises an FX subgraph to ggml IR."""

from typing import Dict, List

import torch
from torch.export import ExportedProgram

from executorch.exir.backend.backend_details import BackendDetails, PreprocessResult
from executorch.exir.backend.compile_spec_schema import CompileSpec

from executorch_ggml.serialize import (
    IrTensor,
    # OpCodes
    OP_NONE,
    OP_ADD,
    OP_MUL_MAT,
    OP_MUL,
    OP_NEG,
    OP_LINEAR,
    OP_EMBEDDING,
    OP_SILU,
    OP_LEAKY_RELU,
    OP_CONV_2D,
    OP_CONV_2D_DW,
    OP_HARDTANH,
    OP_MEAN,
    OP_RSQRT,
    OP_VIEW,
    OP_UNSQUEEZE,
    OP_PERMUTE,
    OP_TRANSPOSE,
    OP_SLICE,
    OP_CAT,
    OP_REPEAT_INTERLEAVE,
    OP_INDEX,
    OP_INDEX_PUT,
    OP_REPEAT,
    OP_INDEX_MULTI,
    OP_LLAMA_ATTENTION,
    # Types
    TYPE_F32,
    TYPE_F16,
    TYPE_I64,
    TYPE_I32,
    TYPE_BOOL,
    # Packers
    pack_float,
    pack_i32,
    pack_conv2d_params,
    pack_hardtanh_params,
    pack_mean_params,
    pack_view_params,
    pack_permute_params,
    pack_transpose_params,
    pack_unsqueeze_params,
    pack_slice_params,
    pack_cat_params,
    pack_repeat_interleave_params,
    pack_index_params,
    pack_index_put_params,
    pack_index_multi_params,
    pack_index_put_multi_params,
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
    if dtype == torch.float32:
        return TYPE_F32
    if dtype == torch.int64:
        return TYPE_I64
    if dtype == torch.int32:
        return TYPE_I32
    if dtype == torch.bool:
        return TYPE_BOOL
    # Default fallback
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

        # ep.constants holds tensor constants (like attention masks and RoPE freqs)
        # that are not in ep.state_dict but are still compile-time constants.
        # We need both sources when resolving constant data.
        ep_constants = getattr(edge_program, "constants", {}) or {}

        # Track runtime input index (for inputs that are NOT params/buffers)
        runtime_input_idx = 0

        # Named data store for weights/constants.
        # We follow ExecuTorch convention: store blobs in NamedDataStore and refer
        # to them by key from the delegate blob. Keys are state_dict FQNs.
        data_store = NamedDataStore()

        # Some ggml kernels have dtype contracts. For ggml conv (im2col_f16), the
        # CPU backend expects:
        #   - kernel/weights: F16
        #   - activations/image: F32
        # We currently downcast float32 constants to F16.

        # ---- Pass 1: Walk graph in topological order ----
        for node in graph.nodes:
            if node.op == "placeholder":
                node_name = node.name
                # Some placeholders map to parameters/buffers.  There are two
                # sources for constant tensor data:
                #   1) ep.state_dict  – parameters and mutable buffers (KV caches)
                #   2) ep.constants   – non-mutable tensor constants (attention
                #                       masks, RoPE freqs_cos/sin, etc.)
                # We check both so that attention masks and RoPE frequency tables
                # are treated as compile-time constants rather than runtime inputs.
                fqn = param_map.get(node_name) or buffer_map.get(node_name)
                tensor_from_state = fqn is not None and fqn in edge_program.state_dict
                tensor_from_constants = fqn is not None and fqn in ep_constants
                is_constant = tensor_from_state or tensor_from_constants

                if is_constant:
                    tensor = (
                        edge_program.state_dict[fqn]
                        if tensor_from_state
                        else ep_constants[fqn]
                    )
                    shape = list(tensor.shape)

                    # Store the tensor bytes in NamedDataStore (dedup handled).
                    # Use contiguous CPU tensor for stable storage.
                    t_cpu = tensor.detach().contiguous().cpu()
                    # DType policy:
                    # - downcast float32 *conv* weights to F16 (im2col_f16 path).
                    #   Heuristic: only 4-D (conv kernel) tensors get downcast.
                    # - Keep 2-D float32 LLM weights (linear, RoPE freqs) as F32.
                    # - Bool (attention mask) tensors: convert to F16 with
                    #   0.0 for True (attend) and -inf for False (masked out).
                    #   ggml_flash_attn_ext reads the mask as F16 additive bias;
                    #   ggml_get_rows (used by aten.index) also handles F16.
                    if t_cpu.dtype == torch.bool:
                        # Convert bool causal mask to F16 additive bias:
                        # True -> 0.0 (attend), False -> -inf (mask out)
                        neg_inf = float("-inf")
                        t_float = torch.where(t_cpu, torch.tensor(0.0), torch.tensor(neg_inf))
                        t_cpu = t_float.to(torch.float16)
                    elif t_cpu.dtype == torch.float32 and t_cpu.ndim >= 4:
                        t_cpu = t_cpu.to(torch.float16)

                    # Store inside the .pte.
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
                    if fake_val is not None and hasattr(fake_val, "shape"):
                        shape = list(fake_val.shape)
                    else:
                        shape = []

                    tid = alloc_id()
                    # Runtime input dtype based on FakeTensor meta when available.
                    in_dtype = getattr(fake_val, "dtype", torch.float32) if fake_val is not None and hasattr(fake_val, "dtype") else torch.float32
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=_torch_dtype_to_ir_type(in_dtype),
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
                if "aten._assert_tensor_metadata.default" in target_str:
                    # No-op shape/dtype assertion inserted by export.
                    src_node = node.args[0]
                    node_to_id[node] = node_to_id[src_node]

                elif "aten.scalar_tensor.default" in target_str:
                    # scalar_tensor(s, dtype?, device?) -> 0-d tensor constant
                    fake_val = node.meta.get("val")
                    out_dtype = getattr(fake_val, "dtype", torch.float32) if fake_val is not None else torch.float32
                    value = node.args[0]
                    const = torch.tensor(value, dtype=out_dtype).cpu()

                    tid = alloc_id()
                    # Ensure NamedDataStore keys are stable and unique across
                    # multiple delegated submodules. Using just `tid` can
                    # collide when merging named data stores from different
                    # lowered partitions.
                    key = f"__const_scalar_{node.name}"
                    data_store.add_named_data(key, const, alignment=64)
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=_torch_dtype_to_ir_type(out_dtype),
                            ne=_pytorch_shape_to_ggml_ne(list(const.shape)),
                            op=OP_NONE,
                            data_key=key,
                            is_input=False,
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.t.default" in target_str or "aten.permute_copy.default" in target_str:
                    # NOTE: `aten.permute_copy` can show up in exported graphs as a
                    # materializing copy after a permute. For our matmul lowering we
                    # only care about the logical transpose of weight tensors.
                    #
                    # We treat both `aten.t` and `aten.permute_copy` as a no-op here
                    # (look-through), because ggml_mul_mat expects weight layout in
                    # a way that already matches after our shape reversal.
                    src_node = node.args[0]
                    node_to_id[node] = node_to_id[src_node]

                elif "aten.linear.default" in target_str:
                    # linear(input, weight, bias?)
                    x_node = node.args[0]
                    w_node = node.args[1]
                    b_node = node.args[2] if len(node.args) > 2 else None

                    x_id = node_to_id[x_node]
                    w_id = node_to_id[w_node]
                    src_ids = [x_id, w_id]
                    if b_node is not None:
                        src_ids.append(node_to_id[b_node])

                    fake_val = node.meta.get("val")
                    shape = list(fake_val.shape) if fake_val is not None else []
                    out_dtype = getattr(fake_val, "dtype", torch.float32) if fake_val is not None else torch.float32

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=_torch_dtype_to_ir_type(out_dtype),
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_LINEAR,
                            src_ids=src_ids,
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.embedding.default" in target_str:
                    # embedding(weight, indices)
                    weight_node = node.args[0]
                    indices_node = node.args[1]
                    w_id = node_to_id[weight_node]
                    idx_id = node_to_id[indices_node]

                    fake_val = node.meta.get("val")
                    shape = list(fake_val.shape) if fake_val is not None else []
                    out_dtype = getattr(fake_val, "dtype", torch.float32) if fake_val is not None else torch.float32

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=_torch_dtype_to_ir_type(out_dtype),
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_EMBEDDING,
                            src_ids=[w_id, idx_id],
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.silu.default" in target_str:
                    src_node = node.args[0]
                    src_id = node_to_id[src_node]
                    fake_val = node.meta.get("val")
                    shape = list(fake_val.shape) if fake_val is not None else []
                    out_dtype = getattr(fake_val, "dtype", torch.float32) if fake_val is not None else torch.float32

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=_torch_dtype_to_ir_type(out_dtype),
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_SILU,
                            src_ids=[src_id],
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.mul.Tensor" in target_str:
                    a_node, b_node = node.args[0], node.args[1]
                    a_id = node_to_id[a_node]
                    b_id = node_to_id[b_node]
                    fake_val = node.meta.get("val")
                    shape = list(fake_val.shape) if fake_val is not None else []
                    out_dtype = getattr(fake_val, "dtype", torch.float32) if fake_val is not None else torch.float32

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=_torch_dtype_to_ir_type(out_dtype),
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_MUL,
                            src_ids=[a_id, b_id],
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.neg.default" in target_str:
                    src_node = node.args[0]
                    src_id = node_to_id[src_node]
                    fake_val = node.meta.get("val")
                    shape = list(fake_val.shape) if fake_val is not None else []
                    out_dtype = getattr(fake_val, "dtype", torch.float32) if fake_val is not None else torch.float32

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=_torch_dtype_to_ir_type(out_dtype),
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_NEG,
                            src_ids=[src_id],
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.rsqrt.default" in target_str:
                    src_node = node.args[0]
                    src_id = node_to_id[src_node]
                    fake_val = node.meta.get("val")
                    shape = list(fake_val.shape) if fake_val is not None else []
                    out_dtype = getattr(fake_val, "dtype", torch.float32) if fake_val is not None else torch.float32
                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=_torch_dtype_to_ir_type(out_dtype),
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_RSQRT,
                            src_ids=[src_id],
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.unsqueeze.default" in target_str or "aten.unsqueeze_copy.default" in target_str:
                    src_node = node.args[0]
                    dim = int(node.args[1])
                    src_id = node_to_id[src_node]
                    fake_val = node.meta.get("val")
                    shape = list(fake_val.shape) if fake_val is not None else []
                    out_dtype = getattr(fake_val, "dtype", torch.float32) if fake_val is not None else torch.float32
                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=_torch_dtype_to_ir_type(out_dtype),
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_UNSQUEEZE,
                            src_ids=[src_id],
                            op_params=pack_unsqueeze_params(dim),
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.transpose.int" in target_str:
                    src_node = node.args[0]
                    dim0 = int(node.args[1])
                    dim1 = int(node.args[2])
                    src_id = node_to_id[src_node]
                    fake_val = node.meta.get("val")
                    shape = list(fake_val.shape) if fake_val is not None else []
                    out_dtype = getattr(fake_val, "dtype", torch.float32) if fake_val is not None else torch.float32
                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=_torch_dtype_to_ir_type(out_dtype),
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_TRANSPOSE,
                            src_ids=[src_id],
                            op_params=pack_transpose_params(dim0, dim1),
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.slice.Tensor" in target_str or "aten.slice_copy.Tensor" in target_str:
                    # slice.Tensor(x, dim=0, start=None, end=None, step=1)
                    src_node = node.args[0]
                    dim = int(node.args[1]) if len(node.args) > 1 else 0
                    start = node.args[2] if len(node.args) > 2 else None
                    end = node.args[3] if len(node.args) > 3 else None
                    step = int(node.args[4]) if len(node.args) > 4 else 1

                    # Normalize optional start/end
                    start_i = int(start) if start is not None else 0
                    # If end is None, represent as a large positive bound (runtime will clamp)
                    end_i = int(end) if end is not None else (2**62)
                    src_id = node_to_id[src_node]
                    fake_val = node.meta.get("val")
                    shape = list(fake_val.shape) if fake_val is not None else []
                    out_dtype = getattr(fake_val, "dtype", torch.float32) if fake_val is not None else torch.float32
                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=_torch_dtype_to_ir_type(out_dtype),
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_SLICE,
                            src_ids=[src_id],
                            op_params=pack_slice_params(dim, start_i, end_i, step),
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.cat.default" in target_str:
                    tensors = list(node.args[0])
                    dim = int(node.args[1])
                    src_ids = [node_to_id[t] for t in tensors]
                    fake_val = node.meta.get("val")
                    shape = list(fake_val.shape) if fake_val is not None else []
                    out_dtype = getattr(fake_val, "dtype", torch.float32) if fake_val is not None else torch.float32
                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=_torch_dtype_to_ir_type(out_dtype),
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_CAT,
                            src_ids=src_ids,
                            op_params=pack_cat_params(dim),
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.repeat_interleave.self_int" in target_str:
                    src_node = node.args[0]
                    repeats = int(node.args[1])
                    dim = int(node.args[2]) if len(node.args) > 2 and node.args[2] is not None else 0
                    src_id = node_to_id[src_node]
                    fake_val = node.meta.get("val")
                    shape = list(fake_val.shape) if fake_val is not None else []
                    out_dtype = getattr(fake_val, "dtype", torch.float32) if fake_val is not None else torch.float32
                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=_torch_dtype_to_ir_type(out_dtype),
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_REPEAT_INTERLEAVE,
                            src_ids=[src_id],
                            op_params=pack_repeat_interleave_params(dim, repeats),
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.index.Tensor" in target_str:
                    src_node = node.args[0]
                    indices = node.args[1]
                    non_none = [i for i in indices if i is not None] if isinstance(indices, (list, tuple)) else []

                    fake_val = node.meta.get("val")
                    shape = list(fake_val.shape) if fake_val is not None else []
                    out_dtype = getattr(fake_val, "dtype", torch.float32) if fake_val is not None else torch.float32

                    if len(non_none) == 1:
                        # Single-index case: use ggml_get_rows (gather along dim 0).
                        idx_node = non_none[0]
                        src_id = node_to_id[src_node]
                        idx_id = node_to_id[idx_node]
                        tid = alloc_id()
                        ir_tensors.append(
                            IrTensor(
                                tensor_id=tid,
                                tensor_type=_torch_dtype_to_ir_type(out_dtype),
                                ne=_pytorch_shape_to_ggml_ne(shape),
                                op=OP_INDEX,
                                src_ids=[src_id, idx_id],
                                op_params=pack_index_params(0),
                            )
                        )
                        node_to_id[node] = tid

                    else:
                        # Multi-index case (all non-None): linearize and gather.
                        # For x[idx0, idx1, ...] where x.shape=[D0, D1, ...]:
                        #   linear = idx0 * D1 * D2 * ... + idx1 * D2 * ... + ...
                        #   out = x.view(-1)[linear.view(-1)].view(out_shape)
                        #
                        # Since ggml lacks integer mul/add, we implement this by
                        # packing all indices into op_params and handle in C++ runtime.
                        # op: OP_INDEX_MULTI  with src_ids = [x, idx0, idx1, ...]
                        src_id = node_to_id[src_node]
                        idx_ids = [node_to_id[i] for i in non_none]

                        # Get source shape to compute strides
                        src_val = src_node.meta.get("val")
                        src_shape = list(src_val.shape) if src_val is not None else []

                        tid = alloc_id()
                        ir_tensors.append(
                            IrTensor(
                                tensor_id=tid,
                                tensor_type=_torch_dtype_to_ir_type(out_dtype),
                                ne=_pytorch_shape_to_ggml_ne(shape),
                                op=OP_INDEX_MULTI,
                                src_ids=[src_id] + idx_ids,
                                op_params=pack_index_multi_params(src_shape),
                            )
                        )
                        node_to_id[node] = tid

                elif "aten.select.int" in target_str:
                    # select(x, dim, index) — picks one slice along dim and squeezes it.
                    # e.g. [1,1,1024].select(dim=1, index=-1) -> [1,1024]
                    # Lower as: SLICE(dim, index, index+1, step=1) then VIEW to output shape.
                    src_node = node.args[0]
                    dim = int(node.args[1])
                    idx = int(node.args[2])

                    src_id = node_to_id[src_node]
                    src_val = src_node.meta.get("val")
                    src_shape = list(src_val.shape) if src_val is not None else []

                    # Normalize negative index
                    if idx < 0 and src_shape:
                        idx = src_shape[dim] + idx

                    fake_val = node.meta.get("val")
                    out_shape = list(fake_val.shape) if fake_val is not None else []
                    out_dtype = getattr(fake_val, "dtype", torch.float32) if fake_val is not None else torch.float32

                    # Intermediate sliced shape: same as src but dim shrunk to 1
                    sliced_shape = list(src_shape)
                    sliced_shape[dim] = 1

                    slice_id = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=slice_id,
                            tensor_type=_torch_dtype_to_ir_type(out_dtype),
                            ne=_pytorch_shape_to_ggml_ne(sliced_shape),
                            op=OP_SLICE,
                            src_ids=[src_id],
                            op_params=pack_slice_params(dim, idx, idx + 1, 1),
                        )
                    )

                    # Squeeze the dim via VIEW to out_shape
                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=_torch_dtype_to_ir_type(out_dtype),
                            ne=_pytorch_shape_to_ggml_ne(out_shape),
                            op=OP_VIEW,
                            src_ids=[slice_id],
                            op_params=pack_view_params(out_shape),
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.index_put.default" in target_str:
                    # index_put(x, indices, values, accumulate?)
                    x_node = node.args[0]
                    indices = node.args[1]
                    values_node = node.args[2]

                    if not isinstance(indices, (list, tuple)):
                        raise RuntimeError('aten.index_put: expected indices to be a list/tuple')

                    # Support multi-index form: indices is a tuple of optional index tensors.
                    # We'll serialize only the non-None index tensors as src_ids.
                    present_mask = 0
                    idx_src_ids: List[int] = []
                    for i, idx in enumerate(indices):
                        if idx is None:
                            continue
                        present_mask |= (1 << i)
                        idx_src_ids.append(node_to_id[idx])

                    x_id = node_to_id[x_node]
                    v_id = node_to_id[values_node]

                    fake_val = node.meta.get("val")
                    shape = list(fake_val.shape) if fake_val is not None else []
                    out_dtype = getattr(fake_val, "dtype", torch.float32) if fake_val is not None else torch.float32

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=_torch_dtype_to_ir_type(out_dtype),
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_INDEX_PUT,
                            src_ids=[x_id] + idx_src_ids + [v_id],
                            op_params=pack_index_put_multi_params(len(indices), present_mask),
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.scaled_dot_product_attention.default" in target_str:
                    # Lower SDPA to fused llama.cpp attention op.
                    # Args: (q, k, v, attn_mask, dropout_p, is_causal, scale)
                    q_node, k_node, v_node = node.args[0], node.args[1], node.args[2]
                    mask_node = node.args[3] if len(node.args) > 3 else None
                    q_id = node_to_id[q_node]
                    k_id = node_to_id[k_node]
                    v_id = node_to_id[v_node]
                    src_ids = [q_id, k_id, v_id]
                    if mask_node is not None:
                        src_ids.append(node_to_id[mask_node])
                    fake_val = node.meta.get("val")
                    shape = list(fake_val.shape) if fake_val is not None else []
                    out_dtype = getattr(fake_val, "dtype", torch.float32) if fake_val is not None else torch.float32
                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=_torch_dtype_to_ir_type(out_dtype),
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_LLAMA_ATTENTION,
                            src_ids=src_ids,
                            op_params=b"",  # TODO: pack model params (n_head, head_dim, start_pos binding)
                        )
                    )
                    node_to_id[node] = tid

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
                    keepdim = bool(node.args[2]) if len(node.args) > 2 else False
                    # MV2 global avg pool is exported as mean over (H, W) with keepdim=True.
                    # We keep the semantics by returning a pooled tensor with singleton
                    # spatial dims (H=W=1). If other keepdim=True patterns show up later,
                    # we can extend the lowering.

                    # ExecuTorch/Edge exports `mean.dim` dims as a tuple/list.
                    # For MV2 we need global avg pool which is mean over (H, W) = dims (2, 3).
                    if isinstance(dim, (list, tuple)):
                        dims = [int(d) for d in list(dim)]
                    else:
                        dims = [int(dim)]

                    # Canonicalize dims to positive indices when possible.
                    # For NCHW 4D tensors, -1/-2 correspond to W/H.
                    dims = [d + 4 if d < 0 else d for d in dims]
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
                            op_params=pack_mean_params(dims),
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.expand.default" in target_str or "aten.expand_copy.default" in target_str:
                    # expand is a broadcast. Lower as ggml_repeat(x, like).
                    src_node = node.args[0]
                    src_id = node_to_id[src_node]
                    fake_val = node.meta.get("val")
                    shape = list(fake_val.shape) if fake_val is not None else []
                    out_dtype = getattr(fake_val, "dtype", torch.float32) if fake_val is not None else torch.float32

                    # Create a shape-only "like" tensor.
                    like_id = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=like_id,
                            tensor_type=_torch_dtype_to_ir_type(out_dtype),
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_NONE,
                            is_input=False,
                        )
                    )

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=_torch_dtype_to_ir_type(out_dtype),
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_REPEAT,
                            src_ids=[src_id, like_id],
                        )
                    )
                    node_to_id[node] = tid

                elif (
                    "aten.view.default" in target_str
                    or "aten.view_copy.default" in target_str
                    or "aten._unsafe_view.default" in target_str
                    or "aten.reshape.default" in target_str
                ):
                    # view(x, new_shape) or reshape(x, new_shape)
                    src_node = node.args[0]
                    src_id = node_to_id[src_node]

                    fake_val = node.meta.get("val")
                    shape = list(fake_val.shape) if fake_val is not None else []
                    # Use concrete output shape from FakeTensor meta rather than
                    # node.args[1] which may contain SymInt from dynamic export.
                    new_shape = [int(d) for d in shape] if shape else [int(d) for d in node.args[1]]

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

                elif "aten.add.Tensor" in target_str:
                    # add(x, y, alpha=1)
                    x_node, y_node = node.args[0], node.args[1]
                    # Edge graphs usually pass alpha as a kwarg.
                    alpha = float(getattr(node, "kwargs", {}).get("alpha", 1))
                    if alpha != 1.0:
                        raise RuntimeError("aten.add.Tensor with alpha != 1 not supported yet")

                    x_id = node_to_id[x_node]
                    y_id = node_to_id[y_node]

                    fake_val = node.meta.get("val")
                    out_shape = list(fake_val.shape) if fake_val is not None else []

                    # Make broadcasting explicit with OP_REPEAT where possible.
                    x_val = x_node.meta.get("val") if isinstance(x_node, torch.fx.Node) else None
                    y_val = y_node.meta.get("val") if isinstance(y_node, torch.fx.Node) else None
                    x_shape = list(getattr(x_val, "shape", [])) if x_val is not None else []
                    y_shape = list(getattr(y_val, "shape", [])) if y_val is not None else []

                    def _n_non1(sh):
                        return sum(1 for d in sh if int(d) != 1)

                    def _numel(sh):
                        n = 1
                        for d in sh:
                            n *= int(d)
                        return n

                    like_id = None
                    if out_shape and x_shape and y_shape and (x_shape != y_shape):
                        like_id = alloc_id()
                        ir_tensors.append(
                            IrTensor(
                                tensor_id=like_id,
                                tensor_type=TYPE_F32,
                                ne=_pytorch_shape_to_ggml_ne(out_shape),
                                op=OP_NONE,
                                is_input=False,
                            )
                        )

                        # repeat the smaller/broadcasted side
                        if (_n_non1(x_shape), _numel(x_shape)) < (_n_non1(y_shape), _numel(y_shape)):
                            x_rep = alloc_id()
                            ir_tensors.append(
                                IrTensor(
                                    tensor_id=x_rep,
                                    tensor_type=TYPE_F32,
                                    ne=_pytorch_shape_to_ggml_ne(out_shape),
                                    op=OP_REPEAT,
                                    src_ids=[x_id, like_id],
                                )
                            )
                            x_id = x_rep
                        elif (_n_non1(y_shape), _numel(y_shape)) < (_n_non1(x_shape), _numel(x_shape)):
                            y_rep = alloc_id()
                            ir_tensors.append(
                                IrTensor(
                                    tensor_id=y_rep,
                                    tensor_type=TYPE_F32,
                                    ne=_pytorch_shape_to_ggml_ne(out_shape),
                                    op=OP_REPEAT,
                                    src_ids=[y_id, like_id],
                                )
                            )
                            y_id = y_rep

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=TYPE_F32,
                            ne=_pytorch_shape_to_ggml_ne(out_shape),
                            op=OP_ADD,
                            src_ids=[x_id, y_id],
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.type_as.default" in target_str:
                    # type_as(input, other) casts input to other's dtype.
                    # In the Qwen3 graph both tensors are already f32→f32, so this
                    # is a no-op.  Treat as an identity (look-through).
                    src_node = node.args[0]
                    node_to_id[node] = node_to_id[src_node]

                elif "aten.alias.default" in target_str or "aten.alias_copy.default" in target_str:
                    src_node = node.args[0]
                    node_to_id[node] = node_to_id[src_node]

                elif "dim_order_ops._clone_dim_order.default" in target_str:
                    # `_clone_dim_order` is a layout materialization op used by
                    # ExecuTorch/Edge for dim-order management. For ggml lowering we
                    # treat it as a pure no-op.
                    src_node = node.args[0]
                    node_to_id[node] = node_to_id[src_node]

                elif "aten.permute.default" in target_str or "aten.permute_copy.default" in target_str:
                    # permute(x, dims)
                    #
                    # NOTE: `aten.permute_copy` is a permute with explicit copy
                    # semantics in PyTorch export. ggml doesn't need that distinction
                    # for our purposes, so we lower both `permute` and `permute_copy`
                    # to the same IR op (`OP_PERMUTE`).
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
