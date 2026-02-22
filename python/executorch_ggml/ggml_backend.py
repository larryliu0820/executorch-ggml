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
    OP_SUB,
    OP_MUL_MAT,
    OP_MUL,
    OP_MUL_SCALAR,
    OP_NEG,
    OP_POW,
    OP_COS,
    OP_SIN,
    OP_BMM,
    OP_SIGMOID,
    OP_SOFTMAX,
    OP_WHERE,
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
    OP_CAST,
    OP_LLAMA_ATTENTION,
    # Mask computation ops
    OP_ARANGE,
    OP_FULL,
    OP_CUMSUM,
    OP_EQ,
    OP_NE,
    OP_LE,
    OP_LT,
    OP_GT,
    OP_GE,
    OP_BITWISE_AND,
    OP_BITWISE_OR,
    OP_LOGICAL_NOT,
    OP_ANY,
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
    pack_cast_params,
    pack_softmax_params,
    pack_pow_params,
    pack_arange_params,
    pack_full_params,
    pack_cumsum_params,
    pack_comparison_params,
    pack_any_params,
    serialize_graph,
)

from executorch.exir._serialize._named_data_store import NamedDataStore


def _pytorch_shape_to_ggml_ne(shape: List[int]) -> List[int]:
    """PyTorch [d0, d1, ..., dn] → ggml ne [dn, ..., d1, d0], padded/collapsed to 4D.

    For >4D tensors, collapse leading dimensions: [a,b,c,d,e] → [a*b, c, d, e] (reversed).
    """
    if len(shape) <= 4:
        ne = list(reversed(shape))
        while len(ne) < 4:
            ne.append(1)
        return ne[:4]
    else:
        # Collapse leading dimensions for >4D tensors
        # e.g., [1, 8, 2, 512, 128] → [1*8, 2, 512, 128] → reversed: [128, 512, 2, 8]
        leading_prod = 1
        for d in shape[:-4]:
            leading_prod *= d
        ne = [leading_prod] + list(shape[-4:])
        return list(reversed(ne))


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

        def _look_through_transpose(n: torch.fx.Node) -> torch.fx.Node:
            """Look through aten.t, aten.permute_copy, aten.permute that
            transposes a 2D weight tensor. ggml_mul_mat already does an
            implicit transpose, so mm/addmm should pass the original weight."""
            if n.op != "call_function":
                return n
            t = str(n.target)
            if "aten.t" in t or "aten.permute_copy" in t or "aten.permute" in t:
                return n.args[0]
            return n

        # Build mapping from param/buffer placeholder names → FQN → tensor data
        sig = edge_program.graph_signature
        param_map = dict(sig.inputs_to_parameters)  # node_name → param FQN
        buffer_map = dict(sig.inputs_to_buffers)  # node_name → buffer FQN

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
                        t_float = torch.where(
                            t_cpu, torch.tensor(0.0), torch.tensor(neg_inf)
                        )
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
                    in_dtype = (
                        getattr(fake_val, "dtype", torch.float32)
                        if fake_val is not None and hasattr(fake_val, "dtype")
                        else torch.float32
                    )
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
                    out_dtype = (
                        getattr(fake_val, "dtype", torch.float32)
                        if fake_val is not None
                        else torch.float32
                    )
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

                elif "aten.t.default" in target_str:
                    # aten.t is a 2D transpose. For matmul lowering, ggml_mul_mat
                    # expects weight layout that already matches after shape reversal,
                    # so we treat it as a no-op (look-through).
                    # NOTE: aten.permute_copy is handled later via the full permute path.
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
                    out_dtype = (
                        getattr(fake_val, "dtype", torch.float32)
                        if fake_val is not None
                        else torch.float32
                    )

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

                    # Cast I64 indices to I32 (ggml_get_rows requires I32).
                    idx_fv = indices_node.meta.get("val")
                    idx_dtype = (
                        getattr(idx_fv, "dtype", torch.int64)
                        if idx_fv is not None
                        else torch.int64
                    )
                    if idx_dtype == torch.int64 or idx_dtype == torch.long:
                        idx_shape = list(idx_fv.shape) if idx_fv is not None else []
                        cast_tid = alloc_id()
                        ir_tensors.append(
                            IrTensor(
                                tensor_id=cast_tid,
                                tensor_type=TYPE_I32,
                                ne=_pytorch_shape_to_ggml_ne(idx_shape),
                                op=OP_CAST,
                                src_ids=[idx_id],
                                op_params=pack_cast_params(TYPE_I32),
                            )
                        )
                        idx_id = cast_tid

                    fake_val = node.meta.get("val")
                    shape = list(fake_val.shape) if fake_val is not None else []
                    out_dtype = (
                        getattr(fake_val, "dtype", torch.float32)
                        if fake_val is not None
                        else torch.float32
                    )

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
                    out_dtype = (
                        getattr(fake_val, "dtype", torch.float32)
                        if fake_val is not None
                        else torch.float32
                    )

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
                    out_shape = list(fake_val.shape) if fake_val is not None else []
                    out_dtype = (
                        getattr(fake_val, "dtype", torch.float32)
                        if fake_val is not None
                        else torch.float32
                    )

                    # Make broadcasting explicit with OP_REPEAT where needed.
                    # ggml_mul requires ggml_can_repeat(b, a) which fails for
                    # cases like [1,3,1] * [1,1,64] where neither can repeat into other.
                    a_val = (
                        a_node.meta.get("val")
                        if isinstance(a_node, torch.fx.Node)
                        else None
                    )
                    b_val = (
                        b_node.meta.get("val")
                        if isinstance(b_node, torch.fx.Node)
                        else None
                    )
                    a_shape = (
                        list(getattr(a_val, "shape", [])) if a_val is not None else []
                    )
                    b_shape = (
                        list(getattr(b_val, "shape", [])) if b_val is not None else []
                    )

                    like_id = None
                    if (
                        out_shape
                        and a_shape
                        and b_shape
                        and (a_shape != out_shape or b_shape != out_shape)
                    ):
                        # Check if ggml_repeat can handle the broadcast
                        # Convert to ggml shapes for compatibility check
                        a_ne = _pytorch_shape_to_ggml_ne(a_shape)
                        b_ne = _pytorch_shape_to_ggml_ne(b_shape)
                        out_ne = _pytorch_shape_to_ggml_ne(out_shape)

                        # Check if a can repeat to output shape
                        a_can_repeat = all(
                            a_ne[i] == 1 or a_ne[i] == out_ne[i] for i in range(4)
                        )
                        # Check if b can repeat to output shape
                        b_can_repeat = all(
                            b_ne[i] == 1 or b_ne[i] == out_ne[i] for i in range(4)
                        )

                        if not (a_can_repeat and b_can_repeat):
                            # Can't handle this broadcast in ggml, fall back to host
                            raise RuntimeError(
                                f"MUL broadcast not supported: a={a_shape} -> {a_ne}, "
                                f"b={b_shape} -> {b_ne}, out={out_shape} -> {out_ne}"
                            )

                        # Create a "like" tensor for REPEAT target shape
                        like_id = alloc_id()
                        ir_tensors.append(
                            IrTensor(
                                tensor_id=like_id,
                                tensor_type=_torch_dtype_to_ir_type(out_dtype),
                                ne=out_ne,
                                op=OP_NONE,
                                is_input=False,
                            )
                        )

                        # REPEAT a if it doesn't match output shape
                        if a_shape != out_shape:
                            a_rep = alloc_id()
                            ir_tensors.append(
                                IrTensor(
                                    tensor_id=a_rep,
                                    tensor_type=_torch_dtype_to_ir_type(out_dtype),
                                    ne=out_ne,
                                    op=OP_REPEAT,
                                    src_ids=[a_id, like_id],
                                )
                            )
                            a_id = a_rep

                        # REPEAT b if it doesn't match output shape
                        if b_shape != out_shape:
                            b_rep = alloc_id()
                            ir_tensors.append(
                                IrTensor(
                                    tensor_id=b_rep,
                                    tensor_type=_torch_dtype_to_ir_type(out_dtype),
                                    ne=out_ne,
                                    op=OP_REPEAT,
                                    src_ids=[b_id, like_id],
                                )
                            )
                            b_id = b_rep

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=_torch_dtype_to_ir_type(out_dtype),
                            ne=_pytorch_shape_to_ggml_ne(out_shape),
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
                    out_dtype = (
                        getattr(fake_val, "dtype", torch.float32)
                        if fake_val is not None
                        else torch.float32
                    )

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
                    out_dtype = (
                        getattr(fake_val, "dtype", torch.float32)
                        if fake_val is not None
                        else torch.float32
                    )
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

                elif (
                    "aten.unsqueeze.default" in target_str
                    or "aten.unsqueeze_copy.default" in target_str
                ):
                    src_node = node.args[0]
                    dim = int(node.args[1])
                    src_id = node_to_id[src_node]
                    fake_val = node.meta.get("val")
                    shape = list(fake_val.shape) if fake_val is not None else []
                    out_dtype = (
                        getattr(fake_val, "dtype", torch.float32)
                        if fake_val is not None
                        else torch.float32
                    )
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
                    out_dtype = (
                        getattr(fake_val, "dtype", torch.float32)
                        if fake_val is not None
                        else torch.float32
                    )
                    ndim = len(shape)
                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=_torch_dtype_to_ir_type(out_dtype),
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_TRANSPOSE,
                            src_ids=[src_id],
                            op_params=pack_transpose_params(dim0, dim1, ndim),
                        )
                    )
                    node_to_id[node] = tid

                elif (
                    "aten.slice.Tensor" in target_str
                    or "aten.slice_copy.Tensor" in target_str
                ):
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
                    out_dtype = (
                        getattr(fake_val, "dtype", torch.float32)
                        if fake_val is not None
                        else torch.float32
                    )
                    # Get the source tensor's PyTorch rank for correct axis computation.
                    src_val = (
                        src_node.meta.get("val")
                        if isinstance(src_node, torch.fx.Node)
                        else None
                    )
                    src_shape = (
                        list(getattr(src_val, "shape", []))
                        if src_val is not None
                        else []
                    )
                    ndim = len(src_shape) if src_shape else len(shape)
                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=_torch_dtype_to_ir_type(out_dtype),
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_SLICE,
                            src_ids=[src_id],
                            op_params=pack_slice_params(
                                dim, start_i, end_i, step, ndim
                            ),
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.cat.default" in target_str:
                    tensors = list(node.args[0])
                    dim = int(node.args[1])
                    src_ids = [node_to_id[t] for t in tensors]
                    fake_val = node.meta.get("val")
                    shape = list(fake_val.shape) if fake_val is not None else []
                    out_dtype = (
                        getattr(fake_val, "dtype", torch.float32)
                        if fake_val is not None
                        else torch.float32
                    )
                    # Normalize negative dim and compute ggml axis directly.
                    # ggml axis = (rank - 1) - dim, where rank is the full PyTorch rank.
                    ndim = len(shape)
                    if dim < 0:
                        dim = ndim + dim
                    ggml_axis = (ndim - 1) - dim
                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=_torch_dtype_to_ir_type(out_dtype),
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_CAT,
                            src_ids=src_ids,
                            op_params=pack_cat_params(ggml_axis),
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.repeat_interleave.self_int" in target_str:
                    src_node = node.args[0]
                    repeats = int(node.args[1])
                    dim = (
                        int(node.args[2])
                        if len(node.args) > 2 and node.args[2] is not None
                        else 0
                    )
                    src_id = node_to_id[src_node]
                    fake_val = node.meta.get("val")
                    shape = list(fake_val.shape) if fake_val is not None else []
                    out_dtype = (
                        getattr(fake_val, "dtype", torch.float32)
                        if fake_val is not None
                        else torch.float32
                    )
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
                    if not isinstance(indices, (list, tuple)):
                        raise RuntimeError(
                            "aten.index.Tensor: expected indices to be list/tuple"
                        )

                    non_none_pos = [
                        i for i, idx in enumerate(indices) if idx is not None
                    ]
                    non_none = [indices[i] for i in non_none_pos]

                    fake_val = node.meta.get("val")
                    shape = list(fake_val.shape) if fake_val is not None else []
                    out_dtype = (
                        getattr(fake_val, "dtype", torch.float32)
                        if fake_val is not None
                        else torch.float32
                    )

                    if (
                        len(non_none) == 1
                        and len(non_none_pos) == 1
                        and non_none_pos[0] == 0
                    ):
                        # Single-index case: use ggml_get_rows (gather along dim 0).
                        idx_node = non_none[0]
                        src_id = node_to_id[src_node]
                        idx_id = node_to_id[idx_node]

                        # Cast I64 indices to I32 (ggml_get_rows requires I32).
                        idx_fv = idx_node.meta.get("val")
                        idx_dtype = (
                            getattr(idx_fv, "dtype", torch.int64)
                            if idx_fv is not None
                            else torch.int64
                        )
                        if idx_dtype == torch.int64 or idx_dtype == torch.long:
                            idx_shape = list(idx_fv.shape) if idx_fv is not None else []
                            cast_tid = alloc_id()
                            ir_tensors.append(
                                IrTensor(
                                    tensor_id=cast_tid,
                                    tensor_type=TYPE_I32,
                                    ne=_pytorch_shape_to_ggml_ne(idx_shape),
                                    op=OP_CAST,
                                    src_ids=[idx_id],
                                    op_params=pack_cast_params(TYPE_I32),
                                )
                            )
                            idx_id = cast_tid

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

                    elif len(non_none) > 1 and len(non_none) == len(indices):
                        # Multi-index case (all indices present):
                        # lower to runtime custom gather op.
                        src_val = src_node.meta.get("val")
                        src_shape = list(src_val.shape) if src_val is not None else []
                        if len(src_shape) == 0 or len(src_shape) > 4:
                            raise RuntimeError(
                                "aten.index.Tensor multi-index: unsupported source rank "
                                f"{len(src_shape)}"
                            )
                        if len(indices) != len(src_shape):
                            raise RuntimeError(
                                "aten.index.Tensor multi-index: indices rank "
                                f"{len(indices)} does not match source rank {len(src_shape)}"
                            )

                        src_id = node_to_id[src_node]
                        idx_ids = [node_to_id[i] for i in indices]
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

                    else:
                        raise RuntimeError(
                            "aten.index.Tensor: unsupported indexing pattern for ggml lowering"
                        )

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
                    out_dtype = (
                        getattr(fake_val, "dtype", torch.float32)
                        if fake_val is not None
                        else torch.float32
                    )

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
                    ggml_ne = _pytorch_shape_to_ggml_ne(out_shape)
                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=_torch_dtype_to_ir_type(out_dtype),
                            ne=ggml_ne,
                            op=OP_VIEW,
                            src_ids=[slice_id],
                            op_params=pack_view_params(ggml_ne),
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.index_put.default" in target_str:
                    # index_put(x, indices, values, accumulate?)
                    x_node = node.args[0]
                    indices = node.args[1]
                    values_node = node.args[2]

                    if not isinstance(indices, (list, tuple)):
                        raise RuntimeError(
                            "aten.index_put: expected indices to be a list/tuple"
                        )

                    # Support multi-index form: indices is a tuple of optional index tensors.
                    # We'll serialize only the non-None index tensors as src_ids.
                    present_mask = 0
                    idx_src_ids: List[int] = []
                    for i, idx in enumerate(indices):
                        if idx is None:
                            continue
                        present_mask |= 1 << i
                        idx_src_ids.append(node_to_id[idx])

                    x_id = node_to_id[x_node]
                    v_id = node_to_id[values_node]

                    fake_val = node.meta.get("val")
                    shape = list(fake_val.shape) if fake_val is not None else []
                    out_dtype = (
                        getattr(fake_val, "dtype", torch.float32)
                        if fake_val is not None
                        else torch.float32
                    )

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=_torch_dtype_to_ir_type(out_dtype),
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_INDEX_PUT,
                            src_ids=[x_id] + idx_src_ids + [v_id],
                            op_params=pack_index_put_multi_params(
                                len(indices), present_mask
                            ),
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
                    out_dtype = (
                        getattr(fake_val, "dtype", torch.float32)
                        if fake_val is not None
                        else torch.float32
                    )
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
                    # ggml_mul_mat(a, b) computes b @ a^T, so pass the
                    # original (un-transposed) weight.
                    input_node, weight_t_node = node.args
                    orig_w_node = _look_through_transpose(weight_t_node)
                    weight_id = node_to_id[orig_w_node]
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
                    # ggml_mul_mat(a, b) computes b @ a^T, so pass the
                    # original (un-transposed) weight.
                    bias_node, input_node, weight_t_node = node.args

                    orig_w_node = _look_through_transpose(weight_t_node)
                    weight_id = node_to_id[orig_w_node]
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

                elif (
                    "aten.convolution.default" in target_str
                    or "aten.conv2d.default" in target_str
                ):
                    # aten.convolution.default or aten.conv2d.default
                    # Args: (input, weight, bias?, stride, padding, dilation, transposed, output_padding, groups)
                    # For conv2d: (input, weight, bias, stride, padding, dilation, groups)

                    if "aten.convolution.default" in target_str:
                        if len(node.args) < 9:
                            raise RuntimeError(
                                f"Expected 9 args for convolution, got {len(node.args)}"
                            )
                        input_node = node.args[0]
                        weight_node = node.args[1]
                        bias_node = (
                            node.args[2]
                            if len(node.args) > 2 and node.args[2] is not None
                            else None
                        )
                        stride = list(node.args[3]) if len(node.args) > 3 else [1, 1]
                        padding = list(node.args[4]) if len(node.args) > 4 else [0, 0]
                        dilation = list(node.args[5]) if len(node.args) > 5 else [1, 1]
                        transposed = node.args[6] if len(node.args) > 6 else False
                        output_padding = (
                            list(node.args[7]) if len(node.args) > 7 else [0, 0]
                        )
                        groups = node.args[8] if len(node.args) > 8 else 1
                    else:  # conv2d
                        if len(node.args) < 7:
                            raise RuntimeError(
                                f"Expected 7 args for conv2d, got {len(node.args)}"
                            )
                        input_node = node.args[0]
                        weight_node = node.args[1]
                        bias_node = (
                            node.args[2]
                            if len(node.args) > 2 and node.args[2] is not None
                            else None
                        )
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

                elif (
                    "aten.hardtanh.default" in target_str
                    or "aten.clamp.default" in target_str
                ):
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
                            op_params=pack_hardtanh_params(
                                float(min_val), float(max_val)
                            ),
                        )
                    )
                    node_to_id[node] = tid

                elif (
                    "aten.mean.dim" in target_str
                    or "aten._mean_dim.default" in target_str
                ):
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

                elif (
                    "aten.expand.default" in target_str
                    or "aten.expand_copy.default" in target_str
                ):
                    # expand is a broadcast. Lower as ggml_repeat(x, like) if compatible,
                    # otherwise just use the source (let consumer handle broadcast).
                    src_node = node.args[0]
                    src_id = node_to_id[src_node]
                    fake_val = node.meta.get("val")
                    shape = list(fake_val.shape) if fake_val is not None else []
                    out_dtype = (
                        getattr(fake_val, "dtype", torch.float32)
                        if fake_val is not None
                        else torch.float32
                    )

                    src_fake = (
                        src_node.meta.get("val") if hasattr(src_node, "meta") else None
                    )
                    src_shape = list(src_fake.shape) if src_fake is not None else []

                    # Check if ggml_repeat can handle this broadcast
                    # ggml_repeat requires each dim of src to be 1 or equal to target dim
                    src_ne = _pytorch_shape_to_ggml_ne(src_shape)
                    dst_ne = _pytorch_shape_to_ggml_ne(shape)
                    can_repeat = all(
                        src_ne[i] == 1 or src_ne[i] == dst_ne[i] for i in range(4)
                    )

                    if can_repeat:
                        # Create a shape-only "like" tensor.
                        like_id = alloc_id()
                        ir_tensors.append(
                            IrTensor(
                                tensor_id=like_id,
                                tensor_type=_torch_dtype_to_ir_type(out_dtype),
                                ne=dst_ne,
                                op=OP_NONE,
                                is_input=False,
                            )
                        )

                        tid = alloc_id()
                        ir_tensors.append(
                            IrTensor(
                                tensor_id=tid,
                                tensor_type=_torch_dtype_to_ir_type(out_dtype),
                                ne=dst_ne,
                                op=OP_REPEAT,
                                src_ids=[src_id, like_id],
                            )
                        )
                        node_to_id[node] = tid
                    else:
                        # Just use source tensor - consumer ops have broadcasting
                        node_to_id[node] = src_id

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
                    new_shape = (
                        [int(d) for d in shape]
                        if shape
                        else [int(d) for d in node.args[1]]
                    )

                    # Pack the shape in ggml ne order (reversed from PyTorch)
                    # since the C++ runtime passes these directly to ggml_reshape_4d.
                    ggml_ne = _pytorch_shape_to_ggml_ne(new_shape)

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=TYPE_F32,
                            ne=ggml_ne,
                            op=OP_VIEW,
                            src_ids=[src_id],
                            op_params=pack_view_params(ggml_ne),
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.add.Tensor" in target_str:
                    # add(x, y, alpha=1)
                    x_node, y_node = node.args[0], node.args[1]
                    # Edge graphs usually pass alpha as a kwarg.
                    alpha = float(getattr(node, "kwargs", {}).get("alpha", 1))
                    if alpha != 1.0:
                        raise RuntimeError(
                            "aten.add.Tensor with alpha != 1 not supported yet"
                        )

                    x_id = node_to_id[x_node]
                    y_id = node_to_id[y_node]

                    fake_val = node.meta.get("val")
                    out_shape = list(fake_val.shape) if fake_val is not None else []

                    # Make broadcasting explicit with OP_REPEAT where possible.
                    x_val = (
                        x_node.meta.get("val")
                        if isinstance(x_node, torch.fx.Node)
                        else None
                    )
                    y_val = (
                        y_node.meta.get("val")
                        if isinstance(y_node, torch.fx.Node)
                        else None
                    )
                    x_shape = (
                        list(getattr(x_val, "shape", [])) if x_val is not None else []
                    )
                    y_shape = (
                        list(getattr(y_val, "shape", [])) if y_val is not None else []
                    )

                    def _n_non1(sh):
                        return sum(1 for d in sh if int(d) != 1)

                    def _numel(sh):
                        n = 1
                        for d in sh:
                            n *= int(d)
                        return n

                    like_id = None
                    if out_shape and x_shape and y_shape and (x_shape != y_shape):
                        # Check if ggml_repeat can handle the broadcast
                        x_ne = _pytorch_shape_to_ggml_ne(x_shape)
                        y_ne = _pytorch_shape_to_ggml_ne(y_shape)
                        out_ne = _pytorch_shape_to_ggml_ne(out_shape)

                        x_can_repeat = all(
                            x_ne[i] == 1 or x_ne[i] == out_ne[i] for i in range(4)
                        )
                        y_can_repeat = all(
                            y_ne[i] == 1 or y_ne[i] == out_ne[i] for i in range(4)
                        )

                        if not (x_can_repeat and y_can_repeat):
                            # Can't handle this broadcast in ggml, fall back to host
                            raise RuntimeError(
                                f"ADD broadcast not supported: x={x_shape} -> {x_ne}, "
                                f"y={y_shape} -> {y_ne}, out={out_shape} -> {out_ne}"
                            )

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
                        if (_n_non1(x_shape), _numel(x_shape)) < (
                            _n_non1(y_shape),
                            _numel(y_shape),
                        ):
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
                        elif (_n_non1(y_shape), _numel(y_shape)) < (
                            _n_non1(x_shape),
                            _numel(x_shape),
                        ):
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
                        else:
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

                elif (
                    "aten.alias.default" in target_str
                    or "aten.alias_copy.default" in target_str
                ):
                    src_node = node.args[0]
                    node_to_id[node] = node_to_id[src_node]

                elif "dim_order_ops._clone_dim_order.default" in target_str:
                    # `_clone_dim_order` is a layout materialization op used by
                    # ExecuTorch/Edge for dim-order management. For ggml lowering we
                    # treat it as a pure no-op.
                    src_node = node.args[0]
                    node_to_id[node] = node_to_id[src_node]

                elif (
                    "aten.permute.default" in target_str
                    or "aten.permute_copy.default" in target_str
                ):
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

                    # Convert PyTorch permutation to ggml permutation.
                    # PyTorch axes are reversed relative to ggml: pytorch_axis = (ndim-1) - ggml_axis
                    # For each ggml axis j, the source axis is:
                    #   ggml_perm[j] = (ndim - 1) - perm[(ndim - 1) - j]
                    ndim = len(perm)
                    ggml_perm = [0, 1, 2, 3]
                    for j in range(4):
                        if j < ndim:
                            pt_result_axis = (ndim - 1) - j
                            pt_source_axis = perm[pt_result_axis]
                            ggml_perm[j] = (ndim - 1) - pt_source_axis
                        else:
                            ggml_perm[j] = j

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=TYPE_F32,
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_PERMUTE,
                            src_ids=[src_id],
                            op_params=pack_permute_params(ggml_perm),
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.sub.Tensor" in target_str:
                    # sub(x, y, alpha=1)
                    x_node, y_node = node.args[0], node.args[1]
                    x_id = node_to_id[x_node]
                    y_id = node_to_id[y_node]

                    fake_val = node.meta.get("val")
                    out_shape = list(fake_val.shape) if fake_val is not None else []
                    out_dtype = (
                        getattr(fake_val, "dtype", torch.float32)
                        if fake_val is not None
                        else torch.float32
                    )

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=_torch_dtype_to_ir_type(out_dtype),
                            ne=_pytorch_shape_to_ggml_ne(out_shape),
                            op=OP_SUB,
                            src_ids=[x_id, y_id],
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.mul.Scalar" in target_str:
                    # mul(x, scalar)
                    src_node = node.args[0]
                    scalar = float(node.args[1])
                    src_id = node_to_id[src_node]

                    fake_val = node.meta.get("val")
                    shape = list(fake_val.shape) if fake_val is not None else []
                    out_dtype = (
                        getattr(fake_val, "dtype", torch.float32)
                        if fake_val is not None
                        else torch.float32
                    )

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=_torch_dtype_to_ir_type(out_dtype),
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_MUL_SCALAR,
                            src_ids=[src_id],
                            op_params=pack_float(scalar),
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.pow.Tensor_Scalar" in target_str:
                    # pow(x, exponent)
                    src_node = node.args[0]
                    exponent = float(node.args[1])
                    src_id = node_to_id[src_node]

                    fake_val = node.meta.get("val")
                    shape = list(fake_val.shape) if fake_val is not None else []
                    out_dtype = (
                        getattr(fake_val, "dtype", torch.float32)
                        if fake_val is not None
                        else torch.float32
                    )

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=_torch_dtype_to_ir_type(out_dtype),
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_POW,
                            src_ids=[src_id],
                            op_params=pack_pow_params(exponent),
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.cos.default" in target_str:
                    src_node = node.args[0]
                    src_id = node_to_id[src_node]

                    fake_val = node.meta.get("val")
                    shape = list(fake_val.shape) if fake_val is not None else []
                    out_dtype = (
                        getattr(fake_val, "dtype", torch.float32)
                        if fake_val is not None
                        else torch.float32
                    )

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=_torch_dtype_to_ir_type(out_dtype),
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_COS,
                            src_ids=[src_id],
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.sin.default" in target_str:
                    src_node = node.args[0]
                    src_id = node_to_id[src_node]

                    fake_val = node.meta.get("val")
                    shape = list(fake_val.shape) if fake_val is not None else []
                    out_dtype = (
                        getattr(fake_val, "dtype", torch.float32)
                        if fake_val is not None
                        else torch.float32
                    )

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=_torch_dtype_to_ir_type(out_dtype),
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_SIN,
                            src_ids=[src_id],
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.bmm.default" in target_str:
                    # bmm(input, mat2) -> batch matrix multiply
                    a_node, b_node = node.args[0], node.args[1]
                    a_id = node_to_id[a_node]
                    b_id = node_to_id[b_node]

                    fake_val = node.meta.get("val")
                    shape = list(fake_val.shape) if fake_val is not None else []
                    out_dtype = (
                        getattr(fake_val, "dtype", torch.float32)
                        if fake_val is not None
                        else torch.float32
                    )

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=_torch_dtype_to_ir_type(out_dtype),
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_BMM,
                            src_ids=[a_id, b_id],
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.sigmoid.default" in target_str:
                    src_node = node.args[0]
                    src_id = node_to_id[src_node]

                    fake_val = node.meta.get("val")
                    shape = list(fake_val.shape) if fake_val is not None else []
                    out_dtype = (
                        getattr(fake_val, "dtype", torch.float32)
                        if fake_val is not None
                        else torch.float32
                    )

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=_torch_dtype_to_ir_type(out_dtype),
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_SIGMOID,
                            src_ids=[src_id],
                        )
                    )
                    node_to_id[node] = tid

                elif "aten._softmax.default" in target_str:
                    # _softmax(x, dim, half_to_float)
                    src_node = node.args[0]
                    dim = int(node.args[1])
                    src_id = node_to_id[src_node]

                    fake_val = node.meta.get("val")
                    shape = list(fake_val.shape) if fake_val is not None else []
                    out_dtype = (
                        getattr(fake_val, "dtype", torch.float32)
                        if fake_val is not None
                        else torch.float32
                    )
                    ndim = len(shape)

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=_torch_dtype_to_ir_type(out_dtype),
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_SOFTMAX,
                            src_ids=[src_id],
                            op_params=pack_softmax_params(dim, ndim),
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.where.self" in target_str:
                    # where(condition, x, y)
                    cond_node, x_node, y_node = node.args[0], node.args[1], node.args[2]
                    cond_id = node_to_id[cond_node]
                    x_id = node_to_id[x_node]
                    y_id = node_to_id[y_node]

                    fake_val = node.meta.get("val")
                    shape = list(fake_val.shape) if fake_val is not None else []
                    out_dtype = (
                        getattr(fake_val, "dtype", torch.float32)
                        if fake_val is not None
                        else torch.float32
                    )

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=_torch_dtype_to_ir_type(out_dtype),
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_WHERE,
                            src_ids=[cond_id, x_id, y_id],
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.full_like.default" in target_str:
                    # full_like(input, fill_value, ...) - create tensor like input filled with fill_value
                    # This is typically used to create constant tensors (e.g., -inf for masking).
                    fill_value = float(node.args[1])

                    fake_val = node.meta.get("val")
                    shape = list(fake_val.shape) if fake_val is not None else []
                    out_dtype = (
                        getattr(fake_val, "dtype", torch.float32)
                        if fake_val is not None
                        else torch.float32
                    )

                    # Create a scalar constant and use MUL_SCALAR with a tensor of ones
                    # Actually, simpler: create a VIEW of a scalar tensor.
                    # For now, store the fill value as a named constant.
                    import hashlib
                    import numpy as np

                    # Create numpy array with fill value
                    numel = 1
                    for d in shape:
                        numel *= d
                    const_data = np.full(numel, fill_value, dtype=np.float32)
                    const_key = f"_full_like_{hashlib.sha256(const_data.tobytes()).hexdigest()[:16]}"

                    data_store.add_named_data(
                        const_key, const_data.tobytes(), alignment=64
                    )

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=_torch_dtype_to_ir_type(out_dtype),
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_NONE,  # Constant
                            data_key=const_key,
                        )
                    )
                    node_to_id[node] = tid

                elif "dim_order_ops._to_dim_order_copy.default" in target_str:
                    # Treat as no-op / identity for ggml purposes
                    src_node = node.args[0]
                    node_to_id[node] = node_to_id[src_node]

                elif "aten.arange.start_step" in target_str:
                    # arange(start, end, step, ...) - generates [start, start+step, ...]
                    start = float(node.args[0]) if len(node.args) > 0 else 0.0
                    # end = node.args[1]  # We use output shape instead
                    step = float(node.args[2]) if len(node.args) > 2 else 1.0

                    fake_val = node.meta.get("val")
                    shape = list(fake_val.shape) if fake_val is not None else []
                    out_dtype = (
                        getattr(fake_val, "dtype", torch.int64)
                        if fake_val is not None
                        else torch.int64
                    )

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=_torch_dtype_to_ir_type(out_dtype),
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_ARANGE,
                            src_ids=[],
                            op_params=pack_arange_params(start, step),
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.full.default" in target_str:
                    # full(size, fill_value, ...) - creates tensor filled with fill_value
                    fill_value = float(node.args[1]) if len(node.args) > 1 else 0.0

                    fake_val = node.meta.get("val")
                    shape = list(fake_val.shape) if fake_val is not None else []
                    out_dtype = (
                        getattr(fake_val, "dtype", torch.float32)
                        if fake_val is not None
                        else torch.float32
                    )

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=_torch_dtype_to_ir_type(out_dtype),
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_FULL,
                            src_ids=[],
                            op_params=pack_full_params(fill_value),
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.cumsum.default" in target_str:
                    # cumsum(x, dim) - cumulative sum along dimension
                    src_node = node.args[0]
                    dim = int(node.args[1]) if len(node.args) > 1 else 0
                    src_id = node_to_id[src_node]

                    fake_val = node.meta.get("val")
                    shape = list(fake_val.shape) if fake_val is not None else []
                    out_dtype = (
                        getattr(fake_val, "dtype", torch.int64)
                        if fake_val is not None
                        else torch.int64
                    )
                    ndim = len(shape)

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=_torch_dtype_to_ir_type(out_dtype),
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_CUMSUM,
                            src_ids=[src_id],
                            op_params=pack_cumsum_params(dim, ndim),
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.eq.Scalar" in target_str:
                    # eq(x, scalar) - element-wise equality with scalar
                    src_node = node.args[0]
                    scalar = float(node.args[1])
                    src_id = node_to_id[src_node]

                    fake_val = node.meta.get("val")
                    shape = list(fake_val.shape) if fake_val is not None else []

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=TYPE_BOOL,
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_EQ,
                            src_ids=[src_id],
                            op_params=pack_comparison_params(scalar, is_scalar=True),
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.eq.Tensor" in target_str:
                    # eq(x, y) - element-wise equality
                    a_node, b_node = node.args[0], node.args[1]
                    a_id = node_to_id[a_node]
                    b_id = node_to_id[b_node]

                    fake_val = node.meta.get("val")
                    shape = list(fake_val.shape) if fake_val is not None else []

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=TYPE_BOOL,
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_EQ,
                            src_ids=[a_id, b_id],
                            op_params=pack_comparison_params(0.0, is_scalar=False),
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.ne.Scalar" in target_str:
                    # ne(x, scalar) - element-wise not-equal with scalar
                    src_node = node.args[0]
                    scalar = float(node.args[1])
                    src_id = node_to_id[src_node]

                    fake_val = node.meta.get("val")
                    shape = list(fake_val.shape) if fake_val is not None else []

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=TYPE_BOOL,
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_NE,
                            src_ids=[src_id],
                            op_params=pack_comparison_params(scalar, is_scalar=True),
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.le.Tensor" in target_str:
                    # le(x, y) - element-wise less-than-or-equal
                    a_node, b_node = node.args[0], node.args[1]
                    a_id = node_to_id[a_node]
                    b_id = node_to_id[b_node]

                    fake_val = node.meta.get("val")
                    shape = list(fake_val.shape) if fake_val is not None else []

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=TYPE_BOOL,
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_LE,
                            src_ids=[a_id, b_id],
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.lt.Tensor" in target_str:
                    # lt(x, y) - element-wise less-than
                    a_node, b_node = node.args[0], node.args[1]
                    a_id = node_to_id[a_node]
                    b_id = node_to_id[b_node]

                    fake_val = node.meta.get("val")
                    shape = list(fake_val.shape) if fake_val is not None else []

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=TYPE_BOOL,
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_LT,
                            src_ids=[a_id, b_id],
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.gt.Tensor" in target_str:
                    # gt(x, y) - element-wise greater-than
                    a_node, b_node = node.args[0], node.args[1]
                    a_id = node_to_id[a_node]
                    b_id = node_to_id[b_node]

                    fake_val = node.meta.get("val")
                    shape = list(fake_val.shape) if fake_val is not None else []

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=TYPE_BOOL,
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_GT,
                            src_ids=[a_id, b_id],
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.ge.Tensor" in target_str:
                    # ge(x, y) - element-wise greater-than-or-equal
                    a_node, b_node = node.args[0], node.args[1]
                    a_id = node_to_id[a_node]
                    b_id = node_to_id[b_node]

                    fake_val = node.meta.get("val")
                    shape = list(fake_val.shape) if fake_val is not None else []

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=TYPE_BOOL,
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_GE,
                            src_ids=[a_id, b_id],
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.bitwise_and.Tensor" in target_str:
                    # bitwise_and(x, y)
                    a_node, b_node = node.args[0], node.args[1]
                    a_id = node_to_id[a_node]
                    b_id = node_to_id[b_node]

                    fake_val = node.meta.get("val")
                    shape = list(fake_val.shape) if fake_val is not None else []
                    out_dtype = (
                        getattr(fake_val, "dtype", torch.bool)
                        if fake_val is not None
                        else torch.bool
                    )

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=_torch_dtype_to_ir_type(out_dtype),
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_BITWISE_AND,
                            src_ids=[a_id, b_id],
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.bitwise_or.Tensor" in target_str:
                    # bitwise_or(x, y)
                    a_node, b_node = node.args[0], node.args[1]
                    a_id = node_to_id[a_node]
                    b_id = node_to_id[b_node]

                    fake_val = node.meta.get("val")
                    shape = list(fake_val.shape) if fake_val is not None else []
                    out_dtype = (
                        getattr(fake_val, "dtype", torch.bool)
                        if fake_val is not None
                        else torch.bool
                    )

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=_torch_dtype_to_ir_type(out_dtype),
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_BITWISE_OR,
                            src_ids=[a_id, b_id],
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.logical_not.default" in target_str:
                    # logical_not(x)
                    src_node = node.args[0]
                    src_id = node_to_id[src_node]

                    fake_val = node.meta.get("val")
                    shape = list(fake_val.shape) if fake_val is not None else []

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=TYPE_BOOL,
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_LOGICAL_NOT,
                            src_ids=[src_id],
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.any.dim" in target_str:
                    # any(x, dim, keepdim)
                    src_node = node.args[0]
                    dim = int(node.args[1]) if len(node.args) > 1 else 0
                    src_id = node_to_id[src_node]

                    fake_val = node.meta.get("val")
                    shape = list(fake_val.shape) if fake_val is not None else []

                    src_val = (
                        src_node.meta.get("val") if hasattr(src_node, "meta") else None
                    )
                    src_shape = list(src_val.shape) if src_val is not None else shape
                    ndim = len(src_shape)

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=TYPE_BOOL,
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_ANY,
                            src_ids=[src_id],
                            op_params=pack_any_params(dim, ndim),
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
