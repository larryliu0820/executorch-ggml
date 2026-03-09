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
    OP_RELU,
    OP_TANH,
    OP_LEAKY_RELU,
    OP_CONV_2D,
    OP_CONV_2D_DW,
    OP_CONV_1D,
    OP_CONV_1D_DW,
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
    OP_UPDATE_CACHE,
    OP_LAYER_NORM,
    OP_BATCH_NORM,
    OP_ARGMAX,
    OP_DIV,
    OP_PAD,
    # Types
    TYPE_F32,
    TYPE_F16,
    TYPE_I64,
    TYPE_I32,
    TYPE_BOOL,
    TYPE_BF16,
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
    pack_update_cache_params,
    pack_layer_norm_params,
    pack_batch_norm_params,
    pack_argmax_params,
    pack_conv1d_params,
    pack_pad_params,
    serialize_graph,
)

from executorch.exir._serialize._named_data_store import NamedDataStore


def _concrete_int(s) -> int:
    """Convert a SymInt (or plain int) to a concrete int without installing
    guards on the shape environment.  This avoids mutating
    ``shape_env.var_to_range`` during ``preprocess``."""
    if isinstance(s, int):
        return s
    # FX Node — resolve from FakeTensor metadata (scalar_tensor output).
    if isinstance(s, torch.fx.Node):
        fv = s.meta.get("val")
        if fv is not None and hasattr(fv, "item"):
            return int(fv.item())
        # Fall back to a large sentinel for "end" values
        return 2**62
    # torch.SymInt — read the hint (concrete value used during tracing)
    # without creating a guard that narrows var_to_range.
    return s.node.hint


def _resolve_shape(fake_val) -> List[int]:
    """Get a concrete integer shape list from a FakeTensor without guarding."""
    if fake_val is None or not hasattr(fake_val, "shape"):
        return []
    return [_concrete_int(s) for s in fake_val.shape]


# ---------------------------------------------------------------------------
# Symbolic expression bytecode compiler
# ---------------------------------------------------------------------------
#
# Opcodes for postfix bytecode encoding of sympy expressions.
# Used to serialize derived dynamic shape expressions (e.g. ((s0-1)//8)+1)
# from strided convolutions into the FlatBuffer IR.
#
# The C++ runtime evaluates these with a simple stack machine.

SYM_OP_PUSH_SYM  = 0x01  # 1-byte operand: sym_id
SYM_OP_PUSH_CONST = 0x02  # 4-byte operand: int32 LE
SYM_OP_ADD       = 0x10
SYM_OP_SUB       = 0x11
SYM_OP_MUL       = 0x12
SYM_OP_FLOORDIV  = 0x13
SYM_OP_MOD       = 0x14
SYM_OP_NEG       = 0x15

import struct as _struct


def _sympy_to_bytecode(expr, sym_id_map: dict) -> bytes:
    """Compile a sympy expression to postfix bytecode.

    Handles: Symbol, Integer, One, NegativeOne, Add, Mul,
    FloorDiv (torch.utils._sympy.functions.FloorDiv), Mod.
    """
    import sympy

    # Try to import torch's FloorDiv
    try:
        from torch.utils._sympy.functions import FloorDiv
    except ImportError:
        FloorDiv = None

    buf = bytearray()

    def _emit(node):
        if isinstance(node, sympy.Symbol):
            name = str(node)
            if name not in sym_id_map:
                sym_id_map[name] = len(sym_id_map)
            buf.append(SYM_OP_PUSH_SYM)
            buf.append(sym_id_map[name] & 0xFF)
            return

        if isinstance(node, sympy.Integer):
            val = int(node)
            buf.append(SYM_OP_PUSH_CONST)
            buf.extend(_struct.pack("<i", val))
            return

        # Add: sum of args
        if isinstance(node, sympy.Add):
            args = list(node.args)
            _emit(args[0])
            for a in args[1:]:
                _emit(a)
                buf.append(SYM_OP_ADD)
            return

        # Mul: product of args
        if isinstance(node, sympy.Mul):
            args = list(node.args)
            # Handle Mul(-1, x) as NEG(x)
            if len(args) == 2 and args[0] == sympy.S.NegativeOne:
                _emit(args[1])
                buf.append(SYM_OP_NEG)
                return
            _emit(args[0])
            for a in args[1:]:
                _emit(a)
                buf.append(SYM_OP_MUL)
            return

        # FloorDiv (torch-specific or sympy)
        if FloorDiv is not None and isinstance(node, FloorDiv):
            _emit(node.args[0])
            _emit(node.args[1])
            buf.append(SYM_OP_FLOORDIV)
            return

        # sympy.floor(a/b) pattern
        if isinstance(node, sympy.floor):
            inner = node.args[0]
            if isinstance(inner, sympy.Mul) and len(inner.args) == 2:
                a, b = inner.args
                if isinstance(b, sympy.Pow) and b.args[1] == -1:
                    _emit(a)
                    _emit(b.args[0])
                    buf.append(SYM_OP_FLOORDIV)
                    return
            # Fallback: emit floor arg, floordiv by 1 (identity)
            _emit(inner)
            return

        # Mod
        if isinstance(node, sympy.Mod):
            _emit(node.args[0])
            _emit(node.args[1])
            buf.append(SYM_OP_MOD)
            return

        raise ValueError(f"Unsupported sympy node type: {type(node).__name__} ({node})")

    _emit(expr)
    return bytes(buf)


def _eval_bytecode(code: bytes, sym_values: dict) -> int:
    """Evaluate postfix bytecode with given symbol values. Python-side mirror
    of the C++ eval_sym_expr() for testing."""
    stack = []
    i = 0
    while i < len(code):
        op = code[i]
        i += 1
        if op == SYM_OP_PUSH_SYM:
            sid = code[i]
            i += 1
            stack.append(sym_values[sid])
        elif op == SYM_OP_PUSH_CONST:
            val = _struct.unpack_from("<i", code, i)[0]
            i += 4
            stack.append(val)
        elif op == SYM_OP_ADD:
            b, a = stack.pop(), stack.pop()
            stack.append(a + b)
        elif op == SYM_OP_SUB:
            b, a = stack.pop(), stack.pop()
            stack.append(a - b)
        elif op == SYM_OP_MUL:
            b, a = stack.pop(), stack.pop()
            stack.append(a * b)
        elif op == SYM_OP_FLOORDIV:
            b, a = stack.pop(), stack.pop()
            # Python-style floor division (rounds toward -inf)
            stack.append(a // b)
        elif op == SYM_OP_MOD:
            b, a = stack.pop(), stack.pop()
            stack.append(a % b)
        elif op == SYM_OP_NEG:
            stack.append(-stack.pop())
        else:
            raise ValueError(f"Unknown bytecode op: 0x{op:02x}")
    assert len(stack) == 1, f"Stack not empty after eval: {stack}"
    return stack[0]


def _get_sym_dim_info(s, sym_id_map: dict):
    """Return (sym_dim_id, bytecode_or_None) for a single SymInt dimension.

    Returns:
      (-1, None)        — static dimension
      (id, None)        — simple symbol (direct lookup)
      (-2, bytecode)    — derived expression (evaluate bytecode)
    """
    if not isinstance(s, torch.SymInt):
        return (-1, None)
    import sympy
    expr = s.node.expr
    if isinstance(expr, sympy.Symbol):
        name = str(expr)
        if name not in sym_id_map:
            sym_id_map[name] = len(sym_id_map)
        return (sym_id_map[name], None)
    # Derived expression — compile to bytecode
    free = expr.free_symbols
    if len(free) >= 1:
        # Ensure all free symbols are registered
        for sym in free:
            name = str(sym)
            if name not in sym_id_map:
                sym_id_map[name] = len(sym_id_map)
        bytecode = _sympy_to_bytecode(expr, sym_id_map)
        return (-2, bytecode)
    # No free symbols — treat as static
    return (-1, None)


def _sym_dim_info_ggml(fake_val, sym_id_map: dict):
    """Compute ggml-order sym_dim_ids and packed sym_dim_exprs.

    Returns (sym_dim_ids_or_None, sym_dim_exprs_or_None).
    """
    from typing import Optional
    if fake_val is None or not hasattr(fake_val, "shape"):
        return (None, None)

    # Collect per-dim info in PyTorch order
    pt_info = [_get_sym_dim_info(s, sym_id_map) for s in fake_val.shape]
    if not pt_info or all(sid == -1 for sid, _ in pt_info):
        return (None, None)

    # Reverse to ggml order, pad to 4
    ggml_info = list(reversed(pt_info))
    while len(ggml_info) < 4:
        ggml_info.append((-1, None))
    ggml_info = ggml_info[:4]

    ggml_sym = [sid for sid, _ in ggml_info]
    has_exprs = any(bc is not None for _, bc in ggml_info)

    if not has_exprs:
        # No derived expressions — just return sym_dim_ids (backwards compat)
        if all(sid == -1 for sid in ggml_sym):
            return (None, None)
        return (ggml_sym, None)

    # Pack sym_dim_exprs: 4 entries, each prefixed by uint16 length
    exprs_buf = bytearray()
    for _, bc in ggml_info:
        if bc is not None:
            exprs_buf.extend(_struct.pack("<H", len(bc)))
            exprs_buf.extend(bc)
        else:
            exprs_buf.extend(_struct.pack("<H", 0))

    return (ggml_sym, bytes(exprs_buf))


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
    if dtype == torch.bfloat16:
        return TYPE_BF16
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

        # Maps symbolic variable name (e.g. "s0") → unique integer ID
        sym_id_map: Dict[str, int] = {}

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
        lifted_const_map = dict(
            getattr(sig, "inputs_to_lifted_tensor_constants", {}) or {}
        )  # node_name → constant FQN

        # ep.constants holds tensor constants (like attention masks and RoPE freqs)
        # that are not in ep.state_dict but are still compile-time constants.
        # We need both sources when resolving constant data.
        # ep.tensor_constants is the same pool under a different accessor.
        ep_constants = getattr(edge_program, "constants", {}) or {}
        ep_tensor_constants = getattr(edge_program, "tensor_constants", {}) or {}

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
                # Some placeholders map to parameters, buffers, or lifted
                # tensor constants.  There are three sources:
                #   1) ep.state_dict  – parameters and mutable buffers (KV caches)
                #   2) ep.constants   – non-mutable tensor constants (attention
                #                       masks, RoPE freqs_cos/sin, etc.)
                #   3) lifted_tensor_constants – scalar/tensor constants lifted
                #      by torch.export (e.g. scaling factors); these live in
                #      ep.constants or ep.tensor_constants.
                # We check all three so that every compile-time constant is
                # handled as an IR constant rather than a runtime input.
                fqn = (param_map.get(node_name)
                       or buffer_map.get(node_name)
                       or lifted_const_map.get(node_name))
                tensor_from_state = fqn is not None and fqn in edge_program.state_dict
                tensor_from_constants = fqn is not None and (
                    fqn in ep_constants or fqn in ep_tensor_constants
                )
                is_constant = tensor_from_state or tensor_from_constants

                if is_constant:
                    if tensor_from_state:
                        tensor = edge_program.state_dict[fqn]
                    elif fqn in ep_constants:
                        tensor = ep_constants[fqn]
                    else:
                        tensor = ep_tensor_constants[fqn]
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
                    elif t_cpu.dtype == torch.float32 and t_cpu.ndim >= 3:
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

                    # Skip non-tensor placeholders (e.g. SymInt from
                    # dynamic shapes).  These are passed by ExecuTorch
                    # for output-shape bookkeeping but are not real
                    # tensor inputs to the ggml graph.
                    if fake_val is None or not hasattr(fake_val, "shape"):
                        continue

                    shape = _resolve_shape(fake_val)

                    # Compute per-dimension symbolic variable IDs + expressions.
                    ggml_sym, ggml_exprs = _sym_dim_info_ggml(fake_val, sym_id_map)

                    tid = alloc_id()
                    # Runtime input dtype based on FakeTensor meta when available.
                    in_dtype = (
                        getattr(fake_val, "dtype", torch.float32)
                        if fake_val is not None and hasattr(fake_val, "dtype")
                        else torch.float32
                    )
                    # Declare int64 inputs as int32 in the IR.  ggml ops
                    # (get_rows, set_rows) work with I32 indices, and the
                    # C++ execute() already converts ET int64 → ggml I32
                    # during input copy.  This avoids the need for deferred
                    # I64→I32 casts and handles VIEW→CAST chains correctly.
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
                    shape = _resolve_shape(fake_val)
                    out_dtype = (
                        getattr(fake_val, "dtype", torch.float32)
                        if fake_val is not None
                        else torch.float32
                    )
                    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, sym_id_map)

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=_torch_dtype_to_ir_type(out_dtype),
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_LINEAR,
                            src_ids=src_ids,
                            sym_dim_ids=_vsym,
                            sym_dim_exprs=_vexprs,
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
                        idx_shape = _resolve_shape(idx_fv)
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
                    shape = _resolve_shape(fake_val)
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
                    shape = _resolve_shape(fake_val)
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

                elif "aten.relu.default" in target_str:
                    src_node = node.args[0]
                    src_id = node_to_id[src_node]
                    fake_val = node.meta.get("val")
                    shape = _resolve_shape(fake_val)
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
                            op=OP_RELU,
                            src_ids=[src_id],
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.tanh.default" in target_str:
                    src_node = node.args[0]
                    src_id = node_to_id[src_node]
                    fake_val = node.meta.get("val")
                    shape = _resolve_shape(fake_val)
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
                            op=OP_TANH,
                            src_ids=[src_id],
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.mul.Tensor" in target_str:
                    # mul(a, b)
                    # NOTE: Broadcasting should be handled by BroadcastCanonicalizationPass
                    # which inserts explicit expand_copy ops. This lowering expects inputs
                    # to already have matching shapes.
                    a_node, b_node = node.args[0], node.args[1]
                    a_id = node_to_id[a_node]
                    b_id = node_to_id[b_node]
                    fake_val = node.meta.get("val")
                    out_shape = _resolve_shape(fake_val)
                    out_dtype = (
                        getattr(fake_val, "dtype", torch.float32)
                        if fake_val is not None
                        else torch.float32
                    )

                    # Broadcasting is handled natively by the C++ ggml backend
                    # (ggml_mul supports ggml_can_repeat(b, a)).
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

                elif (
                    "aten.div.Tensor_mode" in target_str
                    or "aten.div.Tensor" in target_str
                ):
                    # div(a, b) → OP_DIV
                    a_node, b_node = node.args[0], node.args[1]
                    a_id = node_to_id[a_node]
                    b_id = node_to_id[b_node]
                    fake_val = node.meta.get("val")
                    out_shape = _resolve_shape(fake_val)
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
                            op=OP_DIV,
                            src_ids=[a_id, b_id],
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.neg.default" in target_str:
                    src_node = node.args[0]
                    src_id = node_to_id[src_node]
                    fake_val = node.meta.get("val")
                    shape = _resolve_shape(fake_val)
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
                    shape = _resolve_shape(fake_val)
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
                    "aten.squeeze.dims" in target_str
                    or "aten.squeeze_copy.dims" in target_str
                ):
                    # squeeze(x, dims) — remove size-1 dims → equivalent to reshape/VIEW
                    src_node = node.args[0]
                    src_id = node_to_id[src_node]
                    fake_val = node.meta.get("val")
                    shape = _resolve_shape(fake_val)
                    out_dtype = (
                        getattr(fake_val, "dtype", torch.float32)
                        if fake_val is not None
                        else torch.float32
                    )

                    ggml_ne = _pytorch_shape_to_ggml_ne(shape)
                    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, sym_id_map)
                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=_torch_dtype_to_ir_type(out_dtype),
                            ne=ggml_ne,
                            op=OP_VIEW,
                            src_ids=[src_id],
                            op_params=pack_view_params(ggml_ne),
                            sym_dim_ids=_vsym,
                            sym_dim_exprs=_vexprs,
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
                    src_val = src_node.meta.get("val")
                    src_shape = _resolve_shape(src_val)
                    src_ndim = len(src_shape) if len(src_shape) > 0 else 1
                    if dim < 0:
                        dim += src_ndim + 1
                    fake_val = node.meta.get("val")
                    shape = _resolve_shape(fake_val)
                    out_dtype = (
                        getattr(fake_val, "dtype", torch.float32)
                        if fake_val is not None
                        else torch.float32
                    )
                    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, sym_id_map)
                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=_torch_dtype_to_ir_type(out_dtype),
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_UNSQUEEZE,
                            src_ids=[src_id],
                            op_params=pack_unsqueeze_params(dim, src_ndim),
                            sym_dim_ids=_vsym,
                            sym_dim_exprs=_vexprs,
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.transpose.int" in target_str:
                    src_node = node.args[0]
                    dim0 = int(node.args[1])
                    dim1 = int(node.args[2])
                    src_id = node_to_id[src_node]
                    fake_val = node.meta.get("val")
                    shape = _resolve_shape(fake_val)
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
                    start_i = _concrete_int(start) if start is not None else 0
                    # If end is None, represent as a large positive bound (runtime will clamp)
                    end_i = _concrete_int(end) if end is not None else (2**62)
                    src_id = node_to_id[src_node]
                    fake_val = node.meta.get("val")
                    shape = _resolve_shape(fake_val)
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
                    src_shape = _resolve_shape(src_val)
                    ndim = len(src_shape) if src_shape else len(shape)

                    # If start or end hit the 2**62 sentinel (unresolvable SymInt),
                    # derive concrete trace-time values from the output shape.
                    if start_i == 2**62:
                        start_i = 0
                    if end_i == 2**62 and shape:
                        # output_shape[dim] == (end - start) / step
                        d = dim if dim >= 0 else ndim + dim
                        if d < len(shape):
                            end_i = start_i + shape[d] * step

                    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, sym_id_map)
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
                            sym_dim_ids=_vsym,
                            sym_dim_exprs=_vexprs,
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.cat.default" in target_str:
                    tensors = list(node.args[0])
                    dim = int(node.args[1]) if len(node.args) > 1 else 0
                    src_ids = [node_to_id[t] for t in tensors]
                    fake_val = node.meta.get("val")
                    shape = _resolve_shape(fake_val)
                    out_dtype = (
                        getattr(fake_val, "dtype", torch.float32)
                        if fake_val is not None
                        else torch.float32
                    )
                    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, sym_id_map)
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
                            sym_dim_ids=_vsym,
                            sym_dim_exprs=_vexprs,
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
                    shape = _resolve_shape(fake_val)
                    out_dtype = (
                        getattr(fake_val, "dtype", torch.float32)
                        if fake_val is not None
                        else torch.float32
                    )
                    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, sym_id_map)
                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=_torch_dtype_to_ir_type(out_dtype),
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_REPEAT_INTERLEAVE,
                            src_ids=[src_id],
                            op_params=pack_repeat_interleave_params(dim, repeats),
                            sym_dim_ids=_vsym,
                            sym_dim_exprs=_vexprs,
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
                    shape = _resolve_shape(fake_val)
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
                            idx_shape = _resolve_shape(idx_fv)
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
                        src_shape = _resolve_shape(src_val)
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
                    src_shape = _resolve_shape(src_val)

                    # Normalize negative index
                    if idx < 0 and src_shape:
                        idx = src_shape[dim] + idx

                    fake_val = node.meta.get("val")
                    out_shape = _resolve_shape(fake_val)
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
                    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, sym_id_map)
                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=_torch_dtype_to_ir_type(out_dtype),
                            ne=ggml_ne,
                            op=OP_VIEW,
                            src_ids=[slice_id],
                            op_params=pack_view_params(ggml_ne),
                            sym_dim_ids=_vsym,
                            sym_dim_exprs=_vexprs,
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
                    shape = _resolve_shape(fake_val)
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
                    shape = _resolve_shape(fake_val)
                    out_dtype = (
                        getattr(fake_val, "dtype", torch.float32)
                        if fake_val is not None
                        else torch.float32
                    )
                    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, sym_id_map)
                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=_torch_dtype_to_ir_type(out_dtype),
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_LLAMA_ATTENTION,
                            src_ids=src_ids,
                            op_params=b"",  # TODO: pack model params (n_head, head_dim, start_pos binding)
                            sym_dim_ids=_vsym,
                            sym_dim_exprs=_vexprs,
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
                    shape = _resolve_shape(fake_val)
                    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, sym_id_map)

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=TYPE_F32,
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_MUL_MAT,
                            src_ids=[weight_id, input_id],
                            sym_dim_ids=_vsym,
                            sym_dim_exprs=_vexprs,
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
                    shape = _resolve_shape(fake_val)
                    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, sym_id_map)

                    # First: MUL_MAT
                    mm_id = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=mm_id,
                            tensor_type=TYPE_F32,
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_MUL_MAT,
                            src_ids=[weight_id, input_id],
                            sym_dim_ids=_vsym,
                            sym_dim_exprs=_vexprs,
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
                            sym_dim_ids=_vsym,
                            sym_dim_exprs=_vexprs,
                        )
                    )
                    node_to_id[node] = add_id

                elif "aten.leaky_relu.default" in target_str:
                    src_node = node.args[0]
                    negative_slope = node.args[1] if len(node.args) > 1 else 0.01
                    src_id = node_to_id[src_node]

                    fake_val = node.meta.get("val")
                    shape = _resolve_shape(fake_val)

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

                elif "aten.constant_pad_nd.default" in target_str:
                    # constant_pad_nd(input, pad_list, value=0.0)
                    src_node = node.args[0]
                    src_id = node_to_id[src_node]
                    pad_list = [int(p) for p in node.args[1]]
                    fill_value = float(node.args[2]) if len(node.args) > 2 else 0.0

                    fake_val = node.meta.get("val")
                    out_shape = _resolve_shape(fake_val)
                    out_dtype = (
                        getattr(fake_val, "dtype", torch.float32)
                        if fake_val is not None
                        else torch.float32
                    )
                    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, sym_id_map)

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=_torch_dtype_to_ir_type(out_dtype),
                            ne=_pytorch_shape_to_ggml_ne(out_shape),
                            op=OP_PAD,
                            src_ids=[src_id],
                            op_params=pack_pad_params(pad_list, fill_value),
                            sym_dim_ids=_vsym,
                            sym_dim_exprs=_vexprs,
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

                    weight_id = node_to_id[weight_node]
                    if bias_node is not None:
                        bias_id = node_to_id[bias_node]
                    else:
                        bias_id = None

                    input_id = node_to_id[input_node]

                    fake_val = node.meta.get("val")
                    shape = _resolve_shape(fake_val)

                    is_depthwise = groups > 1

                    # Build src_ids: [weight, input, bias?]
                    if bias_id is not None:
                        src_ids = [weight_id, input_id, bias_id]
                    else:
                        src_ids = [weight_id, input_id]

                    # Detect 1D vs 2D by stride length
                    is_conv1d = len(stride) == 1

                    if is_conv1d:
                        op_params = pack_conv1d_params(
                            stride[0], padding[0], dilation[0], groups
                        )
                        op_code = OP_CONV_1D_DW if is_depthwise else OP_CONV_1D
                    else:
                        op_params = pack_conv2d_params(stride, padding, dilation, groups)
                        op_code = OP_CONV_2D_DW if is_depthwise else OP_CONV_2D

                    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, sym_id_map)
                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=TYPE_F32,
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=op_code,
                            src_ids=src_ids,
                            op_params=op_params,
                            sym_dim_ids=_vsym,
                            sym_dim_exprs=_vexprs,
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
                    shape = _resolve_shape(fake_val)

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
                        dims = [_concrete_int(d) for d in list(dim)]
                    else:
                        dims = [_concrete_int(dim)]

                    # Canonicalize dims to positive indices when possible.
                    # For NCHW 4D tensors, -1/-2 correspond to W/H.
                    dims = [d + 4 if d < 0 else d for d in dims]
                    src_id = node_to_id[src_node]

                    fake_val = node.meta.get("val")
                    shape = _resolve_shape(fake_val)

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
                    # expand is a broadcast.  If ggml_repeat can handle the
                    # shape at trace time, emit REPEAT; otherwise pass through
                    # the source and rely on consumer ops for broadcast.
                    src_node = node.args[0]
                    src_id = node_to_id[src_node]
                    fake_val = node.meta.get("val")
                    shape = _resolve_shape(fake_val)
                    out_dtype = (
                        getattr(fake_val, "dtype", torch.float32)
                        if fake_val is not None
                        else torch.float32
                    )

                    src_fake = (
                        src_node.meta.get("val") if hasattr(src_node, "meta") else None
                    )
                    src_shape = _resolve_shape(src_fake)

                    src_ne = _pytorch_shape_to_ggml_ne(src_shape)
                    dst_ne = _pytorch_shape_to_ggml_ne(shape)
                    can_repeat = all(
                        src_ne[i] == 1 or src_ne[i] == dst_ne[i] for i in range(4)
                    )

                    if can_repeat and src_ne != dst_ne:
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
                        # Same shape or not repeatable — pass through.
                        node_to_id[node] = src_id

                elif "aten.repeat.default" in target_str:
                    # repeat(input, repeats) → tile input by repeats along each dim
                    src_node = node.args[0]
                    src_id = node_to_id[src_node]
                    fake_val = node.meta.get("val")
                    out_shape = _resolve_shape(fake_val)
                    out_dtype = (
                        getattr(fake_val, "dtype", torch.float32)
                        if fake_val is not None
                        else torch.float32
                    )
                    dst_ne = _pytorch_shape_to_ggml_ne(out_shape)
                    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, sym_id_map)

                    # Create a shape-only "like" tensor for ggml_repeat.
                    like_id = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=like_id,
                            tensor_type=_torch_dtype_to_ir_type(out_dtype),
                            ne=dst_ne,
                            op=OP_NONE,
                            is_input=False,
                            sym_dim_ids=_vsym,
                            sym_dim_exprs=_vexprs,
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
                            sym_dim_ids=_vsym,
                            sym_dim_exprs=_vexprs,
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
                    shape = _resolve_shape(fake_val)
                    # Use concrete output shape from FakeTensor meta rather than
                    # node.args[1] which may contain SymInt from dynamic export.
                    new_shape = (
                        [_concrete_int(d) for d in shape]
                        if shape
                        else [_concrete_int(d) for d in node.args[1]]
                    )

                    # Pack the shape in ggml ne order (reversed from PyTorch)
                    # since the C++ runtime passes these directly to ggml_reshape_4d.
                    ggml_ne = _pytorch_shape_to_ggml_ne(new_shape)
                    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, sym_id_map)

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=TYPE_F32,
                            ne=ggml_ne,
                            op=OP_VIEW,
                            src_ids=[src_id],
                            op_params=pack_view_params(ggml_ne),
                            sym_dim_ids=_vsym,
                            sym_dim_exprs=_vexprs,
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.add.Tensor" in target_str:
                    # add(x, y, alpha=1)
                    # NOTE: Broadcasting should be handled by BroadcastCanonicalizationPass
                    # which inserts explicit expand_copy ops. This lowering expects inputs
                    # to already have matching shapes.
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
                    out_shape = _resolve_shape(fake_val)
                    out_dtype = (
                        getattr(fake_val, "dtype", torch.float32)
                        if fake_val is not None
                        else torch.float32
                    )

                    # Broadcasting is handled natively by the C++ ggml backend
                    # (ggml_add supports broadcast via ggml_can_repeat).
                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=_torch_dtype_to_ir_type(out_dtype),
                            ne=_pytorch_shape_to_ggml_ne(out_shape),
                            op=OP_ADD,
                            src_ids=[x_id, y_id],
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.type_as.default" in target_str:
                    # type_as(input, other) casts input to other's dtype.
                    src_node = node.args[0]
                    other_node = node.args[1]
                    src_id = node_to_id[src_node]

                    src_val = src_node.meta.get("val") if hasattr(src_node, "meta") else None
                    other_val = other_node.meta.get("val") if hasattr(other_node, "meta") else None
                    src_dtype = getattr(src_val, "dtype", torch.float32) if src_val is not None else torch.float32
                    target_dtype = getattr(other_val, "dtype", torch.float32) if other_val is not None else torch.float32

                    if src_dtype == target_dtype:
                        # No cast needed - use identity
                        node_to_id[node] = src_id
                    else:
                        # Need to cast
                        fake_val = node.meta.get("val")
                        shape = _resolve_shape(fake_val)
                        tid = alloc_id()
                        ir_tensors.append(
                            IrTensor(
                                tensor_id=tid,
                                tensor_type=_torch_dtype_to_ir_type(target_dtype),
                                ne=_pytorch_shape_to_ggml_ne(shape),
                                op=OP_CAST,
                                src_ids=[src_id],
                                op_params=pack_cast_params(_torch_dtype_to_ir_type(target_dtype)),
                            )
                        )
                        node_to_id[node] = tid

                elif (
                    "aten.alias.default" in target_str
                    or "aten.alias_copy.default" in target_str
                ):
                    src_node = node.args[0]
                    node_to_id[node] = node_to_id[src_node]

                elif "aten.clone.default" in target_str:
                    # clone(x) - treat as identity for ggml (no explicit copy needed)
                    src_node = node.args[0]
                    node_to_id[node] = node_to_id[src_node]

                elif "aten._to_copy.default" in target_str:
                    # _to_copy(x, dtype=...) - dtype conversion with copy
                    src_node = node.args[0]
                    src_id = node_to_id[src_node]

                    src_val = src_node.meta.get("val") if hasattr(src_node, "meta") else None
                    src_dtype = getattr(src_val, "dtype", torch.float32) if src_val is not None else torch.float32

                    fake_val = node.meta.get("val")
                    out_dtype = fake_val.dtype if fake_val is not None else torch.float32
                    out_shape = _resolve_shape(fake_val)

                    if src_dtype == out_dtype:
                        # No cast needed - use identity
                        node_to_id[node] = src_id
                    else:
                        # Need to cast
                        tid = alloc_id()
                        ir_tensors.append(
                            IrTensor(
                                tensor_id=tid,
                                tensor_type=_torch_dtype_to_ir_type(out_dtype),
                                ne=_pytorch_shape_to_ggml_ne(out_shape),
                                op=OP_CAST,
                                src_ids=[src_id],
                                op_params=pack_cast_params(_torch_dtype_to_ir_type(out_dtype)),
                            )
                        )
                        node_to_id[node] = tid

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
                    shape = _resolve_shape(fake_val)

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
                    # NOTE: Broadcasting should be handled by BroadcastCanonicalizationPass
                    # which inserts explicit expand_copy ops. This lowering expects inputs
                    # to already have matching shapes.
                    x_node, y_node = node.args[0], node.args[1]
                    x_id = node_to_id[x_node]
                    y_id = node_to_id[y_node]

                    fake_val = node.meta.get("val")
                    out_shape = _resolve_shape(fake_val)
                    out_dtype = (
                        getattr(fake_val, "dtype", torch.float32)
                        if fake_val is not None
                        else torch.float32
                    )

                    # Broadcasting is handled natively by ggml_sub.
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
                    shape = _resolve_shape(fake_val)
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
                    shape = _resolve_shape(fake_val)
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
                    shape = _resolve_shape(fake_val)
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
                    shape = _resolve_shape(fake_val)
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
                    shape = _resolve_shape(fake_val)
                    out_dtype = (
                        getattr(fake_val, "dtype", torch.float32)
                        if fake_val is not None
                        else torch.float32
                    )
                    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, sym_id_map)

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=_torch_dtype_to_ir_type(out_dtype),
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_BMM,
                            src_ids=[a_id, b_id],
                            sym_dim_ids=_vsym,
                            sym_dim_exprs=_vexprs,
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.sigmoid.default" in target_str:
                    src_node = node.args[0]
                    src_id = node_to_id[src_node]

                    fake_val = node.meta.get("val")
                    shape = _resolve_shape(fake_val)
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
                    shape = _resolve_shape(fake_val)
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
                    shape = _resolve_shape(fake_val)
                    out_dtype = (
                        getattr(fake_val, "dtype", torch.float32)
                        if fake_val is not None
                        else torch.float32
                    )
                    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, sym_id_map)

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=_torch_dtype_to_ir_type(out_dtype),
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_WHERE,
                            src_ids=[cond_id, x_id, y_id],
                            sym_dim_ids=_vsym,
                            sym_dim_exprs=_vexprs,
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.full_like.default" in target_str:
                    # full_like(input, fill_value, ...) - create tensor like input filled with fill_value
                    # This is typically used to create constant tensors (e.g., -inf for masking).
                    fill_value = float(node.args[1])

                    fake_val = node.meta.get("val")
                    shape = _resolve_shape(fake_val)
                    out_dtype = (
                        getattr(fake_val, "dtype", torch.float32)
                        if fake_val is not None
                        else torch.float32
                    )

                    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, sym_id_map)
                    has_dynamic = _vsym is not None and any(s != -1 for s in _vsym)

                    if has_dynamic:
                        # Dynamic shape — emit OP_FULL so C++ fills at runtime
                        tid = alloc_id()
                        ir_tensors.append(
                            IrTensor(
                                tensor_id=tid,
                                tensor_type=_torch_dtype_to_ir_type(out_dtype),
                                ne=_pytorch_shape_to_ggml_ne(shape),
                                op=OP_FULL,
                                src_ids=[],
                                op_params=pack_full_params(fill_value),
                                sym_dim_ids=_vsym,
                                sym_dim_exprs=_vexprs,
                            )
                        )
                    else:
                        # Static shape — keep existing constant path (OP_NONE + baked data)
                        import hashlib
                        import numpy as np

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
                    shape = _resolve_shape(fake_val)
                    out_dtype = (
                        getattr(fake_val, "dtype", torch.int64)
                        if fake_val is not None
                        else torch.int64
                    )

                    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, sym_id_map)
                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=_torch_dtype_to_ir_type(out_dtype),
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_ARANGE,
                            src_ids=[],
                            op_params=pack_arange_params(start, step),
                            sym_dim_ids=_vsym,
                            sym_dim_exprs=_vexprs,
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.arange.default" in target_str:
                    # arange(end, ...) - generates [0, 1, ..., end-1]
                    # Different from start_step: only end is specified
                    start = 0.0
                    step = 1.0

                    fake_val = node.meta.get("val")
                    shape = _resolve_shape(fake_val)
                    out_dtype = (
                        getattr(fake_val, "dtype", torch.int64)
                        if fake_val is not None
                        else torch.int64
                    )

                    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, sym_id_map)
                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=_torch_dtype_to_ir_type(out_dtype),
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_ARANGE,
                            src_ids=[],
                            op_params=pack_arange_params(start, step),
                            sym_dim_ids=_vsym,
                            sym_dim_exprs=_vexprs,
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.matmul.default" in target_str:
                    # matmul(a, b) - general matrix multiplication
                    a_node = node.args[0]
                    b_node = node.args[1]
                    a_id = node_to_id[a_node]
                    b_id = node_to_id[b_node]

                    fake_val = node.meta.get("val")
                    shape = _resolve_shape(fake_val)
                    out_dtype = (
                        getattr(fake_val, "dtype", torch.float32)
                        if fake_val is not None
                        else torch.float32
                    )
                    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, sym_id_map)

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=_torch_dtype_to_ir_type(out_dtype),
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_MUL_MAT,  # Use MUL_MAT for matmul
                            src_ids=[a_id, b_id],
                            sym_dim_ids=_vsym,
                            sym_dim_exprs=_vexprs,
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.to.dtype" in target_str or "aten.to.dtype_layout" in target_str:
                    # Type casting - use CAST op
                    src_node = node.args[0]
                    src_id = node_to_id[src_node]

                    fake_val = node.meta.get("val")
                    shape = _resolve_shape(fake_val)
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
                            op=OP_CAST,
                            src_ids=[src_id],
                            op_params=pack_cast_params(_torch_dtype_to_ir_type(out_dtype)),
                        )
                    )
                    node_to_id[node] = tid

                # Note: aten._assert_scalar and aten.sym_constrain_range_for_size
                # are removed by RemoveGraphAssertsPass before lowering.

                elif "aten.full.default" in target_str:
                    # full(size, fill_value, ...) - creates tensor filled with fill_value
                    fill_value = float(node.args[1]) if len(node.args) > 1 else 0.0

                    fake_val = node.meta.get("val")
                    shape = _resolve_shape(fake_val)
                    out_dtype = (
                        getattr(fake_val, "dtype", torch.float32)
                        if fake_val is not None
                        else torch.float32
                    )

                    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, sym_id_map)
                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=_torch_dtype_to_ir_type(out_dtype),
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_FULL,
                            src_ids=[],
                            op_params=pack_full_params(fill_value),
                            sym_dim_ids=_vsym,
                            sym_dim_exprs=_vexprs,
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.cumsum.default" in target_str:
                    # cumsum(x, dim) - cumulative sum along dimension
                    src_node = node.args[0]
                    dim = int(node.args[1]) if len(node.args) > 1 else 0
                    src_id = node_to_id[src_node]

                    fake_val = node.meta.get("val")
                    shape = _resolve_shape(fake_val)
                    out_dtype = (
                        getattr(fake_val, "dtype", torch.int64)
                        if fake_val is not None
                        else torch.int64
                    )
                    ndim = len(shape)
                    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, sym_id_map)

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=_torch_dtype_to_ir_type(out_dtype),
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_CUMSUM,
                            src_ids=[src_id],
                            op_params=pack_cumsum_params(dim, ndim),
                            sym_dim_ids=_vsym,
                            sym_dim_exprs=_vexprs,
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.eq.Scalar" in target_str:
                    # eq(x, scalar) - element-wise equality with scalar
                    src_node = node.args[0]
                    scalar = float(node.args[1])
                    src_id = node_to_id[src_node]

                    fake_val = node.meta.get("val")
                    shape = _resolve_shape(fake_val)
                    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, sym_id_map)

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=TYPE_BOOL,
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_EQ,
                            src_ids=[src_id],
                            op_params=pack_comparison_params(scalar, is_scalar=True),
                            sym_dim_ids=_vsym,
                            sym_dim_exprs=_vexprs,
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.eq.Tensor" in target_str:
                    # eq(x, y) - element-wise equality
                    a_node, b_node = node.args[0], node.args[1]
                    a_id = node_to_id[a_node]
                    b_id = node_to_id[b_node]

                    fake_val = node.meta.get("val")
                    shape = _resolve_shape(fake_val)
                    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, sym_id_map)

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=TYPE_BOOL,
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_EQ,
                            src_ids=[a_id, b_id],
                            op_params=pack_comparison_params(0.0, is_scalar=False),
                            sym_dim_ids=_vsym,
                            sym_dim_exprs=_vexprs,
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.ne.Scalar" in target_str:
                    # ne(x, scalar) - element-wise not-equal with scalar
                    src_node = node.args[0]
                    scalar = float(node.args[1])
                    src_id = node_to_id[src_node]

                    fake_val = node.meta.get("val")
                    shape = _resolve_shape(fake_val)
                    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, sym_id_map)

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=TYPE_BOOL,
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_NE,
                            src_ids=[src_id],
                            op_params=pack_comparison_params(scalar, is_scalar=True),
                            sym_dim_ids=_vsym,
                            sym_dim_exprs=_vexprs,
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.le.Tensor" in target_str:
                    # le(x, y) - element-wise less-than-or-equal
                    a_node, b_node = node.args[0], node.args[1]
                    a_id = node_to_id[a_node]
                    b_id = node_to_id[b_node]

                    fake_val = node.meta.get("val")
                    shape = _resolve_shape(fake_val)
                    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, sym_id_map)

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=TYPE_BOOL,
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_LE,
                            src_ids=[a_id, b_id],
                            sym_dim_ids=_vsym,
                            sym_dim_exprs=_vexprs,
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.lt.Tensor" in target_str:
                    # lt(x, y) - element-wise less-than
                    a_node, b_node = node.args[0], node.args[1]
                    a_id = node_to_id[a_node]
                    b_id = node_to_id[b_node]

                    fake_val = node.meta.get("val")
                    shape = _resolve_shape(fake_val)
                    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, sym_id_map)

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=TYPE_BOOL,
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_LT,
                            src_ids=[a_id, b_id],
                            sym_dim_ids=_vsym,
                            sym_dim_exprs=_vexprs,
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.gt.Tensor" in target_str:
                    # gt(x, y) - element-wise greater-than
                    a_node, b_node = node.args[0], node.args[1]
                    a_id = node_to_id[a_node]
                    b_id = node_to_id[b_node]

                    fake_val = node.meta.get("val")
                    shape = _resolve_shape(fake_val)
                    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, sym_id_map)

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=TYPE_BOOL,
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_GT,
                            src_ids=[a_id, b_id],
                            sym_dim_ids=_vsym,
                            sym_dim_exprs=_vexprs,
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.ge.Tensor" in target_str:
                    # ge(x, y) - element-wise greater-than-or-equal
                    a_node, b_node = node.args[0], node.args[1]
                    a_id = node_to_id[a_node]
                    b_id = node_to_id[b_node]

                    fake_val = node.meta.get("val")
                    shape = _resolve_shape(fake_val)
                    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, sym_id_map)

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=TYPE_BOOL,
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_GE,
                            src_ids=[a_id, b_id],
                            sym_dim_ids=_vsym,
                            sym_dim_exprs=_vexprs,
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.bitwise_and.Tensor" in target_str:
                    # bitwise_and(x, y)
                    a_node, b_node = node.args[0], node.args[1]
                    a_id = node_to_id[a_node]
                    b_id = node_to_id[b_node]

                    fake_val = node.meta.get("val")
                    shape = _resolve_shape(fake_val)
                    out_dtype = (
                        getattr(fake_val, "dtype", torch.bool)
                        if fake_val is not None
                        else torch.bool
                    )
                    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, sym_id_map)

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=_torch_dtype_to_ir_type(out_dtype),
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_BITWISE_AND,
                            src_ids=[a_id, b_id],
                            sym_dim_ids=_vsym,
                            sym_dim_exprs=_vexprs,
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.logical_and.default" in target_str:
                    # logical_and(x, y) — identical to bitwise_and for bool tensors
                    a_node, b_node = node.args[0], node.args[1]
                    a_id = node_to_id[a_node]
                    b_id = node_to_id[b_node]

                    fake_val = node.meta.get("val")
                    shape = _resolve_shape(fake_val)
                    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, sym_id_map)

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=TYPE_BOOL,
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_BITWISE_AND,
                            src_ids=[a_id, b_id],
                            sym_dim_ids=_vsym,
                            sym_dim_exprs=_vexprs,
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.bitwise_or.Tensor" in target_str:
                    # bitwise_or(x, y)
                    a_node, b_node = node.args[0], node.args[1]
                    a_id = node_to_id[a_node]
                    b_id = node_to_id[b_node]

                    fake_val = node.meta.get("val")
                    shape = _resolve_shape(fake_val)
                    out_dtype = (
                        getattr(fake_val, "dtype", torch.bool)
                        if fake_val is not None
                        else torch.bool
                    )
                    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, sym_id_map)

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=_torch_dtype_to_ir_type(out_dtype),
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_BITWISE_OR,
                            src_ids=[a_id, b_id],
                            sym_dim_ids=_vsym,
                            sym_dim_exprs=_vexprs,
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.logical_not.default" in target_str:
                    # logical_not(x)
                    src_node = node.args[0]
                    src_id = node_to_id[src_node]

                    fake_val = node.meta.get("val")
                    shape = _resolve_shape(fake_val)
                    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, sym_id_map)

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=TYPE_BOOL,
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_LOGICAL_NOT,
                            src_ids=[src_id],
                            sym_dim_ids=_vsym,
                            sym_dim_exprs=_vexprs,
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.bitwise_not.default" in target_str:
                    # bitwise_not(x) — identical to logical_not for bool tensors
                    src_node = node.args[0]
                    src_id = node_to_id[src_node]

                    fake_val = node.meta.get("val")
                    shape = _resolve_shape(fake_val)
                    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, sym_id_map)

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=TYPE_BOOL,
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_LOGICAL_NOT,
                            src_ids=[src_id],
                            sym_dim_ids=_vsym,
                            sym_dim_exprs=_vexprs,
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.any.dim" in target_str:
                    # any(x, dim, keepdim)
                    src_node = node.args[0]
                    dim = int(node.args[1]) if len(node.args) > 1 else 0
                    src_id = node_to_id[src_node]

                    fake_val = node.meta.get("val")
                    shape = _resolve_shape(fake_val)
                    _vsym, _vexprs = _sym_dim_info_ggml(fake_val, sym_id_map)

                    src_val = (
                        src_node.meta.get("val") if hasattr(src_node, "meta") else None
                    )
                    src_shape = _resolve_shape(src_val) or shape
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
                            sym_dim_ids=_vsym,
                            sym_dim_exprs=_vexprs,
                        )
                    )
                    node_to_id[node] = tid

                elif "llama.update_cache.default" in target_str:
                    # llama.update_cache(value, cache, start_pos) -> cache
                    # Updates cache at start_pos with new values.
                    # Args: value (new K/V), cache (mutable buffer), start_pos (int64 scalar)
                    value_node = node.args[0]
                    cache_node = node.args[1]

                    value_id = node_to_id[value_node]
                    cache_id = node_to_id[cache_node]

                    # start_pos comes from: item(select(cache_position, 0, 0))
                    # We need to trace back to find the original cache_position tensor
                    start_pos_node = node.args[2]

                    # Helper to trace back through item/select to find the tensor
                    def trace_to_tensor(n):
                        if n in node_to_id:
                            return node_to_id[n]
                        # Check if this is an item or select node
                        if hasattr(n, "target"):
                            target_name = str(n.target)
                            if "item" in target_name or "select" in target_name:
                                # Trace to input
                                if n.args:
                                    return trace_to_tensor(n.args[0])
                        return None

                    start_pos_id = trace_to_tensor(start_pos_node)
                    if start_pos_id is None:
                        raise RuntimeError(
                            f"Could not find tensor for update_cache start_pos: {start_pos_node}"
                        )

                    fake_val = node.meta.get("val")
                    shape = _resolve_shape(fake_val)
                    out_dtype = (
                        getattr(fake_val, "dtype", torch.float32)
                        if fake_val is not None
                        else torch.float32
                    )

                    # Determine sequence dimension from cache shape
                    # Typical shapes: [batch, seq, n_heads, head_dim] -> seq_dim=1
                    #                 [batch, n_heads, seq, head_dim] -> seq_dim=2
                    cache_val = cache_node.meta.get("val") if hasattr(cache_node, "meta") else None
                    cache_shape = _resolve_shape(cache_val)
                    value_val = value_node.meta.get("val") if hasattr(value_node, "meta") else None
                    value_shape = _resolve_shape(value_val)

                    # The seq dim is where cache_shape differs from value_shape
                    seq_dim = 1  # default
                    if len(cache_shape) == len(value_shape) and len(cache_shape) >= 2:
                        for i in range(1, len(cache_shape)):
                            if cache_shape[i] != value_shape[i]:
                                seq_dim = i
                                break

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=_torch_dtype_to_ir_type(out_dtype),
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_UPDATE_CACHE,
                            src_ids=[cache_id, value_id, start_pos_id],
                            op_params=pack_update_cache_params(seq_dim),
                        )
                    )
                    node_to_id[node] = tid

                elif "aten.split_with_sizes_copy.default" in target_str:
                    # split_with_sizes_copy(x, split_sizes, dim=0) → tuple of slices
                    # Decompose into multiple SLICE ops
                    src_node = node.args[0]
                    split_sizes = list(node.args[1])
                    dim = int(node.args[2]) if len(node.args) > 2 else 0
                    src_id = node_to_id[src_node]

                    src_val = src_node.meta.get("val")
                    src_shape = _resolve_shape(src_val)
                    ndim = len(src_shape) if src_shape else 4

                    # Normalize negative dim
                    if dim < 0:
                        dim = ndim + dim

                    # Emit one SLICE per chunk
                    chunk_ids = []
                    offset = 0
                    fake_vals = node.meta.get("val")  # tuple of FakeTensors
                    for i, sz in enumerate(split_sizes):
                        start_i = offset
                        end_i = offset + sz
                        offset = end_i

                        # Get shape from FakeTensor tuple
                        if isinstance(fake_vals, (list, tuple)) and i < len(fake_vals):
                            chunk_fv = fake_vals[i]
                            chunk_shape = _resolve_shape(chunk_fv)
                            chunk_dtype = getattr(chunk_fv, "dtype", torch.float32)
                        else:
                            chunk_fv = None
                            chunk_shape = list(src_shape)
                            chunk_shape[dim] = sz
                            chunk_dtype = torch.float32

                        _vsym, _vexprs = _sym_dim_info_ggml(chunk_fv, sym_id_map)
                        chunk_tid = alloc_id()
                        ir_tensors.append(
                            IrTensor(
                                tensor_id=chunk_tid,
                                tensor_type=_torch_dtype_to_ir_type(chunk_dtype),
                                ne=_pytorch_shape_to_ggml_ne(chunk_shape),
                                op=OP_SLICE,
                                src_ids=[src_id],
                                op_params=pack_slice_params(
                                    dim, start_i, end_i, 1, ndim
                                ),
                                sym_dim_ids=_vsym,
                                sym_dim_exprs=_vexprs,
                            )
                        )
                        chunk_ids.append(chunk_tid)

                    # Store as list for getitem resolution
                    node_to_id[node] = chunk_ids

                elif "aten.native_layer_norm.default" in target_str:
                    # native_layer_norm(input, normalized_shape, weight, bias, eps)
                    # Returns tuple (output, mean, rstd)
                    input_node = node.args[0]
                    # normalized_shape = node.args[1]  # not needed for IR
                    weight_node = node.args[2] if len(node.args) > 2 else None
                    bias_node = node.args[3] if len(node.args) > 3 else None
                    eps = float(node.args[4]) if len(node.args) > 4 else 1e-5

                    input_id = node_to_id[input_node]
                    has_weight = weight_node is not None and not (
                        isinstance(weight_node, type(None))
                    )
                    has_bias = bias_node is not None and not (
                        isinstance(bias_node, type(None))
                    )

                    src_ids = [input_id]
                    if has_weight:
                        src_ids.append(node_to_id[weight_node])
                    if has_bias:
                        src_ids.append(node_to_id[bias_node])

                    # Output shape comes from the first element of the tuple
                    fake_val = node.meta.get("val")
                    if isinstance(fake_val, (list, tuple)):
                        out_fv = fake_val[0]
                    else:
                        out_fv = fake_val
                    shape = _resolve_shape(out_fv)
                    out_dtype = (
                        getattr(out_fv, "dtype", torch.float32)
                        if out_fv is not None
                        else torch.float32
                    )

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=_torch_dtype_to_ir_type(out_dtype),
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_LAYER_NORM,
                            src_ids=src_ids,
                            op_params=pack_layer_norm_params(
                                eps, has_weight, has_bias
                            ),
                        )
                    )
                    # Store as single int — getitem(0) will resolve to this
                    node_to_id[node] = tid

                elif "aten._native_batch_norm_legit_no_training.default" in target_str:
                    # _native_batch_norm_legit_no_training(input, weight, bias, running_mean, running_var, momentum, eps)
                    # Returns tuple (output, mean, rstd)
                    input_node = node.args[0]
                    weight_node = node.args[1] if len(node.args) > 1 else None
                    bias_node = node.args[2] if len(node.args) > 2 else None
                    mean_node = node.args[3] if len(node.args) > 3 else None
                    var_node = node.args[4] if len(node.args) > 4 else None
                    # momentum = node.args[5]  # not used
                    eps = float(node.args[6]) if len(node.args) > 6 else 1e-5

                    input_id = node_to_id[input_node]

                    src_ids = [input_id]
                    has_w = isinstance(weight_node, torch.fx.Node) and weight_node in node_to_id
                    has_b = isinstance(bias_node, torch.fx.Node) and bias_node in node_to_id
                    has_m = isinstance(mean_node, torch.fx.Node) and mean_node in node_to_id
                    has_v = isinstance(var_node, torch.fx.Node) and var_node in node_to_id

                    if has_w:
                        src_ids.append(node_to_id[weight_node])
                    if has_b:
                        src_ids.append(node_to_id[bias_node])
                    if has_m:
                        src_ids.append(node_to_id[mean_node])
                    if has_v:
                        src_ids.append(node_to_id[var_node])

                    # Output shape from tuple[0]
                    fake_val = node.meta.get("val")
                    if isinstance(fake_val, (list, tuple)):
                        out_fv = fake_val[0]
                    else:
                        out_fv = fake_val
                    shape = _resolve_shape(out_fv)
                    out_dtype = (
                        getattr(out_fv, "dtype", torch.float32)
                        if out_fv is not None
                        else torch.float32
                    )

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=_torch_dtype_to_ir_type(out_dtype),
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_BATCH_NORM,
                            src_ids=src_ids,
                            op_params=pack_batch_norm_params(eps),
                        )
                    )
                    # Store as single int — getitem(0) will resolve to this
                    node_to_id[node] = tid

                elif "aten.argmax.default" in target_str:
                    # argmax(x, dim=None, keepdim=False)
                    src_node = node.args[0]
                    dim = int(node.args[1]) if len(node.args) > 1 and node.args[1] is not None else -1
                    src_id = node_to_id[src_node]

                    fake_val = node.meta.get("val")
                    shape = _resolve_shape(fake_val)

                    src_val = src_node.meta.get("val")
                    src_shape = _resolve_shape(src_val)
                    ndim = len(src_shape) if src_shape else len(shape) + 1

                    tid = alloc_id()
                    ir_tensors.append(
                        IrTensor(
                            tensor_id=tid,
                            tensor_type=TYPE_I32,  # ggml_argmax returns I32
                            ne=_pytorch_shape_to_ggml_ne(shape),
                            op=OP_ARGMAX,
                            src_ids=[src_id],
                            op_params=pack_argmax_params(dim, ndim),
                        )
                    )
                    node_to_id[node] = tid

                elif callable(target) and "getitem" in str(target):
                    # Generic getitem handler for tuple outputs
                    # (layer_norm, batch_norm, split_with_sizes_copy)
                    src_node = node.args[0]
                    idx = int(node.args[1])
                    src_val = node_to_id.get(src_node)

                    if isinstance(src_val, list):
                        # split_with_sizes_copy: list of tensor IDs
                        node_to_id[node] = src_val[idx]
                    elif isinstance(src_val, int):
                        # layer_norm / batch_norm: single tensor ID (only index 0 is the output)
                        if idx == 0:
                            node_to_id[node] = src_val
                        else:
                            # mean/rstd outputs — not used in inference, skip
                            pass
                    else:
                        raise RuntimeError(
                            f"getitem: unexpected source type {type(src_val)} for {src_node}"
                        )

                else:
                    raise RuntimeError(
                        f"GgmlBackend.preprocess: unsupported op {target}"
                    )

            elif node.op == "output":
                # Mark output tensors with their position in the return
                # tuple.  ExecuTorch passes output args in this same order,
                # so we store the index in input_index (reused for outputs)
                # and sort by it in the C++ build_graph.
                output_args = node.args[0]
                if not isinstance(output_args, (list, tuple)):
                    output_args = [output_args]
                for out_idx, out_node in enumerate(output_args):
                    if out_node in node_to_id:
                        tid = node_to_id[out_node]
                        # Find the IrTensor and mark it as output
                        for ir_t in ir_tensors:
                            if ir_t.id == tid:
                                ir_t.is_output = True
                                ir_t.input_index = out_idx  # output ordering
                                break

        # Serialize to FlatBuffer
        fb_bytes = serialize_graph(ir_tensors)
        return PreprocessResult(
            processed_bytes=fb_bytes,
            data_store_output=data_store.get_named_data_store_output(),
        )
