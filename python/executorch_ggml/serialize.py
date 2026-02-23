"""FlatBuffer serialization helpers for the ggml IR.

Uses the `flatbuffers` Python library to construct ggml_ir.GgmlGraph
byte blobs that can be consumed by the C++ runtime.
"""

import struct
from typing import List, Optional

import flatbuffers

# ---------------------------------------------------------------------------
# Inline constants mirroring schema/ggml_ir.fbs enums
# (avoids requiring generated code at import time)
# ---------------------------------------------------------------------------

# OpCode (keep in sync with schema/ggml_ir.fbs)
OP_NONE = 0

# Basic math
OP_ADD = 1
OP_MUL_MAT = 2
OP_MUL = 10
OP_NEG = 11
OP_SUB = 12
OP_MUL_SCALAR = 13
OP_POW = 14

# Trigonometric
OP_COS = 15
OP_SIN = 16

# BMM (batch matmul)
OP_BMM = 17

# Activations
OP_SIGMOID = 18
OP_SOFTMAX = 19

# NN
OP_LEAKY_RELU = 3
OP_LINEAR = 20
OP_EMBEDDING = 21
OP_SILU = 22

# Vision
OP_CONV_2D = 4
OP_CONV_2D_DW = 5     # depthwise conv2d

# Activations / reductions / views
OP_HARDTANH = 6        # clamp
OP_MEAN = 7            # mean(dim)
OP_RSQRT = 30
OP_VIEW = 8            # reshape
OP_UNSQUEEZE = 31

# Layout / indexing
OP_PERMUTE = 9         # permute dims
OP_TRANSPOSE = 40
OP_SLICE = 41
OP_CAT = 42
OP_REPEAT_INTERLEAVE = 43
OP_INDEX = 44
OP_INDEX_PUT = 45
OP_REPEAT = 46
OP_INDEX_MULTI = 47  # Multi-index gather: x[idx0, idx1, ...] via linearized lookup
OP_CAST = 48  # Type cast: src_ids=[x], op_params=int32 target_type (TensorType enum)

# Conditional
OP_WHERE = 50

# Mask computation / comparison ops
OP_ARANGE = 51
OP_FULL = 52
OP_CUMSUM = 53
OP_EQ = 54
OP_NE = 55
OP_LE = 56
OP_LT = 57
OP_GT = 58
OP_GE = 59
OP_BITWISE_AND = 70
OP_BITWISE_OR = 71
OP_LOGICAL_NOT = 72
OP_ANY = 73

# KV cache ops
OP_UPDATE_CACHE = 74

# Fused attention (llama.cpp/ggml)
OP_LLAMA_ATTENTION = 60

# TensorType
TYPE_F32 = 0
TYPE_F16 = 1
TYPE_I64 = 2
TYPE_I32 = 3
TYPE_BOOL = 4
TYPE_BF16 = 5


# ---------------------------------------------------------------------------
# Data class for an IR tensor before serialization
# ---------------------------------------------------------------------------

class IrTensor:
    """Intermediate representation of a single tensor in the ggml graph."""

    __slots__ = (
        "id",
        "tensor_type",
        "ne",
        "op",
        "src_ids",
        "op_params",
        "data_key",
        "is_input",
        "is_output",
        "input_index",
    )

    def __init__(
        self,
        tensor_id: int,
        tensor_type: int = TYPE_F32,
        ne: Optional[List[int]] = None,
        op: int = OP_NONE,
        src_ids: Optional[List[int]] = None,
        op_params: Optional[bytes] = None,
        data_key: Optional[str] = None,
        is_input: bool = False,
        is_output: bool = False,
        input_index: int = -1,
    ):
        self.id = tensor_id
        self.tensor_type = tensor_type
        self.ne = ne or []
        self.op = op
        self.src_ids = src_ids or []
        self.op_params = op_params or b""
        self.data_key = data_key or ""
        self.is_input = is_input
        self.is_output = is_output
        self.input_index = input_index


# ---------------------------------------------------------------------------
# FlatBuffer serialization
# ---------------------------------------------------------------------------

def _pytorch_shape_to_ggml_ne(shape: List[int]) -> List[int]:
    """Convert PyTorch shape [d0, d1, ..., dn] → ggml ne [dn, ..., d1, d0].

    Result is padded to 4 dimensions with trailing 1s.
    """
    ne = list(reversed(shape))
    while len(ne) < 4:
        ne.append(1)
    return ne[:4]


def serialize_graph(tensors: List[IrTensor], n_threads: int = 1) -> bytes:
    """Serialize a list of IrTensor objects into a FlatBuffer GgmlGraph blob."""
    builder = flatbuffers.Builder(4096)

    # Build tensors in reverse order (FlatBuffers vectors are built back-to-front)
    tensor_offsets = []
    for t in reversed(tensors):
        # Vectors
        if t.ne:
            ne_vec = builder.CreateNumpyVector(
                _to_int64_array(t.ne)
            ) if _has_numpy() else _create_int64_vector(builder, t.ne)
        else:
            ne_vec = None

        if t.src_ids:
            src_ids_vec = _create_int32_vector(builder, t.src_ids)
        else:
            src_ids_vec = None

        if t.op_params:
            op_params_vec = _create_uint8_vector(builder, t.op_params)
        else:
            op_params_vec = None

        if t.data_key:
            data_key_off = builder.CreateString(t.data_key)
        else:
            data_key_off = None

        # Build the Tensor table
        # Start table with the right number of fields (10 fields)
        builder.StartObject(10)

        builder.PrependInt32Slot(0, t.id, 0)           # id
        builder.PrependInt32Slot(1, t.tensor_type, 0)  # type
        if ne_vec is not None:
            builder.PrependUOffsetTRelativeSlot(2, ne_vec, 0)  # ne
        builder.PrependInt32Slot(3, t.op, 0)           # op
        if src_ids_vec is not None:
            builder.PrependUOffsetTRelativeSlot(4, src_ids_vec, 0)  # src_ids
        if op_params_vec is not None:
            builder.PrependUOffsetTRelativeSlot(5, op_params_vec, 0)  # op_params
        if data_key_off is not None:
            builder.PrependUOffsetTRelativeSlot(6, data_key_off, 0)  # data_key
        builder.PrependBoolSlot(7, t.is_input, False)    # is_input
        builder.PrependBoolSlot(8, t.is_output, False)   # is_output
        builder.PrependInt32Slot(9, t.input_index, -1)   # input_index

        tensor_offsets.append(builder.EndObject())

    # Reverse so they match the original order
    tensor_offsets.reverse()

    # Build tensors vector
    builder.StartVector(4, len(tensor_offsets), 4)
    for off in reversed(tensor_offsets):
        builder.PrependUOffsetTRelative(off)
    tensors_vec = builder.EndVector()

    # Build GgmlGraph table (2 fields: tensors, n_threads)
    builder.StartObject(2)
    builder.PrependUOffsetTRelativeSlot(0, tensors_vec, 0)  # tensors
    builder.PrependInt32Slot(1, n_threads, 1)               # n_threads
    graph_offset = builder.EndObject()

    builder.Finish(graph_offset)
    return bytes(builder.Output())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_numpy() -> bool:
    try:
        import numpy  # noqa: F401
        return True
    except ImportError:
        return False


def _to_int64_array(values: List[int]):
    import numpy as np
    return np.array(values, dtype=np.int64)


def _create_int64_vector(builder: flatbuffers.Builder, values: List[int]):
    builder.StartVector(8, len(values), 8)
    for v in reversed(values):
        builder.PrependInt64(v)
    return builder.EndVector()


def _create_int32_vector(builder: flatbuffers.Builder, values: List[int]):
    builder.StartVector(4, len(values), 4)
    for v in reversed(values):
        builder.PrependInt32(v)
    return builder.EndVector()


def _create_uint8_vector(builder: flatbuffers.Builder, data: bytes):
    builder.StartVector(1, len(data), 1)
    for b in reversed(data):
        builder.PrependByte(b)
    return builder.EndVector()


def pack_float(value: float) -> bytes:
    """Pack a single float into little-endian bytes (for op_params)."""
    return struct.pack("<f", value)


def pack_i32(value: int) -> bytes:
    return struct.pack("<i", int(value))


def pack_transpose_params(dim0: int, dim1: int, ndim: int) -> bytes:
    """Pack transpose parameters. ndim is the full PyTorch rank (not ggml_n_dims)."""
    return struct.pack("<iii", int(dim0), int(dim1), int(ndim))


def pack_unsqueeze_params(dim: int) -> bytes:
    return pack_i32(dim)


def pack_cat_params(ggml_axis: int) -> bytes:
    """Pack cat params. Takes the pre-computed ggml axis (not PyTorch dim)."""
    return pack_i32(ggml_axis)


def pack_repeat_interleave_params(dim: int, repeats: int) -> bytes:
    return struct.pack("<ii", int(dim), int(repeats))


def pack_index_params(dim: int) -> bytes:
    return pack_i32(dim)


def pack_index_put_params(dim: int) -> bytes:
    return pack_i32(dim)


def pack_index_put_multi_params(nindices: int, present_mask: int) -> bytes:
    # Layout: int32 nindices, int32 present_mask
    return struct.pack('<ii', int(nindices), int(present_mask))


def pack_slice_params(dim: int, start: int, end: int, step: int, ndim: int = 4) -> bytes:
    """Pack slice parameters. ndim is the source tensor's PyTorch rank."""
    return struct.pack("<iqqqI", int(dim), int(start), int(end), int(step), int(ndim))


def pack_conv2d_params(
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    groups: int,
) -> bytes:
    """Pack conv2d parameters into little-endian bytes.

    Layout: stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups
    Each as int32 (4 bytes).
    """
    return struct.pack(
        "<iiiiiii",
        stride[0] if len(stride) > 0 else 1,
        stride[1] if len(stride) > 1 else 1,
        padding[0] if len(padding) > 0 else 0,
        padding[1] if len(padding) > 1 else 0,
        dilation[0] if len(dilation) > 0 else 1,
        dilation[1] if len(dilation) > 1 else 1,
        groups,
    )


def pack_hardtanh_params(min_val: float, max_val: float) -> bytes:
    """Pack hardtanh/clamp parameters into little-endian bytes.

    Layout: min_val, max_val (each as float32, 4 bytes).
    """
    return struct.pack("<ff", min_val, max_val)


def pack_mean_params(dims) -> bytes:
    """Pack mean parameters into little-endian bytes.

    Layout:
      ndims (int32)
      dims[0..ndims-1] (int32 each)

    We keep the dims in *PyTorch* dimension order.
    """
    if isinstance(dims, int):
        dims = [dims]
    dims = list(dims)
    ndims = len(dims)
    fmt = f"<i{ndims}i"
    return struct.pack(fmt, ndims, *dims)


def pack_view_params(new_shape: List[int]) -> bytes:
    """Pack view/reshape parameters into little-endian bytes.

    Layout: ndims (int32), followed by ndims int64 values.
    """
    ndims = len(new_shape)
    fmt = f"<i{ndims}q"  # i for ndims, q for each int64
    return struct.pack(fmt, ndims, *new_shape)


def pack_permute_params(perm: List[int]) -> bytes:
    """Pack permute parameters into little-endian bytes.

    Layout: ndims (int32), followed by ndims int32 values.
    """
    ndims = len(perm)
    fmt = f"<i{ndims}i"  # i for ndims, i for each int32
    return struct.pack(fmt, ndims, *perm)


def pack_index_multi_params(src_shape: List[int]) -> bytes:
    """Pack multi-index gather parameters into little-endian bytes.

    Encodes the source tensor shape so the C++ runtime can compute
    row strides for linearized multi-dimensional indexing.

    Layout: ndims (int32), followed by ndims int64 values (src dimension sizes).
    """
    ndims = len(src_shape)
    fmt = f"<i{ndims}q"  # i for ndims, q for each int64
    return struct.pack(fmt, ndims, *[int(d) for d in src_shape])


def pack_cast_params(target_type: int) -> bytes:
    """Pack cast parameters into little-endian bytes.

    Layout: target_type (int32) - TensorType enum value.
    """
    return struct.pack("<i", target_type)


def pack_softmax_params(dim: int, ndim: int) -> bytes:
    """Pack softmax parameters: dim (int32), ndim (int32)."""
    return struct.pack("<ii", dim, ndim)


def pack_pow_params(exponent: float) -> bytes:
    """Pack pow parameters: exponent (float32)."""
    return struct.pack("<f", exponent)


def pack_arange_params(start: float, step: float) -> bytes:
    """Pack arange parameters: start (float64), step (float64)."""
    return struct.pack("<dd", float(start), float(step))


def pack_full_params(fill_value: float) -> bytes:
    """Pack full parameters: fill_value (float64)."""
    return struct.pack("<d", float(fill_value))


def pack_cumsum_params(dim: int, ndim: int) -> bytes:
    """Pack cumsum parameters: dim (int32), ndim (int32)."""
    return struct.pack("<ii", int(dim), int(ndim))


def pack_comparison_params(scalar: float = 0.0, is_scalar: bool = False) -> bytes:
    """Pack comparison parameters: scalar (float64), is_scalar (int32)."""
    return struct.pack("<di", float(scalar), int(is_scalar))


def pack_any_params(dim: int, ndim: int) -> bytes:
    """Pack any.dim parameters: dim (int32), ndim (int32)."""
    return struct.pack("<ii", int(dim), int(ndim))


def pack_update_cache_params(seq_dim: int = 1) -> bytes:
    """Pack update_cache parameters: seq_dim (int32) - the sequence dimension in the cache tensor.

    The update_cache op inserts new values into a cache tensor along the sequence dimension.
    For typical KV cache with shape [batch, seq_len, n_heads, head_dim], seq_dim=1.
    For shape [batch, n_heads, seq_len, head_dim], seq_dim=2.
    """
    return struct.pack("<i", int(seq_dim))
