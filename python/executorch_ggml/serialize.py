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

# OpCode
OP_NONE = 0
OP_ADD = 1
OP_MUL_MAT = 2
OP_LEAKY_RELU = 3

# TensorType
TYPE_F32 = 0
TYPE_F16 = 1


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
        "data",
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
        data: Optional[bytes] = None,
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
        self.data = data or b""
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

        if t.data:
            data_vec = _create_uint8_vector(builder, t.data)
        else:
            data_vec = None

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
        if data_vec is not None:
            builder.PrependUOffsetTRelativeSlot(6, data_vec, 0)  # data
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
