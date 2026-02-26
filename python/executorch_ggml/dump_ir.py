#!/usr/bin/env python3
"""Deserialize a .pte file and print the ggml IR graph.

Usage:
    python -m executorch_ggml.dump_ir path/to/model.pte
"""

import json
import struct
import sys
from pathlib import Path

from executorch.exir._serialize._flatbuffer import _program_flatbuffer_to_json

# Generated FlatBuffer Python bindings
from executorch_ggml.ggml_ir.GgmlGraph import GgmlGraph
from executorch_ggml.ggml_ir.OpCode import OpCode
from executorch_ggml.ggml_ir.TensorType import TensorType


# --- OpCode names -----------------------------------------------------------

_OPCODE_NAMES = {
    OpCode.NONE: "NONE",
    OpCode.ADD: "ADD", OpCode.MUL_MAT: "MUL_MAT", OpCode.MUL: "MUL",
    OpCode.NEG: "NEG", OpCode.SUB: "SUB", OpCode.MUL_SCALAR: "MUL_SCALAR",
    OpCode.POW: "POW", OpCode.COS: "COS", OpCode.SIN: "SIN",
    OpCode.BMM: "BMM", OpCode.SIGMOID: "SIGMOID", OpCode.SOFTMAX: "SOFTMAX",
    OpCode.LINEAR: "LINEAR", OpCode.EMBEDDING: "EMBEDDING", OpCode.SILU: "SILU",
    OpCode.LEAKY_RELU: "LEAKY_RELU", OpCode.CONV_2D: "CONV_2D",
    OpCode.CONV_2D_DW: "CONV_2D_DW", OpCode.HARDTANH: "HARDTANH",
    OpCode.MEAN: "MEAN", OpCode.RSQRT: "RSQRT", OpCode.VIEW: "VIEW",
    OpCode.UNSQUEEZE: "UNSQUEEZE", OpCode.PERMUTE: "PERMUTE",
    OpCode.TRANSPOSE: "TRANSPOSE", OpCode.SLICE: "SLICE", OpCode.CAT: "CAT",
    OpCode.REPEAT_INTERLEAVE: "REPEAT_INTERLEAVE", OpCode.INDEX: "INDEX",
    OpCode.INDEX_PUT: "INDEX_PUT", OpCode.REPEAT: "REPEAT",
    OpCode.INDEX_MULTI: "INDEX_MULTI", OpCode.CAST: "CAST",
    OpCode.WHERE: "WHERE",
    OpCode.ARANGE: "ARANGE", OpCode.FULL: "FULL", OpCode.CUMSUM: "CUMSUM",
    OpCode.EQ: "EQ", OpCode.NE: "NE", OpCode.LE: "LE", OpCode.LT: "LT",
    OpCode.GT: "GT", OpCode.GE: "GE",
    OpCode.LLAMA_ATTENTION: "LLAMA_ATTENTION",
    OpCode.BITWISE_AND: "BITWISE_AND", OpCode.BITWISE_OR: "BITWISE_OR",
    OpCode.LOGICAL_NOT: "LOGICAL_NOT", OpCode.ANY: "ANY",
    OpCode.UPDATE_CACHE: "UPDATE_CACHE",
}

_TYPE_NAMES = {
    TensorType.F32: "f32", TensorType.F16: "f16", TensorType.I64: "i64",
    TensorType.I32: "i32", TensorType.BOOL: "bool", TensorType.BF16: "bf16",
}


def _decode_op_params(op: int, raw: bytes) -> str:
    """Decode op_params bytes into a human-readable string."""
    if not raw:
        return ""
    try:
        if op == OpCode.VIEW:
            ndims = struct.unpack_from("<i", raw, 0)[0]
            dims = [struct.unpack_from("<q", raw, 4 + i * 8)[0] for i in range(ndims)]
            return f"view_shape={dims}"
        elif op == OpCode.PERMUTE:
            ndims = struct.unpack_from("<i", raw, 0)[0]
            perm = [struct.unpack_from("<i", raw, 4 + i * 4)[0] for i in range(ndims)]
            return f"perm={perm}"
        elif op == OpCode.TRANSPOSE:
            d0, d1, nd = struct.unpack_from("<iii", raw, 0)
            return f"dim0={d0} dim1={d1} ndim={nd}"
        elif op == OpCode.SLICE:
            dim = struct.unpack_from("<i", raw, 0)[0]
            start, end, step = struct.unpack_from("<qqq", raw, 4)
            parts = [f"dim={dim} start={start} end={end} step={step}"]
            if len(raw) >= 32:
                nd = struct.unpack_from("<I", raw, 28)[0]
                parts.append(f"ndim={nd}")
            return " ".join(parts)
        elif op == OpCode.MUL_SCALAR:
            s = struct.unpack_from("<f", raw, 0)[0]
            return f"scalar={s}"
        elif op == OpCode.SOFTMAX:
            dim, ndim = struct.unpack_from("<ii", raw, 0)
            return f"dim={dim} ndim={ndim}"
        elif op in (OpCode.EQ, OpCode.NE):
            scalar, is_scalar = struct.unpack_from("<di", raw, 0)
            return f"scalar={scalar} is_scalar={is_scalar}"
        elif op in (OpCode.LE, OpCode.LT, OpCode.GT, OpCode.GE):
            return ""
        elif op in (OpCode.CUMSUM, OpCode.ANY):
            dim, ndim = struct.unpack_from("<ii", raw, 0)
            return f"dim={dim} ndim={ndim}"
        elif op == OpCode.ARANGE:
            start, step = struct.unpack_from("<dd", raw, 0)
            return f"start={start} step={step}"
        elif op == OpCode.FULL:
            fill = struct.unpack_from("<d", raw, 0)[0]
            return f"fill={fill}"
        elif op == OpCode.CAST:
            tt = struct.unpack_from("<i", raw, 0)[0]
            return f"target={_TYPE_NAMES.get(tt, str(tt))}"
        elif op == OpCode.REPEAT_INTERLEAVE:
            dim, reps = struct.unpack_from("<ii", raw, 0)
            return f"dim={dim} repeats={reps}"
        elif op == OpCode.INDEX:
            dim = struct.unpack_from("<i", raw, 0)[0]
            return f"dim={dim}"
        elif op == OpCode.INDEX_PUT:
            nidx, pm = struct.unpack_from("<ii", raw, 0)
            return f"nidx={nidx} present_mask={pm}"
        elif op == OpCode.UPDATE_CACHE:
            seq_dim = struct.unpack_from("<i", raw, 0)[0]
            return f"seq_dim={seq_dim}"
        elif op == OpCode.POW:
            exp = struct.unpack_from("<f", raw, 0)[0]
            return f"exp={exp}"
    except struct.error:
        pass
    return f"raw[{len(raw)}]"


# --- Extract ggml flatbuffer from .pte ----------------------------------------

def extract_ggml_blobs(pte_path: str) -> list[bytes]:
    """Extract all ggml delegate FlatBuffer blobs from a .pte file."""
    with open(pte_path, "rb") as f:
        pte_data = f.read()

    # Use ExecuTorch's flatc to get the program JSON
    result = _program_flatbuffer_to_json(pte_data)
    program = json.loads(result)

    blobs = []
    segments = program.get("segments", [])

    # The segment offsets are relative to the extended header.
    # ExecuTorch .pte layout: [flatbuffer header] [padding] [segment data...]
    # The segments section starts after the main flatbuffer.
    # We need to find the segment base offset.

    # The main flatbuffer size is stored in the first 8 bytes as:
    #   bytes 0-3: flatbuffer "magic" or offset
    #   The segments start after the flatbuffer data.
    # Actually, ExecuTorch extended header is:
    #   0-3: magic "et12"  (or flatbuffer root offset)
    #   4-7: flatbuffer size
    #   8+: flatbuffer data
    # Segments are appended after the flatbuffer.

    # Let's figure out the segment base from the flatbuffer size.
    # The flatbuffer starts at offset 0. Its size = position of first segment.
    # For segment[1] (offset=0, the ggml IR), the actual file offset is:
    #   segment_base + segment.offset

    # Heuristic: find segment_base by reading the flatbuffer size.
    # FlatBuffers store the root table offset at byte 0 as a uint32.
    fb_root_offset = struct.unpack_from("<I", pte_data, 0)[0]
    # A rough estimate: the flatbuffer ends around fb_root_offset + a few KB.
    # But actually, ExecuTorch uses an extended header format.
    # Let's look for "eh00" or other magic.

    # Simpler: ExecuTorch .pte extended format has:
    #   header_length (4 bytes) at offset 4 if magic matches.
    # If it's a plain flatbuffer, segment_base = total_fb_size.
    # The easiest way: check each segment offset to see if it makes sense.

    # Find segment base by looking at what offset gives valid flatbuffer data
    # for segment 1 (the ggml IR blob).
    if len(segments) < 2:
        return blobs

    # The segment base is typically right after the main flatbuffer.
    # We can detect it by scanning for the segment base that produces
    # valid FlatBuffer data for segment 1.
    # Try common ExecuTorch header sizes.

    # Find the ExecuTorch extended header ("eh00") which contains the
    # segment base offset.  The header is embedded in the flatbuffer data
    # (typically near offset 8).
    eh_pos = pte_data.find(b"eh00")
    if eh_pos >= 0 and eh_pos < 256:
        header_size = struct.unpack_from("<I", pte_data, eh_pos + 4)[0]
        program_size = struct.unpack_from("<Q", pte_data, eh_pos + 8)[0]
        segment_base_offset = struct.unpack_from("<Q", pte_data, eh_pos + 16)[0]
    else:
        # Fallback: estimate from file size and last segment
        last_seg = segments[-1]
        last_seg_end = last_seg["offset"] + last_seg["size"]
        segment_base_offset = len(pte_data) - last_seg_end
        segment_base_offset = (segment_base_offset // 4096) * 4096

    for plan in program.get("execution_plan", []):
        for delegate in plan.get("delegates", []):
            if delegate["id"] != "GgmlBackend":
                continue
            processed = delegate.get("processed", {})
            if isinstance(processed, dict) and processed.get("location") == "SEGMENT":
                seg_idx = processed["index"]
                seg = segments[seg_idx]
                offset = segment_base_offset + seg["offset"]
                size = seg["size"]
                blob = pte_data[offset:offset + size]
                blobs.append(blob)
                print(f"Extracted ggml IR from segment[{seg_idx}]: "
                      f"offset={offset}, size={size}")

    return blobs


# --- Deserialize and print ----------------------------------------------------

def dump_graph(blob: bytes):
    """Deserialize a ggml IR FlatBuffer blob and print the graph."""
    graph = GgmlGraph.GetRootAs(blob)
    n = graph.TensorsLength()
    print(f"\nGgml IR Graph: {n} tensors, n_threads={graph.NThreads()}")
    print("=" * 90)

    for i in range(n):
        t = graph.Tensors(i)
        tid = t.Id()
        op = t.Op()
        op_name = _OPCODE_NAMES.get(op, f"OP_{op}")
        ttype = _TYPE_NAMES.get(t.Type(), f"type_{t.Type()}")

        # Shape
        ne = [t.Ne(d) for d in range(t.NeLength())] if t.NeLength() > 0 else [1]

        # Source IDs
        src_ids = [t.SrcIds(s) for s in range(t.SrcIdsLength())] if t.SrcIdsLength() > 0 else []

        # Flags
        flags = []
        if t.IsInput():
            flags.append(f"INPUT[{t.InputIndex()}]")
        if t.IsOutput():
            flags.append(f"OUTPUT[{t.InputIndex()}]")
        if t.DataKey():
            key = t.DataKey().decode()
            if key:
                short_key = key.split(".")[-1] if "." in key else key
                flags.append(f"CONST({short_key})")

        # Dynamic dims
        dd = [t.DynamicDims(d) for d in range(t.DynamicDimsLength())] if t.DynamicDimsLength() > 0 else []
        if any(dd):
            dyn_str = "".join("D" if d else "." for d in dd)
            flags.append(f"dyn=[{dyn_str}]")

        # Op params
        op_params_raw = bytes(t.OpParamsAsNumpy()) if t.OpParamsLength() > 0 else b""
        params_str = _decode_op_params(op, op_params_raw)

        # Format output
        ne_str = ",".join(str(x) for x in ne)
        src_str = ",".join(str(x) for x in src_ids)
        flag_str = " ".join(flags)

        line = f"  t{tid:3d}  {op_name:20s}  {ttype:4s}  ({ne_str:20s})"
        if src_ids:
            line += f"  <- [{src_str}]"
        if params_str:
            line += f"  {params_str}"
        if flag_str:
            line += f"  {flag_str}"

        print(line)

    print("=" * 90)


# --- Main ---------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print(f"Usage: python -m executorch_ggml.dump_ir <pte_file>")
        sys.exit(1)

    pte_path = sys.argv[1]
    print(f"Loading {pte_path} ({Path(pte_path).stat().st_size:,} bytes)")

    blobs = extract_ggml_blobs(pte_path)
    if not blobs:
        print("No ggml delegate blobs found!")
        sys.exit(1)

    for i, blob in enumerate(blobs):
        if len(blobs) > 1:
            print(f"\n--- Delegate {i} ---")
        dump_graph(blob)


if __name__ == "__main__":
    main()
