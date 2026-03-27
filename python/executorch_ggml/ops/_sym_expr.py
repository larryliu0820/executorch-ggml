"""Symbolic expression bytecode compiler and evaluator.

Opcodes for postfix bytecode encoding of sympy expressions.
Used to serialize derived dynamic shape expressions (e.g. ((s0-1)//8)+1)
from strided convolutions into the FlatBuffer IR.

The C++ runtime evaluates these with a simple stack machine.
"""

import struct as _struct

import torch

SYM_OP_PUSH_SYM  = 0x01  # 1-byte operand: sym_id
SYM_OP_PUSH_CONST = 0x02  # 4-byte operand: int32 LE
SYM_OP_ADD       = 0x10
SYM_OP_SUB       = 0x11
SYM_OP_MUL       = 0x12
SYM_OP_FLOORDIV  = 0x13
SYM_OP_MOD       = 0x14
SYM_OP_NEG       = 0x15


def _sympy_to_bytecode(expr, sym_id_map: dict) -> bytes:
    """Compile a sympy expression to postfix bytecode.

    Handles: Symbol, Integer, One, NegativeOne, Add, Mul,
    FloorDiv (torch.utils._sympy.functions.FloorDiv), Mod.
    """
    import sympy

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

        if isinstance(node, sympy.Add):
            args = list(node.args)
            _emit(args[0])
            for a in args[1:]:
                _emit(a)
                buf.append(SYM_OP_ADD)
            return

        if isinstance(node, sympy.Mul):
            args = list(node.args)
            if len(args) == 2 and args[0] == sympy.S.NegativeOne:
                _emit(args[1])
                buf.append(SYM_OP_NEG)
                return
            _emit(args[0])
            for a in args[1:]:
                _emit(a)
                buf.append(SYM_OP_MUL)
            return

        if FloorDiv is not None and isinstance(node, FloorDiv):
            _emit(node.args[0])
            _emit(node.args[1])
            buf.append(SYM_OP_FLOORDIV)
            return

        if isinstance(node, sympy.floor):
            inner = node.args[0]
            if isinstance(inner, sympy.Mul) and len(inner.args) == 2:
                a, b = inner.args
                if isinstance(b, sympy.Pow) and b.args[1] == -1:
                    _emit(a)
                    _emit(b.args[0])
                    buf.append(SYM_OP_FLOORDIV)
                    return
            _emit(inner)
            return

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
    free = expr.free_symbols
    if len(free) >= 1:
        for sym in free:
            name = str(sym)
            if name not in sym_id_map:
                sym_id_map[name] = len(sym_id_map)
        bytecode = _sympy_to_bytecode(expr, sym_id_map)
        return (-2, bytecode)
    return (-1, None)


def _sym_dim_info_ggml(fake_val, sym_id_map: dict):
    """Compute ggml-order sym_dim_ids and packed sym_dim_exprs.

    Returns (sym_dim_ids_or_None, sym_dim_exprs_or_None).
    """
    if fake_val is None or not hasattr(fake_val, "shape"):
        return (None, None)

    pt_info = [_get_sym_dim_info(s, sym_id_map) for s in fake_val.shape]
    if not pt_info or all(sid == -1 for sid, _ in pt_info):
        return (None, None)

    ggml_info = list(reversed(pt_info))
    while len(ggml_info) < 4:
        ggml_info.append((-1, None))
    ggml_info = ggml_info[:4]

    ggml_sym = [sid for sid, _ in ggml_info]
    has_exprs = any(bc is not None for _, bc in ggml_info)

    if not has_exprs:
        if all(sid == -1 for sid in ggml_sym):
            return (None, None)
        return (ggml_sym, None)

    exprs_buf = bytearray()
    for _, bc in ggml_info:
        if bc is not None:
            exprs_buf.extend(_struct.pack("<H", len(bc)))
            exprs_buf.extend(bc)
        else:
            exprs_buf.extend(_struct.pack("<H", 0))

    return (ggml_sym, bytes(exprs_buf))
