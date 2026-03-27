#pragma once

#include <cstdint>
#include <cstring>
#include <unordered_map>

#include <flatbuffers/flatbuffers.h>

namespace executorch_ggml {

// ---------------------------------------------------------------------------
// Symbolic expression bytecode evaluator
// ---------------------------------------------------------------------------
// Opcodes match the Python SYM_OP_* constants in ggml_backend.py.

enum SymExprOp : uint8_t {
  SYM_PUSH_SYM   = 0x01,  // 1-byte operand: sym_id
  SYM_PUSH_CONST = 0x02,  // 4-byte operand: int32 LE
  SYM_ADD        = 0x10,
  SYM_SUB        = 0x11,
  SYM_MUL        = 0x12,
  SYM_FLOORDIV   = 0x13,
  SYM_MOD        = 0x14,
  SYM_NEG        = 0x15,
};

// Evaluate postfix bytecode with given symbol values.
// Returns the computed int64_t value, or 0 on error.
static int64_t eval_sym_expr(
    const uint8_t* code, size_t code_len,
    const std::unordered_map<int32_t, int64_t>& sym_values) {
  int64_t stack[16];
  int sp = 0;
  size_t pc = 0;
  while (pc < code_len) {
    uint8_t op = code[pc++];
    switch (op) {
      case SYM_PUSH_SYM: {
        if (pc >= code_len) return 0;
        uint8_t sid = code[pc++];
        auto it = sym_values.find(static_cast<int32_t>(sid));
        stack[sp++] = (it != sym_values.end()) ? it->second : 0;
        break;
      }
      case SYM_PUSH_CONST: {
        if (pc + 4 > code_len) return 0;
        int32_t val;
        memcpy(&val, code + pc, 4);
        pc += 4;
        stack[sp++] = static_cast<int64_t>(val);
        break;
      }
      case SYM_ADD: {
        if (sp < 2) return 0;
        int64_t b = stack[--sp], a = stack[--sp];
        stack[sp++] = a + b;
        break;
      }
      case SYM_SUB: {
        if (sp < 2) return 0;
        int64_t b = stack[--sp], a = stack[--sp];
        stack[sp++] = a - b;
        break;
      }
      case SYM_MUL: {
        if (sp < 2) return 0;
        int64_t b = stack[--sp], a = stack[--sp];
        stack[sp++] = a * b;
        break;
      }
      case SYM_FLOORDIV: {
        if (sp < 2) return 0;
        int64_t b = stack[--sp], a = stack[--sp];
        if (b == 0) return 0;
        // C++ integer division truncates; Python floor-divides toward -inf.
        // For positive divisors (our use case), they're equivalent when a>=0.
        // Handle general case correctly:
        int64_t q = a / b;
        int64_t r = a % b;
        if (r != 0 && ((r ^ b) < 0)) q--;
        stack[sp++] = q;
        break;
      }
      case SYM_MOD: {
        if (sp < 2) return 0;
        int64_t b = stack[--sp], a = stack[--sp];
        if (b == 0) return 0;
        int64_t r = a % b;
        if (r != 0 && ((r ^ b) < 0)) r += b;
        stack[sp++] = r;
        break;
      }
      case SYM_NEG: {
        if (sp < 1) return 0;
        stack[sp - 1] = -stack[sp - 1];
        break;
      }
      default:
        return 0;  // Unknown opcode
    }
    if (sp > 15) return 0;  // Stack overflow
  }
  return (sp == 1) ? stack[0] : 0;
}

// Unpack per-dim bytecode from the packed sym_dim_exprs vector.
// Layout: 4 entries, each prefixed by uint16 length.
// Returns true if bytecode was found for the given dim.
static bool get_dim_expr_bytecode(
    const ::flatbuffers::Vector<uint8_t>* exprs,
    size_t dim,
    const uint8_t*& out_code,
    size_t& out_len) {
  if (!exprs || dim >= 4) return false;
  const uint8_t* data = exprs->data();
  size_t total = exprs->size();
  size_t offset = 0;
  for (size_t d = 0; d < 4 && offset + 2 <= total; ++d) {
    uint16_t len;
    memcpy(&len, data + offset, 2);
    offset += 2;
    if (d == dim) {
      if (len == 0 || offset + len > total) return false;
      out_code = data + offset;
      out_len = len;
      return true;
    }
    offset += len;
  }
  return false;
}

} // namespace executorch_ggml
