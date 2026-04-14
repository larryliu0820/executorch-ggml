"""Operator handlers for the ggml backend.

Importing this package triggers registration of all operator handlers
via the _registry module.
"""

# Re-export infrastructure for use by operator modules
from executorch_ggml.ops._helpers import (
    _concrete_int,
    _resolve_shape,
    _type_elem_size,
    _pytorch_shape_to_ggml_ne,
    _torch_dtype_to_ir_type,
)

from executorch_ggml.ops._sym_expr import (
    SYM_OP_PUSH_SYM,
    SYM_OP_PUSH_CONST,
    SYM_OP_ADD,
    SYM_OP_SUB,
    SYM_OP_MUL,
    SYM_OP_FLOORDIV,
    SYM_OP_MOD,
    SYM_OP_NEG,
    _sympy_to_bytecode,
    _eval_bytecode,
    _get_sym_dim_info,
    _sym_dim_info_ggml,
)

from executorch_ggml.ops._context import PreprocessContext
from executorch_ggml.ops._registry import register_op, dispatch_op

# Import all operator modules to trigger registration.
# Each module uses @register_op to add its handlers to _OP_HANDLERS.
from executorch_ggml.ops import (  # noqa: F401
    misc,
    activation,
    arithmetic,
    linalg,
    shape,
    normalization,
    comparison,
    tensor_creation,
    indexing,
    convolution,
    special,
    moe_ops,
)

__all__ = [
    "PreprocessContext",
    "_concrete_int",
    "_resolve_shape",
    "_type_elem_size",
    "_pytorch_shape_to_ggml_ne",
    "_torch_dtype_to_ir_type",
    "_sympy_to_bytecode",
    "_eval_bytecode",
    "_get_sym_dim_info",
    "_sym_dim_info_ggml",
    "register_op",
    "dispatch_op",
]
