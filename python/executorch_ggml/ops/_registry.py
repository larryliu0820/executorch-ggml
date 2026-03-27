"""Operator dispatch registry for ggml backend preprocess."""

from typing import Callable, List, Tuple

# Global registry: list of (match_string, handler_function) tuples.
# Order matters — first match wins, matching the original if/elif semantics.
_OP_HANDLERS: List[Tuple[str, Callable]] = []


def register_op(*match_strs: str):
    """Decorator: register a handler for target strings containing match_str.

    Usage:
        @register_op("aten.add.Tensor", "aten.add.Scalar")
        def handle_add(ctx, node, target_str):
            ...
    """
    def wrapper(fn):
        for match_str in match_strs:
            _OP_HANDLERS.append((match_str, fn))
        return fn
    return wrapper


def dispatch_op(ctx, node, target_str: str) -> bool:
    """Try all registered handlers. Returns True if one matched."""
    for pattern, handler in _OP_HANDLERS:
        if pattern in target_str:
            handler(ctx, node, target_str)
            return True
    return False
