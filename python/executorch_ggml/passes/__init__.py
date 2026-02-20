"""Graph/pipeline passes for executorch-ggml."""

from .broadcast_pass import BroadcastCanonicalizationPass

__all__ = [
    "BroadcastCanonicalizationPass",
]
