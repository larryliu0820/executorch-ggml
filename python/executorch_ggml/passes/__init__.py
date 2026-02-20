"""Graph/pipeline passes for executorch-ggml."""

from .bn_folding_pass import BatchNormFoldingPass
from .bn_folding_rewrite_pass import BatchNormFoldingRewritePass
from .broadcast_pass import BroadcastCanonicalizationPass

__all__ = [
    "BatchNormFoldingPass",
    "BatchNormFoldingRewritePass",
    "BroadcastCanonicalizationPass",
]
