"""Module swap: replace CausalConv1d F.pad with built-in Conv1d padding.

The original CausalConv1d does F.pad(x, (k-1, 0)) followed by Conv1d(padding=0).
This creates a separate PAD op in the ggml graph. When the first conv's input is the
mel spectrogram (a graph input on CPU), the ggml scheduler places PAD on CPU, creating
a graph split that prevents CUDA graph capture and adds transfer overhead.

Fix: use Conv1d with symmetric padding=(k-1) and slice off the right padding.
The conv's IM2COL handles padding internally (CUDA-native), and the slice is a
zero-cost view op in ggml. Eliminates 2 PAD ops and 1 graph split.

Usage (before export):
    from executorch_ggml.modules.voxtral_conv import swap_causal_conv1d
    swap_causal_conv1d(model)
"""

import torch.nn as nn


class SymPadConv1d(nn.Module):
    """Conv1d with symmetric padding + right truncation (= causal padding).

    Mathematically equivalent to CausalConv1d but avoids the explicit PAD op.
    Does NOT include GELU — the encoder's forward applies GELU after calling this.
    """

    def __init__(self, conv: nn.Conv1d, pad_length: int, stride: int):
        super().__init__()
        self.conv = conv
        self.pad_length = pad_length
        self.stride = stride
        self.conv.padding = (pad_length,)

    def forward(self, x):
        y = self.conv(x)
        # Symmetric padding adds pad_length on the right too. Trim those elements.
        trim = (self.pad_length + self.stride - 1) // self.stride
        return y[:, :, :-trim]


def swap_causal_conv1d(model: nn.Module) -> int:
    """Replace CausalConv1d modules with SymPadConv1d.

    Returns the number of modules swapped.
    """
    from executorch.examples.models.voxtral_realtime.model import CausalConv1d

    encoder = model.encoder if hasattr(model, "encoder") else model
    if not hasattr(encoder, "conv_layers"):
        return 0

    count = 0
    for i, layer in enumerate(encoder.conv_layers):
        if isinstance(layer, CausalConv1d):
            stride = layer.conv.stride[0]
            new_layer = SymPadConv1d(layer.conv, layer.pad_length, stride)
            encoder.conv_layers[i] = new_layer
            count += 1

    return count
