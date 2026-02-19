import torch
import torch.nn as nn
from torch.export import export

from executorch_ggml.passes.bn_folding_pass import BatchNormFoldingPass, find_conv_bn_patterns


class ConvBn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(8)

    def forward(self, x):
        return self.bn(self.conv(x))


def test_detect_and_fold_pattern():
    m = ConvBn().eval()
    x = torch.randn(1, 3, 16, 16)
    ep = export(m, (x,))

    pats = find_conv_bn_patterns(ep)
    assert len(pats) == 1

    folded = BatchNormFoldingPass().run(ep)
    assert len(folded) == 1

    # sanity: folded bias exists and shapes match
    params = next(iter(folded.values()))
    assert params.weight.shape[0] == 8
    assert params.bias.shape == (8,)
