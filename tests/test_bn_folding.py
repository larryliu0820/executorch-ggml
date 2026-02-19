import torch
import torch.nn as nn

from executorch_ggml.bn_folding import fold_conv_bn_weights


def test_bn_folding_matches_conv_bn():
    torch.manual_seed(0)
    conv = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False).eval()
    bn = nn.BatchNorm2d(16).eval()

    # Randomize BN running stats and affine params to exercise formula.
    bn.running_mean = torch.randn_like(bn.running_mean)
    bn.running_var = torch.rand_like(bn.running_var) + 0.5
    bn.weight.data = torch.randn_like(bn.weight)
    bn.bias.data = torch.randn_like(bn.bias)

    x = torch.randn(1, 8, 32, 32)

    with torch.no_grad():
        y_ref = bn(conv(x))

    w_fold, b_fold = fold_conv_bn_weights(
        conv.weight,
        None,
        bn.weight,
        bn.bias,
        bn.running_mean,
        bn.running_var,
        bn.eps,
    )

    conv_fold = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=True).eval()
    with torch.no_grad():
        conv_fold.weight.copy_(w_fold)
        conv_fold.bias.copy_(b_fold)
        y_fold = conv_fold(x)

    assert torch.allclose(y_ref, y_fold, rtol=1e-4, atol=1e-4)
