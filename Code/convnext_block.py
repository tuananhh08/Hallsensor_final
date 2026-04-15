import torch
import torch.nn as nn


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep = 1 - drop_prob
    mask = torch.empty(x.shape[0], 1, 1, 1, device=x.device).bernoulli_(keep) / keep
    return x * mask


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class ConvNeXtBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, drop_path_rate: float = 0.0):
        super().__init__()

        self.dw   = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.bn1  = nn.BatchNorm2d(in_ch)
        self.act1 = nn.LeakyReLU(0.01, inplace=True)

        self.pw   = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2  = nn.BatchNorm2d(out_ch)
        self.act2 = nn.LeakyReLU(0.01, inplace=True)

        self.drop_path = DropPath(drop_path_rate)

        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        out = self.act1(self.bn1(self.dw(x)))
        out = self.act2(self.bn2(self.pw(out)))
        return self.drop_path(out) + residual