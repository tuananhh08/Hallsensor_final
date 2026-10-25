"""MobilePoseNet backbone adapted to this benchmark's scalar 8×8 Hall input."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn
from .IRAB_block import IRAB


def _conv_bn_relu(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))


class MobilePoseNet(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.stem = _conv_bn_relu(1, 32)
        self.features = nn.Sequential(
            IRAB(32, 16, expansion=1, stride=1),
            IRAB(16, 32, expansion=6, stride=1),
            IRAB(32, 64, expansion=6, stride=2),
            IRAB(64, 128, expansion=6, stride=1),
            IRAB(128, 256, expansion=1, stride=1),
            _conv_bn_relu(256, 512), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.head_xyz, self.head_mvec = nn.Linear(512, 3), nn.Linear(512, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1:] != (1, 8, 8):
            raise ValueError(f"MobilePoseNet expects [B,1,8,8], got {tuple(x.shape)}")
        feature = self.features(self.stem(x))
        return torch.cat((self.head_xyz(feature), F.normalize(self.head_mvec(feature), dim=1, eps=1e-6)), dim=1)
