"""Inverted Residual Attention Block (Table I of MobilePoseNet)."""
from __future__ import annotations
import torch
from torch import nn


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, channels), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x).view(x.shape[0], x.shape[1], 1, 1)


class IRAB(nn.Module):
    """MobileNetV2 block with the paper's SE channel-attention after projection."""
    def __init__(self, in_channels: int, out_channels: int, expansion: int, stride: int):
        super().__init__()
        if stride not in (1, 2): raise ValueError("IRAB stride must be 1 or 2")
        expanded = in_channels * expansion
        self.expand = nn.Sequential(nn.Conv2d(in_channels, expanded, 1, bias=False), nn.BatchNorm2d(expanded), nn.ReLU6(inplace=True))
        self.depthwise = nn.Sequential(nn.Conv2d(expanded, expanded, 3, stride=stride, padding=1, groups=expanded, bias=False), nn.BatchNorm2d(expanded), nn.ReLU6(inplace=True))
        self.project = nn.Sequential(nn.Conv2d(expanded, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels))
        self.attention = SEBlock(out_channels)
        self.use_skip = stride == 1 and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.project(self.depthwise(self.expand(x)))
        if self.use_skip: y = y + x
        return self.attention(y)
