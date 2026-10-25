"""LocNet from CRLocNet: MSC + channel attention + 1-D residual backbone."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class TeLU(nn.Module):
    """TeLU activation used by the paper's multi-scale convolution block."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(torch.exp(x))


class MultiScaleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        branch = max(1, out_ch // 3)
        self.branches = nn.ModuleList([nn.Conv1d(in_ch, branch, k, padding=k // 2, bias=False)
                                       for k in (3, 5, 7)])
        self.project = nn.Sequential(nn.Conv1d(branch * 3, out_ch, 1, bias=False),
                                     nn.BatchNorm1d(out_ch), TeLU())
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(torch.cat([conv(x) for conv in self.branches], dim=1))


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.net = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Conv1d(channels, hidden, 1),
                                 nn.ReLU(inplace=True), nn.Conv1d(hidden, channels, 1), nn.Sigmoid())
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, attention: bool = False):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(channels, channels, 3, padding=1, bias=False),
                                  nn.BatchNorm1d(channels), TeLU(),
                                  nn.Conv1d(channels, channels, 3, padding=1, bias=False), nn.BatchNorm1d(channels))
        self.attn = ChannelAttention(channels) if attention else nn.Identity()
        self.act = TeLU()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.attn(self.conv(x)))


class LocNet(nn.Module):
    """Estimate ``[x,y,z,mx,my,mz]`` from corrected + theoretical EPM field."""
    def __init__(self, sensor_channels: int = 1, width: int = 64):
        super().__init__()
        self.msc = MultiScaleConv(sensor_channels * 2, width)
        self.blocks = nn.Sequential(ResidualBlock(width), ResidualBlock(width, attention=True),
                                    ResidualBlock(width, attention=True))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Linear(width, width), TeLU())
        self.xyz = nn.Linear(width, 3)
        self.direction = nn.Linear(width, 3)
    def forward(self, corrected: torch.Tensor, epm: torch.Tensor) -> torch.Tensor:
        if corrected.ndim == 4:
            corrected = corrected.flatten(2)
        if epm.ndim == 4:
            epm = epm.flatten(2)
        if corrected.shape != epm.shape:
            raise ValueError(f"Corrected and EPM fields must have equal shape; got {tuple(corrected.shape)} and {tuple(epm.shape)}")
        x = self.blocks(self.msc(torch.cat((corrected, epm), dim=1)))
        x = self.head(self.pool(x).squeeze(-1))
        return torch.cat((self.xyz(x), F.normalize(self.direction(x), dim=1, eps=1e-6)), dim=1)
