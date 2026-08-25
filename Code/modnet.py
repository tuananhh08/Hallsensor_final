from __future__ import annotations

import torch
from torch import nn


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int, groups: int = 4):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode="replicate", bias=False),
            nn.GroupNorm(min(groups, channels), channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode="replicate", bias=False),
            nn.GroupNorm(min(groups, channels), channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class ModNet(nn.Module):

    def __init__(self, channels: int = 16, num_blocks: int = 2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1, padding_mode="replicate", bias=False),
            nn.GroupNorm(4, channels),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[_ResidualBlock(channels) for _ in range(num_blocks)])
        self.to_residual = nn.Conv2d(channels, 1, 3, padding=1, padding_mode="replicate")
        nn.init.zeros_(self.to_residual.weight)
        nn.init.zeros_(self.to_residual.bias)

    def forward(self, x: torch.Tensor, return_residual: bool = False):
        if x.ndim != 4 or x.shape[1:] != (1, 8, 8):
            raise ValueError(f"ModNet expects (V, 1, 8, 8), got {tuple(x.shape)}")
        residual = self.to_residual(self.blocks(self.stem(x)))
        corrected = x + residual
        return (corrected, residual) if return_residual else corrected
