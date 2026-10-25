from __future__ import annotations
import torch
from torch import nn
from .modnet import ModNet
from .locnet import LocNet

class CRLocNet(nn.Module):
    """Paper pipeline: raw sensor field → ModNet → LocNet(corrected, EPM)."""
    def __init__(self, sensor_channels: int = 1, width: int = 64):
        super().__init__()
        self.modnet = ModNet(sensor_channels, width)
        self.locnet = LocNet(sensor_channels, width)
    def forward(self, raw: torch.Tensor, epm: torch.Tensor | None = None, return_features: bool = False):
        # A zero EPM is a valid compatibility mode for single-source datasets.
        # For the paper's multi-source setting, always provide theoretical EPM.
        if epm is None:
            epm = torch.zeros_like(raw)
        corrected = self.modnet(raw)
        pred = self.locnet(corrected, epm)
        return (pred, {"corrected": corrected}) if return_features else pred
