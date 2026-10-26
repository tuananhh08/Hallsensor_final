"""ModNet: maps measured Hall fields into the analytical-model domain."""
from __future__ import annotations

import torch
from torch import nn


class ModNet(nn.Module):
    """Residual 1-D calibration network; input/output are ``[B, C, S]``.

    The paper specifies ModNet's domain-alignment role but not its layer widths.
    This compact residual implementation preserves every sensor node and works
    for the repository's 64 scalar sensors (C=1) and triaxial arrays (C=3).
    """
    def __init__(self, channels: int = 1, width: int = 64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(channels, width, 3, padding=1, bias=False), nn.BatchNorm1d(width), nn.GELU(),
            nn.Conv1d(width, width, 3, padding=1, bias=False), nn.BatchNorm1d(width), nn.GELU(),
            nn.Conv1d(width, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial_shape = None
        if x.ndim == 4:  # cache used by the existing benchmark: [B,C,8,8]
            spatial_shape = x.shape
            x = x.flatten(2)
        if x.ndim != 3:
            raise ValueError(f"Expected [B,C,S] or [B,C,H,W], got {tuple(x.shape)}")
        corrected = x + self.body(x)
        # Keep the caller's layout.  This is required for ModNet pretraining,
        # where the cached clean target remains [B, C, 8, 8], and for the
        # Stage-3 alignment loss.  LocNet already flattens 4-D fields itself.
        return corrected.reshape(spatial_shape) if spatial_shape is not None else corrected
