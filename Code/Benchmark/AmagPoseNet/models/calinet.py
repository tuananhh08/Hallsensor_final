"""AMagPoseNet – calinet.py

CaliNet: maps raw sensor data to clean/model-domain sensor data.
Paper Fig. 3 (CaliNet section):
    Conv3×3, 32, BN, ReLU
    Conv3×3, 64, BN, ReLU
    Conv3×3, 32, BN, ReLU
    Conv3×3,  3, BN, ReLU    ← paper: 3 output channels (triaxial)

Adaptation for 1-channel 8×8 Hall sensor array:
    Conv3×3, 32, BN, ReLU    ← 1 input  channel  (scalar Hall voltage)
    Conv3×3, 64, BN, ReLU
    Conv3×3, 32, BN, ReLU
    Conv3×3,  1, BN, ReLU    ← 1 output channel (same as input)

Changes vs paper:
  - in_ch: 3 → 1
  - out_ch: 3 → 1
  - spatial: 5×5 → 8×8 (padding=1 preserves spatial size throughout)
  All other topology is identical.

No bias in any Conv layer (paper Sec. III-B formula convention).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CaliNet(nn.Module):
    """Calibration / domain-gap network.

    Input : [B, 1, 8, 8]  – scaled raw Hall voltages
    Output: [B, 1, 8, 8]  – corrected (model-domain) Hall voltages

    Paper: plain 4-layer sequential Conv3×3 encoder, no residual connections,
           no downsampling (all spatial dims preserved).
    """

    def __init__(self):
        super().__init__()
        # Layer 1: 1 → 32
        self.conv1 = nn.Conv2d(1,  32, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(32)

        # Layer 2: 32 → 64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(64)

        # Layer 3: 64 → 32
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False)
        self.bn3   = nn.BatchNorm2d(32)

        # Layer 4: 32 → 1 (paper: 32 → 3)
        self.conv4 = nn.Conv2d(32,  1, kernel_size=3, padding=1, bias=False)
        self.bn4   = nn.BatchNorm2d(1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, return_residual: bool = False):
        """
        Args:
            x: [B, 1, 8, 8] scaled raw Hall voltages.
            return_residual: if True, return (corrected, residual).

        Returns:
            corrected: [B, 1, 8, 8]
            residual (optional): [B, 1, 8, 8]  – corrected - x
        """
        if x.ndim != 4 or x.shape[1:] != (1, 8, 8):
            raise ValueError(f"CaliNet expects (B, 1, 8, 8), got {tuple(x.shape)}")

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)))
        out = self.relu(self.bn4(self.conv4(out)))   # corrected sensor image

        if return_residual:
            return out, out - x
        return out
