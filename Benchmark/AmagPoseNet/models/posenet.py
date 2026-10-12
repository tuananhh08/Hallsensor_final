"""AMagPoseNet – posenet.py

PoseNet: estimates magnet pose from (CaliNet-corrected) sensor data.

Paper Fig. 3 + Sec. III-B:
    "PoseNet is similar to ResNet-18 … reduction of the kernel size in the first
     convolutional layer and the network depth and the splitting into four FC
     output branches."

Architecture (my adaptation, faithful to paper):
    Stem : 1×1 Conv(1→64), BN, ReLU          [B, 64, 8, 8]
    Stage1: 2 × BasicBlock(64, 64)            [B, 64, 8, 8]
    Stage2: BasicBlock(64,128,stride=2)
          + BasicBlock(128,128)               [B,128, 4, 4]
    Stage3: BasicBlock(128,256,stride=2)
          + BasicBlock(256,256)               [B,256, 2, 2]
    AvgPool 2×2                               [B,256, 1, 1]
    Flatten                                   [B,256]
    head_xyz  : Linear(256, 3)   – no activation
    head_mvec : Linear(256, 3) + F.normalize  – unit direction vector

Changes vs paper:
  - in_ch: 3 → 1   (scalar Hall voltage vs triaxial B-field)
  - spatial: 5×5 → 8×8 (two stride-2 stages give 2×2 before pool – same as paper)
  - output: 9-D Euler-variant → 6-D [xyz + unit mvec]
  - heads: 4 (paper) → 2 (position + orientation unit vector)
  No other backbone change.

No bias in Conv or FC (paper formula convention).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .residual_blocks import BasicBlock


class PoseNet(nn.Module):
    """Pose estimation backbone.

    Input : [B, 1, 8, 8]  – CaliNet-corrected (or raw synthetic) sensor data
    Output: [B, 6]         – [x, y, z, mx, my, mz]
              x,y,z  : StandardScaler-normalised position
              mx,my,mz: unit magnetic-moment direction vector (||m||=1)
    """

    def __init__(self):
        super().__init__()

        # ── Stem: 1×1 Conv replaces ResNet's 7×7 (paper Fig. 3) ──────────
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # ── Stage 1: 2 residual blocks @ 64 channels, no downsample ──────
        # Spatial: 8×8 → 8×8
        self.stage1 = nn.Sequential(
            BasicBlock(64, 64),
            BasicBlock(64, 64),
        )

        # ── Stage 2: transition 64→128, stride-2 downsample ──────────────
        # Spatial: 8×8 → 4×4
        self.stage2 = nn.Sequential(
            BasicBlock(64,  128, stride=2),
            BasicBlock(128, 128),
        )

        # ── Stage 3: transition 128→256, stride-2 downsample ─────────────
        # Spatial: 4×4 → 2×2
        self.stage3 = nn.Sequential(
            BasicBlock(128, 256, stride=2),
            BasicBlock(256, 256),
        )

        # ── 2×2 Average pool → [B, 256, 1, 1] ────────────────────────────
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        # After pooling: 2×2 → 1×1, feature vector = 256-D

        # ── FC heads (paper: 4 branches; adaptation: 2 branches) ─────────
        # head_xyz  : [256] → [3]   (position, no activation)
        # head_mvec : [256] → [3]   (raw orientation, then L2-normalised)
        # No bias in FC (paper formula convention).
        self.head_xyz  = nn.Linear(256, 3,  bias=False)
        self.head_mvec = nn.Linear(256, 3,  bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 1, 8, 8]

        Returns:
            [B, 6] = [x, y, z, mx, my, mz]
        """
        out = self.stem(x)      # [B,  64, 8, 8]
        out = self.stage1(out)  # [B,  64, 8, 8]
        out = self.stage2(out)  # [B, 128, 4, 4]
        out = self.stage3(out)  # [B, 256, 2, 2]
        out = self.avgpool(out) # [B, 256, 1, 1]
        feat = out.flatten(1)   # [B, 256]

        xyz   = self.head_xyz(feat)                          # [B, 3]
        m_raw = self.head_mvec(feat)                         # [B, 3]
        m_hat = F.normalize(m_raw, dim=-1, eps=1e-6)         # [B, 3] unit vector

        return torch.cat([xyz, m_hat], dim=1)                # [B, 6]
