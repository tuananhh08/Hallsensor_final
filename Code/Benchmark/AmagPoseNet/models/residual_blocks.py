"""AMagPoseNet – residual_blocks.py

BasicBlock matching Fig. 3 of the paper:
    Conv3×3 – BN – ReLU → Conv3×3 – BN  (+  shortcut)

No bias terms (paper Sec. III-B, FLOPs / NPs formula uses no-bias convention).
The projection shortcut (1×1 Conv) is used only when channels or stride change,
exactly as in ResNet-18.
"""
from __future__ import annotations

import torch.nn as nn


class BasicBlock(nn.Module):
    """Single residual block used inside PoseNet.

    Architecture (paper Fig. 3):
        x → Conv3×3(in,out,stride) – BN – ReLU
          → Conv3×3(out,out,1)    – BN
          (+shortcut) → ReLU → output

    Shortcut is a 1×1 Conv (+ BN) when in_ch != out_ch or stride != 1.
    No bias in any Conv layer.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_ch, out_ch, kernel_size=3, stride=stride,
            padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_ch, out_ch, kernel_size=3, stride=1,
            padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_ch)

        # 1×1 projection shortcut (paper Fig. 3 "transition block")
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out
