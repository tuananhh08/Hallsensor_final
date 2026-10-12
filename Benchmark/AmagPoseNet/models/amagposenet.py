"""AMagPoseNet – amagposenet.py

Full AMagPoseNet pipeline:
    RAW SENSOR DATA  [B, 1, 8, 8]
          ↓
       CaliNet          (domain correction)
          ↓
    CLEAN DATA       [B, 1, 8, 8]
          ↓
       PoseNet          (pose regression)
          ↓
    [B, 6]  =  [x, y, z, mx, my, mz]
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .calinet import CaliNet
from .posenet import PoseNet


class AMagPoseNet(nn.Module):
    """Full AMagPoseNet: CaliNet → PoseNet pipeline.

    Args:
        use_calinet: if False, CaliNet is bypassed (identity).
                     Useful for Phase 2 (pre-train PoseNet on clean synthetic data).

    Forward signature mirrors train_mvec.py's Model.forward for drop-in compatibility:
        model(x)                         → [B, 6]
        model(x, return_features=True)   → ([B, 6], {"input": …, "corrected": …, "residual": …})
    """

    def __init__(self, use_calinet: bool = True):
        super().__init__()
        self.use_calinet = use_calinet
        self.calinet = CaliNet()
        self.posenet = PoseNet()

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ):
        """
        Args:
            x: [B, 1, 8, 8] scaled raw (or synthetic) Hall voltages.
            return_features: if True, also return intermediate tensors dict.

        Returns:
            pred: [B, 6]
            features (optional): dict with keys "input", "corrected", "residual"
        """
        input_x = x

        if self.use_calinet:
            corrected, residual = self.calinet(x, return_residual=True)
        else:
            corrected = x
            residual  = torch.zeros_like(x)

        pred = self.posenet(corrected)

        if return_features:
            return pred, {
                "input":     input_x,
                "corrected": corrected,
                "residual":  residual,
            }
        return pred
