"""AMagPoseNet – calinet_loss.py

CaliNet denoising / calibration loss.

Paper Sec. III-B:
    "the L1-norm is utilised to evaluate the calibration performance of CaliNet"
    Loss = ||CaliNet(x_noisy) - x_clean||_1

Used in:
    Stage 1  : L_calib = L1(CaliNet(raw), clean)
    Stage 3  : same term added alongside pose loss
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CaliNetLoss(nn.Module):
    """L1 loss between CaliNet output and the clean reference sensor image.

    Paper: L1-norm chosen for robustness to saturated sensor measurements (outliers).

    Args:
        reduction: 'mean' (default) or 'sum' – passed to F.l1_loss.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction not in ("mean", "sum"):
            raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction!r}")
        self.reduction = reduction

    def forward(
        self,
        corrected: torch.Tensor,
        clean: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            corrected: [B, 1, 8, 8] – CaliNet(raw)
            clean:     [B, 1, 8, 8] – target clean sensor image

        Returns:
            Scalar L1 loss.
        """
        if corrected.shape != clean.shape:
            raise ValueError(
                f"corrected and clean must have the same shape; "
                f"got {tuple(corrected.shape)} vs {tuple(clean.shape)}"
            )
        return F.l1_loss(corrected.float(), clean.float(), reduction=self.reduction)
