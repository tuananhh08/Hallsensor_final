"""AMagPoseNet – pose_loss.py

Pose regression loss, adapted from AMagPoseNet paper Eq. (11)–(12):
    L(M) = Lp(M) + β * Lo(M)

Paper uses L1-norm ("more robust on datasets that contain saturated sensor data").

=== Position loss ===
    Lp = L1( pred_xyz, target_xyz )
    pred_xyz / target_xyz are StandardScaler-normalised [x, y, z].
    Directly compatible with paper's L1 position term.

=== Orientation loss ===
    Paper: L1 on 8-D Euler-variant representation
           [sinα,cosα, sinβ,cosβ, sin2γ,cos2γ]
    My data: (mx,my,mz) is a unit magnetic-moment direction vector.
             Euler-variant cannot be applied without an unsupported convention.
    Adaptation: L1 on the normalised 3-D unit vector.
           Lo = L1( F.normalize(pred_mvec), F.normalize(target_mvec) )
    Both representations measure orientation deviation in a Euclidean embedding
    of the unit sphere – the adaptation is principled and minimally invasive.
    (Explicit deviation documented in implementation plan.)

=== Combined loss (finetune phase) ===
    L_total = lambda_calib * L_calib + lambda_pos * Lp + beta * Lo
    where lambda_calib, lambda_pos, and beta are all configurable.

=== Beta ===
    Paper does not report a specific numerical β value.
    Default: beta=1.0 (equal weighting). User-configurable.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .calinet_loss import CaliNetLoss


class AMagPoseLoss(nn.Module):
    """Pose regression loss (Stage 2 – PoseNet pre-training).

    L = lambda_pos * L1(pred_xyz, target_xyz)
      + beta       * L1(norm(pred_mvec), norm(target_mvec))

    Args:
        beta:       Weight for orientation loss. Default 1.0 (paper value not reported).
        lambda_pos: Weight for position loss. Default 1.0.
    """

    def __init__(self, beta: float = 1.0, lambda_pos: float = 1.0):
        super().__init__()
        if beta < 0:
            raise ValueError(f"beta must be >= 0, got {beta}")
        if lambda_pos < 0:
            raise ValueError(f"lambda_pos must be >= 0, got {lambda_pos}")
        self.beta       = beta
        self.lambda_pos = lambda_pos

    def forward(
        self,
        pred:   torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            pred:   [B, 6] – model output [x,y,z, mx,my,mz]
            target: [B, 6] – ground-truth [x,y,z, mx,my,mz]

        Returns:
            (total_loss, loss_pos, loss_ori)  – all scalars
        """
        pred   = pred.float()
        target = target.float()

        # ── Position: L1 on normalised xyz ────────────────────────────────
        loss_pos = F.l1_loss(pred[:, :3], target[:, :3])

        # ── Orientation: L1 on unit direction vectors ─────────────────────
        m_pred = F.normalize(pred[:, 3:],   dim=1, eps=1e-6)
        m_gt   = F.normalize(target[:, 3:], dim=1, eps=1e-6)
        loss_ori = F.l1_loss(m_pred, m_gt)

        total = self.lambda_pos * loss_pos + self.beta * loss_ori
        return total, loss_pos, loss_ori


class CalibPoseLoss(nn.Module):
    """Combined calibration + pose loss (Stage 1 and Stage 3).

    Stage 1 (calibnet phase): only calib term is active (pose weight = 0).
    Stage 3 (finetune phase): both terms active.

    L_total = lambda_calib * CaliNetLoss(corrected, clean)
            + AMagPoseLoss(pred, target)

    Args:
        beta:           β for orientation in AMagPoseLoss.
        lambda_pos:     Weight for position loss.
        lambda_calib:   Weight for CaliNet denoising loss.
    """

    def __init__(
        self,
        beta:         float = 1.0,
        lambda_pos:   float = 1.0,
        lambda_calib: float = 0.1,
        calib_reduction: str = "mean",
    ):
        super().__init__()
        if lambda_calib < 0:
            raise ValueError(f"lambda_calib must be >= 0, got {lambda_calib}")
        self.lambda_calib = lambda_calib
        self.pose_loss  = AMagPoseLoss(beta=beta, lambda_pos=lambda_pos)
        self.calib_loss = CaliNetLoss(reduction=calib_reduction)

    def calibration_term(
        self,
        corrected: torch.Tensor,
        clean:     torch.Tensor,
    ) -> torch.Tensor:
        """Standalone calibration loss (used in Phase 1)."""
        return self.calib_loss(corrected, clean)

    def forward(
        self,
        pred:      torch.Tensor,
        target:    torch.Tensor,
        corrected: torch.Tensor,
        clean:     torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            pred:      [B, 6]       – AMagPoseNet output
            target:    [B, 6]       – ground-truth pose
            corrected: [B, 1, 8, 8] – CaliNet(raw)
            clean:     [B, 1, 8, 8] – target clean sensor image

        Returns:
            (total, loss_calib, loss_pos, loss_ori)
        """
        pose_total, loss_pos, loss_ori = self.pose_loss(pred, target)
        loss_calib = self.calib_loss(corrected, clean)
        total = pose_total + self.lambda_calib * loss_calib
        return total, loss_calib, loss_pos, loss_ori
