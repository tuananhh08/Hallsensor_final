"""MobilePoseNet pose objective from Sec. III-C.

The paper uses Euclidean position and direction-vector errors, with β
balancing their disparate scales.  Direction vectors are normalised before
comparison; reported angular error is derived separately during evaluation.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn


class MobilePoseLoss(nn.Module):
    def __init__(self, beta: float = 1.0) -> None:
        super().__init__()
        if beta < 0: raise ValueError("beta must be non-negative")
        self.beta = beta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        position = torch.linalg.vector_norm(pred[:, :3] - target[:, :3], dim=1).mean()
        direction = torch.linalg.vector_norm(F.normalize(pred[:, 3:], dim=1, eps=1e-6) - F.normalize(target[:, 3:], dim=1, eps=1e-6), dim=1).mean()
        return self.beta * position + direction, position, direction
