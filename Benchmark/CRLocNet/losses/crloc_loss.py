"""Losses from CRLocNet Sec. 3, including differentiable physical consistency."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F


class ModNetLoss(nn.Module):
    """Paper's Euclidean domain-alignment loss, averaged over samples."""
    def forward(self, corrected: torch.Tensor, model_field: torch.Tensor) -> torch.Tensor:
        return torch.linalg.vector_norm((corrected - model_field).flatten(1), dim=1).mean()


class DipoleForward(nn.Module):
    """Differentiable magnetic dipole forward model for the physics loss.

    ``sensor_positions`` are metres in the model/world frame.  The default
    creates an 8×8 planar array; pass measured coordinates for a real rig.
    For scalar Hall arrays, only ``measurement_axis`` is returned.
    """
    def __init__(self, sensor_positions: torch.Tensor, sensor_channels: int = 1,
                 moment_am2: float = 1.0, measurement_axis: int = 2):
        super().__init__()
        self.register_buffer("sensor_positions", sensor_positions.float())
        self.sensor_channels, self.moment_am2, self.measurement_axis = sensor_channels, moment_am2, measurement_axis
    @classmethod
    def planar_8x8(cls, pitch_m: float = 0.01, **kwargs):
        xy = (torch.arange(8, dtype=torch.float32) - 3.5) * pitch_m
        yy, xx = torch.meshgrid(xy, xy, indexing="ij")
        return cls(torch.stack((xx.flatten(), yy.flatten(), torch.zeros(64)), dim=1), **kwargs)
    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        pos, direction = pose[:, :3], F.normalize(pose[:, 3:6], dim=1, eps=1e-6)
        r = self.sensor_positions[None] - pos[:, None]                   # [B,S,3]
        r2 = r.square().sum(-1, keepdim=True).clamp_min(1e-12)
        dot = (r * direction[:, None]).sum(-1, keepdim=True)
        coeff = 1e-7 * self.moment_am2
        field = coeff * (3.0 * r * dot / r2.pow(2.5) - direction[:, None] / r2.pow(1.5))
        if self.sensor_channels == 1:
            return field[..., self.measurement_axis].unsqueeze(1)
        return field.permute(0, 2, 1)


class ScaledDipoleForward(nn.Module):
    """Physical forward model expressed in the benchmark's scaled field domain.

    The pose network sees standardised XYZ and min-max-scaled Hall fields. This
    wrapper reverses the XYZ scaling, evaluates :class:`DipoleForward`, then
    applies the *same* per-sensor field scaling.  It makes L_ph differentiable
    with respect to the network output, unlike comparing two input fields.
    Input field units and ``moment_am2`` must be mutually consistent (normally
    tesla and A m²).
    """
    def __init__(self, forward: DipoleForward, xyz_mean, xyz_scale, field_min, field_max):
        super().__init__()
        self.forward = forward
        self.register_buffer("xyz_mean", torch.as_tensor(xyz_mean, dtype=torch.float32))
        self.register_buffer("xyz_scale", torch.as_tensor(xyz_scale, dtype=torch.float32))
        self.register_buffer("field_min", torch.as_tensor(field_min, dtype=torch.float32).reshape(1, -1, 1))
        self.register_buffer("field_range", (torch.as_tensor(field_max, dtype=torch.float32)-torch.as_tensor(field_min, dtype=torch.float32)).clamp_min(1e-12).reshape(1, -1, 1))
    def forward(self, scaled_pose: torch.Tensor) -> torch.Tensor:
        pose = scaled_pose.clone()
        pose[:, :3] = pose[:, :3] * self.xyz_scale + self.xyz_mean
        physical = self.forward(pose)
        return (physical - self.field_min) / self.field_range


class CRLocLoss(nn.Module):
    """L = Lp + lambda_direction*Lo + lambda_physics*Lph (Sec. 3.1)."""
    def __init__(self, lambda_direction: float = .2, lambda_physics: float = 1e-4):
        super().__init__(); self.lambda_direction = lambda_direction; self.lambda_physics = lambda_physics
    def forward(self, pred: torch.Tensor, target: torch.Tensor, predicted_field: torch.Tensor | None = None,
                measured_field: torch.Tensor | None = None):
        # Lp: Euclidean positional distance; Lo: 1-cos(theta); Lph: MAE as Sec. 3.2.
        l_pos = torch.linalg.vector_norm(pred[:, :3] - target[:, :3], dim=1).mean()
        l_dir = (1.0 - (F.normalize(pred[:, 3:], dim=1, eps=1e-6) *
                        F.normalize(target[:, 3:], dim=1, eps=1e-6)).sum(1).clamp(-1, 1)).mean()
        l_phy = pred.new_zeros(())
        if predicted_field is not None and measured_field is not None:
            # Sec. 3.1/3.2: MAE between G(predicted pose) and measured IPM field.
            l_phy = F.l1_loss(predicted_field, measured_field)
        total = l_pos + self.lambda_direction * l_dir + self.lambda_physics * l_phy
        return total, l_pos, l_dir, l_phy
