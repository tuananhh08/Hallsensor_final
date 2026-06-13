import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from loss_calib import load_calib_csv


class HuberPoseLossMVec(nn.Module):
    MU_0_4PI = 1e-7   # μ₀ / (4π)  [T·m/A]

    def __init__(self,
                 lambda_ori:     float = 1.0,
                 delta_xyz:      float = 0.061,
                 lambda_pos:     float = 1.0,
                 lambda_physics: float = 1e-4,
                 calib_csv:      str   = None,
                 volt_scaler            = None,
                 label_scaler           = None,
                 m0:             float = 1.0):

        super().__init__()

        self.lambda_ori     = lambda_ori
        self.delta_xyz      = delta_xyz
        self.lambda_pos     = lambda_pos
        self.lambda_physics = lambda_physics
        self.m0             = m0

        self.register_buffer("latest_loss_physics", torch.tensor(0.0))

        if calib_csv is not None:
            calib = load_calib_csv(calib_csv)

            self.register_buffer(
                "calib_offset",
                torch.tensor(calib["offset"], dtype=torch.float32))

            self.register_buffer(
                "calib_gain",
                torch.tensor(calib["gain"], dtype=torch.float32))

            self.register_buffer(
                "sensor_pos",
                torch.tensor(calib["sensor_pos"], dtype=torch.float32))

            print(f"[HuberPoseLossMVec] Loaded calib from: {calib_csv}")
            print(f"  offset  : {calib['offset'].min():.4f} ~ "
                  f"{calib['offset'].max():.4f} V")
            print(f"  gain    : {calib['gain'].min():.4f} ~ "
                  f"{calib['gain'].max():.4f} V/T")
            print(f"  pos_x   : {calib['sensor_pos'][:,0].min():.4f} ~ "
                  f"{calib['sensor_pos'][:,0].max():.4f} m")
            print(f"  pos_y   : {calib['sensor_pos'][:,1].min():.4f} ~ "
                  f"{calib['sensor_pos'][:,1].max():.4f} m")
            print(f"  pos_z   : {calib['sensor_pos'][:,2].min():.4f} ~ "
                  f"{calib['sensor_pos'][:,2].max():.4f} m")
        else:
            self.calib_offset = None
            self.calib_gain   = None
            self.sensor_pos   = None

        if volt_scaler is not None:
            self.register_buffer(
                "volt_min",
                torch.tensor(volt_scaler.data_min_, dtype=torch.float32))
            self.register_buffer(
                "volt_scale",
                torch.tensor(
                    volt_scaler.data_max_ - volt_scaler.data_min_,
                    dtype=torch.float32))
        else:
            self.volt_min   = None
            self.volt_scale = None

        # xyz-only label scaler (PosLabelScaler)
        if label_scaler is not None:
            self.register_buffer(
                "label_mean",
                torch.tensor(label_scaler.xyz_scaler.mean_, dtype=torch.float32))
            self.register_buffer(
                "label_scale",
                torch.tensor(label_scaler.xyz_scaler.scale_, dtype=torch.float32))
        else:
            self.label_mean  = None
            self.label_scale = None

    def _inverse_xyz(self, pred: torch.Tensor) -> torch.Tensor:
        pred_orig = pred[:, :3] * self.label_scale + self.label_mean
        return pred_orig

    def _get_m_vec(self, pred: torch.Tensor) -> torch.Tensor:
        return self.m0 * pred[:, 3:]

    def _compute_Bz_dipole(self,
                            pred_pos: torch.Tensor,
                            m_vec:    torch.Tensor) -> torch.Tensor:
        r_vec  = self.sensor_pos.unsqueeze(0) - pred_pos.unsqueeze(1)

        r_norm = torch.linalg.norm(r_vec, dim=-1)
        r_norm = torch.clamp(r_norm, min=1e-3)

        m_dot_r = torch.sum(m_vec.unsqueeze(1) * r_vec, dim=-1)
        r_vec_z = r_vec[:, :, 2]
        m_vec_z = m_vec[:, 2].unsqueeze(1)

        Bz_pred = self.MU_0_4PI * (
            3.0 * m_dot_r * r_vec_z / (r_norm ** 5)
            - m_vec_z / (r_norm ** 3)
        )
        return Bz_pred

    def _bz_to_voltage_calib(self, Bz_pred: torch.Tensor) -> torch.Tensor:
        V_pred = (self.calib_offset.unsqueeze(0)
                  + self.calib_gain.unsqueeze(0) * Bz_pred)
        return V_pred

    def _inverse_voltage(self, X_b: torch.Tensor) -> torch.Tensor:
        volt_scaled = X_b.view(-1, 64)
        V_input     = volt_scaled * self.volt_scale + self.volt_min
        return V_input

    def forward(self,
                pred:   torch.Tensor,
                target: torch.Tensor,
                X_b:    torch.Tensor = None):
        pred   = pred.float()
        target = target.float()

        loss_xyz = F.huber_loss(pred[:, :3], target[:, :3],
                                delta=self.delta_xyz)

        m_pred = pred[:, 3:]
        m_gt   = F.normalize(target[:, 3:], dim=-1)
        loss_ori = torch.mean(1.0 - torch.sum(m_pred * m_gt, dim=-1))

        if torch.isnan(loss_xyz) or torch.isnan(loss_ori):
            self.latest_loss_physics.fill_(0.0)
            zero = pred.sum() * 0.0
            return zero, zero.detach(), zero.detach()

        loss_physics = torch.zeros(1, device=pred.device).squeeze()

        physics_ready = (
            X_b                is not None
            and self.calib_offset is not None
            and self.calib_gain   is not None
            and self.sensor_pos   is not None
            and self.volt_min     is not None
            and self.label_mean   is not None
        )

        if physics_ready:
            try:
                X_b = X_b.float()

                pred_pos = self._inverse_xyz(pred)
                m_vec    = self._get_m_vec(pred)

                Bz_pred = self._compute_Bz_dipole(pred_pos, m_vec)
                V_pred  = self._bz_to_voltage_calib(Bz_pred)
                V_input = self._inverse_voltage(X_b)

                loss_physics = torch.mean(torch.abs(V_pred - V_input))

                if not torch.isfinite(loss_physics):
                    loss_physics = torch.zeros(1, device=pred.device).squeeze()

            except Exception:
                loss_physics = torch.zeros(1, device=pred.device).squeeze()

        self.latest_loss_physics = loss_physics.detach().clone()

        total = (self.lambda_pos     * loss_xyz
               + self.lambda_ori     * loss_ori
               + self.lambda_physics * loss_physics)

        return total, loss_xyz, loss_ori

