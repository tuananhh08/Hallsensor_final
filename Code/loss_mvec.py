import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd


def load_calib_csv(path: str) -> dict:
    df = pd.read_csv(path).sort_values("sensor_index").reset_index(drop=True)
    return {
        "offset"     : df["offset_a_V"].values.astype(np.float32),
        "gain"       : df["gain_g_V_per_T"].values.astype(np.float32),
        "sensor_pos" : df[["x", "y", "z"]].values.astype(np.float32),
    }


class HuberPoseLossMVec(nn.Module):
    MU_0_4PI = 1e-7   # μ₀ / (4π)  [T·m/A]

    def __init__(self,
                 lambda_ori:     float = 1.0,
                 delta_xyz:      float = 0.061,
                 lambda_pos:     float = 1.0,
                 lambda_physics: float = 0.05,  
                 physics_delta:  float = 0.05,   # THÊM MỚI: Huber delta cho relative error (5%)
                 calib_csv:      str   = None,
                 volt_scaler            = None,
                 label_scaler           = None,
                 m0:             float = 1.0):

        super().__init__()

        self.lambda_ori     = lambda_ori
        self.delta_xyz      = delta_xyz
        self.lambda_pos     = lambda_pos
        self.lambda_physics = lambda_physics
        self.physics_delta  = physics_delta   # THÊM MỚI
        self.m0             = m0

        self.register_buffer("latest_loss_physics", torch.tensor(0.0))

        # ── Calib params ─────────────────────────────────────────────────────
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

            print(f"[HuberPoseLossMVec] Loaded calib: {calib_csv}")
            print(f"  offset : [{calib['offset'].min():.4f}, {calib['offset'].max():.4f}] V")
            print(f"  gain   : [{calib['gain'].min():.4f},  {calib['gain'].max():.4f}] V/T")
        else:
            self.calib_offset = None
            self.calib_gain   = None
            self.sensor_pos   = None

        # ── Voltage scaler buffers ────────────────────────────────────────────
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

        # ── Label scaler buffers (xyz only) ───────────────────────────────────
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

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _inverse_xyz(self, pred: torch.Tensor) -> torch.Tensor:
        """Chuyển xyz từ scaled space về đơn vị mét."""
        return pred[:, :3] * self.label_scale + self.label_mean

    def _inverse_voltage(self, X_b: torch.Tensor) -> torch.Tensor:
        """Chuyển voltage từ [0,1] về đơn vị Volt."""
        return X_b.view(-1, 64) * self.volt_scale + self.volt_min

    def _compute_Bz_dipole(self,
                            pred_pos: torch.Tensor,
                            m_vec:    torch.Tensor) -> torch.Tensor:
        """Tính Bz tại 64 sensor theo dipole model."""
        r_vec   = self.sensor_pos.unsqueeze(0) - pred_pos.unsqueeze(1)  # (B, 64, 3)
        r_norm  = torch.linalg.norm(r_vec, dim=-1).clamp(min=1e-3)      # (B, 64)
        m_dot_r = torch.sum(m_vec.unsqueeze(1) * r_vec, dim=-1)         # (B, 64)

        Bz = self.MU_0_4PI * (
            3.0 * m_dot_r * r_vec[:, :, 2] / (r_norm ** 5)
            - m_vec[:, 2].unsqueeze(1)      / (r_norm ** 3)
        )
        return Bz   # (B, 64)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self,
                pred:   torch.Tensor,
                target: torch.Tensor,
                X_b:    torch.Tensor = None):

        pred   = pred.float()
        target = target.float()

        # ── 1. Position loss — Huber (scaled space) ──────────────────────────
        loss_xyz = F.huber_loss(
            pred[:, :3], target[:, :3], delta=self.delta_xyz)

        # ── 2. Orientation loss — CosSim + MSE ───────────────────────────────
        # MSE component để nhạy hơn ở sai số nhỏ
        m_pred  = pred[:, 3:]                           # unit vector từ model
        m_gt    = F.normalize(target[:, 3:], dim=-1)    # đảm bảo unit vector

        cos_sim  = torch.sum(m_pred * m_gt, dim=-1)     # (B,)  ∈ [-1, 1]
        loss_cos = torch.mean(1.0 - cos_sim)            # ∈ [0, 2]
        loss_mse_m = F.mse_loss(m_pred, m_gt)           # MSE trên components

        # Kết hợp: CosSim tốt khi lệch lớn, MSE tốt khi lệch nhỏ
        loss_ori = loss_cos + 0.5 * loss_mse_m

        # ── 3. NaN guard ─────────────────────────────────────────────────────
        if not torch.isfinite(loss_xyz) or not torch.isfinite(loss_ori):
            self.latest_loss_physics.fill_(0.0)
            zero = pred.sum() * 0.0
            return zero, zero.detach(), zero.detach()

        # ── 4. Physics loss — Huber trên relative error ───────────────────────
        # THAY ĐỔI: relative error thay vì MAE tuyệt đối
        # → scale-invariant, lambda_physics có thể dùng range [0.01, 0.5]
        loss_physics = torch.zeros(1, device=pred.device).squeeze()

        physics_ready = (
            X_b               is not None
            and self.calib_offset is not None
            and self.calib_gain   is not None
            and self.sensor_pos   is not None
            and self.volt_min     is not None
            and self.label_mean   is not None
        )

        if physics_ready:
            try:
                # Inverse xyz về mét
                pred_pos = self._inverse_xyz(pred)          # (B, 3) [m]

                # m_vec scale về moment gốc
                m_vec    = self.m0 * m_pred                 # unit vector * m0

                # Tính Bz dipole → V_pred
                Bz_pred  = self._compute_Bz_dipole(pred_pos, m_vec)  # (B, 64)
                V_pred   = (self.calib_offset.unsqueeze(0)
                            + self.calib_gain.unsqueeze(0) * Bz_pred)  # (B, 64) [V]

                # Inverse voltage về Volt
                V_input  = self._inverse_voltage(X_b)      # (B, 64) [V]

                # Relative error: (V_pred - V_input) / |V_input|
                V_ref       = torch.abs(V_input).clamp(min=1e-3)
                rel_err     = (V_pred - V_input) / V_ref   

                # Huber trên relative error
                loss_physics = F.huber_loss(
                    rel_err,
                    torch.zeros_like(rel_err),
                    delta=self.physics_delta)   # nhạy với sai số > physics_delta*100 %

                if not torch.isfinite(loss_physics):
                    loss_physics = torch.zeros(1, device=pred.device).squeeze()

            except Exception as e:
                loss_physics = torch.zeros(1, device=pred.device).squeeze()

        self.latest_loss_physics = loss_physics.detach().clone()

        # ── 5. Total ─────────────────────────────────────────────────────────
        total = (self.lambda_pos     * loss_xyz
               + self.lambda_ori     * loss_ori
               + self.lambda_physics * loss_physics)

        return total, loss_xyz, loss_ori