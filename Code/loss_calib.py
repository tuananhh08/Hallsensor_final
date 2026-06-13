import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd


# =============================================================================
# Helper: load calibration từ CSV
# =============================================================================

def load_calib_csv(path: str) -> dict:
    df = pd.read_csv(path)

    # Kiểm tra columns cần thiết
    required = {"sensor_index", "offset_a_V", "gain_g_V_per_T", "x", "y", "z"}
    missing  = required - set(df.columns)
    assert not missing, f"CSV thiếu columns: {missing}"
    assert len(df) == 64, f"CSV phải có 64 rows (sensors), got {len(df)}"

    # Sort theo sensor_index để đảm bảo thứ tự 1→64
    df = df.sort_values("sensor_index").reset_index(drop=True)

    return {
        "offset"     : df["offset_a_V"].values.astype(np.float32),       # (64,)
        "gain"       : df["gain_g_V_per_T"].values.astype(np.float32),   # (64,)
        "sensor_pos" : df[["x", "y", "z"]].values.astype(np.float32),   # (64, 3)
    }


# =============================================================================
# Loss class
# =============================================================================

class HuberPoseLossCalib(nn.Module):
    MU_0_4PI = 1e-7   # μ₀ / (4π)  [T·m/A]

    def __init__(self,
                 ang_weight:     float = 1.0,
                 delta_xyz:      float = 0.061,
                 delta_ang:      float = 0.21,
                 lambda_pos:     float = 1.0,
                 lambda_physics: float = 1e-4,
                 calib_csv:      str   = None,
                 volt_scaler            = None,
                 label_scaler           = None,
                 m0:             float = 1.0):

        super().__init__()

        self.ang_weight     = ang_weight
        self.delta_xyz      = delta_xyz
        self.delta_ang      = delta_ang
        self.lambda_pos     = lambda_pos
        self.lambda_physics = lambda_physics
        self.m0             = m0

        self.register_buffer("latest_loss_physics", torch.tensor(0.0))

        # ── Calibration parameters ────────────────────────────────────────────
        if calib_csv is not None:
            calib = load_calib_csv(calib_csv)

            # Per-sensor offset  (64,)  [V]
            self.register_buffer(
                "calib_offset",
                torch.tensor(calib["offset"], dtype=torch.float32))

            # Per-sensor gain  (64,)  [V/T]  — có dấu âm
            self.register_buffer(
                "calib_gain",
                torch.tensor(calib["gain"], dtype=torch.float32))

            # Per-sensor tọa độ  (64, 3)  [m]
            self.register_buffer(
                "sensor_pos",
                torch.tensor(calib["sensor_pos"], dtype=torch.float32))

            print(f"[HuberPoseLossCalib] Loaded calib from: {calib_csv}")
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

        # ── Voltage scaler  (inverse transform X_b → V original) ─────────────
        if volt_scaler is not None:
            self.register_buffer(
                "volt_min",
                torch.tensor(volt_scaler.data_min_, dtype=torch.float32))  # (64,)
            self.register_buffer(
                "volt_scale",
                torch.tensor(
                    volt_scaler.data_max_ - volt_scaler.data_min_,
                    dtype=torch.float32))                                   # (64,)
        else:
            self.volt_min   = None
            self.volt_scale = None

        # ── Label scaler  (inverse transform pred → original units) ──────────
        if label_scaler is not None:
            self.register_buffer(
                "label_mean",
                torch.tensor(label_scaler.mean_,  dtype=torch.float32))   # (5,)
            self.register_buffer(
                "label_scale",
                torch.tensor(label_scaler.scale_, dtype=torch.float32))   # (5,)
        else:
            self.label_mean  = None
            self.label_scale = None

    # =========================================================================
    # Private helpers
    # =========================================================================

    def _inverse_label(self, pred: torch.Tensor):
        pred_orig = pred * self.label_scale + self.label_mean   # (B, 5)
        pred_pos  = pred_orig[:, :3]                            # (B, 3)
        cos_pitch = torch.clamp(pred_orig[:, 3], -1.0 + 1e-6,  1.0 - 1e-6)
        cos_yaw   = torch.clamp(pred_orig[:, 4], -1.0 + 1e-6,  1.0 - 1e-6)
        return pred_pos, cos_pitch, cos_yaw

    def _build_m_vec(self,
                     cos_pitch: torch.Tensor,
                     cos_yaw:   torch.Tensor) -> torch.Tensor:

        sin_pitch = torch.sqrt(torch.clamp(1.0 - cos_pitch**2, min=1e-12))
        sin_yaw   = torch.sqrt(torch.clamp(1.0 - cos_yaw**2,   min=1e-12))
        mx = cos_pitch * cos_yaw
        my = cos_pitch * sin_yaw
        mz = sin_pitch
        return torch.stack([mx, my, mz], dim=-1)   # (B, 3)

    def _compute_Bz_dipole(self,
                            pred_pos: torch.Tensor,
                            m_vec:    torch.Tensor) -> torch.Tensor:
        # (B, 64, 3)
        r_vec  = self.sensor_pos.unsqueeze(0) - pred_pos.unsqueeze(1)

        # (B, 64)
        r_norm = torch.linalg.norm(r_vec, dim=-1)
        r_norm = torch.clamp(r_norm, min=1e-3)      # tránh chia 0

        m_dot_r = torch.sum(m_vec.unsqueeze(1) * r_vec, dim=-1)  # (B, 64)
        r_vec_z = r_vec[:, :, 2]                                   # (B, 64)
        m_vec_z = m_vec[:, 2].unsqueeze(1)                        # (B,  1)

        Bz_pred = self.MU_0_4PI * (
            3.0 * m_dot_r * r_vec_z / (r_norm ** 5)
            - m_vec_z / (r_norm ** 3)
        )                                                           # (B, 64)
        return Bz_pred

    def _bz_to_voltage_calib(self, Bz_pred: torch.Tensor) -> torch.Tensor:
        # broadcast (64,) → (1, 64)
        V_pred = (self.calib_offset.unsqueeze(0)
                  + self.calib_gain.unsqueeze(0) * Bz_pred)        # (B, 64)
        return V_pred

    def _inverse_voltage(self, X_b: torch.Tensor) -> torch.Tensor:
        volt_scaled = X_b.view(-1, 64)                                 # (B, 64)
        V_input     = volt_scaled * self.volt_scale + self.volt_min    # (B, 64)
        return V_input

    # =========================================================================
    # Forward
    # =========================================================================

    def forward(self,
                pred:   torch.Tensor,
                target: torch.Tensor,
                X_b:    torch.Tensor = None):
        pred   = pred.float()
        target = target.float()

        # ── 1. Huber losses ───────────────────────────────────────────────────
        loss_xyz = F.huber_loss(pred[:, :3], target[:, :3],
                                delta=self.delta_xyz)
        loss_ang = F.huber_loss(pred[:, 3:], target[:, 3:],
                                delta=self.delta_ang)

        # Fallback nếu NaN (không nên xảy ra nhưng giữ an toàn)
        if torch.isnan(loss_xyz) or torch.isnan(loss_ang):
            self.latest_loss_physics.fill_(0.0)
            zero = pred.sum() * 0.0
            return zero, zero.detach(), zero.detach()

        # ── 2. Physics consistency loss ───────────────────────────────────────
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

                # 2a. Inverse transform pred → original units
                pred_pos, cos_pitch, cos_yaw = self._inverse_label(pred)

                # 2b. Magnetic moment vector  (B, 3)
                m_vec = self.m0 * self._build_m_vec(cos_pitch, cos_yaw)

                # 2c. Bz tại 64 sensors theo dipole model  (B, 64)
                Bz_pred = self._compute_Bz_dipole(pred_pos, m_vec)

                # 2d. Bz → V dùng calibrated per-sensor params  (B, 64)
                V_pred = self._bz_to_voltage_calib(Bz_pred)

                # 2e. Inverse transform input voltage  (B, 64)
                V_input = self._inverse_voltage(X_b)

                # 2f. Physics loss = MAE(V_pred_calib, V_measured)
                loss_physics = torch.mean(torch.abs(V_pred - V_input))

                if not torch.isfinite(loss_physics):
                    loss_physics = torch.zeros(1, device=pred.device).squeeze()

            except Exception:
                loss_physics = torch.zeros(1, device=pred.device).squeeze()

        self.latest_loss_physics = loss_physics.detach().clone()

        # ── 3. Total loss ─────────────────────────────────────────────────────
        total = (self.lambda_pos     * loss_xyz
               + self.ang_weight     * loss_ang
               + self.lambda_physics * loss_physics)

        return total, loss_xyz, loss_ang
