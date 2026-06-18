import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd


# =============================================================================
# Helper: load calibration CSV
# =============================================================================

def load_calib_csv(path: str) -> dict:

    df = pd.read_csv(path).sort_values("sensor_index").reset_index(drop=True)
    assert len(df) == 64, f"Calib CSV phải có 64 rows, got {len(df)}"
    return {
        "offset"     : df["offset_a_V"].values.astype(np.float32),
        "gain"       : df["gain_g_V_per_T"].values.astype(np.float32),
        "sensor_pos" : df[["x", "y", "z"]].values.astype(np.float32),
    }


# =============================================================================
# Loss class
# =============================================================================

class HuberPoseLossMVec(nn.Module):
    """
    Huber Pose Loss + Physics-Informed Consistency Loss (Bz domain).

    Total loss:
        L = lambda_pos * L_huber_xyz
          + lambda_ori * L_ori
          + lambda_physics * L_physics_Bz

    L_ori = cosine_loss + 0.5 * mse_loss   
    
    L_physics_Bz = Huber(Bz_pred, Bz_measured, delta=physics_delta)
        Bz_measured_i = (V_measured_i - offset_i) / gain_i
        Bz_pred       = dipole_model(pred_pos, m_vec)

    Parameters
    ----------
    lambda_ori     : hệ số nhân orientation loss
    delta_xyz      : Huber delta cho xyz (scaled space)
    lambda_pos     : hệ số nhân position loss
    lambda_physics : hệ số nhân physics loss
    physics_delta  : Huber delta cho Bz error [T]
                     default=0.002 T ≈ 1-std của Bz thực nghiệm
    calib_csv      : path đến calibration CSV
    volt_scaler    : sklearn MinMaxScaler fit trên voltage
    label_scaler   : PosLabelScaler (scale xyz only)
    m0             : magnetic moment magnitude [A·m²] (default 1.0)
    """

    MU_0_4PI = 1e-7   # μ₀ / (4π)  [T·m/A]

    def __init__(self,
                 lambda_ori:     float = 1.0,
                 delta_xyz:      float = 0.061,
                 lambda_pos:     float = 1.0,
                 lambda_physics: float = 1e-4,
                 physics_delta:  float = 0.002,   # [T] — ~1 std Bz thực nghiệm
                 calib_csv:      str   = None,
                 volt_scaler            = None,
                 label_scaler           = None,
                 m0:             float = 1.0):

        super().__init__()

        self.lambda_ori     = lambda_ori
        self.delta_xyz      = delta_xyz
        self.lambda_pos     = lambda_pos
        self.lambda_physics = lambda_physics
        self.physics_delta  = physics_delta
        self.m0             = m0

        self.register_buffer("latest_loss_physics", torch.tensor(0.0))

        # ── Calibration parameters ────────────────────────────────────────────
        if calib_csv is not None:
            calib = load_calib_csv(calib_csv)

            self.register_buffer(
                "calib_offset",
                torch.tensor(calib["offset"], dtype=torch.float32))   # (64,)

            self.register_buffer(
                "calib_gain",
                torch.tensor(calib["gain"], dtype=torch.float32))     # (64,) âm

            self.register_buffer(
                "sensor_pos",
                torch.tensor(calib["sensor_pos"], dtype=torch.float32))  # (64,3)

            print(f"[HuberPoseLossMVec] Loaded calib: {calib_csv}")
            print(f"  offset : [{calib['offset'].min():.4f}, {calib['offset'].max():.4f}] V")
            print(f"  gain   : [{calib['gain'].min():.4f}, {calib['gain'].max():.4f}] V/T")
            print(f"  physics_delta = {physics_delta:.4f} T  (Bz domain)")
        else:
            self.calib_offset = None
            self.calib_gain   = None
            self.sensor_pos   = None

        # ── Voltage scaler buffers ────────────────────────────────────────────
        if volt_scaler is not None:
            self.register_buffer(
                "volt_min",
                torch.tensor(volt_scaler.data_min_, dtype=torch.float32))   # (64,)
            self.register_buffer(
                "volt_scale",
                torch.tensor(
                    volt_scaler.data_max_ - volt_scaler.data_min_,
                    dtype=torch.float32))                                    # (64,)
        else:
            self.volt_min   = None
            self.volt_scale = None

        # ── Label scaler buffers (xyz only) ───────────────────────────────────
        if label_scaler is not None:
            self.register_buffer(
                "label_mean",
                torch.tensor(label_scaler.xyz_scaler.mean_,  dtype=torch.float32))  # (3,)
            self.register_buffer(
                "label_scale",
                torch.tensor(label_scaler.xyz_scaler.scale_, dtype=torch.float32))  # (3,)
        else:
            self.label_mean  = None
            self.label_scale = None

    # =========================================================================
    # Private helpers
    # =========================================================================

    def _inverse_xyz(self, pred: torch.Tensor) -> torch.Tensor:
        """pred[:, :3] scaled → mét."""
        return pred[:, :3] * self.label_scale + self.label_mean   # (B, 3)

    def _inverse_voltage(self, X_b: torch.Tensor) -> torch.Tensor:
        """X_b scaled [0,1] → Volt."""
        return X_b.view(-1, 64) * self.volt_scale + self.volt_min  # (B, 64)

    def _voltage_to_Bz(self, V_measured: torch.Tensor) -> torch.Tensor:

        # broadcast (64,) → (1, 64)
        Bz = (V_measured - self.calib_offset.unsqueeze(0)) / \
             self.calib_gain.unsqueeze(0)                    # (B, 64)
        return Bz

    def _compute_Bz_dipole(self,
                            pred_pos: torch.Tensor,
                            m_vec:    torch.Tensor) -> torch.Tensor:

        r_vec  = self.sensor_pos.unsqueeze(0) - pred_pos.unsqueeze(1)  # (B, 64, 3)
        r_norm = torch.linalg.norm(r_vec, dim=-1).clamp(min=1e-3)      # (B, 64)

        m_dot_r = torch.sum(m_vec.unsqueeze(1) * r_vec, dim=-1)        # (B, 64)
        r_vec_z = r_vec[:, :, 2]                                        # (B, 64)
        m_vec_z = m_vec[:, 2].unsqueeze(1)                             # (B, 1)

        Bz_pred = self.MU_0_4PI * (
            3.0 * m_dot_r * r_vec_z / (r_norm ** 5)
            - m_vec_z / (r_norm ** 3)
        )                                                               # (B, 64)
        return Bz_pred

    # =========================================================================
    # Forward
    # =========================================================================

    def forward(self,
                pred:   torch.Tensor,
                target: torch.Tensor,
                X_b:    torch.Tensor = None):

        pred   = pred.float()
        target = target.float()

        # ── 1. Position loss — Huber (scaled space) ──────────────────────────
        loss_xyz = F.huber_loss(
            pred[:, :3], target[:, :3], delta=self.delta_xyz)

        # ── 2. Orientation loss — cosine + MSE ───────────────────────────────
        # m_pred từ model output (đã được normalize trong model.forward)
        m_pred = pred[:, 3:]                                # (B, 3)
        m_gt   = F.normalize(target[:, 3:], dim=-1)        # (B, 3) unit vector

        cos_sim    = torch.sum(m_pred * m_gt, dim=-1)       # (B,)
        loss_cos   = torch.mean(1.0 - cos_sim)              # ∈ [0, 2]
        loss_mse_m = F.mse_loss(m_pred, m_gt)

        # Cosine: nhạy khi lệch lớn; MSE: nhạy khi lệch nhỏ → kết hợp
        loss_ori = loss_cos + 0.5 * loss_mse_m

        # ── 3. NaN guard ─────────────────────────────────────────────────────
        if not torch.isfinite(loss_xyz) or not torch.isfinite(loss_ori):
            self.latest_loss_physics.fill_(0.0)
            zero = pred.sum() * 0.0
            return zero, zero.detach(), zero.detach()

        # ── 4. Physics loss — Huber trên Bz domain [T] ───────────────────────
        #
        # Quy trình:
        #   V_measured  →  Bz_measured  (inverse calib)
        #   pred_pose   →  Bz_predicted (dipole model)
        #   L_physics   =  Huber(Bz_pred, Bz_measured, delta=physics_delta)
        # ─────────────────────────────────────────────────────────────────────
        loss_physics = torch.zeros(1, device=pred.device).squeeze()

        physics_ready = (
            X_b                  is not None
            and self.calib_offset is not None
            and self.calib_gain   is not None
            and self.sensor_pos   is not None
            and self.volt_min     is not None
            and self.label_mean   is not None
        )

        if physics_ready:
            try:
                X_b = X_b.float()

                # 4a. Inverse transform: voltage scaled → V → Bz_measured
                V_measured  = self._inverse_voltage(X_b)          # (B, 64) [V]
                Bz_measured = self._voltage_to_Bz(V_measured)     # (B, 64) [T]

                # 4b. Inverse transform: xyz scaled → mét
                pred_pos = self._inverse_xyz(pred)                 # (B, 3)  [m]

                # 4c. m_vec (đã unit) scale lên m0
                m_vec = self.m0 * m_pred                           # (B, 3)  [A·m²]

                # 4d. Dipole model → Bz_predicted
                Bz_pred = self._compute_Bz_dipole(pred_pos, m_vec) # (B, 64) [T]

                # 4e. Huber loss trực tiếp trên Bz [T]
                #     delta = 0.002 T được chọn từ thống kê:
                #       - std(Bz) = 0.0024 T
                #       - 84th percentile ≈ 0.0024 T
                loss_physics = F.huber_loss(
                    Bz_pred, Bz_measured,
                    delta=self.physics_delta)                       # scalar [T²/T]

                if not torch.isfinite(loss_physics):
                    loss_physics = torch.zeros(1, device=pred.device).squeeze()

            except Exception:
                loss_physics = torch.zeros(1, device=pred.device).squeeze()

        self.latest_loss_physics = loss_physics.detach().clone()

        # ── 5. Total loss ─────────────────────────────────────────────────────
        total = (self.lambda_pos     * loss_xyz
               + self.lambda_ori     * loss_ori
               + self.lambda_physics * loss_physics)

        return total, loss_xyz, loss_ori