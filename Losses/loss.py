import torch
import torch.nn as nn
import torch.nn.functional as F


class HuberPoseLoss(nn.Module):

    def __init__(self,
                 ang_weight: float = 1.0,
                 delta_xyz:  float = 0.055,
                 delta_ang:  float = 0.21,
                 lambda_pos: float = 1.0,
                 lambda_physics: float = 1e-4,
                 calib_csv: str = None,
                 volt_scaler = None,
                 label_scaler = None):

        super().__init__()
        self.ang_weight     = ang_weight
        self.delta_xyz      = delta_xyz
        self.delta_ang      = delta_ang
        self.lambda_pos     = lambda_pos
        self.lambda_physics = lambda_physics
        self.register_buffer("latest_loss_physics", torch.tensor(0.0))

        if calib_csv is not None:
            # Load calibration data from CSV
            pass
        else:
            self.calib_data = None

        if volt_scaler is not None:
            volt_min   = torch.tensor(volt_scaler.data_min_, dtype=torch.float32)
            volt_scale = torch.tensor(volt_scaler.data_max_ - volt_scaler.data_min_,
                                      dtype=torch.float32)
            self.register_buffer("volt_min",   volt_min)
            self.register_buffer("volt_scale", volt_scale)
        else:
            self.volt_min   = None
            self.volt_scale = None

        if label_scaler is not None:
            label_mean  = torch.tensor(label_scaler.mean_,  dtype=torch.float32)
            label_scale = torch.tensor(label_scaler.scale_, dtype=torch.float32)
            self.register_buffer("label_mean",  label_mean)
            self.register_buffer("label_scale", label_scale)
        else:
            self.label_mean  = None
            self.label_scale = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                X_b: torch.Tensor = None):

    
        pred   = pred.float()
        target = target.float()

        loss_xyz = F.huber_loss(pred[:, :3], target[:, :3], delta=self.delta_xyz)
        loss_ang = F.huber_loss(pred[:, 3:], target[:, 3:], delta=self.delta_ang)

      
        if torch.isnan(loss_xyz) or torch.isnan(loss_ang):
            self.latest_loss_physics.fill_(0.0)
            zero = pred.sum() * 0.0         
            return zero, zero.detach(), zero.detach()

        loss_physics = torch.zeros(1, device=pred.device).squeeze()

        if (X_b is not None
                and self.sensor_pos is not None
                and self.volt_min   is not None
                and self.label_mean is not None):
            try:
                X_b = X_b.float()

                # 1. Đưa prediction về đơn vị gốc
                pred_pos_orig = pred[:, :3] * self.label_scale[:3] + self.label_mean[:3]
                cos_pitch = pred[:, 3] * self.label_scale[3] + self.label_mean[3]
                cos_yaw   = pred[:, 4] * self.label_scale[4] + self.label_mean[4]

                cos_pitch = torch.clamp(cos_pitch, -1.0 + 1e-6, 1.0 - 1e-6)
                cos_yaw   = torch.clamp(cos_yaw,   -1.0 + 1e-6, 1.0 - 1e-6)

                sin_pitch = torch.sqrt(torch.clamp(1.0 - cos_pitch**2, min=1e-12))
                sin_yaw   = torch.sqrt(torch.clamp(1.0 - cos_yaw**2,   min=1e-12))

                mx = cos_pitch * cos_yaw
                my = cos_pitch * sin_yaw
                mz = sin_pitch
                m_vec = torch.stack([mx, my, mz], dim=-1)   # (B, 3)

                # 2. Đưa voltage scaled về volt gốc
                volt_scaled = X_b.view(-1, 64)
                V_input = volt_scaled * self.volt_scale + self.volt_min  # (B, 64)

                # 3. Tính Bz tại các sensor theo mô hình dipole
                r_vec  = self.sensor_pos.unsqueeze(0) - pred_pos_orig.unsqueeze(1)  # (B, 64, 3)
                r_norm = torch.linalg.norm(r_vec, dim=-1)                           # (B, 64)
                r_norm = torch.clamp(r_norm, min=1e-3)   

                m_dot_r = torch.sum(m_vec.unsqueeze(1) * r_vec, dim=-1)  # (B, 64)
                r_vec_z = r_vec[:, :, 2]                                  # (B, 64)
                m_vec_z = m_vec[:, 2].unsqueeze(1)                        # (B, 1)

                MU_0_4PI = 1e-7
                Bz_pred = MU_0_4PI * (
                    3.0 * m_dot_r * r_vec_z / (r_norm ** 5)
                    - m_vec_z / (r_norm ** 3)
                )                                                         # (B, 64)

                # Bz → voltage
                V_Q    = 1.65
                SENS   = 7.5
                V_pred = V_Q + SENS * Bz_pred                            # (B, 64)

                # Physics loss = MAE
                loss_physics = torch.mean(torch.abs(V_pred - V_input))

                if not torch.isfinite(loss_physics):
                    loss_physics = torch.zeros(1, device=pred.device).squeeze()

            except Exception:
                loss_physics = torch.zeros(1, device=pred.device).squeeze()

        self.latest_loss_physics = loss_physics.detach().clone()

        total = (self.lambda_pos     * loss_xyz
               + self.ang_weight     * loss_ang
               + self.lambda_physics * loss_physics)
        return total, loss_xyz, loss_ang
    