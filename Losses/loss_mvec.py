from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_SENSORS = 64

def load_physical_calib_csv(path: str) -> dict:
    
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Calibration CSV not found: {path}")
    
    df = pd.read_csv(path)
    required_columns = {"sensor_index", "x", "y", "z", "offset", "gain"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    
    df = df.sort_values("sensor_index").reset_index(drop=True)
    expected = np.arange(NUM_SENSORS)
    indices = df["sensor_index"].to_numpy()
    if len(df) != NUM_SENSORS or not np.array_equal(indices, expected):
        raise ValueError (f"{path} must contain exactly {NUM_SENSORS} rows with sensor_index from 0 to {NUM_SENSORS-1}. Found indices: {indices}")
    
    values = df[["x", "y", "z", "offset", "gain"]].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"{path} contains non-finite values in x, y, z, offset, or gain columns.")
    if np.any(np.isclose(values[:, 4], 0)):
        raise ValueError(f"{path} contains zero gain values.")
    return {
        "offset"     : values[:, 3],
        "gain"       : values[:, 4],
        "sensor_pos" : values[:, :3],
    }

def load_alpha_calib_csv(path: str) -> dict:
    " Load the alpha(h) = c0 + c1*h "
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Alpha calibration file not found: {path}")
    
    df = pd.read_csv(path)
    required = {"coefficient", "value"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    coeffs = dict(zip(df["coefficient"].astype(str).str.strip(), df["value"].astype(float)))
    missing_coeffs = {"c0", "c1"}.difference(coeffs.keys())
    if missing_coeffs: 
        raise ValueError(f"{path} is missing required coefficients: {sorted(missing_coeffs)}")
    c0, c1 = float(coeffs["c0"]), float(coeffs["c1"])
    if not np.isfinite([c0, c1]).all():
        raise ValueError(f"{path} contains non-finite values for c0 or c1.")
    return {"c0": c0, "c1": c1}

class HuberPoseLossMVec(nn.Module):
    MU_0_4PI = 1e-7   # mu0 / (4*pi)
    def __init__(self,
                 lambda_ori:     float = 1.0,
                 delta_xyz:      float = 0.061,
                 lambda_pos:     float = 1.0,
                 lambda_physics: float = 1e-4,
                 physics_delta:  float = 0.002,   
                 calib_physical_csv:      str | None  = None,
                 calib_alpha_csv:         str | None  = None,
                 volt_scaler            = None,
                 label_scaler           = None,
                 m0:             float = 1.0):
    
        super().__init__()
        if physics_delta <= 0:
            raise ValueError(f"physics_delta must be positive. Got {physics_delta}")
        if m0 <= 0:
            raise ValueError(f"m0 must be positive. Got {m0}")
        if (calib_physical_csv is None) != (calib_alpha_csv is None):
            raise ValueError("Both calib_physical_csv and calib_alpha_csv must be provided together, or both None.")
       
        self.lambda_ori     = lambda_ori
        self.delta_xyz      = delta_xyz
        self.lambda_pos     = lambda_pos
        self.lambda_physics = lambda_physics
        self.physics_delta  = physics_delta
        self.m0             = m0

        self.register_buffer("latest_loss_physics", torch.tensor(0.0))
        # Physics-free ablations must not require calibration/scaler artefacts.
        self._register_calibration(calib_physical_csv if lambda_physics != 0 else None,
                                   calib_alpha_csv if lambda_physics != 0 else None)
        self._register_scalers(volt_scaler if lambda_physics != 0 else None,
                               label_scaler if lambda_physics != 0 else None)

    def _register_calibration(self, physical_path, alpha_path) -> None:
        if physical_path is None: 
            self.calib_offset = self.calib_gain = self.sensor_pos = None
            self.alpha_c0 = self.alpha_c1 = None
            return
        calib = load_physical_calib_csv(physical_path)
        alpha = load_alpha_calib_csv(alpha_path)
        self.register_buffer("calib_offset", torch.tensor(calib["offset"], dtype=torch.float32))  # (64,)
        self.register_buffer("calib_gain",   torch.tensor(calib["gain"],   dtype=torch.float32))  # (64,)
        self.register_buffer("sensor_pos",   torch.tensor(calib["sensor_pos"], dtype=torch.float32))  # (64, 3)
        self.register_buffer("alpha_c0",     torch.tensor(alpha["c0"], dtype=torch.float32))  # scalar
        self.register_buffer("alpha_c1",     torch.tensor(alpha["c1"], dtype=torch.float32))  # scalar
        print(f"[HuberPoseLoss] Physical calibration: {physical_path}")
        print(f"[HuberPoseLoss] Alpha calibration: {alpha_path}")
        print(f" alpha(h) = {alpha['c0']:.8f} + {alpha['c1']:.8f} * h")
        
    def _register_scalers(self, volt_scaler, label_scaler) -> None: 
        if volt_scaler is None: 
            self.volt_min = self.volt_scale = None
        else: 
            if not hasattr(volt_scaler, "data_min_") or not hasattr(volt_scaler, "data_max_"):
                raise TypeError("volt_scaler must be a fitted sklearn MinMaxScaler")
            volt_min = np.asarray(volt_scaler.data_min_, dtype=np.float32)
            volt_scale = np.asarray(volt_scaler.data_max_ - volt_scaler.data_min_, dtype=np.float32)
            if volt_min.shape != (NUM_SENSORS,) or volt_scale.shape != (NUM_SENSORS,):
                raise ValueError(f"volt_scaler must be fitted to 64 voltage features")
            if not np.isfinite(volt_min).all() or not np.isfinite(volt_scale).all():
                raise ValueError("volt_scaler contains non-finite values")
            
            self.register_buffer("volt_min", torch.tensor(volt_min, dtype=torch.float32))
            self.register_buffer("volt_scale", torch.tensor(volt_scale, dtype=torch.float32))


        if label_scaler is None or not hasattr(label_scaler, "xyz_scaler"):
            self.label_mean  = None
            self.label_scale = None
        else:
            scaler = label_scaler.xyz_scaler
            if not hasattr(scaler, "mean_") or not hasattr(scaler, "scale_"):
                raise TypeError("label_scaler.xyz_scaler must be fitted")
            mean = np.asarray(scaler.mean_, dtype=np.float32)
            scale = np.asarray(scaler.scale_, dtype=np.float32)
            if mean.shape != (3,) or scale.shape != (3,) or not np.isfinite(mean).all() or not np.isfinite(scale).all():
                raise ValueError("label_scaler must contain finite xyz mean/scale values")
            self.register_buffer("label_mean", torch.tensor(mean, dtype=torch.float32))
            self.register_buffer("label_scale", torch.tensor(scale, dtype=torch.float32))
            
    @property
    def physics_ready(self) -> bool:
        return all(x is not None for x in (
            self.calib_offset, self.calib_gain, self.sensor_pos, self.alpha_c0,
            self.alpha_c1, self.volt_min, self.volt_scale, self.label_mean, self.label_scale,
        ))
        
    def _inverse_xyz(self, pred: torch.Tensor) -> torch.Tensor:
        return pred[:, :3] * self.label_scale + self.label_mean   # (B, 3)

    def _inverse_voltage(self, x_batch: torch.Tensor) -> torch.Tensor:
        flat = x_batch.reshape(-1, NUM_SENSORS)  # (B, 64)
        return flat * self.volt_scale + self.volt_min  # (B, 64)

    def _compute_Bz_raw(self, pred_pos: torch.Tensor, m_vec: torch.Tensor) -> torch.Tensor:
        r_vec = self.sensor_pos.unsqueeze(0) - pred_pos.unsqueeze(1)          # (B, 64, 3)
        r_norm = torch.linalg.norm(r_vec, dim=-1).clamp(min=1e-4)            # (B, 64)
        m_dot_r = torch.sum(m_vec.unsqueeze(1) * r_vec, dim=-1)               # (B, 64)
        B_vec = self.MU_0_4PI * (3 * m_dot_r.unsqueeze(-1) * r_vec / r_norm.unsqueeze(-1)**5
                                  - m_vec.unsqueeze(1) / r_norm.unsqueeze(-1)**3)   # (B, 64, 3)
        # Sensors are calibrated with a FIXED direction straight up [0, 0, 1]
        # (see calib_linear.py: sensor_dir = [0, 0, 1], B_proj = B @ sensor_dir),
        # so the projected scalar field is simply the z-component.
        return B_vec[..., 2]   # (B, 64)

    def physics_term(self, pred: torch.Tensor, x_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.physics_ready: 
            raise RuntimeError("Physics terms require both calibration files and fitted scalers")
        if x_batch is None:
            raise ValueError("x_batch must be provided for physics term computation")
        if pred.ndim != 2 or pred.shape[1] != 6:
            raise ValueError(f"pred must be of shape (B, 6). Got {tuple(pred.shape)}")
        if x_batch.numel() % NUM_SENSORS:
            raise ValueError(f"x_batch must contain a multiple of {NUM_SENSORS} values")
        
        pred = pred.float()
        voltage = self._inverse_voltage(x_batch.float())
        Bz_measured = (voltage - self.calib_offset.unsqueeze(0))/ self.calib_gain.unsqueeze(0)
        pred_pos = self._inverse_xyz(pred)
        Bz_raw = self._compute_Bz_raw(pred_pos, self.m0 * pred[:, 3:])
        h = pred_pos[:, 2:3] - self.sensor_pos[:, 2].unsqueeze(0)
        Bz_predicted = (self.alpha_c0 + self.alpha_c1*h) * Bz_raw
        return Bz_predicted, Bz_measured 
    
    def physics_per_sample(self, pred: torch.Tensor, x_batch: torch.Tensor) -> dict[str, torch.Tensor]:
        Bz_predicted, Bz_measured = self.physics_term(pred, x_batch)
        error = Bz_predicted - Bz_measured
        
        return {
            "Bz_predicted": Bz_predicted,
            "Bz_measured" : Bz_measured,
            "physics_huber": F.huber_loss(Bz_predicted, Bz_measured, delta = self.physics_delta, reduction = 'none').mean(dim = 1),
            "Bz_mae": error.abs().mean(dim = 1),
            "Bz_rmse": error.square().mean(dim = 1).sqrt(),
        } 

    def forward(self, pred: torch.Tensor, target: torch.Tensor, X_b: torch.Tensor | None = None):
        pred, target = pred.float(), target.float()
        loss_xyz = F.huber_loss(pred[:, :3], target[:, :3], delta = self.delta_xyz)
        m_pred = pred[:, 3:]
        m_gt = F.normalize(target[:, 3:], dim = 1)
        loss_ori = (1.0 - torch.sum(m_pred * m_gt, dim = -1)).mean() + 0.5 * F.mse_loss(m_pred, m_gt)
    
        if not torch.isfinite(loss_xyz) or not torch.isfinite(loss_ori):
            self.latest_loss_physics.zero_()
            zero = pred.sum() * 0.0
            return zero, zero.detach(), zero.detach()

        if X_b is None or self.lambda_physics == 0:
            loss_physics = pred.new_zeros(())
            self.latest_loss_physics.zero_()
        else:
            Bz_predicted, Bz_measured = self.physics_term(pred, X_b)
            loss_physics = F.huber_loss(Bz_predicted, Bz_measured, delta = self.physics_delta)
            if not torch.isfinite(loss_physics):
                raise FloatingPointError ("Physics loss is non-finite")

            self.latest_loss_physics.copy_(loss_physics.detach())

        total = self.lambda_pos * loss_xyz + self.lambda_ori * loss_ori + self.lambda_physics * loss_physics
        return total, loss_xyz, loss_ori


class CalibLocLoss(nn.Module):
    
    def __init__(self, pose_loss: HuberPoseLossMVec,
                 lambda_calib: float = 0.1, calib_delta: float = 0.05):
        super().__init__()
        if lambda_calib < 0:
            raise ValueError(f"lambda_calib must be >= 0. Got {lambda_calib}")
        if calib_delta <= 0:
            raise ValueError(f"calib_delta must be positive. Got {calib_delta}")
        self.pose_loss = pose_loss
        self.lambda_calib = lambda_calib
        self.calib_delta = calib_delta
        self.register_buffer("latest_loss_calib", torch.tensor(0.0))

    def calibration_term(self, corrected: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        if corrected.shape != clean.shape:
            raise ValueError(
                f"corrected and clean must have the same shape; got "
                f"{tuple(corrected.shape)} and {tuple(clean.shape)}"
            )
        return F.huber_loss(corrected.float(), clean.float(), delta=self.calib_delta)

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                corrected: torch.Tensor, clean: torch.Tensor):
        pose_total, loss_xyz, loss_ori = self.pose_loss(pred, target, X_b=corrected)
        loss_calib = self.calibration_term(corrected, clean)
        if not torch.isfinite(loss_calib):
            raise FloatingPointError("Calibration loss is non-finite")
        self.latest_loss_calib.copy_(loss_calib.detach())
        total = pose_total + self.lambda_calib * loss_calib
        return total, loss_calib, pose_total, loss_xyz, loss_ori
