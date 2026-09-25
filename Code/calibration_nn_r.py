"""
    !python calib_alpha_r_colab.py \
        --sensor_positions "/content/drive/MyDrive/Dataset/Hall_sensor_positions.csv" \
        --robot_pose       "/content/drive/MyDrive/Dataset/Grid_points_coordinates.csv" \
        --voltage          "/content/drive/MyDrive/Dataset/Grid_data.csv" \
        --offset_init      "/content/drive/MyDrive/Dataset/Offset_Sens.csv" \
        --output_dir       "/content/drive/MyDrive/Dataset/alpha_r_out" \
        --n_trials 30 
        --max_epochs 150 
        --patience 15
"""

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
except ImportError:
    sys.exit("PyTorch is required.\n  !pip install torch --quiet")

try:
    from scipy.optimize import least_squares
except ImportError:
    sys.exit("scipy is required.\n  !pip install scipy --quiet")

try:
    import optuna
except ImportError:
    sys.exit("optuna is required for Stage-2 hyperparameter search.\n  !pip install optuna --quiet")


MU0_OVER_4PI = 1e-7
N_SENSORS = 64


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# CLI
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="2-stage Hall-sensor calibration: Stage 1 physical params "
                     "(least_squares, unchanged) + Stage 2 distance-only "
                     "nonlinear alpha(r) NN, Optuna-tuned training "
                     "hyperparameters, fixed architecture.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--sensor_positions", required=True)
    p.add_argument("--robot_pose", required=True,
                    help="CSV columns: x,y,z,mx,my,mz")
    p.add_argument("--voltage", required=True,
                    help="CSV of measured voltages, one column per sensor, row-aligned with --robot_pose.")
    p.add_argument("--offset_init", required=True,
                    help="CSV columns: sensor_index, offset_a_V")
    p.add_argument("--output_dir", required=True)

    # ---- Stage 1 / Stage 2 pool sizes (preserved exactly) ----
    p.add_argument("--n_total_samples", type=int, default=3200)
    p.add_argument("--n_stage1_samples", type=int, default=400)

    # ---- Stage 2 data split: fractions of the Stage-2 pool (must sum to 1) ----
    p.add_argument("--train_fraction", type=float, default=0.70)
    p.add_argument("--val_fraction", type=float, default=0.18)
    p.add_argument("--test_fraction", type=float, default=0.07)

    # ---- Stage 1 regularization weights (preserved exactly) ----
    p.add_argument("--lambda_pos", type=float, default=2000)
    p.add_argument("--lambda_gain", type=float, default=9e-3)
    p.add_argument("--lambda_offset", type=float, default=750)

    # ---- Stage 2 NN: FIXED architecture  ----
    p.add_argument("--hidden_dim", type=int, default=64,
                    help="Shared scalar MLP hidden width: 1 -> hidden -> SiLU -> hidden -> SiLU -> 1.")
    p.add_argument("--mlp_n_hidden_layers", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=64,
                    help="Fixed training batch size (not tuned by Optuna).")

    # ---- Stage 2 Optuna: training hyperparameters only ----
    p.add_argument("--n_trials", type=int, default=35,
                    help="Optuna trials for the NN Stage-2 model.")
    p.add_argument("--max_epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--min_delta", type=float, default=1e-8)
    p.add_argument("--huber_delta_v", type=float, default=1e-3,
                    help="Huber loss delta, in volts.")
    p.add_argument("--n_representative_sensors", type=int, default=5,
                    help="How many sensors to plot alpha(r) curves for.")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None, help="Force 'cuda' or 'cpu'. Default: auto-detect.")
    return p


# =============================================================================
# DIPOLE MODEL -- Stage 1 
# =============================================================================

def dipole_field(r_vec, m_vec):
    r = np.linalg.norm(r_vec, axis=1, keepdims=True)
    r3 = np.maximum(r ** 3, 1e-12)
    r5 = np.maximum(r ** 5, 1e-12)
    mdotr = np.sum(m_vec * r_vec, axis=1, keepdims=True)
    B = MU0_OVER_4PI * (3.0 * r_vec * mdotr / r5 - m_vec / r3)
    return B


# =============================================================================
# LOADERS
# =============================================================================

def load_sensor_positions(file_path):
    df = pd.read_csv(file_path)
    sensor_positions = df.values
    print(f"Loaded sensor positions: {sensor_positions.shape}")
    return sensor_positions


def load_robot_pose(file_path):
    df = pd.read_csv(file_path)
    required_cols = ["x", "y", "z", "mx", "my", "mz"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    positions = df[["x", "y", "z"]].values
    m_world = df[["mx", "my", "mz"]].values
    norm = np.linalg.norm(m_world, axis=1, keepdims=True)
    m_world = m_world / norm
    print(f"Loaded robot positions: {positions.shape}")
    print(f"Loaded magnetic orientations: {m_world.shape}")
    return positions, m_world


def load_voltage_data(file_path):
    df = pd.read_csv(file_path)
    voltage = df.values
    print(f"Loaded voltage data: {voltage.shape}")
    return voltage


def load_offset_initial_values(file_path, n_sensors):
    df = pd.read_csv(file_path)
    required_columns = {"sensor_index", "offset_a_V"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing column(s) in {Path(file_path).name}: {sorted(missing_columns)}")
    if df["sensor_index"].duplicated().any():
        raise ValueError("offset_init CSV contains duplicate sensor_index values.")
    df = df.sort_values("sensor_index").reset_index(drop=True)
    expected_indices = np.arange(n_sensors)
    actual_indices = df["sensor_index"].to_numpy()
    if not np.array_equal(actual_indices, expected_indices):
        raise ValueError(f"offset_init CSV must contain exactly sensor_index values 0 to {n_sensors - 1}.")
    offset_initial_values = df["offset_a_V"].to_numpy(dtype=float)
    if not np.isfinite(offset_initial_values).all():
        raise ValueError("Column offset_a_V contains missing or non-finite values.")
    print(f"Loaded per-sensor offset initial values: {offset_initial_values.shape}")
    return offset_initial_values


# =============================================================================
# STAGE 1 -- least_squares physical fit
# =============================================================================

def sensor_residuals(params, robot_positions, m_world, voltage_sensor,
                      pos_prior, offset_prior, g0, lambda_pos, lambda_gain, lambda_offset):
    x, y, z, a, g = params
    sensor_dir = np.array([0.0, 0.0, 1.0])
    sensor_pos = np.array([x, y, z])
    r_vec = sensor_pos - robot_positions
    B = dipole_field(r_vec, m_world)
    B_proj = B @ sensor_dir
    voltage_pred = a + g * B_proj
    r_voltage = voltage_sensor - voltage_pred
    if not np.all(np.isfinite(r_voltage)):
        print(f"WARNING: Non-finite residuals detected at params: {params}")
    x0, y0, z0 = pos_prior
    r_pos = np.sqrt(lambda_pos) * np.array([x - x0, y - y0, z - z0])
    r_gain = np.sqrt(lambda_gain) * np.array([g - g0])
    r_offset = np.sqrt(lambda_offset) * np.array([a - offset_prior])
    return np.concatenate([r_voltage, r_pos, r_offset, r_gain])


def calibrate_single_sensor(sensor_index, sensor_pos_init, robot_positions,
                             m_world, voltage_sensor, offset_init,
                             lambda_pos, lambda_gain, lambda_offset):
    g0 = 7.5
    x0 = np.array([sensor_pos_init[0], sensor_pos_init[1], sensor_pos_init[2], offset_init, g0])
    pos_tol = 0.001
    lower = [sensor_pos_init[0] - pos_tol, sensor_pos_init[1] - pos_tol,
             sensor_pos_init[2] - pos_tol, offset_init - 0.0011, 7]
    upper = [sensor_pos_init[0] + pos_tol, sensor_pos_init[1] + pos_tol,
             sensor_pos_init[2] + pos_tol, offset_init + 0.0011, 8]
    result = least_squares(
        sensor_residuals, x0, bounds=(lower, upper),
        args=(robot_positions, m_world, voltage_sensor,
              (sensor_pos_init[0], sensor_pos_init[1], sensor_pos_init[2]),
              offset_init, g0, lambda_pos, lambda_gain, lambda_offset),
        method="trf", max_nfev=250,
    )
    params_opt = result.x
    theta_opt, phi_opt = 0.0, 0.0
    nx, ny, nz = 0.0, 0.0, 1.0
    params_extended = np.array([
        params_opt[0], params_opt[1], params_opt[2], params_opt[3], params_opt[4],
        nx, ny, nz, theta_opt, phi_opt,
    ])
    n_voltage = voltage_sensor.shape[0]
    rmse = np.sqrt(np.mean(result.fun[:n_voltage] ** 2))
    print(f"Sensor {sensor_index + 1:02d} | RMSE = {rmse:.6f}")
    return params_extended, rmse


def run_calibration(sensor_positions, robot_positions, m_world, voltage_data,
                     offset_initial_values, lambda_pos, lambda_gain, lambda_offset):
    n_sensors = sensor_positions.shape[0]
    if len(offset_initial_values) != n_sensors:
        raise ValueError(f"Expected {n_sensors} initial offsets, got {len(offset_initial_values)}.")
    results, rmses = [], []
    for i in range(n_sensors):
        params, rmse = calibrate_single_sensor(
            sensor_index=i, sensor_pos_init=sensor_positions[i],
            robot_positions=robot_positions, m_world=m_world,
            voltage_sensor=voltage_data[:, i], offset_init=offset_initial_values[i],
            lambda_pos=lambda_pos, lambda_gain=lambda_gain, lambda_offset=lambda_offset,
        )
        results.append(params)
        rmses.append(rmse)
    return np.array(results), np.array(rmses)


def select_stage1_stage2_split(robot_positions, m_world, voltage_data, n_total, n_stage1):
    n_samples = robot_positions.shape[0]
    if n_total > n_samples:
        raise ValueError(f"Requested {n_total} calibration points but dataset only has {n_samples} samples.")
    all_idx = np.random.choice(n_samples, size=n_total, replace=False)
    stage1_idx = all_idx[:n_stage1]
    stage2_idx = all_idx[n_stage1:]
    print(f"\n[Sampling] Drew {n_total} random points out of {n_samples} total input samples.")
    print(f"  Stage 1 (physical param fit): {len(stage1_idx)} points")
    print(f"  Stage 2 (alpha(r) fit):       {len(stage2_idx)} points")
    rp1, mw1, vd1 = robot_positions[stage1_idx], m_world[stage1_idx], voltage_data[stage1_idx]
    rp2, mw2, vd2 = robot_positions[stage2_idx], m_world[stage2_idx], voltage_data[stage2_idx]
    return (stage1_idx, rp1, mw1, vd1), (stage2_idx, rp2, mw2, vd2)


# =============================================================================
# STAGE 2 -- data split 
# =============================================================================

def split_stage2_data(rp, mw, vd, train_fraction, val_fraction, test_fraction, seed):
    if abs(train_fraction + val_fraction + test_fraction - 1.0) > 1e-6:
        raise ValueError("train_fraction + val_fraction + test_fraction must sum to 1.0")
    n = len(rp)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    n_test = int(round(test_fraction * n))
    n_val = int(round(val_fraction * n))
    test_idx = indices[:n_test]
    val_idx = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]
    # print("\n[Stage 2 split] (70/15/15 of the Stage-2 pool)")
    print(f"  Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}")
    return (
        (rp[train_idx], mw[train_idx], vd[train_idx]),
        (rp[val_idx], mw[val_idx], vd[val_idx]),
        (rp[test_idx], mw[test_idx], vd[test_idx]),
    )


# =============================================================================
# STAGE 2 -- distance normalization 
# =============================================================================

def compute_distance_stats(sensor_pos_np: np.ndarray, rp_train: np.ndarray):
    r_vec = sensor_pos_np[None, :, :] - rp_train[:, None, :]      # (n_train, 64, 3)
    r_mat = np.linalg.norm(r_vec, axis=-1)                        # (n_train, 64)
    return float(r_mat.mean()), float(max(r_mat.std(), 1e-8))


# =============================================================================
# STAGE 2 -- alpha(r) models
# =============================================================================

# =============================================================================
# LINEAR BASELINE — temporarily disabled
# =============================================================================
# The linear alpha(r) model is intentionally commented out for this experiment.
# We train and evaluate only the nonlinear NN alpha(r) model.
#
# class SensorwiseLinearAlpha(nn.Module):
#     ...
#
class MLPCalibration(nn.Module):
    """alpha_i(r) = 1 + f_theta(r), ONE shared scalar MLP applied per sensor.

    Reshape (B, 64) -> (B*64, 1) -> f_theta -> (B*64, 1) -> (B, 64) means
    delta_alpha_i can only ever be a function of r_i -- there is no path for
    information from r_j (j != i) to reach output i. Verified numerically in
    run_sanity_checks().
    """

    def __init__(self, hidden_dim: int = 32, n_hidden_layers: int = 2, output_scale_init: float = 0.05):
        super().__init__()
        layers = []
        in_dim = 1
        for _ in range(n_hidden_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
            in_dim = hidden_dim
        layers += [nn.Linear(in_dim, 1)]
        self.net = nn.Sequential(*layers)

        # Zero-init the last layer -> f_theta(r) == 0 everywhere at init -> alpha == 1.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

        self.output_scale = nn.Parameter(torch.tensor(float(output_scale_init)))

        # Distance normalization stats, fit on TRAIN split only (see
        # compute_distance_stats), stored as buffers so they travel with
        # .to(device) and are saved/loaded automatically via state_dict().
        self.register_buffer("r_mean", torch.tensor(0.0))
        self.register_buffer("r_std", torch.tensor(1.0))

    def set_normalization(self, r_mean: float, r_std: float) -> None:
        self.r_mean.fill_(r_mean)
        self.r_std.fill_(r_std)

    def forward(self, r_raw: torch.Tensor) -> torch.Tensor:
        shape = r_raw.shape                                  # (B, n_sensors)
        r_norm = (r_raw - self.r_mean) / self.r_std
        flat = r_norm.reshape(-1, 1)                          # (B*n_sensors, 1)
        delta_flat = self.net(flat) * self.output_scale       # (B*n_sensors, 1)
        return delta_flat.reshape(shape)                      # (B, n_sensors)

    def alpha_curve(self, sensor_index: int, r_values: np.ndarray) -> np.ndarray:
        # sensor_index is irrelevant here -- the function is SHARED across
        # sensors by construction -- kept as an arg only so callers can treat
        # both model types uniformly.
        with torch.no_grad():
            r_t = torch.tensor(r_values, dtype=torch.float32).unsqueeze(0)  # (1, len(r_values))
            delta = self.forward(r_t).squeeze(0).cpu().numpy()
        return 1.0 + delta


def build_model(cfg: dict, output_scale_init: float = 0.05) -> nn.Module:
    """Build the Stage-2 nonlinear NN alpha(r) model only."""
    return MLPCalibration(
        hidden_dim=cfg["mlp_hidden_dim"],
        n_hidden_layers=cfg["mlp_n_hidden_layers"],
        output_scale_init=output_scale_init,
    )


# =============================================================================
# STAGE 2 -- dipole model 
# =============================================================================

def torch_dipole_field(r_vec, m_vec):
    r = torch.linalg.norm(r_vec, dim=-1, keepdim=True)
    r3 = torch.clamp(r ** 3, min=1e-12)
    r5 = torch.clamp(r ** 5, min=1e-12)
    mdotr = torch.sum(m_vec * r_vec, dim=-1, keepdim=True)
    return MU0_OVER_4PI * (3.0 * r_vec * mdotr / r5 - m_vec / r3)


def check_torch_numpy_dipole_consistency(sensor_pos_np, rp_probe, mw_probe, rtol=1e-5, atol=1e-9):
    n_sensors = sensor_pos_np.shape[0]
    B_numpy = np.zeros((rp_probe.shape[0], n_sensors, 3), dtype=np.float64)
    for s in range(n_sensors):
        r_vec = sensor_pos_np[s] - rp_probe
        B_numpy[:, s, :] = dipole_field(r_vec, mw_probe)
    sensor_pos_t = torch.tensor(sensor_pos_np, dtype=torch.float64)
    rp_t = torch.tensor(rp_probe, dtype=torch.float64)
    mw_t = torch.tensor(mw_probe, dtype=torch.float64)
    r_vec_t = sensor_pos_t.unsqueeze(0) - rp_t.unsqueeze(1)
    mw_expanded = mw_t.unsqueeze(1).expand_as(r_vec_t)
    B_torch = torch_dipole_field(r_vec_t, mw_expanded).numpy()
    max_abs_diff = np.max(np.abs(B_torch - B_numpy))
    ok = np.allclose(B_torch, B_numpy, rtol=rtol, atol=atol)
    print(f"\n[Consistency check] max |B_torch - B_numpy| = {max_abs_diff:.3e} T | within tolerance: {ok}")
    if not ok:
        raise AssertionError(
            f"Torch Stage-2 dipole field disagrees with the NumPy Stage-1 dipole field "
            f"(max abs diff = {max_abs_diff:.3e} T)."
        )
    return max_abs_diff


def build_frozen_physics(physical_results, device):
    return {
        "sensor_pos": torch.tensor(physical_results[:, 0:3], dtype=torch.float32, device=device),
        "offset": torch.tensor(physical_results[:, 3], dtype=torch.float32, device=device),
        "gain": torch.tensor(physical_results[:, 4], dtype=torch.float32, device=device),
        "sensor_dir": torch.tensor(physical_results[:, 5:8], dtype=torch.float32, device=device),
    }


def forward_voltage_conditioned(model, pose, moment, frozen_phys):
    """model: any module with forward(r_raw: (B,64)) -> delta_alpha: (B,64)."""
    sensor_pos = frozen_phys["sensor_pos"]
    r_vec = sensor_pos.unsqueeze(0) - pose.unsqueeze(1)                  # (B, 64, 3)
    r_raw = torch.linalg.norm(r_vec, dim=-1)                             # (B, 64)
    moment_expanded = moment.unsqueeze(1).expand_as(r_vec)
    B = torch_dipole_field(r_vec, moment_expanded)
    b_sensor = torch.sum(B * frozen_phys["sensor_dir"].unsqueeze(0), dim=-1)   # (B, 64)

    delta_alpha = model(r_raw)                                          # (B, 64) -- ONLY r as input
    alpha = 1.0 + delta_alpha
    voltage_pred = frozen_phys["offset"].unsqueeze(0) + frozen_phys["gain"].unsqueeze(0) * b_sensor * alpha
    return voltage_pred, delta_alpha, alpha, b_sensor, r_raw


# =============================================================================
# STAGE 2 -- loss
# =============================================================================

def huber_voltage_loss(voltage_pred, voltage_raw, delta):
    error = voltage_pred - voltage_raw
    abs_error = torch.abs(error)
    quadratic = torch.minimum(abs_error, torch.tensor(delta, device=error.device, dtype=error.dtype))
    linear = abs_error - quadratic
    return (0.5 * quadratic ** 2 + delta * linear).mean()


def stage2_loss(voltage_pred, voltage_raw, delta_alpha, lambda_alpha, huber_delta_v):
    loss_voltage = huber_voltage_loss(voltage_pred, voltage_raw, huber_delta_v)
    loss_alpha = torch.mean(delta_alpha ** 2)
    total_loss = loss_voltage + lambda_alpha * loss_alpha
    return total_loss, loss_voltage, loss_alpha


# =============================================================================
# DATALOADER
# =============================================================================

def make_loader(rp, mw, vd, batch_size, shuffle=False):
    dataset = TensorDataset(
        torch.tensor(rp, dtype=torch.float32),
        torch.tensor(mw, dtype=torch.float32),
        torch.tensor(vd, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


# =============================================================================
# ONE EPOCH
# =============================================================================

def run_epoch(model, loader, optimizer, frozen_phys, lambda_alpha, huber_delta_v, device, training):
    model.train(training)
    total = total_voltage = total_alpha = 0.0
    n = 0
    for pose, moment, voltage_raw in loader:
        pose, moment, voltage_raw = pose.to(device), moment.to(device), voltage_raw.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            voltage_pred, delta_alpha, _, _, _ = forward_voltage_conditioned(model, pose, moment, frozen_phys)
            loss, loss_voltage, loss_alpha = stage2_loss(voltage_pred, voltage_raw, delta_alpha,
                                                          lambda_alpha, huber_delta_v)
            if training:
                loss.backward()
                optimizer.step()
        batch_n = voltage_raw.shape[0]
        total += loss.item() * batch_n
        total_voltage += loss_voltage.item() * batch_n
        total_alpha += loss_alpha.item() * batch_n
        n += batch_n
    return {"loss": total / n, "voltage_loss": total_voltage / n, "alpha_reg": total_alpha / n}


def compute_voltage_rmse(model, data, frozen_phys, device) -> float:
    rp, mw, vd = data
    with torch.no_grad():
        pose = torch.tensor(rp, dtype=torch.float32, device=device)
        moment = torch.tensor(mw, dtype=torch.float32, device=device)
        voltage_raw = torch.tensor(vd, dtype=torch.float32, device=device)
        model.eval()
        voltage_pred, _, _, _, _ = forward_voltage_conditioned(model, pose, moment, frozen_phys)
        return torch.sqrt(torch.mean((voltage_pred - voltage_raw) ** 2)).item()


# =============================================================================
# TRAIN LOOP -- early-stop (model selection) or fixed-epoch
# =============================================================================

def train_alpha_model(model, train_loader, val_data, frozen_phys,
                       lr, weight_decay, lambda_alpha, huber_delta_v, device,
                       max_epochs, patience, min_delta, batch_size,
                       verbose=False, early_stop=True):
    val_loader = make_loader(*val_data, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val, best_state, best_epoch, epochs_no_improve = float("inf"), None, -1, 0
    val_rmse = float("inf")
    history = []

    for epoch in range(max_epochs):
        train_metrics = run_epoch(model, train_loader, optimizer, frozen_phys,
                                   lambda_alpha, huber_delta_v, device, training=True)
        run_epoch(model, val_loader, None, frozen_phys, lambda_alpha, huber_delta_v, device, training=False)
        val_rmse = compute_voltage_rmse(model, val_data, frozen_phys, device)
        history.append({"epoch": epoch, "train_loss": train_metrics["loss"], "val_rmse": val_rmse})

        if verbose and epoch % 2 == 0:
            tag = "val" if early_stop else "val(monitor only)"
            print(f"  epoch {epoch:4d} | train loss = {train_metrics['loss']:.6e} | {tag} RMSE = {val_rmse:.6f} V")

        if not early_stop:
            continue
        if val_rmse < best_val - min_delta:
            best_val, best_epoch = val_rmse, epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if early_stop:
        if best_state is not None:
            model.load_state_dict(best_state)
        return model, best_val, best_epoch, history
    return model, val_rmse, max_epochs - 1, history


# =============================================================================
# OPTUNA -- training hyperparameters only 
# =============================================================================

def optuna_objective(trial, train_data, val_data, sensor_pos_np, frozen_phys, cfg, device):
    """Optuna objective for NN training hyperparameters only."""
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    lambda_alpha = trial.suggest_float("lambda_alpha", 1e-6, 1e-1, log=True)
    output_scale_init = trial.suggest_float("output_scale_init", 1e-3, 0.3, log=True)

    set_seed(cfg["seed"] + trial.number)

    rp_tr, mw_tr, vd_tr = train_data
    train_loader = make_loader(
        rp_tr, mw_tr, vd_tr,
        batch_size=cfg["batch_size"],
        shuffle=True,
    )

    model = build_model(cfg, output_scale_init).to(device)

    r_mean, r_std = compute_distance_stats(sensor_pos_np, rp_tr)
    model.set_normalization(r_mean, r_std)

    _, val_rmse, best_epoch, _ = train_alpha_model(
        model, train_loader, val_data, frozen_phys,
        lr=lr,
        weight_decay=weight_decay,
        lambda_alpha=lambda_alpha,
        huber_delta_v=cfg["huber_delta_v"],
        device=device,
        max_epochs=cfg["max_epochs"],
        patience=cfg["patience"],
        min_delta=cfg["min_delta"],
        batch_size=cfg["batch_size"],
        early_stop=True,
    )

    trial.set_user_attr("best_epoch", best_epoch)
    return val_rmse


def calibrate_stage2_model(train_data, val_data, test_data,
                           physical_results, cfg, device):
    """Train/select/evaluate the Stage-2 nonlinear NN alpha(r) model only.

    The held-out test set is used only once for the honest final evaluation.
    """
    rp_train, mw_train, vd_train = train_data
    sensor_pos_np = physical_results[:, 0:3]
    frozen_phys = build_frozen_physics(physical_results, device)
    frozen_snapshot = {k: v.detach().clone() for k, v in frozen_phys.items()}

    print("\n===================================")
    print("STAGE 2 [NN]: OPTUNA HYPERPARAMETER SEARCH")
    print("===================================")
    print(
        f"Architecture (fixed): 1 -> {cfg['mlp_hidden_dim']} -> SiLU "
        f"(x{cfg['mlp_n_hidden_layers']}) -> 1, shared independently across "
        f"all {N_SENSORS} sensors"
    )
    print(
        f"{cfg['n_trials']} trials over "
        "[lr, weight_decay, lambda_alpha, output_scale_init] | "
        f"train={len(rp_train)} | val={len(val_data[0])} | test={len(test_data[0])}"
    )

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=cfg["seed"]),
    )
    study.optimize(
        lambda trial: optuna_objective(
            trial, train_data, val_data, sensor_pos_np,
            frozen_phys, cfg, device
        ),
        n_trials=cfg["n_trials"],
        show_progress_bar=False,
    )

    print(f"\n[Optuna/NN] Best val RMSE = {study.best_value:.6f} V")
    print(f"[Optuna/NN] Best params = {study.best_params}")

    best = study.best_params
    best_epoch = study.best_trial.user_attrs["best_epoch"]
    output_scale_init = best["output_scale_init"]

    # Train the selected model on TRAIN only for model selection.
    set_seed(cfg["seed"])
    train_loader = make_loader(
        rp_train, mw_train, vd_train,
        batch_size=cfg["batch_size"],
        shuffle=True,
    )

    selection_model = build_model(cfg, output_scale_init).to(device)
    r_mean, r_std = compute_distance_stats(sensor_pos_np, rp_train)
    selection_model.set_normalization(r_mean, r_std)

    selection_model, selection_val_rmse, _, history = train_alpha_model(
        selection_model,
        train_loader,
        val_data,
        frozen_phys,
        lr=best["lr"],
        weight_decay=best["weight_decay"],
        lambda_alpha=best["lambda_alpha"],
        huber_delta_v=cfg["huber_delta_v"],
        device=device,
        max_epochs=cfg["max_epochs"],
        patience=cfg["patience"],
        min_delta=cfg["min_delta"],
        batch_size=cfg["batch_size"],
        early_stop=True,
        verbose=True,
    )

    # Honest held-out test evaluation.
    test_rmse = compute_voltage_rmse(
        selection_model, test_data, frozen_phys, device
    )
    print(
        f"\n[Stage 2 / NN] HONEST held-out TEST RMSE "
        f"(never used for Optuna or early stopping) = {test_rmse:.6f} V"
    )

    rp_val, mw_val, vd_val = val_data
    rp_te, mw_te, vd_te = test_data

    rp_full = np.concatenate([rp_train, rp_val, rp_te])
    mw_full = np.concatenate([mw_train, mw_val, mw_te])
    vd_full = np.concatenate([vd_train, vd_val, vd_te])

    set_seed(cfg["seed"])
    loader_full = make_loader(
        rp_full, mw_full, vd_full,
        batch_size=cfg["batch_size"],
        shuffle=True,
    )

    deploy_model = build_model(cfg, output_scale_init).to(device)
    r_mean_full, r_std_full = compute_distance_stats(sensor_pos_np, rp_full)
    deploy_model.set_normalization(r_mean_full, r_std_full)

    fixed_epochs = best_epoch + 1 if best_epoch >= 0 else cfg["max_epochs"]

    deploy_model, _, _, _ = train_alpha_model(
        deploy_model,
        loader_full,
        val_data,
        frozen_phys,
        lr=best["lr"],
        weight_decay=best["weight_decay"],
        lambda_alpha=best["lambda_alpha"],
        huber_delta_v=cfg["huber_delta_v"],
        device=device,
        max_epochs=fixed_epochs,
        patience=cfg["patience"],
        min_delta=cfg["min_delta"],
        batch_size=cfg["batch_size"],
        early_stop=False,
        verbose=True,
    )

    # Verify that Stage-1 physical parameters remained frozen.
    for name, tensor in frozen_phys.items():
        assert torch.equal(
            tensor, frozen_snapshot[name]
        ), f"Frozen Stage-1 tensor {name} changed during Stage 2!"

    checkpoint = {
        "kind": "nn",
        "model_state_dict": deploy_model.state_dict(),
        "mlp_hidden_dim": cfg["mlp_hidden_dim"],
        "mlp_n_hidden_layers": cfg["mlp_n_hidden_layers"],
        "output_scale_init": output_scale_init,
        "lambda_alpha": best["lambda_alpha"],
        "lr": best["lr"],
        "weight_decay": best["weight_decay"],
        "huber_delta_v": cfg["huber_delta_v"],
        "best_epoch": int(best_epoch),
        "selection_val_rmse": selection_val_rmse,
        "held_out_test_rmse": test_rmse,
        "deploy_fixed_epochs": fixed_epochs,
        "optuna_best_params": best,
        "seed": cfg["seed"],
    }

    return {
        "selection_model": selection_model,
        "deploy_model": deploy_model,
        "frozen_phys": frozen_phys,
        "checkpoint": checkpoint,
        "history": history,
        "val_rmse": selection_val_rmse,
        "test_rmse": test_rmse,
    }


# =============================================================================
# EVALUATION -- overall metrics + per-sensor RMSE
# =============================================================================

def evaluate_stage2(model, data, frozen_phys, lambda_alpha, huber_delta_v, device, split_name):
    rp, mw, vd = data
    loader = make_loader(rp, mw, vd, batch_size=64, shuffle=False)
    metrics = run_epoch(model, loader, None, frozen_phys, lambda_alpha, huber_delta_v, device, training=False)
    model.eval()

    all_delta_alpha, all_voltage_pred = [], []
    with torch.no_grad():
        for pose, moment, voltage_raw in loader:
            pose, moment, voltage_raw = pose.to(device), moment.to(device), voltage_raw.to(device)
            voltage_pred, delta_alpha, _, _, _ = forward_voltage_conditioned(model, pose, moment, frozen_phys)
            all_delta_alpha.append(delta_alpha.cpu().numpy())
            all_voltage_pred.append(voltage_pred.cpu().numpy())

    delta_alpha_np = np.concatenate(all_delta_alpha, axis=0)
    voltage_pred_np = np.concatenate(all_voltage_pred, axis=0)
    voltage_error = voltage_pred_np - vd
    alpha_np = 1.0 + delta_alpha_np

    sensor_rmse = np.sqrt(np.mean(voltage_error ** 2, axis=0))   # (64,)
    voltage_rmse = np.sqrt(np.mean(voltage_error ** 2))
    voltage_mae = np.mean(np.abs(voltage_error))

    print(f"\n----------------------------------- {split_name}")
    print(f"Voltage RMSE       : {voltage_rmse:.8e} V")
    print(f"Voltage MAE        : {voltage_mae:.8e} V")
    print(f"Mean |delta_alpha| : {np.mean(np.abs(delta_alpha_np)):.8e}")
    print(f"Max  |delta_alpha| : {np.max(np.abs(delta_alpha_np)):.8e}")
    print(f"Alpha stats        : mean={alpha_np.mean():.6f} std={alpha_np.std():.6f} "
          f"min={alpha_np.min():.6f} max={alpha_np.max():.6f}")

    return {
        "loss": metrics["loss"], "voltage_loss": metrics["voltage_loss"],
        "voltage_rmse": float(voltage_rmse), "voltage_mae": float(voltage_mae),
        "mean_abs_delta_alpha": float(np.mean(np.abs(delta_alpha_np))),
        "max_abs_delta_alpha": float(np.max(np.abs(delta_alpha_np))),
        "alpha_mean": float(alpha_np.mean()), "alpha_std": float(alpha_np.std()),
        "alpha_min": float(alpha_np.min()), "alpha_max": float(alpha_np.max()),
        "sensor_rmse": sensor_rmse,  # (64,) numpy array
    }


# =============================================================================
# SANITY CHECKS
# =============================================================================

def run_sanity_checks(model, frozen_phys, frozen_snapshot, sample_data, device, kind: str):
    rp, mw, vd = sample_data
    pose = torch.tensor(rp[:1], dtype=torch.float32, device=device, requires_grad=True)
    moment = torch.tensor(mw[:1], dtype=torch.float32, device=device, requires_grad=True)

    model.eval()
    voltage_pred, delta_alpha, alpha, b_sensor, r_raw = forward_voltage_conditioned(model, pose, moment, frozen_phys)

    print(f"\n===================================")
    print(f"STAGE 2 [{kind.upper()}] SANITY CHECKS")
    print(f"===================================")

    assert delta_alpha.shape == (1, N_SENSORS)
    assert voltage_pred.shape == (1, N_SENSORS)
    print(f"delta_alpha shape : {tuple(delta_alpha.shape)}")

    for name, tensor in frozen_phys.items():
        unchanged = torch.equal(tensor, frozen_snapshot[name])
        print(f"Frozen '{name}' unchanged : {unchanged}")
        if not unchanged:
            raise RuntimeError(f"Frozen Stage-1 tensor '{name}' was modified!")

    alpha_ok = torch.allclose(alpha, 1.0 + delta_alpha)
    print(f"alpha == 1 + delta_alpha  : {alpha_ok}")
    if not alpha_ok:
        raise RuntimeError("alpha formula check failed.")

    expected_v = frozen_phys["offset"].unsqueeze(0) + frozen_phys["gain"].unsqueeze(0) * b_sensor * alpha
    v_pred_ok = torch.allclose(voltage_pred, expected_v)
    print(f"V_pred == a + g*Bz*alpha  : {v_pred_ok}")
    if not v_pred_ok:
        raise RuntimeError("V_pred formula check failed.")

    grad_pose, grad_moment = torch.autograd.grad(voltage_pred.sum(), [pose, moment], allow_unused=True)
    print(f"Pose gradient exists      : {grad_pose is not None}")
    print(f"Moment gradient exists    : {grad_moment is not None}")
    if grad_pose is None or not torch.isfinite(grad_pose).all():
        raise RuntimeError("Pose gradient check failed.")
    if grad_moment is None or not torch.isfinite(grad_moment).all():
        raise RuntimeError("Moment gradient check failed.")

    # No-cross-sensor-coupling check: d(delta_alpha_i)/d(r_j) must be ~0 for j != i.
    r_probe = torch.rand(1, N_SENSORS, requires_grad=True, device=device)
    delta_probe = model(r_probe)
    grad_r = torch.autograd.grad(delta_probe[0, 0], r_probe)[0]  # (1, 64)
    cross_leak = grad_r[0, 1:].abs().max().item()
    own_grad = grad_r[0, 0].abs().item()
    print(f"d(delta_alpha_0)/d(r_j!=0) max |grad| = {cross_leak:.3e} "
          f"(own d(delta_alpha_0)/d(r_0) = {own_grad:.3e}) -- should be ~0")
    if cross_leak > 1e-6:
        raise RuntimeError(f"Cross-sensor coupling detected in {kind} model: alpha_0 depends on r_j, j!=0.")

    print("All sanity checks passed.")


# =============================================================================
# SAVE / PLOT
# =============================================================================

def save_physical_results(results, output_file):
    df = pd.DataFrame({
        "sensor_index": np.arange(len(results)),
        "x": results[:, 0], "y": results[:, 1], "z": results[:, 2],
        "offset": results[:, 3], "gain": results[:, 4],
    })
    df.to_csv(output_file, index=False)
    print(f"\nSaved Stage-1 physical parameters: {output_file}")


def plot_stage1_rmse(rmses, output_path):
    plt.figure(figsize=(10, 5))
    plt.bar(np.arange(len(rmses)), rmses)
    plt.xlabel("Sensor Index"); plt.ylabel("RMSE"); plt.title("Stage-1 Calibration RMSE")
    plt.grid(True); plt.tight_layout()
    plt.savefig(output_path, dpi=130); plt.close()
    print(f"Saved: {output_path}")


def save_distance_scaler_json(nn_checkpoint, output_path):
    r_mean = nn_checkpoint["model_state_dict"]["r_mean"].item()
    r_std = nn_checkpoint["model_state_dict"]["r_std"].item()
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"r_mean": r_mean, "r_std": r_std}, f, indent=2)
    print(f"Saved: {output_path}")


def save_history_csv(history, output_path):
    pd.DataFrame(history).to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


def plot_training_validation_loss(history, output_path, title="Stage 2 [NN] Training"):
    df = pd.DataFrame(history)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(df["epoch"], df["train_loss"], label="train loss (Huber+reg)", color="#2b6cb0")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("train loss"); ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(df["epoch"], df["val_rmse"], label="val voltage RMSE", color="#c53030")
    ax2.set_ylabel("val voltage RMSE (V)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    plt.title(title); fig.tight_layout()
    plt.savefig(output_path, dpi=130); plt.close()
    print(f"Saved: {output_path}")


def save_sensor_metrics_csv(sensor_rmse_nn, output_path):
    df = pd.DataFrame({
        "sensor_index": np.arange(N_SENSORS),
        "test_rmse_nn": sensor_rmse_nn,
    })
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


def plot_sensor_rmse(sensor_rmse_nn, output_path):
    x = np.arange(N_SENSORS)
    plt.figure(figsize=(14, 5))
    plt.bar(x, sensor_rmse_nn)
    plt.xlabel("Sensor Index")
    plt.ylabel("Test voltage RMSE (V)")
    plt.title("Stage 2 test RMSE per sensor: NN alpha(r)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=130)
    plt.close()
    print(f"Saved: {output_path}")


def plot_alpha_vs_r_representative_sensors(
    nn_model,
    physical_results,
    test_data,
    sensor_indices,
    output_dir,
):
    """Plot empirical alpha_hat and the NN alpha(r) curve for selected sensors."""
    rp_te, mw_te, vd_te = test_data
    sensor_pos = physical_results[:, 0:3]
    offset = physical_results[:, 3]
    gain = physical_results[:, 4]

    # Empirical alpha_hat computed from the frozen Stage-1 physical model.
    r_vec = sensor_pos[None, :, :] - rp_te[:, None, :]
    r_all = np.linalg.norm(r_vec, axis=-1)

    B = np.zeros_like(r_all)
    for s in range(N_SENSORS):
        B[:, s] = dipole_field(
            sensor_pos[s] - rp_te, mw_te
        )[:, 2]

    with np.errstate(divide="ignore", invalid="ignore"):
        alpha_hat = (vd_te - offset[None, :]) / (gain[None, :] * B)

    valid = np.isfinite(alpha_hat) & (np.abs(B) > 1e-15)

    for s in sensor_indices:
        r_s = r_all[:, s]
        mask_s = valid[:, s]

        if not np.any(mask_s):
            continue

        r_range = np.linspace(r_s.min(), r_s.max(), 200)

        # NN prediction only. The linear alpha(r) baseline is intentionally
        # not used in this experiment.
        alpha_nn = nn_model.alpha_curve(s, r_range)

        plt.figure(figsize=(7, 5))
        plt.scatter(
            r_s[mask_s],
            alpha_hat[mask_s, s],
            s=6,
            alpha=0.15,
            color="gray",
            label="empirical alpha_hat (test set)",
        )
        plt.plot(
            r_range,
            alpha_nn,
            color="#c53030",
            lw=2,
            label="NN alpha(r)",
        )
        plt.axhline(
            1.0,
            color="black",
            linestyle="--",
            linewidth=0.8,
            label="alpha = 1",
        )
        plt.xlabel("r (m)")
        plt.ylabel("alpha")
        plt.title(f"Sensor {s:02d} -- NN alpha(r)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        out_path = output_dir / f"alpha_vs_r_sensor_{s:02d}.png"
        plt.savefig(out_path, dpi=130)
        plt.close()
        print(f"Saved: {out_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = build_arg_parser().parse_args()
    set_seed(args.seed)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    physical_output_path = output_dir / "physical_params.csv"
    nn_checkpoint_path = output_dir / "stage2_alpha_r_nn.pt"
    scaler_path = output_dir / "stage2_distance_scaler.json"
    history_path = output_dir / "stage2_history.csv"
    sensor_metrics_path = output_dir / "stage2_sensor_metrics.csv"
    summary_path = output_dir / "calibration_summary.json"
    stage1_rmse_plot_path = output_dir / "stage1_rmse.png"
    loss_plot_path = output_dir / "training_validation_loss.png"
    sensor_rmse_plot_path = output_dir / "sensor_rmse.png"

    cfg = {
        "seed": args.seed, "mlp_hidden_dim": args.mlp_hidden_dim,
        "mlp_n_hidden_layers": args.mlp_n_hidden_layers, "batch_size": args.batch_size,
        "n_trials": args.n_trials, "max_epochs": args.max_epochs, "patience": args.patience,
        "min_delta": args.min_delta, "huber_delta_v": args.huber_delta_v,
    }

    print("=" * 70)
    print("TWO-STAGE HALL SENSOR CALIBRATION  |  device =", device)
    print("=" * 70)
    print("Stage 1: physical calibration (least_squares) -- unchanged")
    print("Stage 2: r(64) -> alpha(64), nonlinear shared scalar-MLP NN only")

    sensor_positions = load_sensor_positions(args.sensor_positions)
    robot_positions, m_world = load_robot_pose(args.robot_pose)
    voltage_data = load_voltage_data(args.voltage)
    offset_initial_values = load_offset_initial_values(args.offset_init, n_sensors=sensor_positions.shape[0])

    if sensor_positions.shape[0] != N_SENSORS:
        raise ValueError(f"Expected {N_SENSORS} sensors, got {sensor_positions.shape[0]}.")
    if voltage_data.shape[1] != N_SENSORS:
        raise ValueError(f"Expected voltage_data to have {N_SENSORS} columns, got {voltage_data.shape[1]}.")

    n_samples = min(len(robot_positions), len(voltage_data))
    robot_positions, m_world, voltage_data = robot_positions[:n_samples], m_world[:n_samples], voltage_data[:n_samples]

    print("\n===================================")
    print(f"SAMPLING: {args.n_total_samples} random points -> "
          f"{args.n_stage1_samples} Stage 1 / {args.n_total_samples - args.n_stage1_samples} Stage 2")
    print("===================================")
    stage1_split, stage2_split = select_stage1_stage2_split(
        robot_positions, m_world, voltage_data, n_total=args.n_total_samples, n_stage1=args.n_stage1_samples)
    _, rp1, mw1, vd1 = stage1_split
    _, rp2, mw2, vd2 = stage2_split

    print("\n===================================")
    print("STAGE 1: PHYSICAL PARAMETER FIT")
    print("===================================")
    results, rmses = run_calibration(
        sensor_positions, rp1, mw1, vd1, offset_initial_values=offset_initial_values,
        lambda_pos=args.lambda_pos, lambda_gain=args.lambda_gain, lambda_offset=args.lambda_offset,
    )
    print(f"\nMean RMSE = {np.mean(rmses):.6f} | Max RMSE = {np.max(rmses):.6f} | Min RMSE = {np.min(rmses):.6f}")
    save_physical_results(results, physical_output_path)
    plot_stage1_rmse(rmses, stage1_rmse_plot_path)

    train_data, val_data, test_data = split_stage2_data(
        rp2, mw2, vd2, train_fraction=args.train_fraction, val_fraction=args.val_fraction,
        test_fraction=args.test_fraction, seed=args.seed)

    # ---- consistency check before any Stage-2 training ----
    n_probe = min(50, len(train_data[0]))
    check_torch_numpy_dipole_consistency(results[:, 0:3], train_data[0][:n_probe], train_data[1][:n_probe])

    # ---- NN only ----
    nn_result = calibrate_stage2_model(
        train_data, val_data, test_data, results, cfg, device
    )

    nn_snapshot = {
        k: v.detach().clone()
        for k, v in nn_result["frozen_phys"].items()
    }

    run_sanity_checks(
        nn_result["selection_model"],
        nn_result["frozen_phys"],
        nn_snapshot,
        train_data,
        device,
        kind="nn",
    )

    nn_val_metrics = evaluate_stage2(
        nn_result["selection_model"],
        val_data,
        nn_result["frozen_phys"],
        nn_result["checkpoint"]["lambda_alpha"],
        cfg["huber_delta_v"],
        device,
        "VALIDATION [nn]",
    )

    nn_test_metrics = evaluate_stage2(
        nn_result["selection_model"],
        test_data,
        nn_result["frozen_phys"],
        nn_result["checkpoint"]["lambda_alpha"],
        cfg["huber_delta_v"],
        device,
        "TEST [nn, honest]",
    )

    torch.save(nn_result["checkpoint"], nn_checkpoint_path)
    print(f"Saved: {nn_checkpoint_path}")

    save_distance_scaler_json(
        nn_result["checkpoint"], scaler_path
    )
    save_history_csv(
        nn_result["history"], history_path
    )
    plot_training_validation_loss(
        nn_result["history"], loss_plot_path
    )

    save_sensor_metrics_csv(
        nn_test_metrics["sensor_rmse"],
        sensor_metrics_path,
    )
    plot_sensor_rmse(
        nn_test_metrics["sensor_rmse"],
        sensor_rmse_plot_path,
    )

    representative = np.unique(
        np.linspace(
            0,
            N_SENSORS - 1,
            args.n_representative_sensors,
        ).astype(int)
    )

    plot_alpha_vs_r_representative_sensors(
        nn_result["selection_model"],
        results,
        test_data,
        representative,
        output_dir,
    )

    summary = {
        "stage1": {
            "n_samples": int(len(rp1)), "mean_rmse": float(np.mean(rmses)),
            "max_rmse": float(np.max(rmses)), "min_rmse": float(np.min(rmses)),
        },
        "stage2": {
            "n_samples": int(len(rp2)),
            "train_samples": int(len(train_data[0])),
            "val_samples": int(len(val_data[0])),
            "test_samples": int(len(test_data[0])),
            "nn": {
                "architecture": (
                    f"1 -> {cfg['mlp_hidden_dim']} -> "
                    f"SiLU (x{cfg['mlp_n_hidden_layers']}) -> 1, shared"
                ),
                "optuna_best_params": nn_result["checkpoint"]["optuna_best_params"],
                "best_epoch": nn_result["checkpoint"]["best_epoch"],
                "val_voltage_rmse": nn_val_metrics["voltage_rmse"],
                "test_voltage_rmse": nn_test_metrics["voltage_rmse"],
                "test_voltage_mae": nn_test_metrics["voltage_mae"],
                "test_mean_abs_delta_alpha": nn_test_metrics["mean_abs_delta_alpha"],
                "test_max_abs_delta_alpha": nn_test_metrics["max_abs_delta_alpha"],
                "test_alpha_mean": nn_test_metrics["alpha_mean"],
                "test_alpha_std": nn_test_metrics["alpha_std"],
                "test_alpha_min": nn_test_metrics["alpha_min"],
                "test_alpha_max": nn_test_metrics["alpha_max"],
            },
        },
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n===================================")
    print("ALL STAGES FINISHED")
    print("===================================")
    print(f"Summary saved to: {summary_path}")
    print(f"NN     test RMSE : {nn_test_metrics['voltage_rmse']:.6f} V")
    print(f"\nOutputs written to: {output_dir}")


if __name__ == "__main__":
    main()