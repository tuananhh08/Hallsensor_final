"""
    !python calib_residual_vraw_colab.py \
        --sensor_positions "/content/drive/MyDrive/Dataset/Hall_sensor_positions.csv" \
        --robot_pose       "/content/drive/MyDrive/Dataset/Grid_points_coordinates.csv" \
        --voltage          "/content/drive/MyDrive/Dataset/Grid_data.csv" \
        --offset_init      "/content/drive/MyDrive/Dataset/Offset_Sens.csv" \
        --output_dir       "/content/drive/MyDrive/Dataset/residual_nn_out" \
        --n_trials 30 \
        --max_epochs 20 \
        --patience 8
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
    sys.exit(
        "PyTorch is required. In a Colab cell, run:\n"
        "  !pip install torch --quiet\n"
        "before running this script."
    )

try:
    from scipy.optimize import least_squares
except ImportError:
    sys.exit("scipy is required: !pip install scipy --quiet")

try:
    import optuna
except ImportError:
    sys.exit(
        "optuna is required for Stage-2 hyperparameter search.\n"
        "  !pip install optuna --quiet"
    )


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
        description="2-stage Hall-sensor calibration: Stage 1 physical "
                     "params (least_squares) + Stage 2 residual NN on "
                     "V_raw (64->64 delta_alpha), Optuna-tuned training "
                     "hyperparameters, fixed architecture.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- input files ----
    p.add_argument("--sensor_positions", required=True,
                    help="CSV of nominal/design sensor positions (Hall_sensor_positions.csv).")
    p.add_argument("--robot_pose", required=True,
                    help="CSV of capsule positions + magnetic moment orientation "
                         "(columns: x,y,z,mx,my,mz).")
    p.add_argument("--voltage", required=True,
                    help="CSV of measured voltages, one column per sensor, "
                         "row-aligned with --robot_pose.")
    p.add_argument("--offset_init", required=True,
                    help="CSV of per-sensor initial offset guesses "
                         "(columns: sensor_index, offset_a_V).")

    # ---- outputs ----
    p.add_argument("--output_dir", required=True,
                    help="Directory to write all outputs (calibration CSVs, "
                         "Stage-2 checkpoint, history, summary, plots). "
                         "Created if it doesn't exist.")

    # ---- Stage 1 / Stage 2 split sizes ----
    p.add_argument("--n_total_samples", type=int, default=2000,
                    help="Total number of (robot_pose, voltage) rows drawn "
                         "at random for calibration (Stage1 + Stage2 pool).")
    p.add_argument("--n_stage1_samples", type=int, default=400,
                    help="How many of --n_total_samples go to Stage 1 "
                         "(physical parameter fit). The rest form the "
                         "Stage 2 pool (train/val/test).")
    p.add_argument("--val_fraction", type=float, default=0.15,
                    help="Fraction of the (Stage-2 pool minus test) used "
                         "for Optuna/early-stopping validation.")
    p.add_argument("--test_fraction", type=float, default=0.1,
                    help="Fraction of the Stage-2 pool held out as an "
                         "untouched final test set (never used by Optuna "
                         "or early stopping).")

    # ---- Stage 1 regularization weights (physical priors) ----
    p.add_argument("--lambda_pos", type=float, default=2000,
                    help="Stage 1 ridge weight pulling (x,y,z) toward the "
                         "nominal design position.")
    p.add_argument("--lambda_gain", type=float, default=9e-3,
                    help="Stage 1 ridge weight pulling gain toward g0=7.5.")
    p.add_argument("--lambda_offset", type=float, default=750,
                    help="Stage 1 ridge weight pulling offset toward its "
                         "--offset_init value.")

    # ---- Stage 2 fixed architecture ----
    p.add_argument("--hidden_dim", type=int, default=128,
                    help="Fixed hidden width: 64 -> hidden_dim -> "
                         "[n_residual_blocks residual blocks] -> 64.")
    p.add_argument("--n_residual_blocks", type=int, default=3,
                    help="Fixed number of residual blocks (each hidden_dim -> hidden_dim).")
    p.add_argument("--batch_size", type=int, default=64,
                    help="Fixed training batch size (not tuned by Optuna).")

    # ---- Stage 2 Optuna: training hyperparameters only ----
    p.add_argument("--n_trials", type=int, default=35,
                    help="Number of Optuna trials over [lr, weight_decay, "
                         "lambda_alpha, output_scale_init].")
    p.add_argument("--max_epochs", type=int, default=200,
                    help="Max training epochs per Optuna trial / model-selection run.")
    p.add_argument("--patience", type=int, default=20,
                    help="Early-stopping patience (epochs without val RMSE improvement).")
    p.add_argument("--min_delta", type=float, default=1e-8,
                    help="Minimum val RMSE improvement to reset early-stopping patience.")
    p.add_argument("--huber_delta_v", type=float, default=1e-3,
                    help="Huber loss delta, in volts, for the voltage-domain loss.")

    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--device", default=None,
                    help="Force 'cuda' or 'cpu'. Default: auto-detect.")

    return p


# =============================================================================
# DIPOLE MODEL 
# =============================================================================

def dipole_field(r_vec, m_vec):
    r = np.linalg.norm(r_vec, axis=1, keepdims=True)

    r3 = np.maximum(r**3, 1e-12)
    r5 = np.maximum(r**5, 1e-12)

    mdotr = np.sum(m_vec * r_vec, axis=1, keepdims=True)

    B = MU0_OVER_4PI * (
        3.0 * r_vec * mdotr / r5 - m_vec / r3
    )

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

    required_cols = ['x', 'y', 'z', 'mx', 'my', 'mz']
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    positions = df[['x', 'y', 'z']].values
    m_world = df[['mx', 'my', 'mz']].values

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
        raise ValueError(
            f"Missing column(s) in {Path(file_path).name}: {sorted(missing_columns)}"
        )

    if df["sensor_index"].duplicated().any():
        raise ValueError("offset_init CSV contains duplicate sensor_index values.")

    df = df.sort_values("sensor_index").reset_index(drop=True)

    expected_indices = np.arange(n_sensors)
    actual_indices = df["sensor_index"].to_numpy()

    if not np.array_equal(actual_indices, expected_indices):
        raise ValueError(
            "offset_init CSV must contain exactly sensor_index values "
            f"0 to {n_sensors - 1}."
        )

    offset_initial_values = df["offset_a_V"].to_numpy(dtype=float)

    if not np.isfinite(offset_initial_values).all():
        raise ValueError("Column offset_a_V contains missing or non-finite values.")

    print(f"Loaded per-sensor offset initial values: {offset_initial_values.shape}")
    
    return offset_initial_values


# =============================================================================
# STAGE 1 -- RESIDUALS LEAST SQUARES (Regularization)
# =============================================================================

def sensor_residuals(
    params, robot_positions, m_world, voltage_sensor,
    pos_prior, offset_prior, g0, lambda_pos, lambda_gain, lambda_offset,
):
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

    x0 = np.array([sensor_pos_init[0], sensor_pos_init[1], sensor_pos_init[2],
                    offset_init, g0])

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
        method='trf', max_nfev=250,
    )

    params_opt = result.x
    theta_opt, phi_opt = 0.0, 0.0
    nx, ny, nz = 0.0, 0.0, 1.0

    params_extended = np.array([
        params_opt[0], params_opt[1], params_opt[2], params_opt[3], params_opt[4],
        nx, ny, nz, theta_opt, phi_opt,
    ])

    n_voltage = voltage_sensor.shape[0]
    rmse = np.sqrt(np.mean(result.fun[:n_voltage]**2))
    print(f"Sensor {sensor_index+1:02d} | RMSE = {rmse:.6f}")

    return params_extended, rmse


def run_calibration(sensor_positions, robot_positions, m_world, voltage_data,
                     offset_initial_values, lambda_pos, lambda_gain, lambda_offset):
    n_sensors = sensor_positions.shape[0]

    if len(offset_initial_values) != n_sensors:
        raise ValueError(
            f"Expected {n_sensors} initial offsets, got {len(offset_initial_values)}."
        )

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


# =============================================================================
# STAGE 1 / STAGE 2 SPLIT -- RANDOM SAMPLING
# =============================================================================

def select_stage1_stage2_split(robot_positions, m_world, voltage_data,
                                n_total, n_stage1):
    n_samples = robot_positions.shape[0]

    if n_total > n_samples:
        raise ValueError(
            f"Requested {n_total} calibration points but dataset only has "
            f"{n_samples} samples."
        )

    all_idx = np.random.choice(n_samples, size=n_total, replace=False)

    stage1_idx = all_idx[:n_stage1]
    stage2_idx = all_idx[n_stage1:]

    print(f"\n[Sampling] Drew {n_total} random points out of "
          f"{n_samples} total input samples.")
    print(f"  Stage 1 (physical param fit): {len(stage1_idx)} points")
    print(f"  Stage 2 (NN alpha fit):       {len(stage2_idx)} points")

    rp1, mw1, vd1 = robot_positions[stage1_idx], m_world[stage1_idx], voltage_data[stage1_idx]
    rp2, mw2, vd2 = robot_positions[stage2_idx], m_world[stage2_idx], voltage_data[stage2_idx]

    return (stage1_idx, rp1, mw1, vd1), (stage2_idx, rp2, mw2, vd2)


# =============================================================================
# STAGE 2 -- NORMALIZATION (train-only)
# =============================================================================

def fit_voltage_normalization(voltage_train):
    mean = np.mean(voltage_train, axis=0)
    std = np.std(voltage_train, axis=0)
    std = np.maximum(std, 1e-8)
    
    return mean.astype(np.float32), std.astype(np.float32)


# =============================================================================
# STAGE 2 -- RESIDUAL BLOCK
# =============================================================================

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        residual = self.fc1(x)
        residual = self.act(residual)
        residual = self.fc2(residual)
        return x + residual


# =============================================================================
# STAGE 2 -- NN ARCHITECTURE 
# =============================================================================

class ResidualNN(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, n_blocks,
                 output_scale_init=0.05):
        super().__init__()

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.input_activation = nn.SiLU()

        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim) for _ in range(n_blocks)]
        )

        self.output_layer = nn.Linear(hidden_dim, output_dim)

        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

        self.output_scale = nn.Parameter(torch.tensor(float(output_scale_init)))

    def forward(self, voltage_normalized):
        x = self.input_layer(voltage_normalized)
        x = self.input_activation(x)
        x = self.blocks(x)
        return self.output_layer(x) * self.output_scale


# =============================================================================
# STAGE 2 -- TORCH DIPOLE
# =============================================================================

def torch_dipole_field(r_vec, m_vec):
    r = torch.linalg.norm(r_vec, dim=-1, keepdim=True)

    r3 = torch.clamp(r**3, min=1e-12)
    r5 = torch.clamp(r**5, min=1e-12)

    mdotr = torch.sum(m_vec * r_vec, dim=-1, keepdim=True)

    return MU0_OVER_4PI * (
        3.0 * r_vec * mdotr / r5 - m_vec / r3
    )


# =============================================================================
# NUMERICAL CONSISTENCY CHECK 
# =============================================================================

def check_torch_numpy_dipole_consistency(sensor_pos_np, rp_probe, mw_probe,
                                          rtol=1e-5, atol=1e-9):
    """Compare torch_dipole_field vs dipole_field on a probe batch.

    Raises AssertionError if they disagree beyond tolerance.
    """
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

    print("\n[Consistency check] Torch dipole vs NumPy dipole "
          f"(probe batch: {rp_probe.shape[0]} points x {n_sensors} sensors)")
    print(f"  max |B_torch - B_numpy| = {max_abs_diff:.3e} T")
    print(f"  within tolerance (rtol={rtol:.0e}, atol={atol:.0e}): {ok}")

    if not ok:
        raise AssertionError(
            "Torch Stage-2 dipole field disagrees with the NumPy Stage-1 "
            f"dipole field (max abs diff = {max_abs_diff:.3e} T). "
            "Stage 2 would silently train against different physics than "
            "Stage 1 was calibrated with -- fix before proceeding."
        )

    return max_abs_diff


# =============================================================================
# STAGE 2 PHYSICS FORWARD
# =============================================================================

def forward_voltage_conditioned(model, voltage_raw, pose, moment,
                                 frozen_phys, voltage_mean, voltage_std):
    voltage_normalized = (voltage_raw - voltage_mean) / voltage_std
    delta_alpha = model(voltage_normalized)
    alpha = 1.0 + delta_alpha

    sensor_pos = frozen_phys["sensor_pos"]
    sensor_dir = frozen_phys["sensor_dir"]
    offset = frozen_phys["offset"]
    gain = frozen_phys["gain"]

    r_vec = sensor_pos.unsqueeze(0) - pose.unsqueeze(1)
    moment_expanded = moment.unsqueeze(1).expand_as(r_vec)
    B = torch_dipole_field(r_vec, moment_expanded)
    b_sensor = torch.sum(B * sensor_dir.unsqueeze(0), dim=-1)

    voltage_pred = offset.unsqueeze(0) + gain.unsqueeze(0) * b_sensor * alpha

    return voltage_pred, delta_alpha, alpha, b_sensor


# =============================================================================
# STAGE 2 -- LOSS
# =============================================================================

def huber_voltage_loss(voltage_pred, voltage_raw, delta):
    error = voltage_pred - voltage_raw
    abs_error = torch.abs(error)
    quadratic = torch.minimum(
        abs_error, torch.tensor(delta, device=error.device, dtype=error.dtype))
    linear = abs_error - quadratic
    
    return (0.5 * quadratic**2 + delta * linear).mean()


def stage2_loss(voltage_pred, voltage_raw, delta_alpha, lambda_alpha, huber_delta_v):
    loss_voltage = huber_voltage_loss(voltage_pred, voltage_raw, huber_delta_v)
    loss_alpha = torch.mean(delta_alpha**2)
    total_loss = loss_voltage + lambda_alpha * loss_alpha
    
    return total_loss, loss_voltage, loss_alpha


# =============================================================================
# FROZEN PARAMETER AFTER STAGE-1
# =============================================================================

def build_frozen_physics(physical_results, device):
    return {
        "sensor_pos": torch.tensor(physical_results[:, 0:3], dtype=torch.float32, device=device),
        "offset": torch.tensor(physical_results[:, 3], dtype=torch.float32, device=device),
        "gain": torch.tensor(physical_results[:, 4], dtype=torch.float32, device=device),
        "sensor_dir": torch.tensor(physical_results[:, 5:8], dtype=torch.float32, device=device),
    }


# =============================================================================
# STAGE 2 -- TRAIN/VAL/TEST SPLIT
# =============================================================================

def split_stage2_data(rp, mw, vd, val_fraction, test_fraction, seed):
    n = len(rp)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)

    n_test = int(round(test_fraction * n))
    remaining = n - n_test
    n_val = int(round(val_fraction * remaining))

    test_idx = indices[:n_test]
    remaining_idx = indices[n_test:]

    val_idx = remaining_idx[:n_val]
    train_idx = remaining_idx[n_val:]

    print("\n[Stage 2 split]")
    print(f"  Train: {len(train_idx)}")
    print(f"  Val:   {len(val_idx)}")
    print(f"  Test:  {len(test_idx)}")

    return (
        (rp[train_idx], mw[train_idx], vd[train_idx]),
        (rp[val_idx], mw[val_idx], vd[val_idx]),
        (rp[test_idx], mw[test_idx], vd[test_idx]),
    )


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

def run_epoch(model, loader, optimizer, frozen_phys, voltage_mean, voltage_std,
              lambda_alpha, huber_delta_v, device, training):
    model.train(training)

    total = total_voltage = total_alpha = 0.0
    n = 0

    for pose, moment, voltage_raw in loader:
        pose, moment, voltage_raw = pose.to(device), moment.to(device), voltage_raw.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            voltage_pred, delta_alpha, _, _ = forward_voltage_conditioned(
                model, voltage_raw, pose, moment, frozen_phys, voltage_mean, voltage_std)

            loss, loss_voltage, loss_alpha = stage2_loss(
                voltage_pred, voltage_raw, delta_alpha, lambda_alpha, huber_delta_v)

            if training:
                loss.backward()
                optimizer.step()

        batch_n = voltage_raw.shape[0]
        total += loss.item() * batch_n
        total_voltage += loss_voltage.item() * batch_n
        total_alpha += loss_alpha.item() * batch_n
        n += batch_n

    return {"loss": total / n, "voltage_loss": total_voltage / n, "alpha_reg": total_alpha / n}


# =============================================================================
# TRAIN LOOP WITH EARLY-STOP / FIXED-EPOCH MODES 
# =============================================================================

def train_alpha_nn(model, train_loader, val_data, frozen_phys, voltage_mean, voltage_std,
                    lr, weight_decay, lambda_alpha, huber_delta_v, device,
                    max_epochs, patience, min_delta, batch_size,
                    verbose=False, early_stop=True):
    """
    early_stop=True  -> normal model-selection mode: monitor val RMSE, keep
                         the best checkpoint, stop after `patience` epochs
                         without improvement. Returns (model, best_val_rmse,
                         best_epoch).
    early_stop=False -> fixed-epoch "deploy" mode: run exactly max_epochs,
                         never checkpoint/stop against val (val is only used
                         to LOG progress here). Use this for the final
                         refit on train+val+test so that split can never
                         leak into model selection.
    """
    val_loader = make_loader(*val_data, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = float("inf")
    best_state = None
    best_epoch = -1
    epochs_no_improve = 0
    val_rmse = float("inf")

    for epoch in range(max_epochs):
        run_epoch(model, train_loader, optimizer, frozen_phys, voltage_mean, voltage_std,
                  lambda_alpha, huber_delta_v, device, training=True)

        run_epoch(model, val_loader, None, frozen_phys, voltage_mean, voltage_std,
                  lambda_alpha, huber_delta_v, device, training=False)

        with torch.no_grad():
            rp_v, mw_v, vd_v = val_data
            pose = torch.tensor(rp_v, dtype=torch.float32, device=device)
            moment = torch.tensor(mw_v, dtype=torch.float32, device=device)
            voltage_raw = torch.tensor(vd_v, dtype=torch.float32, device=device)
            model.eval()
            voltage_pred, _, _, _ = forward_voltage_conditioned(
                model, voltage_raw, pose, moment, frozen_phys, voltage_mean, voltage_std)
            val_rmse = torch.sqrt(torch.mean((voltage_pred - voltage_raw) ** 2)).item()

        if verbose and epoch % 5 == 0:
            tag = "val" if early_stop else "val(monitor only)"
            print(f"  epoch {epoch:4d} | {tag} RMSE = {val_rmse:.6f} V | "
                  f"output_scale = {model.output_scale.item():.5f}")

        if not early_stop:
            continue

        if val_rmse < best_val - min_delta:
            best_val = val_rmse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if early_stop:
        if best_state is not None:
            model.load_state_dict(best_state)
        return model, best_val, best_epoch
    else:
        return model, val_rmse, max_epochs - 1


# =============================================================================
# OPTUNA OBJECTIVE + SEARCH -- training hyperparameters only
# =============================================================================

def optuna_objective(trial, train_data, val_data, frozen_phys, cfg, device):
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    lambda_alpha = trial.suggest_float("lambda_alpha", 1e-6, 1e-1, log=True)
    output_scale_init = trial.suggest_float("output_scale_init", 1e-3, 0.3, log=True)

    set_seed(cfg["seed"] + trial.number)

    rp_tr, mw_tr, vd_tr = train_data
    voltage_mean_np, voltage_std_np = fit_voltage_normalization(vd_tr)
    voltage_mean = torch.tensor(voltage_mean_np, dtype=torch.float32, device=device)
    voltage_std = torch.tensor(voltage_std_np, dtype=torch.float32, device=device)

    train_loader = make_loader(rp_tr, mw_tr, vd_tr, batch_size=cfg["batch_size"], shuffle=True)

    model = ResidualNN(
        input_dim=N_SENSORS, hidden_dim=cfg["hidden_dim"],
        output_dim=N_SENSORS, n_blocks=cfg["n_residual_blocks"],
        output_scale_init=output_scale_init,
    ).to(device)

    _, val_rmse, best_epoch = train_alpha_nn(
        model, train_loader, val_data, frozen_phys, voltage_mean, voltage_std,
        lr=lr, weight_decay=weight_decay, lambda_alpha=lambda_alpha,
        huber_delta_v=cfg["huber_delta_v"], device=device,
        max_epochs=cfg["max_epochs"], patience=cfg["patience"],
        min_delta=cfg["min_delta"], batch_size=cfg["batch_size"], early_stop=True,
    )

    trial.set_user_attr("best_epoch", best_epoch)
    
    return val_rmse


# =============================================================================
# STAGE 2 -- FULL CALIBRATION 
# =============================================================================

def calibrate_stage2(train_data, val_data, test_data, physical_results, cfg, device):
    rp_train, mw_train, vd_train = train_data

    frozen_phys = build_frozen_physics(physical_results, device)
    frozen_snapshot = {k: v.detach().clone() for k, v in frozen_phys.items()}

    n_probe = min(50, rp_train.shape[0])
    check_torch_numpy_dipole_consistency(
        physical_results[:, 0:3], rp_train[:n_probe], mw_train[:n_probe]
    )

    print("\n===================================")
    print("STAGE 2: OPTUNA HYPERPARAMETER SEARCH")
    print("===================================")
    print(f"Fixed architecture: {N_SENSORS} -> {cfg['hidden_dim']} -> "
          f"{cfg['n_residual_blocks']} residual blocks (dim={cfg['hidden_dim']}) "
          f"-> {N_SENSORS}, batch_size={cfg['batch_size']} (NOT tuned)")
    print(f"{cfg['n_trials']} trials over [lr, weight_decay, lambda_alpha, "
          f"output_scale_init] | train={len(rp_train)} | "
          f"val={len(val_data[0])} | test={len(test_data[0])} "
          f"(held out, untouched until the end)")

    study = optuna.create_study(direction="minimize",
                                 sampler=optuna.samplers.TPESampler(seed=cfg["seed"]))
    study.optimize(
        lambda trial: optuna_objective(trial, train_data, val_data, frozen_phys, cfg, device),
        n_trials=cfg["n_trials"], show_progress_bar=False,
    )

    print(f"\n[Optuna] Best val RMSE = {study.best_value:.6f} V")
    print(f"[Optuna] Best params = {study.best_params}")
    print(f"[Optuna] Best trial's early-stop epoch = "
          f"{study.best_trial.user_attrs['best_epoch']}")

    best = study.best_params
    best_epoch = study.best_trial.user_attrs["best_epoch"]

    set_seed(cfg["seed"])
    voltage_mean_np, voltage_std_np = fit_voltage_normalization(vd_train)
    voltage_mean = torch.tensor(voltage_mean_np, dtype=torch.float32, device=device)
    voltage_std = torch.tensor(voltage_std_np, dtype=torch.float32, device=device)

    train_loader = make_loader(rp_train, mw_train, vd_train,
                                batch_size=cfg["batch_size"], shuffle=True)

    selection_model = ResidualNN(
        input_dim=N_SENSORS, hidden_dim=cfg["hidden_dim"],
        output_dim=N_SENSORS, n_blocks=cfg["n_residual_blocks"],
        output_scale_init=best["output_scale_init"],
    ).to(device)

    selection_model, selection_val_rmse, _ = train_alpha_nn(
        selection_model, train_loader, val_data, frozen_phys, voltage_mean, voltage_std,
        lr=best["lr"], weight_decay=best["weight_decay"], lambda_alpha=best["lambda_alpha"],
        huber_delta_v=cfg["huber_delta_v"], device=device,
        max_epochs=cfg["max_epochs"], patience=cfg["patience"],
        min_delta=cfg["min_delta"], batch_size=cfg["batch_size"],
        early_stop=True, verbose=True,
    )

    # ---- HONEST held-out test evaluation, exactly once ----
    with torch.no_grad():
        rp_te, mw_te, vd_te = test_data
        pose = torch.tensor(rp_te, dtype=torch.float32, device=device)
        moment = torch.tensor(mw_te, dtype=torch.float32, device=device)
        voltage_raw = torch.tensor(vd_te, dtype=torch.float32, device=device)
        selection_model.eval()
        voltage_pred, _, _, _ = forward_voltage_conditioned(
            selection_model, voltage_raw, pose, moment, frozen_phys,
            voltage_mean, voltage_std)
        test_rmse = torch.sqrt(torch.mean((voltage_pred - voltage_raw) ** 2)).item()

    print(f"\n[Stage 2] HONEST held-out TEST RMSE (never used for Optuna or "
          f"early stopping) = {test_rmse:.6f} V")

    rp_val, mw_val, vd_val = val_data
    rp_full = np.concatenate([rp_train, rp_val, rp_te])
    mw_full = np.concatenate([mw_train, mw_val, mw_te])
    vd_full = np.concatenate([vd_train, vd_val, vd_te])

    set_seed(cfg["seed"])
    voltage_mean_full_np, voltage_std_full_np = fit_voltage_normalization(vd_full)
    voltage_mean_full = torch.tensor(voltage_mean_full_np, dtype=torch.float32, device=device)
    voltage_std_full = torch.tensor(voltage_std_full_np, dtype=torch.float32, device=device)

    loader_full = make_loader(rp_full, mw_full, vd_full,
                               batch_size=cfg["batch_size"], shuffle=True)

    deploy_model = ResidualNN(
        input_dim=N_SENSORS, hidden_dim=cfg["hidden_dim"],
        output_dim=N_SENSORS, n_blocks=cfg["n_residual_blocks"],
        output_scale_init=best["output_scale_init"],
    ).to(device)

    fixed_epochs = best_epoch + 1 if best_epoch >= 0 else cfg["max_epochs"]
    deploy_model, _, _ = train_alpha_nn(
        deploy_model, loader_full, val_data, frozen_phys,
        voltage_mean_full, voltage_std_full,
        lr=best["lr"], weight_decay=best["weight_decay"], lambda_alpha=best["lambda_alpha"],
        huber_delta_v=cfg["huber_delta_v"], device=device,
        max_epochs=fixed_epochs, patience=cfg["patience"],
        min_delta=cfg["min_delta"], batch_size=cfg["batch_size"],
        early_stop=False, verbose=True,
    )

    checkpoint = {
        "model_state_dict": deploy_model.state_dict(),
        "input_dim": N_SENSORS,
        "hidden_dim": cfg["hidden_dim"],
        "output_dim": N_SENSORS,
        "n_residual_blocks": cfg["n_residual_blocks"],
        "batch_size": cfg["batch_size"],
        "output_scale_init": best["output_scale_init"],
        "voltage_mean": voltage_mean_full_np,
        "voltage_std": voltage_std_full_np,
        "lambda_alpha": best["lambda_alpha"],
        "lr": best["lr"],
        "weight_decay": best["weight_decay"],
        "huber_delta_v": cfg["huber_delta_v"],
        "best_epoch": best_epoch,
        "selection_val_rmse": selection_val_rmse,
        "held_out_test_rmse": test_rmse,
        "deploy_fixed_epochs": fixed_epochs,
        "optuna_best_params": best,
        "seed": cfg["seed"],
    }
    torch.save(checkpoint, cfg["stage2_checkpoint_path"])
    print(f"\nSaved Stage-2 checkpoint: {cfg['stage2_checkpoint_path']}")

    for name, tensor in frozen_phys.items():
        assert torch.equal(tensor, frozen_snapshot[name]), \
            f"Frozen Stage-1 tensor {name} changed during Stage 2!"

    return (
        deploy_model, selection_model, frozen_phys,
        voltage_mean_full, voltage_std_full,
        voltage_mean, voltage_std,
        study, checkpoint,
    )


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_stage2(model, data, frozen_phys, voltage_mean, voltage_std,
                     lambda_alpha, huber_delta_v, device, split_name):
    rp, mw, vd = data

    loader = make_loader(rp, mw, vd, batch_size=64, shuffle=False)

    metrics = run_epoch(model, loader, None, frozen_phys, voltage_mean, voltage_std,
                         lambda_alpha, huber_delta_v, device, training=False)

    model.eval()

    all_delta_alpha, all_voltage_pred = [], []

    with torch.no_grad():
        for pose, moment, voltage_raw in loader:
            pose, moment, voltage_raw = pose.to(device), moment.to(device), voltage_raw.to(device)

            voltage_pred, delta_alpha, _, _ = forward_voltage_conditioned(
                model, voltage_raw, pose, moment, frozen_phys, voltage_mean, voltage_std)

            all_delta_alpha.append(delta_alpha.cpu().numpy())
            all_voltage_pred.append(voltage_pred.cpu().numpy())

    delta_alpha_np = np.concatenate(all_delta_alpha, axis=0)
    voltage_pred_np = np.concatenate(all_voltage_pred, axis=0)

    voltage_error = voltage_pred_np - vd
    alpha_np = 1.0 + delta_alpha_np

    voltage_rmse = np.sqrt(np.mean(voltage_error**2))
    voltage_mae = np.mean(np.abs(voltage_error))

    print("\n-----------------------------------")
    print(f"{split_name} evaluation")
    print("-----------------------------------")
    print(f"Huber total loss   : {metrics['loss']:.8e}")
    print(f"Voltage loss       : {metrics['voltage_loss']:.8e}")
    print(f"Voltage RMSE       : {voltage_rmse:.8e} V")
    print(f"Voltage MAE        : {voltage_mae:.8e} V")
    print(f"Mean |delta_alpha|: {np.mean(np.abs(delta_alpha_np)):.8e}")
    print(f"Max  |delta_alpha|: {np.max(np.abs(delta_alpha_np)):.8e}")
    print(f"Alpha range       : [{alpha_np.min():.8e}, {alpha_np.max():.8e}]")

    return {
        "loss": metrics["loss"], "voltage_loss": metrics["voltage_loss"],
        "voltage_rmse": voltage_rmse, "voltage_mae": voltage_mae,
        "mean_abs_delta_alpha": np.mean(np.abs(delta_alpha_np)),
        "max_abs_delta_alpha": np.max(np.abs(delta_alpha_np)),
        "alpha_min": alpha_np.min(), "alpha_max": alpha_np.max(),
    }


# =============================================================================
# SANITY CHECK
# =============================================================================

def run_sanity_checks(model, frozen_phys, frozen_snapshot, voltage_mean, voltage_std,
                       sample_data, device):
    rp, mw, vd = sample_data

    pose = torch.tensor(rp[:1], dtype=torch.float32, device=device, requires_grad=True)
    moment = torch.tensor(mw[:1], dtype=torch.float32, device=device, requires_grad=True)
    voltage_raw = torch.tensor(vd[:1], dtype=torch.float32, device=device)

    model.eval()

    voltage_pred, delta_alpha, alpha, b_sensor = forward_voltage_conditioned(
        model, voltage_raw, pose, moment, frozen_phys, voltage_mean, voltage_std)

    print("\n===================================")
    print("STAGE 2 SANITY CHECK")
    print("===================================")

    assert delta_alpha.shape == (1, 64)
    assert voltage_pred.shape == (1, 64)
    print(f"delta_alpha shape : {tuple(delta_alpha.shape)}")
    print(f"voltage_pred shape: {tuple(voltage_pred.shape)}")

    for name, tensor in frozen_phys.items():
        unchanged = torch.equal(tensor, frozen_snapshot[name])
        print(f"Frozen '{name}' unchanged : {unchanged}")
        if not unchanged:
            raise RuntimeError(f"Frozen Stage-1 tensor '{name}' was modified!")

    alpha_expected = 1.0 + delta_alpha
    alpha_ok = torch.allclose(alpha, alpha_expected)
    print(f"alpha == 1 + delta_alpha  : {alpha_ok}")
    if not alpha_ok:
        raise RuntimeError("alpha formula check failed.")

    expected_v = (
        frozen_phys["offset"].unsqueeze(0)
        + frozen_phys["gain"].unsqueeze(0) * b_sensor * alpha
    )
    v_pred_ok = torch.allclose(voltage_pred, expected_v)
    print(f"V_pred == a + g*Bz*alpha  : {v_pred_ok}")
    if not v_pred_ok:
        raise RuntimeError("V_pred formula check failed.")

    grad_pose, grad_moment = torch.autograd.grad(
        voltage_pred.sum(), [pose, moment], allow_unused=True)
    print(f"Pose gradient exists     : {grad_pose is not None}")
    print(f"Moment gradient exists   : {grad_moment is not None}")

    if grad_pose is None or not torch.isfinite(grad_pose).all():
        raise RuntimeError("Pose gradient check failed.")
    if grad_moment is None or not torch.isfinite(grad_moment).all():
        raise RuntimeError("Moment gradient check failed.")

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


def plot_rmse(rmses, output_path):
    plt.figure(figsize=(10, 5))
    plt.bar(np.arange(len(rmses)), rmses)
    plt.xlabel("Sensor Index")
    plt.ylabel("RMSE")
    plt.title("Stage-1 Calibration RMSE")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=130)
    plt.close()
    print(f"Saved: {output_path}")


def save_stage2_alpha_summary(test_metrics, output_file):
    summary_df = pd.DataFrame({
        "metric": ["mean_abs_delta_alpha", "max_abs_delta_alpha", "alpha_min", "alpha_max"],
        "value": [test_metrics["mean_abs_delta_alpha"], test_metrics["max_abs_delta_alpha"],
                  test_metrics["alpha_min"], test_metrics["alpha_max"]],
    })
    summary_df.to_csv(output_file, index=False)
    print(f"Saved: {output_file}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = build_arg_parser().parse_args()
    set_seed(args.seed)

    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    physical_output_path = output_dir / "Calibration_Physical_Residual_NN.csv"
    alpha_output_path = output_dir / "Calibration_Alpha_Residual_NN.csv"
    stage2_checkpoint_path = output_dir / "Calibration_Stage2_Residual_NN.pt"
    summary_path = output_dir / "Calibration_Residual_NN_summary.json"
    stage1_rmse_plot_path = output_dir / "stage1_rmse.png"

    cfg = {
        "seed": args.seed,
        "hidden_dim": args.hidden_dim,
        "n_residual_blocks": args.n_residual_blocks,
        "batch_size": args.batch_size,
        "n_trials": args.n_trials,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "huber_delta_v": args.huber_delta_v,
        "stage2_checkpoint_path": stage2_checkpoint_path,
    }

    print("=" * 70)
    print("TWO-STAGE HALL SENSOR CALIBRATION  |  device =", device)
    print("=" * 70)
    print("Stage 1: physical calibration (least_squares)")
    print(f"Stage 2: Vraw(64) -> ResidualNN({N_SENSORS}->{cfg['hidden_dim']}->"
          f"{cfg['n_residual_blocks']} blocks->{N_SENSORS}) -> delta_alpha(64), "
          f"Optuna-tuned training hyperparameters")

    sensor_positions = load_sensor_positions(args.sensor_positions)
    robot_positions, m_world = load_robot_pose(args.robot_pose)
    voltage_data = load_voltage_data(args.voltage)
    offset_initial_values = load_offset_initial_values(
        args.offset_init, n_sensors=sensor_positions.shape[0])

    if sensor_positions.shape[0] != N_SENSORS:
        raise ValueError(f"Expected {N_SENSORS} sensors, got {sensor_positions.shape[0]}.")
    if voltage_data.shape[1] != N_SENSORS:
        raise ValueError(
            f"Expected voltage_data to have {N_SENSORS} columns, got {voltage_data.shape[1]}.")

    n_samples = min(len(robot_positions), len(voltage_data))
    robot_positions = robot_positions[:n_samples]
    m_world = m_world[:n_samples]
    voltage_data = voltage_data[:n_samples]

    print("\n===================================")
    print(f"SAMPLING: {args.n_total_samples} random points -> "
          f"{args.n_stage1_samples} Stage 1 / "
          f"{args.n_total_samples - args.n_stage1_samples} Stage 2")
    print("===================================")

    stage1_split, stage2_split = select_stage1_stage2_split(
        robot_positions, m_world, voltage_data,
        n_total=args.n_total_samples, n_stage1=args.n_stage1_samples)
    stage1_idx, rp1, mw1, vd1 = stage1_split
    stage2_idx, rp2, mw2, vd2 = stage2_split

    print("\n===================================")
    print("STAGE 1: PHYSICAL PARAMETER FIT")
    print("===================================")
    results, rmses = run_calibration(
        sensor_positions, rp1, mw1, vd1, offset_initial_values=offset_initial_values,
        lambda_pos=args.lambda_pos, lambda_gain=args.lambda_gain,
        lambda_offset=args.lambda_offset,
    )

    print("\n========================")
    print(f"Mean RMSE = {np.mean(rmses):.6f}")
    print(f"Max RMSE  = {np.max(rmses):.6f}")
    print(f"Min RMSE  = {np.min(rmses):.6f}")
    print("========================")

    save_physical_results(results, physical_output_path)
    plot_rmse(rmses, stage1_rmse_plot_path)

    train_data, val_data, test_data = split_stage2_data(
        rp2, mw2, vd2, val_fraction=args.val_fraction,
        test_fraction=args.test_fraction, seed=args.seed)

    (
        deploy_model, selection_model, frozen_phys,
        voltage_mean_deploy, voltage_std_deploy,
        voltage_mean_sel, voltage_std_sel,
        study, checkpoint,
    ) = calibrate_stage2(train_data, val_data, test_data, results, cfg, device)

    frozen_snapshot = {k: v.detach().clone() for k, v in frozen_phys.items()}

    run_sanity_checks(
        selection_model, frozen_phys, frozen_snapshot,
        voltage_mean_sel, voltage_std_sel, train_data, device,
    )

    val_metrics = evaluate_stage2(
        selection_model, val_data, frozen_phys, voltage_mean_sel, voltage_std_sel,
        checkpoint["lambda_alpha"], cfg["huber_delta_v"], device,
        "VALIDATION (selection model)",
    )
    test_metrics = evaluate_stage2(
        selection_model, test_data, frozen_phys, voltage_mean_sel, voltage_std_sel,
        checkpoint["lambda_alpha"], cfg["huber_delta_v"], device,
        "TEST (selection model, honest, untouched until now)",
    )

    save_stage2_alpha_summary(test_metrics, alpha_output_path)

    summary = {
        "stage1": {
            "n_samples": int(len(rp1)),
            "mean_rmse": float(np.mean(rmses)),
            "max_rmse": float(np.max(rmses)),
            "min_rmse": float(np.min(rmses)),
        },
        "stage2": {
            "n_samples": int(len(rp2)),
            "train_samples": int(len(train_data[0])),
            "val_samples": int(len(val_data[0])),
            "test_samples": int(len(test_data[0])),
            "hidden_dim": cfg["hidden_dim"],
            "n_residual_blocks": cfg["n_residual_blocks"],
            "batch_size": cfg["batch_size"],
            "optuna_n_trials": cfg["n_trials"],
            "optuna_best_params": checkpoint["optuna_best_params"],
            "best_epoch": int(checkpoint["best_epoch"]),
            "val_voltage_rmse": float(val_metrics["voltage_rmse"]),
            "val_voltage_mae": float(val_metrics["voltage_mae"]),
            "test_voltage_rmse": float(test_metrics["voltage_rmse"]),
            "test_voltage_mae": float(test_metrics["voltage_mae"]),
            "test_mean_abs_delta_alpha": float(test_metrics["mean_abs_delta_alpha"]),
            "test_max_abs_delta_alpha": float(test_metrics["max_abs_delta_alpha"]),
            "test_alpha_min": float(test_metrics["alpha_min"]),
            "test_alpha_max": float(test_metrics["alpha_max"]),
            "deploy_fixed_epochs": int(checkpoint["deploy_fixed_epochs"]),
        },
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n===================================")
    print("ALL STAGES FINISHED")
    print("===================================")
    print(f"Summary saved to: {summary_path}")
    print(f"Honest held-out TEST RMSE: {test_metrics['voltage_rmse']:.6f} V")
    print("(Deploy model was refit on train+val+test for "
          f"{checkpoint['deploy_fixed_epochs']} fixed epochs -- not "
          "evaluated against test again, to avoid leakage into this metric.)")
    print(f"\nOutputs written to: {output_dir}")


if __name__ == "__main__":
    main()