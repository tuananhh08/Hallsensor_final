"""
2-stage calibration framework 
------------------------------------------------------------------------------
Stage 1 : per-sensor physical parameters (x, y, z, offset, gain), fit via
          scipy.optimize.least_squares (dipole model, bounded trust-region,
          priors pulling toward nominal/design values).
Stage 2 : physical parameters frozen. A shared neural network learns only the
          multiplicative residual alpha = 1 + delta_alpha from normalized
          dipole-geometry features.  The voltage is still formed by the
          hard-coded dipole forward model, never directly by the network.

Run from the command line, e.g. in a Colab cell after `!git clone ...`:

    !python calib_nn_h.py \
        --sensor_positions "/content/drive/MyDrive/Dataset/Hall_sensor_positions.csv" \
        --robot_pose       "/content/drive/MyDrive/Dataset/Grid_points_coordinates.csv" \
        --voltage          "/content/drive/MyDrive/Dataset/Grid_data.csv" \
        --offset_init      "/content/drive/MyDrive/Dataset/Offset_Sens.csv" \
        --output_dir       "/content/drive/MyDrive/Dataset" \
        --n_trials 30 \
        --max_epochs 20 \
        --patience 8

Everything that used to be a hardcoded path/constant at the top of the
script is now a CLI flag (with the same default values as before) -- see
`build_arg_parser()` below for the full list.
"""

import argparse
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
    sys.exit("optuna is required: !pip install optuna --quiet")


MU0_OVER_4PI = 1e-7


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


# =============================================================================
# CLI
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="2-stage Hall-sensor calibration: Stage 1 physical "
                     "params (least_squares) + Stage 2 physics-informed "
                     "neural residual correction (Optuna-tuned).",
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
                         "the trained physics-feature residual model and diagnostics). "
                         "Created if it doesn't exist.")

    # ---- Stage 1 / Stage 2 split sizes ----
    p.add_argument("--n_total_samples", type=int, default=1800,
                    help="Total number of (robot_pose, voltage) rows drawn "
                         "at random for calibration (Stage1 + Stage2 pool).")
    p.add_argument("--n_stage1_samples", type=int, default=300,
                    help="How many of --n_total_samples go to Stage 1 "
                         "(physical parameter fit). The rest form the "
                         "Stage 2 pool (train/val/test).")
    p.add_argument("--val_fraction", type=float, default=0.2,
                    help="Fraction of the Stage-2 pool (after removing the "
                         "test split) used for Optuna/early-stopping validation.")
    p.add_argument("--test_fraction", type=float, default=0.15,
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

    # ---- Stage 2 NN: Optuna search budget ----
    p.add_argument("--n_trials", type=int, default=35,
                    help="Number of Optuna trials for Stage-2 hyperparameter search.")
    p.add_argument("--max_epochs", type=int, default=200,
                    help="Max training epochs per Optuna trial / model-selection run.")
    p.add_argument("--patience", type=int, default=15,
                    help="Early-stopping patience (epochs without val improvement).")

    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--device", default=None,
                    help="Force 'cuda' or 'cpu'. Default: auto-detect.")

    return p


# =============================================================================
# DIPOLE MODEL
# =============================================================================

def dipole_field(r_vec, m_vec):
    """Calculate magnetic field from dipole model. r_vec, m_vec: (N, 3)."""
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
        raise ValueError(
            f"offset_init CSV must contain exactly sensor_index values 0 to {n_sensors - 1}."
        )
    offset_initial_values = df["offset_a_V"].to_numpy(dtype=float)
    if not np.isfinite(offset_initial_values).all():
        raise ValueError("Column offset_a_V contains missing or non-finite values.")
    print(f"Loaded per-sensor offset initial values: {offset_initial_values.shape}")
    return offset_initial_values


# =============================================================================
# STAGE 1: PER-SENSOR PHYSICAL PARAMETER FIT
# =============================================================================

def sensor_residuals(params, robot_positions, m_world, voltage_sensor,
                      pos_prior, offset_prior, g0,
                      lambda_pos, lambda_gain, lambda_offset):
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
        method="trf", max_nfev=250,
    )

    params_opt = result.x
    nx, ny, nz = 0.0, 0.0, 1.0
    params_extended = np.array([
        params_opt[0], params_opt[1], params_opt[2], params_opt[3], params_opt[4],
        nx, ny, nz,
    ])

    n_voltage = voltage_sensor.shape[0]
    rmse = np.sqrt(np.mean(result.fun[:n_voltage] ** 2))
    print(f"Sensor {sensor_index + 1:02d} | RMSE = {rmse:.6f} V")
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
            lambda_pos=lambda_pos, lambda_gain=lambda_gain, lambda_offset=lambda_offset,)
        results.append(params)
        rmses.append(rmse)
    return np.array(results), np.array(rmses)


# =============================================================================
# SAMPLING: Stage1 / Stage2(train) / Stage2(val) / Stage2(test) split
# =============================================================================

def select_splits(robot_positions, m_world, voltage_data,
                   n_total, n_stage1, val_fraction, test_fraction, seed):
    n_samples = robot_positions.shape[0]
    if n_total > n_samples:
        raise ValueError(f"Requested {n_total} calibration points but dataset only has {n_samples}.")

    rng = np.random.default_rng(seed)
    all_idx = rng.choice(n_samples, size=n_total, replace=False)
    stage1_idx = all_idx[:n_stage1]
    stage2_idx = all_idx[n_stage1:]

    n_test = int(len(stage2_idx) * test_fraction)
    stage2_test_idx = stage2_idx[:n_test]
    remaining_idx = stage2_idx[n_test:]

    n_val = int(len(remaining_idx) * val_fraction)
    stage2_val_idx = remaining_idx[:n_val]
    stage2_train_idx = remaining_idx[n_val:]

    print(f"\n[Sampling] seed={seed} | total={n_total} -> "
          f"Stage1={len(stage1_idx)}, Stage2-train={len(stage2_train_idx)}, "
          f"Stage2-val={len(stage2_val_idx)}, Stage2-test={len(stage2_test_idx)}")

    def gather(idx):
        return robot_positions[idx], m_world[idx], voltage_data[idx]

    return (
        (stage1_idx, *gather(stage1_idx)),
        (stage2_train_idx, *gather(stage2_train_idx)),
        (stage2_val_idx, *gather(stage2_val_idx)),
        (stage2_test_idx, *gather(stage2_test_idx)),
    )


# =============================================================================
# STAGE 2: PHYSICS-INFORMED NEURAL delta_alpha CORRECTION
# =============================================================================

class DeltaAlphaNet(nn.Module):
    """Shared residual model.  It predicts only delta_alpha, never voltage."""
    HIDDEN_DIMS = (32, 64, 64, 16)

    def __init__(self, input_dim: int = 1, output_scale_init: float = 0.05):
        super().__init__()
        layers = []
        in_dim = input_dim
        for hidden_dim in self.HIDDEN_DIMS:
            layers += [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
            in_dim = hidden_dim
        layers += [nn.Linear(in_dim, 1)]
        self.net = nn.Sequential(*layers)

        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

        # Learnable output scale 
        self.output_scale = nn.Parameter(torch.tensor(float(output_scale_init)))

    def forward(self, h_norm: torch.Tensor) -> torch.Tensor:
        delta = self.net(h_norm) * self.output_scale
        return delta.squeeze(-1)


FEATURE_NAMES = ("dx", "dy", "dz", "r", "Bx", "By", "Bz", "Bmag", "cos_theta_m")
EPS = 1e-12


def frozen_physical_tensors(physical_results, device):
    """Turn Stage-1 output into immutable Stage-2 constants."""
    return {
        "sensor_pos": torch.tensor(physical_results[:, 0:3], dtype=torch.float32,
                                   device=device, requires_grad=False),
        "offset": torch.tensor(physical_results[:, 3], dtype=torch.float32,
                               device=device, requires_grad=False),
        "gain": torch.tensor(physical_results[:, 4], dtype=torch.float32,
                             device=device, requires_grad=False),
        "sensor_dir": torch.tensor(physical_results[:, 5:8], dtype=torch.float32,
                                   device=device, requires_grad=False),
    }


def torch_dipole_field(r_vec, m_vec):
    """Differentiable counterpart of the unchanged NumPy Stage-1 dipole model."""
    r = torch.linalg.vector_norm(r_vec, dim=-1, keepdim=True).clamp_min(EPS)
    r3, r5 = r.pow(3), r.pow(5)
    mdotr = (m_vec * r_vec).sum(dim=-1, keepdim=True)
    return MU0_OVER_4PI * (3.0 * r_vec * mdotr / r5 - m_vec / r3)


def build_physical_features(r_vec, B, m_world):
    """Features contain only geometry and dipole quantities, never measured voltage."""
    r = torch.linalg.vector_norm(r_vec, dim=-1, keepdim=True).clamp_min(EPS)
    bmag = torch.linalg.vector_norm(B, dim=-1, keepdim=True)
    m_norm = torch.linalg.vector_norm(m_world, dim=-1, keepdim=True).clamp_min(EPS)
    cos_theta_m = (r_vec * m_world).sum(dim=-1, keepdim=True) / (r * m_norm)
    return torch.cat((r_vec, r, B, bmag, cos_theta_m), dim=-1)


def forward_dipole_alpha(model, pose, m_world, frozen_phys, feature_mean, feature_std):
    """Differentiable Stage-2 physics forward model for calibration and future LM use.

    Args have shapes pose=(N,3), m_world=(N,3); returned fields are (N,S).
    """
    r_vec = frozen_phys["sensor_pos"].unsqueeze(0) - pose.unsqueeze(1)
    m_expanded = m_world.unsqueeze(1).expand_as(r_vec)
    B = torch_dipole_field(r_vec, m_expanded)
    B_sensor = (B * frozen_phys["sensor_dir"].unsqueeze(0)).sum(dim=-1)
    features = build_physical_features(r_vec, B, m_expanded)
    x = features.reshape(-1, len(FEATURE_NAMES))
    x_norm = (x - feature_mean) / feature_std
    delta_alpha = model(x_norm).reshape_as(B_sensor)
    alpha = 1.0 + delta_alpha
    v_pred = frozen_phys["offset"].unsqueeze(0) + \
        frozen_phys["gain"].unsqueeze(0) * B_sensor * alpha
    return v_pred, alpha, delta_alpha, B, B_sensor, features


def make_pose_tensors(rp, mw, vd):
    return (torch.tensor(rp, dtype=torch.float32),
            torch.tensor(mw, dtype=torch.float32),
            torch.tensor(vd, dtype=torch.float32))


def feature_statistics(rp, mw, frozen_phys):
    """Fit normalisation only from the Stage-2 training split."""
    with torch.no_grad():
        pose = torch.tensor(rp, dtype=torch.float32, device=frozen_phys["offset"].device)
        moment = torch.tensor(mw, dtype=torch.float32, device=pose.device)
        r_vec = frozen_phys["sensor_pos"].unsqueeze(0) - pose.unsqueeze(1)
        B = torch_dipole_field(r_vec, moment.unsqueeze(1).expand_as(r_vec))
        x = build_physical_features(r_vec, B, moment.unsqueeze(1).expand_as(r_vec))
        x = x.reshape(-1, len(FEATURE_NAMES))
        return x.mean(dim=0), x.std(dim=0, unbiased=False).clamp_min(1e-8)


def evaluate_stage2(model, tensors, frozen_phys, feature_mean, feature_std, device):
    pose, moment, voltage = (t.to(device) for t in tensors)
    model.eval()
    with torch.no_grad():
        v_pred, alpha, delta, _, _, features = forward_dipole_alpha(
            model, pose, moment, frozen_phys, feature_mean, feature_std)
        error = v_pred - voltage
        return {
            "rmse": torch.sqrt(torch.mean(error ** 2)).item(),
            "mae": torch.mean(torch.abs(error)).item(),
            "v_pred": v_pred.cpu(), "alpha": alpha.cpu(), "delta": delta.cpu(),
            "features": features.cpu(), "error": error.cpu(),
        }


def train_alpha_nn(model, train_loader, val_tensors, frozen_phys, feature_mean, feature_std,
                    lr, weight_decay, delta_l2, device,
                    max_epochs, patience, verbose=False, early_stop=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    huber = nn.HuberLoss(delta=1e-3)

    best_val = float("inf")
    best_state = None
    best_epoch = -1
    epochs_no_improve = 0
    val_rmse = float("inf")

    for epoch in range(max_epochs):
        model.train()
        for pose_b, moment_b, v_b in train_loader:
            pose_b, moment_b, v_b = (t.to(device) for t in (pose_b, moment_b, v_b))
            optimizer.zero_grad()
            v_pred, _, delta_alpha, _, _, _ = forward_dipole_alpha(
                model, pose_b, moment_b, frozen_phys, feature_mean, feature_std)
            loss_data = huber(v_pred, v_b)
            loss_reg = delta_l2 * (delta_alpha ** 2).mean()
            loss = loss_data + loss_reg
            loss.backward()
            optimizer.step()

        val_rmse = evaluate_stage2(model, val_tensors, frozen_phys, feature_mean,
                                   feature_std, device)["rmse"]

        if verbose and epoch % 5 == 0:
            tag = "val" if early_stop else "val(monitor only)"
            print(f"  epoch {epoch:4d} | {tag} RMSE = {val_rmse:.6f} V | "
                  f"output_scale = {model.output_scale.item():.5f}")

        if not early_stop:
            continue

        if val_rmse < best_val - 1e-9:
            best_val = val_rmse
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
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


def optuna_objective(trial, train_tensors, val_tensors, frozen_phys, feature_mean,
                     feature_std, device, max_epochs, patience):
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    delta_l2 = trial.suggest_float("delta_l2", 1e-6, 1e-1, log=True)
    output_scale_init = trial.suggest_float("output_scale_init", 1e-3, 0.3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64])

    set_seed(42 + trial.number)

    dataset = TensorDataset(*train_tensors)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DeltaAlphaNet(input_dim=len(FEATURE_NAMES),
                          output_scale_init=output_scale_init).to(device)

    _, val_rmse, best_epoch = train_alpha_nn(
        model, loader, val_tensors, frozen_phys, feature_mean, feature_std,
        lr=lr, weight_decay=weight_decay, delta_l2=delta_l2,
        device=device, max_epochs=max_epochs, patience=patience,
        early_stop=True,
    )

    trial.set_user_attr("output_scale_init", output_scale_init)
    trial.set_user_attr("best_epoch", best_epoch)
    return val_rmse


def calibrate_alpha_nn(physical_results,
                        rp_train, mw_train, vd_train,
                        rp_val, mw_val, vd_val,
                        rp_test, mw_test, vd_test,
                        n_trials, max_epochs, patience, seed, device):
    frozen_phys = frozen_physical_tensors(physical_results, device)
    frozen_snapshot = {name: tensor.detach().clone() for name, tensor in frozen_phys.items()}
    train_tensors = make_pose_tensors(rp_train, mw_train, vd_train)
    val_tensors = make_pose_tensors(rp_val, mw_val, vd_val)
    test_tensors = make_pose_tensors(rp_test, mw_test, vd_test)
    feature_mean, feature_std = feature_statistics(rp_train, mw_train, frozen_phys)

    # Numerical consistency check against the existing (unchanged) NumPy Stage-1 field.
    with torch.no_grad():
        check_pose = torch.tensor(rp_train[:2], dtype=torch.float32, device=device)
        check_moment = torch.tensor(mw_train[:2], dtype=torch.float32, device=device)
        check_r = frozen_phys["sensor_pos"].unsqueeze(0) - check_pose.unsqueeze(1)
        check_m = check_moment.unsqueeze(1).expand_as(check_r)
        check_b_torch = torch_dipole_field(check_r, check_m)
        check_b_numpy = dipole_field(check_r.detach().cpu().numpy().reshape(-1, 3),
                                     check_m.detach().cpu().numpy().reshape(-1, 3))
        check_b_numpy = torch.tensor(check_b_numpy, dtype=torch.float32, device=device).reshape_as(check_b_torch)
        assert torch.allclose(check_b_torch, check_b_numpy, rtol=2e-5, atol=1e-10), \
            "Torch Stage-2 dipole field disagrees with the Stage-1 dipole field"

    print(f"\n[Stage 2 / Optuna] {n_trials} trials, "
          f"{vd_train.size} train pairs, {vd_val.size} val pairs, "
          f"{vd_test.size} test pairs (held out, untouched until the end)")
    print(f"[Stage 2] features = {FEATURE_NAMES} (statistics fit on train only)")
    print(f"[Stage 2] Fixed architecture: {DeltaAlphaNet.HIDDEN_DIMS} (SiLU) | device={device}")

    study = optuna.create_study(direction="minimize",
                                 sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(
        lambda trial: optuna_objective(trial, train_tensors, val_tensors, frozen_phys,
                                        feature_mean, feature_std,
                                        device, max_epochs, patience),
        n_trials=n_trials, show_progress_bar=False,
    )

    print(f"\n[Stage 2 / Optuna] Best val RMSE = {study.best_value:.6f} V")
    print(f"[Stage 2 / Optuna] Best params = {study.best_params}")
    print(f"[Stage 2 / Optuna] Best trial's early-stop epoch = "
          f"{study.best_trial.user_attrs['best_epoch']}")

    best = study.best_params
    best_epoch = study.best_trial.user_attrs["best_epoch"]

    set_seed(seed)
    loader_train = DataLoader(TensorDataset(*train_tensors),
                               batch_size=best["batch_size"], shuffle=True)
    selection_model = DeltaAlphaNet(input_dim=len(FEATURE_NAMES),
                                    output_scale_init=best["output_scale_init"]).to(device)
    selection_model, selection_val_rmse, _ = train_alpha_nn(
        selection_model, loader_train, val_tensors, frozen_phys, feature_mean, feature_std,
        lr=best["lr"], weight_decay=best["weight_decay"], delta_l2=best["delta_l2"],
        device=device, max_epochs=max_epochs, patience=patience,
        early_stop=True, verbose=True,
    )

    test_metrics = evaluate_stage2(selection_model, test_tensors, frozen_phys,
                                   feature_mean, feature_std, device)
    test_rmse = test_metrics["rmse"]

    print(f"\n[Stage 2] HONEST held-out TEST RMSE (never used for Optuna or "
          f"early stopping) = {test_rmse:.6f} V")

    full_tensors = tuple(torch.cat(parts, dim=0) for parts in zip(
        train_tensors, val_tensors, test_tensors))
    loader_full = DataLoader(TensorDataset(*full_tensors),
                              batch_size=best["batch_size"], shuffle=True)

    set_seed(seed)
    deploy_model = DeltaAlphaNet(input_dim=len(FEATURE_NAMES),
                                 output_scale_init=best["output_scale_init"]).to(device)
    fixed_epochs = best_epoch + 1 if best_epoch >= 0 else max_epochs
    deploy_model, _, _ = train_alpha_nn(
        deploy_model, loader_full, val_tensors, frozen_phys, feature_mean, feature_std,
        lr=best["lr"], weight_decay=best["weight_decay"], delta_l2=best["delta_l2"],
        device=device, max_epochs=fixed_epochs, patience=patience,
        early_stop=False, verbose=True,
    )

    meta = {
        "feature_names": list(FEATURE_NAMES),
        "feature_mean": feature_mean.detach().cpu().tolist(),
        "feature_std": feature_std.detach().cpu().tolist(),
        "hidden_dims": "-".join(map(str, DeltaAlphaNet.HIDDEN_DIMS)),
        "activation": "SiLU",
        "output_scale_init": best["output_scale_init"],
        "output_scale_final": deploy_model.output_scale.item(),
        "selection_val_rmse": selection_val_rmse,
        "held_out_test_rmse": test_rmse,
        "held_out_test_mae": test_metrics["mae"],
        "deploy_fixed_epochs": fixed_epochs,
    }
    # Calibration-time checks: physical parameters remain constants and pose is differentiable.
    for name, tensor in frozen_phys.items():
        assert not tensor.requires_grad, f"Frozen Stage-1 tensor {name} unexpectedly requires gradients"
        assert torch.equal(tensor, frozen_snapshot[name]), f"Frozen Stage-1 tensor {name} changed during Stage 2"
    probe_pose = train_tensors[0][:1].to(device).detach().clone().requires_grad_(True)
    probe_moment = train_tensors[1][:1].to(device).detach().clone().requires_grad_(True)
    probe_v, probe_alpha, probe_delta, _, probe_bsensor, _ = forward_dipole_alpha(
        selection_model, probe_pose, probe_moment, frozen_phys, feature_mean, feature_std)
    assert torch.allclose(probe_alpha, 1.0 + probe_delta)
    expected_v = frozen_phys["offset"].unsqueeze(0) + frozen_phys["gain"].unsqueeze(0) * probe_bsensor * probe_alpha
    assert torch.allclose(probe_v, expected_v)
    probe_v.sum().backward()
    assert probe_pose.grad is not None and torch.isfinite(probe_pose.grad).all()
    assert probe_moment.grad is not None and torch.isfinite(probe_moment.grad).all()
    return deploy_model, meta, study


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
    print(f"\nSaved: {output_file}")


def save_alpha_nn(model, meta, model_path, meta_path):
    """Checkpoint includes normalisation needed for identical deployment inference."""
    torch.save({"model_state_dict": model.state_dict(),
                "feature_names": meta["feature_names"],
                "feature_mean": meta["feature_mean"],
                "feature_std": meta["feature_std"]}, model_path)
    meta_csv = meta.copy()
    for key in ("feature_names", "feature_mean", "feature_std"):
        meta_csv[key] = repr(meta_csv[key])
    pd.DataFrame([meta_csv]).to_csv(meta_path, index=False)
    print(f"Saved: {model_path}")
    print(f"Saved: {meta_path}")


def plot_rmse(rmses, output_path):
    plt.figure(figsize=(10, 5))
    plt.bar(np.arange(len(rmses)), rmses)
    plt.xlabel("Sensor Index")
    plt.ylabel("RMSE")
    plt.title("Stage 1 Calibration RMSE")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=130)
    plt.close()
    print(f"Saved: {output_path}")


def save_stage2_diagnostics(model, physical_results, splits, meta, output_path, device):
    """Voltage-domain metrics; alpha_emp is intentionally not used here."""
    frozen_phys = frozen_physical_tensors(physical_results, device)
    mean = torch.tensor(meta["feature_mean"], dtype=torch.float32, device=device)
    std = torch.tensor(meta["feature_std"], dtype=torch.float32, device=device)
    rows = []
    for name, (rp, mw, vd) in splits.items():
        metrics = evaluate_stage2(model, make_pose_tensors(rp, mw, vd), frozen_phys, mean, std, device)
        # The Stage-1 baseline is alpha=1, computed from the same hard-coded physics.
        with torch.no_grad():
            pose = torch.tensor(rp, dtype=torch.float32, device=device)
            moment = torch.tensor(mw, dtype=torch.float32, device=device)
            r_vec = frozen_phys["sensor_pos"].unsqueeze(0) - pose.unsqueeze(1)
            B = torch_dipole_field(r_vec, moment.unsqueeze(1).expand_as(r_vec))
            b_sensor = (B * frozen_phys["sensor_dir"].unsqueeze(0)).sum(dim=-1)
            baseline = frozen_phys["offset"].unsqueeze(0) + frozen_phys["gain"].unsqueeze(0) * b_sensor
            base_error = baseline.cpu() - torch.tensor(vd, dtype=torch.float32)
        stage2_error = metrics["error"]
        aggregate = {"record_type": "aggregate", "split": name,
                     "stage1_rmse": torch.sqrt(torch.mean(base_error ** 2)).item(),
                     "stage1_mae": torch.mean(torch.abs(base_error)).item(),
                     "stage2_rmse": metrics["rmse"], "stage2_mae": metrics["mae"],
                     "alpha_mean": metrics["alpha"].mean().item(), "alpha_std": metrics["alpha"].std().item(),
                     "delta_alpha_mean": metrics["delta"].mean().item(), "delta_alpha_std": metrics["delta"].std().item()}
        rows.append(aggregate)
        for sensor_index in range(vd.shape[1]):
            e1, e2 = base_error[:, sensor_index], stage2_error[:, sensor_index]
            rows.append({"record_type": "sensor", "split": name, "sensor_index": sensor_index,
                         "stage1_rmse": torch.sqrt(torch.mean(e1 ** 2)).item(),
                         "stage1_mae": torch.mean(torch.abs(e1)).item(),
                         "stage2_rmse": torch.sqrt(torch.mean(e2 ** 2)).item(),
                         "stage2_mae": torch.mean(torch.abs(e2)).item()})
        distance = metrics["features"][..., 3].reshape(-1).numpy()
        e1_flat, e2_flat = base_error.reshape(-1).numpy(), stage2_error.reshape(-1).numpy()
        bin_edges = np.linspace(distance.min(), distance.max(), 6)
        for bin_index in range(len(bin_edges) - 1):
            lower, upper = bin_edges[bin_index], bin_edges[bin_index + 1]
            mask = (distance >= lower) & ((distance < upper) if bin_index < len(bin_edges) - 2 else (distance <= upper))
            if not mask.any():
                continue
            rows.append({"record_type": "distance_bin", "split": name, "distance_bin": bin_index,
                         "distance_min_m": lower, "distance_max_m": upper, "count": int(mask.sum()),
                         "stage1_rmse": float(np.sqrt(np.mean(e1_flat[mask] ** 2))),
                         "stage1_mae": float(np.mean(np.abs(e1_flat[mask]))),
                         "stage2_rmse": float(np.sqrt(np.mean(e2_flat[mask] ** 2))),
                         "stage2_mae": float(np.mean(np.abs(e2_flat[mask])))})
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


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

    physical_output_path = output_dir / "Calibration_Physical_NN.csv"
    alpha_nn_output_path = output_dir / "Calibration_AlphaNN_physics.pt"
    alpha_nn_meta_path = output_dir / "Calibration_AlphaNN_physics_meta.csv"
    stage1_rmse_plot_path = output_dir / "stage1_rmse.png"
    stage2_diagnostics_path = output_dir / "stage2_voltage_diagnostics.csv"

    print("=" * 70)
    print("2-STAGE CALIBRATION  |  device =", device)
    print("=" * 70)

    sensor_positions = load_sensor_positions(args.sensor_positions)
    robot_positions, m_world = load_robot_pose(args.robot_pose)
    voltage_data = load_voltage_data(args.voltage)
    offset_initial_values = load_offset_initial_values(
        args.offset_init, n_sensors=sensor_positions.shape[0]
    )

    n_samples = min(len(robot_positions), len(voltage_data))
    robot_positions = robot_positions[:n_samples]
    m_world = m_world[:n_samples]
    voltage_data = voltage_data[:n_samples]

    print("\n===================================")
    print("SAMPLING: Stage1 / Stage2-train / Stage2-val / Stage2-test split")
    print("===================================")
    (s1_idx, rp1, mw1, vd1), (s2t_idx, rp2t, mw2t, vd2t), \
        (s2v_idx, rp2v, mw2v, vd2v), (s2te_idx, rp2te, mw2te, vd2te) = \
        select_splits(robot_positions, m_world, voltage_data,
                       n_total=args.n_total_samples, n_stage1=args.n_stage1_samples,
                       val_fraction=args.val_fraction, test_fraction=args.test_fraction,
                       seed=args.seed)

    print("\n===================================")
    print("STAGE 1: PHYSICAL PARAMETER FIT")
    print("===================================")
    results, rmses = run_calibration(
        sensor_positions, rp1, mw1, vd1, offset_initial_values=offset_initial_values,
        lambda_pos=args.lambda_pos, lambda_gain=args.lambda_gain,
        lambda_offset=args.lambda_offset,
    )
    print(f"\nStage 1 Mean RMSE = {np.mean(rmses):.6f} | "
          f"Max = {np.max(rmses):.6f} | Min = {np.min(rmses):.6f}")
    save_physical_results(results, physical_output_path)
    plot_rmse(rmses, stage1_rmse_plot_path)

    print("\n===================================")
    print("STAGE 2: PHYSICS-INFORMED NEURAL delta_alpha CORRECTION (Optuna-tuned)")
    print("===================================")
    deploy_model, meta, study = calibrate_alpha_nn(
        results, rp2t, mw2t, vd2t, rp2v, mw2v, vd2v, rp2te, mw2te, vd2te,
        n_trials=args.n_trials, max_epochs=args.max_epochs, patience=args.patience,
        seed=args.seed, device=device,
    )
    save_alpha_nn(deploy_model, meta, alpha_nn_output_path, alpha_nn_meta_path)
    save_stage2_diagnostics(
        deploy_model, results,
        {"train": (rp2t, mw2t, vd2t), "validation": (rp2v, mw2v, vd2v),
         "test": (rp2te, mw2te, vd2te)},
        meta, stage2_diagnostics_path, device,
    )

    print("\n===================================")
    print("ALL STAGES FINISHED")
    print(f"Honest held-out TEST RMSE (Stage 2, model-selection run): "
          f"{meta['held_out_test_rmse']:.6f} V")
    print(f"(Deploy model was refit on train+val+test for "
          f"{meta['deploy_fixed_epochs']} fixed epochs -- not evaluated "
          f"against test again, to avoid leakage into this metric.)")
    print(f"\nOutputs written to: {output_dir}")
    print("===================================")


if __name__ == "__main__":
    main()
