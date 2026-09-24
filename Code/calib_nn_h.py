"""
2-stage calibration framework (CLI version, for `git clone` + Colab `!python ...`)
------------------------------------------------------------------------------
Stage 1 : per-sensor physical parameters (x, y, z, offset, gain), fit via
          scipy.optimize.least_squares (dipole model, bounded trust-region,
          priors pulling toward nominal/design values).
Stage 2 : physical parameters frozen. A small neural network learns a
          multiplicative correction alpha(h) = 1 + delta_alpha(h), where h is
          the signed height offset (capsule z - calibrated sensor z) along
          the sensor's fixed sensitive axis.

Run from the command line, e.g. in a Colab cell after `!git clone ...`:

    !python calib_alpha_nn_h.py \
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
                     "params (least_squares) + Stage 2 neural alpha(h) "
                     "correction (Optuna-tuned).",
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
                         "the trained alpha(h) model, and diagnostic plots). "
                         "Created if it doesn't exist.")

    # ---- Stage 1 / Stage 2 split sizes ----
    p.add_argument("--n_total_samples", type=int, default=3200,
                    help="Total number of (robot_pose, voltage) rows drawn "
                         "at random for calibration (Stage1 + Stage2 pool).")
    p.add_argument("--n_stage1_samples", type=int, default=400,
                    help="How many of --n_total_samples go to Stage 1 "
                         "(physical parameter fit). The rest form the "
                         "Stage 2 pool (train/val/test).")
    p.add_argument("--val_fraction", type=float, default=0.15,
                    help="Fraction of the Stage-2 pool (after removing the "
                         "test split) used for Optuna/early-stopping validation.")
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

    # ---- Stage 2 NN: Optuna search budget ----
    p.add_argument("--n_trials", type=int, default=35,
                    help="Number of Optuna trials for Stage-2 hyperparameter search.")
    p.add_argument("--max_epochs", type=int, default=200,
                    help="Max training epochs per Optuna trial / model-selection run.")
    p.add_argument("--patience", type=int, default=20,
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
# STAGE 2: NEURAL alpha(h) = 1 + delta_alpha(h)
# =============================================================================

class DeltaAlphaNet(nn.Module):
    HIDDEN_DIMS = (32, 64, 64, 64, 16)

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


def build_stage2_features(physical_results, rp, mw):
    sensor_pos = physical_results[:, 0:3]
    a = physical_results[:, 3]
    g = physical_results[:, 4]
    sensor_dir = physical_results[:, 5:8]

    n_samples = rp.shape[0]
    n_sensors = physical_results.shape[0]

    h_mat = np.zeros((n_samples, n_sensors))
    gB_mat = np.zeros((n_samples, n_sensors))
    for s in range(n_sensors):
        r_vec = sensor_pos[s] - rp
        h_mat[:, s] = rp[:, 2] - sensor_pos[s, 2]
        B = dipole_field(r_vec, mw)
        gB_mat[:, s] = g[s] * (B @ sensor_dir[s])

    a_mat = np.broadcast_to(a[None, :], (n_samples, n_sensors))
    return h_mat.ravel(), gB_mat.ravel(), a_mat.ravel()


def make_tensors(h_flat, gB_flat, a_flat, v_flat, h_mean, h_std):
    h_norm = (h_flat - h_mean) / h_std
    return (
        torch.tensor(h_norm, dtype=torch.float32).unsqueeze(-1),
        torch.tensor(gB_flat, dtype=torch.float32),
        torch.tensor(a_flat, dtype=torch.float32),
        torch.tensor(v_flat, dtype=torch.float32),
    )


def train_alpha_nn(model, train_loader, val_h, val_gB, val_a, val_v,
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
        for h_b, gB_b, a_b, v_b in train_loader:
            h_b, gB_b, a_b, v_b = (t.to(device) for t in (h_b, gB_b, a_b, v_b))
            optimizer.zero_grad()
            delta_alpha = model(h_b)
            alpha = 1.0 + delta_alpha
            v_pred = a_b + gB_b * alpha
            loss_data = huber(v_pred, v_b)
            loss_reg = delta_l2 * (delta_alpha ** 2).mean()
            loss = loss_data + loss_reg
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            delta_alpha_val = model(val_h.to(device))
            alpha_val = 1.0 + delta_alpha_val
            v_pred_val = val_a.to(device) + val_gB.to(device) * alpha_val
            val_rmse = torch.sqrt(torch.mean((v_pred_val - val_v.to(device)) ** 2)).item()

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


def optuna_objective(trial, train_tensors, val_tensors, device, max_epochs, patience):
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    delta_l2 = trial.suggest_float("delta_l2", 1e-6, 1e-1, log=True)
    output_scale_init = trial.suggest_float("output_scale_init", 1e-3, 0.3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64])

    set_seed(42 + trial.number)

    h_tr, gB_tr, a_tr, v_tr = train_tensors
    dataset = TensorDataset(h_tr, gB_tr, a_tr, v_tr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DeltaAlphaNet(output_scale_init=output_scale_init).to(device)

    _, val_rmse, best_epoch = train_alpha_nn(
        model, loader, *val_tensors,
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
    h_train, gB_train, a_train = build_stage2_features(physical_results, rp_train, mw_train)
    v_train = vd_train.ravel()
    h_val, gB_val, a_val = build_stage2_features(physical_results, rp_val, mw_val)
    v_val = vd_val.ravel()
    h_test, gB_test, a_test = build_stage2_features(physical_results, rp_test, mw_test)
    v_test = vd_test.ravel()

    h_mean, h_std = h_train.mean(), h_train.std()

    train_tensors = make_tensors(h_train, gB_train, a_train, v_train, h_mean, h_std)
    val_tensors = make_tensors(h_val, gB_val, a_val, v_val, h_mean, h_std)
    test_tensors = make_tensors(h_test, gB_test, a_test, v_test, h_mean, h_std)

    print(f"\n[Stage 2 / Optuna] {n_trials} trials, "
          f"{len(v_train)} train pairs, {len(v_val)} val pairs, "
          f"{len(v_test)} test pairs (held out, untouched until the end)")
    print(f"[Stage 2] h range train=[{h_train.min():.4f}, {h_train.max():.4f}] m")
    print(f"[Stage 2] Fixed architecture: {DeltaAlphaNet.HIDDEN_DIMS} (SiLU) | device={device}")

    study = optuna.create_study(direction="minimize",
                                 sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(
        lambda trial: optuna_objective(trial, train_tensors, val_tensors,
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
    h_tr, gB_tr, a_tr, v_tr = train_tensors
    loader_train = DataLoader(TensorDataset(h_tr, gB_tr, a_tr, v_tr),
                               batch_size=best["batch_size"], shuffle=True)
    selection_model = DeltaAlphaNet(output_scale_init=best["output_scale_init"]).to(device)
    selection_model, selection_val_rmse, _ = train_alpha_nn(
        selection_model, loader_train, *val_tensors,
        lr=best["lr"], weight_decay=best["weight_decay"], delta_l2=best["delta_l2"],
        device=device, max_epochs=max_epochs, patience=patience,
        early_stop=True, verbose=True,
    )

    selection_model.eval()
    with torch.no_grad():
        h_te, gB_te, a_te, v_te = test_tensors
        delta_alpha_test = selection_model(h_te.to(device))
        alpha_test = 1.0 + delta_alpha_test
        v_pred_test = a_te.to(device) + gB_te.to(device) * alpha_test
        test_rmse = torch.sqrt(torch.mean((v_pred_test - v_te.to(device)) ** 2)).item()

    print(f"\n[Stage 2] HONEST held-out TEST RMSE (never used for Optuna or "
          f"early stopping) = {test_rmse:.6f} V")

    h_full = np.concatenate([h_train, h_val, h_test])
    gB_full = np.concatenate([gB_train, gB_val, gB_test])
    a_full = np.concatenate([a_train, a_val, a_test])
    v_full = np.concatenate([v_train, v_val, v_test])
    full_tensors = make_tensors(h_full, gB_full, a_full, v_full, h_mean, h_std)
    h_f, gB_f, a_f, v_f = full_tensors
    loader_full = DataLoader(TensorDataset(h_f, gB_f, a_f, v_f),
                              batch_size=best["batch_size"], shuffle=True)

    set_seed(seed)
    deploy_model = DeltaAlphaNet(output_scale_init=best["output_scale_init"]).to(device)
    fixed_epochs = best_epoch + 1 if best_epoch >= 0 else max_epochs
    deploy_model, _, _ = train_alpha_nn(
        deploy_model, loader_full, *val_tensors,
        lr=best["lr"], weight_decay=best["weight_decay"], delta_l2=best["delta_l2"],
        device=device, max_epochs=fixed_epochs, patience=patience,
        early_stop=False, verbose=True,
    )

    meta = {
        "h_mean": float(h_mean), "h_std": float(h_std),
        "hidden_dims": "-".join(map(str, DeltaAlphaNet.HIDDEN_DIMS)),
        "activation": "SiLU",
        "output_scale_init": best["output_scale_init"],
        "output_scale_final": deploy_model.output_scale.item(),
        "selection_val_rmse": selection_val_rmse,
        "held_out_test_rmse": test_rmse,
        "deploy_fixed_epochs": fixed_epochs,
    }
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
    torch.save(model.state_dict(), model_path)
    pd.DataFrame([meta]).to_csv(meta_path, index=False)
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


def plot_alpha_curve(model, meta, output_path, device, h_range=(-0.02, 0.15)):
    model.eval()
    h_plot = np.linspace(*h_range, 300)
    h_norm = (h_plot - meta["h_mean"]) / meta["h_std"]
    with torch.no_grad():
        h_t = torch.tensor(h_norm, dtype=torch.float32).unsqueeze(-1).to(device)
        delta_alpha = model(h_t).cpu().numpy()
    alpha = 1.0 + delta_alpha
    plt.figure(figsize=(8, 5))
    plt.plot(h_plot * 1000, alpha)
    plt.xlabel("h (mm)")
    plt.ylabel("alpha(h) = 1 + delta_alpha(h)")
    plt.title("Learned NN correction curve (deploy model)")
    plt.grid(True)
    plt.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=130)
    plt.close()
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

    physical_output_path = output_dir / "Calibration_Physical_NN_h.csv"
    alpha_nn_output_path = output_dir / "Calibration_AlphaNN_h.pt"
    alpha_nn_meta_path = output_dir / "Calibration_AlphaNN_h_meta.csv"
    stage1_rmse_plot_path = output_dir / "stage1_rmse.png"
    alpha_curve_plot_path = output_dir / "alpha_h_curve.png"

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
    print("SAMPLING: Stage1 / Stage2-train / Stage2-val / Stage2-test split (seeded)")
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
    print("STAGE 2: NEURAL alpha(h) CORRECTION (Optuna-tuned)")
    print("===================================")
    deploy_model, meta, study = calibrate_alpha_nn(
        results, rp2t, mw2t, vd2t, rp2v, mw2v, vd2v, rp2te, mw2te, vd2te,
        n_trials=args.n_trials, max_epochs=args.max_epochs, patience=args.patience,
        seed=args.seed, device=device,
    )
    save_alpha_nn(deploy_model, meta, alpha_nn_output_path, alpha_nn_meta_path)
    plot_alpha_curve(deploy_model, meta, alpha_curve_plot_path, device)

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