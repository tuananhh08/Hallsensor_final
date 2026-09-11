import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import least_squares

import torch
from torch import nn

# =============================================================================
# FILE PATHS
# =============================================================================
# BASE_DIR = Path(r"/Users/tuananhnguyen/Downloads/Hallsensor_final/Data_8_2026")  # MAC
BASE_DIR = Path(r"D:\Downloads\Hallsensor_final\Data_8_2026")  # WINDOWS

SENSOR_POSITIONS_PATH = BASE_DIR / "Hall_sensor_positions.csv"
ROBOT_POSE_PATH = BASE_DIR / "Grid_points_coordinates.csv"
VOLTAGE_DATA_PATH = BASE_DIR / "Grid_data.csv"
OFFSET_INIT_PATH = BASE_DIR / "Offset_Sens.csv"

PHYSICAL_OUTPUT_PATH = BASE_DIR / "Calibration_Physical_NN.csv"
NN_MODEL_PATH = BASE_DIR / "Calibration_Alpha_NN.pt"
NN_CONFIG_PATH = BASE_DIR / "Calibration_Alpha_NN_config.json"
NN_HISTORY_PATH = BASE_DIR / "Calibration_Alpha_NN_history.csv"
RMSE_OUTPUT_PATH = BASE_DIR / "Calibration_RMSE_NN.png"
ALPHA_PLOT_PATH = BASE_DIR / "Calibration_Alpha_NN_curve.png"

# =============================================================================
# CONSTANTS / SAMPLE SPLIT
# =============================================================================
MU0_OVER_4PI = 1e-7
N_TOTAL_CALIB_SAMPLES = 800
N_STAGE1_SAMPLES = 300
N_STAGE2_SAMPLES = N_TOTAL_CALIB_SAMPLES - N_STAGE1_SAMPLES

# Inner validation split for Stage 2 NN only.
STAGE2_VAL_FRACTION = 0.20
RANDOM_SEED = 42

# =============================================================================
# STAGE 1 REGULARIZATION
# =============================================================================
LAMBDA_POS = 2000
LAMBDA_GAIN = 9e-3
LAMBDA_OFFSET = 750

# =============================================================================
# STAGE 2 NN
# Shared global network: r -> delta_alpha, alpha = 1 + delta_alpha.
# =============================================================================
NN_HIDDEN_1 = 32
NN_HIDDEN_2 = 32
NN_HIDDEN_3 = 16
NN_EPOCHS = 200
NN_LR = 1e-3
NN_WEIGHT_DECAY = 1e-5
LAMBDA_SMOOTH = 1e-4
NN_PATIENCE = 50
NN_MIN_DELTA = 1e-9

# =============================================================================
# DIPOLE MODEL
# =============================================================================
def dipole_field(r_vec, m_vec):
    r = np.linalg.norm(r_vec, axis=1, keepdims=True)
    r3 = np.maximum(r**3, 1e-12)
    r5 = np.maximum(r**5, 1e-12)
    mdotr = np.sum(m_vec * r_vec, axis=1, keepdims=True)
    return MU0_OVER_4PI * (
        3.0 * r_vec * mdotr / r5 - m_vec / r3
    )

# =============================================================================
# LOADERS
# =============================================================================
def load_sensor_positions(file_path):
    df = pd.read_csv(file_path)
    sensor_positions = df.values.astype(float)
    print(f"Loaded sensor positions: {sensor_positions.shape}")
    return sensor_positions


def load_robot_pose(file_path):
    df = pd.read_csv(file_path)
    required_cols = ["x", "y", "z", "mx", "my", "mz"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    positions = df[["x", "y", "z"]].values.astype(float)
    m_world = df[["mx", "my", "mz"]].values.astype(float)
    norm = np.linalg.norm(m_world, axis=1, keepdims=True)
    if np.any(norm <= 0):
        raise ValueError("Found zero-norm magnetic orientation vector.")
    m_world = m_world / norm
    print(f"Loaded robot positions: {positions.shape}")
    print(f"Loaded magnetic orientations: {m_world.shape}")
    return positions, m_world


def load_voltage_data(file_path):
    df = pd.read_csv(file_path)
    voltage = df.values.astype(float)
    print(f"Loaded voltage data: {voltage.shape}")
    return voltage


def load_offset_initial_values(file_path, n_sensors):
    df = pd.read_csv(file_path)
    required_columns = {"sensor_index", "offset_a_V"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing column(s) in {file_path.name}: {sorted(missing_columns)}"
        )
    if df["sensor_index"].duplicated().any():
        raise ValueError("Offset_Sens.csv contains duplicate sensor_index values.")
    df = df.sort_values("sensor_index").reset_index(drop=True)
    expected_indices = np.arange(n_sensors)
    actual_indices = df["sensor_index"].to_numpy()
    if not np.array_equal(actual_indices, expected_indices):
        raise ValueError(
            "Offset_Sens.csv must contain exactly sensor_index values "
            f"0 to {n_sensors - 1}."
        )
    offsets = df["offset_a_V"].to_numpy(dtype=float)
    if not np.isfinite(offsets).all():
        raise ValueError("Column offset_a_V contains missing or non-finite values.")
    print(f"Loaded offset initial values: {offsets.shape}")
    return offsets

# =============================================================================
# STAGE 1 RESIDUAL
# =============================================================================
def sensor_residuals(
    params,
    robot_positions,
    m_world,
    voltage_sensor,
    pos_prior=None,
    offset_prior=None,
    g0=7.5,
    lambda_pos=LAMBDA_POS,
    lambda_gain=LAMBDA_GAIN,
    lambda_offset=LAMBDA_OFFSET,
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
        raise ValueError("Non-finite Stage 1 voltage residual detected.")

    if pos_prior is not None:
        x0, y0, z0 = pos_prior
        r_pos = np.sqrt(lambda_pos) * np.array([x - x0, y - y0, z - z0])
    else:
        r_pos = np.array([])

    r_gain = np.sqrt(lambda_gain) * np.array([g - g0])

    if offset_prior is not None:
        r_offset = np.sqrt(lambda_offset) * np.array([a - offset_prior])
    else:
        r_offset = np.array([])

    return np.concatenate([r_voltage, r_pos, r_offset, r_gain])

# =============================================================================
# STAGE 1 SINGLE SENSOR
# =============================================================================
def calibrate_single_sensor(
    sensor_index,
    sensor_pos_init,
    robot_positions,
    m_world,
    voltage_sensor,
    offset_init,
):
    g0 = 7.5
    x0 = np.array(
        [
            sensor_pos_init[0],
            sensor_pos_init[1],
            sensor_pos_init[2],
            offset_init,
            g0,
        ],
        dtype=float,
    )
    pos_tol = 0.001
    lower = [
        sensor_pos_init[0] - pos_tol,
        sensor_pos_init[1] - pos_tol,
        sensor_pos_init[2] - pos_tol,
        offset_init - 0.0011,
        7.0,
    ]
    upper = [
        sensor_pos_init[0] + pos_tol,
        sensor_pos_init[1] + pos_tol,
        sensor_pos_init[2] + pos_tol,
        offset_init + 0.0011,
        8.0,
    ]
    result = least_squares(
        sensor_residuals,
        x0,
        bounds=(lower, upper),
        args=(robot_positions, m_world, voltage_sensor),
        kwargs=dict(
            pos_prior=(sensor_pos_init[0], sensor_pos_init[1], sensor_pos_init[2]),
            offset_prior=offset_init,
            g0=g0,
        ),
        method="trf",
        max_nfev=250,
    )
    params_opt = result.x
    nx, ny, nz = 0.0, 0.0, 1.0
    theta_opt, phi_opt = 0.0, 0.0
    params_extended = np.array(
        [
            params_opt[0], params_opt[1], params_opt[2], params_opt[3], params_opt[4],
            nx, ny, nz, theta_opt, phi_opt,
        ]
    )
    n_voltage = voltage_sensor.shape[0]
    rmse = np.sqrt(np.mean(result.fun[:n_voltage] ** 2))
    print(
        f"Sensor {sensor_index + 1:02d} | RMSE = {rmse:.6f} V | "
        f"x={params_opt[0]:.6f}, y={params_opt[1]:.6f}, "
        f"z={params_opt[2]:.6f}, a={params_opt[3]:.6f}, "
        f"g={params_opt[4]:.6f}"
    )
    return params_extended, rmse

# =============================================================================
# STAGE 1 ALL SENSORS
# =============================================================================
def run_calibration(
    sensor_positions,
    robot_positions,
    m_world,
    voltage_data,
    offset_initial_values,
):
    n_sensors = sensor_positions.shape[0]
    if len(offset_initial_values) != n_sensors:
        raise ValueError(
            f"Expected {n_sensors} initial offsets, got {len(offset_initial_values)}."
        )
    results, rmses = [], []
    for i in range(n_sensors):
        params, rmse = calibrate_single_sensor(
            sensor_index=i,
            sensor_pos_init=sensor_positions[i],
            robot_positions=robot_positions,
            m_world=m_world,
            voltage_sensor=voltage_data[:, i],
            offset_init=offset_initial_values[i],
        )
        results.append(params)
        rmses.append(rmse)
    return np.array(results), np.array(rmses)

# =============================================================================
# RANDOM SPLIT
# =============================================================================
def select_stage1_stage2_split(
    robot_positions,
    m_world,
    voltage_data,
    n_total=N_TOTAL_CALIB_SAMPLES,
    n_stage1=N_STAGE1_SAMPLES,
):
    n_samples = robot_positions.shape[0]
    if n_total > n_samples:
        raise ValueError(
            f"Requested {n_total} calibration points but dataset only has {n_samples} samples."
        )
    all_idx = np.random.choice(n_samples, size=n_total, replace=False)
    stage1_idx = all_idx[:n_stage1]
    stage2_idx = all_idx[n_stage1:]
    print(f"\n[Sampling] Drew {n_total} random points out of {n_samples}.")
    print(f"  Stage 1: {len(stage1_idx)} points")
    print(f"  Stage 2: {len(stage2_idx)} points")
    stage1_data = (
        stage1_idx,
        robot_positions[stage1_idx],
        m_world[stage1_idx],
        voltage_data[stage1_idx],
    )
    stage2_data = (
        stage2_idx,
        robot_positions[stage2_idx],
        m_world[stage2_idx],
        voltage_data[stage2_idx],
    )
    return stage1_data, stage2_data

# =============================================================================
# STAGE 2 TRAIN/VALIDATION SPLIT
# =============================================================================
def split_stage2_train_val(stage2_idx, rp2, mw2, vd2):
    n = len(stage2_idx)
    if n < 2:
        raise ValueError("Stage 2 needs at least 2 samples for train/validation.")
    rng = np.random.default_rng(RANDOM_SEED)
    perm = rng.permutation(n)
    n_val = max(1, int(round(n * STAGE2_VAL_FRACTION)))
    val_local = perm[:n_val]
    train_local = perm[n_val:]
    return (
        stage2_idx[train_local],
        rp2[train_local],
        mw2[train_local],
        vd2[train_local],
        stage2_idx[val_local],
        rp2[val_local],
        mw2[val_local],
        vd2[val_local],
    )

# =============================================================================
# BUILD FROZEN STAGE 1 FEATURES FOR STAGE 2
# =============================================================================
def build_stage2_features(
    physical_results,
    robot_positions,
    m_world,
    voltage_data,
):
    n_samples = robot_positions.shape[0]
    n_sensors = physical_results.shape[0]
    sensor_pos = physical_results[:, 0:3]
    offset = physical_results[:, 3]
    gain = physical_results[:, 4]
    sensor_dir = physical_results[:, 5:8]

    r_distance = np.zeros((n_samples, n_sensors), dtype=np.float64)
    B_proj = np.zeros((n_samples, n_sensors), dtype=np.float64)

    for s in range(n_sensors):
        r_vec = sensor_pos[s] - robot_positions
        r_distance[:, s] = np.linalg.norm(r_vec, axis=1)
        B = dipole_field(r_vec, m_world)
        B_proj[:, s] = B @ sensor_dir[s]

    gB = B_proj * gain[None, :]
    v_minus_a = voltage_data - offset[None, :]

    return r_distance, B_proj, gB, v_minus_a

# =============================================================================
# NORMALIZATION
# =============================================================================
def fit_r_normalization(r_train):
    mean = float(np.mean(r_train))
    std = float(np.std(r_train))
    if std < 1e-12:
        std = 1.0
    return mean, std


def normalize_r(r, mean, std):
    return (r - mean) / std

# =============================================================================
# NN MODEL
# =============================================================================
class AlphaCorrectionNN(nn.Module):
    """
    Shared global correction network:

        r -> MLP -> delta_alpha
        alpha(r) = 1 + delta_alpha(r)

    Final layer is zero-initialized, so the initial network gives alpha(r)=1.
    """

    def __init__(self, hidden1=NN_HIDDEN_1, hidden2=NN_HIDDEN_2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden1),
            nn.SiLU(),
            nn.Linear(hidden1, hidden2),
            nn.SiLU(),
            nn.Linear(hidden2, 1),
        )
        last = self.net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, r_norm):
        delta_alpha = self.net(r_norm)
        alpha = 1.0 + delta_alpha
        return alpha, delta_alpha

# =============================================================================
# TORCH TENSORS
# =============================================================================
def make_stage2_tensors(r_distance, gB, v_minus_a, r_mean, r_std, device):
    r_norm = normalize_r(r_distance, r_mean, r_std).reshape(-1, 1)
    gB_flat = gB.reshape(-1, 1)
    y_flat = v_minus_a.reshape(-1, 1)
    return (
        torch.tensor(r_norm, dtype=torch.float32, device=device),
        torch.tensor(gB_flat, dtype=torch.float32, device=device),
        torch.tensor(y_flat, dtype=torch.float32, device=device),
    )

# =============================================================================
# FORWARD MODEL
# =============================================================================
def voltage_prediction_from_alpha(model, r_norm, gB):
    alpha, delta_alpha = model(r_norm)
    v_pred_minus_a = gB * alpha
    return v_pred_minus_a, alpha, delta_alpha

# =============================================================================
# SMOOTHNESS REGULARIZATION
# =============================================================================
def smoothness_loss(
    model,
    r_min,
    r_max,
    r_mean,
    r_std,
    device,
    n_grid=256,
):
    r_grid = torch.linspace(
        r_min, r_max, n_grid, device=device
    ).reshape(-1, 1)
    r_grid_norm = (r_grid - r_mean) / r_std
    _, delta = model(r_grid_norm)
    d2 = delta[2:] - 2.0 * delta[1:-1] + delta[:-2]
    return torch.mean(d2**2)

# =============================================================================
# EVALUATE
# =============================================================================
@torch.no_grad()
def evaluate_stage2(
    model,
    r_distance,
    gB,
    v_minus_a,
    r_mean,
    r_std,
    device,
):
    model.eval()
    r_norm, gB_t, y_t = make_stage2_tensors(
        r_distance, gB, v_minus_a, r_mean, r_std, device
    )
    v_pred_minus_a, alpha, delta_alpha = voltage_prediction_from_alpha(
        model, r_norm, gB_t
    )
    residual = y_t - v_pred_minus_a
    mse = torch.mean(residual**2)
    rmse = torch.sqrt(mse)
    mae = torch.mean(torch.abs(residual))
    return {
        "rmse": float(rmse.item()),
        "mae": float(mae.item()),
        "alpha_min": float(alpha.min().item()),
        "alpha_max": float(alpha.max().item()),
        "alpha_mean": float(alpha.mean().item()),
        "delta_mean": float(delta_alpha.mean().item()),
    }

# =============================================================================
# TRAIN NN
# =============================================================================
def train_alpha_nn(
    r_train,
    gB_train,
    v_minus_a_train,
    r_val,
    gB_val,
    v_minus_a_val,
    device,
):
    r_mean, r_std = fit_r_normalization(r_train)

    r_train_t, gB_train_t, y_train_t = make_stage2_tensors(
        r_train, gB_train, v_minus_a_train, r_mean, r_std, device
    )
    r_val_t, gB_val_t, y_val_t = make_stage2_tensors(
        r_val, gB_val, v_minus_a_val, r_mean, r_std, device
    )

    model = AlphaCorrectionNN().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=NN_LR,
        weight_decay=NN_WEIGHT_DECAY,
    )

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    history = []

    r_min = float(np.min(r_train))
    r_max = float(np.max(r_train))

    for epoch in range(1, NN_EPOCHS + 1):
        model.train()

        v_pred_minus_a, alpha, delta_alpha = voltage_prediction_from_alpha(
            model, r_train_t, gB_train_t
        )

        residual = y_train_t - v_pred_minus_a
        voltage_loss = torch.mean(residual**2)
        smooth_loss = smoothness_loss(
            model,
            r_min,
            r_max,
            r_mean,
            r_std,
            device,
        )
        loss = voltage_loss + LAMBDA_SMOOTH * smooth_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            v_val_minus_a, alpha_val, delta_val = voltage_prediction_from_alpha(
                model, r_val_t, gB_val_t
            )
            val_residual = y_val_t - v_val_minus_a
            val_loss = torch.mean(val_residual**2)
            val_rmse = torch.sqrt(val_loss)
            val_mae = torch.mean(torch.abs(val_residual))

        train_rmse = torch.sqrt(voltage_loss)

        history.append({
            "epoch": epoch,
            "train_voltage_mse": float(voltage_loss.item()),
            "train_rmse_V": float(train_rmse.item()),
            "smooth_loss": float(smooth_loss.item()),
            "total_train_loss": float(loss.item()),
            "val_mse": float(val_loss.item()),
            "val_rmse_V": float(val_rmse.item()),
            "val_mae_V": float(val_mae.item()),
            "alpha_min": float(alpha_val.min().item()),
            "alpha_max": float(alpha_val.max().item()),
        })

        current_val = float(val_loss.item())
        if current_val < best_val_loss - NN_MIN_DELTA:
            best_val_loss = current_val
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch == 1 or epoch % 100 == 0:
            print(
                f"[NN] Epoch {epoch:04d} | "
                f"Train RMSE = {train_rmse.item():.8f} V | "
                f"Val RMSE = {val_rmse.item():.8f} V | "
                f"Alpha range = "
                f"[{alpha_val.min().item():.4f}, {alpha_val.max().item():.4f}]"
            )

        if patience_counter >= NN_PATIENCE:
            print(f"[NN] Early stopping at epoch {epoch}.")
            break

    if best_state is None:
        raise RuntimeError("NN training produced no valid best state.")

    model.load_state_dict(best_state)

    return model, pd.DataFrame(history), r_mean, r_std

# =============================================================================
# SAVE STAGE 1
# =============================================================================
def save_physical_results(results, rmses, output_file):
    df = pd.DataFrame({
        "sensor_index": np.arange(len(results)),
        "x": results[:, 0],
        "y": results[:, 1],
        "z": results[:, 2],
        "offset": results[:, 3],
        "gain": results[:, 4],
    })
    df.to_csv(output_file, index=False)
    print(f"\nSaved Stage 1: {output_file}")

# =============================================================================
# SAVE NN
# =============================================================================
def save_nn(model, r_mean, r_std, history_df):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "hidden_1": NN_HIDDEN_1,
            "hidden_2": NN_HIDDEN_2,
            "r_mean": r_mean,
            "r_std": r_std,
            "formula": "alpha(r) = 1 + delta_alpha(r)",
        },
        NN_MODEL_PATH,
    )

    config = {
        "hidden_1": NN_HIDDEN_1,
        "hidden_2": NN_HIDDEN_2,
        "epochs": NN_EPOCHS,
        "learning_rate": NN_LR,
        "weight_decay": NN_WEIGHT_DECAY,
        "lambda_smooth": LAMBDA_SMOOTH,
        "r_mean": r_mean,
        "r_std": r_std,
        "formula": "alpha(r) = 1 + delta_alpha(r)",
    }

    with open(NN_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    history_df.to_csv(NN_HISTORY_PATH, index=False)

    print(f"Saved NN model: {NN_MODEL_PATH}")
    print(f"Saved NN config: {NN_CONFIG_PATH}")
    print(f"Saved training history: {NN_HISTORY_PATH}")

# =============================================================================
# PLOTS
# =============================================================================
def plot_rmse(rmses, output_file):
    plt.figure(figsize=(10, 5))
    plt.bar(np.arange(len(rmses)), rmses)
    plt.xlabel("Sensor Index")
    plt.ylabel("RMSE (V)")
    plt.title("Stage 1 Calibration RMSE")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()


@torch.no_grad()
def plot_alpha_curve(
    model,
    r_mean,
    r_std,
    r_min,
    r_max,
    device,
    output_file,
):
    model.eval()
    r_grid = np.linspace(r_min, r_max, 500)
    r_norm = normalize_r(r_grid, r_mean, r_std).reshape(-1, 1)
    r_t = torch.tensor(r_norm, dtype=torch.float32, device=device)
    alpha, delta = model(r_t)
    alpha_np = alpha.cpu().numpy().reshape(-1)
    delta_np = delta.cpu().numpy().reshape(-1)

    fig, axes = plt.subplots(2, 1, figsize=(9, 9))

    axes[0].plot(
        r_grid,
        alpha_np,
        linewidth=2.0,
        label=r"$\alpha(r)=1+\Delta\alpha(r)$",
    )
    axes[0].axhline(
        1.0,
        linestyle="--",
        color="black",
        linewidth=1.0,
        label=r"$\alpha=1$",
    )
    axes[0].set_xlabel("r (m)")
    axes[0].set_ylabel(r"$\alpha(r)$")
    axes[0].set_title("Learned Global Correction Function")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        r_grid,
        delta_np,
        linewidth=2.0,
        label=r"$\Delta\alpha(r)$",
    )
    axes[1].axhline(
        0.0,
        linestyle="--",
        color="black",
        linewidth=1.0,
    )
    axes[1].set_xlabel("r (m)")
    axes[1].set_ylabel(r"$\Delta\alpha(r)$")
    axes[1].set_title("Learned Correction Term")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_file, dpi=200)
    plt.close(fig)
    print(f"Saved alpha curve: {output_file}")

# =============================================================================
# MAIN
# =============================================================================
def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n===================================")
    print("2-STAGE CALIBRATION")
    print("Version 2: NN alpha(r)")
    print(f"Device: {device}")
    print("===================================\n")

    sensor_positions = load_sensor_positions(SENSOR_POSITIONS_PATH)
    robot_positions, m_world = load_robot_pose(ROBOT_POSE_PATH)
    voltage_data = load_voltage_data(VOLTAGE_DATA_PATH)
    offsets = load_offset_initial_values(
        OFFSET_INIT_PATH,
        n_sensors=sensor_positions.shape[0],
    )

    n_samples = min(len(robot_positions), len(voltage_data))
    robot_positions = robot_positions[:n_samples]
    m_world = m_world[:n_samples]
    voltage_data = voltage_data[:n_samples]

    if sensor_positions.shape[0] != voltage_data.shape[1]:
        raise ValueError(
            f"Sensor count mismatch: {sensor_positions.shape[0]} calibration sensors vs "
            f"{voltage_data.shape[1]} voltage columns."
        )

    stage1_data, stage2_data = select_stage1_stage2_split(
        robot_positions,
        m_world,
        voltage_data,
    )

    _, rp1, mw1, vd1 = stage1_data
    stage2_idx, rp2, mw2, vd2 = stage2_data

    # =====================================================================
    # STAGE 1
    # =====================================================================
    print("\n===================================")
    print("STAGE 1: PHYSICAL PARAMETER FIT")
    print("===================================\n")

    physical_results, rmses = run_calibration(
        sensor_positions,
        rp1,
        mw1,
        vd1,
        offsets,
    )

    print("\nStage 1 statistics:")
    print(f"Mean RMSE = {np.mean(rmses):.6f} V")
    print(f"Min RMSE  = {np.min(rmses):.6f} V")
    print(f"Max RMSE  = {np.max(rmses):.6f} V")

    save_physical_results(
        physical_results,
        rmses,
        PHYSICAL_OUTPUT_PATH,
    )
    plot_rmse(rmses, RMSE_OUTPUT_PATH)

    # =====================================================================
    # FREEZE STAGE 1
    # =====================================================================
    print("\n===================================")
    print("FREEZING STAGE 1 PARAMETERS")
    print("===================================")

    # =====================================================================
    # STAGE 2 TRAIN/VALIDATION SPLIT
    # =====================================================================
    (
        stage2_train_idx,
        rp2_train,
        mw2_train,
        vd2_train,
        stage2_val_idx,
        rp2_val,
        mw2_val,
        vd2_val,
    ) = split_stage2_train_val(
        stage2_idx,
        rp2,
        mw2,
        vd2,
    )

    print(f"\nStage 2 train samples: {len(stage2_train_idx)}")
    print(f"Stage 2 validation samples: {len(stage2_val_idx)}")

    # =====================================================================
    # BUILD FROZEN FEATURES
    # =====================================================================
    r_train, B_train, gB_train, vma_train = build_stage2_features(
        physical_results,
        rp2_train,
        mw2_train,
        vd2_train,
    )

    r_val, B_val, gB_val, vma_val = build_stage2_features(
        physical_results,
        rp2_val,
        mw2_val,
        vd2_val,
    )

    print("\nStage 2 feature shapes:")
    print(f"  r_train  : {r_train.shape}")
    print(f"  gB_train : {gB_train.shape}")
    print(f"  r_val    : {r_val.shape}")

    # =====================================================================
    # STAGE 2 NN
    # =====================================================================
    print("\n===================================")
    print("STAGE 2: NN CORRECTION alpha(r)")
    print("alpha(r) = 1 + delta_alpha(r)")
    print("===================================\n")

    model, history_df, r_mean, r_std = train_alpha_nn(
        r_train,
        gB_train,
        vma_train,
        r_val,
        gB_val,
        vma_val,
        device,
    )

    train_metrics = evaluate_stage2(
        model,
        r_train,
        gB_train,
        vma_train,
        r_mean,
        r_std,
        device,
    )

    val_metrics = evaluate_stage2(
        model,
        r_val,
        gB_val,
        vma_val,
        r_mean,
        r_std,
        device,
    )

    print("\n===================================")
    print("FINAL STAGE 2 METRICS")
    print("===================================")
    print(f"Train RMSE = {train_metrics['rmse']:.8f} V")
    print(f"Train MAE  = {train_metrics['mae']:.8f} V")
    print(f"Val RMSE   = {val_metrics['rmse']:.8f} V")
    print(f"Val MAE    = {val_metrics['mae']:.8f} V")
    print(
        f"Val alpha range = "
        f"[{val_metrics['alpha_min']:.6f}, {val_metrics['alpha_max']:.6f}]"
    )

    save_nn(
        model,
        r_mean,
        r_std,
        history_df,
    )

    r_plot_min = min(
        float(np.min(r_train)),
        float(np.min(r_val)),
    )
    r_plot_max = max(
        float(np.max(r_train)),
        float(np.max(r_val)),
    )

    plot_alpha_curve(
        model,
        r_mean,
        r_std,
        r_plot_min,
        r_plot_max,
        device,
        ALPHA_PLOT_PATH,
    )

    print("\n===================================")
    print("ALL STAGES FINISHED")
    print("===================================")


if __name__ == "__main__":
    main()
