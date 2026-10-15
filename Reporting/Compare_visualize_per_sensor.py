# alpha(r) per sensor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# FILE PATHS  
# =============================================================================
DATASET_DIR = Path(r"/Users/tuananhnguyen/Downloads/Hallsensor_final/Dataset") #MAC
PHYSICAL_PATH = Path(r"/Users/tuananhnguyen/Downloads/Hallsensor_final/Calibration/Parameters/Linear_r_per_sensor") #MAC
OUTPUT_DIR = Path(r"/Users/tuananhnguyen/Downloads/Hallsensor_final/Reporting/calibration_linear_r_per_sensor") #MAC

# BASE_DIR = Path(r"D:\Downloads\Hallsensor_final\Data_8_2026") #WINDOWS

PHYSICAL_PARAMS = PHYSICAL_PATH / "Calibration_Parameters_r_per_sensor.csv"
ALPHA_PATH = PHYSICAL_PATH / "Calibration_Alpha_r_per_sensor.csv"

VOLTAGE_PATH = DATASET_DIR / "Helix_data_2.csv"
COORDS_PATH = DATASET_DIR / "Helix_points_coordinates_2.csv"

# VOLTAGE_PATH = BASE_DIR / "Grid_data.csv"
# COORDS_PATH = BASE_DIR / "Grid_points_coordinates.csv"
# OUTPUT_DIR = BASE_DIR / "outputs/sensor_plots_Grid_alpha(r)_per_sensor"

Helix2 = OUTPUT_DIR / "Helix2"
Helix2.mkdir(parents=True, exist_ok=True)

VISUALIZATION = Helix2 / "Visualization"
VISUALIZATION.mkdir(parents=True, exist_ok=True)

RMSE_SUMMARY_PATH = Helix2 / "rmse_summary_alpha(r)_per_sensor.csv"

MU0_OVER_4PI = 1e-7


# =============================================================================
# DIPOLE MODEL 
# =============================================================================
def dipole_field(r_vec, m_vec):
    """Calculate magnetic field from dipole model.
    r_vec: (N,3) vector from source(capsule) to sensor
    m_vec: (N,3) unit magnetic moment vector
    """
    r = np.linalg.norm(r_vec, axis=1, keepdims=True)
    r3 = np.maximum(r ** 3, 1e-12)
    r5 = np.maximum(r ** 5, 1e-12)

    mdotr = np.sum(m_vec * r_vec, axis=1, keepdims=True)

    B = MU0_OVER_4PI * (
        3.0 * r_vec * mdotr / r5 - m_vec / r3
    )
    return B


# =============================================================================
# LOAD DATA
# =============================================================================
def load_physical_calib(path):
    """sensor_index, x, y, z, offset, gain, theta, phi"""
    df = pd.read_csv(path)
    df = df.sort_values("sensor_index").reset_index(drop=True)
    return df


def load_alpha_per_sensor(path, n_sensors):
    """sensor_index, c0, c1, rmse -> DataFrame indexed by sensor_index,
    one (c0, c1) pair per sensor (alpha_s(r) = c0_s + c1_s*r)."""
    df = pd.read_csv(path)
    df = df.sort_values("sensor_index").reset_index(drop=True)
    expected_indices = np.arange(n_sensors)
    actual_indices = df["sensor_index"].to_numpy()
    if not np.array_equal(actual_indices, expected_indices):
        raise ValueError(
            "Calibration_Alpha_r_per_sensor.csv must contain exactly "
            f"sensor_index values 0 to {n_sensors - 1}."
        )
    return df


def load_voltage_data(path):
    """grid_data.csv: moi cot la 1 sensor, moi dong la 1 mau"""
    df = pd.read_csv(path)
    return df.values, list(df.columns)


def load_robot_pose(path):
    """grid_points_coordinates.csv: x,y,z,mx,my,mz"""
    df = pd.read_csv(path)
    positions = df[["x", "y", "z"]].values
    m_world = df[["mx", "my", "mz"]].values
    norm = np.linalg.norm(m_world, axis=1, keepdims=True)
    m_world = m_world / norm
    return positions, m_world


# =============================================================================
# ALPHA(R) LOOKUP -- PER SENSOR
# =============================================================================
def alpha_for_r(r, c0, c1):
    """r: (N,) array = khoang cach tu capsule den sensor. Tra ve (N,) mang
    alpha_s(r) = c0_s + c1_s*r cho MOT sensor duy nhat (c0, c1 la scalar
    cua sensor do, khac voi ban global chi co 1 cap (c0, c1) chung)."""
    return c0 + c1 * r


# =============================================================================
# COMPUTE V_pred FOR ONE SENSOR
# =============================================================================
def compute_vpred_for_sensor(sensor_row, robot_positions, m_world, c0, c1):
    x, y, z = sensor_row["x"], sensor_row["y"], sensor_row["z"]
    a = sensor_row["offset"]
    g = sensor_row["gain"]
    # Huong sensor co dinh thang dung (theta=phi=0 trong file calib)
    sensor_dir = np.array([0.0, 0.0, 1.0])

    sensor_pos = np.array([x, y, z])
    r_vec = sensor_pos - robot_positions          # (N,3)

    B = dipole_field(r_vec, m_world)              # (N,3)
    B_proj = B @ sensor_dir                       # (N,)
    r_distance = np.linalg.norm(r_vec, axis=1)     # (N,)

    alpha_sample = alpha_for_r(r_distance, c0, c1)   # (N,) -- rieng cho sensor nay

    v_pred = a + alpha_sample * g * B_proj

    return v_pred


# =============================================================================
# MAIN
# =============================================================================
def main():
    physical_df = load_physical_calib(PHYSICAL_PARAMS)
    voltage_data, voltage_cols = load_voltage_data(VOLTAGE_PATH)
    robot_positions, m_world = load_robot_pose(COORDS_PATH)

    n_samples_v = voltage_data.shape[0]
    n_samples_pos = robot_positions.shape[0]
    assert n_samples_v == n_samples_pos, (
        f"So mau dien ap ({n_samples_v}) khac so mau toa do ({n_samples_pos})"
    )

    n_sensors = physical_df.shape[0]
    assert n_sensors == voltage_data.shape[1], (
        f"So sensor trong file calib ({n_sensors}) khac so cot dien ap "
        f"({voltage_data.shape[1]})"
    )

    alpha_df = load_alpha_per_sensor(ALPHA_PATH, n_sensors)

    sample_idx = np.arange(n_samples_v)

    per_sensor_rmse = []
    per_sensor_rmse_pct = []
    all_v_meas = []
    all_v_pred = []

    for s in range(n_sensors):
        sensor_row = physical_df.iloc[s]
        alpha_row = alpha_df.iloc[s]
        c0_s = alpha_row["c0"]
        c1_s = alpha_row["c1"]

        v_meas = voltage_data[:, s]
        v_pred = compute_vpred_for_sensor(
            sensor_row, robot_positions, m_world, c0_s, c1_s
        )

        rmse_s = np.sqrt(np.mean((v_meas - v_pred) ** 2))
        mean_abs_vmeas_s = np.mean(np.abs(v_meas))
        rmse_pct_s = (rmse_s / mean_abs_vmeas_s * 100.0) if mean_abs_vmeas_s > 0 else float("nan")
        per_sensor_rmse.append(rmse_s)
        per_sensor_rmse_pct.append(rmse_pct_s)

        all_v_meas.append(v_meas)
        all_v_pred.append(v_pred)

        # ---- Plot ----
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(sample_idx, v_meas, label="V measured", linewidth=1.0)
        ax.plot(sample_idx, v_pred, label="V computed",linewidth=1.0)
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Voltage (V)")
        ax.set_title(
            f"Sensor {s+1:02d} | RMSE = {rmse_s:.6f} V  |  Relative RMSE = {rmse_pct_s:.2f} %"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(VISUALIZATION / f"sensor_{s+1:02d}.png", dpi=120)
        plt.close(fig)

        print(f"Sensor {s+1:02d} | RMSE = {rmse_s:.6f} V  |  Relative RMSE = {rmse_pct_s:.2f} %")

    # ---- Overall RMSE ----
    all_v_meas = np.concatenate(all_v_meas)
    all_v_pred = np.concatenate(all_v_pred)
    overall_rmse = np.sqrt(np.mean((all_v_meas - all_v_pred) ** 2))
    overall_mean_abs = np.mean(np.abs(all_v_meas))
    overall_rmse_pct = (overall_rmse / overall_mean_abs * 100.0) if overall_mean_abs > 0 else float("nan")

    print(f"\n==== OVERALL RMSE (all sensors, all samples) = "
          f"{overall_rmse:.6f} V  |  Relative RMSE = {overall_rmse_pct:.2f} % ====")

    # ---- Save RMSE summary CSV ----
    summary_df = pd.DataFrame({
        "sensor_index": np.arange(1, n_sensors + 1),
        "rmse": per_sensor_rmse,
        "rmse_pct": per_sensor_rmse_pct
    })
    summary_df.loc[len(summary_df)] = ["OVERALL", overall_rmse, overall_rmse_pct]
    summary_df.to_csv(RMSE_SUMMARY_PATH, index=False)

    print(f"\nDa luu {n_sensors} anh vao: {VISUALIZATION}")
    print(f"Da luu bang RMSE vao: {RMSE_SUMMARY_PATH}")


if __name__ == "__main__":
    main()

# NN_r calibration

import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "Calibration")
)

from calibration_nn_r import MLPCalibration

# =============================================================================
# CONFIGURATION
# =============================================================================
#
# This visualization follows the SAME Stage-2 pipeline as:
#     Calibration_NN_alpha_r_only.py
#
# Stage 1:
#     V_stage1 = offset + gain * Bz
#
# Stage 2:
#     r_i = ||sensor_i - capsule_position||
#     alpha_i = 1 + f_theta(r_i)
#     V_NN = offset + gain * Bz * alpha
#
# IMPORTANT:
# - The Stage-2 NN receives ONLY distance r.
# - The same scalar MLP is applied independently to every sensor:
#       (B, 64) -> (B*64, 1) -> MLP -> (B, 64)
# - No voltage vector is used as NN input.
# - No linear alpha(r) baseline is used.
# - Stage-1 physical parameters are frozen.
#
# Change only BASE_DIR if your files are stored elsewhere.
# =============================================================================

# PATHS 
DATASET_DIR = Path(r"/Users/tuananhnguyen/Downloads/Hallsensor_final/Dataset")  # MAC
PHYSICAL_PATH = Path(r"/Users/tuananhnguyen/Downloads/Hallsensor_final/Calibration/Parameters/NonLinear_r_per_sensor") #MAC
OUTPUT_DIR = Path(r"/Users/tuananhnguyen/Downloads/Hallsensor_final/Reporting/calibration_non_linear_r_per_sensor") #MAC


# =============================================================================
# INPUT FILES 
# =============================================================================

# Output directory created by the calibration pipeline.
# CALIB_OUTPUT_DIR = DATASET_DIR / "calibration_nn_r_outputs"

# Stage-1 physical parameters produced by the pipeline.
PHYSICAL_PARAMS = PHYSICAL_PATH / "Physical_parameters.csv"

# Stage-2 NN checkpoint produced by the pipeline.
NN_CKPT_PATH = PHYSICAL_PATH / "Alpha_r_nn.pt"

# Optional distance scaler exported by the pipeline.
SCALER_PATH = PHYSICAL_PATH / "stage2_distance_scaler.json"

# External/raw measurement data to visualize.
VOLTAGE_PATH = DATASET_DIR / "Helix_data_2.csv"
COORDS_PATH = DATASET_DIR / "Helix_points_coordinates_2.csv"


# =============================================================================
# OUTPUT
# =============================================================================
OUTPUT_HELIX2 = OUTPUT_DIR / "Helix2"
OUTPUT_HELIX2.mkdir(parents=True, exist_ok=True)

VISUALIZATION = OUTPUT_HELIX2 / "visualization"
VISUALIZATION.mkdir(parents=True, exist_ok=True)

RMSE_SUMMARY_PATH = OUTPUT_HELIX2 / "rmse_stage1_vs_nn.csv"
ALPHA_PATH = OUTPUT_HELIX2 / "alpha_nn_all_samples.csv"
VOLTAGE_PRED_PATH = OUTPUT_HELIX2 / "voltage_predictions_stage1_vs_nn.csv"


# =============================================================================
# CONSTANTS
# =============================================================================

MU0_OVER_4PI = 1e-7
N_SENSORS = 64


# =============================================================================
# DIPOLE MODEL
# =============================================================================

def dipole_field(r_vec, m_vec):
    """
    r_vec : (N, 3)
    m_vec : (N, 3)
    """
    r = np.linalg.norm(r_vec, axis=1, keepdims=True)

    r3 = np.maximum(r ** 3, 1e-12)
    r5 = np.maximum(r ** 5, 1e-12)

    mdotr = np.sum(m_vec * r_vec, axis=1, keepdims=True)

    B = MU0_OVER_4PI * (
        3.0 * r_vec * mdotr / r5
        - m_vec / r3
    )

    return B


# =============================================================================
# STAGE-2 NN
# =============================================================================

class MLPCalibration(nn.Module):
    """
    Input:
        r_raw : (B, 64)

    Internally:
        (B, 64)
          -> normalize r
          -> reshape to (B*64, 1)
          -> shared scalar MLP
          -> reshape back to (B, 64)

    Output:
        delta_alpha : (B, 64)

    alpha = 1 + delta_alpha
    """

    def __init__(
        self,
        hidden_dim=64,
        n_hidden_layers=2,
        output_scale_init=0.05,
    ):
        super().__init__()

        layers = []
        in_dim = 1

        for _ in range(n_hidden_layers):
            layers += [
                nn.Linear(in_dim, hidden_dim),
                nn.SiLU(),
            ]
            in_dim = hidden_dim

        layers += [nn.Linear(in_dim, 1)]

        self.net = nn.Sequential(*layers)

        # Kept exactly as in the training pipeline.
        self.output_scale = nn.Parameter(
            torch.tensor(float(output_scale_init))
        )

        # These are buffers in the training pipeline and therefore
        # are saved inside model_state_dict.
        self.register_buffer("r_mean", torch.tensor(0.0))
        self.register_buffer("r_std", torch.tensor(1.0))

    def set_normalization(self, r_mean, r_std):
        self.r_mean.fill_(float(r_mean))
        self.r_std.fill_(max(float(r_std), 1e-8))

    def forward(self, r_raw):
        shape = r_raw.shape

        r_norm = (
            r_raw - self.r_mean
        ) / self.r_std

        flat = r_norm.reshape(-1, 1)

        delta_flat = (
            self.net(flat)
            * self.output_scale
        )

        return delta_flat.reshape(shape)


# =============================================================================
# LOADERS
# =============================================================================

def load_physical_calibration(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Stage-1 physical calibration not found:\n{path}"
        )

    df = pd.read_csv(path)

    required = [
        "sensor_index",
        "x",
        "y",
        "z",
        "offset",
        "gain",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in physical_params.csv: {missing}"
        )

    df = (
        df.sort_values("sensor_index")
        .reset_index(drop=True)
    )

    if len(df) != N_SENSORS:
        raise ValueError(
            f"Expected {N_SENSORS} sensors, got {len(df)}."
        )

    print(
        f"Loaded Stage-1 physical calibration: "
        f"{df.shape}"
    )

    return df


def load_nn_checkpoint(path, device):
    if not path.exists():
        raise FileNotFoundError(
            f"Stage-2 NN checkpoint not found:\n{path}"
        )

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    if checkpoint.get("kind") != "nn":
        raise ValueError(
            "The checkpoint is not marked as the NN alpha(r) checkpoint."
        )

    hidden_dim = int(
        checkpoint.get("hidden_dim", 64)
    )

    n_hidden_layers = int(
        checkpoint.get("mlp_n_hidden_layers", 2)
    )

    output_scale_init = float(
        checkpoint.get("output_scale_init", 0.05)
    )

    model = MLPCalibration(
        hidden_dim=hidden_dim,
        n_hidden_layers=n_hidden_layers,
        output_scale_init=output_scale_init,
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.eval()

    print("\nLoaded Stage-2 NN checkpoint:")
    print(f"  {path}")
    print(
        "  Architecture: "
        f"1 -> {hidden_dim} -> SiLU "
        f"(x{n_hidden_layers}) -> 1"
    )
    print(
        f"  output_scale_init = "
        f"{output_scale_init:.8f}"
    )

    # r_mean/r_std are stored in model_state_dict.
    r_mean = (
        checkpoint["model_state_dict"]["r_mean"]
        .detach()
        .cpu()
        .item()
    )

    r_std = (
        checkpoint["model_state_dict"]["r_std"]
        .detach()
        .cpu()
        .item()
    )

    print(f"  r_mean = {r_mean:.9f} m")
    print(f"  r_std  = {r_std:.9f} m")

    return model, checkpoint


def load_voltage(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Voltage file not found:\n{path}"
        )

    df = pd.read_csv(path)

    voltage = df.values.astype(float)

    if voltage.shape[1] != N_SENSORS:
        raise ValueError(
            f"Expected {N_SENSORS} voltage columns, "
            f"got {voltage.shape[1]}."
        )

    print(
        f"Loaded voltage data: {voltage.shape}"
    )

    return voltage, list(df.columns)


def load_pose(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Coordinate file not found:\n{path}"
        )

    df = pd.read_csv(path)

    required = [
        "x",
        "y",
        "z",
        "mx",
        "my",
        "mz",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in coordinate file: {missing}"
        )

    positions = (
        df[["x", "y", "z"]]
        .values
        .astype(float)
    )

    m_world = (
        df[["mx", "my", "mz"]]
        .values
        .astype(float)
    )

    norm = np.linalg.norm(
        m_world,
        axis=1,
        keepdims=True,
    )

    if np.any(norm < 1e-12):
        raise ValueError(
            "A magnetic moment vector has zero length."
        )

    # Same normalization as the calibration pipeline.
    m_world = m_world / norm

    print(
        f"Loaded robot pose: {positions.shape}"
    )

    return positions, m_world


# =============================================================================
# PHYSICS + NN PREDICTION
# =============================================================================

def compute_bz_and_distance(
    physical_df,
    robot_positions,
    m_world,
):
    """
    Compute for every sample and every sensor:

        r_i = ||sensor_i - capsule||
        Bz_i = dipole field projected to sensor z-axis
    """

    sensor_pos = physical_df[
        ["x", "y", "z"]
    ].values.astype(float)

    n_samples = len(robot_positions)

    r_all = np.zeros(
        (n_samples, N_SENSORS),
        dtype=np.float64,
    )

    bz_all = np.zeros(
        (n_samples, N_SENSORS),
        dtype=np.float64,
    )

    sensor_dir = np.array(
        [0.0, 0.0, 1.0]
    )

    for s in range(N_SENSORS):

        r_vec = (
            sensor_pos[s]
            - robot_positions
        )

        r_all[:, s] = np.linalg.norm(
            r_vec,
            axis=1,
        )

        B = dipole_field(
            r_vec,
            m_world,
        )

        bz_all[:, s] = (
            B @ sensor_dir
        )

    return r_all, bz_all


def compute_predictions(
    physical_df,
    robot_positions,
    m_world,
    voltage_raw,
    model,
    device,
):
    """
    Apply the exact pipeline:

        Stage 1:
            V_stage1 = offset + gain * Bz

        Stage 2:
            r -> delta_alpha
            alpha = 1 + delta_alpha
            V_NN = offset + gain * Bz * alpha
    """

    r_all, bz_all = compute_bz_and_distance(
        physical_df,
        robot_positions,
        m_world,
    )

    offset = (
        physical_df["offset"]
        .values
        .astype(float)
    )

    gain = (
        physical_df["gain"]
        .values
        .astype(float)
    )

    # ---------------------------------------------------------
    # Stage 1: alpha = 1
    # ---------------------------------------------------------
    voltage_stage1 = (
        offset[None, :]
        + gain[None, :]
        * bz_all
    )

    # ---------------------------------------------------------
    # Stage 2: NN alpha(r)
    # ---------------------------------------------------------
    r_tensor = torch.tensor(
        r_all,
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():
        delta_alpha = (
            model(r_tensor)
            .cpu()
            .numpy()
        )

    alpha_nn = 1.0 + delta_alpha

    voltage_nn = (
        offset[None, :]
        + gain[None, :]
        * bz_all
        * alpha_nn
    )

    return (
        r_all,
        bz_all,
        voltage_stage1,
        delta_alpha,
        alpha_nn,
        voltage_nn,
    )


# =============================================================================
# METRICS
# =============================================================================

def compute_rmse(measured, predicted):
    return np.sqrt(
        np.mean(
            (measured - predicted) ** 2
        )
    )


def compute_per_sensor_metrics(
    voltage_raw,
    voltage_stage1,
    voltage_nn,
):
    rows = []

    for s in range(N_SENSORS):

        v = voltage_raw[:, s]

        rmse_stage1 = compute_rmse(
            v,
            voltage_stage1[:, s],
        )

        rmse_nn = compute_rmse(
            v,
            voltage_nn[:, s],
        )

        improvement = (
            rmse_stage1
            - rmse_nn
        )

        if rmse_stage1 > 0:
            improvement_pct = (
                100.0
                * improvement
                / rmse_stage1
            )
        else:
            improvement_pct = np.nan

        rows.append(
            {
                "sensor_index": s,
                "rmse_stage1_V": rmse_stage1,
                "rmse_nn_V": rmse_nn,
                "rmse_improvement_V": improvement,
                "rmse_improvement_percent": improvement_pct,
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# PLOTS
# =============================================================================

def plot_sensor_voltage_comparison(
    sensor_index,
    voltage_raw,
    voltage_stage1,
    voltage_nn,
    output_dir,
):
    s = sensor_index

    fig, ax = plt.subplots(
        figsize=(11, 4.5)
    )

    sample_idx = np.arange(
        len(voltage_raw)
    )

    ax.plot(
        sample_idx,
        voltage_raw[:, s],
        label="V measured",
        linewidth=1.0,
    )

    ax.plot(
        sample_idx,
        voltage_stage1[:, s],
        label="V Stage-1 physical",
        linewidth=1.0,
    )

    ax.plot(
        sample_idx,
        voltage_nn[:, s],
        label="V Stage-2 NN alpha(r)",
        linewidth=1.0,
    )

    rmse_s1 = compute_rmse(
        voltage_raw[:, s],
        voltage_stage1[:, s],
    )

    rmse_nn = compute_rmse(
        voltage_raw[:, s],
        voltage_nn[:, s],
    )

    ax.set_xlabel("Sample index")
    ax.set_ylabel("Voltage (V)")

    ax.set_title(
        f"Sensor {s + 1:02d} | "
        f"Stage-1 RMSE = {rmse_s1:.6f} V | "
        f"NN RMSE = {rmse_nn:.6f} V"
    )

    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    path = (
        output_dir
        / f"sensor_{s + 1:02d}.png"
    )

    fig.savefig(
        path,
        dpi=130,
    )

    plt.close(fig)


def plot_rmse_comparison(
    metrics,
    output_dir,
):
    sensors = np.arange(
        1,
        N_SENSORS + 1,
    )

    fig, ax = plt.subplots(
        figsize=(13, 5)
    )

    width = 0.38

    ax.bar(
        sensors - width / 2,
        metrics["rmse_stage1_V"],
        width=width,
        label="Stage-1 physical",
    )

    ax.bar(
        sensors + width / 2,
        metrics["rmse_nn_V"],
        width=width,
        label="Stage-2 NN alpha(r)",
    )

    ax.set_xlabel("Sensor index")
    ax.set_ylabel("RMSE (V)")
    ax.set_title(
        "Per-sensor RMSE: Stage-1 physical vs Stage-2 NN alpha(r)"
    )

    ax.set_xticks(sensors)
    ax.legend()
    ax.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()

    path = (
        output_dir
        / "rmse_stage1_vs_nn.png"
    )

    fig.savefig(
        path,
        dpi=140,
    )

    plt.close(fig)


def plot_rmse_improvement(
    metrics,
    output_dir,
):
    sensors = np.arange(
        1,
        N_SENSORS + 1,
    )

    fig, ax = plt.subplots(
        figsize=(13, 5)
    )

    ax.bar(
        sensors,
        metrics["rmse_improvement_V"],
    )

    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=0.8,
    )

    ax.set_xlabel("Sensor index")
    ax.set_ylabel("RMSE reduction (V)")
    ax.set_title(
        "RMSE change after Stage-2 NN alpha(r)"
    )

    ax.set_xticks(sensors)
    ax.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()

    path = (
        output_dir
        / "rmse_improvement_nn.png"
    )

    fig.savefig(
        path,
        dpi=140,
    )

    plt.close(fig)


def plot_alpha_vs_r(
    sensor_index,
    r_all,
    alpha_nn,
    output_dir,
):
    s = sensor_index

    r_s = r_all[:, s]
    alpha_s = alpha_nn[:, s]

    r_min = np.min(r_s)
    r_max = np.max(r_s)

    r_curve = np.linspace(
        r_min,
        r_max,
        300,
    )

    # Evaluate the same NN on a smooth r grid.
    # This is done outside this function in main, where the model is available.
    return r_s, alpha_s, r_curve


def save_alpha_curve(
    model,
    sensor_indices,
    r_all,
    output_dir,
    device,
):
    """
    Save and plot NN alpha(r) for selected sensors.

    Because the current architecture is a SHARED scalar MLP,
    all sensors use the same mathematical function f_theta(r).
    Their evaluated alpha values differ when their distances r_i differ.
    """

    for s in sensor_indices:

        r_s = r_all[:, s]

        r_curve = np.linspace(
            r_s.min(),
            r_s.max(),
            300,
        )

        r_tensor = torch.tensor(
            r_curve[None, :],
            dtype=torch.float32,
            device=device,
        )

        with torch.no_grad():
            alpha_curve = (
                1.0
                + model(r_tensor)
                .cpu()
                .numpy()[0]
            )

        fig, ax = plt.subplots(
            figsize=(7.5, 5)
        )

        ax.scatter(
            r_s,
            1.0
            + (
                model(
                    torch.tensor(
                        r_s[None, :],
                        dtype=torch.float32,
                        device=device,
                    )
                )
                .detach()
                .cpu()
                .numpy()[0]
            ),
            s=8,
            alpha=0.35,
            label="NN alpha at measured r",
        )

        ax.plot(
            r_curve,
            alpha_curve,
            linewidth=2.0,
            label="NN alpha(r)",
        )

        ax.axhline(
            1.0,
            linestyle="--",
            linewidth=0.8,
            label="alpha = 1",
        )

        ax.set_xlabel("r (m)")
        ax.set_ylabel("alpha")
        ax.set_title(
            f"Sensor {s + 1:02d} | NN alpha(r)"
        )

        ax.legend()
        ax.grid(
            True,
            alpha=0.3,
        )

        fig.tight_layout()

        path = (
            output_dir
            / f"alpha_vs_r_sensor_{s + 1:02d}.png"
        )

        fig.savefig(
            path,
            dpi=140,
        )

        plt.close(fig)


def plot_alpha_statistics(
    alpha_nn,
    output_dir,
):
    alpha_mean = np.mean(
        alpha_nn,
        axis=0,
    )

    alpha_std = np.std(
        alpha_nn,
        axis=0,
    )

    sensors = np.arange(
        1,
        N_SENSORS + 1,
    )

    fig, ax = plt.subplots(
        figsize=(13, 5)
    )

    ax.errorbar(
        sensors,
        alpha_mean,
        yerr=alpha_std,
        fmt="o",
        markersize=3,
        capsize=2,
    )

    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=0.8,
        label="alpha = 1",
    )

    ax.set_xlabel("Sensor index")
    ax.set_ylabel("alpha")
    ax.set_title(
        "NN alpha statistics across samples"
    )

    ax.set_xticks(sensors)
    ax.legend()
    ax.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()

    path = (
        output_dir
        / "alpha_statistics.png"
    )

    fig.savefig(
        path,
        dpi=140,
    )

    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 72)
    print("VISUALIZATION")
    print("=" * 72)

    print(f"Device: {device}")
    print(f"DATASET: {DATASET_DIR}")
    print(f"Visualization output: {OUTPUT_HELIX2}")

    # -------------------------------------------------------------------------
    # Load Stage-1 calibration
    # -------------------------------------------------------------------------
    physical_df = load_physical_calibration(
        PHYSICAL_PARAMS
    )

    # -------------------------------------------------------------------------
    # Load Stage-2 NN
    # -------------------------------------------------------------------------
    model, checkpoint = load_nn_checkpoint(
        NN_CKPT_PATH,
        device,
    )

    # -------------------------------------------------------------------------
    # Load external/raw test data
    # -------------------------------------------------------------------------
    voltage_raw, voltage_columns = load_voltage(
        VOLTAGE_PATH
    )

    robot_positions, m_world = load_pose(
        COORDS_PATH
    )

    if len(voltage_raw) != len(robot_positions):
        raise ValueError(
            f"Voltage samples ({len(voltage_raw)}) "
            f"!= pose samples ({len(robot_positions)})."
        )

    # -------------------------------------------------------------------------
    # Compute Stage-1 + Stage-2 NN predictions
    # -------------------------------------------------------------------------
    (
        r_all,
        bz_all,
        voltage_stage1,
        delta_alpha,
        alpha_nn,
        voltage_nn,
    ) = compute_predictions(
        physical_df,
        robot_positions,
        m_world,
        voltage_raw,
        model,
        device,
    )

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------
    metrics = compute_per_sensor_metrics(
        voltage_raw,
        voltage_stage1,
        voltage_nn,
    )

    overall_stage1_rmse = compute_rmse(
        voltage_raw,
        voltage_stage1,
    )

    overall_nn_rmse = compute_rmse(
        voltage_raw,
        voltage_nn,
    )

    print("\n" + "=" * 72)
    print("OVERALL RESULT")
    print("=" * 72)

    print(
        f"Stage-1 physical RMSE : "
        f"{overall_stage1_rmse:.8f} V"
    )

    print(
        f"Stage-2 NN alpha(r) RMSE: "
        f"{overall_nn_rmse:.8f} V"
    )

    print(
        f"RMSE reduction         : "
        f"{overall_stage1_rmse - overall_nn_rmse:.8f} V"
    )

    if overall_stage1_rmse > 0:
        print(
            f"RMSE reduction (%)     : "
            f"{100.0 * (overall_stage1_rmse - overall_nn_rmse) / overall_stage1_rmse:.3f}%"
        )

    print(
        f"\nNN alpha range         : "
        f"[{alpha_nn.min():.8f}, {alpha_nn.max():.8f}]"
    )

    print(
        f"NN delta_alpha range   : "
        f"[{delta_alpha.min():.8f}, {delta_alpha.max():.8f}]"
    )

    print(
        f"Distance r range       : "
        f"[{r_all.min():.6f}, {r_all.max():.6f}] m"
    )

    # -------------------------------------------------------------------------
    # Save CSV: per-sensor RMSE
    # -------------------------------------------------------------------------
    metrics.to_csv(
        RMSE_SUMMARY_PATH,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Save alpha values
    # -------------------------------------------------------------------------
    alpha_df = pd.DataFrame(
        alpha_nn,
        columns=[
            f"sensor_{i + 1:02d}"
            for i in range(N_SENSORS)
        ],
    )

    alpha_df.insert(
        0,
        "sample_index",
        np.arange(len(alpha_df)),
    )

    alpha_df.to_csv(
        ALPHA_PATH,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Save voltage predictions
    # -------------------------------------------------------------------------
    voltage_rows = {}

    for s in range(N_SENSORS):
        voltage_rows[
            f"sensor_{s + 1:02d}_measured_V"
        ] = voltage_raw[:, s]

        voltage_rows[
            f"sensor_{s + 1:02d}_stage1_V"
        ] = voltage_stage1[:, s]

        voltage_rows[
            f"sensor_{s + 1:02d}_nn_V"
        ] = voltage_nn[:, s]

    pd.DataFrame(voltage_rows).to_csv(
        VOLTAGE_PRED_PATH,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Plot every sensor
    # -------------------------------------------------------------------------
    print("\nGenerating 64 sensor voltage plots...")

    for s in range(N_SENSORS):

        plot_sensor_voltage_comparison(
            s,
            voltage_raw,
            voltage_stage1,
            voltage_nn,
            VISUALIZATION,
        )

    # -------------------------------------------------------------------------
    # Global comparison plots
    # -------------------------------------------------------------------------
    plot_rmse_comparison(
        metrics,
        VISUALIZATION,
    )

    plot_rmse_improvement(
        metrics,
        VISUALIZATION,
    )

    plot_alpha_statistics(
        alpha_nn,
        VISUALIZATION,
    )

    # -------------------------------------------------------------------------
    # Alpha(r) curves for representative sensors
    # -------------------------------------------------------------------------
    representative_sensors = [
        0,
        15,
        31,
        47,
        63,
    ]

    print(
        "\nGenerating alpha(r) plots for sensors:",
        [s + 1 for s in representative_sensors],
    )

    save_alpha_curve(
        model,
        representative_sensors,
        r_all,
        VISUALIZATION,
        device,
    )

    # -------------------------------------------------------------------------
    # Console table
    # -------------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("PER-SENSOR RMSE")
    print("=" * 72)

    print(
        metrics.to_string(
            index=False,
            float_format=lambda x: f"{x:.8f}",
        )
    )

    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)

    print(
        f"64 voltage plots saved to:\n"
        f"  {VISUALIZATION}"
    )

    print(
        f"\nRMSE summary:\n"
        f"  {RMSE_SUMMARY_PATH}"
    )

    print(
        f"\nAll NN alpha values:\n"
        f"  {ALPHA_PATH}"
    )

    print(
        f"\nVoltage predictions:\n"
        f"  {VOLTAGE_PRED_PATH}"
    )



if __name__ == "__main__":
    main()