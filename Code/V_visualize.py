"Vẽ đồ thị so sánh điện áp đo được và điện áp tính toán từ mô hình dipole cho từng sensor. Tính toán RMSE cho từng sensor và tổng thể."

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from pyparsing import alphas

# =============================================================================
# FILE PATHS  (sua lai neu ban chay o may khac)
# =============================================================================
BASE_DIR = Path(r"/Users/tuananhnguyen/Downloads/Hallsensor_final/Data set 18.6")

PHYSICAL_PATH = BASE_DIR / "Calibration_Physical.csv"
ALPHA_PATH = BASE_DIR / "Calibration_Alpha.csv"
VOLTAGE_PATH = BASE_DIR / "Helix_data.csv"
COORDS_PATH = BASE_DIR / "Helix_points_coordinates.csv"

OUTPUT_DIR = BASE_DIR / "outputs/sensor_plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RMSE_SUMMARY_PATH = BASE_DIR / "outputs/rmse_summary.csv"

# ----  region boundaries (phai khop voi script calib) ----
REGION1_H_MAX = 0.040
REGION2_H_MAX = 0.055

MU0_OVER_4PI = 1e-7


# =============================================================================
# DIPOLE MODEL (giu nguyen cong thuc goc)
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


def load_alpha(path):
    """Region,Alpha -> dict {1: alpha1, 2: alpha2, 3: alpha3}"""
    df = pd.read_csv(path).dropna()
    alphas = {}
    for _, row in df.iterrows():
        region_num = int(str(row["Region"]).strip().split()[-1])
        alphas[region_num] = float(row["Alpha"])
    return alphas


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
# ALPHA-BY-REGION LOOKUP (giong het logic Stage 2 trong calib_region_based.py)
# =============================================================================
def alpha_for_h(h, alphas):
    """h: (N,) array = z_capsule - z_sensor. Tra ve (N,) mang alpha tuong ung."""
    alpha_arr = np.empty_like(h)
    region1_mask = h < REGION1_H_MAX
    region2_mask = (h >= REGION1_H_MAX) & (h < REGION2_H_MAX)
    region3_mask = h >= REGION2_H_MAX

    alpha_arr[region1_mask] = alphas[1]
    alpha_arr[region2_mask] = alphas[2]
    alpha_arr[region3_mask] = alphas[3]
    return alpha_arr


# =============================================================================
# COMPUTE V_pred FOR ONE SENSOR
# =============================================================================
def compute_vpred_for_sensor(sensor_row, robot_positions, m_world, alphas):
    x, y, z = sensor_row["x"], sensor_row["y"], sensor_row["z"]
    a = sensor_row["offset"]
    g = sensor_row["gain"]
    # Huong sensor co dinh thang dung (theta=phi=0 trong file calib)
    sensor_dir = np.array([0.0, 0.0, 1.0])

    sensor_pos = np.array([x, y, z])
    r_vec = sensor_pos - robot_positions          # (N,3)

    B = dipole_field(r_vec, m_world)              # (N,3)
    B_proj = B @ sensor_dir                       # (N,)
    h = robot_positions[:, 2] - z

    # mask1 = h < 0.045
    # mask2 = (h >= 0.045) & (h < 0.065)
    # mask3 = h >= 0.065
    # for region_name, mask in zip(
    #     ["R1", "R2", "R3"],
    #     [mask1, mask2, mask3]
    # ):

    #     B_region = B_proj[mask]

    #     if len(B_region) == 0:
    #         continue

    #     print(f"\n===== {region_name} =====")
    #     print("Samples :", len(B_region))
    #     print("Mean    :", np.mean(B_region))
    #     print("Median  :", np.median(B_region))
    #     print("Min     :", np.min(B_region))
    #     print("Max     :", np.max(B_region))
    #     print("Positive:", np.sum(B_region > 0))
    #     print("Negative:", np.sum(B_region < 0))

    alpha_sample = alpha_for_h(h, alphas)         # (N,)

    v_pred = a + alpha_sample * g * B_proj
    return v_pred


# =============================================================================
# MAIN
# =============================================================================
def main():
    physical_df = load_physical_calib(PHYSICAL_PATH)
    alphas = load_alpha(ALPHA_PATH)
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

    sample_idx = np.arange(n_samples_v)

    per_sensor_rmse = []
    all_v_meas = []
    all_v_pred = []

    for s in range(n_sensors):
        sensor_row = physical_df.iloc[s]
        v_meas = voltage_data[:, s]
        v_pred = compute_vpred_for_sensor(sensor_row, robot_positions, m_world, alphas)

        rmse_s = np.sqrt(np.mean((v_meas - v_pred) ** 2))
        per_sensor_rmse.append(rmse_s)

        all_v_meas.append(v_meas)
        all_v_pred.append(v_pred)

        # ---- Plot ----
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(sample_idx, v_meas, label="V measured", linewidth=1.0)
        ax.plot(sample_idx, v_pred, label="V computed",
                linewidth=1.0, linestyle="--")
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Voltage (V)")
        ax.set_title(f"Sensor {s+1:02d} | RMSE = {rmse_s:.6f} V")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"sensor_{s+1:02d}.png", dpi=120)
        plt.close(fig)

        print(f"Sensor {s+1:02d} | RMSE = {rmse_s:.6f} V")

    # ---- Overall RMSE (gop tat ca diem cua tat ca sensor) ----
    all_v_meas = np.concatenate(all_v_meas)
    all_v_pred = np.concatenate(all_v_pred)
    overall_rmse = np.sqrt(np.mean((all_v_meas - all_v_pred) ** 2))

    print(f"\n==== OVERALL RMSE (all sensors, all samples) = "
          f"{overall_rmse:.6f} V ====")

    # ---- Save RMSE summary CSV ----
    summary_df = pd.DataFrame({
        "sensor_index": np.arange(1, n_sensors + 1),
        "rmse": per_sensor_rmse
    })
    summary_df.loc[len(summary_df)] = ["OVERALL", overall_rmse]
    summary_df.to_csv(RMSE_SUMMARY_PATH, index=False)

    print(f"\nDa luu {n_sensors} anh vao: {OUTPUT_DIR}")
    print(f"Da luu bang RMSE vao: {RMSE_SUMMARY_PATH}")


if __name__ == "__main__":
    main()