# alpha(r) per sensor

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from pathlib import Path

# # =============================================================================
# # FILE PATHS  
# # =============================================================================
# # BASE_DIR = Path(r"/Users/tuananhnguyen/Downloads/Hallsensor_final/Data_8_2026") #MAC
# BASE_DIR = Path(r"D:\Downloads\Hallsensor_final\Data_8_2026") #WINDOWS

# PHYSICAL_PATH = BASE_DIR / "Calibration_Physical_r.csv"
# ALPHA_PATH = BASE_DIR / "Calibration_Alpha_r_per_sensor.csv"

# VOLTAGE_PATH = BASE_DIR / "Helix_data_2.csv"
# COORDS_PATH = BASE_DIR / "Helix_points_coordinates_2.csv"
# OUTPUT_DIR = BASE_DIR / "outputs/sosanh_linear_alpha(r)_per_sensor_Helix_2"

# # VOLTAGE_PATH = BASE_DIR / "Grid_data.csv"
# # COORDS_PATH = BASE_DIR / "Grid_points_coordinates.csv"
# # OUTPUT_DIR = BASE_DIR / "outputs/sensor_plots_grid_alpha(r)_per_sensor"

# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# RMSE_SUMMARY_PATH = BASE_DIR / "outputs/rmse_summary_alpha(r)_per_sensor.csv"

# MU0_OVER_4PI = 1e-7


# # =============================================================================
# # DIPOLE MODEL 
# # =============================================================================
# def dipole_field(r_vec, m_vec):
#     """Calculate magnetic field from dipole model.
#     r_vec: (N,3) vector from source(capsule) to sensor
#     m_vec: (N,3) unit magnetic moment vector
#     """
#     r = np.linalg.norm(r_vec, axis=1, keepdims=True)
#     r3 = np.maximum(r ** 3, 1e-12)
#     r5 = np.maximum(r ** 5, 1e-12)

#     mdotr = np.sum(m_vec * r_vec, axis=1, keepdims=True)

#     B = MU0_OVER_4PI * (
#         3.0 * r_vec * mdotr / r5 - m_vec / r3
#     )
#     return B


# # =============================================================================
# # LOAD DATA
# # =============================================================================
# def load_physical_calib(path):
#     """sensor_index, x, y, z, offset, gain, theta, phi"""
#     df = pd.read_csv(path)
#     df = df.sort_values("sensor_index").reset_index(drop=True)
#     return df


# def load_alpha_per_sensor(path, n_sensors):
#     """sensor_index, c0, c1, rmse -> DataFrame indexed by sensor_index,
#     one (c0, c1) pair per sensor (alpha_s(r) = c0_s + c1_s*r)."""
#     df = pd.read_csv(path)
#     df = df.sort_values("sensor_index").reset_index(drop=True)
#     expected_indices = np.arange(n_sensors)
#     actual_indices = df["sensor_index"].to_numpy()
#     if not np.array_equal(actual_indices, expected_indices):
#         raise ValueError(
#             "Calibration_Alpha_r_per_sensor.csv must contain exactly "
#             f"sensor_index values 0 to {n_sensors - 1}."
#         )
#     return df


# def load_voltage_data(path):
#     """grid_data.csv: moi cot la 1 sensor, moi dong la 1 mau"""
#     df = pd.read_csv(path)
#     return df.values, list(df.columns)


# def load_robot_pose(path):
#     """grid_points_coordinates.csv: x,y,z,mx,my,mz"""
#     df = pd.read_csv(path)
#     positions = df[["x", "y", "z"]].values
#     m_world = df[["mx", "my", "mz"]].values
#     norm = np.linalg.norm(m_world, axis=1, keepdims=True)
#     m_world = m_world / norm
#     return positions, m_world


# # =============================================================================
# # ALPHA(R) LOOKUP -- PER SENSOR
# # =============================================================================
# def alpha_for_r(r, c0, c1):
#     """r: (N,) array = khoang cach tu capsule den sensor. Tra ve (N,) mang
#     alpha_s(r) = c0_s + c1_s*r cho MOT sensor duy nhat (c0, c1 la scalar
#     cua sensor do, khac voi ban global chi co 1 cap (c0, c1) chung)."""
#     return c0 + c1 * r


# # =============================================================================
# # COMPUTE V_pred FOR ONE SENSOR
# # =============================================================================
# def compute_vpred_for_sensor(sensor_row, robot_positions, m_world, c0, c1):
#     x, y, z = sensor_row["x"], sensor_row["y"], sensor_row["z"]
#     a = sensor_row["offset"]
#     g = sensor_row["gain"]
#     # Huong sensor co dinh thang dung (theta=phi=0 trong file calib)
#     sensor_dir = np.array([0.0, 0.0, 1.0])

#     sensor_pos = np.array([x, y, z])
#     r_vec = sensor_pos - robot_positions          # (N,3)

#     B = dipole_field(r_vec, m_world)              # (N,3)
#     B_proj = B @ sensor_dir                       # (N,)
#     r_distance = np.linalg.norm(r_vec, axis=1)     # (N,)

#     alpha_sample = alpha_for_r(r_distance, c0, c1)   # (N,) -- rieng cho sensor nay

#     v_pred = a + alpha_sample * g * B_proj

#     return v_pred


# # =============================================================================
# # MAIN
# # =============================================================================
# def main():
#     physical_df = load_physical_calib(PHYSICAL_PATH)
#     voltage_data, voltage_cols = load_voltage_data(VOLTAGE_PATH)
#     robot_positions, m_world = load_robot_pose(COORDS_PATH)

#     n_samples_v = voltage_data.shape[0]
#     n_samples_pos = robot_positions.shape[0]
#     assert n_samples_v == n_samples_pos, (
#         f"So mau dien ap ({n_samples_v}) khac so mau toa do ({n_samples_pos})"
#     )

#     n_sensors = physical_df.shape[0]
#     assert n_sensors == voltage_data.shape[1], (
#         f"So sensor trong file calib ({n_sensors}) khac so cot dien ap "
#         f"({voltage_data.shape[1]})"
#     )

#     alpha_df = load_alpha_per_sensor(ALPHA_PATH, n_sensors)

#     sample_idx = np.arange(n_samples_v)

#     per_sensor_rmse = []
#     all_v_meas = []
#     all_v_pred = []

#     for s in range(n_sensors):
#         sensor_row = physical_df.iloc[s]
#         alpha_row = alpha_df.iloc[s]
#         c0_s = alpha_row["c0"]
#         c1_s = alpha_row["c1"]

#         v_meas = voltage_data[:, s]
#         v_pred = compute_vpred_for_sensor(
#             sensor_row, robot_positions, m_world, c0_s, c1_s
#         )

#         rmse_s = np.sqrt(np.mean((v_meas - v_pred) ** 2))
#         per_sensor_rmse.append(rmse_s)

#         all_v_meas.append(v_meas)
#         all_v_pred.append(v_pred)

#         # ---- Plot ----
#         fig, ax = plt.subplots(figsize=(10, 4))
#         ax.plot(sample_idx, v_meas, label="V measured", linewidth=1.0)
#         ax.plot(sample_idx, v_pred, label="V computed",linewidth=1.0)
#         ax.set_xlabel("Sample index")
#         ax.set_ylabel("Voltage (V)")
#         ax.set_title(
#             f"Sensor {s+1:02d} | alpha(r) = {c0_s:.4f} + ({c1_s:.4f})*r "
#             f"| RMSE = {rmse_s:.6f} V"
#         )
#         ax.legend()
#         ax.grid(True, alpha=0.3)
#         fig.tight_layout()
#         fig.savefig(OUTPUT_DIR / f"sensor_{s+1:02d}.png", dpi=120)
#         plt.close(fig)

#         print(f"Sensor {s+1:02d} | RMSE = {rmse_s:.6f} V")

#     # ---- Overall RMSE (gop tat ca diem cua tat ca sensor) ----
#     all_v_meas = np.concatenate(all_v_meas)
#     all_v_pred = np.concatenate(all_v_pred)
#     overall_rmse = np.sqrt(np.mean((all_v_meas - all_v_pred) ** 2))

#     print(f"\n==== OVERALL RMSE (all sensors, all samples) = "
#           f"{overall_rmse:.6f} V ====")

#     # ---- Save RMSE summary CSV ----
#     summary_df = pd.DataFrame({
#         "sensor_index": np.arange(1, n_sensors + 1),
#         "rmse": per_sensor_rmse
#     })
#     summary_df.loc[len(summary_df)] = ["OVERALL", overall_rmse]
#     summary_df.to_csv(RMSE_SUMMARY_PATH, index=False)

#     print(f"\nDa luu {n_sensors} anh vao: {OUTPUT_DIR}")
#     print(f"Da luu bang RMSE vao: {RMSE_SUMMARY_PATH}")


# if __name__ == "__main__":
#     main()

# alpha(h) per sensor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# FILE PATHS  
# =============================================================================
# BASE_DIR = Path(r"/Users/tuananhnguyen/Downloads/Hallsensor_final/Data_8_2026") #MAC
BASE_DIR = Path(r"D:\Downloads\Hallsensor_final\Data_8_2026") #WINDOWS

PHYSICAL_PATH = BASE_DIR / "Calibration_Physical_h_per_sensor.csv"
ALPHA_PATH = BASE_DIR / "Calibration_Alpha_h_per_sensor.csv"

VOLTAGE_PATH = BASE_DIR / "Grid_data.csv"
COORDS_PATH = BASE_DIR / "Grid_points_coordinates.csv"
OUTPUT_DIR = BASE_DIR / "outputs/sosanh_linear_alpha(h)_per_sensor_Grid"

# VOLTAGE_PATH = BASE_DIR / "Grid_data.csv"
# COORDS_PATH = BASE_DIR / "Grid_points_coordinates.csv"
# OUTPUT_DIR = BASE_DIR / "outputs/sensor_plots_grid_alpha(h)_per_sensor"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RMSE_SUMMARY_PATH = BASE_DIR / "outputs/rmse_summary_alpha(h)_per_sensor_Grid.csv"

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
    one (c0, c1) pair per sensor (alpha_s(h) = c0_s + c1_s*h)."""
    df = pd.read_csv(path)
    df = df.sort_values("sensor_index").reset_index(drop=True)
    expected_indices = np.arange(n_sensors)
    actual_indices = df["sensor_index"].to_numpy()
    if not np.array_equal(actual_indices, expected_indices):
        raise ValueError(
            "Calibration_Alpha_per_sensor_new.csv must contain exactly "
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
# ALPHA(H) LOOKUP -- PER SENSOR
# =============================================================================
def alpha_for_h(h, c0, c1):
    """h: (N,) array = z_capsule - z_sensor. Tra ve (N,) mang
    alpha_s(h) = c0_s + c1_s*h cho MOT sensor duy nhat (c0, c1 la scalar
    cua sensor do, khac voi ban global chi co 1 cap (c0, c1) chung)."""
    return c0 + c1 * h


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
    h = robot_positions[:, 2] - z                  # (N,)

    alpha_sample = alpha_for_h(h, c0, c1)         # (N,) -- rieng cho sensor nay

    v_pred = a + alpha_sample * g * B_proj

    return v_pred


# =============================================================================
# MAIN
# =============================================================================
def main():
    physical_df = load_physical_calib(PHYSICAL_PATH)
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
        per_sensor_rmse.append(rmse_s)

        all_v_meas.append(v_meas)
        all_v_pred.append(v_pred)

        # ---- Plot ----
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(sample_idx, v_meas, label="V measured", linewidth=1.0)
        ax.plot(sample_idx, v_pred, label="V computed",linewidth=1.0)
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Voltage (V)")
        ax.set_title(
            f"Sensor {s+1:02d} | alpha(h) = {c0_s:.4f} + ({c1_s:.4f})*h "
            f"| RMSE = {rmse_s:.6f} V"
        )
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