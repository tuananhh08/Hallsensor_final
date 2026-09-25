# Global alpha(h)
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from pathlib import Path

# # =============================================================================
# # FILE PATHS  
# # =============================================================================
# # BASE_DIR = Path(r"/Users/tuananhnguyen/Downloads/Hallsensor_final/Data_8_2026") #MAC
# BASE_DIR = Path(r"D:\Downloads\Hallsensor_final\Data_8_2026") #WINDOWS

# PHYSICAL_PATH = BASE_DIR / "Calibration_Physical_h.csv"
# ALPHA_PATH = BASE_DIR / "Calibration_Alpha_h.csv"

# VOLTAGE_PATH = BASE_DIR / "Grid_data.csv"
# COORDS_PATH = BASE_DIR / "Grid_points_coordinates.csv"
# OUTPUT_DIR = BASE_DIR / "outputs/sosanh_linear_alpha(h)_Grid"

# # VOLTAGE_PATH = BASE_DIR / "Grid_data.csv"
# # COORDS_PATH = BASE_DIR / "Grid_points_coordinates.csv"
# # OUTPUT_DIR = BASE_DIR / "outputs/sensor_plots_grid_alpha(r)"

# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# RMSE_SUMMARY_PATH = BASE_DIR / "outputs/rmse_summary_alpha(h)_Grid.csv"

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


# def load_alpha(path):
#     """coefficient,value -> dict {'c0': c0, 'c1': c1} (alpha(h) = c0 + c1*h)"""
#     df = pd.read_csv(path).dropna()
#     alpha_coeffs = {}
#     for _, row in df.iterrows():
#         alpha_coeffs[str(row["coefficient"]).strip()] = float(row["value"])
#     return alpha_coeffs


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
# # ALPHA(H) LOOKUP 
# # =============================================================================
# def alpha_for_h(h, alpha_coeffs):
#     """h: (N,) array = z_capsule - z_sensor. Tra ve (N,) mang alpha(h) = c0 + c1*h."""
#     c0 = alpha_coeffs["c0"]
#     c1 = alpha_coeffs["c1"]
#     return c0 + c1 * h


# # =============================================================================
# # COMPUTE V_pred FOR ONE SENSOR
# # =============================================================================
# def compute_vpred_for_sensor(sensor_row, robot_positions, m_world, alpha_coeffs):
#     x, y, z = sensor_row["x"], sensor_row["y"], sensor_row["z"]
#     a = sensor_row["offset"]
#     g = sensor_row["gain"]
#     # Huong sensor co dinh thang dung (theta=phi=0 trong file calib)
#     sensor_dir = np.array([0.0, 0.0, 1.0])

#     sensor_pos = np.array([x, y, z])
#     r_vec = sensor_pos - robot_positions          # (N,3)

#     B = dipole_field(r_vec, m_world)              # (N,3)
#     B_proj = B @ sensor_dir                       # (N,)
#     h = robot_positions[:, 2] - z

#     alpha_sample = alpha_for_h(h, alpha_coeffs)   # (N,)

#     v_pred = a + alpha_sample * g * B_proj

#     return v_pred


# # =============================================================================
# # MAIN
# # =============================================================================
# def main():
#     physical_df = load_physical_calib(PHYSICAL_PATH)
#     alpha_coeffs = load_alpha(ALPHA_PATH)
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

#     sample_idx = np.arange(n_samples_v)

#     per_sensor_rmse = []
#     all_v_meas = []
#     all_v_pred = []

#     for s in range(n_sensors):
#         sensor_row = physical_df.iloc[s]
#         v_meas = voltage_data[:, s]
#         v_pred = compute_vpred_for_sensor(sensor_row, robot_positions, m_world, alpha_coeffs)

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
#         ax.set_title(f"Sensor {s+1:02d} | RMSE = {rmse_s:.6f} V")
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

# Global alpha(r)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# FILE PATHS  
# =============================================================================
# BASE_DIR = Path(r"/Users/tuananhnguyen/Downloads/Hallsensor_final/Data_8_2026") #MAC
BASE_DIR = Path(r"D:\Downloads\Hallsensor_final\Data_8_2026") #WINDOWS

PHYSICAL_PATH = BASE_DIR / "Calibration_Physical_r.csv"
ALPHA_PATH = BASE_DIR / "Calibration_Alpha_r.csv"

VOLTAGE_PATH = BASE_DIR / "Helix_data_2.csv"
COORDS_PATH = BASE_DIR / "Helix_points_coordinates_2.csv"
OUTPUT_DIR = BASE_DIR / "outputs/sosanh_linear_alpha(r)_Helix_2"

# VOLTAGE_PATH = BASE_DIR / "Grid_data.csv"
# COORDS_PATH = BASE_DIR / "Grid_points_coordinates.csv"
# OUTPUT_DIR = BASE_DIR / "outputs/sensor_plots_grid_alpha(r)"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RMSE_SUMMARY_PATH = BASE_DIR / "outputs/rmse_summary_alpha(r)_Helix_2.csv"

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


def load_alpha(path):
    """coefficient,value -> dict {'c0': c0, 'c1': c1} (alpha(r) = c0 + c1*r)"""
    df = pd.read_csv(path).dropna()
    alpha_coeffs = {}
    for _, row in df.iterrows():
        alpha_coeffs[str(row["coefficient"]).strip()] = float(row["value"])
    return alpha_coeffs


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
# ALPHA(R) LOOKUP 
# =============================================================================
def alpha_for_r(r, alpha_coeffs):
    """r: (N,) array = khoang cach tu capsule den sensor. Tra ve (N,) mang alpha(r) = c0 + c1*r."""
    c0 = alpha_coeffs["c0"]
    c1 = alpha_coeffs["c1"]
    return c0 + c1 * r


# =============================================================================
# COMPUTE V_pred FOR ONE SENSOR
# =============================================================================
def compute_vpred_for_sensor(sensor_row, robot_positions, m_world, alpha_coeffs):
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

    alpha_sample = alpha_for_r(r_distance, alpha_coeffs)   # (N,)

    v_pred = a + alpha_sample * g * B_proj

    return v_pred


# =============================================================================
# MAIN
# =============================================================================
def main():
    physical_df = load_physical_calib(PHYSICAL_PATH)
    alpha_coeffs = load_alpha(ALPHA_PATH)
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
        v_pred = compute_vpred_for_sensor(sensor_row, robot_positions, m_world, alpha_coeffs)

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


# ve do thi so sanh, su dung NN de tinh alpha(r) 

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import torch
# import torch.nn as nn
# from pathlib import Path


# # =============================================================================
# # FILE PATHS
# # =============================================================================

# # MAC
# BASE_DIR = Path(r"/Users/tuananhnguyen/Downloads/Hallsensor_final/Data_8_2026")

# # WINDOWS
# # BASE_DIR = Path(r"D:\Downloads\Hallsensor_final\Data_8_2026")


# # -------------------------------------------------------------------------
# # Calibration files produced by Neural Network calibration
# # -------------------------------------------------------------------------
# PHYSICAL_PATH = BASE_DIR / "Calibration_Physical_NN2.csv"
# NN_PATH = BASE_DIR / "Calibration_AlphaNN2.pt"
# META_PATH = BASE_DIR / "Calibration_AlphaNN2_meta.csv"


# # -------------------------------------------------------------------------
# # Test / validation data
# # -------------------------------------------------------------------------
# VOLTAGE_PATH = BASE_DIR / "Helix_data_2.csv"
# COORDS_PATH = BASE_DIR / "Helix_points_coordinates_2.csv"


# # -------------------------------------------------------------------------
# # Output
# # -------------------------------------------------------------------------
# OUTPUT_DIR = BASE_DIR / "outputs/sensor_plots_NN_alpha(r)_Helix_2"
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# RMSE_SUMMARY_PATH = (
#     BASE_DIR /
#     "outputs/rmse_summary_NN_alpha(r)_Helix_2.csv"
# )


# # =============================================================================
# # CONSTANTS
# # =============================================================================

# MU0_OVER_4PI = 1e-7

# DEVICE = torch.device(
#     "cuda" if torch.cuda.is_available() else "cpu"
# )

# print(f"Using device: {DEVICE}")


# # =============================================================================
# # DIPOLE MODEL
# # =============================================================================

# def dipole_field(r_vec, m_vec):
#     """
#     Calculate magnetic field from dipole model.

#     r_vec:
#         (N, 3)
#         vector from capsule to sensor

#     m_vec:
#         (N, 3)
#         unit magnetic moment vector

#     return:
#         B : (N, 3)
#     """

#     r = np.linalg.norm(
#         r_vec,
#         axis=1,
#         keepdims=True
#     )

#     r3 = np.maximum(
#         r ** 3,
#         1e-12
#     )

#     r5 = np.maximum(
#         r ** 5,
#         1e-12
#     )

#     mdotr = np.sum(
#         m_vec * r_vec,
#         axis=1,
#         keepdims=True
#     )

#     B = MU0_OVER_4PI * (
#         3.0 * r_vec * mdotr / r5
#         - m_vec / r3
#     )

#     return B


# # =============================================================================
# # LOAD PHYSICAL CALIBRATION
# # =============================================================================

# def load_physical_calib(path):
#     """
#     Load Stage-1 physical calibration.

#     Expected columns:
#         sensor_index
#         x
#         y
#         z
#         offset
#         gain
#     """

#     df = pd.read_csv(path)

#     df = (
#         df
#         .sort_values("sensor_index")
#         .reset_index(drop=True)
#     )

#     required_columns = [
#         "sensor_index",
#         "x",
#         "y",
#         "z",
#         "offset",
#         "gain"
#     ]

#     for col in required_columns:
#         if col not in df.columns:
#             raise ValueError(
#                 f"Missing column '{col}' in {path.name}"
#             )

#     print(
#         f"Loaded physical calibration: "
#         f"{df.shape[0]} sensors"
#     )

#     return df


# # =============================================================================
# # LOAD NN METADATA
# # =============================================================================

# def load_nn_meta(path):
#     """
#     Load NN metadata.

#     Expected:
#         r_mean
#         r_std
#         hidden_dims
#         output_scale_init
#     """

#     df = pd.read_csv(path)

#     if len(df) == 0:
#         raise ValueError(
#             f"Metadata file is empty: {path}"
#         )

#     meta = df.iloc[0]

#     required_columns = [
#         "r_mean",
#         "r_std",
#         "hidden_dims",
#         "output_scale_init"
#     ]

#     for col in required_columns:
#         if col not in df.columns:
#             raise ValueError(
#                 f"Missing metadata column '{col}' "
#                 f"in {path.name}"
#             )

#     r_mean = float(meta["r_mean"])
#     r_std = float(meta["r_std"])

#     # Calibration NN uses the fixed architecture 64-64-32.
#     # Accept both metadata formats for backward compatibility:
#     #   "64,64,32" or "64-64-32".
#     hidden_dims_raw = str(meta["hidden_dims"]).strip()
#     hidden_dims_raw = hidden_dims_raw.replace("-", ",")
#     hidden_dims = [int(x.strip()) for x in hidden_dims_raw.split(",") if x.strip()]

#     if hidden_dims != [32, 64, 64, 64, 32]:
#         raise ValueError(
#             f"Expected fixed NN hidden_dims [32, 64, 64, 64, 32], got {hidden_dims}"
#         )

#     output_scale_init = float(
#         meta["output_scale_init"]
#     )

#     if r_std <= 0:
#         raise ValueError(
#             f"Invalid r_std = {r_std}"
#         )

#     print("\nLoaded NN metadata:")
#     print(f"  r_mean            = {r_mean:.10f} m")
#     print(f"  r_std             = {r_std:.10f} m")
#     print(f"  hidden_dims       = {hidden_dims}")
#     print(f"  output_scale_init = {output_scale_init}")

#     return {
#         "r_mean": r_mean,
#         "r_std": r_std,
#         "hidden_dims": hidden_dims,
#         "output_scale_init": output_scale_init
#     }


# # =============================================================================
# # LOAD VOLTAGE DATA
# # =============================================================================

# def load_voltage_data(path):
#     """
#     Load measured voltage.

#     Rows    = samples
#     Columns = sensors
#     """

#     df = pd.read_csv(path)

#     voltage = df.values.astype(float)

#     print(
#         f"\nLoaded voltage data: "
#         f"{voltage.shape}"
#     )

#     return voltage, list(df.columns)


# # =============================================================================
# # LOAD ROBOT POSE
# # =============================================================================

# def load_robot_pose(path):
#     """
#     Expected columns:

#         x
#         y
#         z
#         mx
#         my
#         mz

#     Position:
#         (N, 3)

#     Magnetic moment:
#         (N, 3)

#     The magnetic moment is normalized.
#     """

#     df = pd.read_csv(path)

#     required_columns = [
#         "x",
#         "y",
#         "z",
#         "mx",
#         "my",
#         "mz"
#     ]

#     for col in required_columns:
#         if col not in df.columns:
#             raise ValueError(
#                 f"Missing column '{col}' "
#                 f"in {path.name}"
#             )

#     positions = df[
#         ["x", "y", "z"]
#     ].values.astype(float)

#     m_world = df[
#         ["mx", "my", "mz"]
#     ].values.astype(float)

#     norm = np.linalg.norm(
#         m_world,
#         axis=1,
#         keepdims=True
#     )

#     if np.any(norm < 1e-12):
#         raise ValueError(
#             "Found zero magnetic-moment vector."
#         )

#     m_world = m_world / norm

#     print(
#         f"Loaded robot positions: "
#         f"{positions.shape}"
#     )

#     print(
#         f"Loaded magnetic orientations: "
#         f"{m_world.shape}"
#     )

#     return positions, m_world


# # =============================================================================
# # SAME NN ARCHITECTURE AS CALIBRATION CODE
# # =============================================================================

# class DeltaAlphaNet(nn.Module):
#     """Fixed architecture: 1 -> 64 -> 64 -> 32 -> 1, SiLU."""

#     def __init__(
#         self,
#         hidden_dims=(32, 64, 64, 64, 32),
#         input_dim=1,
#         output_scale_init=0.05
#     ):

#         super().__init__()

#         if list(hidden_dims) != [32, 64, 64, 64, 32]:
#             raise ValueError(
#                 f"Expected fixed hidden_dims [32, 64, 64, 64, 32], got {list(hidden_dims)}"
#             )

#         layers = []
#         in_dim = input_dim

#         for hidden_dim in hidden_dims:
#             layers += [
#                 nn.Linear(in_dim, hidden_dim),
#                 nn.SiLU()
#             ]
#             in_dim = hidden_dim

#         layers += [nn.Linear(in_dim, 1)]

#         self.net = nn.Sequential(*layers)

#         # Same as calibration code
#         self.output_scale = output_scale_init

#     def forward(self, r_norm):

#         delta = (
#             self.net(r_norm)
#             * self.output_scale
#         )

#         return delta.squeeze(-1)


# # =============================================================================
# # LOAD TRAINED NN
# # =============================================================================

# def load_alpha_nn(nn_path, meta):
#     """
#     Reconstruct the same NN architecture
#     and load trained weights.
#     """

#     model = DeltaAlphaNet(
#         hidden_dims=meta["hidden_dims"],
#         output_scale_init=meta["output_scale_init"]
#     ).to(DEVICE)

#     state_dict = torch.load(
#         nn_path,
#         map_location=DEVICE
#     )

#     model.load_state_dict(
#         state_dict
#     )

#     model.eval()

#     print(
#         f"\nLoaded trained NN:"
#         f"\n  {nn_path}"
#     )

#     return model


# # =============================================================================
# # COMPUTE ALPHA(r) USING TRAINED NN
# # =============================================================================

# def alpha_from_nn(
#     model,
#     r,
#     r_mean,
#     r_std
# ):
#     """
#     Calculate:

#         r_norm = (r - r_mean) / r_std

#         delta_alpha = NN(r_norm)

#         alpha(r) = 1 + delta_alpha

#     r:
#         (N,)
#     """

#     # ---------------------------------------------------------
#     # Same normalization as calibration
#     # ---------------------------------------------------------
#     r_norm = (
#         r - r_mean
#     ) / r_std

#     # ---------------------------------------------------------
#     # Convert to tensor
#     # ---------------------------------------------------------
#     r_tensor = torch.tensor(
#         r_norm,
#         dtype=torch.float32,
#         device=DEVICE
#     ).unsqueeze(-1)

#     # ---------------------------------------------------------
#     # NN inference
#     # ---------------------------------------------------------
#     with torch.no_grad():

#         delta_alpha = model(
#             r_tensor
#         )

#     delta_alpha = (
#         delta_alpha
#         .cpu()
#         .numpy()
#     )

#     # ---------------------------------------------------------
#     # alpha(r)
#     # ---------------------------------------------------------
#     alpha = (
#         1.0
#         + delta_alpha
#     )

#     return alpha


# # =============================================================================
# # COMPUTE V_PRED FOR ONE SENSOR
# # =============================================================================

# def compute_vpred_for_sensor(
#     sensor_row,
#     robot_positions,
#     m_world,
#     model,
#     r_mean,
#     r_std
# ):
#     """
#     Calculate post-calibration voltage
#     for one sensor.

#     Formula:

#         r_vec = sensor_pos - capsule_pos

#         B = dipole_field(r_vec, m)

#         B_proj = Bz

#         r = ||r_vec||

#         alpha(r) = 1 + NN(normalized r)

#         V_computed =
#             offset
#             + gain * B_proj * alpha(r)
#     """

#     # ---------------------------------------------------------
#     # Physical calibration parameters
#     # ---------------------------------------------------------

#     x = float(
#         sensor_row["x"]
#     )

#     y = float(
#         sensor_row["y"]
#     )

#     z = float(
#         sensor_row["z"]
#     )

#     offset = float(
#         sensor_row["offset"]
#     )

#     gain = float(
#         sensor_row["gain"]
#     )


#     # ---------------------------------------------------------
#     # Sensor direction
#     #
#     # DRV5055 sensor direction is fixed along Z
#     # ---------------------------------------------------------

#     sensor_dir = np.array([
#         0.0,
#         0.0,
#         1.0
#     ])


#     # ---------------------------------------------------------
#     # Calibrated sensor position
#     # ---------------------------------------------------------

#     sensor_pos = np.array([
#         x,
#         y,
#         z
#     ])


#     # ---------------------------------------------------------
#     # Vector from capsule to sensor
#     # ---------------------------------------------------------

#     r_vec = (
#         sensor_pos
#         - robot_positions
#     )


#     # ---------------------------------------------------------
#     # Euclidean distance r
#     # ---------------------------------------------------------

#     r = np.linalg.norm(
#         r_vec,
#         axis=1
#     )


#     # ---------------------------------------------------------
#     # Dipole magnetic field
#     # ---------------------------------------------------------

#     B = dipole_field(
#         r_vec,
#         m_world
#     )


#     # ---------------------------------------------------------
#     # Projection onto sensor axis
#     #
#     # Since sensor_dir = [0,0,1]:
#     #
#     # B_proj = Bz
#     # ---------------------------------------------------------

#     B_proj = (
#         B
#         @ sensor_dir
#     )


#     # ---------------------------------------------------------
#     # Neural correction alpha(r)
#     # ---------------------------------------------------------

#     alpha_sample = alpha_from_nn(
#         model=model,
#         r=r,
#         r_mean=r_mean,
#         r_std=r_std
#     )


#     # ---------------------------------------------------------
#     # FINAL FORWARD MODEL
#     #
#     # V = offset + gain * Bz * alpha(r)
#     # ---------------------------------------------------------

#     v_pred = (
#         offset
#         + gain
#         * B_proj
#         * alpha_sample
#     )


#     return v_pred


# # =============================================================================
# # MAIN
# # =============================================================================

# def main():

#     print(
#         "\n=============================================="
#     )
#     print(
#         "NN-CALIBRATED DIPOLE MODEL VALIDATION"
#     )
#     print(
#         "=============================================="
#     )


#     # =========================================================================
#     # LOAD CALIBRATION
#     # =========================================================================

#     physical_df = load_physical_calib(
#         PHYSICAL_PATH
#     )

#     nn_meta = load_nn_meta(
#         META_PATH
#     )

#     model = load_alpha_nn(
#         NN_PATH,
#         nn_meta
#     )


#     # =========================================================================
#     # LOAD TEST DATA
#     # =========================================================================

#     voltage_data, voltage_cols = (
#         load_voltage_data(
#             VOLTAGE_PATH
#         )
#     )

#     robot_positions, m_world = (
#         load_robot_pose(
#             COORDS_PATH
#         )
#     )


#     # =========================================================================
#     # CHECK DATA SIZE
#     # =========================================================================

#     n_samples_v = (
#         voltage_data.shape[0]
#     )

#     n_samples_pos = (
#         robot_positions.shape[0]
#     )

#     assert n_samples_v == n_samples_pos, (
#         f"So mau dien ap ({n_samples_v}) "
#         f"khac so mau toa do ({n_samples_pos})"
#     )


#     n_sensors = (
#         physical_df.shape[0]
#     )

#     assert n_sensors == voltage_data.shape[1], (
#         f"So sensor trong file calib "
#         f"({n_sensors}) khac so cot dien ap "
#         f"({voltage_data.shape[1]})"
#     )


#     # =========================================================================
#     # PRINT INFORMATION
#     # =========================================================================

#     print(
#         f"\nNumber of samples : {n_samples_v}"
#     )

#     print(
#         f"Number of sensors : {n_sensors}"
#     )

#     print(
#         f"r_mean = {nn_meta['r_mean']:.8f} m"
#     )

#     print(
#         f"r_std  = {nn_meta['r_std']:.8f} m"
#     )


#     # =========================================================================
#     # SAMPLE INDEX
#     # =========================================================================

#     sample_idx = np.arange(
#         n_samples_v
#     )


#     # =========================================================================
#     # STORAGE FOR RMSE
#     # =========================================================================

#     per_sensor_rmse = []

#     all_v_meas = []

#     all_v_pred = []


#     # =========================================================================
#     # PROCESS EACH SENSOR
#     # =========================================================================

#     for s in range(n_sensors):

#         print(
#             f"\nProcessing Sensor {s + 1:02d}..."
#         )


#         # ---------------------------------------------------------------------
#         # Calibration parameters
#         # ---------------------------------------------------------------------

#         sensor_row = (
#             physical_df.iloc[s]
#         )


#         # ---------------------------------------------------------------------
#         # Measured voltage
#         # ---------------------------------------------------------------------

#         v_meas = (
#             voltage_data[:, s]
#         )


#         # ---------------------------------------------------------------------
#         # NN-calibrated dipole prediction
#         # ---------------------------------------------------------------------

#         v_pred = compute_vpred_for_sensor(
#             sensor_row=sensor_row,
#             robot_positions=robot_positions,
#             m_world=m_world,
#             model=model,
#             r_mean=nn_meta["r_mean"],
#             r_std=nn_meta["r_std"]
#         )


#         # ---------------------------------------------------------------------
#         # RMSE
#         # ---------------------------------------------------------------------

#         rmse_s = np.sqrt(
#             np.mean(
#                 (
#                     v_meas
#                     - v_pred
#                 ) ** 2
#             )
#         )

#         per_sensor_rmse.append(
#             rmse_s
#         )


#         # ---------------------------------------------------------------------
#         # Store for overall RMSE
#         # ---------------------------------------------------------------------

#         all_v_meas.append(
#             v_meas
#         )

#         all_v_pred.append(
#             v_pred
#         )


#         # =====================================================================
#         # PLOT
#         # =====================================================================

#         fig, ax = plt.subplots(
#             figsize=(10, 4)
#         )


#         # ---------------------------------------------------------------------
#         # Measured voltage
#         # ---------------------------------------------------------------------

#         ax.plot(
#             sample_idx,
#             v_meas,
#             label="V measured",
#             linewidth=1.0
#         )


#         # ---------------------------------------------------------------------
#         # Computed voltage
#         # ---------------------------------------------------------------------

#         ax.plot(
#             sample_idx,
#             v_pred,
#             label="V computed",
#             linewidth=1.0
#         )


#         # ---------------------------------------------------------------------
#         # Labels
#         # ---------------------------------------------------------------------

#         ax.set_xlabel(
#             "Sample index"
#         )

#         ax.set_ylabel(
#             "Voltage (V)"
#         )


#         # ---------------------------------------------------------------------
#         # Title
#         # ---------------------------------------------------------------------

#         ax.set_title(
#             f"Sensor {s + 1:02d} | "
#             f"RMSE = {rmse_s:.6f} V"
#         )


#         # ---------------------------------------------------------------------
#         # Legend
#         # ---------------------------------------------------------------------

#         ax.legend()


#         # ---------------------------------------------------------------------
#         # Grid
#         # ---------------------------------------------------------------------

#         ax.grid(
#             True,
#             alpha=0.3
#         )


#         fig.tight_layout()


#         # ---------------------------------------------------------------------
#         # Save figure
#         # ---------------------------------------------------------------------

#         output_path = (
#             OUTPUT_DIR
#             / f"sensor_{s + 1:02d}.png"
#         )

#         fig.savefig(
#             output_path,
#             dpi=120
#         )

#         plt.close(fig)


#         print(
#             f"Sensor {s + 1:02d} | "
#             f"RMSE = {rmse_s:.6f} V"
#         )


#     # =========================================================================
#     # OVERALL RMSE
#     # =========================================================================

#     all_v_meas = np.concatenate(
#         all_v_meas
#     )

#     all_v_pred = np.concatenate(
#         all_v_pred
#     )


#     overall_rmse = np.sqrt(
#         np.mean(
#             (
#                 all_v_meas
#                 - all_v_pred
#             ) ** 2
#         )
#     )


#     print(
#         "\n=============================================="
#     )

#     print(
#         "OVERALL RMSE"
#     )

#     print(
#         "=============================================="
#     )

#     print(
#         f"Overall RMSE "
#         f"(all sensors, all samples) "
#         f"= {overall_rmse:.6f} V"
#     )


#     # =========================================================================
#     # SAVE RMSE SUMMARY
#     # =========================================================================

#     summary_df = pd.DataFrame({
#         "sensor_index": np.arange(
#             1,
#             n_sensors + 1
#         ),
#         "rmse": per_sensor_rmse
#     })


#     # Add overall row
#     summary_df.loc[
#         len(summary_df)
#     ] = [
#         "OVERALL",
#         overall_rmse
#     ]


#     summary_df.to_csv(
#         RMSE_SUMMARY_PATH,
#         index=False
#     )


#     # =========================================================================
#     # FINAL MESSAGE
#     # =========================================================================

#     print(
#         f"\nDa luu {n_sensors} anh vao:"
#     )

#     print(
#         OUTPUT_DIR
#     )

#     print(
#         "\nDa luu bang RMSE vao:"
#     )

#     print(
#         RMSE_SUMMARY_PATH
#     )


# # =============================================================================
# # RUN
# # =============================================================================

# if __name__ == "__main__":
#     main()


# =============================================================================
# PHYSICS-FEATURE NN TRAJECTORY EVALUATOR (Stage 1 + new Stage 2 checkpoint)
# =============================================================================
# Example:
# python Code\Compare_visualize.py ^
#   --calibration-dir D:\path\to\calibration_outputs ^
#   --data-dir Dataset\Dataset --trajectories all

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# import argparse
# import torch
# import torch.nn as nn
# from pathlib import Path

# PHYSICS_FEATURE_NAMES = (
#     "dx", "dy", "dz", "r", "Bx", "By", "Bz", "Bmag", "cos_theta_m"
# )
# PHYSICS_HIDDEN_DIMS = (32, 64, 64, 16)
# PHYSICS_EPS = 1e-12

# TRAJECTORY_FILES = {
#     "grid": ("Grid_points_coordinates.csv", "Grid_data.csv"),
#     "helix": ("Helix_points_coordinates_2.csv", "Helix_data_2.csv"),
#     "lissajous": ("Lissajous_points_coordinates.csv", "Lissajous_data.csv"),
#     "random": ("Random_points_coordinates.csv", "Random_data.csv"),
# }
# MU0_OVER_4PI = 1e-7

# class PhysicsDeltaAlphaNet(nn.Module):
#     """Architecture for the 9 physics features."""

#     def __init__(self):
#         super().__init__()
#         layers = []
#         input_dim = len(PHYSICS_FEATURE_NAMES)
#         for hidden_dim in PHYSICS_HIDDEN_DIMS:
#             layers += [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
#             input_dim = hidden_dim
#         layers += [nn.Linear(input_dim, 1)]
#         self.net = nn.Sequential(*layers)
#         self.output_scale = nn.Parameter(torch.tensor(0.05))

#     def forward(self, x_norm):
#         return (self.net(x_norm) * self.output_scale).squeeze(-1)

# def load_voltage_data(path):
#     """
#     Load measured voltage.

#     Rows    = samples
#     Columns = sensors
#     """

#     df = pd.read_csv(path)

#     voltage = df.values.astype(float)

#     print(
#         f"\nLoaded voltage data: "
#         f"{voltage.shape}"
#     )

#     return voltage, list(df.columns)

# def load_robot_pose(path):
#     """
#     Expected columns:

#         x
#         y
#         z
#         mx
#         my
#         mz

#     Position:
#         (N, 3)

#     Magnetic moment:
#         (N, 3)

#     The magnetic moment is normalized.
#     """

#     df = pd.read_csv(path)

#     required_columns = [
#         "x",
#         "y",
#         "z",
#         "mx",
#         "my",
#         "mz"
#     ]

#     for col in required_columns:
#         if col not in df.columns:
#             raise ValueError(
#                 f"Missing column '{col}' "
#                 f"in {path.name}"
#             )

#     positions = df[
#         ["x", "y", "z"]
#     ].values.astype(float)

#     m_world = df[
#         ["mx", "my", "mz"]
#     ].values.astype(float)

#     norm = np.linalg.norm(
#         m_world,
#         axis=1,
#         keepdims=True
#     )

#     if np.any(norm < 1e-12):
#         raise ValueError(
#             "Found zero magnetic-moment vector."
#         )

#     m_world = m_world / norm

#     print(
#         f"Loaded robot positions: "
#         f"{positions.shape}"
#     )

#     print(
#         f"Loaded magnetic orientations: "
#         f"{m_world.shape}"
#     )

#     return positions, m_world

# def torch_dipole_field_compare(r_vec, m_vec):
#     """Same dipole formula as the calibration forward model; inputs are (..., 3)."""
#     r = torch.linalg.vector_norm(r_vec, dim=-1, keepdim=True).clamp_min(PHYSICS_EPS)
#     mdotr = (m_vec * r_vec).sum(dim=-1, keepdim=True)
#     return MU0_OVER_4PI * (3.0 * r_vec * mdotr / r.pow(5) - m_vec / r.pow(3))


# def load_physics_checkpoint(checkpoint_path, device):
#     checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
#     required = {"model_state_dict", "feature_names", "feature_mean", "feature_std"}
#     missing = required - set(checkpoint)
#     if missing:
#         raise ValueError(f"Checkpoint {checkpoint_path.name} is missing keys: {sorted(missing)}")
#     if tuple(checkpoint["feature_names"]) != PHYSICS_FEATURE_NAMES:
#         raise ValueError("Checkpoint feature order does not match the physics-feature evaluator.")

#     feature_mean = torch.tensor(checkpoint["feature_mean"], dtype=torch.float32, device=device)
#     feature_std = torch.tensor(checkpoint["feature_std"], dtype=torch.float32, device=device)
#     if feature_mean.numel() != len(PHYSICS_FEATURE_NAMES) or torch.any(feature_std <= 0):
#         raise ValueError("Invalid feature normalisation values in checkpoint.")

#     model = PhysicsDeltaAlphaNet().to(device)
#     model.load_state_dict(checkpoint["model_state_dict"])
#     model.eval()
#     return model, feature_mean, feature_std


# def predict_physics_feature_voltage(physical_df, positions, moments, model,
#                                     feature_mean, feature_std, device):
#     """Return Stage-1 and Stage-2 voltage predictions for all poses/sensors."""
#     required_cols = {"sensor_index", "x", "y", "z", "offset", "gain"}
#     missing = required_cols - set(physical_df.columns)
#     if missing:
#         raise ValueError(f"Physical calibration is missing columns: {sorted(missing)}")

#     sensor_pos = torch.tensor(physical_df[["x", "y", "z"]].to_numpy(),
#                               dtype=torch.float32, device=device)
#     offset = torch.tensor(physical_df["offset"].to_numpy(), dtype=torch.float32, device=device)
#     gain = torch.tensor(physical_df["gain"].to_numpy(), dtype=torch.float32, device=device)
#     pose = torch.tensor(positions, dtype=torch.float32, device=device)
#     moment = torch.tensor(moments, dtype=torch.float32, device=device)

#     with torch.no_grad():
#         r_vec = sensor_pos.unsqueeze(0) - pose.unsqueeze(1)
#         m_expanded = moment.unsqueeze(1).expand_as(r_vec)
#         B = torch_dipole_field_compare(r_vec, m_expanded)
#         # The frozen sensor direction is [0, 0, 1], matching Stage 1.
#         B_sensor = B[..., 2]
#         r = torch.linalg.vector_norm(r_vec, dim=-1, keepdim=True).clamp_min(PHYSICS_EPS)
#         bmag = torch.linalg.vector_norm(B, dim=-1, keepdim=True)
#         mnorm = torch.linalg.vector_norm(m_expanded, dim=-1, keepdim=True).clamp_min(PHYSICS_EPS)
#         cos_theta_m = (r_vec * m_expanded).sum(dim=-1, keepdim=True) / (r * mnorm)
#         features = torch.cat((r_vec, r, B, bmag, cos_theta_m), dim=-1)
#         x_norm = (features.reshape(-1, len(PHYSICS_FEATURE_NAMES)) - feature_mean) / feature_std
#         delta_alpha = model(x_norm).reshape_as(B_sensor)
#         alpha = 1.0 + delta_alpha
#         stage1 = offset.unsqueeze(0) + gain.unsqueeze(0) * B_sensor
#         stage2 = offset.unsqueeze(0) + gain.unsqueeze(0) * B_sensor * alpha
#     return stage1.cpu().numpy(), stage2.cpu().numpy(), alpha.cpu().numpy(), delta_alpha.cpu().numpy()


# def evaluate_trajectory(trajectory_name, data_dir, physical_df, model,
#                         feature_mean, feature_std, output_dir, device):
#     coords_name, voltage_name = TRAJECTORY_FILES[trajectory_name]
#     coords_path, voltage_path = data_dir / coords_name, data_dir / voltage_name
#     if not coords_path.is_file() or not voltage_path.is_file():
#         raise FileNotFoundError(f"Missing files for {trajectory_name}: {coords_path} / {voltage_path}")

#     positions, moments = load_robot_pose(coords_path)
#     voltage, _ = load_voltage_data(voltage_path)
#     if voltage.shape != (len(positions), len(physical_df)):
#         raise ValueError(f"{trajectory_name}: voltage shape {voltage.shape} does not match "
#                          f"{len(positions)} poses and {len(physical_df)} sensors")

#     stage1, stage2, alpha, delta = predict_physics_feature_voltage(
#         physical_df, positions, moments, model, feature_mean, feature_std, device)
#     err1, err2 = stage1 - voltage, stage2 - voltage
#     plot_dir = output_dir / f"sensor_plots_{trajectory_name}_physics_feature_nn"
#     plot_dir.mkdir(parents=True, exist_ok=True)
#     sample_idx = np.arange(voltage.shape[0])
#     rows = []
#     for sensor_index in range(voltage.shape[1]):
#         stage1_rmse = float(np.sqrt(np.mean(err1[:, sensor_index] ** 2)))
#         stage2_rmse = float(np.sqrt(np.mean(err2[:, sensor_index] ** 2)))
#         rows.append({"trajectory": trajectory_name, "record_type": "sensor",
#                      "sensor_index": sensor_index + 1,
#                      "stage1_rmse_V": stage1_rmse,
#                      "stage1_mae_V": float(np.mean(np.abs(err1[:, sensor_index]))),
#                      "stage2_rmse_V": stage2_rmse,
#                      "stage2_mae_V": float(np.mean(np.abs(err2[:, sensor_index]))),
#                      "alpha_mean": float(alpha[:, sensor_index].mean()),
#                      "delta_alpha_mean": float(delta[:, sensor_index].mean())})

#         # Plot comparision
#         fig, ax = plt.subplots(figsize=(10, 4))
#         ax.plot(sample_idx, voltage[:, sensor_index], label="V measured", linewidth=1.0)
#         ax.plot(sample_idx, stage1[:, sensor_index], label="Dipole / Stage 1", linewidth=0.9,
#                 linestyle="--")
#         ax.plot(sample_idx, stage2[:, sensor_index], label="Computed / Stage 1 + NN", linewidth=1.0)
#         ax.set_xlabel("Sample index")
#         ax.set_ylabel("Voltage (V)")
#         ax.set_title(f"{trajectory_name.title()} | Sensor {sensor_index + 1:02d} | "
#                      f"RMSE: Stage 1={stage1_rmse:.6f} V, Stage 1+NN={stage2_rmse:.6f} V")
#         ax.grid(True, alpha=0.3)
#         ax.legend()
#         fig.tight_layout()
#         fig.savefig(plot_dir / f"sensor_{sensor_index + 1:02d}.png", dpi=120)
#         plt.close(fig)
#     rows.append({"trajectory": trajectory_name, "record_type": "overall", "sensor_index": "ALL",
#                  "stage1_rmse_V": float(np.sqrt(np.mean(err1 ** 2))),
#                  "stage1_mae_V": float(np.mean(np.abs(err1))),
#                  "stage2_rmse_V": float(np.sqrt(np.mean(err2 ** 2))),
#                  "stage2_mae_V": float(np.mean(np.abs(err2))),
#                  "alpha_mean": float(alpha.mean()), "delta_alpha_mean": float(delta.mean())})
#     output_path = output_dir / f"rmse_{trajectory_name}_physics_feature_nn.csv"
#     pd.DataFrame(rows).to_csv(output_path, index=False)
#     overall = rows[-1]
#     print(f"{trajectory_name:10s} | Stage 1 RMSE={overall['stage1_rmse_V']:.6f} V | "
#           f"Stage 1+NN RMSE={overall['stage2_rmse_V']:.6f} V | "
#           f"saved {output_path.name} + 64 plots")
#     return overall

# def load_physical_calib(path):
#     """
#     Load Stage-1 physical calibration.

#     Expected columns:
#         sensor_index
#         x
#         y
#         z
#         offset
#         gain
#     """

#     df = pd.read_csv(path)

#     df = (
#         df
#         .sort_values("sensor_index")
#         .reset_index(drop=True)
#     )

#     required_columns = [
#         "sensor_index",
#         "x",
#         "y",
#         "z",
#         "offset",
#         "gain"
#     ]

#     for col in required_columns:
#         if col not in df.columns:
#             raise ValueError(
#                 f"Missing column '{col}' in {path.name}"
#             )

#     print(
#         f"Loaded physical calibration: "
#         f"{df.shape[0]} sensors"
#     )

#     return df


# def physics_feature_compare_main():
#     parser = argparse.ArgumentParser(description="Evaluate the physics-feature Stage-2 checkpoint by trajectory.")
#     parser.add_argument("--calibration-dir", type=Path, required=True,
#                         help="Folder containing Calibration_Physical_NN_h.csv and Calibration_AlphaNN_physics.pt.")
#     parser.add_argument("--data-dir", type=Path, required=True,
#                         help="Folder containing Grid/Helix/Lissajous/Random CSV pairs.")
#     parser.add_argument("--trajectories", nargs="+", choices=(*TRAJECTORY_FILES, "all"), default=["all"])
#     parser.add_argument("--output-dir", type=Path, default=None)
#     parser.add_argument("--device", default=None, choices=("cpu", "cuda"))
#     args = parser.parse_args()

#     device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
#     output_dir = args.output_dir or args.calibration_dir / "trajectory_rmse_physics_feature_nn"
#     output_dir.mkdir(parents=True, exist_ok=True)
#     physical_path = args.calibration_dir / "Calibration_Physical_NN.csv"
#     checkpoint_path = args.calibration_dir / "Calibration_AlphaNN_physics.pt"
#     if not physical_path.is_file() or not checkpoint_path.is_file():
#         raise FileNotFoundError("Expected calibration files were not found: "
#                                 f"{physical_path} and {checkpoint_path}")

#     physical_df = load_physical_calib(physical_path)
#     model, feature_mean, feature_std = load_physics_checkpoint(checkpoint_path, device)
#     selected = list(TRAJECTORY_FILES) if "all" in args.trajectories else args.trajectories
#     summary = [evaluate_trajectory(name, args.data_dir, physical_df, model,
#                                    feature_mean, feature_std, output_dir, device)
#                for name in selected]
#     summary_path = output_dir / "rmse_trajectory_summary_physics_feature_nn.csv"
#     pd.DataFrame(summary).to_csv(summary_path, index=False)
#     print(f"\nTrajectory summary saved: {summary_path}")


# if __name__ == "__main__":
#     physics_feature_compare_main()