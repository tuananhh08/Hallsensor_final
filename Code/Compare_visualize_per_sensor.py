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

# # VOLTAGE_PATH = BASE_DIR / "Helix_data_2.csv"
# # COORDS_PATH = BASE_DIR / "Helix_points_coordinates_2.csv"
# # OUTPUT_DIR = BASE_DIR / "outputs/sosanh_linear_alpha(r)_per_sensor_Helix_2"

# VOLTAGE_PATH = BASE_DIR / "Grid_data.csv"
# COORDS_PATH = BASE_DIR / "Grid_points_coordinates.csv"
# OUTPUT_DIR = BASE_DIR / "outputs/sensor_plots_Grid_alpha(r)_per_sensor"

# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# RMSE_SUMMARY_PATH = BASE_DIR / "outputs/rmse_summary_alpha(r)_per_sensor_Grid.csv"

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

# # alpha(h) per sensor
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from pathlib import Path

# # =============================================================================
# # FILE PATHS  
# # =============================================================================
# # BASE_DIR = Path(r"/Users/tuananhnguyen/Downloads/Hallsensor_final/Data_8_2026") #MAC
# BASE_DIR = Path(r"D:\Downloads\Hallsensor_final\Data_8_2026") #WINDOWS

# PHYSICAL_PATH = BASE_DIR / "Calibration_Physical_h_per_sensor.csv"
# ALPHA_PATH = BASE_DIR / "Calibration_Alpha_h_per_sensor.csv"

# # VOLTAGE_PATH = BASE_DIR / "Helix_data_2.csv"
# # COORDS_PATH = BASE_DIR / "Helix_points_coordinates_2.csv"
# # OUTPUT_DIR = BASE_DIR / "outputs/sosanh_linear_alpha(h)_per_sensor_Helix_2"

# VOLTAGE_PATH = BASE_DIR / "Grid_data.csv"
# COORDS_PATH = BASE_DIR / "Grid_points_coordinates.csv"
# OUTPUT_DIR = BASE_DIR / "outputs/sensor_plots_Grid_alpha(h)_per_sensor"

# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# RMSE_SUMMARY_PATH = BASE_DIR / "outputs/rmse_summary_alpha(h)_per_sensor_Grid.csv"

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
#     one (c0, c1) pair per sensor (alpha_s(h) = c0_s + c1_s*h)."""
#     df = pd.read_csv(path)
#     df = df.sort_values("sensor_index").reset_index(drop=True)
#     expected_indices = np.arange(n_sensors)
#     actual_indices = df["sensor_index"].to_numpy()
#     if not np.array_equal(actual_indices, expected_indices):
#         raise ValueError(
#             "Calibration_Alpha_per_sensor_new.csv must contain exactly "
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
# # ALPHA(H) LOOKUP -- PER SENSOR
# # =============================================================================
# def alpha_for_h(h, c0, c1):
#     """h: (N,) array = z_capsule - z_sensor. Tra ve (N,) mang
#     alpha_s(h) = c0_s + c1_s*h cho MOT sensor duy nhat (c0, c1 la scalar
#     cua sensor do, khac voi ban global chi co 1 cap (c0, c1) chung)."""
#     return c0 + c1 * h


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
#     h = robot_positions[:, 2] - z                  # (N,)

#     alpha_sample = alpha_for_h(h, c0, c1)         # (N,) -- rieng cho sensor nay

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
#             f"Sensor {s+1:02d} | alpha(h) = {c0_s:.4f} + ({c1_s:.4f})*h "
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

# =============================================================================
# alpha(V_raw) per sensor -- NN-based (calibration_nn_based.py)
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from pathlib import Path


# =============================================================================
# FILE PATHS
# =============================================================================

# BASE_DIR = Path(r"/Users/tuananhnguyen/Downloads/Hallsensor_final")  # MAC
BASE_DIR = Path(r"D:\Downloads\Hallsensor_final\Data_8_2026")  # WINDOWS


# -----------------------------------------------------------------------------
# Stage-1 physical calibration output
# -----------------------------------------------------------------------------
PHYSICAL_PATH = (
    BASE_DIR
    / "calibration_nn_based_outputs"
    / "Calibration_Physical_Residual_NN.csv"
)


# -----------------------------------------------------------------------------
# Stage-2 NN checkpoint
#
# Model architecture:
#       64 -> 128 -> 256 -> 128 -> 64
#
# This checkpoint must be trained using the new MLP architecture.
# -----------------------------------------------------------------------------
STAGE2_CKPT_PATH = (
    BASE_DIR
    / "calibration_nn_based_outputs"
    / "Calibration_Stage2_Residual_NN.pt"
)


# -----------------------------------------------------------------------------
# Test data
# -----------------------------------------------------------------------------
VOLTAGE_PATH = BASE_DIR / "Helix_data_2.csv"
COORDS_PATH = BASE_DIR / "Helix_points_coordinates_2.csv"


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "visualize_nn_based_calibration_alpha_per_sensor_Helix_2"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


RMSE_SUMMARY_PATH = (
    BASE_DIR
    / "outputs"
    / "rmse_summary_nn_based_calibration_alpha_Helix_2.csv"
)


# =============================================================================
# CONSTANTS
# =============================================================================

MU0_OVER_4PI = 1e-7
N_SENSORS = 64

# New Stage-2 architecture
INPUT_DIM = 64
HIDDEN_DIM_1 = 128
HIDDEN_DIM_2 = 256
HIDDEN_DIM_3 = 128
OUTPUT_DIM = 64


# =============================================================================
# DIPOLE MODEL
# =============================================================================

def nn_dipole_field(r_vec, m_vec):
    """
    Dipole magnetic field.

    Parameters
    ----------
    r_vec : ndarray, shape (N, 3)
        Vector from capsule position to sensor position.

    m_vec : ndarray, shape (N, 3)
        Magnetic dipole moment vector.

    Returns
    -------
    B : ndarray, shape (N, 3)
        Magnetic field vector.
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
# STAGE-2 NN ARCHITECTURE
#
# 64 -> 128 -> 256 -> 128 -> 64
#
# No residual blocks.
# =============================================================================

class CalibrationMLP(nn.Module):
    """
    Stage-2 calibration neural network.

    Architecture:

        Input
          64
          |
          v
        Linear 64 -> 128
          |
        SiLU
          |
        Linear 128 -> 256
          |
        SiLU
          |
        Linear 256 -> 128
          |
        SiLU
          |
        Linear 128 -> 64
          |
        output_scale
          |
          v
        delta_alpha (64)
    """

    def __init__(
        self,
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        output_scale_init=0.05,
    ):
        super().__init__()

        # ---------------------------------------------------------
        # Fixed architecture:
        #
        # 64 -> 128 -> 256 -> 128 -> 64
        # ---------------------------------------------------------
        self.network = nn.Sequential(

            # 64 -> 128
            nn.Linear(input_dim, 128),
            nn.SiLU(),

            # 128 -> 256
            nn.Linear(128, 256),
            nn.SiLU(),

            # 256 -> 128
            nn.Linear(256, 128),
            nn.SiLU(),

            # 128 -> 64
            nn.Linear(128, output_dim),
        )

        # ---------------------------------------------------------
        # Important:
        # Initialize final layer to zero so that initially:
        #
        # delta_alpha = 0
        # alpha = 1 + delta_alpha = 1
        #
        # This preserves the same initialization idea as the
        # previous model.
        # ---------------------------------------------------------
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

        # Trainable output scaling
        self.output_scale = nn.Parameter(
            torch.tensor(float(output_scale_init))
        )

    def forward(self, voltage_normalized):
        """
        Parameters
        ----------
        voltage_normalized : torch.Tensor
            Shape: (N, 64)

        Returns
        -------
        delta_alpha : torch.Tensor
            Shape: (N, 64)
        """

        delta_alpha = self.network(voltage_normalized)

        return delta_alpha * self.output_scale


# =============================================================================
# LOAD FUNCTIONS
# =============================================================================

def nn_load_physical_calib(path):
    """
    Load Stage-1 physical calibration.

    Required columns:
        sensor_index
        x
        y
        z
        offset
        gain
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Physical calibration file not found:\n{path}"
        )

    df = pd.read_csv(path)

    required_columns = [
        "sensor_index",
        "x",
        "y",
        "z",
        "offset",
        "gain",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in physical calibration file: {missing}"
        )

    df = (
        df
        .sort_values("sensor_index")
        .reset_index(drop=True)
    )

    print(
        f"Loaded physical calib: "
        f"{df.shape[0]} sensors from {path.name}"
    )

    return df


def nn_load_stage2_checkpoint(ckpt_path, device):
    """
    Load Stage-2 checkpoint trained with:

        64 -> 128 -> 256 -> 128 -> 64

    Expected checkpoint fields:

        model_state_dict
        input_dim
        output_dim
        output_scale_init
        voltage_mean
        voltage_std

    Extra fields in the checkpoint are allowed and ignored.

    Returns
    -------
    model
    voltage_mean
    voltage_std
    """

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Stage-2 checkpoint not found:\n{ckpt_path}"
        )

    print("\nLoading Stage-2 checkpoint...")

    ckpt = torch.load(
        ckpt_path,
        map_location=device,
        weights_only=False,
    )

    # -------------------------------------------------------------------------
    # Read dimensions if they exist.
    # The architecture itself remains fixed:
    #
    # 64 -> 128 -> 256 -> 128 -> 64
    # -------------------------------------------------------------------------

    checkpoint_input_dim = int(
        ckpt.get("input_dim", INPUT_DIM)
    )

    checkpoint_output_dim = int(
        ckpt.get("output_dim", OUTPUT_DIM)
    )

    if checkpoint_input_dim != INPUT_DIM:
        raise ValueError(
            f"Checkpoint input_dim = {checkpoint_input_dim}, "
            f"but this visualization code expects {INPUT_DIM}."
        )

    if checkpoint_output_dim != OUTPUT_DIM:
        raise ValueError(
            f"Checkpoint output_dim = {checkpoint_output_dim}, "
            f"but this visualization code expects {OUTPUT_DIM}."
        )

    # -------------------------------------------------------------------------
    # output_scale_init
    # -------------------------------------------------------------------------

    output_scale_init = float(
        ckpt.get("output_scale_init", 0.05)
    )

    # -------------------------------------------------------------------------
    # Create NEW MLP
    #
    # 64 -> 128 -> 256 -> 128 -> 64
    # -------------------------------------------------------------------------

    model = CalibrationMLP(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        output_scale_init=output_scale_init,
    ).to(device)

    # -------------------------------------------------------------------------
    # Load model weights
    # -------------------------------------------------------------------------

    if "model_state_dict" not in ckpt:
        raise KeyError(
            "Checkpoint does not contain 'model_state_dict'."
        )

    state_dict = ckpt["model_state_dict"]

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        raise RuntimeError(
            "\nThe checkpoint is NOT compatible with the new architecture:\n"
            "\n"
            "Expected:\n"
            "    64 -> 128 -> 256 -> 128 -> 64\n"
            "\n"
            "The checkpoint you are loading appears to have been trained "
            "with a different architecture.\n"
            "\n"
            "Please make sure STAGE2_CKPT_PATH points to the checkpoint "
            "created after changing the Stage-2 model.\n"
            f"\nOriginal PyTorch error:\n{e}"
        ) from e

    model.eval()

    # -------------------------------------------------------------------------
    # Voltage normalization parameters
    # -------------------------------------------------------------------------

    if "voltage_mean" not in ckpt:
        raise KeyError(
            "Checkpoint does not contain 'voltage_mean'."
        )

    if "voltage_std" not in ckpt:
        raise KeyError(
            "Checkpoint does not contain 'voltage_std'."
        )

    voltage_mean = torch.tensor(
        ckpt["voltage_mean"],
        dtype=torch.float32,
        device=device,
    )

    voltage_std = torch.tensor(
        ckpt["voltage_std"],
        dtype=torch.float32,
        device=device,
    )

    # Prevent division by zero
    voltage_std = torch.clamp(
        voltage_std,
        min=1e-8,
    )

    # -------------------------------------------------------------------------
    # Print model information
    # -------------------------------------------------------------------------

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Loaded Stage-2 checkpoint: {ckpt_path.name}"
    )

    print(
        "\nStage-2 architecture:"
    )

    print(
        "    64 -> 128 -> 256 -> 128 -> 64"
    )

    print(
        f"    Total parameters     : {total_params:,}"
    )

    print(
        f"    Trainable parameters : {trainable_params:,}"
    )

    print(
        f"    output_scale_init    : {output_scale_init:.8f}"
    )

    # Expected:
    #
    # 64 -> 128 : 8,320
    # 128 -> 256: 33,024
    # 256 -> 128: 32,896
    # 128 -> 64 : 8,256
    # output_scale: 1
    #
    # Total = 82,497
    #
    expected_params = 82497

    if total_params != expected_params:
        print(
            f"WARNING: Expected {expected_params:,} parameters, "
            f"but loaded model has {total_params:,}."
        )
    else:
        print(
            "    Parameter count check: OK (82,497)"
        )

    return (
        model,
        voltage_mean,
        voltage_std,
    )


def nn_load_voltage_data(path):
    """
    Load voltage data.

    Expected:
        Rows    = samples
        Columns = 64 sensors
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Voltage file not found:\n{path}"
        )

    df = pd.read_csv(path)

    print(
        f"Loaded voltage data: {df.shape}"
    )

    return (
        df.values.astype(float),
        list(df.columns),
    )


def nn_load_robot_pose(path):
    """
    Load robot pose.

    Required columns:

        x, y, z
        mx, my, mz

    Returns
    -------
    positions : (N, 3)
    m_world   : (N, 3), normalized
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Robot pose file not found:\n{path}"
        )

    df = pd.read_csv(path)

    required_columns = [
        "x",
        "y",
        "z",
        "mx",
        "my",
        "mz",
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in robot pose file: {missing}"
        )

    positions = df[
        ["x", "y", "z"]
    ].values.astype(float)

    m_world = df[
        ["mx", "my", "mz"]
    ].values.astype(float)

    # Normalize magnetic moment
    norm = np.linalg.norm(
        m_world,
        axis=1,
        keepdims=True,
    )

    if np.any(norm < 1e-12):
        raise ValueError(
            "Robot pose file contains a zero-length "
            "magnetic moment vector."
        )

    m_world = m_world / norm

    print(
        f"Loaded robot pose: {positions.shape}"
    )

    return (
        positions,
        m_world,
    )


# =============================================================================
# COMPUTE V_pred
#
# Stage-1 physical calibration
# +
# Stage-2 NN residual correction
# =============================================================================

def nn_compute_all_vpred(
    physical_df,
    robot_positions,
    m_world,
    voltage_data,
    model,
    voltage_mean,
    voltage_std,
    device,
):
    """
    Compute predicted voltage for all sensors.

    Pipeline:

        1. Raw voltage
              |
              v
        2. Normalize voltage
              |
              v
        3. Stage-2 MLP
              |
              v
        delta_alpha
              |
              v
        alpha = 1 + delta_alpha
              |
              v
        4. Dipole physical model
              |
              v
        Bz
              |
              v
        5. Stage-1 calibration parameters
              |
              v
        V_pred = offset + gain * Bz * alpha

    Returns
    -------
    voltage_pred : (N, 64)
    delta_alpha  : (N, 64)
    """

    n_samples = robot_positions.shape[0]

    # -------------------------------------------------------------------------
    # Fixed sensor direction
    # -------------------------------------------------------------------------

    sensor_dir = np.array(
        [0.0, 0.0, 1.0]
    )

    # -------------------------------------------------------------------------
    # Calculate Bz for every sensor
    #
    # Shape:
    #     Bz_all = (N, 64)
    # -------------------------------------------------------------------------

    Bz_all = np.zeros(
        (n_samples, N_SENSORS),
        dtype=np.float64,
    )

    for s in range(N_SENSORS):

        row = physical_df.iloc[s]

        sensor_position = np.array([
            row["x"],
            row["y"],
            row["z"],
        ])

        # Sensor position - capsule position
        r_vec = (
            sensor_position
            - robot_positions
        )

        # Dipole field
        B = nn_dipole_field(
            r_vec,
            m_world,
        )

        # Sensor measures Bz
        Bz_all[:, s] = (
            B @ sensor_dir
        )

    # -------------------------------------------------------------------------
    # Stage-1 physical calibration parameters
    # -------------------------------------------------------------------------

    offset_np = physical_df[
        "offset"
    ].values.astype(np.float32)

    gain_np = physical_df[
        "gain"
    ].values.astype(np.float32)

    # -------------------------------------------------------------------------
    # Stage-2 neural network
    # -------------------------------------------------------------------------

    voltage_tensor = torch.tensor(
        voltage_data,
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():

        # Normalize voltage using training statistics
        v_norm = (
            voltage_tensor
            - voltage_mean
        ) / voltage_std

        # New MLP:
        #
        # 64 -> 128 -> 256 -> 128 -> 64
        #
        delta_alpha = (
            model(v_norm)
            .cpu()
            .numpy()
        )

    # -------------------------------------------------------------------------
    # alpha = 1 + delta_alpha
    # -------------------------------------------------------------------------

    alpha = 1.0 + delta_alpha

    # -------------------------------------------------------------------------
    # Predicted voltage
    #
    # V_pred = offset + gain * Bz * alpha
    # -------------------------------------------------------------------------

    voltage_pred = (
        offset_np[np.newaxis, :]
        + gain_np[np.newaxis, :]
        * Bz_all
        * alpha
    )

    return (
        voltage_pred,
        delta_alpha,
    )


# =============================================================================
# MAIN
# =============================================================================

def nn_main():

    # -------------------------------------------------------------------------
    # Device
    # -------------------------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Using device: {device}"
    )

    print(
        "\n============================================================"
    )

    print(
        "NN-BASED HALL SENSOR CALIBRATION VISUALIZATION"
    )

    print(
        "Stage-2 architecture: 64 -> 128 -> 256 -> 128 -> 64"
    )

    print(
        "============================================================\n"
    )

    # -------------------------------------------------------------------------
    # Load Stage-1 physical calibration
    # -------------------------------------------------------------------------

    physical_df = nn_load_physical_calib(
        PHYSICAL_PATH
    )

    # -------------------------------------------------------------------------
    # Load Stage-2 neural network
    # -------------------------------------------------------------------------

    (
        model,
        voltage_mean,
        voltage_std,
    ) = nn_load_stage2_checkpoint(
        STAGE2_CKPT_PATH,
        device,
    )

    # -------------------------------------------------------------------------
    # Load voltage data
    # -------------------------------------------------------------------------

    (
        voltage_data,
        volt_cols,
    ) = nn_load_voltage_data(
        VOLTAGE_PATH
    )

    # -------------------------------------------------------------------------
    # Load robot pose
    # -------------------------------------------------------------------------

    (
        robot_positions,
        m_world,
    ) = nn_load_robot_pose(
        COORDS_PATH
    )

    # =========================================================================
    # CHECK DATA DIMENSIONS
    # =========================================================================

    n_samples_v = (
        voltage_data.shape[0]
    )

    n_samples_pos = (
        robot_positions.shape[0]
    )

    # Same number of voltage and pose samples
    assert (
        n_samples_v == n_samples_pos
    ), (
        f"So mau dien ap ({n_samples_v}) "
        f"khac so mau toa do ({n_samples_pos})"
    )

    # Number of sensors from physical calibration
    n_sensors = (
        physical_df.shape[0]
    )

    # Same number of sensors in voltage data
    assert (
        n_sensors == voltage_data.shape[1]
    ), (
        f"So sensor trong file calib ({n_sensors}) "
        f"khac so cot dien ap ({voltage_data.shape[1]})"
    )

    # Exactly 64 sensors
    assert (
        n_sensors == N_SENSORS
    ), (
        f"Expected {N_SENSORS} sensors, "
        f"got {n_sensors}"
    )

    # Input dimension must be 64
    assert (
        voltage_data.shape[1] == INPUT_DIM
    ), (
        f"Expected voltage input dimension "
        f"{INPUT_DIM}, got {voltage_data.shape[1]}"
    )

    # =========================================================================
    # PRINT DATA INFORMATION
    # =========================================================================

    print(
        "\n============================================================"
    )

    print(
        "DATA CHECK"
    )

    print(
        "============================================================"
    )

    print(
        f"Number of samples : {n_samples_v}"
    )

    print(
        f"Number of sensors : {n_sensors}"
    )

    print(
        f"Voltage shape     : {voltage_data.shape}"
    )

    print(
        f"Position shape    : {robot_positions.shape}"
    )

    print(
        f"Moment shape      : {m_world.shape}"
    )

    print(
        "Data dimension check: OK"
    )

    # =========================================================================
    # COMPUTE V_pred
    # =========================================================================

    print(
        "\nComputing V_pred for all sensors..."
    )

    (
        voltage_pred,
        delta_alpha,
    ) = nn_compute_all_vpred(
        physical_df,
        robot_positions,
        m_world,
        voltage_data,
        model,
        voltage_mean,
        voltage_std,
        device,
    )

    # =========================================================================
    # PREPARE RMSE
    # =========================================================================

    sample_idx = np.arange(
        n_samples_v
    )

    per_sensor_rmse = []

    all_v_meas = []
    all_v_pred = []

    # =========================================================================
    # PLOT EACH SENSOR
    # =========================================================================

    print(
        "\n============================================================"
    )

    print(
        "PER-SENSOR RESULTS"
    )

    print(
        "============================================================"
    )

    for s in range(n_sensors):

        # ---------------------------------------------------------------------
        # Measured and predicted voltage
        # ---------------------------------------------------------------------

        v_meas = voltage_data[:, s]

        v_pred = voltage_pred[:, s]

        # ---------------------------------------------------------------------
        # RMSE
        # ---------------------------------------------------------------------

        rmse_s = np.sqrt(
            np.mean(
                (v_meas - v_pred) ** 2
            )
        )

        per_sensor_rmse.append(
            rmse_s
        )

        all_v_meas.append(
            v_meas
        )

        all_v_pred.append(
            v_pred
        )

        # ---------------------------------------------------------------------
        # delta_alpha statistics
        # ---------------------------------------------------------------------

        da_s = delta_alpha[:, s]

        da_mean = np.mean(
            da_s
        )

        da_std = np.std(
            da_s
        )

        da_min = np.min(
            da_s
        )

        da_max = np.max(
            da_s
        )

        # ---------------------------------------------------------------------
        # alpha statistics
        # ---------------------------------------------------------------------

        alpha_s = (
            1.0 + da_s
        )

        alpha_min = np.min(
            alpha_s
        )

        alpha_max = np.max(
            alpha_s
        )

        # ---------------------------------------------------------------------
        # Plot
        # ---------------------------------------------------------------------

        fig, ax = plt.subplots(
            figsize=(10, 4)
        )

        ax.plot(
            sample_idx,
            v_meas,
            label="V measured",
            linewidth=1.0,
        )

        ax.plot(
            sample_idx,
            v_pred,
            label="V computed (NN-based)",
            linewidth=1.0,
        )

        ax.set_xlabel(
            "Sample index"
        )

        ax.set_ylabel(
            "Voltage (V)"
        )

        ax.set_title(
            f"Sensor {s + 1:02d}"
            f" | RMSE = {rmse_s:.6f} V"
        )

        ax.legend()

        ax.grid(
            True,
            alpha=0.3,
        )

        fig.tight_layout()

        fig.savefig(
            OUTPUT_DIR
            / f"sensor_{s + 1:02d}.png",
            dpi=120,
        )

        plt.close(fig)

        # ---------------------------------------------------------------------
        # Console output
        # ---------------------------------------------------------------------

        print(
            f"Sensor {s + 1:02d} | "
            f"RMSE = {rmse_s:.6f} V | "
            f"delta_alpha mean = {da_mean:.6f} | "
            f"std = {da_std:.6f} | "
            f"min = {da_min:.6f} | "
            f"max = {da_max:.6f} | "
            f"alpha min = {alpha_min:.6f} | "
            f"alpha max = {alpha_max:.6f}"
        )

    # =========================================================================
    # OVERALL RMSE
    # =========================================================================

    all_v_meas = np.concatenate(
        all_v_meas
    )

    all_v_pred = np.concatenate(
        all_v_pred
    )

    overall_rmse = np.sqrt(
        np.mean(
            (all_v_meas - all_v_pred) ** 2
        )
    )

    print(
        "\n============================================================"
    )

    print(
        "OVERALL RESULT"
    )

    print(
        "============================================================"
    )

    print(
        f"Overall RMSE "
        f"(all sensors, all samples) = "
        f"{overall_rmse:.6f} V"
    )

    # =========================================================================
    # OVERALL delta_alpha / alpha statistics
    # =========================================================================

    overall_delta_alpha_mean = np.mean(
        delta_alpha
    )

    overall_delta_alpha_std = np.std(
        delta_alpha
    )

    overall_delta_alpha_min = np.min(
        delta_alpha
    )

    overall_delta_alpha_max = np.max(
        delta_alpha
    )

    overall_alpha_min = (
        1.0
        + overall_delta_alpha_min
    )

    overall_alpha_max = (
        1.0
        + overall_delta_alpha_max
    )

    print(
        f"delta_alpha mean = "
        f"{overall_delta_alpha_mean:.6f}"
    )

    print(
        f"delta_alpha std  = "
        f"{overall_delta_alpha_std:.6f}"
    )

    print(
        f"delta_alpha min  = "
        f"{overall_delta_alpha_min:.6f}"
    )

    print(
        f"delta_alpha max  = "
        f"{overall_delta_alpha_max:.6f}"
    )

    print(
        f"alpha min        = "
        f"{overall_alpha_min:.6f}"
    )

    print(
        f"alpha max        = "
        f"{overall_alpha_max:.6f}"
    )

    # =========================================================================
    # SAVE RMSE SUMMARY CSV
    # =========================================================================

    summary_df = pd.DataFrame({
        "sensor_index": np.arange(
            1,
            n_sensors + 1
        ),
        "rmse": per_sensor_rmse,
    })

    summary_df.loc[
        len(summary_df)
    ] = [
        "OVERALL",
        overall_rmse,
    ]

    summary_df.to_csv(
        RMSE_SUMMARY_PATH,
        index=False,
    )

    # =========================================================================
    # FINAL MESSAGE
    # =========================================================================

    print(
        "\n============================================================"
    )

    print(
        "DONE"
    )

    print(
        "============================================================"
    )

    print(
        f"Da luu {n_sensors} anh vao:"
    )

    print(
        f"  {OUTPUT_DIR}"
    )

    print(
        "\nDa luu bang RMSE vao:"
    )

    print(
        f"  {RMSE_SUMMARY_PATH}"
    )

    print(
        "\nStage-2 model:"
    )

    print(
        "  64 -> 128 -> 256 -> 128 -> 64"
    )

    print(
        "  No residual blocks"
    )

    print(
        "  Parameters = 82,497"
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    nn_main()