# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from pathlib import Path

# # ==========================================================
# # PATHS
# # ==========================================================

# BASE_DIR = Path(
#     r"/Users/tuananhnguyen/Downloads/Hallsensor_final/Data set 18.6"
# )

# CALIB_PATH = BASE_DIR / "Calibration_Physical_new.csv"
# ROBOT_POSE_PATH = BASE_DIR / "Helix_points_coordinates.csv"

# OUTPUT_DIR = BASE_DIR / "Residual_vs_h_and_Bz"

# OUTPUT_DIR.mkdir(exist_ok=True)

# # ==========================================================
# # PARAMETERS
# # ==========================================================

# WINDOW = 15
# MU_0_4PI = 1e-7

# # ==========================================================
# # DIPOLE MODEL
# # ==========================================================

# def dipole_field(r_vec, m_vec):

#     r = np.linalg.norm(r_vec, axis=1, keepdims=True)
#     r = np.clip(r, 1e-6, None)

#     m_dot_r = np.sum(m_vec * r_vec, axis=1, keepdims=True)

#     term1 = 3 * m_dot_r * r_vec / r**5
#     term2 = m_vec / r**3

#     return MU_0_4PI * (term1 - term2)


# # ==========================================================
# # MOVING AVERAGE
# # ==========================================================

# def moving_average(y, window):

#     return (
#         pd.Series(y)
#         .rolling(window=window,
#                  center=True,
#                  min_periods=1)
#         .mean()
#         .values
#     )


# # ==========================================================
# # LOAD DATA
# # ==========================================================

# print("Loading calibration...")

# calib = pd.read_csv(CALIB_PATH)

# sensor_pos = calib[["x","y","z"]].values

# offset = calib["offset"].values
# gain   = calib["gain"].values

# if {"nx","ny","nz"}.issubset(calib.columns):
#     sensor_dir = calib[["nx","ny","nz"]].values
# else:
#     sensor_dir = np.tile(np.array([0,0,1]), (64,1))

# print("Loading robot trajectory...")

# traj = pd.read_csv(ROBOT_POSE_PATH)

# robot_pos = traj[["x","y","z"]].values
# m_world   = traj[["mx","my","mz"]].values

# print("Loading measured voltage...")

# V_measured = pd.read_csv(
#     BASE_DIR / "Helix_data.csv"
# ).values

# N, S = V_measured.shape

# print(N, "samples")
# print(S, "sensors")


# # ==========================================================
# # DRAW
# # ==========================================================

# print("\nGenerating plots...\n")

# for s in range(S):
    
#     # Compute Bz

#     r_vec = sensor_pos[s] - robot_pos

#     B = dipole_field(r_vec, m_world)

#     Bz = B @ sensor_dir[s]

#     # Compute residual

#     V_pred = offset[s] + gain[s] * Bz

#     residual = V_measured[:, s] - V_pred

#     # Height

#     h = robot_pos[:,2] - sensor_pos[s,2]

#     # Sort by h

#     idx = np.argsort(h)

#     h_sorted = h[idx]

#     res_h = residual[idx]

#     ma_h = moving_average(res_h, WINDOW)

#     # Sort by Bz

#     idx = np.argsort(Bz)

#     Bz_sorted = Bz[idx]

#     res_B = residual[idx]

#     ma_B = moving_average(res_B, WINDOW)

#     # Plot

#     fig, ax = plt.subplots(
#         1,
#         2,
#         figsize=(14,5)
#     )

#     # -------------------------------------------------
#     # Residual vs h
#     # -------------------------------------------------

#     ax[0].plot(
#         h_sorted,
#         res_h,
#         color="tab:blue",
#         alpha=0.35,
#         linewidth=1.5,
#         label="Residual"
#     )

#     ax[0].plot(
#         h_sorted,
#         ma_h,
#         color="red",
#         linewidth=2,
#         label=f"Moving Avg ({WINDOW})"
#     )

#     ax[0].axhline(
#         0,
#         ls="--",
#         color="black"
#     )

#     ax[0].set_title("Residual vs Height")

#     ax[0].set_xlabel("Height h (m)")

#     ax[0].set_ylabel("Residual (V)")

#     ax[0].grid(True)

#     ax[0].legend()


#     # -------------------------------------------------
#     # Residual vs Bz
#     # -------------------------------------------------

#     ax[1].plot(
#         Bz_sorted,
#         res_B,
#         color="tab:blue",
#         alpha=0.35,
#         linewidth=1.5,
#         label="Residual"
#     )

#     ax[1].plot(
#         Bz_sorted,
#         ma_B,
#         color="red",
#         linewidth=2,
#         label=f"Moving Avg ({WINDOW})"
#     )

#     ax[1].axhline(
#         0,
#         ls="--",
#         color="black"
#     )

#     ax[1].set_title("Residual vs Bz")

#     ax[1].set_xlabel("Bz (Tesla)")

#     ax[1].set_ylabel("Residual (V)")

#     ax[1].grid(True)

#     ax[1].legend()

#     plt.suptitle(
#         f"Sensor {s+1:02d}",
#         fontsize=16
#     )

#     plt.tight_layout()

#     plt.savefig(
#         OUTPUT_DIR /
#         f"Sensor_{s+1:02d}.png",
#         dpi=250
#     )

#     plt.close()

# print("\n===================================")
# print("Finished.")
# print(f"Saved {S} figures.")
# print("Output folder:")
# print(OUTPUT_DIR)
# print("===================================")



# RESIDUAL CHECK = V_measured - V_calib

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from pathlib import Path

# # ==========================================================
# # PATHS
# # ==========================================================

# BASE_DIR = Path(r"/Users/tuananhnguyen/Downloads/Hallsensor_final/Data_8_2026") #MAC

# CALIB_PATH = BASE_DIR / "Calibration_Physical_new.csv"
# ALPHA_PATH = BASE_DIR / "Calibration_Alpha_new.csv"
# ROBOT_POSE_PATH = BASE_DIR / "Helix_points_coordinates_2.csv"

# OUTPUT_DIR = BASE_DIR / "Residual_vs_h_and_Bz_with_alpha"

# OUTPUT_DIR.mkdir(exist_ok=True)

# # ==========================================================
# # PARAMETERS
# # ==========================================================

# WINDOW = 10
# MU_0_4PI = 1e-7

# # ==========================================================
# # DIPOLE MODEL
# # ==========================================================

# def dipole_field(r_vec, m_vec):

#     r = np.linalg.norm(r_vec, axis=1, keepdims=True)
#     r = np.clip(r, 1e-6, None)

#     m_dot_r = np.sum(m_vec * r_vec, axis=1, keepdims=True)

#     term1 = 3 * m_dot_r * r_vec / r**5
#     term2 = m_vec / r**3

#     return MU_0_4PI * (term1 - term2)


# # ==========================================================
# # MOVING AVERAGE
# # ==========================================================

# def moving_average(y, window):

#     return (
#         pd.Series(y)
#         .rolling(window=window,
#                  center=True,
#                  min_periods=1)
#         .mean()
#         .values
#     )


# # ==========================================================
# # LOAD DATA
# # ==========================================================

# print("Loading calibration...")

# calib = pd.read_csv(CALIB_PATH)

# sensor_pos = calib[["x","y","z"]].values

# offset = calib["offset"].values
# gain   = calib["gain"].values

# if {"nx","ny","nz"}.issubset(calib.columns):
#     sensor_dir = calib[["nx","ny","nz"]].values
# else:
#     sensor_dir = np.tile(np.array([0,0,1]), (64,1))

# print("Loading alpha(h) coefficients...")

# alpha_df = pd.read_csv(ALPHA_PATH)

# C0 = alpha_df.loc[alpha_df["coefficient"] == "c0", "value"].values[0]
# C1 = alpha_df.loc[alpha_df["coefficient"] == "c1", "value"].values[0]

# print(f"alpha(h) = {C0:.6f} + ({C1:.6f}) * h")

# print("Loading robot trajectory...")

# traj = pd.read_csv(ROBOT_POSE_PATH)

# robot_pos = traj[["x","y","z"]].values
# m_world   = traj[["mx","my","mz"]].values

# print("Loading measured voltage...")

# V_measured = pd.read_csv(
#     BASE_DIR / "Helix_data_2.csv"
# ).values

# N, S = V_measured.shape

# print(N, "samples")
# print(S, "sensors")


# # ==========================================================
# # DRAW
# # ==========================================================

# print("\nGenerating plots...\n")

# for s in range(S):
    
#     # Height (needed before V_pred, since alpha depends on h)

#     h = robot_pos[:,2] - sensor_pos[s,2]

#     # Compute Bz

#     r_vec = sensor_pos[s] - robot_pos

#     B = dipole_field(r_vec, m_world)

#     Bz = B @ sensor_dir[s]

#     # Compute residual (V_pred now includes alpha(h) = c0 + c1*h)

#     alpha_h = C0 + C1 * h

#     V_pred = offset[s] + gain[s] * Bz * alpha_h

#     residual = V_measured[:, s] - V_pred

#     # Sort by h

#     idx = np.argsort(h)

#     h_sorted = h[idx]

#     res_h = residual[idx]

#     ma_h = moving_average(res_h, WINDOW)

#     # Sort by Bz

#     idx = np.argsort(Bz)

#     Bz_sorted = Bz[idx]

#     res_B = residual[idx]

#     ma_B = moving_average(res_B, WINDOW)

#     # Plot

#     fig, ax = plt.subplots(
#         1,
#         2,
#         figsize=(14,5)
#     )

#     # -------------------------------------------------
#     # Residual vs h
#     # -------------------------------------------------

#     ax[0].plot(
#         h_sorted,
#         res_h,
#         color="tab:blue",
#         alpha=0.35,
#         linewidth=1.5,
#         label="Residual"
#     )

#     ax[0].plot(
#         h_sorted,
#         ma_h,
#         color="red",
#         linewidth=2,
#         label=f"Moving Avg ({WINDOW})"
#     )

#     ax[0].axhline(
#         0,
#         ls="--",
#         color="black"
#     )

#     ax[0].set_title("Residual vs Height")

#     ax[0].set_xlabel("Height h (m)")

#     ax[0].set_ylabel("Residual (V)")

#     ax[0].grid(True)

#     ax[0].legend()


#     # -------------------------------------------------
#     # Residual vs Bz
#     # -------------------------------------------------

#     ax[1].plot(
#         Bz_sorted,
#         res_B,
#         color="tab:blue",
#         alpha=0.35,
#         linewidth=1.5,
#         label="Residual"
#     )

#     ax[1].plot(
#         Bz_sorted,
#         ma_B,
#         color="red",
#         linewidth=2,
#         label=f"Moving Avg ({WINDOW})"
#     )

#     ax[1].axhline(
#         0,
#         ls="--",
#         color="black"
#     )

#     ax[1].set_title("Residual vs Bz")

#     ax[1].set_xlabel("Bz (Tesla)")

#     ax[1].set_ylabel("Residual (V)")

#     ax[1].grid(True)

#     ax[1].legend()

#     plt.suptitle(
#         f"Sensor {s+1:02d}",
#         fontsize=16
#     )

#     plt.tight_layout()

#     plt.savefig(
#         OUTPUT_DIR /
#         f"Sensor_{s+1:02d}.png",
#         dpi=250
#     )

#     plt.close()

# print("\n===================================")
# print("Finished.")
# print(f"Saved {S} figures.")
# print("Output folder:")
# print(OUTPUT_DIR)
# print("===================================")


# VISUALIZE alpha(h)

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from pathlib import Path


# # ==========================================================
# # PATHS
# # ==========================================================

# BASE_DIR = Path(
#     r"/Users/tuananhnguyen/Downloads/Hallsensor_final/Data_8_2026")  # MAC

# CALIB_PATH = BASE_DIR / "Calibration_Physical_new.csv"
# ROBOT_POSE_PATH = BASE_DIR / "Helix_points_coordinates_2.csv"
# VOLTAGE_PATH = BASE_DIR / "Helix_data_2.csv"

# OUTPUT_DIR = BASE_DIR / "Visualize_Alpha_vs_h"

# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# # ==========================================================
# # PARAMETERS
# # ==========================================================

# WINDOW = 10

# MU_0_4PI = 1e-7

# # Loại các điểm mà |gain * Bz| quá nhỏ
# # để tránh alpha -> +/- infinity khi chia cho số gần 0.
# MIN_GB_MAGNITUDE = 1e-4


# # ==========================================================
# # DIPOLE MODEL
# # Giữ nguyên cách tính Bz như code hiện tại
# # ==========================================================

# def dipole_field(r_vec, m_vec):

#     r = np.linalg.norm(
#         r_vec,
#         axis=1,
#         keepdims=True
#     )

#     r = np.clip(
#         r,
#         1e-6,
#         None
#     )

#     m_dot_r = np.sum(
#         m_vec * r_vec,
#         axis=1,
#         keepdims=True
#     )

#     term1 = (
#         3
#         * m_dot_r
#         * r_vec
#         / r**5
#     )

#     term2 = (
#         m_vec
#         / r**3
#     )

#     return MU_0_4PI * (
#         term1 - term2
#     )


# # ==========================================================
# # MOVING AVERAGE
# # ==========================================================

# def moving_average(y, window):

#     return (
#         pd.Series(y)
#         .rolling(
#             window=window,
#             center=True,
#             min_periods=1
#         )
#         .mean()
#         .values
#     )


# # ==========================================================
# # LOAD CALIBRATION
# # ==========================================================

# print("Loading calibration...")

# calib = pd.read_csv(CALIB_PATH)

# sensor_pos = calib[
#     ["x", "y", "z"]
# ].values

# offset = calib[
#     "offset"
# ].values

# gain = calib[
#     "gain"
# ].values


# # ==========================================================
# # SENSOR DIRECTION
# # ==========================================================

# if {"nx", "ny", "nz"}.issubset(calib.columns):

#     sensor_dir = calib[
#         ["nx", "ny", "nz"]
#     ].values

# else:

#     print(
#         "No nx, ny, nz columns found."
#     )

#     print(
#         "Using default sensor direction [0, 0, 1] "
#         "for all sensors."
#     )

#     sensor_dir = np.tile(
#         np.array([0.0, 0.0, 1.0]),
#         (len(calib), 1)
#     )


# # ==========================================================
# # CHECK SENSOR NUMBER
# # ==========================================================

# n_sensors_calib = len(sensor_pos)

# print(
#     f"Number of sensors in calibration: "
#     f"{n_sensors_calib}"
# )

# if n_sensors_calib != 64:

#     print(
#         f"WARNING: Calibration contains "
#         f"{n_sensors_calib} sensors, not 64."
#     )


# # ==========================================================
# # LOAD ROBOT TRAJECTORY
# # ==========================================================

# print("\nLoading robot trajectory...")

# traj = pd.read_csv(
#     ROBOT_POSE_PATH
# )

# robot_pos = traj[
#     ["x", "y", "z"]
# ].values

# m_world = traj[
#     ["mx", "my", "mz"]
# ].values

# print(
#     f"Trajectory samples: "
#     f"{len(robot_pos)}"
# )


# # ==========================================================
# # LOAD MEASURED VOLTAGE
# # ==========================================================

# print("\nLoading measured voltage...")

# V_measured = pd.read_csv(
#     VOLTAGE_PATH
# ).values

# print(
#     f"Voltage shape: "
#     f"{V_measured.shape}"
# )


# # ==========================================================
# # ALIGN NUMBER OF SAMPLES
# # ==========================================================

# N_voltage = V_measured.shape[0]
# N_robot = robot_pos.shape[0]

# N = min(
#     N_voltage,
#     N_robot
# )

# if N_voltage != N_robot:

#     print(
#         "\nWARNING:"
#         "\nNumber of voltage samples and trajectory samples differ."
#         f"\nVoltage samples   = {N_voltage}"
#         f"\nTrajectory samples = {N_robot}"
#         f"\nUsing first {N} samples."
#     )

# V_measured = V_measured[:N]
# robot_pos = robot_pos[:N]
# m_world = m_world[:N]


# # ==========================================================
# # CHECK SENSOR NUMBER
# # ==========================================================

# S_voltage = V_measured.shape[1]

# if S_voltage != n_sensors_calib:

#     raise ValueError(
#         f"Sensor number mismatch!\n"
#         f"Calibration sensors = {n_sensors_calib}\n"
#         f"Voltage sensors     = {S_voltage}"
#     )


# S = S_voltage

# print(
#     f"\nProcessing {S} sensors..."
# )


# # ==========================================================
# # MAIN CALCULATION
# #
# # alpha_emp =
# # (V_measured - offset) / (gain * Bz)
# # ==========================================================

# for s in range(S):

#     print(
#         f"Processing sensor "
#         f"{s + 1:02d}/{S}"
#     )


#     # ------------------------------------------------------
#     # Height h
#     #
#     # h = z_capsule - z_sensor
#     # ------------------------------------------------------

#     h = (
#         robot_pos[:, 2]
#         - sensor_pos[s, 2]
#     )


#     # ------------------------------------------------------
#     # Compute magnetic field B
#     #
#     # Same dipole model as original code
#     # ------------------------------------------------------

#     r_vec = (
#         sensor_pos[s]
#         - robot_pos
#     )

#     B = dipole_field(
#         r_vec,
#         m_world
#     )


#     # ------------------------------------------------------
#     # Project B onto sensor direction
#     # ------------------------------------------------------

#     Bz = B @ sensor_dir[s]


#     # ------------------------------------------------------
#     # Compute denominator
#     #
#     # gain * Bz
#     # ------------------------------------------------------

#     gB = gain[s] * Bz


#     # ------------------------------------------------------
#     # Valid mask
#     #
#     # Avoid division by very small values
#     # ------------------------------------------------------

#     valid_mask = (
#         np.abs(gB)
#         >= MIN_GB_MAGNITUDE
#     )


#     # ------------------------------------------------------
#     # Empirical alpha
#     #
#     # alpha = (V - offset) / (gain * Bz)
#     # ------------------------------------------------------

#     alpha_emp = np.full(
#         N,
#         np.nan
#     )

#     alpha_emp[valid_mask] = (
#         V_measured[valid_mask, s]
#         - offset[s]
#     ) / gB[valid_mask]


#     # ------------------------------------------------------
#     # Diagnostic
#     # ------------------------------------------------------

#     n_valid = np.sum(valid_mask)
#     n_invalid = N - n_valid

#     print(
#         f"    Valid points   : "
#         f"{n_valid}/{N}"
#     )

#     print(
#         f"    Removed points : "
#         f"{n_invalid}"
#     )

#     if n_valid > 0:

#         print(
#             f"    Alpha median   : "
#             f"{np.nanmedian(alpha_emp):.6f}"
#         )

#         print(
#             f"    Alpha mean     : "
#             f"{np.nanmean(alpha_emp):.6f}"
#         )

#         print(
#             f"    Alpha std      : "
#             f"{np.nanstd(alpha_emp):.6f}"
#         )

#         print(
#             f"    Bz range       : "
#             f"[{Bz.min():.6e}, {Bz.max():.6e}] T"
#         )


#     # ======================================================
#     # SORT BY h
#     # ======================================================

#     idx = np.argsort(h)

#     h_sorted = h[idx]

#     alpha_sorted = alpha_emp[idx]


#     # ------------------------------------------------------
#     # Moving average
#     #
#     # np.nan cannot be used directly with normal rolling mean
#     # so use pandas Series, which handles NaN naturally.
#     # ------------------------------------------------------

#     alpha_ma = moving_average(
#         alpha_sorted,
#         WINDOW
#     )


#     # ======================================================
#     # PLOT
#     # ======================================================

#     fig, ax = plt.subplots(
#         figsize=(9, 6)
#     )


#     # ------------------------------------------------------
#     # Raw empirical alpha
#     # ------------------------------------------------------

#     ax.plot(
#         h_sorted,
#         alpha_sorted,
#         color="tab:blue",
#         alpha=0.35,
#         linewidth=1.2,
#         label=r"$\alpha_{emp}$"
#     )


#     # ------------------------------------------------------
#     # Moving average
#     # ------------------------------------------------------

#     ax.plot(
#         h_sorted,
#         alpha_ma,
#         color="red",
#         linewidth=2.0,
#         label=f"Moving Avg ({WINDOW})"
#     )


#     # ------------------------------------------------------
#     # Reference alpha = 1
#     # ------------------------------------------------------

#     ax.axhline(
#         1.0,
#         linestyle="--",
#         color="black",
#         linewidth=1.2,
#         label=r"$\alpha = 1$"
#     )


#     # ------------------------------------------------------
#     # Labels
#     # ------------------------------------------------------

#     ax.set_xlabel(
#         "Height h (m)"
#     )

#     ax.set_ylabel(
#         r"$\alpha_{emp} = \frac{V_{measured}-offset}{gain \cdot B_z}$"
#     )

#     ax.set_title(
#         f"Sensor {s + 1:02d} - "
#         r"Empirical $\alpha(h)$"
#     )

#     ax.grid(
#         True,
#         alpha=0.3
#     )

#     ax.legend()


#     # ------------------------------------------------------
#     # Tight layout
#     # ------------------------------------------------------

#     plt.tight_layout()


#     # ------------------------------------------------------
#     # Save
#     # ------------------------------------------------------

#     output_file = (
#         OUTPUT_DIR
#         / f"Sensor_{s + 1:02d}_alpha_vs_h.png"
#     )

#     plt.savefig(
#         output_file,
#         dpi=250,
#         bbox_inches="tight"
#     )

#     plt.close()


# # ==========================================================
# # FINISHED
# # ==========================================================

# print("\n===================================")
# print("Finished.")
# print(f"Processed {S} sensors.")
# print(f"Saved {S} figures.")
# print("Output folder:")
# print(OUTPUT_DIR)
# print("===================================")



# tinh Bz theo r 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ==========================================================
# PATHS
# ==========================================================

# BASE_DIR = Path(r"/Users/tuananhnguyen/Downloads/Hallsensor_final/Data_8_2026")  # MAC
BASE_DIR = Path(r"D:\Downloads\Hallsensor_final\Data_8_2026") #WINDOWS

CALIB_PATH = BASE_DIR / "Calibration_Physical_new.csv"
ROBOT_POSE_PATH = BASE_DIR / "Helix_points_coordinates_2.csv"

OUTPUT_DIR = BASE_DIR / "Bz_vs_r"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# PARAMETERS
# ==========================================================

MU_0_4PI = 1e-7


# ==========================================================
# DIPOLE MODEL
# Giữ nguyên công thức từ code trước
# ==========================================================

def dipole_field(r_vec, m_vec):

    r = np.linalg.norm(
        r_vec,
        axis=1,
        keepdims=True
    )

    r = np.clip(
        r,
        1e-6,
        None
    )

    m_dot_r = np.sum(
        m_vec * r_vec,
        axis=1,
        keepdims=True
    )

    term1 = (
        3
        * m_dot_r
        * r_vec
        / r**5
    )

    term2 = (
        m_vec
        / r**3
    )

    return MU_0_4PI * (
        term1 - term2
    )


# ==========================================================
# LOAD CALIBRATION
# ==========================================================

print("Loading calibration...")

calib = pd.read_csv(
    CALIB_PATH
)

sensor_pos = calib[
    ["x", "y", "z"]
].values


# ==========================================================
# SENSOR DIRECTION
# ==========================================================

if {"nx", "ny", "nz"}.issubset(calib.columns):

    sensor_dir = calib[
        ["nx", "ny", "nz"]
    ].values

else:

    print(
        "nx, ny, nz not found."
    )

    print(
        "Using default sensor direction [0, 0, 1]."
    )

    sensor_dir = np.tile(
        np.array([0.0, 0.0, 1.0]),
        (len(sensor_pos), 1)
    )


# ==========================================================
# LOAD ROBOT TRAJECTORY
# ==========================================================

print("\nLoading robot trajectory...")

traj = pd.read_csv(
    ROBOT_POSE_PATH
)

robot_pos = traj[
    ["x", "y", "z"]
].values

m_world = traj[
    ["mx", "my", "mz"]
].values


# ==========================================================
# NORMALIZE MAGNETIZATION VECTOR
# ==========================================================

m_norm = np.linalg.norm(
    m_world,
    axis=1,
    keepdims=True
)

m_world_normalized = np.zeros_like(
    m_world
)

valid_m = (
    m_norm[:, 0] > 0
)

m_world_normalized[valid_m] = (
    m_world[valid_m]
    / m_norm[valid_m]
)


# ==========================================================
# NUMBER OF SENSORS
# ==========================================================

S = len(sensor_pos)

print(
    f"Number of sensors: {S}"
)

print(
    f"Number of trajectory samples: "
    f"{len(robot_pos)}"
)


# ==========================================================
# CALCULATE Bz VS r
# ==========================================================

print("\nCalculating Bz(r)...\n")


for s in range(S):

    print(
        f"Processing sensor "
        f"{s + 1:02d}/{S}"
    )


    # ------------------------------------------------------
    # Relative position vector
    #
    # r_vec = sensor_pos - capsule_pos
    # ------------------------------------------------------

    r_vec = (
        sensor_pos[s]
        - robot_pos
    )


    # ------------------------------------------------------
    # Euclidean distance
    #
    # r = |sensor_pos - capsule_pos|
    # ------------------------------------------------------

    r = np.linalg.norm(
        r_vec,
        axis=1
    )


    # ------------------------------------------------------
    # Dipole magnetic field
    # ------------------------------------------------------

    B = dipole_field(
        r_vec,
        m_world_normalized
    )


    # ------------------------------------------------------
    # Projection onto sensor direction
    #
    # Bz = B . sensor_dir
    # ------------------------------------------------------

    Bz = B @ sensor_dir[s]


    # ------------------------------------------------------
    # Sort according to r
    # ------------------------------------------------------

    idx = np.argsort(r)

    r_sorted = r[idx]
    Bz_sorted = Bz[idx]


    # ======================================================
    # PLOT
    # ======================================================

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )


    ax.plot(
        r_sorted,
        Bz_sorted,
        linewidth=1.8,
        label=r"$B_z$"
    )


    # ------------------------------------------------------
    # Bz = 0 reference
    # ------------------------------------------------------

    ax.axhline(
        0,
        linestyle="--",
        color="black",
        linewidth=1,
        label=r"$B_z = 0$"
    )


    # ------------------------------------------------------
    # Labels
    # ------------------------------------------------------

    ax.set_xlabel(
        r"$r = |\mathbf{r}_{sensor} - "
        r"\mathbf{r}_{capsule}|$ (m)"
    )

    ax.set_ylabel(
        r"$B_z$ (Tesla)"
    )

    ax.set_title(
        f"Sensor {s + 1:02d} - "
        r"$B_z$ vs Distance $r$"
    )

    ax.grid(
        True,
        alpha=0.3
    )

    ax.legend()


    plt.tight_layout()


    # ------------------------------------------------------
    # Save figure
    # ------------------------------------------------------

    output_file = (
        OUTPUT_DIR
        / f"Sensor_{s + 1:02d}_Bz_vs_r.png"
    )

    plt.savefig(
        output_file,
        dpi=250,
        bbox_inches="tight"
    )

    plt.close()


    # ------------------------------------------------------
    # Diagnostic
    # ------------------------------------------------------

    print(
        f"    r range  = "
        f"[{r.min():.6e}, {r.max():.6e}] m"
    )

    print(
        f"    Bz range = "
        f"[{Bz.min():.6e}, {Bz.max():.6e}] T"
    )


# ==========================================================
# FINISHED
# ==========================================================

print("\n===================================")
print("Finished.")
print(f"Generated {S} plots.")
print("Output folder:")
print(OUTPUT_DIR)
print("===================================")


# # alpha theo r

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from pathlib import Path


# # ==========================================================
# # PATHS
# # ==========================================================

# # BASE_DIR = Path(r"/Users/tuananhnguyen/Downloads/Hallsensor_final/Data_8_2026")  # MAC
# BASE_DIR = Path(r"D:\Downloads\Hallsensor_final\Data_8_2026")

# CALIB_PATH = BASE_DIR / "Calibration_Physical_new.csv"
# ROBOT_POSE_PATH = BASE_DIR / "Helix_points_coordinates_2.csv"
# VOLTAGE_PATH = BASE_DIR / "Helix_data_2.csv"

# OUTPUT_DIR = BASE_DIR / "Empirical_Alpha_vs_r_grid"

# OUTPUT_DIR.mkdir(
#     parents=True,
#     exist_ok=True
# )


# # ==========================================================
# # PARAMETERS
# # ==========================================================

# MU_0_4PI = 1e-7

# # Loại các điểm có |gain * Bz| quá nhỏ
# # để tránh chia cho số gần 0.
# MIN_GB_MAGNITUDE = 1e-4

# # Moving average
# WINDOW = 10


# # ==========================================================
# # DIPOLE MODEL
# # ==========================================================

# def dipole_field(r_vec, m_vec):

#     r = np.linalg.norm(
#         r_vec,
#         axis=1,
#         keepdims=True
#     )

#     r = np.clip(
#         r,
#         1e-6,
#         None
#     )

#     m_dot_r = np.sum(
#         m_vec * r_vec,
#         axis=1,
#         keepdims=True
#     )

#     term1 = (
#         3
#         * m_dot_r
#         * r_vec
#         / r**5
#     )

#     term2 = (
#         m_vec
#         / r**3
#     )

#     return MU_0_4PI * (
#         term1 - term2
#     )


# # ==========================================================
# # MOVING AVERAGE
# # ==========================================================

# def moving_average(y, window):

#     return (
#         pd.Series(y)
#         .rolling(
#             window=window,
#             center=True,
#             min_periods=1
#         )
#         .mean()
#         .values
#     )


# # ==========================================================
# # LOAD CALIBRATION
# # ==========================================================

# print("Loading calibration...")

# calib = pd.read_csv(
#     CALIB_PATH
# )

# sensor_pos = calib[
#     ["x", "y", "z"]
# ].values

# offset = calib[
#     "offset"
# ].values

# gain = calib[
#     "gain"
# ].values


# # ==========================================================
# # SENSOR DIRECTION
# # ==========================================================

# if {"nx", "ny", "nz"}.issubset(calib.columns):

#     sensor_dir = calib[
#         ["nx", "ny", "nz"]
#     ].values

# else:

#     print(
#         "nx, ny, nz not found."
#     )

#     print(
#         "Using default sensor direction [0, 0, 1] "
#         "for all sensors."
#     )

#     sensor_dir = np.tile(
#         np.array([0.0, 0.0, 1.0]),
#         (len(sensor_pos), 1)
#     )


# # ==========================================================
# # CHECK NUMBER OF SENSORS
# # ==========================================================

# n_sensors_calib = len(sensor_pos)

# print(
#     f"Number of sensors in calibration: "
#     f"{n_sensors_calib}"
# )


# # ==========================================================
# # LOAD ROBOT TRAJECTORY
# # ==========================================================

# print("\nLoading robot trajectory...")

# traj = pd.read_csv(
#     ROBOT_POSE_PATH
# )

# robot_pos = traj[
#     ["x", "y", "z"]
# ].values

# m_world = traj[
#     ["mx", "my", "mz"]
# ].values


# # ==========================================================
# # NORMALIZE MAGNETIZATION VECTOR
# # ==========================================================
# # Giữ m_world theo trajectory nhưng chuẩn hóa về unit vector.

# m_norm = np.linalg.norm(
#     m_world,
#     axis=1,
#     keepdims=True
# )

# m_world_normalized = np.zeros_like(
#     m_world
# )

# valid_m = (
#     m_norm[:, 0] > 0
# )

# m_world_normalized[valid_m] = (
#     m_world[valid_m]
#     / m_norm[valid_m]
# )


# # ==========================================================
# # LOAD MEASURED VOLTAGE
# # ==========================================================

# print("\nLoading measured voltage...")

# V_measured = pd.read_csv(
#     VOLTAGE_PATH
# ).values

# print(
#     f"Voltage shape: {V_measured.shape}"
# )


# # ==========================================================
# # ALIGN NUMBER OF SAMPLES
# # ==========================================================

# N_voltage = V_measured.shape[0]
# N_robot = robot_pos.shape[0]

# N = min(
#     N_voltage,
#     N_robot
# )

# if N_voltage != N_robot:

#     print(
#         "\nWARNING:"
#         "\nNumber of voltage samples and trajectory samples differ."
#         f"\nVoltage samples    = {N_voltage}"
#         f"\nTrajectory samples = {N_robot}"
#         f"\nUsing first {N} samples."
#     )

# V_measured = V_measured[:N]
# robot_pos = robot_pos[:N]
# m_world_normalized = m_world_normalized[:N]


# # ==========================================================
# # CHECK SENSOR NUMBER
# # ==========================================================

# S_voltage = V_measured.shape[1]

# if S_voltage != n_sensors_calib:

#     raise ValueError(
#         f"Sensor number mismatch!\n"
#         f"Calibration sensors = {n_sensors_calib}\n"
#         f"Voltage sensors     = {S_voltage}"
#     )

# S = S_voltage


# # ==========================================================
# # MAIN CALCULATION
# #
# # alpha_emp =
# # (V_measured - offset) / (gain * Bz)
# #
# # r =
# # |sensor_pos - capsule_pos|
# # ==========================================================

# print("\n===================================")
# print("Calculating empirical alpha(r)...")
# print("===================================\n")


# for s in range(S):

#     print(
#         f"Processing sensor {s + 1:02d}/{S}"
#     )


#     # ======================================================
#     # RELATIVE POSITION
#     # ======================================================

#     r_vec = (
#         sensor_pos[s]
#         - robot_pos
#     )


#     # ======================================================
#     # EUCLIDEAN DISTANCE
#     #
#     # r = |sensor_pos - capsule_pos|
#     # ======================================================

#     r = np.linalg.norm(
#         r_vec,
#         axis=1
#     )


#     # ======================================================
#     # DIPOLAR FIELD
#     # ======================================================

#     B = dipole_field(
#         r_vec,
#         m_world_normalized
#     )


#     # ======================================================
#     # PROJECT B ONTO SENSOR DIRECTION
#     # ======================================================

#     Bz = B @ sensor_dir[s]


#     # ======================================================
#     # DENOMINATOR
#     # ======================================================

#     gB = (
#         gain[s]
#         * Bz
#     )


#     # ======================================================
#     # VALID MASK
#     #
#     # Avoid division by values close to zero.
#     # ======================================================

#     valid_mask = (
#         np.abs(gB)
#         >= MIN_GB_MAGNITUDE
#     )


#     # ======================================================
#     # EMPIRICAL ALPHA
#     #
#     # alpha_emp =
#     # (V_measured - offset) / (gain * Bz)
#     # ======================================================

#     alpha_emp = np.full(
#         N,
#         np.nan
#     )

#     alpha_emp[valid_mask] = (
#         V_measured[valid_mask, s]
#         - offset[s]
#     ) / gB[valid_mask]


#     # ======================================================
#     # DIAGNOSTICS
#     # ======================================================

#     n_valid = np.sum(
#         valid_mask
#     )

#     n_invalid = (
#         N - n_valid
#     )

#     print(
#         f"    Valid points : "
#         f"{n_valid}/{N}"
#     )

#     print(
#         f"    Removed      : "
#         f"{n_invalid}"
#     )

#     if n_valid > 0:

#         print(
#             f"    r range      : "
#             f"[{r.min():.6e}, {r.max():.6e}] m"
#         )

#         print(
#             f"    Bz range     : "
#             f"[{Bz.min():.6e}, {Bz.max():.6e}] T"
#         )

#         print(
#             f"    alpha median : "
#             f"{np.nanmedian(alpha_emp):.6f}"
#         )

#         print(
#             f"    alpha mean   : "
#             f"{np.nanmean(alpha_emp):.6f}"
#         )

#         print(
#             f"    alpha std    : "
#             f"{np.nanstd(alpha_emp):.6f}"
#         )


#     # ======================================================
#     # SORT BY r
#     # ======================================================

#     idx = np.argsort(
#         r
#     )

#     r_sorted = r[idx]

#     alpha_sorted = alpha_emp[idx]


#     # ======================================================
#     # MOVING AVERAGE
#     # ======================================================

#     alpha_ma = moving_average(
#         alpha_sorted,
#         WINDOW
#     )


#     # ======================================================
#     # PLOT
#     # ======================================================

#     fig, ax = plt.subplots(
#         figsize=(9, 6)
#     )


#     # ------------------------------------------------------
#     # RAW EMPIRICAL ALPHA
#     # ------------------------------------------------------

#     ax.plot(
#         r_sorted,
#         alpha_sorted,
#         color="tab:blue",
#         alpha=0.35,
#         linewidth=1.2,
#         label=r"$\alpha_{emp}$"
#     )


#     # ------------------------------------------------------
#     # MOVING AVERAGE
#     # ------------------------------------------------------

#     ax.plot(
#         r_sorted,
#         alpha_ma,
#         color="red",
#         linewidth=2.0,
#         label=f"Moving Avg ({WINDOW})"
#     )


#     # ------------------------------------------------------
#     # REFERENCE alpha = 1
#     # ------------------------------------------------------

#     ax.axhline(
#         1.0,
#         linestyle="--",
#         color="black",
#         linewidth=1.2,
#         label=r"$\alpha = 1$"
#     )


#     # ======================================================
#     # AXIS LABELS
#     # ======================================================

#     ax.set_xlabel(
#         r"$r = |\mathbf{r}_{sensor} - "
#         r"\mathbf{r}_{capsule}|$ (m)"
#     )

#     ax.set_ylabel(
#         r"$\alpha_{emp}"
#         r"=\frac{V_{measured}-offset}"
#         r"{gain\cdot B_z}$"
#     )

#     ax.set_title(
#         f"Sensor {s + 1:02d} - "
#         r"Empirical $\alpha(r)$"
#     )

#     ax.grid(
#         True,
#         alpha=0.3
#     )

#     ax.legend()


#     # ======================================================
#     # SAVE
#     # ======================================================

#     plt.tight_layout()

#     output_file = (
#         OUTPUT_DIR
#         / f"Sensor_{s + 1:02d}_alpha_vs_r.png"
#     )

#     plt.savefig(
#         output_file,
#         dpi=250,
#         bbox_inches="tight"
#     )

#     plt.close()


# # ==========================================================
# # FINISHED
# # ==========================================================

# print("\n===================================")
# print("Finished.")
# print(f"Processed {S} sensors.")
# print(f"Saved {S} figures.")
# print("Output folder:")
# print(OUTPUT_DIR)
# print("===================================")