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



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = Path(
    r"/Users/tuananhnguyen/Downloads/Hallsensor_final/Data set 18.6"
)

CALIB_PATH = BASE_DIR / "Calibration_Physical_new.csv"
ALPHA_PATH = BASE_DIR / "Calibration_Alpha_new.csv"
ROBOT_POSE_PATH = BASE_DIR / "Helix_points_coordinates.csv"

OUTPUT_DIR = BASE_DIR / "Residual_vs_h_and_Bz_with_alpha"

OUTPUT_DIR.mkdir(exist_ok=True)

# ==========================================================
# PARAMETERS
# ==========================================================

WINDOW = 15
MU_0_4PI = 1e-7

# ==========================================================
# DIPOLE MODEL
# ==========================================================

def dipole_field(r_vec, m_vec):

    r = np.linalg.norm(r_vec, axis=1, keepdims=True)
    r = np.clip(r, 1e-6, None)

    m_dot_r = np.sum(m_vec * r_vec, axis=1, keepdims=True)

    term1 = 3 * m_dot_r * r_vec / r**5
    term2 = m_vec / r**3

    return MU_0_4PI * (term1 - term2)


# ==========================================================
# MOVING AVERAGE
# ==========================================================

def moving_average(y, window):

    return (
        pd.Series(y)
        .rolling(window=window,
                 center=True,
                 min_periods=1)
        .mean()
        .values
    )


# ==========================================================
# LOAD DATA
# ==========================================================

print("Loading calibration...")

calib = pd.read_csv(CALIB_PATH)

sensor_pos = calib[["x","y","z"]].values

offset = calib["offset"].values
gain   = calib["gain"].values

if {"nx","ny","nz"}.issubset(calib.columns):
    sensor_dir = calib[["nx","ny","nz"]].values
else:
    sensor_dir = np.tile(np.array([0,0,1]), (64,1))

print("Loading alpha(h) coefficients...")

alpha_df = pd.read_csv(ALPHA_PATH)

C0 = alpha_df.loc[alpha_df["coefficient"] == "c0", "value"].values[0]
C1 = alpha_df.loc[alpha_df["coefficient"] == "c1", "value"].values[0]

print(f"alpha(h) = {C0:.6f} + ({C1:.6f}) * h")

print("Loading robot trajectory...")

traj = pd.read_csv(ROBOT_POSE_PATH)

robot_pos = traj[["x","y","z"]].values
m_world   = traj[["mx","my","mz"]].values

print("Loading measured voltage...")

V_measured = pd.read_csv(
    BASE_DIR / "Helix_data.csv"
).values

N, S = V_measured.shape

print(N, "samples")
print(S, "sensors")


# ==========================================================
# DRAW
# ==========================================================

print("\nGenerating plots...\n")

for s in range(S):
    
    # Height (needed before V_pred, since alpha depends on h)

    h = robot_pos[:,2] - sensor_pos[s,2]

    # Compute Bz

    r_vec = sensor_pos[s] - robot_pos

    B = dipole_field(r_vec, m_world)

    Bz = B @ sensor_dir[s]

    # Compute residual (V_pred now includes alpha(h) = c0 + c1*h)

    alpha_h = C0 + C1 * h

    V_pred = offset[s] + gain[s] * Bz * alpha_h

    residual = V_measured[:, s] - V_pred

    # Sort by h

    idx = np.argsort(h)

    h_sorted = h[idx]

    res_h = residual[idx]

    ma_h = moving_average(res_h, WINDOW)

    # Sort by Bz

    idx = np.argsort(Bz)

    Bz_sorted = Bz[idx]

    res_B = residual[idx]

    ma_B = moving_average(res_B, WINDOW)

    # Plot

    fig, ax = plt.subplots(
        1,
        2,
        figsize=(14,5)
    )

    # -------------------------------------------------
    # Residual vs h
    # -------------------------------------------------

    ax[0].plot(
        h_sorted,
        res_h,
        color="tab:blue",
        alpha=0.35,
        linewidth=1.5,
        label="Residual"
    )

    ax[0].plot(
        h_sorted,
        ma_h,
        color="red",
        linewidth=2,
        label=f"Moving Avg ({WINDOW})"
    )

    ax[0].axhline(
        0,
        ls="--",
        color="black"
    )

    ax[0].set_title("Residual vs Height")

    ax[0].set_xlabel("Height h (m)")

    ax[0].set_ylabel("Residual (V)")

    ax[0].grid(True)

    ax[0].legend()


    # -------------------------------------------------
    # Residual vs Bz
    # -------------------------------------------------

    ax[1].plot(
        Bz_sorted,
        res_B,
        color="tab:blue",
        alpha=0.35,
        linewidth=1.5,
        label="Residual"
    )

    ax[1].plot(
        Bz_sorted,
        ma_B,
        color="red",
        linewidth=2,
        label=f"Moving Avg ({WINDOW})"
    )

    ax[1].axhline(
        0,
        ls="--",
        color="black"
    )

    ax[1].set_title("Residual vs Bz")

    ax[1].set_xlabel("Bz (Tesla)")

    ax[1].set_ylabel("Residual (V)")

    ax[1].grid(True)

    ax[1].legend()

    plt.suptitle(
        f"Sensor {s+1:02d}",
        fontsize=16
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        f"Sensor_{s+1:02d}.png",
        dpi=250
    )

    plt.close()

print("\n===================================")
print("Finished.")
print(f"Saved {S} figures.")
print("Output folder:")
print(OUTPUT_DIR)
print("===================================")