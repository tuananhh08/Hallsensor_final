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

# CALIB_PATH = BASE_DIR / "Calibration_Physical.csv"
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

CALIB_PATH = BASE_DIR / "Calibration_Physical.csv"
ROBOT_POSE_PATH = BASE_DIR / "Helix_points_coordinates.csv"
PHYSICAL_PATH = BASE_DIR / "Calibration_Physical.csv"
ALPHA_PATH = BASE_DIR / "Calibration_Alpha.csv"   # MOI: file alpha theo region

OUTPUT_DIR = BASE_DIR / "Residual_vs_h_and_Bz_with_alpha"  # MOI: doi ten folder de khong de len ket qua khong-alpha truoc do

OUTPUT_DIR.mkdir(exist_ok=True)

# ==========================================================
# PARAMETERS
# ==========================================================

WINDOW = 15
MU_0_4PI = 1e-7

REGION1_H_MAX = 0.040
REGION2_H_MAX = 0.055
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

def load_alpha(path):
    """Region,Alpha -> dict {1: alpha1, 2: alpha2, 3: alpha3}"""
    df = pd.read_csv(path).dropna()
    alphas = {}
    for _, row in df.iterrows():
        region_num = int(str(row["Region"]).strip().split()[-1])
        alphas[region_num] = float(row["Alpha"])
    return alphas

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


print("Loading alpha (per region)...")   # MOI

alphas = load_alpha(ALPHA_PATH)          # MOI
print(f"  alpha = {alphas}")             # MOI

# ==========================================================
# DRAW
# ==========================================================

print("\nGenerating plots...\n")

for s in range(S):
    
    # Compute Bz

    r_vec = sensor_pos[s] - robot_pos

    B = dipole_field(r_vec, m_world)

    Bz = B @ sensor_dir[s]

    # Height (chuyen len TRUOC V_pred vi gio can h de tra alpha theo region)

    h = robot_pos[:,2] - sensor_pos[s,2]

    # Alpha theo region cua tung mau (MOI)

    alpha_sample = alpha_for_h(h, alphas)

    # Compute residual (MOI: them alpha_sample vao V_pred)

    V_pred = offset[s] + alpha_sample * gain[s] * Bz

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

    ax[0].set_title("Residual vs Height (with alpha)")

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

    ax[1].set_title("Residual vs Bz (with alpha)")

    ax[1].set_xlabel("Bz (Tesla)")

    ax[1].set_ylabel("Residual (V)")

    ax[1].grid(True)

    ax[1].legend()

    plt.suptitle(
        f"Sensor {s+1:02d} (V_computed includes alpha-by-region)",
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