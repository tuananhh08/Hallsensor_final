import numpy as np
import pandas as pd
from pathlib import Path

# =============================================================================
# CONSTANTS
# =============================================================================

MU_0_4PI = 1e-7

BASE_DIR = Path(r"/Users/tuananhnguyen/Downloads/Hallsensor_final/Data set 18.6")

# =============================================================================
# DIPOLE MODEL
# =============================================================================

def dipole_field(r_vec, m_vec):
    """
    Parameters
    ----------
    r_vec : (N,3)
        Vector from magnet -> sensor

    m_vec : (N,3)
        Magnetic moment

    Returns
    -------
    B : (N,3)
    """

    r = np.linalg.norm(r_vec, axis=1, keepdims=True)
    r = np.clip(r, 1e-6, None)

    m_dot_r = np.sum(m_vec * r_vec, axis=1, keepdims=True)

    term1 = 3.0 * m_dot_r * r_vec / (r ** 5)
    term2 = m_vec / (r ** 3)

    B = MU_0_4PI * (term1 - term2)

    return B


# =============================================================================
# LOAD CALIBRATION
# =============================================================================

print("Loading calibration...")

calib_df = pd.read_csv(BASE_DIR / "Calibration_Physical.csv")

sensor_pos = calib_df[["x", "y", "z"]].values

# Nếu file có hướng sensor thì dùng
if {"nx", "ny", "nz"}.issubset(calib_df.columns):
    sensor_dir = calib_df[["nx", "ny", "nz"]].values
else:
    sensor_dir = np.tile(np.array([0.0, 0.0, 1.0]), (len(sensor_pos), 1))

print(f"{len(sensor_pos)} sensors loaded")


# =============================================================================
# LOAD ROBOT TRAJECTORY
# =============================================================================

print("Loading trajectory...")

coord_df = pd.read_csv(BASE_DIR / "Helix_points_coordinates.csv")

robot_positions = coord_df[["x", "y", "z"]].values

m_world = coord_df[["mx", "my", "mz"]].values

print(f"{len(robot_positions)} positions loaded")


# =============================================================================
# COMPUTE Bz
# =============================================================================

print("Computing Bz...")

N = len(robot_positions)
S = len(sensor_pos)

Bz = np.zeros((N, S))

for s in range(S):

    r_vec = sensor_pos[s] - robot_positions

    B = dipole_field(r_vec, m_world)

    Bz[:, s] = B @ sensor_dir[s]

print("Done.")

print("Shape:", Bz.shape)


# =============================================================================
# REGION STATISTICS
# =============================================================================

# Tính h = z_capsule - z_sensor
# h có kích thước (N,64)

h = robot_positions[:, 2][:, None] - sensor_pos[:, 2][None, :]

regions = [
    ("Region 1", h < 0.045),                           # h < 45 mm
    ("Region 2", (h >= 0.045) & (h < 0.065)),          # 45 ~ 65 mm
    ("Region 3", h >= 0.065),                          # >=65 mm
]

print("\n========== GLOBAL Bz ==========")
print("Min :", np.min(Bz))
print("Max :", np.max(Bz))
print("Mean:", np.mean(Bz))
print("Positive:", np.sum(Bz > 0))
print("Negative:", np.sum(Bz < 0))
print("Zero    :", np.sum(Bz == 0))

for region_name, mask in regions:

    B = Bz[mask]

    print(f"\n========== {region_name} ==========")
    print("Samples :", len(B))
    print("Positive:", np.sum(B > 0))
    print("Negative:", np.sum(B < 0))
    print("Mean    :", np.mean(B))
    print("Median  :", np.median(B))
    print("Min     :", np.min(B))
    print("Max     :", np.max(B))


# =============================================================================
# SAVE
# =============================================================================

columns = [f"sensor_{i+1}" for i in range(S)]

Bz_df = pd.DataFrame(Bz, columns=columns)

output_file = BASE_DIR / "Computed_Bz.csv"

Bz_df.to_csv(output_file, index=False)

print(f"\nSaved to:\n{output_file}")