import numpy as np
import pandas as pd
from pathlib import Path

# =============================================================================
# CONSTANTS
# =============================================================================

MU_0_4PI = 1e-7

BASE_DIR = Path(r"D:\Downloads\Hallsensor_final\Data set 18.6")

# =============================================================================
# FILE PATHS
# =============================================================================

ORIGINAL_SENSOR_PATH = BASE_DIR / "Hall_sensor_positions.csv"

CALIB_REGION1_PATH = BASE_DIR / "calibration_region1.csv"
CALIB_REGION2_PATH = BASE_DIR / "calibration_region2.csv"
CALIB_REGION3_PATH = BASE_DIR / "calibration_region3.csv"

COORD_PATH = BASE_DIR / "grid_points_coordinates.csv"

OUTPUT_PATH = BASE_DIR / "Grid_data_computed.csv"

# =============================================================================
# LOAD CALIBRATION
# =============================================================================

def load_calibration(file_path):

    df = pd.read_csv(file_path)

    df = df.sort_values(
        "sensor_id"
    ).reset_index(drop=True)

    sensor_pos = df[
        ["x", "y", "z"]
    ].values

    offset_arr = df[
        "offset_a"
    ].values

    gain_arr = df[
        "gain_g"
    ].values

    return (
        sensor_pos,
        offset_arr,
        gain_arr
    )

# =============================================================================
# LOAD ORIGINAL SENSOR HEIGHT
# =============================================================================

def load_original_sensor_height(file_path):

    df = pd.read_csv(file_path)

    sensor_pos = df.values

    sensor_z = sensor_pos[0, 2]

    print(
        f"Reference sensor z = {sensor_z:.6f} m"
    )

    return sensor_z

# =============================================================================
# DIPOLE MODEL
# =============================================================================

def compute_Bz_single_region(
        roi_xyz,
        sensor_pos,
        m_vecs):

    r_vec = (
        sensor_pos[None, :, :]
        - roi_xyz[:, None, :]
    )

    r_norm = np.linalg.norm(
        r_vec,
        axis=2,
        keepdims=True
    )

    r_norm = np.clip(
        r_norm,
        1e-4,
        None
    )

    m_dot_r = np.sum(
        m_vecs[:, None, :]
        * r_vec,
        axis=2,
        keepdims=True
    )

    term1 = (
        3.0
        * m_dot_r
        * r_vec
        / (r_norm ** 5)
    )

    term2 = (
        m_vecs[:, None, :]
        / (r_norm ** 3)
    )

    B_vec = MU_0_4PI * (
        term1 - term2
    )

    return B_vec[:, :, 2]

# =============================================================================
# LOAD ROI DATA
# =============================================================================

print("Loading trajectory...")

coord_df = pd.read_csv(
    COORD_PATH
)

roi_xyz = coord_df[
    ["x", "y", "z"]
].values

m_vecs = coord_df[
    ["mx", "my", "mz"]
].values

N = len(roi_xyz)

print(
    f"Number of positions = {N}"
)

# =============================================================================
# LOAD CALIBRATION FILES
# =============================================================================

print("Loading calibration files...")

sensor_pos_r1, offset_r1, gain_r1 = load_calibration(
    CALIB_REGION1_PATH
)

sensor_pos_r2, offset_r2, gain_r2 = load_calibration(
    CALIB_REGION2_PATH
)

sensor_pos_r3, offset_r3, gain_r3 = load_calibration(
    CALIB_REGION3_PATH
)

num_sensors = len(sensor_pos_r1)

print(
    f"Number of sensors = {num_sensors}"
)

# =============================================================================
# LOAD ORIGINAL SENSOR HEIGHT
# =============================================================================

sensor_z_ref = load_original_sensor_height(
    ORIGINAL_SENSOR_PATH
)

# =============================================================================
# OUTPUT VOLTAGE MATRIX
# =============================================================================

V = np.zeros(
    (N, num_sensors)
)

# =============================================================================
# REGION MASKS
# =============================================================================

h = (
    roi_xyz[:, 2]
    - sensor_z_ref
)

region1_idx = np.where(
    h < 0.045
)[0]

region2_idx = np.where(
    (h >= 0.045)
    &
    (h < 0.065)
)[0]

region3_idx = np.where(
    h >= 0.065
)[0]

print(
    f"Region1 samples = {len(region1_idx)}"
)

print(
    f"Region2 samples = {len(region2_idx)}"
)

print(
    f"Region3 samples = {len(region3_idx)}"
)

# =============================================================================
# REGION 1
# =============================================================================

if len(region1_idx) > 0:

    Bz_r1 = compute_Bz_single_region(

        roi_xyz[region1_idx],

        sensor_pos_r1,

        m_vecs[region1_idx]
    )

    V[region1_idx] = (
        offset_r1[None, :]
        +
        gain_r1[None, :]
        * Bz_r1
    )

# =============================================================================
# REGION 2
# =============================================================================

if len(region2_idx) > 0:

    Bz_r2 = compute_Bz_single_region(

        roi_xyz[region2_idx],

        sensor_pos_r2,

        m_vecs[region2_idx]
    )

    V[region2_idx] = (
        offset_r2[None, :]
        +
        gain_r2[None, :]
        * Bz_r2
    )

# =============================================================================
# REGION 3
# =============================================================================

if len(region3_idx) > 0:

    Bz_r3 = compute_Bz_single_region(

        roi_xyz[region3_idx],

        sensor_pos_r3,

        m_vecs[region3_idx]
    )

    V[region3_idx] = (
        offset_r3[None, :]
        +
        gain_r3[None, :]
        * Bz_r3
    )

# =============================================================================
# SAVE CSV
# =============================================================================

column_names = [
    f"sensor {i}"
    for i in range(
        1,
        num_sensors + 1
    )
]

voltage_df = pd.DataFrame(
    V,
    columns=column_names
)

voltage_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print(
    f"\nSaved: {OUTPUT_PATH}"
)

print(
    f"Output shape: {voltage_df.shape}"
)