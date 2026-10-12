import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(r"D:\Downloads\Hallsensor_final\Data set 18.6")

CALIB_PATH = BASE_DIR / "calibration_joint_params.csv"

COORD_PATH = BASE_DIR / "Helix_points_coordinates.csv"

MEASURED_PATH = BASE_DIR / "Helix_data.csv"

# =============================================================================
# CONSTANTS
# =============================================================================

MU_0_4PI = 1e-7

# =============================================================================
# DIPOLE MODEL
# =============================================================================

def compute_projected_field(
    roi_xyz,
    sensor_pos,
    sensor_dir,
    m_vecs
):
    """
    roi_xyz    : (N,3)
    sensor_pos : (64,3)
    sensor_dir : (64,3)
    m_vecs     : (N,3)

    return:
        B_proj : (N,64)
    """

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
        m_vecs[:, None, :] * r_vec,
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

    B_proj = np.sum(
        B_vec
        * sensor_dir[None, :, :],
        axis=2
    )

    return B_proj

# =============================================================================
# LOAD CALIBRATION
# =============================================================================

print("Loading calibration...")

calib_df = pd.read_csv(CALIB_PATH)

calib_df = calib_df.sort_values(
    "sensor_id"
).reset_index(drop=True)

sensor_pos = calib_df[
    ["x", "y", "z"]
].values

sensor_dir = calib_df[
    ["nx", "ny", "nz"]
].values

offset_arr = calib_df[
    "offset_a"
].values

gain_arr = calib_df[
    "gain_g"
].values

print(
    f"Loaded {len(sensor_pos)} sensors"
)

# =============================================================================
# LOAD HELIX COORDINATES
# =============================================================================

print("Loading helix coordinates...")

coord_df = pd.read_csv(
    COORD_PATH
)

roi_xyz = coord_df[
    ["x", "y", "z"]
].values

m_vecs = coord_df[
    ["mx", "my", "mz"]
].values

print(
    f"Loaded {len(roi_xyz)} positions"
)

# =============================================================================
# LOAD MEASURED VOLTAGE
# =============================================================================

print("Loading measured voltage...")

measured_df = pd.read_csv(
    MEASURED_PATH
)

V_measured = measured_df.values

print(
    f"Measured shape: {V_measured.shape}"
)

# =============================================================================
# COMPUTE PROJECTED FIELD
# =============================================================================

print("Computing projected field...")

B_proj = compute_projected_field(
    roi_xyz=roi_xyz,
    sensor_pos=sensor_pos,
    sensor_dir=sensor_dir,
    m_vecs=m_vecs
)

print(
    f"B_proj shape: {B_proj.shape}"
)

# =============================================================================
# COMPUTE HEIGHT
# =============================================================================

sensor_z_ref = sensor_pos[0, 2]

h = (
    roi_xyz[:, 2]
    - sensor_z_ref
)

print(
    f"h range = [{h.min():.4f}, {h.max():.4f}] m"
)

# =============================================================================
# COMPUTE ALPHA
# =============================================================================

FIELD_THRESHOLD = 5e-5  # Tesla

valid_mask = (
    np.abs(B_proj)
    > FIELD_THRESHOLD
)

alpha = np.full(
    B_proj.shape,
    np.nan
)

alpha = np.full_like(B_proj, np.nan)

alpha[valid_mask] = (
    (V_measured - offset_arr[None,:])[valid_mask]
    /
    (gain_arr[None,:] * B_proj)[valid_mask]
)

print(
    f"Valid samples: "
    f"{np.sum(valid_mask):,} / {valid_mask.size:,}"
)

print(
    f"Valid ratio: "
    f"{100*np.mean(valid_mask):.2f}%"
)
# =============================================================================
# GLOBAL STATISTICS
# =============================================================================

print("\n===== ALPHA STATISTICS =====")

print(
    f"Alpha mean   : {np.nanmean(alpha):.6f}"
)

print(
    f"Alpha median : {np.nanmedian(alpha):.6f}"
)

print(
    f"Alpha std    : {np.nanstd(alpha):.6f}"
)

print(
    "\nAlpha percentiles:"
)

print(
    np.nanpercentile(
        alpha,
        [1, 5, 25, 50, 75, 95, 99]
    )
)
# =============================================================================
# PLOT ALL 64 SENSORS
# =============================================================================

fig, axes = plt.subplots(
    8,
    8,
    figsize=(22, 22),
    sharex=True
)

axes = axes.flatten()

for i in range(64):

    axes[i].scatter(
        h,
        alpha[:, i],
        s=4
    )

    axes[i].set_title(
        f"S{i+1}"
    )

    axes[i].grid(True)

fig.suptitle(
    "Alpha vs Height - All Sensors",
    fontsize=18
)

plt.tight_layout()

plt.show()

# =============================================================================
# GLOBAL SCATTER
# =============================================================================

h_all = np.repeat(
    h,
    64
)

alpha_all = alpha.flatten()

valid_global = (
    np.isfinite(alpha_all)
)

plt.figure(
    figsize=(12, 6)
)

plt.scatter(
    h_all[valid_global],
    alpha_all[valid_global],
    s=1
)

plt.xlabel(
    "Height h (m)"
)

plt.ylabel(
    "Alpha"
)

plt.title(
    f"Global Alpha vs Height "
    f"(|B| > {FIELD_THRESHOLD:.1e} T)"
)

plt.grid(True)

plt.show()

# =============================================================================
# BINNED MEAN ALPHA
# =============================================================================

n_bins = 30

bins = np.linspace(
    h.min(),
    h.max(),
    n_bins + 1
)

bin_centers = (
    bins[:-1]
    + bins[1:]
) / 2

alpha_mean = []

h_all = np.repeat(
    h,
    64
)

alpha_all = alpha.flatten()

for i in range(n_bins):

    mask = (
        (h_all >= bins[i])
        &
        (h_all < bins[i+1])
    )

    alpha_mean.append(
        np.nanmean(
            alpha_all[mask]
        )
    )

alpha_mean = np.array(
    alpha_mean
)

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    bin_centers,
    alpha_mean,
    marker='o'
)

plt.xlabel(
    "Height h (m)"
)

plt.ylabel(
    "Mean Alpha"
)

plt.title(
    "Mean Alpha vs Height"
)

plt.grid(True)

plt.show()

# =============================================================================
# POSE-WISE ALPHA
# =============================================================================

alpha_pose = np.full(
    len(h),
    np.nan
)

for i in range(len(h)):

    mask = valid_mask[i]

    if np.sum(mask) < 5:
        continue

    x = (
        gain_arr[mask]
        * B_proj[i, mask]
    )

    y = (
        V_measured[i, mask]
        - offset_arr[mask]
    )

    alpha_pose[i] = (
        np.sum(x * y)
        /
        np.sum(x * x)
    )

print("\n===== POSE ALPHA =====")

print(
    f"Mean   : {np.nanmean(alpha_pose):.6f}"
)

print(
    f"Median : {np.nanmedian(alpha_pose):.6f}"
)

print(
    f"Std    : {np.nanstd(alpha_pose):.6f}"
)

plt.figure(
    figsize=(10,6)
)

plt.scatter(
    h,
    alpha_pose,
    s=10
)

plt.xlabel(
    "Height h (m)"
)

plt.ylabel(
    "Pose-wise alpha"
)

plt.title(
    "Pose-wise Alpha vs Height"
)

plt.grid(True)

plt.show()