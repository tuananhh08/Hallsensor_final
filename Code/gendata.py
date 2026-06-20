"""
synthetic_grid.py
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# ─── Args ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--calib",      default="Calibration_PARAM.csv")
parser.add_argument("--out_coord",  default="synthetic_grid.csv")
parser.add_argument("--out_volt",   default="synthetic_data.csv")
parser.add_argument("--seed",       type=int, default=42)
args = parser.parse_args()

# BASE_DIR = Path(r"/Users/tuananhnguyen/Downloads/Hallsensor_final/Data set/Coordinate/") # MAC
BASE_DIR = Path(r"D:\Downloads\Hallsensor_final\Data set 18.6") # WINDOWS

np.random.seed(args.seed)

# ─── Constants ────────────────────────────────────────────────────────────────
MU_0_4PI = 1e-7   # mu0 / (4*pi)

# ─── Load calibration (sensor positions + calib params) ──────────────────────
print("Loading calibration file...")
calib_df = pd.read_csv(BASE_DIR/ args.calib)
calib_df = calib_df.sort_values("sensor_index").reset_index(drop=True)

sensor_pos = calib_df[["x", "y", "z"]].values          # (64, 3)
offset_arr = calib_df["offset_a_V"].values              # (64,)
gain_arr   = calib_df["gain_g_V_per_T"].values          # (64,)
Ns         = len(sensor_pos)

print(f"  Sensors         : {Ns}")
print(f"  offset range    : [{offset_arr.min():.4f}, {offset_arr.max():.4f}] V")
print(f"  gain range      : [{gain_arr.min():.4f},  {gain_arr.max():.4f}] V/T")

# ─── ROI definition ───────────────────────────────────────────────────────────
sensor_center = sensor_pos.mean(axis=0)
print(f"  Sensor center   : {sensor_center}")

roi_width  = 0.14    # m
roi_depth  = 0.14    # m
roi_height = 0.075   # m

x_min = sensor_center[0] - roi_width  / 2
x_max = sensor_center[0] + roi_width  / 2
y_min = sensor_center[1] - roi_depth  / 2
y_max = sensor_center[1] + roi_depth  / 2
z_min = sensor_center[2] + 0.015
z_max = sensor_center[2] + roi_height

num_xy = 22
num_z  = 20

x_vals = np.linspace(x_min, x_max, num_xy)
y_vals = np.linspace(y_min, y_max, num_xy)
z_vals = np.linspace(z_min, z_max, num_z)

print(f"\n  ROI x           : [{x_min:.4f}, {x_max:.4f}] m")
print(f"  ROI y           : [{y_min:.4f}, {y_max:.4f}] m")
print(f"  ROI z           : [{z_min:.4f}, {z_max:.4f}] m")

N = num_xy * num_xy * num_z
print(f"  Total poses     : {num_xy} × {num_xy} × {num_z} = {N:,}")

# ─── Build coordinate grid ────────────────────────────────────────────────────
print("\nBuilding coordinate grid...")
xx, yy, zz = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
roi_xyz = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)   # (N, 3)


# ─── Save synthetic_grid.csv ──────────────────────────────────────────────────
coord_df = pd.DataFrame({
    "x" : roi_xyz[:, 0],
    "y" : roi_xyz[:, 1],
    "z" : roi_xyz[:, 2],
    "mx": m_vecs[:, 0],
    "my": m_vecs[:, 1],
    "mz": m_vecs[:, 2],
})
coord_path = BASE_DIR / args.out_coord
coord_df.to_csv(coord_path, index=False)
print(f"  Saved {coord_path}  ({len(coord_df):,} rows)")

# ─── Compute Bz (dipole model) ────────────────────────────────────────────────
print("\nComputing Bz at all sensors (dipole model)...")
print(f"  Matrix size: ({N}, {Ns}) = {N*Ns:,} values")

# r_vec: (N, Ns, 3)
r_vec   = sensor_pos[None, :, :] - roi_xyz[:, None, :]
r_norm  = np.linalg.norm(r_vec, axis=2, keepdims=True)
r_norm  = np.clip(r_norm, 1e-4, None)                  # avoid singularity

m_dot_r = np.sum(m_vecs[:, None, :] * r_vec, axis=2, keepdims=True)

term1   = 3 * m_dot_r * r_vec / (r_norm ** 5)
term2   = m_vecs[:, None, :] / (r_norm ** 3)

B_vec   = MU_0_4PI * (term1 - term2)                   # (N, Ns, 3)
Bz      = B_vec[:, :, 2]                               # (N, Ns) — chỉ lấy Bz

print(f"  Bz range        : [{Bz.min():.6f}, {Bz.max():.6f}] T")

# # ─── Voltage from calib params ────────────────────────────────────────────────
# print("Computing voltage...")
# V_clean = offset_arr[None, :] + gain_arr[None, :] * Bz  # (N, Ns)
# print(f"  V_clean range   : [{V_clean.min():.4f}, {V_clean.max():.4f}] V")

# # ─── Add noise (SNR per sample) ───────────────────────────────────────────────
# print("Adding noise (SNR ∈ [28, 55] dB, step 0.4 dB)...")

# # Chọn ngẫu nhiên 1 giá trị SNR cho mỗi sample (hàng)
# snr_levels_db  = np.arange(28.0, 55.5, 0.4)            # 28, 28.4, ..., 55.4
# snr_per_sample = np.random.choice(snr_levels_db, size=N)  # (N,)
# snr_linear     = 10 ** (snr_per_sample / 10.0)           # (N,)

# # Signal power per sample (mean square across 64 sensors)
# sig_power = np.mean(V_clean ** 2, axis=1)               # (N,)

# # Noise std per sample: sigma = sqrt(P_signal / SNR)
# noise_std = np.sqrt(sig_power / snr_linear)             # (N,)

# # Add Gaussian noise
# noise = np.random.randn(N, Ns) * noise_std[:, None]     # (N, Ns)
# V_noisy = V_clean + noise                               # (N, Ns)

# print(f"  SNR range used  : {snr_per_sample.min():.1f} - {snr_per_sample.max():.1f} dB")
# print(f"  Noise std range : [{noise_std.min():.6f}, {noise_std.max():.6f}] V")
# print(f"  V_noisy range   : [{V_noisy.min():.4f}, {V_noisy.max():.4f}] V")

# ─── Save synthetic_data.csv ──────────────────────────────────────────────────
print("\nSaving synthetic_data.csv...")
col_names = [f"S{i+1}" for i in range(Ns)]
volt_df   = pd.DataFrame(V_clean, columns=col_names)
volt_path = BASE_DIR / args.out_volt
volt_df.to_csv(volt_path, index=False)

print(f"  Saved {volt_path}")
print(f"\n─── Summary ─────────────────────────────────────────")
print(f"  synthetic_grid.csv : {N:,} rows × 6 cols  (x,y,z,mx,my,mz)")
print(f"  synthetic_data.csv : {N:,} rows × {Ns} cols (voltage per sensor)")
print(f"  Total voltage vals : {N*Ns:,}")
print(f"─────────────────────────────────────────────────────")


# import argparse
# import numpy as np
# import pandas as pd
# from pathlib import Path

# # ─── Args ─────────────────────────────────────────────────────────────────────
# parser = argparse.ArgumentParser()
# parser.add_argument("--calib",      default="Calibration_PARAM.csv")
# parser.add_argument("--grid",       default="Grid_points_coordinates.csv")
# parser.add_argument("--output",     default="V_out_results.csv")
# parser.add_argument("--m_magnitude",type=float, default=0.1,
#                     help="Magnetic dipole moment magnitude (A·m²)")
# parser.add_argument("--noise_std",  type=float, default=0.0,
#                     help="Gaussian noise std on V_out (V), 0 = no noise")
# parser.add_argument("--seed",       type=int,   default=42)
# args = parser.parse_args()

# # BASE_DIR = Path(r"/Users/tuananhnguyen/Downloads/Hallsensor_final/Data set/Coordinate/") # MAC
# BASE_DIR = Path(r"D:\Downloads\Hallsensor_final\Data set 18.6")  # WINDOWS

# np.random.seed(args.seed)

# MU_0_4PI = 1e-7   # mu0 / (4*pi)

# # ─── Load calibration ─────────────────────────────────────────────────────────
# # Columns: sensor_index, offset_a_V, gain_g_V_per_T, x, y, z
# print("Loading calibration file...")
# calib_df   = pd.read_csv(BASE_DIR / args.calib)
# calib_df   = calib_df.sort_values("sensor_index").reset_index(drop=True)

# sensor_pos = calib_df[["x", "y", "z"]].values      # (Ns, 3)
# offset_arr = calib_df["offset_a_V"].values          # (Ns,)  [V]
# gain_arr   = calib_df["gain_g_V_per_T"].values      # (Ns,)  [V/T]
# Ns         = len(sensor_pos)
# print(f"  Loaded {Ns} sensors")
# print(f"  Sensor X range: [{sensor_pos[:,0].min():.4f}, {sensor_pos[:,0].max():.4f}]")
# print(f"  Sensor Y range: [{sensor_pos[:,1].min():.4f}, {sensor_pos[:,1].max():.4f}]")
# print(f"  Sensor Z range: [{sensor_pos[:,2].min():.4f}, {sensor_pos[:,2].max():.4f}]")

# # ─── Load grid points ─────────────────────────────────────────────────────────
# # Columns: x, y, z, mx, my, mz
# print("\nLoading grid points file...")
# grid_df = pd.read_csv(BASE_DIR / args.grid)

# roi_xyz  = grid_df[["x", "y", "z"]].values          # (N, 3) — vị trí capsule
# m_dir    = grid_df[["mx", "my", "mz"]].values        # (N, 3) — hướng moment từ (unit vector)

# # Normalize từng hàng để đảm bảo là unit vector, rồi nhân với magnitude
# m_norms  = np.linalg.norm(m_dir, axis=1, keepdims=True)
# m_norms  = np.where(m_norms < 1e-12, 1.0, m_norms)  # tránh chia 0
# m_vecs   = (m_dir / m_norms) * args.m_magnitude      # (N, 3)  [A·m²]

# N = len(roi_xyz)
# print(f"  Loaded {N} grid points")
# print(f"  Capsule X range: [{roi_xyz[:,0].min():.4f}, {roi_xyz[:,0].max():.4f}]")
# print(f"  Capsule Y range: [{roi_xyz[:,1].min():.4f}, {roi_xyz[:,1].max():.4f}]")
# print(f"  Capsule Z range: [{roi_xyz[:,2].min():.4f}, {roi_xyz[:,2].max():.4f}]")

# # ─── Compute Bz via dipole model ──────────────────────────────────────────────
# # B = (mu0/4pi) * [ 3(m·r)r/|r|^5  -  m/|r|^3 ]
# # Chỉ lấy thành phần z vì Hall sensor chỉ đo Bz
# print(f"\nComputing Bz (dipole model)...")
# print(f"  Matrix size: ({N} points × {Ns} sensors) = {N*Ns:,} values")

# # r_vec[i,s] = sensor_pos[s] - roi_xyz[i]   shape: (N, Ns, 3)
# r_vec   = sensor_pos[None, :, :] - roi_xyz[:, None, :]      # (N, Ns, 3)
# r_norm  = np.linalg.norm(r_vec, axis=2, keepdims=True)      # (N, Ns, 1)
# r_norm  = np.clip(r_norm, 1e-4, None)

# # m_dot_r[i,s] = m_vecs[i] · r_vec[i,s]
# m_dot_r = np.sum(m_vecs[:, None, :] * r_vec, axis=2, keepdims=True)  # (N, Ns, 1)

# term1   = 3 * m_dot_r * r_vec / (r_norm ** 5)
# term2   = m_vecs[:, None, :] / (r_norm ** 3)
# B_vec   = MU_0_4PI * (term1 - term2)                        # (N, Ns, 3)
# Bz      = B_vec[:, :, 2]                                    # (N, Ns)

# print(f"  Bz range: [{Bz.min():.4e}, {Bz.max():.4e}] T")

# # ─── Voltage: V_out = offset + gain * Bz ──────────────────────────────────────
# print("Computing V_out...")
# V_out = offset_arr[None, :] + gain_arr[None, :] * Bz        # (N, Ns)

# if args.noise_std > 0:
#     V_out += np.random.normal(0, args.noise_std, V_out.shape)
#     print(f"  Added Gaussian noise std={args.noise_std} V")

# print(f"  V_out range: [{V_out.min():.4f}, {V_out.max():.4f}] V")

# # ─── Save ─────────────────────────────────────────────────────────────────────
# print(f"\nSaving to {args.output}...")
# sensor_cols = [f"V_s{i}" for i in range(Ns)]
# result_df   = pd.DataFrame(V_out, columns=sensor_cols)

# # Thêm tọa độ + hướng moment
# result_df.insert(0, "mz", grid_df["mz"].values)
# result_df.insert(0, "my", grid_df["my"].values)
# result_df.insert(0, "mx", grid_df["mx"].values)
# result_df.insert(0, "z",  roi_xyz[:, 2])
# result_df.insert(0, "y",  roi_xyz[:, 1])
# result_df.insert(0, "x",  roi_xyz[:, 0])

# result_df.to_csv(BASE_DIR / args.output, index=False)
# print(f"  Saved {N} rows × {len(result_df.columns)} cols  →  {BASE_DIR / args.output}")

# # ─── Sanity check ─────────────────────────────────────────────────────────────
# print("\n─── Sanity check (first 3 points) ───")
# for i in range(min(3, N)):
#     print(f"  [{i}] xyz=({roi_xyz[i,0]:.4f},{roi_xyz[i,1]:.4f},{roi_xyz[i,2]:.4f}) "
#           f"m=({m_vecs[i,0]:.3f},{m_vecs[i,1]:.3f},{m_vecs[i,2]:.3f}) "
#           f"→ V_s0={V_out[i,0]:.4f}  V_s32={V_out[i,32]:.4f}  V_s63={V_out[i,63]:.4f} V")