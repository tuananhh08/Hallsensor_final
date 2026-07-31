"""
synthetic_grid.py
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# ─── Args ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--calib",      default="Calibration_Physical_new.csv")
parser.add_argument("--alpha",      default="Calibration_Alpha_new.csv")
parser.add_argument("--out_coord",  default="synthetic_grid_coordinates.csv")
parser.add_argument("--out_volt",   default="synthetic_grid_data.csv")
parser.add_argument("--seed",       type=int, default=42)
args = parser.parse_args()

# BASE_DIR = Path(r"/Users/tuananhnguyen/Downloads/Hallsensor_final/Data set 18.6") #MAC
BASE_DIR = Path(r"D:\Downloads\Hallsensor_final\Data set 18.6") # WINDOWS

np.random.seed(args.seed)

# ─── Constants ────────────────────────────────────────────────────────────────
MU_0_4PI = 1e-7   # mu0 / (4*pi)

# ─── Load calibration (sensor positions + calib params) ──────────────────────
print("Loading calibration file...")
calib_df = pd.read_csv(BASE_DIR/ args.calib)
calib_df = calib_df.sort_values("sensor_index").reset_index(drop=True)

sensor_pos = calib_df[["x", "y", "z"]].values          # (64, 3)
offset_arr = calib_df["offset"].values              # (64,)
gain_arr   = calib_df["gain"].values          # (64,)
Ns         = len(sensor_pos)

print(f"  Sensors         : {Ns}")
print(f"  offset range    : [{offset_arr.min():.4f}, {offset_arr.max():.4f}] V")
print(f"  gain range      : [{gain_arr.min():.4f},  {gain_arr.max():.4f}] V/T")

# ─── ROI definition ───────────────────────────────────────────────────────────
sensor_center = sensor_pos.mean(axis=0)
print(f"  Sensor center   : {sensor_center}")

x_min = -0.045  # -4.5cm
x_max = 0.11    # 11cm

y_min = 0.63   # 63cm
y_max = 0.78   # 78cm

z_min = -0.0936  # 2.5cm above sensor plane
z_max = -0.0436  # 7.5cm above sensor plane

num_xy = 21
num_z  = 16

# x_vals = np.linspace(x_min, x_max, num_xy)
# y_vals = np.linspace(y_min, y_max, num_xy)
# z_vals = np.linspace(z_min, z_max, num_z)


# print(f"\n  ROI x           : [{x_min:.4f}, {x_max:.4f}] m")
# print(f"  ROI y           : [{y_min:.4f}, {y_max:.4f}] m")
# print(f"  ROI z           : [{z_min:.4f}, {z_max:.4f}] m")

# N = num_xy * num_xy * num_z
# print(f"  Total poses     : {num_xy} × {num_xy} × {num_z} = {N:,}")

# # ─── Build coordinate grid ────────────────────────────────────────────────────
# print("\nBuilding coordinate grid...")
# xx, yy, zz = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
# roi_xyz = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)   # (N, 3)


# # ─── Save synthetic_grid.csv ──────────────────────────────────────────────────
# coord_df = pd.DataFrame({
#     "x" : roi_xyz[:, 0],
#     "y" : roi_xyz[:, 1],
#     "z" : roi_xyz[:, 2],
#     "mx": m_vecs[:, 0],
#     "my": m_vecs[:, 1],
#     "mz": m_vecs[:, 2],
# })
# coord_path = BASE_DIR / args.out_coord
# coord_df.to_csv(coord_path, index=False)
# print(f"  Saved {coord_path}  ({len(coord_df):,} rows)")

# # ─── Compute Bz (dipole model) ────────────────────────────────────────────────
# print("\nComputing Bz at all sensors (dipole model)...")
# print(f"  Matrix size: ({N}, {Ns}) = {N*Ns:,} values")

# # r_vec: (N, Ns, 3)
# r_vec   = sensor_pos[None, :, :] - roi_xyz[:, None, :]
# r_norm  = np.linalg.norm(r_vec, axis=2, keepdims=True)
# r_norm  = np.clip(r_norm, 1e-4, None)                  # avoid singularity

# m_dot_r = np.sum(m_vecs[:, None, :] * r_vec, axis=2, keepdims=True)

# term1   = 3 * m_dot_r * r_vec / (r_norm ** 5)
# term2   = m_vecs[:, None, :] / (r_norm ** 3)

# B_vec   = MU_0_4PI * (term1 - term2)                   # (N, Ns, 3)
# Bz      = B_vec[:, :, 2]                               # (N, Ns) — chỉ lấy Bz

# print(f"  Bz range        : [{Bz.min():.6f}, {Bz.max():.6f}] T")


# # ─── Save synthetic_data.csv ──────────────────────────────────────────────────
# print("\nSaving synthetic_data.csv...")
# col_names = [f"S{i+1}" for i in range(Ns)]
# volt_df   = pd.DataFrame(V_clean, columns=col_names)
# volt_path = BASE_DIR / args.out_volt
# volt_df.to_csv(volt_path, index=False)

# print(f"  Saved {volt_path}")
# print(f"\n─── Summary ─────────────────────────────────────────")
# print(f"  synthetic_grid.csv : {N:,} rows × 6 cols  (x,y,z,mx,my,mz)")
# print(f"  synthetic_data.csv : {N:,} rows × {Ns} cols (voltage per sensor)")
# print(f"  Total voltage vals : {N*Ns:,}")
# print(f"─────────────────────────────────────────────────────")

