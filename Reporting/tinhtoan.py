import numpy as np
import pandas as pd
import os
# --------------------------------------------------
# Load voltage data
# --------------------------------------------------

voltage_file = r"Data set 18.6\grid_data.csv"

V = pd.read_csv(voltage_file,header = 0).values.astype(np.float32)

# --------------------------------------------------
# Load calibration
# --------------------------------------------------

calib_file = r"Data set 18.6\Calibration_PARAM.csv"

calib = pd.read_csv(calib_file)
calib = calib.sort_values("sensor_index").reset_index(drop=True)

offset = calib["offset_a_V"].values.astype(np.float32)
gain   = calib["gain_g_V_per_T"].values.astype(np.float32)

# --------------------------------------------------
# Shape check
# --------------------------------------------------

print("Voltage shape:", V.shape)
print("Offset shape :", offset.shape)
print("Gain shape   :", gain.shape)

# --------------------------------------------------
# Recover Bz
# --------------------------------------------------

Bz = (V - offset) / gain

# --------------------------------------------------
# Global statistics
# --------------------------------------------------

print("\n===== GLOBAL Bz STATISTICS =====")
print(f"Bz range      : [{Bz.min():.6f}, {Bz.max():.6f}] T")
print(f"Bz mean(abs)  : {np.abs(Bz).mean():.6f} T")
print(f"Bz std        : {Bz.std():.6f} T")
print(f"|Bz| median   : {np.median(np.abs(Bz)):.6f} T")

# --------------------------------------------------
# Per-sensor statistics
# --------------------------------------------------

per_sensor_std = Bz.std(axis=0)

print("\n===== PER SENSOR =====")
print(f"std min       : {per_sensor_std.min():.6f} T")
print(f"std max       : {per_sensor_std.max():.6f} T")
print(f"std mean      : {per_sensor_std.mean():.6f} T")

# --------------------------------------------------
# Useful percentiles
# --------------------------------------------------

abs_B = np.abs(Bz)

print("\n===== |Bz| PERCENTILES =====")
print(f"50% : {np.percentile(abs_B,50):.6f} T")
print(f"90% : {np.percentile(abs_B,90):.6f} T")
print(f"95% : {np.percentile(abs_B,95):.6f} T")
print(f"99% : {np.percentile(abs_B,99):.6f} T")