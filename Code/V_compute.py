# import argparse
# import numpy as np
# import pandas as pd
# from pathlib import Path

# # ─── Constants 
# MU_0_4PI    = 1e-7              
# VCC         = 3.3                
# V_Q         = VCC / 2            
# SENSITIVITY = 7.5e-3 / 1e-3     
# m0          = 1.0

# # ─── Args 
# parser = argparse.ArgumentParser()
# parser.add_argument("--coords",  default=".csv",
#                     help="File toa do capsule (x,y,z,cos_alpha,cos_beta)")
# parser.add_argument("--sensors", default="Hall_sensor_positions.csv",
#                     help="File vi tri 64 cam bien Hall")
# parser.add_argument("--out",     default="Grid_voltage_no_calib.csv",
#                     help="File output voltage")
# args = parser.parse_args()

# BASE_DIR = Path(__file__).parent

# # ─── Load data 
# print(f"Loading sensor positions from {args.sensors} ...")
# sensor_pos = pd.read_csv(BASE_DIR / args.sensors).values   # (64, 3)
# Ns = sensor_pos.shape[0]
# print(f"  Loaded {Ns} sensors")

# print(f"Loading coordinates from {args.coords} ...")
# coord_df  = pd.read_csv(BASE_DIR / args.coords)
# roi_xyz   = coord_df[["x", "y", "z"]].values              # (N, 3)
# cos_alpha = coord_df["cos_alpha"].values                   # (N,)
# cos_beta  = coord_df["cos_beta"].values                    # (N,)
# N = len(roi_xyz)
# print(f"  Loaded {N} poses")


# # Functions 
# def compute_m_vectors(cos_alpha, cos_beta):
#     """
#     Tinh vector moment tu truong m tu cos_alpha va cos_beta.
#     sin >= 0 vi alpha, beta thuoc [0, 180 deg].
#     Khi cos_alpha=1, cos_beta=1 -> m = [1, 0, 0].
#     """
#     sin_alpha = np.sqrt(np.clip(1 - cos_alpha**2, 0, 1))
#     sin_beta  = np.sqrt(np.clip(1 - cos_beta**2,  0, 1))

#     mx = m0 * cos_alpha * cos_beta
#     my = m0 * cos_alpha * sin_beta
#     mz = m0 * sin_alpha

#     return np.stack([mx, my, mz], axis=1)   # (N, 3)


# def compute_Bz(roi_xyz, sensor_pos, m_vecs):
#     """
#     Tinh thanh phan Bz (theo truc z) tai tung sensor tu cong thuc dipole.
#     Chi lay Bz vi cam bien Hall chi nhaycam voi thanh phan B doc truc cam bien.

#     """
#     r_vec  = sensor_pos[None, :, :] - roi_xyz[:, None, :]  # (N, Ns, 3)
#     r_norm = np.linalg.norm(r_vec, axis=2, keepdims=True)
#     r_norm = np.clip(r_norm, 1e-4, None)                    # tranh singularity < 0.1mm

#     m_dot_r = np.sum(m_vecs[:, None, :] * r_vec, axis=2, keepdims=True)

#     term1 = 3 * m_dot_r * r_vec / (r_norm ** 5)
#     term2 = m_vecs[:, None, :] / (r_norm ** 3)

#     B_vec = MU_0_4PI * (term1 - term2)   
#     Bz    = B_vec[:, :, 2]               

#     return Bz


# def Bz_to_voltage(Bz):
#     """
#     V = V_Q + SENSITIVITY * Bz
#     Sensitivity chung cho tat ca sensor: 7.5 V/T
#     """
#     return V_Q + SENSITIVITY * Bz


# # ─── Main 
# print("\nComputing magnetic moment vectors ...")
# m_vecs = compute_m_vectors(cos_alpha, cos_beta)

# # Kiem tra m vector voi mau dau tien
# print(f"  m[0] = [{m_vecs[0,0]:.4f}, {m_vecs[0,1]:.4f}, {m_vecs[0,2]:.4f}]"
#       f"  (ky vong [1,0,0] khi cos_alpha=cos_beta=1)")

# print("Computing Bz at all sensors ...")
# Bz = compute_Bz(roi_xyz, sensor_pos, m_vecs)   # (N, 64)
# print(f"  Bz range: [{Bz.min():.6f}, {Bz.max():.6f}] T")

# print("Converting Bz to voltage ...")
# V_all = Bz_to_voltage(Bz)                       # (N, 64)

# # Kiem tra khoang cach min
# r_min = np.linalg.norm(
#     roi_xyz[:, None, :] - sensor_pos[None, :, :], axis=2).min()
# if r_min < 0.005:
#     print(f"  [WARNING] Khoang cach min toi sensor = {r_min*100:.2f} cm < 0.5cm")

# # Clip ve [0, VCC] — sensor vat ly bi bao hoa
# n_clipped = np.sum((V_all < 0) | (V_all > VCC))
# V_all     = np.clip(V_all, 0, VCC)

# print(f"\n─── Ket qua ─────────────────────────────────")
# print(f"  Samples  : {N:,}")
# print(f"  V range  : [{V_all.min():.4f}, {V_all.max():.4f}] V")
# print(f"  V_Q      : {V_Q} V  |  Sensitivity: {SENSITIVITY} V/T")
# print(f"  Clipped  : {n_clipped:,} / {V_all.size:,} "
#       f"({100*n_clipped/V_all.size:.2f}%)")
# print(f"─────────────────────────────────────────────")

# # Save — khong header, khong index
# out_path = BASE_DIR / args.out
# pd.DataFrame(V_all).to_csv(out_path, header=False, index=False)
# print(f"\nSaved -> {out_path}")



# import numpy as np
# import pandas as pd
# from pathlib import Path

# # ─── Constants ────────────────────────────────────────────────────────────────
# MU_0_4PI = 1e-7
# m0       = 1.0

# BASE_DIR = Path(r"D:\Downloads\Hallsensor_final\Data set\Coordinate")

# # ─── Load data ────────────────────────────────────────────────────────────────
# print("Loading data...")

# # Sensor positions
# sensor_pos = pd.read_csv(BASE_DIR / "Hall_sensor_positions.csv").values  # (64, 3)
# Ns = sensor_pos.shape[0]
# print(f"  Sensors        : {Ns}")

# # Calib params — moi sensor 1 bo rieng
# calib_df = pd.read_csv(BASE_DIR / "Calibration_GRID_Offset_Sens.csv")
# calib_df = calib_df.sort_values("sensor_index").reset_index(drop=True)
# assert len(calib_df) == Ns, \
#     f"Calib co {len(calib_df)} sensors, sensor_pos co {Ns} sensors"

# VQ_arr   = calib_df["offset_a_V"].to_numpy()        # (64,)
# SENS_arr = calib_df["gain_g_V_per_T"].to_numpy()    # (64,)

# print(f"  V_Q   range    : [{VQ_arr.min():.4f}, {VQ_arr.max():.4f}] V")
# print(f"  SENS  range    : [{SENS_arr.min():.4f}, {SENS_arr.max():.4f}] V/T")

# # Grid coordinates
# coord_df  = pd.read_csv(DATASET_DIR / "grid_points_random_150.csv")
# roi_xyz   = coord_df[["x", "y", "z"]].values        # (N, 3)
# cos_alpha = coord_df["cos_alpha"].values             # (N,)
# cos_beta  = coord_df["cos_beta"].values              # (N,)
# N = len(roi_xyz)
# print(f"  Poses          : {N}")

# # ─── Functions ────────────────────────────────────────────────────────────────
# def compute_m_vectors(cos_alpha, cos_beta):
#     """m = [1,0,0] khi cos_alpha=cos_beta=1"""
#     sin_alpha = np.sqrt(np.clip(1 - cos_alpha**2, 0, 1))
#     sin_beta  = np.sqrt(np.clip(1 - cos_beta**2,  0, 1))
#     mx = m0 * cos_alpha * cos_beta
#     my = m0 * cos_alpha * sin_beta
#     mz = m0 * sin_alpha
#     return np.stack([mx, my, mz], axis=1)            # (N, 3)


# def compute_Bz(roi_xyz, sensor_pos, m_vecs):
#     """Chi lay thanh phan Bz — cam bien Hall nhaycam voi truc z"""
#     r_vec   = sensor_pos[None, :, :] - roi_xyz[:, None, :]  # (N, Ns, 3)
#     r_norm  = np.linalg.norm(r_vec, axis=2, keepdims=True)
#     r_norm  = np.clip(r_norm, 1e-4, None)
#     m_dot_r = np.sum(m_vecs[:, None, :] * r_vec, axis=2, keepdims=True)
#     term1   = 3 * m_dot_r * r_vec / (r_norm ** 5)
#     term2   = m_vecs[:, None, :] / (r_norm ** 3)
#     B_vec   = MU_0_4PI * (term1 - term2)             # (N, Ns, 3)
#     return B_vec[:, :, 2]                             # (N, Ns)


# def Bz_to_voltage_calib(Bz, VQ_arr, SENS_arr):
    
#     return VQ_arr[None, :] + SENS_arr[None, :] * Bz  # (N, Ns)


# # ─── Main ─────────────────────────────────────────────────────────────────────
# print("\nComputing m vectors...")
# m_vecs = compute_m_vectors(cos_alpha, cos_beta)
# print(f"  m[0] = [{m_vecs[0,0]:.3f}, {m_vecs[0,1]:.3f}, {m_vecs[0,2]:.3f}]"
#       f"  (ky vong [1,0,0] khi cos_alpha=cos_beta=1)")

# print("Computing Bz...")
# Bz = compute_Bz(roi_xyz, sensor_pos, m_vecs)         # (N, 64)
# print(f"  Bz range       : [{Bz.min():.6f}, {Bz.max():.6f}] T")

# # Kiem tra khoang cach min
# r_min = np.linalg.norm(
#     roi_xyz[:, None, :] - sensor_pos[None, :, :], axis=2).min()
# if r_min < 0.005:
#     print(f"  [WARNING] Khoang cach min = {r_min*100:.2f} cm < 0.5cm")

# print("Converting Bz to voltage using calib params...")
# V_all = Bz_to_voltage_calib(Bz, VQ_arr, SENS_arr)   # (N, 64)

# # Clip ve [0, 3.3V]
# VCC       = 3.3
# n_clipped = np.sum((V_all < 0) | (V_all > VCC))
# V_all     = np.clip(V_all, 0, VCC)

# print(f"\n─── Ket qua ─────────────────────────────────────────")
# print(f"  Samples        : {N:,}")
# print(f"  V range        : [{V_all.min():.4f}, {V_all.max():.4f}] V")
# print(f"  Clipped        : {n_clipped:,} / {V_all.size:,} "
#       f"({100*n_clipped/V_all.size:.2f}%)")
# print(f"─────────────────────────────────────────────────────")

# # Luu — khong header, khong index
# out_path = DATASET_DIR / "V_grid_calib.csv"
# pd.DataFrame(V_all).to_csv(out_path, header=False, index=False)
# print(f"\nSaved -> {out_path}  ({N} rows x {Ns} cols)")




import numpy as np
import pandas as pd
from pathlib import Path

# ─── Constants ────────────────────────────────────────────────────────────────
MU_0_4PI = 1e-7
m0       = 1.0
VCC      = 3.3
V_Q      = VCC / 2          # 1.65 V
SENS     = 7.5              # V/T  (7.5 mV/mT)

BASE_DIR = Path(r"D:\Downloads\Hallsensor_final\Data set\Coordinate")

# ─── Load data ────────────────────────────────────────────────────────────────
print("Loading data...")

sensor_pos = pd.read_csv(BASE_DIR / "Hall_sensor_positions.csv").values  # (64, 3)
Ns = sensor_pos.shape[0]
print(f"  Sensors        : {Ns}")

coord_df  = pd.read_csv(BASE_DIR / "ROI_data.csv")
roi_xyz   = coord_df[["x", "y", "z"]].values          # (N, 3)
cos_pitch = coord_df["cos_pitch"].values               # (N,)
cos_yaw   = coord_df["cos_yaw"].values                 # (N,)
N = len(roi_xyz)
print(f"  Poses          : {N:,}")

# ─── Functions ────────────────────────────────────────────────────────────────
def compute_m_vectors(cos_pitch, cos_yaw):
    """
    Tính vector moment từ m từ cos_pitch và cos_yaw.
    Quy ước:
      alpha = pitch (quay quanh Y)  →  cos_alpha = cos_pitch
      beta  = yaw   (quay quanh Z)  →  cos_beta  = cos_yaw
    sin lấy dương (góc trong [20,160] → sin >= 0)

      mx = cos_pitch * cos_yaw
      my = cos_pitch * sin_yaw      (sin_yaw  = sqrt(1 - cos_yaw^2))
      mz = sin_pitch                (sin_pitch = sqrt(1 - cos_pitch^2))
    """
    sin_pitch = np.sqrt(np.clip(1.0 - cos_pitch**2, 0, None))
    sin_yaw   = np.sqrt(np.clip(1.0 - cos_yaw**2,   0, None))

    mx = cos_pitch * cos_yaw   # (N,)
    my = cos_pitch * sin_yaw   # (N,)
    mz = sin_pitch             # (N,)

    return np.stack([mx, my, mz], axis=1) * m0   # (N, 3)


def compute_Bz(roi_xyz, sensor_pos, m_vecs):
    """
    Tính thành phần Bz tại mỗi sensor theo mô hình dipole.
    Returns: (N, Ns)
    """
    # r_vec[i,j] = s_j - p_i  →  (N, Ns, 3)
    r_vec  = sensor_pos[None, :, :] - roi_xyz[:, None, :]
    r_norm = np.linalg.norm(r_vec, axis=2, keepdims=True)   # (N, Ns, 1)
    r_norm = np.clip(r_norm, 1e-4, None)                    # tránh chia 0

    m_dot_r = np.sum(m_vecs[:, None, :] * r_vec, axis=2, keepdims=True)  # (N, Ns, 1)

    term1  = 3.0 * m_dot_r * r_vec / (r_norm ** 5)         # (N, Ns, 3)
    term2  = m_vecs[:, None, :] / (r_norm ** 3)            # (N, Ns, 3)
    B_vec  = MU_0_4PI * (term1 - term2)                    # (N, Ns, 3)

    return B_vec[:, :, 2]                                   # (N, Ns) — chỉ Bz


def add_snr_noise(V, snr_min_db=25.0, snr_max_db=50.0, snr_res_db=0.4):
    """
    Thêm nhiễu Gaussian với SNR random trong [snr_min_db, snr_max_db].
    Resolution SNR = snr_res_db (bước lượng tử hóa SNR).

    SNR (dB) = 10 * log10(P_signal / P_noise)
    → sigma_noise = rms(V) / 10^(SNR_dB / 20)
    """
    # Lượng tử hóa SNR theo resolution 0.4 dB
    snr_levels = np.arange(snr_min_db, snr_max_db + snr_res_db, snr_res_db)

    # Random 1 giá trị SNR cho mỗi sample (mỗi hàng)
    chosen_snr_db = np.random.choice(snr_levels, size=V.shape[0])  # (N,)

    # Tính sigma noise theo từng sample
    rms_signal  = np.sqrt(np.mean(V**2, axis=1))                   # (N,)
    sigma_noise = rms_signal / (10.0 ** (chosen_snr_db / 20.0))    # (N,)

    # Sinh nhiễu Gaussian N(0, sigma) cho từng sample
    noise = np.random.randn(*V.shape) * sigma_noise[:, None]       # (N, Ns)

    return V + noise


# ─── Main ─────────────────────────────────────────────────────────────────────
print("\nComputing moment vectors...")
m_vecs = compute_m_vectors(cos_pitch, cos_yaw)
print(f"  m[0] = [{m_vecs[0,0]:.4f}, {m_vecs[0,1]:.4f}, {m_vecs[0,2]:.4f}]")

print("Computing Bz field...")
Bz = compute_Bz(roi_xyz, sensor_pos, m_vecs)    # (N, 64)
print(f"  Bz range       : [{Bz.min():.6f}, {Bz.max():.6f}] T")

# Kiểm tra khoảng cách min
r_min = np.linalg.norm(
    roi_xyz[:, None, :] - sensor_pos[None, :, :], axis=2).min()
print(f"  Min distance   : {r_min*100:.2f} cm", end="")
if r_min < 0.005:
    print("  [WARNING] < 0.5 cm — có thể bão hòa tín hiệu")
else:
    print()

print("Converting Bz to voltage...")
V_clean = V_Q + SENS * Bz                       # (N, 64)
print(f"  Voltage range  : [{V_clean.min():.4f}, {V_clean.max():.4f}] V")

print("Adding SNR noise [25, 50] dB (step 0.4 dB)...")
V_noisy = add_snr_noise(V_clean,
                        snr_min_db=25.0,
                        snr_max_db=50.0,
                        snr_res_db=0.4)
print(f"  Noisy V range  : [{V_noisy.min():.4f}, {V_noisy.max():.4f}] V")

# ─── Export theo chunk ────────────────────────────────────────────────────────
output_path = BASE_DIR / "grid_generated_data.csv"
CHUNK_SIZE  = 50_000

print(f"\nWriting {N:,} rows to {output_path} ...")

for start in range(0, N, CHUNK_SIZE):
    end     = min(start + CHUNK_SIZE, N)
    chunk   = V_noisy[start:end]                # (chunk, 64)

    pd.DataFrame(chunk).to_csv(
        output_path,
        mode   = 'w' if start == 0 else 'a',
        header = False,
        index  = False,
    )
    print(f"  Written {end:>10,} / {N:,} rows", end='\r')

print(f"\nDone! Saved {N:,} rows x {Ns} cols → {output_path}")