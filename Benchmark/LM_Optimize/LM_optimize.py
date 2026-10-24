"""LM_optimize.py – 5-DoF Levenberg–Marquardt Magnetic Pose Estimator
=======================================================================
Benchmark implementation of the nonlinear least-squares inverse problem
for capsule endoscopy localization using a Hall sensor array and a
calibrated magnetic dipole model (Section 3.2 – Graduation Project).

Forward model (per sensor i):
    B_i   = (μ₀/4π) * [ 3(m·r̂)r̂ - m ] / |r_i|³
    V_pred,i = a_i + α(r_i) * g_i * (B_i · ẑ)

where α(r) = c0_i + c1_i * |r_i|  (Stage 2 per-sensor linear correction)

Inverse problem:
    min  Σ_i (V_meas,i - V_pred,i(X,Y,Z,α_tilt,β_tilt))²

solved with scipy.optimize.least_squares (Levenberg–Marquardt via 'lm' method).

Usage:
    python LM_optimize.py \\
        --test_voltage  ../../Dataset/Helix_data_2.csv \\
        --test_label    ../../Dataset/Helix_points_coordinates_2.csv \\
        --calib_params  ../../Calibration/Parameters/Linear_r_per_sensor/Calibration_Parameters_r_per_sensor.csv \\
        --calib_alpha   ../../Calibration/Parameters/Linear_r_per_sensor/Calibration_Alpha_r_per_sensor.csv \\
        --out           test_result_lm.png
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

_here      = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_here, "..", ".."))
sys.path.insert(0, _repo_root)

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULT_CALIB = os.path.join(
    _repo_root, "Calibration", "Parameters", "Linear_r_per_sensor")

parser = argparse.ArgumentParser(
    description="5-DoF Levenberg–Marquardt magnetic pose estimator (benchmark)")
parser.add_argument("--test_voltage", required=True,
                    help="CSV of measured Hall voltages (N × 64)")
parser.add_argument("--test_label",   required=True,
                    help="CSV of ground-truth poses  (N × 6: x,y,z,mx,my,mz)")
parser.add_argument("--calib_params",
                    default=os.path.join(_DEFAULT_CALIB,
                                         "Calibration_Parameters_r_per_sensor.csv"),
                    help="Stage-1 calibration CSV (sensor_index,x,y,z,offset,gain)")
parser.add_argument("--calib_alpha",
                    default=os.path.join(_DEFAULT_CALIB,
                                         "Calibration_Alpha_r_per_sensor.csv"),
                    help="Stage-2 alpha CSV (sensor_index,c0,c1,rmse)")
parser.add_argument("--out",  required=True, help="Output 3-D trajectory PNG")
parser.add_argument("--out-csv",  default=None,
                    help="Output CSV path (default: same dir as --out)")
parser.add_argument("--init", choices=("zero", "centroid", "random"),
                    default="centroid",
                    help="Initial-guess strategy for (X,Y,Z,α,β)")
parser.add_argument("--max-nfev", type=int, default=2000,
                    help="Max function evaluations per LM solve")
parser.add_argument("--ftol",  type=float, default=1e-8)
parser.add_argument("--xtol",  type=float, default=1e-8)
parser.add_argument("--gtol",  type=float, default=1e-8)
parser.add_argument("--timing-repeats", type=int, default=100,
                    help="Repeats for single-sample inference timing (benchmark)")
args = parser.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# Physical constants
# ─────────────────────────────────────────────────────────────────────────────
MU0_OVER_4PI = 1e-7         # μ₀/4π  [T·m/A]
SENSOR_DIR   = np.array([0.0, 0.0, 1.0])  # Hall sensors measure Z-component


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read_csv_auto(path: str) -> pd.DataFrame:
    """Read CSV, auto-detect header, coerce to numeric."""
    df = pd.read_csv(path, header=None)
    try:
        df.iloc[0].astype(float)
        has_header = False
    except (ValueError, TypeError):
        has_header = True
    if has_header:
        df = pd.read_csv(path, header=0)
    return df.apply(pd.to_numeric, errors="coerce")


def _read_paired(volt_path: str, label_path: str):
    vdf = _read_csv_auto(volt_path)
    ldf = _read_csv_auto(label_path)
    n   = min(len(vdf), len(ldf))
    vdf, ldf = vdf.iloc[:n], ldf.iloc[:n]
    mask = (vdf.notna().all(axis=1).to_numpy()
            & ldf.notna().all(axis=1).to_numpy())
    return (vdf[mask].reset_index(drop=True),
            ldf[mask].reset_index(drop=True))


def orientation_error_deg(m_pred: np.ndarray, m_gt: np.ndarray) -> np.ndarray:
    """Angular error in degrees between two sets of unit direction vectors."""
    m_pred_n = m_pred / (np.linalg.norm(m_pred, axis=1, keepdims=True) + 1e-14)
    m_gt_n   = m_gt   / (np.linalg.norm(m_gt,   axis=1, keepdims=True) + 1e-14)
    dot      = np.clip(np.sum(m_pred_n * m_gt_n, axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


# ─────────────────────────────────────────────────────────────────────────────
# Dipole forward model
# ─────────────────────────────────────────────────────────────────────────────

def dipole_field_vectorized(r_vec: np.ndarray, m_vec: np.ndarray) -> np.ndarray:
    """
    B at multiple positions from a single dipole moment m_vec.

    Parameters
    ----------
    r_vec : (N_sensors, 3)  displacement vectors (sensor_pos - magnet_pos)
    m_vec : (3,)            magnetic dipole moment [A·m²]

    Returns
    -------
    B : (N_sensors, 3)  magnetic flux density [T]
    """
    r      = np.linalg.norm(r_vec, axis=1, keepdims=True)          # (N,1)
    r3     = np.maximum(r ** 3, 1e-30)
    r5     = np.maximum(r ** 5, 1e-30)
    mdotr  = r_vec @ m_vec                                          # (N,)
    B = MU0_OVER_4PI * (
        3.0 * r_vec * mdotr[:, None] / r5 - m_vec[None, :] / r3
    )
    return B                                                        # (N, 3)


def pose_to_mvec(alpha: float, beta: float) -> np.ndarray:
    """
    Convert orientation angles (α, β) to unit magnetic moment direction.

    Convention (matches Graduation Project §3.2 / robot data):
        α = rotation about Y-axis  (elevation-like angle)
        β = rotation about Z-axis  (azimuth-like angle)

    m = Rz(β) · Ry(α) · [0, 0, 1]ᵀ
    """
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta),  np.sin(beta)
    mx = -sa * cb
    my = -sa * sb
    mz =  ca
    return np.array([mx, my, mz])


def mvec_to_pose_angles(m: np.ndarray):
    """
    Recover (α, β) from a unit direction vector m = [mx, my, mz].
    Inverse of pose_to_mvec.
    """
    m = m / (np.linalg.norm(m) + 1e-14)
    alpha = np.arccos(np.clip(m[2], -1.0, 1.0))        # [0, π]
    beta  = np.arctan2(m[1], m[0]) + np.pi             # shift for sin(-α·cb)
    # More robust: directly use atan2 on the projected components
    # m = [-sa*cb, -sa*sb, ca] → beta = atan2(-sa*sb, -sa*cb) = atan2(my, mx)
    beta = np.arctan2(m[1], m[0])
    return alpha, beta


# ─────────────────────────────────────────────────────────────────────────────
# Calibrated voltage prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict_voltages(
    pose5: np.ndarray,
    sensor_pos: np.ndarray,
    offset: np.ndarray,
    gain: np.ndarray,
    c0: np.ndarray,
    c1: np.ndarray,
    m_moment_mag: float = 1.0,
) -> np.ndarray:
    """
    Predict Hall sensor voltages for a given 5-DoF pose.

    Parameters
    ----------
    pose5         : [X, Y, Z, alpha, beta]  magnet pose
    sensor_pos    : (N_s, 3)  calibrated sensor positions [m]
    offset, gain  : (N_s,)    Stage-1 calibration params
    c0, c1        : (N_s,)    Stage-2 alpha coefficients
    m_moment_mag  : scalar    |magnetic moment| magnitude [A·m²]

    Returns
    -------
    V_pred : (N_s,)
    """
    X, Y, Z, alpha, beta = pose5
    m_dir = pose_to_mvec(alpha, beta)
    m_vec = m_moment_mag * m_dir

    r_vec = sensor_pos - np.array([X, Y, Z])           # (N_s, 3)
    B     = dipole_field_vectorized(r_vec, m_vec)       # (N_s, 3)
    B_z   = B @ SENSOR_DIR                              # (N_s,)

    r_dist = np.linalg.norm(r_vec, axis=1)              # (N_s,)
    alpha_r = c0 + c1 * r_dist                          # (N_s,)

    V_pred = offset + alpha_r * gain * B_z              # (N_s,)
    return V_pred


def residuals(
    pose5: np.ndarray,
    V_meas: np.ndarray,
    sensor_pos: np.ndarray,
    offset: np.ndarray,
    gain: np.ndarray,
    c0: np.ndarray,
    c1: np.ndarray,
    m_moment_mag: float,
) -> np.ndarray:
    """Residual vector for LM optimizer: V_meas - V_pred."""
    V_pred = predict_voltages(pose5, sensor_pos, offset, gain, c0, c1, m_moment_mag)
    return V_meas - V_pred


# ─────────────────────────────────────────────────────────────────────────────
# Initial-guess strategy
# ─────────────────────────────────────────────────────────────────────────────

def _make_init_guess(strategy: str, sensor_pos: np.ndarray,
                     gt_xyz: np.ndarray | None = None) -> np.ndarray:
    """Return [X0, Y0, Z0, alpha0, beta0]."""
    if strategy == "zero":
        return np.array([0.0, 0.0, 0.1, 0.0, 0.0])
    elif strategy == "centroid":
        cx, cy = sensor_pos[:, 0].mean(), sensor_pos[:, 1].mean()
        cz_above = sensor_pos[:, 2].mean() + 0.08   # ~8 cm above sensor plane
        return np.array([cx, cy, cz_above, 0.0, 0.0])
    elif strategy == "random":
        rng = np.random.default_rng()
        return np.array([
            rng.uniform(-0.06, 0.06),
            rng.uniform( 0.58, 0.78),
            rng.uniform(-0.04, 0.00),
            rng.uniform(-0.5,   0.5),
            rng.uniform(-0.5,   0.5),
        ])
    raise ValueError(f"Unknown init strategy: {strategy}")


# ─────────────────────────────────────────────────────────────────────────────
# Single-sample LM solve
# ─────────────────────────────────────────────────────────────────────────────

def solve_lm(
    V_meas: np.ndarray,
    sensor_pos: np.ndarray,
    offset: np.ndarray,
    gain: np.ndarray,
    c0: np.ndarray,
    c1: np.ndarray,
    m_moment_mag: float,
    x0: np.ndarray,
    max_nfev: int,
    ftol: float,
    xtol: float,
    gtol: float,
) -> tuple[np.ndarray, bool]:
    """
    Run LM optimization for one measurement vector.

    Returns
    -------
    pose5_opt : [X, Y, Z, alpha, beta]
    success   : bool
    """
    result = least_squares(
        residuals,
        x0,
        method="lm",
        args=(V_meas, sensor_pos, offset, gain, c0, c1, m_moment_mag),
        max_nfev=max_nfev,
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
    )
    return result.x, result.success


# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("  LM Optimization – 5-DoF Magnetic Pose Estimator")
print("=" * 65)
print(f"  Voltage : {args.test_voltage}")
print(f"  Label   : {args.test_label}")
print(f"  Params  : {args.calib_params}")
print(f"  Alpha   : {args.calib_alpha}\n")

volt_df, label_df = _read_paired(args.test_voltage, args.test_label)
assert volt_df.shape[1]  == 64, f"Expected 64 voltage columns, got {volt_df.shape[1]}"
assert label_df.shape[1] == 6,  f"Expected 6 label columns [x,y,z,mx,my,mz], got {label_df.shape[1]}"

V_all   = volt_df.values.astype(np.float64)    # (N, 64)
labels  = label_df.values.astype(np.float64)   # (N, 6)
N       = len(V_all)
print(f"  Test samples: {N}")

gt_xyz  = labels[:, :3]                        # (N, 3) [m]
gt_mvec = labels[:, 3:6]                       # (N, 3) unit direction
norms   = np.linalg.norm(gt_mvec, axis=1, keepdims=True)
gt_mvec = gt_mvec / (norms + 1e-14)

# ─────────────────────────────────────────────────────────────────────────────
# Load calibration parameters
# ─────────────────────────────────────────────────────────────────────────────

print(f"  Loading calibration from:\n"
      f"    {args.calib_params}\n    {args.calib_alpha}")

phys_df  = pd.read_csv(args.calib_params)
alpha_df = pd.read_csv(args.calib_alpha)

# Sort by sensor_index for safety
phys_df  = phys_df.sort_values("sensor_index").reset_index(drop=True)
alpha_df = alpha_df.sort_values("sensor_index").reset_index(drop=True)

N_SENSORS = len(phys_df)
assert N_SENSORS == volt_df.shape[1], (
    f"Calibration has {N_SENSORS} sensors but voltage data has {volt_df.shape[1]} columns.")

sensor_pos = phys_df[["x", "y", "z"]].values.astype(np.float64)   # (64, 3)
offset     = phys_df["offset"].values.astype(np.float64)           # (64,)
gain       = phys_df["gain"].values.astype(np.float64)             # (64,)
c0         = alpha_df["c0"].values.astype(np.float64)              # (64,)
c1         = alpha_df["c1"].values.astype(np.float64)              # (64,)

# Magnetic moment magnitude: set to 1.0 (absorbed into gain during calibration)
# The calibration was done with |m|=1, so gain already encodes the actual moment.
M_MAG = 1.0

print(f"  Loaded {N_SENSORS} sensors | offset range [{offset.min():.4f}, {offset.max():.4f}] V"
      f" | gain range [{gain.min():.3f}, {gain.max():.3f}] V/T\n")

# ─────────────────────────────────────────────────────────────────────────────
# Warm-up + single-sample inference timing  (benchmark-fair, same as test_mvec.py)
# ─────────────────────────────────────────────────────────────────────────────

print("Running inference ...")

x0_fixed = _make_init_guess(args.init, sensor_pos)

# Warm-up (không tính giờ)
_lm_kwargs = dict(
    sensor_pos=sensor_pos, offset=offset, gain=gain, c0=c0, c1=c1,
    m_moment_mag=M_MAG, x0=x0_fixed.copy(),
    max_nfev=args.max_nfev, ftol=args.ftol, xtol=args.xtol, gtol=args.gtol)
_ = solve_lm(V_all[0], **_lm_kwargs)

# --- Đo inference time: 100 lần lặp trên 1 sample đơn lẻ (benchmark-fair) ---
_TIMING_REPEATS = args.timing_repeats
_times_ms: list[float] = []
for _ in range(_TIMING_REPEATS):
    _t0 = time.perf_counter()
    _ = solve_lm(V_all[0], **_lm_kwargs)
    _times_ms.append((time.perf_counter() - _t0) * 1000.0)

infer_mean_ms = float(np.mean(_times_ms))
infer_std_ms  = float(np.std(_times_ms))
infer_fps     = 1000.0 / infer_mean_ms if infer_mean_ms > 0 else float("inf")

# --- Full-dataset solve (dùng cho metrics) ---
print(f"  Solving {N} poses with LM (init='{args.init}', max_nfev={args.max_nfev}) ...")
_t_full_start = time.perf_counter()

pred_poses   = np.zeros((N, 5))    # [X, Y, Z, alpha, beta]
n_converged  = 0
for i in range(N):
    pose_opt, ok = solve_lm(V_all[i], **_lm_kwargs)
    pred_poses[i] = pose_opt
    if ok:
        n_converged += 1

_t_full_end    = time.perf_counter()
total_infer_ms = (_t_full_end - _t_full_start) * 1000.0
per_sample_ms  = total_infer_ms / N if N > 0 else 0.0

print(f"  Converged: {n_converged}/{N} samples")

# Extract predicted XYZ and magnetic vector
pred_xyz  = pred_poses[:, :3]                               # (N, 3) [m]
pred_mvec = np.stack([                                      # (N, 3)
    pose_to_mvec(pred_poses[i, 3], pred_poses[i, 4])
    for i in range(N)
])

# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

pos_errors   = np.linalg.norm(pred_xyz - gt_xyz, axis=1)   # [m]
pos_err_mm   = pos_errors * 1000.0
mae_xyz      = np.abs(pred_xyz - gt_xyz).mean(axis=0)
mean_pos     = pos_errors.mean()
rmse_pos     = np.sqrt(np.mean(pos_errors ** 2))
max_pos      = pos_errors.max()

ori_err_deg  = orientation_error_deg(pred_mvec, gt_mvec)
mean_ori     = ori_err_deg.mean()
rmse_ori     = np.sqrt(np.mean(ori_err_deg ** 2))
max_ori      = ori_err_deg.max()

sample_idx        = np.arange(1, N + 1)
pos_mean_running  = np.cumsum(pos_err_mm)   / sample_idx
pos_rmse_running  = np.sqrt(np.cumsum(pos_err_mm  ** 2) / sample_idx)
ori_mean_running  = np.cumsum(ori_err_deg)  / sample_idx
ori_rmse_running  = np.sqrt(np.cumsum(ori_err_deg ** 2) / sample_idx)

# ─────────────────────────────────────────────────────────────────────────────
# Console output (format matches test_mvec.py for easy benchmark comparison)
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n  {'Pt':<5} {'PX':>8} {'PY':>8} {'PZ':>8} "
      f"{'GX':>8} {'GY':>8} {'GZ':>8} {'PosErr(mm)':>11} {'OriErr(°)':>10}")
print("  " + "-" * 95)
for i in range(N):
    print(f"  {i:<5} "
          f"{pred_xyz[i,0]:>8.4f} {pred_xyz[i,1]:>8.4f} {pred_xyz[i,2]:>8.4f} "
          f"{gt_xyz[i,0]:>8.4f} {gt_xyz[i,1]:>8.4f} {gt_xyz[i,2]:>8.4f} "
          f"{pos_err_mm[i]:>11.2f} {ori_err_deg[i]:>10.3f}")

print("\n─── LM Optimizer Test Results ───────────────────────────────────────")
print(f"  Số điểm test              : {N}")
print(f"  Mean position error       : {mean_pos * 1000:.2f} mm")
print(f"  RMSE position             : {rmse_pos * 1000:.2f} mm")
print(f"  Max  position error       : {max_pos  * 1000:.2f} mm")
print(f"  MAE x / y / z             : {mae_xyz[0]*1000:.2f} / "
      f"{mae_xyz[1]*1000:.2f} / {mae_xyz[2]*1000:.2f} mm")
print(f"  Mean orientation error    : {mean_ori:.3f}°")
print(f"  RMSE orientation          : {rmse_ori:.3f}°")
print(f"  Max  orientation error    : {max_ori:.3f}°")
print("─────────────────────────────────────────────────────────────────────")
print(f"  LM converged              : {n_converged}/{N}")
print(f"  Inference time (1 sample) : {infer_mean_ms:.3f} ± {infer_std_ms:.3f} ms  "
      f"(over {_TIMING_REPEATS} runs)")
print(f"  Estimated throughput      : {infer_fps:.1f} FPS")
print(f"  Full-set infer time       : {total_infer_ms:.1f} ms total  "
      f"≈ {per_sample_ms:.1f} ms/sample  ({N} samples)")
print("─────────────────────────────────────────────────────────────────────\n")

# ─────────────────────────────────────────────────────────────────────────────
# Save CSV  (identical format to test_mvec.py)
# ─────────────────────────────────────────────────────────────────────────────

out_dir  = os.path.dirname(os.path.abspath(args.out))
os.makedirs(out_dir, exist_ok=True)

csv_path = args.out_csv or os.path.join(out_dir, "testresult_lm.csv")
os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Point",
        "Pred_X", "Pred_Y", "Pred_Z",
        "GT_X",   "GT_Y",   "GT_Z",
        "PosErr_mm",
        "Pred_mx", "Pred_my", "Pred_mz",
        "GT_mx",   "GT_my",   "GT_mz",
        "OriErr_deg",
        "Converged",
    ])
    for i in range(N):
        writer.writerow([
            i,
            round(float(pred_xyz[i, 0]),  4), round(float(pred_xyz[i, 1]),  4),
            round(float(pred_xyz[i, 2]),  4),
            round(float(gt_xyz[i, 0]),    4), round(float(gt_xyz[i, 1]),    4),
            round(float(gt_xyz[i, 2]),    4),
            round(float(pos_err_mm[i]),   2),
            round(float(pred_mvec[i, 0]), 4), round(float(pred_mvec[i, 1]), 4),
            round(float(pred_mvec[i, 2]), 4),
            round(float(gt_mvec[i, 0]),   4), round(float(gt_mvec[i, 1]),   4),
            round(float(gt_mvec[i, 2]),   4),
            round(float(ori_err_deg[i]),  3),
            int(i < n_converged),          # approximate – use per-sample flag if needed
        ])
print(f"Saved CSV : {csv_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Figures (same layout as test_mvec.py)
# ─────────────────────────────────────────────────────────────────────────────

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

fig1 = plt.figure(figsize=(10, 7))
ax   = fig1.add_subplot(111, projection="3d")
ax.scatter(gt_xyz[:, 0],   gt_xyz[:, 1],   gt_xyz[:, 2],
           color="blue", s=35, label="Ground Truth", zorder=5)
ax.scatter(pred_xyz[:, 0], pred_xyz[:, 1], pred_xyz[:, 2],
           color="red", s=35, label="Predicted (LM)", zorder=5, marker="x")
for i in range(N):
    ax.plot([gt_xyz[i, 0], pred_xyz[i, 0]],
            [gt_xyz[i, 1], pred_xyz[i, 1]],
            [gt_xyz[i, 2], pred_xyz[i, 2]],
            color="gray", linewidth=0.5, alpha=0.5)
ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
ax.set_title(
    f"LM Optimizer – Test Set ({N} pts)\n"
    f"Position Error — Mean: {mean_pos*1000:.2f} mm  RMSE: {rmse_pos*1000:.2f} mm  "
    f"Max: {max_pos*1000:.2f} mm\n"
    f"Orientation Error — Mean: {mean_ori:.2f}°  RMSE: {rmse_ori:.2f}°  Max: {max_ori:.2f}°",
    fontsize=10)
ax.legend(fontsize=10); ax.grid(True); plt.tight_layout()
fig1.savefig(args.out, dpi=150, bbox_inches="tight")
print(f"Saved Figure 1 : {args.out}")

fig2, ax2 = plt.subplots(figsize=(11, 5))
ax2.plot(sample_idx, pos_err_mm,
         color="#5B9BD5", lw=0.8, alpha=0.55, label="Per-sample")
ax2.plot(sample_idx, pos_mean_running,
         color="#2E75B6", lw=2.0,
         label=f"Mean  ({pos_mean_running[-1]:.2f} mm)")
ax2.plot(sample_idx, pos_rmse_running,
         color="#C55A11", lw=2.0, ls="--",
         label=f"RMSE  ({pos_rmse_running[-1]:.2f} mm)")
ax2.axhline(mean_pos * 1000, color="#2E75B6", lw=0.8, ls=":")
ax2.axhline(rmse_pos * 1000, color="#C55A11", lw=0.8, ls=":")
ax2.set_xlabel("Sample index", fontsize=12)
ax2.set_ylabel("Position error (mm)", fontsize=12)
ax2.set_title(
    f"LM Optimizer – Position Error per Sample\n"
    f"Mean = {mean_pos*1000:.2f} mm  |  RMSE = {rmse_pos*1000:.2f} mm  |  "
    f"Max = {max_pos*1000:.2f} mm", fontsize=12)
ax2.legend(fontsize=10); ax2.grid(True, ls="--", alpha=0.5)
ax2.set_xlim(1, N); ax2.set_ylim(bottom=0); plt.tight_layout()
out_fig2 = os.path.join(out_dir, "lm_position_error.png")
fig2.savefig(out_fig2, dpi=150, bbox_inches="tight")
print(f"Saved Figure 2 : {out_fig2}")

fig3, ax3 = plt.subplots(figsize=(11, 5))
ax3.plot(sample_idx, ori_err_deg,
         color="#70AD47", lw=0.8, alpha=0.55, label="Per-sample")
ax3.plot(sample_idx, ori_mean_running,
         color="#375623", lw=2.0,
         label=f"Mean  ({ori_mean_running[-1]:.3f}°)")
ax3.plot(sample_idx, ori_rmse_running,
         color="#843C0C", lw=2.0, ls="--",
         label=f"RMSE  ({ori_rmse_running[-1]:.3f}°)")
ax3.axhline(mean_ori, color="#375623", lw=0.8, ls=":")
ax3.axhline(rmse_ori, color="#843C0C", lw=0.8, ls=":")
ax3.set_xlabel("Sample index", fontsize=12)
ax3.set_ylabel("Orientation error (°)", fontsize=12)
ax3.set_title(
    f"LM Optimizer – Orientation Error per Sample\n"
    f"Mean = {mean_ori:.3f}°  |  RMSE = {rmse_ori:.3f}°  |  Max = {max_ori:.3f}°",
    fontsize=12)
ax3.legend(fontsize=10); ax3.grid(True, ls="--", alpha=0.5)
ax3.set_xlim(1, N); ax3.set_ylim(bottom=0); plt.tight_layout()
out_fig3 = os.path.join(out_dir, "lm_orientation_error.png")
fig3.savefig(out_fig3, dpi=150, bbox_inches="tight")
print(f"Saved Figure 3 : {out_fig3}")

plt.show()
