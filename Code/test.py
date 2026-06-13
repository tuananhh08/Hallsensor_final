# =============================================================================
# split train/val
# =============================================================================

import argparse, sys, os, pickle, json, platform, csv
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

parser = argparse.ArgumentParser()
parser.add_argument("--test_voltage", default="test_voltage.csv")
parser.add_argument("--test_label",   default="test_labels.csv")
parser.add_argument("--ckpt_dir",     default="./ckpt2")
parser.add_argument("--code_dir",     default=".")
parser.add_argument("--out",          default="test_result.png")
args = parser.parse_args()

sys.path.insert(0, args.code_dir)
from model import Model  # noqa: E402


# =============================================================================
# Helpers
# =============================================================================

def _read(path):
    df = pd.read_csv(path, header=None)
    try:
        df.iloc[0].astype(float); has_header = False
    except (ValueError, TypeError):
        has_header = True
    if has_header:
        df = pd.read_csv(path, header=0)
    return df.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)


def cos_to_mvec(cos_pitch: np.ndarray, cos_yaw: np.ndarray) -> np.ndarray:

    cos_pitch = np.clip(cos_pitch, -1.0, 1.0)
    cos_yaw   = np.clip(cos_yaw,   -1.0, 1.0)
    sin_pitch = np.sqrt(np.maximum(0.0, 1.0 - cos_pitch**2))
    sin_yaw   = np.sqrt(np.maximum(0.0, 1.0 - cos_yaw**2))
    mx = cos_pitch * cos_yaw
    my = cos_pitch * sin_yaw
    mz = sin_pitch
    return np.stack([mx, my, mz], axis=1)   # (N, 3)


def orientation_error_deg(m_pred: np.ndarray, m_gt: np.ndarray) -> np.ndarray:

    # Normalize để tránh ảnh hưởng của magnitude
    m_pred_n = m_pred / (np.linalg.norm(m_pred, axis=1, keepdims=True) + 1e-12)
    m_gt_n   = m_gt   / (np.linalg.norm(m_gt,   axis=1, keepdims=True) + 1e-12)
    dot      = np.sum(m_pred_n * m_gt_n, axis=1)
    dot      = np.clip(dot, -1.0, 1.0)        # numerical safety
    return np.degrees(np.arccos(dot))          # (N,)


# =============================================================================
# Load data
# =============================================================================

print("=" * 65)
print("  Test Evaluation")
print("=" * 65)
print(f"  Voltage : {args.test_voltage}")
print(f"  Label   : {args.test_label}")
print(f"  Ckpt    : {args.ckpt_dir}\n")

volt_df  = _read(args.test_voltage)
label_df = _read(args.test_label)
voltages = volt_df.values.astype(np.float32)
labels   = label_df.values.astype(np.float32)
N        = min(len(voltages), len(labels))
voltages, labels = voltages[:N], labels[:N]
print(f"  Test samples: {N}")

scaler_path = os.path.join(args.ckpt_dir, "scalers.pkl")
print(f"  Loading scalers from {scaler_path} ...")
with open(scaler_path, "rb") as f:
    scalers = pickle.load(f)
volt_scaler  = scalers["volt"]
label_scaler = scalers["label"]

# Transform test bằng scaler của train 
volt_test   = volt_scaler.transform(voltages)
volt_tensor = torch.tensor(volt_test, dtype=torch.float32).view(-1, 1, 8, 8)

gt_xyz     = labels[:, :3]          # (N, 3)  [m]
gt_cos     = labels[:, 3:]          # (N, 2)  [cos_pitch, cos_yaw]

# Ground truth magnetic moment vector (N, 3)
gt_mvec = cos_to_mvec(gt_cos[:, 0], gt_cos[:, 1])

# =============================================================================
# Load model + checkpoint
# =============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device  : {device}")

model = Model(out_dim=5).to(device)

if platform.system() != "Windows":
    try:    model = torch.compile(model); print("  torch.compile enabled")
    except: print("  torch.compile not available - skipping")
else:
    print("  torch.compile disabled (Windows)")

ckpt_path = os.path.join(args.ckpt_dir, "best.pt")
print(f"  Loading checkpoint from {ckpt_path} ...")

ckpt      = torch.load(ckpt_path, map_location=device, weights_only=False)
raw_state = ckpt["model"]

# Luôn strip _orig_mod. prefix trước khi load vào raw_model.
# raw_model là Model thuần (chưa compile) nên keys không được có prefix.
# Checkpoint có thể được lưu từ compiled model (có prefix) hoặc không —
# cần normalize về dạng không prefix trong mọi trường hợp.
state = {k.replace("_orig_mod.", ""): v for k, v in raw_state.items()}

# Pop sensor_pos nếu còn sót trong checkpoint cũ
state.pop("sensor_pos", None)

raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
raw_model.load_state_dict(state, strict=True)
model.eval()

print(f"  Checkpoint epoch : {ckpt.get('epoch', '?')}")
print(f"  Best val loss    : {ckpt.get('best_val', 0):.6f}\n")

# =============================================================================
# Inference
# =============================================================================

print("Running inference ...")
with torch.no_grad():
    pred_scaled = model(volt_tensor.to(device)).float().cpu().numpy()  # (N, 5)

pred_full = label_scaler.inverse_transform(pred_scaled)
pred_xyz  = pred_full[:, :3]    # (N, 3)  [m]
pred_cos  = pred_full[:, 3:]    # (N, 2)  [cos_pitch, cos_yaw]

# Predicted magnetic moment vector (N, 3)
pred_mvec = cos_to_mvec(pred_cos[:, 0], pred_cos[:, 1])

# =============================================================================
# Metrics
# =============================================================================

# ── Position ─────────────────────────────────────────────────────────────────
pos_errors         = np.linalg.norm(pred_xyz - gt_xyz, axis=1)   # (N,) [m]
pos_err_mm         = pos_errors * 1000                            # (N,) [mm]
mae_xyz            = np.abs(pred_xyz - gt_xyz).mean(axis=0)       # (3,) [m]
mean_pos           = pos_errors.mean()
rmse_pos           = np.sqrt(np.mean(pos_errors**2))
max_pos            = pos_errors.max()

# ── Orientation — góc giữa 2 vector từ moment ────────────────────────────────
ori_err_deg        = orientation_error_deg(pred_mvec, gt_mvec)    # (N,) [°]
mean_ori           = ori_err_deg.mean()
rmse_ori           = np.sqrt(np.mean(ori_err_deg**2))
max_ori            = ori_err_deg.max()

# ── Running stats ─────────────────────────────────────────────────────────────
sample_idx         = np.arange(1, N + 1)
pos_mean_running   = np.cumsum(pos_err_mm)    / sample_idx
pos_rmse_running   = np.sqrt(np.cumsum(pos_err_mm**2)    / sample_idx)
ori_mean_running   = np.cumsum(ori_err_deg)   / sample_idx
ori_rmse_running   = np.sqrt(np.cumsum(ori_err_deg**2)   / sample_idx)

# =============================================================================
# Console output
# =============================================================================

print(f"\n  {'Pt':<5} {'PX':>8} {'PY':>8} {'PZ':>8} "
      f"{'GX':>8} {'GY':>8} {'GZ':>8} {'PosErr(mm)':>11} {'OriErr(°)':>10}")
print("  " + "-" * 95)
for i in range(N):
    print(f"  {i:<5} "
          f"{pred_xyz[i,0]:>8.4f} {pred_xyz[i,1]:>8.4f} {pred_xyz[i,2]:>8.4f} "
          f"{gt_xyz[i,0]:>8.4f} {gt_xyz[i,1]:>8.4f} {gt_xyz[i,2]:>8.4f} "
          f"{pos_err_mm[i]:>11.2f} {ori_err_deg[i]:>10.3f}")

print("\n─── Test Results ───────────────────────────────────────────────────")
print(f"  Số điểm test              : {N}")
print(f"  Mean position error       : {mean_pos * 1000:.2f} mm")
print(f"  RMSE position             : {rmse_pos * 1000:.2f} mm")
print(f"  Max  position error       : {max_pos  * 1000:.2f} mm")
print(f"  MAE x / y / z             : {mae_xyz[0]*1000:.2f} / "
      f"{mae_xyz[1]*1000:.2f} / {mae_xyz[2]*1000:.2f} mm")
print(f"  Mean orientation error    : {mean_ori:.3f}°")
print(f"  RMSE orientation          : {rmse_ori:.3f}°")
print(f"  Max  orientation error    : {max_ori:.3f}°")
print("────────────────────────────────────────────────────────────────────\n")

# =============================================================================
# Save CSV
# =============================================================================

csv_path = os.path.join(args.ckpt_dir, "testresult.csv")
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
    ])
    for i in range(N):
        writer.writerow([
            i,
            round(float(pred_xyz[i,0]), 4), round(float(pred_xyz[i,1]), 4),
            round(float(pred_xyz[i,2]), 4),
            round(float(gt_xyz[i,0]),   4), round(float(gt_xyz[i,1]),   4),
            round(float(gt_xyz[i,2]),   4),
            round(float(pos_err_mm[i]), 2),
            round(float(pred_mvec[i,0]), 4), round(float(pred_mvec[i,1]), 4),
            round(float(pred_mvec[i,2]), 4),
            round(float(gt_mvec[i,0]),   4), round(float(gt_mvec[i,1]),   4),
            round(float(gt_mvec[i,2]),   4),
            round(float(ori_err_deg[i]), 3),
        ])
print(f"Saved CSV : {csv_path}")

# =============================================================================
# Figures
# =============================================================================

out_dir = os.path.dirname(os.path.abspath(args.out))

# ── Figure 1: 3D scatter ──────────────────────────────────────────────────────
fig1 = plt.figure(figsize=(10, 7))
ax   = fig1.add_subplot(111, projection="3d")
ax.scatter(gt_xyz[:,0],   gt_xyz[:,1],   gt_xyz[:,2],
           color="blue", s=35, label="Ground Truth", zorder=5)
ax.scatter(pred_xyz[:,0], pred_xyz[:,1], pred_xyz[:,2],
           color="red", s=35, label="Predicted", zorder=5, marker="x")
for i in range(N):
    ax.plot([gt_xyz[i,0], pred_xyz[i,0]],
            [gt_xyz[i,1], pred_xyz[i,1]],
            [gt_xyz[i,2], pred_xyz[i,2]],
            color="gray", linewidth=0.5, alpha=0.5)
ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
ax.set_title(
    f"Test Set ({N} pts)\n"
    f"Position Error — Mean: {mean_pos*1000:.2f} mm  RMSE: {rmse_pos*1000:.2f} mm  Max: {max_pos*1000:.2f} mm\n"
    f"Orientation Error — Mean: {mean_ori:.2f}°  RMSE: {rmse_ori:.2f}°  Max: {max_ori:.2f}°",
    fontsize=10)
ax.legend(fontsize=10); ax.grid(True); plt.tight_layout()
fig1.savefig(args.out, dpi=150, bbox_inches="tight")
print(f"Saved Figure 1 : {args.out}")

# ── Figure 2: Position error per sample ──────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(11, 5))
ax2.plot(sample_idx, pos_err_mm,
         color="#5B9BD5", lw=0.8, alpha=0.55, label="Per-sample")
ax2.plot(sample_idx, pos_mean_running,
         color="#2E75B6", lw=2.0,
         label=f"Mean  ({pos_mean_running[-1]:.2f} mm)")
ax2.plot(sample_idx, pos_rmse_running,
         color="#C55A11", lw=2.0, ls="--",
         label=f"RMSE  ({pos_rmse_running[-1]:.2f} mm)")
ax2.axhline(mean_pos*1000, color="#2E75B6", lw=0.8, ls=":")
ax2.axhline(rmse_pos*1000, color="#C55A11", lw=0.8, ls=":")
ax2.set_xlabel("Sample index", fontsize=12)
ax2.set_ylabel("Position error (mm)", fontsize=12)
ax2.set_title(
    f"Position Error per Sample\n"
    f"Mean = {mean_pos*1000:.2f} mm  |  RMSE = {rmse_pos*1000:.2f} mm  |  Max = {max_pos*1000:.2f} mm",
    fontsize=12)
ax2.legend(fontsize=10); ax2.grid(True, ls="--", alpha=0.5)
ax2.set_xlim(1, N); ax2.set_ylim(bottom=0); plt.tight_layout()
out_fig2 = os.path.join(out_dir, "position_error.png")
fig2.savefig(out_fig2, dpi=150, bbox_inches="tight")
print(f"Saved Figure 2 : {out_fig2}")

# ── Figure 3: Orientation error per sample (single subplot) ──────────────────
#
# Dùng góc giữa 2 vector từ moment, nhất quán với các bài báo trong lĩnh vực:
#   err = arccos( dot(m_pred, m_gt) )   [degrees]
#
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
    f"Orientation Error per Sample\n"
    f"Mean Error= {mean_ori:.3f}°  |  RMSE = {rmse_ori:.3f}°  |  Max = {max_ori:.3f}°",
    fontsize=12)
ax3.legend(fontsize=10); ax3.grid(True, ls="--", alpha=0.5)
ax3.set_xlim(1, N); ax3.set_ylim(bottom=0); plt.tight_layout()
out_fig3 = os.path.join(out_dir, "orientation_error.png")
fig3.savefig(out_fig3, dpi=150, bbox_inches="tight")
print(f"Saved Figure 3 : {out_fig3}")

plt.show()

# =============================================================================
# train/val/test
# =============================================================================
# import argparse, sys, os, pickle, json, platform, csv
# import numpy as np
# import pandas as pd
# import torch
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# # ─── Args ─────────────────────────────────────────────────────────────────────
# parser = argparse.ArgumentParser()
# parser.add_argument("--test_voltage", default="grid_calib_data.csv")
# parser.add_argument("--test_label",   default="Grid_points_coordinates.csv")
# parser.add_argument("--ckpt_dir",     default="./ckpt")   # khớp với train/val/test
# parser.add_argument("--code_dir",     default=".")
# parser.add_argument("--out",          default="test_result_grid.png")
# args = parser.parse_args()

# # ─── Import model ─────────────────────────────────────────────────────────────
# sys.path.insert(0, args.code_dir)
# from model import Model  # noqa: E402

# # ─── Load data ────────────────────────────────────────────────────────────────
# def _read(path):
#     df = pd.read_csv(path, header=None)
#     try:
#         df.iloc[0].astype(float); has_header = False
#     except (ValueError, TypeError):
#         has_header = True
#     if has_header:
#         df = pd.read_csv(path, header=0)
#     return df.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)

# print("Loading data...")
# print(f"  Voltage : {args.test_voltage}")
# print(f"  Label   : {args.test_label}")
# volt_df  = _read(args.test_voltage)
# label_df = _read(args.test_label)

# voltages = volt_df.values.astype(np.float32)
# labels   = label_df.values.astype(np.float32)
# N        = min(len(voltages), len(labels))
# voltages, labels = voltages[:N], labels[:N]
# print(f"  Voltage shape : {voltages.shape}")
# print(f"  Label shape   : {labels.shape}")

# # Load test_idx từ split_info.json
# split_path = os.path.join(args.ckpt_dir, "split_info.json")
# print(f"Loading split info from {split_path} ...")
# with open(split_path) as f:
#     split_info = json.load(f)
# test_idx = np.array(split_info["test"])
# print(f"  Test samples: {len(test_idx)}")

# # ─── Load scalers ─────────────────────────────────────────────────────────────
# scaler_path = os.path.join(args.ckpt_dir, "scalers.pkl")
# print(f"Loading scalers from {scaler_path} ...")
# with open(scaler_path, "rb") as f:
#     scalers = pickle.load(f)
# volt_scaler  = scalers["volt"]
# label_scaler = scalers["label"]

# # Transform test bằng scaler của train — KHÔNG fit lại
# volt_test   = volt_scaler.transform(voltages[test_idx])
# volt_tensor = torch.tensor(volt_test, dtype=torch.float32).view(-1, 1, 8, 8)

# gt_labels = labels[test_idx]
# gt_xyz    = gt_labels[:, :3]   # (N_test, 3) — m
# gt_ang    = gt_labels[:, 3:]   # (N_test, 2) — cos(alpha), cos(beta)
# N_test    = len(test_idx)

# # ─── Build model ──────────────────────────────────────────────────────────────
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Device: {device}")
# model = Model(out_dim=5).to(device)

# if platform.system() != "Windows":
#     try:    model = torch.compile(model); print("torch.compile enabled")
#     except: print("torch.compile not available - skipping")
# else:
#     print("torch.compile disabled (Windows)")

# # ─── Load checkpoint ──────────────────────────────────────────────────────────
# ckpt_path = os.path.join(args.ckpt_dir, "best.pt")
# print(f"Loading checkpoint from {ckpt_path} ...")
# ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

# raw_state = ckpt["model"]
# state = {k.replace("_orig_mod.", ""): v for k, v in raw_state.items()}
# state.pop("sensor_pos", None)

# raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
# raw_model.load_state_dict(state, strict=True)
# model.eval()
# print(f"  Checkpoint epoch : {ckpt.get('epoch', '?')}")
# print(f"  Best val loss    : {ckpt.get('best_val', 0):.6f}")

# # ─── Inference ────────────────────────────────────────────────────────────────
# print("Running inference...")
# with torch.no_grad():
    
#     pred_scaled = model(volt_tensor.to(device)).float().cpu().numpy()   # (N_test, 5)

# # Inverse transform → giá trị thực
# pred_full = label_scaler.inverse_transform(pred_scaled)
# pred_xyz  = pred_full[:, :3]   # (N_test, 3) — m
# pred_cos  = pred_full[:, 3:]   # (N_test, 2) — cos values

# # ─── Chuyển đổi góc: cos → degree ────────────────────────────────────────────
# pred_cos_clipped = np.clip(pred_cos, -1.0, 1.0)
# gt_cos_clipped   = np.clip(gt_ang,   -1.0, 1.0)
# pred_ang_deg = np.degrees(np.arccos(pred_cos_clipped))   # (N_test, 2)
# gt_ang_deg   = np.degrees(np.arccos(gt_cos_clipped))     # (N_test, 2)

# # ─── Metrics ──────────────────────────────────────────────────────────────────
# pos_err_per_sample = np.linalg.norm(pred_xyz - gt_xyz, axis=1) * 1000   # mm
# ang_err_alpha      = np.abs(pred_ang_deg[:, 0] - gt_ang_deg[:, 0])      # deg
# ang_err_beta       = np.abs(pred_ang_deg[:, 1] - gt_ang_deg[:, 1])      # deg

# pos_errors = np.linalg.norm(pred_xyz - gt_xyz, axis=1)
# mae_xyz    = np.abs(pred_xyz - gt_xyz).mean(axis=0)
# rmse_pos   = np.sqrt(np.mean(pos_errors**2))
# mean_pos   = pos_errors.mean()
# max_pos    = pos_errors.max()

# rmse_alpha = np.sqrt(np.mean(ang_err_alpha**2))
# rmse_beta  = np.sqrt(np.mean(ang_err_beta**2))
# mean_alpha = ang_err_alpha.mean()
# mean_beta  = ang_err_beta.mean()
# max_alpha  = ang_err_alpha.max()
# max_beta   = ang_err_beta.max()

# sample_idx       = np.arange(1, N_test + 1)
# pos_mean_running = np.cumsum(pos_err_per_sample) / sample_idx
# pos_rmse_running = np.sqrt(np.cumsum(pos_err_per_sample**2) / sample_idx)

# # ─── Print table ──────────────────────────────────────────────────────────────
# print(f"\n  {'Pt':<5} {'PX':>8} {'PY':>8} {'PZ':>8} "
#       f"{'GX':>8} {'GY':>8} {'GZ':>8} {'Err(mm)':>9} "
#       f"{'Pa°':>8} {'Pb°':>8} {'Ga°':>8} {'Gb°':>8}")
# print("  " + "-" * 110)
# for i in range(N_test):
#     print(f"  {i:<5} "
#           f"{pred_xyz[i,0]:>8.4f} {pred_xyz[i,1]:>8.4f} {pred_xyz[i,2]:>8.4f} "
#           f"{gt_xyz[i,0]:>8.4f} {gt_xyz[i,1]:>8.4f} {gt_xyz[i,2]:>8.4f} "
#           f"{pos_err_per_sample[i]:>9.2f} "
#           f"{pred_ang_deg[i,0]:>8.3f} {pred_ang_deg[i,1]:>8.3f} "
#           f"{gt_ang_deg[i,0]:>8.3f} {gt_ang_deg[i,1]:>8.3f}")

# # ─── Print metrics ────────────────────────────────────────────────────────────
# print("\n─── Kết quả test set ───────────────────────────────────────────────")
# print(f"  Số điểm test         : {N_test}")
# print(f"  Mean Euclidean error : {mean_pos * 1000:.2f} mm")
# print(f"  RMSE position        : {rmse_pos * 1000:.2f} mm")
# print(f"  Max position error   : {max_pos  * 1000:.2f} mm")
# print(f"  MAE x                : {mae_xyz[0] * 1000:.2f} mm")
# print(f"  MAE y                : {mae_xyz[1] * 1000:.2f} mm")
# print(f"  MAE z                : {mae_xyz[2] * 1000:.2f} mm")
# print(f"  Mean alpha error     : {mean_alpha:.3f}°   RMSE: {rmse_alpha:.3f}°   Max: {max_alpha:.3f}°")
# print(f"  Mean beta  error     : {mean_beta:.3f}°   RMSE: {rmse_beta:.3f}°   Max: {max_beta:.3f}°")
# print("────────────────────────────────────────────────────────────────────\n")

# # ─── Xuất CSV ─────────────────────────────────────────────────────────────────
# csv_path = os.path.join(args.ckpt_dir, "testresult.csv")
# with open(csv_path, "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["Point",
#                      "Pred_X","Pred_Y","Pred_Z",
#                      "GT_X","GT_Y","GT_Z","PosErr_mm",
#                      "Pred_alpha_deg","Pred_beta_deg",
#                      "GT_alpha_deg","GT_beta_deg",
#                      "Err_alpha_deg","Err_beta_deg"])
#     for i in range(N_test):
#         writer.writerow([i,
#             round(float(pred_xyz[i,0]),4), round(float(pred_xyz[i,1]),4),
#             round(float(pred_xyz[i,2]),4), round(float(gt_xyz[i,0]),4),
#             round(float(gt_xyz[i,1]),4),   round(float(gt_xyz[i,2]),4),
#             round(float(pos_err_per_sample[i]),2),
#             round(float(pred_ang_deg[i,0]),4), round(float(pred_ang_deg[i,1]),4),
#             round(float(gt_ang_deg[i,0]),4),   round(float(gt_ang_deg[i,1]),4),
#             round(float(ang_err_alpha[i]),4),   round(float(ang_err_beta[i]),4)])
# print(f"Saved CSV: {csv_path}")

# # ─── Figure 1: 3D scatter GT vs Predicted ─────────────────────────────────────
# fig1 = plt.figure(figsize=(10, 7))
# ax   = fig1.add_subplot(111, projection="3d")
# ax.scatter(gt_xyz[:,0],   gt_xyz[:,1],   gt_xyz[:,2],
#            color="blue", s=30, label="Ground Truth", zorder=5)
# ax.scatter(pred_xyz[:,0], pred_xyz[:,1], pred_xyz[:,2],
#            color="red",  s=30, label="Predicted",    zorder=5, marker="x")
# for i in range(N_test):
#     ax.plot([gt_xyz[i,0], pred_xyz[i,0]],
#             [gt_xyz[i,1], pred_xyz[i,1]],
#             [gt_xyz[i,2], pred_xyz[i,2]],
#             color="gray", linewidth=0.5, alpha=0.5)
# ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
# ax.set_title(f"Test Set ({N_test} pts)\n"
#              f"Mean: {mean_pos*1000:.2f} mm  RMSE: {rmse_pos*1000:.2f} mm  "
#              f"Max: {max_pos*1000:.2f} mm", fontsize=11)
# ax.legend(fontsize=10); ax.grid(True); plt.tight_layout()
# fig1.savefig(args.out, dpi=150, bbox_inches="tight")
# print(f"Saved Figure 1: {args.out}")

# # ─── Figure 2: Position error per sample ──────────────────────────────────────
# fig2, ax2 = plt.subplots(figsize=(11, 5))
# ax2.plot(sample_idx, pos_err_per_sample, color="#5B9BD5", lw=0.8, alpha=0.55, label="Per-sample")
# ax2.plot(sample_idx, pos_mean_running,   color="#2E75B6", lw=2.0,
#          label=f"Mean (final {pos_mean_running[-1]:.2f} mm)")
# ax2.plot(sample_idx, pos_rmse_running,   color="#C55A11", lw=2.0, ls="--",
#          label=f"RMSE (final {pos_rmse_running[-1]:.2f} mm)")
# ax2.axhline(mean_pos*1000, color="#2E75B6", lw=0.8, ls=":")
# ax2.axhline(rmse_pos*1000, color="#C55A11", lw=0.8, ls=":")
# ax2.set_xlabel("Sample index", fontsize=12)
# ax2.set_ylabel("Position error (mm)", fontsize=12)
# ax2.set_title("Position Error per Sample", fontsize=13)
# ax2.legend(fontsize=10); ax2.grid(True, ls="--", alpha=0.5)
# ax2.set_xlim(1, N_test); ax2.set_ylim(bottom=0); plt.tight_layout()
# out_fig2 = os.path.join(os.path.dirname(args.out), "position_error.png")
# fig2.savefig(out_fig2, dpi=150, bbox_inches="tight")
# print(f"Saved Figure 2: {out_fig2}")

# # ─── Figure 3: Orientation error per sample ───────────────────────────────────
# fig3, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
# for ax_i, (err_arr, name, mean_val, rmse_val) in zip(
#     axes,
#     [(ang_err_alpha, "Alpha (pitch)", mean_alpha, rmse_alpha),
#      (ang_err_beta,  "Beta  (yaw)",   mean_beta,  rmse_beta)]
# ):
#     run_mean = np.cumsum(err_arr) / sample_idx
#     run_rmse = np.sqrt(np.cumsum(err_arr**2) / sample_idx)
#     ax_i.plot(sample_idx, err_arr,   color="#70AD47", lw=0.8, alpha=0.55, label="Per-sample")
#     ax_i.plot(sample_idx, run_mean,  color="#375623", lw=2.0,
#               label=f"Mean (final {run_mean[-1]:.3f}°)")
#     ax_i.plot(sample_idx, run_rmse,  color="#843C0C", lw=2.0, ls="--",
#               label=f"RMSE (final {run_rmse[-1]:.3f}°)")
#     ax_i.axhline(mean_val, color="#375623", lw=0.8, ls=":")
#     ax_i.axhline(rmse_val, color="#843C0C", lw=0.8, ls=":")
#     ax_i.set_ylabel(f"{name} error (°)", fontsize=11)
#     ax_i.set_title(f"Orientation Error — {name}", fontsize=12)
#     ax_i.legend(fontsize=9); ax_i.grid(True, ls="--", alpha=0.5); ax_i.set_ylim(bottom=0)
# axes[-1].set_xlabel("Sample index", fontsize=12)
# axes[-1].set_xlim(1, N_test)
# fig3.suptitle("Orientation Error per Sample", fontsize=13, y=1.01)
# plt.tight_layout()
# out_fig3 = os.path.join(os.path.dirname(args.out), "orientation_error.png")
# fig3.savefig(out_fig3, dpi=150, bbox_inches="tight")
# print(f"Saved Figure 3: {out_fig3}")

# plt.show()
