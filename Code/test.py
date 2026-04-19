# import argparse, sys, os, pickle, json, platform, csv
# import numpy as np
# import pandas as pd
# import torch
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# # ─── Args ─────────────────────────────────────────────────────────────────────
# parser = argparse.ArgumentParser()

# parser.add_argument("--test_voltage", default="test_voltage.csv",
#                     help="File voltage của tập test riêng biệt")
# parser.add_argument("--test_label",   default="test_labels.csv",
#                     help="File label của tập test riêng biệt")
# parser.add_argument("--ckpt_dir", default="./ckpt2")
# parser.add_argument("--code_dir", default=".")
# parser.add_argument("--out",      default="test_result.png")
# args = parser.parse_args()

# # ─── Import model ─────────────────────────────────────────────────────────────
# sys.path.insert(0, args.code_dir)
# from model import Model  # noqa: E402

# # ─── Load data ────────────────────────────────────────────────────────────────
# def _read(path):
#     df = pd.read_csv(path, header=None)
#     try:
#         df.iloc[0].astype(float)
#         has_header = False
#     except (ValueError, TypeError):
#         has_header = True
#     if has_header:
#         df = pd.read_csv(path, header=0)
#     return df.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)

# print("Loading separate test data...")
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
# print(f"  Test samples  : {N}")

# # ─── Load scalers (đã fit trên train, dùng lại để transform test) ─────────────
# scaler_path = os.path.join(args.ckpt_dir, "scalers.pkl")
# print(f"\nLoading scalers from {scaler_path} ...")
# with open(scaler_path, "rb") as f:
#     scalers = pickle.load(f)
# volt_scaler  = scalers["volt"]
# label_scaler = scalers["label"]

# # Transform test bằng scaler của train — KHÔNG fit lại
# volt_test   = volt_scaler.transform(voltages)
# volt_tensor = torch.tensor(volt_test, dtype=torch.float32).view(-1, 1, 8, 8)

# gt_xyz = labels[:, :3]   # (N, 3) — giá trị gốc chưa scale
# gt_ang = labels[:, 3:]   # (N, 2) — cos(alpha), cos(beta)

# # ─── Build model ──────────────────────────────────────────────────────────────
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Device: {device}")
# model = Model(out_dim=5).to(device)

# if platform.system() != "Windows":
#     try:
#         model = torch.compile(model)
#         print("torch.compile enabled")
#     except Exception:
#         print("torch.compile not available - skipping")
# else:
#     print("torch.compile disabled (Windows)")

# # ─── Load checkpoint ──────────────────────────────────────────────────────────
# ckpt_path = os.path.join(args.ckpt_dir, "best.pt")
# print(f"Loading checkpoint from {ckpt_path} ...")
# ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

# raw_state   = ckpt["model"]
# is_compiled = hasattr(model, "_orig_mod")
# if is_compiled:
#     state = (raw_state if any(k.startswith("_orig_mod.") for k in raw_state)
#              else {"_orig_mod." + k: v for k, v in raw_state.items()})
# else:
#     state = {k.replace("_orig_mod.", ""): v for k, v in raw_state.items()}

# model.load_state_dict(state)
# model.eval()
# print(f"  Checkpoint epoch : {ckpt.get('epoch', '?')}")
# print(f"  Best val loss    : {ckpt.get('best_val', 0):.6f}")

# # ─── Inference ────────────────────────────────────────────────────────────────
# print("\nRunning inference...")
# with torch.no_grad():
#     pred_scaled = model(volt_tensor.to(device)).cpu().numpy()   # (N, 5)

# # Inverse transform → giá trị thực
# pred_full = label_scaler.inverse_transform(pred_scaled)
# pred_xyz  = pred_full[:, :3]
# pred_cos  = pred_full[:, 3:]

# # ─── Chuyển đổi góc: cos → rad → degree ──────────────────────────────────────
# pred_cos_clipped = np.clip(pred_cos, -1.0, 1.0)
# gt_cos_clipped   = np.clip(gt_ang,   -1.0, 1.0)

# pred_ang_rad = np.arccos(pred_cos_clipped)
# gt_ang_rad   = np.arccos(gt_cos_clipped)

# pred_ang_deg = np.degrees(pred_ang_rad)
# gt_ang_deg   = np.degrees(gt_ang_rad)

# # ─── Metrics ──────────────────────────────────────────────────────────────────
# pos_err_per_sample = np.linalg.norm(pred_xyz - gt_xyz, axis=1) * 1000  # mm
# ang_err_pitch      = np.abs(pred_ang_deg[:, 0] - gt_ang_deg[:, 0])
# ang_err_yaw       = np.abs(pred_ang_deg[:, 1] - gt_ang_deg[:, 1])
# ang_err_total      = ang_err_pitch  
# pos_errors = np.linalg.norm(pred_xyz - gt_xyz, axis=1)
# mae_xyz    = np.abs(pred_xyz - gt_xyz).mean(axis=0)
# rmse_pos   = np.sqrt(np.mean(pos_errors**2))
# mean_pos   = pos_errors.mean()
# max_pos    = pos_errors.max()

# rmse_pitch = np.sqrt(np.mean(ang_err_pitch**2))
# rmse_yaw   = np.sqrt(np.mean(ang_err_yaw**2))
# mean_pitch = ang_err_pitch.mean()
# mean_yaw  = ang_err_yaw.mean()
# max_pitch  = ang_err_pitch.max()
# max_yaw   = ang_err_yaw.max()

# # Running stats cho plot
# sample_idx = np.arange(1, N + 1)

# pos_mean_running = np.cumsum(pos_err_per_sample) / sample_idx
# pos_rmse_running = np.sqrt(np.cumsum(pos_err_per_sample**2) / sample_idx)

# pitch_mean_running = np.cumsum(ang_err_pitch) / sample_idx
# pitch_rmse_running = np.sqrt(np.cumsum(ang_err_pitch**2) / sample_idx)
# yaw_mean_running  = np.cumsum(ang_err_yaw)  / sample_idx
# yaw_rmse_running  = np.sqrt(np.cumsum(ang_err_yaw**2)  / sample_idx)

# # ─── Print table ──────────────────────────────────────────────────────────────
# print(f"\n  {'Pt':<5} {'PX':>8} {'PY':>8} {'PZ':>8} "
#       f"{'GX':>8} {'GY':>8} {'GZ':>8} {'Err(mm)':>9} "
#       f"{'P_pitch°':>8} {'P_yaw°':>8} {'G_pitch°':>8} {'G_yaw°':>8}")
# print("  " + "-" * 110)
# for i in range(N):
#     print(f"  {i:<5} "
#           f"{pred_xyz[i,0]:>8.4f} {pred_xyz[i,1]:>8.4f} {pred_xyz[i,2]:>8.4f} "
#           f"{gt_xyz[i,0]:>8.4f} {gt_xyz[i,1]:>8.4f} {gt_xyz[i,2]:>8.4f} "
#           f"{pos_err_per_sample[i]:>9.2f} "
#           f"{pred_ang_deg[i,0]:>8.3f} {pred_ang_deg[i,1]:>8.3f} "
#           f"{gt_ang_deg[i,0]:>8.3f} {gt_ang_deg[i,1]:>8.3f}")

# # ─── Print metrics ────────────────────────────────────────────────────────────
# print("\n─── Kết quả test set (riêng biệt, không leakage) ──────────────────")
# print(f"  Số điểm test         : {N}")
# print(f"  Mean Euclidean error : {mean_pos * 1000:.2f} mm")
# print(f"  RMSE position        : {rmse_pos * 1000:.2f} mm")
# print(f"  Max position error   : {max_pos  * 1000:.2f} mm")
# print(f"  MAE x                : {mae_xyz[0] * 1000:.2f} mm")
# print(f"  MAE y                : {mae_xyz[1] * 1000:.2f} mm")
# print(f"  MAE z                : {mae_xyz[2] * 1000:.2f} mm")
# print(f"  Mean pitch error     : {mean_pitch:.3f}°   RMSE: {rmse_pitch:.3f}°   Max: {max_pitch:.3f}°")
# print(f"  Mean yaw  error     : {mean_yaw:.3f}°   RMSE: {rmse_yaw:.3f}°   Max: {max_yaw:.3f}°")
# print("────────────────────────────────────────────────────────────────────\n")

# # ─── Xuất CSV ─────────────────────────────────────────────────────────────────
# csv_path = os.path.join(args.ckpt_dir, "testresult.csv")
# with open(csv_path, "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["Point",
#                      "Pred_X", "Pred_Y", "Pred_Z",
#                      "GT_X",   "GT_Y",   "GT_Z",   "PosErr_mm",
#                      "Pred_pitch_deg", "Pred_yaw_deg",
#                      "GT_pitch_deg",   "GT_yaw_deg",
#                      "Err_pitch_deg",  "Err_yaw_deg"])
#     for i in range(N):
#         writer.writerow([
#             i,
#             round(float(pred_xyz[i, 0]),    4),
#             round(float(pred_xyz[i, 1]),    4),
#             round(float(pred_xyz[i, 2]),    4),
#             round(float(gt_xyz[i, 0]),      4),
#             round(float(gt_xyz[i, 1]),      4),
#             round(float(gt_xyz[i, 2]),      4),
#             round(float(pos_err_per_sample[i]), 2),
#             round(float(pred_ang_deg[i, 0]), 4),
#             round(float(pred_ang_deg[i, 1]), 4),
#             round(float(gt_ang_deg[i, 0]),   4),
#             round(float(gt_ang_deg[i, 1]),   4),
#             round(float(ang_err_pitch[i]),   4),
#             round(float(ang_err_yaw[i]),    4),
#         ])
# print(f"Saved CSV: {csv_path}")

# # ─── Figure 1: 3D scatter GT vs Predicted ─────────────────────────────────────
# fig1 = plt.figure(figsize=(10, 7))
# ax   = fig1.add_subplot(111, projection="3d")

# ax.scatter(gt_xyz[:, 0],   gt_xyz[:, 1],   gt_xyz[:, 2],
#            color="blue", s=30, label="Ground Truth", zorder=5)
# ax.scatter(pred_xyz[:, 0], pred_xyz[:, 1], pred_xyz[:, 2],
#            color="red",  s=30, label="Predicted", zorder=5, marker="x")

# # for i in range(N):
# #     ax.plot([gt_xyz[i,0], pred_xyz[i,0]],
# #             [gt_xyz[i,1], pred_xyz[i,1]],
# #             [gt_xyz[i,2], pred_xyz[i,2]],
# #             color="gray", linewidth=0.5, alpha=0.5)

# ax.set_xlabel("X (m)")
# ax.set_ylabel("Y (m)")
# ax.set_zlabel("Z (m)")
# ax.set_title(
#     f"Test Set Inference ({N} points)\n"
#     f"Mean err: {mean_pos*1000:.2f} mm  |  "
#     f"RMSE: {rmse_pos*1000:.2f} mm  |  "
#     f"Max: {max_pos*1000:.2f} mm",
#     fontsize=11
# )
# ax.legend(fontsize=10)
# ax.grid(True)
# plt.tight_layout()

# out_fig1 = args.out
# fig1.savefig(out_fig1, dpi=150, bbox_inches="tight")
# print(f"Saved Figure 1: {out_fig1}")

# # ─── Figure 2: Position error per sample ──────────────────────────────────────
# fig2, ax2 = plt.subplots(figsize=(11, 5))

# ax2.plot(sample_idx, pos_err_per_sample,
#          color="#5B9BD5", linewidth=0.8, alpha=0.55, label="Per-sample error")
# ax2.plot(sample_idx, pos_mean_running,
#          color="#2E75B6", linewidth=2.0,
#          label=f"Mean error  (final {pos_mean_running[-1]:.2f} mm)")
# ax2.plot(sample_idx, pos_rmse_running,
#          color="#C55A11", linewidth=2.0, linestyle="--",
#          label=f"RMSE (final {pos_rmse_running[-1]:.2f} mm)")

# ax2.axhline(mean_pos * 1000, color="#2E75B6", linewidth=0.8, linestyle=":")
# ax2.axhline(rmse_pos * 1000, color="#C55A11", linewidth=0.8, linestyle=":")

# ax2.set_xlabel("Sample index", fontsize=12)
# ax2.set_ylabel("Position error (mm)", fontsize=12)
# ax2.set_title("Position Error per Sample", fontsize=13)
# ax2.legend(fontsize=10)
# ax2.grid(True, linestyle="--", alpha=0.5)
# ax2.set_xlim(1, N)
# ax2.set_ylim(bottom=0)
# plt.tight_layout()

# out_fig2 = os.path.join(os.path.dirname(args.out), "position_error.png")
# fig2.savefig(out_fig2, dpi=150, bbox_inches="tight")
# print(f"Saved Figure 2: {out_fig2}")

# # ─── Figure 3: Orientation error per sample ───────────────────────────────────
# fig3, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

# for ax_i, (err_arr, mean_run, rmse_run, name, mean_val, rmse_val) in zip(
#     axes,
#     [(ang_err_pitch, pitch_mean_running, pitch_rmse_running,
#       "Pitch", mean_pitch, rmse_pitch),
#      (ang_err_yaw,  yaw_mean_running,  yaw_rmse_running,
#       "Yaw",  mean_yaw,  rmse_yaw)]
# ):
#     ax_i.plot(sample_idx, err_arr,
#               color="#70AD47", linewidth=0.8, alpha=0.55, label="Per-sample error")
#     ax_i.plot(sample_idx, mean_run,
#               color="#375623", linewidth=2.0,
#               label=f"Mean error  (final {mean_run[-1]:.3f}°)")
#     ax_i.plot(sample_idx, rmse_run,
#               color="#843C0C", linewidth=2.0, linestyle="--",
#               label=f"RMSE (final {rmse_run[-1]:.3f}°)")

#     ax_i.axhline(mean_val, color="#375623", linewidth=0.8, linestyle=":")
#     ax_i.axhline(rmse_val, color="#843C0C", linewidth=0.8, linestyle=":")

#     ax_i.set_ylabel(f"{name} error (°)", fontsize=11)
#     ax_i.set_title(f"Orientation Error — {name}", fontsize=12)
#     ax_i.legend(fontsize=9)
#     ax_i.grid(True, linestyle="--", alpha=0.5)
#     ax_i.set_ylim(bottom=0)

# axes[-1].set_xlabel("Sample index", fontsize=12)
# axes[-1].set_xlim(1, N)
# fig3.suptitle("Orientation Error per Sample (degree)",
#               fontsize=13, y=1.01)
# plt.tight_layout()

# out_fig3 = os.path.join(os.path.dirname(args.out), "orientation_error.png")
# fig3.savefig(out_fig3, dpi=150, bbox_inches="tight")
# print(f"Saved Figure 3: {out_fig3}")

# plt.show()


import argparse, sys, os, pickle, json, platform, csv
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ─── Args ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--test_voltage",  default="grid_calib_data.csv")
parser.add_argument("--test_label",    default="Grid_points_coordinates.csv")
parser.add_argument("--ckpt_dir", default="./ckpt")
parser.add_argument("--code_dir", default=".")
parser.add_argument("--out",      default="test_result_grid.png")
args = parser.parse_args()

# ─── Import model ─────────────────────────────────────────────────────────────
sys.path.insert(0, args.code_dir)
from model import Model  # noqa: E402

# ─── Load data ────────────────────────────────────────────────────────────────
def _read(path):
    df = pd.read_csv(path, header=None)
    try:
        df.iloc[0].astype(float)
        has_header = False
    except (ValueError, TypeError):
        has_header = True
    if has_header:
        df = pd.read_csv(path, header=0)
    return df.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)

print("Loading data...")
volt_df  = _read(args.test_voltage)
label_df = _read(args.test_label)

voltages = volt_df.values.astype(np.float32)         # (N, 64)
labels   = label_df.values.astype(np.float32)        # (N, 5)
N        = min(len(voltages), len(labels))
voltages, labels = voltages[:N], labels[:N]
print(f"  Voltage : {voltages.shape}")
print(f"  Labels  : {labels.shape}")

# Load test_idx từ split_info.json
split_path = os.path.join(args.ckpt_dir, "split_info.json")
print(f"Loading split info from {split_path} ...")
with open(split_path) as f:
    split_info = json.load(f)
test_idx = np.array(split_info["test"])
print(f"  Test samples: {len(test_idx)}")

# ─── Load scalers ─────────────────────────────────────────────────────────────
scaler_path = os.path.join(args.ckpt_dir, "scalers.pkl")
print(f"Loading scalers from {scaler_path} ...")
with open(scaler_path, "rb") as f:
    scalers = pickle.load(f)
volt_scaler  = scalers["volt"]
label_scaler = scalers["label"]

volt_test   = volt_scaler.transform(voltages[test_idx])
volt_tensor = torch.tensor(volt_test, dtype=torch.float32).view(-1, 1, 8, 8)

gt_labels = labels[test_idx]          # (N_test, 5) — giá trị gốc chưa scale
gt_xyz    = gt_labels[:, :3]          # (N_test, 3) đơn vị m
gt_ang    = gt_labels[:, 3:]          # (N_test, 2) — col 3: alpha, col 4: beta
                                      # Giả sử label gốc lưu dưới dạng cos(angle)

# ─── Build model ──────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
model = Model(out_dim=5).to(device)

if platform.system() != "Windows":
    try:
        model = torch.compile(model)
        print("torch.compile enabled")
    except Exception:
        print("torch.compile not available - skipping")
else:
    print("torch.compile disabled (Windows)")

# ─── Load checkpoint ──────────────────────────────────────────────────────────
ckpt_path = os.path.join(args.ckpt_dir, "best.pt")
print(f"Loading checkpoint from {ckpt_path} ...")
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

raw_state   = ckpt["model"]
is_compiled = hasattr(model, "_orig_mod")
if is_compiled:
    state = (raw_state if any(k.startswith("_orig_mod.") for k in raw_state)
             else {"_orig_mod." + k: v for k, v in raw_state.items()})
else:
    state = {k.replace("_orig_mod.", ""): v for k, v in raw_state.items()}

model.load_state_dict(state)
model.eval()
print(f"  Checkpoint epoch : {ckpt.get('epoch', '?')}")
print(f"  Best val loss    : {ckpt.get('best_val', 0):.6f}")

# ─── Inference ────────────────────────────────────────────────────────────────
print("Running inference...")
with torch.no_grad():
    pred_scaled = model(volt_tensor.to(device)).cpu().numpy()   # (N_test, 5)

# Inverse transform toàn bộ 5 chiều
pred_full = label_scaler.inverse_transform(pred_scaled)         # (N_test, 5)
pred_xyz  = pred_full[:, :3]                                    # (N_test, 3) — m
pred_cos  = pred_full[:, 3:]                                    # (N_test, 2) — cos values

# ─── Chuyển đổi góc: cos → rad → degree ──────────────────────────────────────
# clip để tránh domain error khi cos hơi vượt [-1, 1] do floating point
pred_cos_clipped = np.clip(pred_cos, -1.0, 1.0)
gt_cos_clipped   = np.clip(gt_ang,   -1.0, 1.0)

pred_ang_rad = np.arccos(pred_cos_clipped)          # (N_test, 2) — radian
gt_ang_rad   = np.arccos(gt_cos_clipped)            # (N_test, 2) — radian

pred_ang_deg = np.degrees(pred_ang_rad)             # (N_test, 2) — degree
gt_ang_deg   = np.degrees(gt_ang_rad)               # (N_test, 2) — degree

# ─── Metrics per sample ───────────────────────────────────────────────────────
pos_err_per_sample  = np.linalg.norm(pred_xyz - gt_xyz, axis=1) * 1000      # mm
ang_err_per_sample  = np.abs(pred_ang_deg - gt_ang_deg)                      # deg, (N,2)
ang_err_total       = ang_err_per_sample.mean(axis=1)                        # mean of alpha+beta

# Cumulative RMSE per sample (running)
pos_rmse_per_sample = np.sqrt(np.cumsum(pos_err_per_sample**2)
                               / np.arange(1, len(pos_err_per_sample) + 1))
ang_rmse_per_sample = np.sqrt(np.cumsum(ang_err_total**2)
                               / np.arange(1, len(ang_err_total) + 1))

# Running mean per sample
pos_mean_running = np.cumsum(pos_err_per_sample) / np.arange(1, len(pos_err_per_sample) + 1)
ang_mean_running = np.cumsum(ang_err_total)      / np.arange(1, len(ang_err_total) + 1)

# Global metrics
pos_errors = np.linalg.norm(pred_xyz - gt_xyz, axis=1)
mae_xyz    = np.abs(pred_xyz - gt_xyz).mean(axis=0)
rmse_pos   = np.sqrt(np.mean(pos_errors**2))
mean_pos   = pos_errors.mean()
max_pos    = pos_errors.max()

ang_err_alpha = np.abs(pred_ang_deg[:, 0] - gt_ang_deg[:, 0])
ang_err_beta  = np.abs(pred_ang_deg[:, 1] - gt_ang_deg[:, 1])
rmse_alpha    = np.sqrt(np.mean(ang_err_alpha**2))
rmse_beta     = np.sqrt(np.mean(ang_err_beta**2))
mean_alpha    = ang_err_alpha.mean()
mean_beta     = ang_err_beta.mean()
max_alpha     = ang_err_alpha.max()
max_beta      = ang_err_beta.max()

# ─── Print table ──────────────────────────────────────────────────────────────
print(f"\n  {'Pt':<5} {'PX':>8} {'PY':>8} {'PZ':>8} "
      f"{'GX':>8} {'GY':>8} {'GZ':>8} {'Err(mm)':>9} "
      f"{'Pa°':>8} {'Pb°':>8} {'Ga°':>8} {'Gb°':>8}")
print("  " + "-" * 110)
for i in range(len(pred_xyz)):
    err = pos_err_per_sample[i]
    print(f"  {i:<5} "
          f"{pred_xyz[i,0]:>8.4f} {pred_xyz[i,1]:>8.4f} {pred_xyz[i,2]:>8.4f} "
          f"{gt_xyz[i,0]:>8.4f} {gt_xyz[i,1]:>8.4f} {gt_xyz[i,2]:>8.4f} "
          f"{err:>9.2f} "
          f"{pred_ang_deg[i,0]:>8.3f} {pred_ang_deg[i,1]:>8.3f} "
          f"{gt_ang_deg[i,0]:>8.3f} {gt_ang_deg[i,1]:>8.3f}")

# ─── Print metrics ────────────────────────────────────────────────────────────
print("\n─── Kết quả test set ───────────────────────────────────────────────")
print(f"  Số điểm test         : {len(test_idx)}")
print(f"  Mean Euclidean error : {mean_pos * 1000:.2f} mm")
print(f"  RMSE position        : {rmse_pos * 1000:.2f} mm")
print(f"  Max position error   : {max_pos  * 1000:.2f} mm")
print(f"  MAE x                : {mae_xyz[0] * 1000:.2f} mm")
print(f"  MAE y                : {mae_xyz[1] * 1000:.2f} mm")
print(f"  MAE z                : {mae_xyz[2] * 1000:.2f} mm")
print(f"  Mean alpha error     : {mean_alpha:.3f}°   RMSE: {rmse_alpha:.3f}°   Max: {max_alpha:.3f}°")
print(f"  Mean beta  error     : {mean_beta:.3f}°   RMSE: {rmse_beta:.3f}°   Max: {max_beta:.3f}°")
print("────────────────────────────────────────────────────────────────────\n")

# ─── Xuất CSV ─────────────────────────────────────────────────────────────────
csv_path = os.path.join(args.ckpt_dir, "testresult.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Point",
                     "Pred_X", "Pred_Y", "Pred_Z",
                     "GT_X",   "GT_Y",   "GT_Z",   "PosErr_mm",
                     "Pred_alpha_deg", "Pred_beta_deg",
                     "GT_alpha_deg",   "GT_beta_deg",
                     "Err_alpha_deg",  "Err_beta_deg"])
    for i in range(len(pred_xyz)):
        writer.writerow([
            i,
            round(float(pred_xyz[i, 0]),    4),
            round(float(pred_xyz[i, 1]),    4),
            round(float(pred_xyz[i, 2]),    4),
            round(float(gt_xyz[i, 0]),      4),
            round(float(gt_xyz[i, 1]),      4),
            round(float(gt_xyz[i, 2]),      4),
            round(float(pos_err_per_sample[i]), 2),
            round(float(pred_ang_deg[i, 0]), 4),
            round(float(pred_ang_deg[i, 1]), 4),
            round(float(gt_ang_deg[i, 0]),   4),
            round(float(gt_ang_deg[i, 1]),   4),
            round(float(ang_err_alpha[i]),   4),
            round(float(ang_err_beta[i]),    4),
        ])
print(f"Saved CSV: {csv_path}")

# ─── Figure 1: 3D scatter GT vs Predicted ─────────────────────────────────────
fig1 = plt.figure(figsize=(10, 7))
ax   = fig1.add_subplot(111, projection="3d")

ax.scatter(gt_xyz[:, 0],   gt_xyz[:, 1],   gt_xyz[:, 2],
           color="blue", s=30, label="Ground Truth", zorder=5)
ax.scatter(pred_xyz[:, 0], pred_xyz[:, 1], pred_xyz[:, 2],
           color="red",  s=30, label="Predicted", zorder=5, marker="x")

for i in range(len(pred_xyz)):
    ax.plot([gt_xyz[i,0], pred_xyz[i,0]],
            [gt_xyz[i,1], pred_xyz[i,1]],
            [gt_xyz[i,2], pred_xyz[i,2]],
            color="gray", linewidth=0.5, alpha=0.5)

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title(
    f"Test Set Inference ({len(test_idx)} points)\n"
    f"Mean err: {mean_pos*1000:.2f} mm  |  "
    f"RMSE: {rmse_pos*1000:.2f} mm  |  "
    f"Max: {max_pos*1000:.2f} mm",
    fontsize=11
)
ax.legend(fontsize=10)
ax.grid(True)
plt.tight_layout()

out_fig1 = args.out
fig1.savefig(out_fig1, dpi=150, bbox_inches="tight")
print(f"Saved Figure 1: {out_fig1}")

# ─── Figure 2: Position error per sample ──────────────────────────────────────
sample_idx = np.arange(1, len(pos_err_per_sample) + 1)

fig2, ax2 = plt.subplots(figsize=(11, 5))

ax2.plot(sample_idx, pos_err_per_sample,
         color="#5B9BD5", linewidth=0.8, alpha=0.55, label="Per-sample error")
ax2.plot(sample_idx, pos_mean_running,
         color="#2E75B6", linewidth=2.0, label=f"Mean Error  (final {pos_mean_running[-1]:.2f} mm)")
ax2.plot(sample_idx, pos_rmse_per_sample,
         color="#C55A11", linewidth=2.0, linestyle="--",
         label=f"RMSE  (final {pos_rmse_per_sample[-1]:.2f} mm)")

ax2.axhline(mean_pos * 1000, color="#2E75B6", linewidth=0.8, linestyle=":")
ax2.axhline(rmse_pos * 1000, color="#C55A11", linewidth=0.8, linestyle=":")

ax2.set_xlabel("Sample index", fontsize=12)
ax2.set_ylabel("Position error (mm)", fontsize=12)
ax2.set_title("Position Error per Sample", fontsize=13)
ax2.legend(fontsize=10)
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.set_xlim(1, len(sample_idx))
ax2.set_ylim(bottom=0)
plt.tight_layout()

out_fig2 = os.path.join(os.path.dirname(args.out), "position_error.png")
fig2.savefig(out_fig2, dpi=150, bbox_inches="tight")
print(f"Saved Figure 2: {out_fig2}")

# ─── Figure 3: Orientation error per sample ───────────────────────────────────
fig3, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

for ax_i, (err_arr, name, rmse_val, mean_val) in zip(
    axes,
    [(ang_err_alpha, "Pitch", rmse_alpha, mean_alpha),
     (ang_err_beta,  "Yaw",   rmse_beta,  mean_beta)]
):
    running_mean = np.cumsum(err_arr) / np.arange(1, len(err_arr) + 1)
    running_rmse = np.sqrt(np.cumsum(err_arr**2) / np.arange(1, len(err_arr) + 1))

    ax_i.plot(sample_idx, err_arr,
              color="#70AD47", linewidth=0.8, alpha=0.55, label="Per-sample error")
    ax_i.plot(sample_idx, running_mean,
              color="#375623", linewidth=2.0,
              label=f"Mean Error  (final {running_mean[-1]:.3f}°)")
    ax_i.plot(sample_idx, running_rmse,
              color="#843C0C", linewidth=2.0, linestyle="--",
              label=f"RMSE  (final {running_rmse[-1]:.3f}°)")

    ax_i.axhline(mean_val, color="#375623", linewidth=0.8, linestyle=":")
    ax_i.axhline(rmse_val, color="#843C0C", linewidth=0.8, linestyle=":")

    ax_i.set_ylabel(f"{name} error (°)", fontsize=11)
    ax_i.set_title(f"Orientation Error — {name}", fontsize=12)
    ax_i.legend(fontsize=9)
    ax_i.grid(True, linestyle="--", alpha=0.5)
    ax_i.set_ylim(bottom=0)

axes[-1].set_xlabel("Sample index", fontsize=12)
axes[-1].set_xlim(1, len(sample_idx))
fig3.suptitle("Orientation Error per Sample", fontsize=13, y=1.01)
plt.tight_layout()

out_fig3 = os.path.join(os.path.dirname(args.out), "orientation_error.png")
fig3.savefig(out_fig3, dpi=150, bbox_inches="tight")
print(f"Saved Figure 3: {out_fig3}")

plt.show()