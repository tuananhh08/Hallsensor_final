# =============================================================================
# Test evaluation -- split train/valid
# =============================================================================

import argparse, sys, os, pickle, json, platform, csv
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

parser = argparse.ArgumentParser()
parser.add_argument("--test_voltage", default="../Data set 18.6/grid_data.csv")
parser.add_argument("--test_label",   default="../Data set 18.6/grid_points_coordinates.csv")
parser.add_argument("--ckpt_dir",     default="./ckpt_mvec")
parser.add_argument("--checkpoint",   default=None,
                    help="Checkpoint path; defaults to <ckpt_dir>/full_model_best.pt then best.pt.")
parser.add_argument("--use_calibnet", action=argparse.BooleanOptionalAction, default=None,
                    help="Override checkpoint architecture for baseline ablation.")
parser.add_argument("--code_dir",     default=".")
parser.add_argument("--out",          default="test_result.png")
args = parser.parse_args()

sys.path.insert(0, args.code_dir)
from model import Model  # noqa: E402

# =============================================================================
# Helpers
# =============================================================================

def _read(path):
    """Read a CSV, auto-detecting a header row, and coerce to numeric (NaNs kept as-is)."""
    df = pd.read_csv(path, header=None)
    try:
        df.iloc[0].astype(float); has_header = False
    except (ValueError, TypeError):
        has_header = True
    if has_header:
        df = pd.read_csv(path, header=0)
    return df.apply(pd.to_numeric, errors="coerce")


def _read_paired(volt_path, label_path):
    """Read voltage + label CSVs and keep only rows valid (finite) in BOTH files.

    Dropping NaNs independently in each file (the previous behaviour) can remove
    different row indices from each side; after truncating both to the same
    length N, row i of voltages would no longer correspond to row i of labels.
    A joint mask keeps the two files aligned, mirroring data_mvec.py's approach.
    """
    volt_df, label_df = _read(volt_path), _read(label_path)
    n = min(len(volt_df), len(label_df))
    volt_df, label_df = volt_df.iloc[:n], label_df.iloc[:n]
    mask = volt_df.notna().all(axis=1).to_numpy() & label_df.notna().all(axis=1).to_numpy()
    return volt_df[mask].reset_index(drop=True), label_df[mask].reset_index(drop=True)

# Hàm tính sai số góc
def orientation_error_deg(m_pred: np.ndarray, m_gt: np.ndarray) -> np.ndarray:
    m_pred_n = m_pred / (np.linalg.norm(m_pred, axis=1, keepdims=True) + 1e-14)
    m_gt_n   = m_gt   / (np.linalg.norm(m_gt,   axis=1, keepdims=True) + 1e-14)
    dot      = np.sum(m_pred_n * m_gt_n, axis=1)
    dot      = np.clip(dot, -1.0, 1.0)
    return np.degrees(np.arccos(dot))


# =============================================================================
# Load data
# =============================================================================

print("=" * 65)
print("  Test Evaluation (m_vec pipeline)")
print("=" * 65)
print(f"  Voltage : {args.test_voltage}")
print(f"  Label   : {args.test_label}")
print(f"  Ckpt    : {args.ckpt_dir}\n")

volt_df, label_df = _read_paired(args.test_voltage, args.test_label)

assert label_df.shape[1] == 6, f"Label must have 6 cols [x,y,z,mx,my,mz], got {label_df.shape[1]}"

voltages = volt_df.values.astype(np.float32)
labels   = label_df.values.astype(np.float32)
N        = len(voltages)
print(f"  Test samples: {N}")

scaler_path = os.path.join(args.ckpt_dir, "scalers.pkl")
print(f"  Loading scalers from {scaler_path} ...")
with open(scaler_path, "rb") as f:
    scalers = pickle.load(f)

assert scalers.get("label_format") == "mvec", (
    f"Expected label_format='mvec' in scalers.pkl, got {scalers.get('label_format')!r}. "
    "Use test_mvec.py with checkpoints from train_mvec.py."
)

volt_scaler  = scalers["volt"]
label_scaler = scalers["label"]

volt_test   = volt_scaler.transform(voltages)
volt_tensor = torch.tensor(volt_test, dtype=torch.float32).view(-1, 1, 8, 8)

gt_xyz  = labels[:, :3]
gt_mvec = labels[:, 3:6]
m_norms = np.linalg.norm(gt_mvec, axis=1, keepdims=True)
gt_mvec = gt_mvec / (m_norms + 1e-14)

# =============================================================================
# Load model + checkpoint
# =============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device  : {device}")

ckpt_path = args.checkpoint or os.path.join(args.ckpt_dir, "full_model_best.pt")
if not os.path.exists(ckpt_path):
    ckpt_path = os.path.join(args.ckpt_dir, "best.pt")
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
is_full = "model_state_dict" in ckpt or any(k.startswith("calibnet.") for k in ckpt.get("model", {}))
use_calibnet = is_full if args.use_calibnet is None else args.use_calibnet
model = Model(out_dim=6, use_calibnet=use_calibnet).to(device)

if platform.system() != "Windows":
    try:    model = torch.compile(model); print("torch.compile enabled")
    except: print("torch.compile not available - skipping")
else:
    print("torch.compile disabled (Windows)")

print(f"  Loading checkpoint from {ckpt_path} ...")
raw_state = ckpt.get("model_state_dict", ckpt.get("model", ckpt.get("locnet_state_dict")))
state = {k.replace("_orig_mod.", ""): v for k, v in raw_state.items()}
state.pop("sensor_pos", None)
if not any(k.startswith("locnet.") for k in state):
    # Old checkpoints stored only the localization backbone.
    state = {"locnet." + k: v for k, v in state.items()}

raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
missing, unexpected = raw_model.load_state_dict(state, strict=False)
if missing or unexpected:
    print(f"  Warning checkpoint mismatch: missing={list(missing)}, unexpected={list(unexpected)}")
model.eval()

print(f"  Checkpoint epoch : {ckpt.get('epoch', '?')}")
print(f"  Best val loss    : {ckpt.get('best_val', 0):.6f}\n")

# =============================================================================
# Inference
# =============================================================================

print("Running inference ...")
with torch.no_grad():
    pred_scaled = model(volt_tensor.to(device)).float().cpu().numpy()  # (N, 6)

pred_full = label_scaler.inverse_transform(pred_scaled)
pred_xyz  = pred_full[:, :3]
pred_mvec = pred_scaled[:, 3:]

# =============================================================================
# Metrics
# =============================================================================

pos_errors         = np.linalg.norm(pred_xyz - gt_xyz, axis=1)
pos_err_mm         = pos_errors * 1000
mae_xyz            = np.abs(pred_xyz - gt_xyz).mean(axis=0)
mean_pos           = pos_errors.mean()
rmse_pos           = np.sqrt(np.mean(pos_errors**2))
max_pos            = pos_errors.max()

ori_err_deg        = orientation_error_deg(pred_mvec, gt_mvec)
mean_ori           = ori_err_deg.mean()
rmse_ori           = np.sqrt(np.mean(ori_err_deg**2))
max_ori            = ori_err_deg.max()

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

