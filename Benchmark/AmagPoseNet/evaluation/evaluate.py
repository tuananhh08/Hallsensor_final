"""AMagPoseNet – evaluation/evaluate.py

Evaluation script with identical output format to Code/test_mvec.py.
Loads a trained AMagPoseNet checkpoint and runs inference on a test CSV.

Usage:
    python -m Benchmark.AmagPoseNet.evaluation.evaluate \
        --test_voltage  ../../"Dataset"/Helix_data_2.csv \
        --test_label    ../../"Dataset"/Helix_points_coordinates_2.csv \
        --ckpt_dir      ../../training_ckpt/amagposenet \
        --checkpoint    full_model_best.pt \
        --out           test_result_amagposenet.png
"""
from __future__ import annotations

import argparse
import csv
import os
import pickle
import platform
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Allow running from repo root or Benchmark/AmagPoseNet/evaluation/
_here = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_here, "..", "..", ".."))
sys.path.insert(0, _repo_root)
sys.path.insert(0, os.path.join(_here, ".."))

from models.amagposenet import AMagPoseNet
from evaluation.metrics import compute_metrics


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Evaluate AMagPoseNet benchmark")
parser.add_argument("--test_voltage",  required=True)
parser.add_argument("--test_label",    required=True)
parser.add_argument("--ckpt_dir",      required=True)
parser.add_argument("--checkpoint",    default=None,
                    help="Checkpoint filename inside ckpt_dir (default: full_model_best.pt)")
parser.add_argument("--use_calibnet",  action=argparse.BooleanOptionalAction, default=None,
                    help="Default: infer from the checkpoint type.")
parser.add_argument("--scalers",       default=None,
                    help="Path to scalers.pkl (default: ckpt_dir/scalers.pkl)")
parser.add_argument("--batch-size",    type=int, default=512)
parser.add_argument("--device",        choices=("auto", "cuda", "cpu"), default="auto")
parser.add_argument("--compile",       action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--out",           required=True, help="Output 3-D trajectory PNG.")
parser.add_argument("--out-csv",       default=None, help="Default: <ckpt_dir>/testresult_amagposenet.csv")
parser.add_argument("--results-log",   default=None,
                    help="JSON summary path (default: <ckpt_dir>/test_results_amagposenet.json).")
args = parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (mirror test_mvec.py)
# ─────────────────────────────────────────────────────────────────────────────

def _read(path):
    df = pd.read_csv(path, header=None)
    try:
        df.iloc[0].astype(float); has_header = False
    except (ValueError, TypeError):
        has_header = True
    if has_header:
        df = pd.read_csv(path, header=0)
    return df.apply(pd.to_numeric, errors="coerce")


def _read_paired(volt_path, label_path):
    volt_df, label_df = _read(volt_path), _read(label_path)
    n = min(len(volt_df), len(label_df))
    volt_df, label_df = volt_df.iloc[:n], label_df.iloc[:n]
    mask = (volt_df.notna().all(axis=1).to_numpy()
            & label_df.notna().all(axis=1).to_numpy())
    return volt_df[mask].reset_index(drop=True), label_df[mask].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("  AMagPoseNet Evaluation")
print("=" * 65)
print(f"  Voltage : {args.test_voltage}")
print(f"  Label   : {args.test_label}")
print(f"  Ckpt    : {args.ckpt_dir}\n")

volt_df, label_df = _read_paired(args.test_voltage, args.test_label)
assert label_df.shape[1] == 6, f"Label must have 6 cols [x,y,z,mx,my,mz], got {label_df.shape[1]}"
assert volt_df.shape[1] == 64, f"Voltage must have 64 columns, got {volt_df.shape[1]}"

voltages = volt_df.values.astype(np.float32)
labels   = label_df.values.astype(np.float32)
N        = len(voltages)
print(f"  Test samples: {N}")

scaler_path = args.scalers or os.path.join(args.ckpt_dir, "scalers.pkl")
print(f"  Loading scalers from {scaler_path} ...")
with open(scaler_path, "rb") as f:
    scalers = pickle.load(f)

assert scalers.get("label_format") == "mvec", (
    f"Expected label_format='mvec', got {scalers.get('label_format')!r}"
)

volt_scaler  = scalers["volt"]
label_scaler = scalers["label"]

volt_test   = volt_scaler.transform(voltages)
volt_tensor = torch.tensor(volt_test, dtype=torch.float32).view(-1, 1, 8, 8)

gt_xyz  = labels[:, :3]
gt_mvec = labels[:, 3:6]
m_norms = np.linalg.norm(gt_mvec, axis=1, keepdims=True)
gt_mvec = gt_mvec / (m_norms + 1e-14)


# ─────────────────────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────────────────────

if args.device == "auto":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(args.device)
if device.type == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("--device cuda was requested but CUDA is unavailable.")
print(f"  Device  : {device}")

ckpt_name = args.checkpoint or "full_model_best.pt"
ckpt_path = os.path.join(args.ckpt_dir, ckpt_name)
if not os.path.exists(ckpt_path):
    ckpt_path = os.path.join(args.ckpt_dir, "posenet_pretrained.pt")
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

# Detect phase from checkpoint
is_full = "model_state_dict" in ckpt or "calinet_state_dict" in ckpt
use_calibnet = is_full if args.use_calibnet is None else args.use_calibnet

model = AMagPoseNet(use_calinet=use_calibnet).to(device)

if args.compile and platform.system() != "Windows":
    try:
        model = torch.compile(model)
        print("torch.compile enabled")
    except Exception:
        print("torch.compile not available - skipping")

print(f"  Loading checkpoint: {ckpt_path} ...")
if "model_state_dict" in ckpt:
    raw_state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
elif "locnet_state_dict" in ckpt:
    # Stage 2 checkpoint: only PoseNet
    raw_state = {"posenet." + k.replace("_orig_mod.", ""): v
                 for k, v in ckpt["locnet_state_dict"].items()}
else:
    raw_state = {k.replace("_orig_mod.", ""): v for k, v in ckpt.items()}

raw_model = getattr(model, "_orig_mod", model)
missing, unexpected = raw_model.load_state_dict(raw_state, strict=False)
if missing or unexpected:
    print(f"  Warning: missing={list(missing)}, unexpected={list(unexpected)}")

model.eval()
print(f"  Checkpoint epoch : {ckpt.get('epoch', '?')}")
print(f"  Best val loss    : {ckpt.get('best_val', 0):.6f}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Number of Parameters
# ─────────────────────────────────────────────────────────────────────────────

raw_model_np = getattr(model, "_orig_mod", model)

def _count_params(module):
    """Count total and trainable parameters of a module."""
    total     = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable

total_params, total_trainable = _count_params(raw_model_np)

# Sub-network breakdown
subnet_info = {}
if hasattr(raw_model_np, "calinet"):
    c_total, c_train = _count_params(raw_model_np.calinet)
    subnet_info["CaliNet"] = (c_total, c_train)
if hasattr(raw_model_np, "posenet"):
    p_total, p_train = _count_params(raw_model_np.posenet)
    subnet_info["PoseNet"] = (p_total, p_train)

print("─── Model Parameters ─────────────────────────────────────────────────")
for name, (tot, train) in subnet_info.items():
    print(f"  {name:<12} : {tot:>10,} params  ({train:>10,} trainable)")
print(f"  {'AMagPoseNet':<12} : {total_params:>10,} params  ({total_trainable:>10,} trainable)")
print("──────────────────────────────────────────────────────────────────────\n")


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

print("Running inference ...")
if args.batch_size <= 0:
    raise ValueError("--batch-size must be positive.")

# --- Warm-up run (không tính thời gian) ---
_warmup_in = volt_tensor[:min(args.batch_size, N)].to(device)
with torch.no_grad():
    _ = model(_warmup_in)
if device.type == "cuda":
    torch.cuda.synchronize()

# --- Đo inference time: lặp 100 lần trên từng sample đơn lẻ (theo paper) ---
_TIMING_REPEATS = 100
_single = volt_tensor[:1].to(device)
_times_ms: list[float] = []
with torch.no_grad():
    for _ in range(_TIMING_REPEATS):
        if device.type == "cuda":
            torch.cuda.synchronize()
        _t0 = time.perf_counter()
        _ = model(_single)
        if device.type == "cuda":
            torch.cuda.synchronize()
        _times_ms.append((time.perf_counter() - _t0) * 1000.0)

infer_mean_ms = float(np.mean(_times_ms))
infer_std_ms  = float(np.std(_times_ms))
infer_fps     = 1000.0 / infer_mean_ms if infer_mean_ms > 0 else float("inf")

# --- Full-dataset batched inference (dùng cho metrics) ---
_t_full_start = time.perf_counter()
with torch.no_grad():
    pred_scaled = np.concatenate([
        model(volt_tensor[start:start + args.batch_size].to(device)).float().cpu().numpy()
        for start in range(0, N, args.batch_size)
    ])
if device.type == "cuda":
    torch.cuda.synchronize()
_t_full_end = time.perf_counter()
total_infer_ms = (_t_full_end - _t_full_start) * 1000.0
per_sample_ms  = total_infer_ms / N if N > 0 else 0.0

pred_full = label_scaler.inverse_transform(pred_scaled)
pred_xyz  = pred_full[:, :3]
pred_mvec = pred_scaled[:, 3:]   # already unit-normalised by model


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

m = compute_metrics(pred_xyz, gt_xyz, pred_mvec, gt_mvec)
pos_err_mm  = m["pos_errors_mm"]
ori_err_deg = m["ori_errors_deg"]
sample_idx  = np.arange(1, N + 1)

pos_mean_running = np.cumsum(pos_err_mm) / sample_idx
pos_rmse_running = np.sqrt(np.cumsum(pos_err_mm ** 2) / sample_idx)
ori_mean_running = np.cumsum(ori_err_deg) / sample_idx
ori_rmse_running = np.sqrt(np.cumsum(ori_err_deg ** 2) / sample_idx)


# ─────────────────────────────────────────────────────────────────────────────
# Console output (same format as test_mvec.py)
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n  {'Pt':<5} {'PX':>8} {'PY':>8} {'PZ':>8} "
      f"{'GX':>8} {'GY':>8} {'GZ':>8} {'PosErr(mm)':>11} {'OriErr(°)':>10}")
print("  " + "-" * 95)
for i in range(N):
    print(f"  {i:<5} "
          f"{pred_xyz[i,0]:>8.4f} {pred_xyz[i,1]:>8.4f} {pred_xyz[i,2]:>8.4f} "
          f"{gt_xyz[i,0]:>8.4f} {gt_xyz[i,1]:>8.4f} {gt_xyz[i,2]:>8.4f} "
          f"{pos_err_mm[i]:>11.2f} {ori_err_deg[i]:>10.3f}")

print("\n─── AMagPoseNet Test Results ─────────────────────────────────────────")
print(f"  Số điểm test              : {N}")
print(f"  Mean position error       : {m['mean_pos_mm']:.2f} mm")
print(f"  RMSE position             : {m['rmse_pos_mm']:.2f} mm")
print(f"  Max  position error       : {m['max_pos_mm']:.2f} mm")
print(f"  MAE x / y / z             : {m['mae_xyz_mm'][0]:.2f} / "
      f"{m['mae_xyz_mm'][1]:.2f} / {m['mae_xyz_mm'][2]:.2f} mm")
print(f"  Mean orientation error    : {m['mean_ori_deg']:.3f}°")
print(f"  RMSE orientation          : {m['rmse_ori_deg']:.3f}°")
print(f"  Max  orientation error    : {m['max_ori_deg']:.3f}°")
print("─────────────────────────────────────────────────────────────────────")
print(f"  Total params (AMagPoseNet): {total_params:,}")
for name, (tot, _) in subnet_info.items():
    print(f"    {name:<10} : {tot:,}")
print(f"  Inference time (1 sample) : {infer_mean_ms:.3f} ± {infer_std_ms:.3f} ms  "
      f"(over {_TIMING_REPEATS} runs, device={device})")
print(f"  Estimated throughput      : {infer_fps:.1f} FPS")
print(f"  Full-set infer time       : {total_infer_ms:.1f} ms total  "
      f"≈ {per_sample_ms:.3f} ms/sample  ({N} samples, batch={args.batch_size})")
print("─────────────────────────────────────────────────────────────────────\n")


# ─────────────────────────────────────────────────────────────────────────────
# Save CSV (same columns as test_mvec.py)
# ─────────────────────────────────────────────────────────────────────────────

out_dir  = os.path.dirname(os.path.abspath(args.out))
os.makedirs(out_dir, exist_ok=True)
csv_path = args.out_csv or os.path.join(args.ckpt_dir, "testresult_amagposenet.csv")
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
    ])
    for i in range(N):
        writer.writerow([
            i,
            round(float(pred_xyz[i, 0]), 4), round(float(pred_xyz[i, 1]), 4),
            round(float(pred_xyz[i, 2]), 4),
            round(float(gt_xyz[i, 0]),   4), round(float(gt_xyz[i, 1]),   4),
            round(float(gt_xyz[i, 2]),   4),
            round(float(pos_err_mm[i]),  2),
            round(float(pred_mvec[i, 0]), 4), round(float(pred_mvec[i, 1]), 4),
            round(float(pred_mvec[i, 2]), 4),
            round(float(gt_mvec[i, 0]),   4), round(float(gt_mvec[i, 1]),   4),
            round(float(gt_mvec[i, 2]),   4),
            round(float(ori_err_deg[i]), 3),
        ])
print(f"Saved CSV : {csv_path}")

results_log = args.results_log or os.path.join(args.ckpt_dir, "test_results_amagposenet.json")
os.makedirs(os.path.dirname(os.path.abspath(results_log)), exist_ok=True)
with open(results_log, "w", encoding="utf-8") as f:
    json.dump({
        "model": "AMagPoseNet",
        "checkpoint": os.path.abspath(ckpt_path),
        "checkpoint_epoch": ckpt.get("epoch"),
        "test_samples": N,
        "metrics": {
            key: value.tolist() if isinstance(value, np.ndarray) else float(value)
            for key, value in m.items() if key not in {"pos_errors_mm", "ori_errors_deg"}
        },
        "timing": {
            "single_sample_mean_ms": infer_mean_ms,
            "single_sample_std_ms": infer_std_ms,
            "full_set_ms": total_infer_ms,
            "per_sample_ms": per_sample_ms,
            "throughput_fps": infer_fps,
            "timing_repeats": _TIMING_REPEATS,
        },
        "parameters": {"total": total_params, "trainable": total_trainable},
    }, f, indent=2)
print(f"Saved results log : {results_log}")


# ─────────────────────────────────────────────────────────────────────────────
# Figures (same layout as test_mvec.py)
# ─────────────────────────────────────────────────────────────────────────────

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

fig1 = plt.figure(figsize=(10, 7))
ax   = fig1.add_subplot(111, projection="3d")
ax.scatter(gt_xyz[:, 0],   gt_xyz[:, 1],   gt_xyz[:, 2],
           color="blue", s=35, label="Ground Truth", zorder=5)
ax.scatter(pred_xyz[:, 0], pred_xyz[:, 1], pred_xyz[:, 2],
           color="red", s=35, label="Predicted", zorder=5, marker="x")
for i in range(N):
    ax.plot([gt_xyz[i, 0], pred_xyz[i, 0]],
            [gt_xyz[i, 1], pred_xyz[i, 1]],
            [gt_xyz[i, 2], pred_xyz[i, 2]],
            color="gray", linewidth=0.5, alpha=0.5)
ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
ax.set_title(
    f"AMagPoseNet – Test Set ({N} pts)\n"
    f"Position Error — Mean: {m['mean_pos_mm']:.2f} mm  RMSE: {m['rmse_pos_mm']:.2f} mm  "
    f"Max: {m['max_pos_mm']:.2f} mm\n"
    f"Orientation Error — Mean: {m['mean_ori_deg']:.2f}°  RMSE: {m['rmse_ori_deg']:.2f}°  "
    f"Max: {m['max_ori_deg']:.2f}°",
    fontsize=10,
)
ax.legend(fontsize=10); ax.grid(True); plt.tight_layout()
fig1.savefig(args.out, dpi=150, bbox_inches="tight")
print(f"Saved Figure 1 : {args.out}")

fig2, ax2 = plt.subplots(figsize=(11, 5))
ax2.plot(sample_idx, pos_err_mm, color="#5B9BD5", lw=0.8, alpha=0.55, label="Per-sample")
ax2.plot(sample_idx, pos_mean_running, color="#2E75B6", lw=2.0,
         label=f"Mean  ({pos_mean_running[-1]:.2f} mm)")
ax2.plot(sample_idx, pos_rmse_running, color="#C55A11", lw=2.0, ls="--",
         label=f"RMSE  ({pos_rmse_running[-1]:.2f} mm)")
ax2.set_xlabel("Sample index", fontsize=12)
ax2.set_ylabel("Position error (mm)", fontsize=12)
ax2.set_title(
    f"AMagPoseNet – Position Error\n"
    f"Mean = {m['mean_pos_mm']:.2f} mm  |  RMSE = {m['rmse_pos_mm']:.2f} mm  |  "
    f"Max = {m['max_pos_mm']:.2f} mm",
    fontsize=12,
)
ax2.legend(fontsize=10); ax2.grid(True, ls="--", alpha=0.5)
ax2.set_xlim(1, N); ax2.set_ylim(bottom=0); plt.tight_layout()
out_fig2 = os.path.join(out_dir, "amagposenet_position_error.png")
fig2.savefig(out_fig2, dpi=150, bbox_inches="tight")
print(f"Saved Figure 2 : {out_fig2}")

fig3, ax3 = plt.subplots(figsize=(11, 5))
ax3.plot(sample_idx, ori_err_deg, color="#70AD47", lw=0.8, alpha=0.55, label="Per-sample")
ax3.plot(sample_idx, ori_mean_running, color="#375623", lw=2.0,
         label=f"Mean  ({ori_mean_running[-1]:.3f}°)")
ax3.plot(sample_idx, ori_rmse_running, color="#843C0C", lw=2.0, ls="--",
         label=f"RMSE  ({ori_rmse_running[-1]:.3f}°)")
ax3.set_xlabel("Sample index", fontsize=12)
ax3.set_ylabel("Orientation error (°)", fontsize=12)
ax3.set_title(
    f"AMagPoseNet – Orientation Error\n"
    f"Mean = {m['mean_ori_deg']:.3f}°  |  RMSE = {m['rmse_ori_deg']:.3f}°  |  "
    f"Max = {m['max_ori_deg']:.3f}°",
    fontsize=12,
)
ax3.legend(fontsize=10); ax3.grid(True, ls="--", alpha=0.5)
ax3.set_xlim(1, N); ax3.set_ylim(bottom=0); plt.tight_layout()
out_fig3 = os.path.join(out_dir, "amagposenet_orientation_error.png")
fig3.savefig(out_fig3, dpi=150, bbox_inches="tight")
print(f"Saved Figure 3 : {out_fig3}")

plt.show()
