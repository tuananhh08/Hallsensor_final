"""Evaluate CRLocNet with AMagPoseNet-compatible metrics and timing output."""
from __future__ import annotations

import argparse
import csv
import pickle
import platform
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from Benchmark.CRLocNet.evaluation.metrics import compute_metrics
from Benchmark.CRLocNet.models import CRLocNet


def _read(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, header=None)
    try:
        frame.iloc[0].astype(float)
    except (ValueError, TypeError):
        frame = pd.read_csv(path)
    return frame.apply(pd.to_numeric, errors="coerce")


def _read_paired(voltage_path: str, label_path: str, epm_path: str | None):
    """Drop invalid rows without ever de-aligning voltage, pose, and EPM rows."""
    voltage, labels = _read(voltage_path), _read(label_path)
    epm = _read(epm_path) if epm_path else None
    n = min(len(voltage), len(labels), len(epm) if epm is not None else len(voltage))
    voltage, labels = voltage.iloc[:n], labels.iloc[:n]
    if epm is not None:
        epm = epm.iloc[:n]
    valid = voltage.notna().all(axis=1).to_numpy() & labels.notna().all(axis=1).to_numpy()
    if epm is not None:
        valid &= epm.notna().all(axis=1).to_numpy()
    voltage, labels = voltage.loc[valid].reset_index(drop=True), labels.loc[valid].reset_index(drop=True)
    return voltage, labels, epm.loc[valid].reset_index(drop=True) if epm is not None else None


def _count_params(module: torch.nn.Module) -> tuple[int, int]:
    return sum(p.numel() for p in module.parameters()), sum(p.numel() for p in module.parameters() if p.requires_grad)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CRLocNet benchmark")
    parser.add_argument("--test-voltage", required=True); parser.add_argument("--test-label", required=True)
    parser.add_argument("--test-epm"); parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--checkpoint", default="full_model_best.pt"); parser.add_argument("--scalers")
    parser.add_argument("--batch-size", type=int, default=512); parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--timing-repeats", type=int, default=100); parser.add_argument("--out", required=True); parser.add_argument("--out-csv")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.timing_repeats <= 0:
        raise ValueError("--batch-size and --timing-repeats must be positive.")

    voltage_df, label_df, epm_df = _read_paired(args.test_voltage, args.test_label, args.test_epm)
    if voltage_df.shape[1] != 64 or label_df.shape[1] != 6:
        raise ValueError("Expected 64 voltage columns and 6 labels [x,y,z,mx,my,mz].")
    if epm_df is not None and epm_df.shape[1] != 64:
        raise ValueError("--test-epm must have exactly 64 field columns.")
    raw, labels = voltage_df.to_numpy(np.float32), label_df.to_numpy(np.float32)
    epm = np.zeros_like(raw) if epm_df is None else epm_df.to_numpy(np.float32)
    n = len(raw)
    if not n:
        raise ValueError("No valid, row-aligned test samples remain after CSV parsing.")

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable.")
    scaler_path = Path(args.scalers) if args.scalers else Path(args.ckpt_dir) / "scalers.pkl"
    with scaler_path.open("rb") as file:
        scalers = pickle.load(file)
    x = torch.tensor(scalers["volt"].transform(raw).reshape(-1, 1, 8, 8), dtype=torch.float32)
    e = torch.tensor(scalers["volt"].transform(epm).reshape(-1, 1, 8, 8), dtype=torch.float32)
    checkpoint_path = Path(args.ckpt_dir) / args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = CRLocNet().to(device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict({key.replace("_orig_mod.", ""): value for key, value in state.items()}, strict=True)
    if args.compile and platform.system() != "Windows":
        try:
            model = torch.compile(model); print("torch.compile enabled")
        except Exception:
            print("torch.compile not available - skipping")
    model.eval(); raw_model = getattr(model, "_orig_mod", model)
    total_params, total_trainable = _count_params(raw_model)

    print("=" * 65 + "\n  CRLocNet Evaluation\n" + "=" * 65)
    print(f"  Test samples: {n} | Device: {device} | EPM: {'provided' if epm_df is not None else 'zero baseline'}")
    print(f"  Checkpoint epoch: {checkpoint.get('epoch', '?')} | Best val loss: {checkpoint.get('best_val', 0):.6f}")
    print("─── Model Parameters ─────────────────────────────────────────────────")
    for name, module in (("ModNet", raw_model.modnet), ("LocNet", raw_model.locnet)):
        total, trainable = _count_params(module); print(f"  {name:<12} : {total:>10,} params  ({trainable:>10,} trainable)")
    print(f"  {'CRLocNet':<12} : {total_params:>10,} params  ({total_trainable:>10,} trainable)")

    with torch.no_grad():
        model(x[:min(args.batch_size, n)].to(device), e[:min(args.batch_size, n)].to(device))
    if device.type == "cuda": torch.cuda.synchronize()
    one_x, one_e, latency_ms = x[:1].to(device), e[:1].to(device), []
    with torch.no_grad():
        for _ in range(args.timing_repeats):
            if device.type == "cuda": torch.cuda.synchronize()
            start = time.perf_counter(); model(one_x, one_e)
            if device.type == "cuda": torch.cuda.synchronize()
            latency_ms.append((time.perf_counter() - start) * 1000.0)
        if device.type == "cuda": torch.cuda.synchronize()
        start = time.perf_counter()
        pred_scaled = np.concatenate([model(x[i:i + args.batch_size].to(device), e[i:i + args.batch_size].to(device)).float().cpu().numpy() for i in range(0, n, args.batch_size)])
        if device.type == "cuda": torch.cuda.synchronize()
        full_infer_ms = (time.perf_counter() - start) * 1000.0

    pred_full = scalers["label"].inverse_transform(pred_scaled)
    pred_xyz, pred_mvec, gt_xyz = pred_full[:, :3], pred_scaled[:, 3:], labels[:, :3]
    gt_mvec = labels[:, 3:] / (np.linalg.norm(labels[:, 3:], axis=1, keepdims=True) + 1e-14)
    metrics = compute_metrics(pred_xyz, gt_xyz, pred_mvec, gt_mvec)
    pos_err, ori_err = metrics["pos_errors_mm"], metrics["ori_errors_deg"]
    sample_idx = np.arange(1, n + 1)
    pos_mean, pos_rmse = np.cumsum(pos_err) / sample_idx, np.sqrt(np.cumsum(pos_err ** 2) / sample_idx)
    ori_mean, ori_rmse = np.cumsum(ori_err) / sample_idx, np.sqrt(np.cumsum(ori_err ** 2) / sample_idx)
    latency_mean, latency_std = float(np.mean(latency_ms)), float(np.std(latency_ms))

    print(f"\n  {'Pt':<5} {'PX':>8} {'PY':>8} {'PZ':>8} {'GX':>8} {'GY':>8} {'GZ':>8} {'PosErr(mm)':>11} {'OriErr(°)':>10}")
    print("  " + "-" * 95)
    for i in range(n):
        print(f"  {i:<5} {pred_xyz[i, 0]:>8.4f} {pred_xyz[i, 1]:>8.4f} {pred_xyz[i, 2]:>8.4f} "
              f"{gt_xyz[i, 0]:>8.4f} {gt_xyz[i, 1]:>8.4f} {gt_xyz[i, 2]:>8.4f} "
              f"{pos_err[i]:>11.2f} {ori_err[i]:>10.3f}")
    print("\n─── CRLocNet Test Results ───────────────────────────────────────────")
    print(f"  Số điểm test              : {n}")
    print(f"  Mean position error       : {metrics['mean_pos_mm']:.2f} mm\n  RMSE position             : {metrics['rmse_pos_mm']:.2f} mm\n  Max  position error       : {metrics['max_pos_mm']:.2f} mm")
    print(f"  MAE x / y / z             : {metrics['mae_xyz_mm'][0]:.2f} / {metrics['mae_xyz_mm'][1]:.2f} / {metrics['mae_xyz_mm'][2]:.2f} mm")
    print(f"  Mean orientation error    : {metrics['mean_ori_deg']:.3f}°\n  RMSE orientation          : {metrics['rmse_ori_deg']:.3f}°\n  Max  orientation error    : {metrics['max_ori_deg']:.3f}°")
    print(f"  Inference time (1 sample) : {latency_mean:.3f} ± {latency_std:.3f} ms (over {args.timing_repeats} runs, device={device})")
    print(f"  Estimated throughput      : {1000.0 / latency_mean:.1f} FPS")
    print(f"  Full-set infer time       : {full_infer_ms:.1f} ms total ≈ {full_infer_ms / n:.3f} ms/sample ({n} samples, batch={args.batch_size})")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.out_csv) if args.out_csv else Path(args.ckpt_dir) / "testresult_crlocnet.csv"; csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as file:
        writer = csv.writer(file); writer.writerow(["Point", "Pred_X", "Pred_Y", "Pred_Z", "GT_X", "GT_Y", "GT_Z", "PosErr_mm", "Pred_mx", "Pred_my", "Pred_mz", "GT_mx", "GT_my", "GT_mz", "OriErr_deg"])
        for i in range(n): writer.writerow([i, *np.round(pred_xyz[i], 4), *np.round(gt_xyz[i], 4), round(float(pos_err[i]), 2), *np.round(pred_mvec[i], 4), *np.round(gt_mvec[i], 4), round(float(ori_err[i]), 3)])
    print(f"Saved CSV : {csv_path}")

    fig = plt.figure(figsize=(10, 7)); ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*gt_xyz.T, color="blue", s=35, label="Ground Truth"); ax.scatter(*pred_xyz.T, color="red", s=35, marker="x", label="Predicted")
    for i in range(n): ax.plot([gt_xyz[i, 0], pred_xyz[i, 0]], [gt_xyz[i, 1], pred_xyz[i, 1]], [gt_xyz[i, 2], pred_xyz[i, 2]], color="gray", linewidth=.5, alpha=.5)
    ax.set(xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)", title=f"CRLocNet – Test Set ({n} pts)\nPosition Error — Mean: {metrics['mean_pos_mm']:.2f} mm  RMSE: {metrics['rmse_pos_mm']:.2f} mm  Max: {metrics['max_pos_mm']:.2f} mm\nOrientation Error — Mean: {metrics['mean_ori_deg']:.2f}°  RMSE: {metrics['rmse_ori_deg']:.2f}°  Max: {metrics['max_ori_deg']:.2f}°")
    ax.legend(); fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); print(f"Saved Figure 1 : {out}")
    for name, err, running_mean, running_rmse, unit, color in (("position", pos_err, pos_mean, pos_rmse, "mm", "#2E75B6"), ("orientation", ori_err, ori_mean, ori_rmse, "°", "#375623")):
        fig, ax = plt.subplots(figsize=(11, 5)); ax.plot(sample_idx, err, alpha=.55, label="Per-sample"); ax.plot(sample_idx, running_mean, color=color, lw=2, label=f"Mean ({running_mean[-1]:.3f} {unit})"); ax.plot(sample_idx, running_rmse, color="#C55A11", lw=2, ls="--", label=f"RMSE ({running_rmse[-1]:.3f} {unit})")
        ax.set(xlabel="Sample index", ylabel=f"{name.title()} error ({unit})", title=f"CRLocNet – {name.title()} Error"); ax.set_xlim(1, n); ax.set_ylim(bottom=0); ax.grid(alpha=.4); ax.legend(); fig.tight_layout(); figure_path = out.parent / f"crlocnet_{name}_error.png"; fig.savefig(figure_path, dpi=150, bbox_inches="tight"); print(f"Saved Figure : {figure_path}")
    plt.show()


if __name__ == "__main__":
    main()
