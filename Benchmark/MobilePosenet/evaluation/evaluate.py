"""Evaluate MobilePoseNet with the exact benchmark outputs of AMagPoseNet/test_mvec."""
from __future__ import annotations
import argparse, csv, pickle, sys, time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))
from Benchmark.MobilePosenet.evaluation.metrics import compute_metrics
from Benchmark.MobilePosenet.models import MobilePoseNet

def read(path):
    frame = pd.read_csv(path, header=None)
    try: frame.iloc[0].astype(float)
    except (ValueError, TypeError): frame = pd.read_csv(path)
    return frame.apply(pd.to_numeric, errors="coerce")

def paired(voltage, label):
    voltage, label = read(voltage), read(label); n = min(len(voltage), len(label)); voltage, label = voltage.iloc[:n], label.iloc[:n]
    mask = voltage.notna().all(axis=1).to_numpy() & label.notna().all(axis=1).to_numpy()
    return voltage.loc[mask].reset_index(drop=True), label.loc[mask].reset_index(drop=True)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test-voltage", required=True); p.add_argument("--test-label", required=True); p.add_argument("--ckpt-dir", required=True); p.add_argument("--checkpoint", default="mobileposenet_best.pt"); p.add_argument("--scalers"); p.add_argument("--out", required=True); p.add_argument("--out-csv"); p.add_argument("--batch-size", type=int, default=512); p.add_argument("--timing-repeats", type=int, default=100); p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto"); a = p.parse_args()
    voltage, label = paired(a.test_voltage, a.test_label)
    if voltage.shape[1] != 64 or label.shape[1] != 6: raise ValueError("Expected 64 voltage columns and 6 labels [x,y,z,mx,my,mz].")
    raw, gt = voltage.to_numpy(np.float32), label.to_numpy(np.float32); n = len(raw)
    with open(a.scalers or str(Path(a.ckpt_dir) / "scalers.pkl"), "rb") as f: scalers = pickle.load(f)
    x = torch.tensor(scalers["volt"].transform(raw).reshape(-1, 1, 8, 8), dtype=torch.float32)
    device = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available() else ("cpu" if a.device == "auto" else a.device))
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    ck = torch.load(Path(a.ckpt_dir) / a.checkpoint, map_location=device, weights_only=False); model = MobilePoseNet().to(device); model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ck["model_state_dict"].items()}); model.eval()
    total, trainable = sum(q.numel() for q in model.parameters()), sum(q.numel() for q in model.parameters() if q.requires_grad)
    with torch.no_grad(): model(x[:1].to(device))
    if device.type == "cuda": torch.cuda.synchronize()
    timings = []
    with torch.no_grad():
        for _ in range(a.timing_repeats):
            if device.type == "cuda": torch.cuda.synchronize()
            start = time.perf_counter(); model(x[:1].to(device))
            if device.type == "cuda": torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000.)
        if device.type == "cuda": torch.cuda.synchronize()
        start = time.perf_counter(); scaled = np.concatenate([model(x[i:i+a.batch_size].to(device)).cpu().numpy() for i in range(0, n, a.batch_size)])
        if device.type == "cuda": torch.cuda.synchronize()
        full_ms = (time.perf_counter() - start) * 1000.
    full = scalers["label"].inverse_transform(scaled); gt_mvec = gt[:, 3:] / (np.linalg.norm(gt[:, 3:], axis=1, keepdims=True) + 1e-14)
    m = compute_metrics(full[:, :3], gt[:, :3], scaled[:, 3:], gt_mvec); pos, ori = m["pos_errors_mm"], m["ori_errors_deg"]; index = np.arange(1, n+1)
    pos_mean, pos_rmse = np.cumsum(pos)/index, np.sqrt(np.cumsum(pos**2)/index); ori_mean, ori_rmse = np.cumsum(ori)/index, np.sqrt(np.cumsum(ori**2)/index)
    latency, latency_std = float(np.mean(timings)), float(np.std(timings))
    print("─── Model Parameters ─────────────────────────────────────────────────"); print(f"  {'MobilePoseNet':<12} : {total:>10,} params  ({trainable:>10,} trainable)")
    print(f"\n  {'Pt':<5} {'PX':>8} {'PY':>8} {'PZ':>8} {'GX':>8} {'GY':>8} {'GZ':>8} {'PosErr(mm)':>11} {'OriErr(°)':>10}\n  " + "-"*95)
    for i in range(n): print(f"  {i:<5} {full[i,0]:>8.4f} {full[i,1]:>8.4f} {full[i,2]:>8.4f} {gt[i,0]:>8.4f} {gt[i,1]:>8.4f} {gt[i,2]:>8.4f} {pos[i]:>11.2f} {ori[i]:>10.3f}")
    print("\n─── MobilePoseNet Test Results ──────────────────────────────────────"); print(f"  Số điểm test              : {n}\n  Mean position error       : {m['mean_pos_mm']:.2f} mm\n  RMSE position             : {m['rmse_pos_mm']:.2f} mm\n  Max  position error       : {m['max_pos_mm']:.2f} mm")
    mae = m["mae_xyz_mm"]; print(f"  MAE x / y / z             : {mae[0]:.2f} / {mae[1]:.2f} / {mae[2]:.2f} mm\n  Mean orientation error    : {m['mean_ori_deg']:.3f}°\n  RMSE orientation          : {m['rmse_ori_deg']:.3f}°\n  Max  orientation error    : {m['max_ori_deg']:.3f}°")
    print(f"  Total params (MobilePoseNet): {total:,}\n  Inference time (1 sample) : {latency:.3f} ± {latency_std:.3f} ms (over {a.timing_repeats} runs, device={device})\n  Estimated throughput      : {1000/latency:.1f} FPS\n  Full-set infer time       : {full_ms:.1f} ms total ≈ {full_ms/n:.3f} ms/sample ({n} samples, batch={a.batch_size})")
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True); csv_path = Path(a.out_csv) if a.out_csv else Path(a.ckpt_dir)/"testresult_mobileposenet.csv"; csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f); writer.writerow(["Point","Pred_X","Pred_Y","Pred_Z","GT_X","GT_Y","GT_Z","PosErr_mm","Pred_mx","Pred_my","Pred_mz","GT_mx","GT_my","GT_mz","OriErr_deg"])
        for i in range(n): writer.writerow([i,*np.round(full[i,:3],4),*np.round(gt[i,:3],4),round(float(pos[i]),2),*np.round(scaled[i,3:],4),*np.round(gt_mvec[i],4),round(float(ori[i]),3)])
    print(f"Saved CSV : {csv_path}")
    fig = plt.figure(figsize=(10,7)); ax=fig.add_subplot(111,projection="3d"); ax.scatter(*gt[:,:3].T,color="blue",s=35,label="Ground Truth"); ax.scatter(*full[:,:3].T,color="red",s=35,marker="x",label="Predicted")
    for i in range(n): ax.plot([gt[i,0],full[i,0]],[gt[i,1],full[i,1]],[gt[i,2],full[i,2]],color="gray",lw=.5,alpha=.5)
    ax.set(xlabel="X (m)",ylabel="Y (m)",zlabel="Z (m)",title=f"MobilePoseNet – Test Set ({n} pts)\nPosition Error — Mean: {m['mean_pos_mm']:.2f} mm  RMSE: {m['rmse_pos_mm']:.2f} mm  Max: {m['max_pos_mm']:.2f} mm\nOrientation Error — Mean: {m['mean_ori_deg']:.2f}°  RMSE: {m['rmse_ori_deg']:.2f}°  Max: {m['max_ori_deg']:.2f}°"); ax.legend(); fig.tight_layout(); fig.savefig(out,dpi=150,bbox_inches="tight"); print(f"Saved Figure 1 : {out}")
    for name, errors, means, rmses, unit in (("position",pos,pos_mean,pos_rmse,"mm"),("orientation",ori,ori_mean,ori_rmse,"°")):
        fig,ax=plt.subplots(figsize=(11,5)); ax.plot(index,errors,alpha=.55,label="Per-sample"); ax.plot(index,means,lw=2,label=f"Mean ({means[-1]:.3f} {unit})"); ax.plot(index,rmses,lw=2,ls="--",label=f"RMSE ({rmses[-1]:.3f} {unit})"); ax.set(xlabel="Sample index",ylabel=f"{name.title()} error ({unit})",title=f"MobilePoseNet – {name.title()} Error"); ax.set_xlim(1,n); ax.set_ylim(bottom=0); ax.grid(alpha=.4); ax.legend(); fig.tight_layout(); path=out.parent/f"mobileposenet_{name}_error.png"; fig.savefig(path,dpi=150,bbox_inches="tight"); print(f"Saved Figure : {path}")
    plt.show()
if __name__ == "__main__": main()
