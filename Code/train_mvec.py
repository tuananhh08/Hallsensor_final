from __future__ import annotations

import argparse
import json
import os
import pickle
import platform
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from data_mvec import PosLabelScaler, StreamingMinMaxScaler
from loss_mvec import CalibLocLoss, HuberPoseLossMVec
from model import Model
from training_loop import default_num_workers, make_loaders


def get_config():
    root = Path(__file__).resolve().parent.parent / "Dataset" / "Dataset"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("calibnet", "locnet", "finetune"), required=True)
    p.add_argument("--raw-voltage", default=str(root / "Grid_data.csv"))
    p.add_argument("--clean-voltage", default=str(root / "Grid_data_computed.csv"))
    p.add_argument("--raw-label", default=str(root / "Grid_points_coordinates.csv"))
    p.add_argument("--synthetic-voltage", default=str(root / "synthetic_grid_data.csv"))
    p.add_argument("--synthetic-label", default=str(root / "synthetic_grid_coordinates.csv"))
    p.add_argument("--calib-physical-csv", default=str(root / "Calibration_Physical_new.csv"))
    p.add_argument("--calib-alpha-csv", default=str(root / "Calibration_Alpha_new.csv"))
    p.add_argument("--ckpt-dir", default="./ckpt_mvec")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--scaler-file", default=None, help="Existing scalers.pkl; never refit it.")
    p.add_argument("--calibnet-checkpoint", default=None)
    p.add_argument("--locnet-checkpoint", default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-epochs", "--epochs", dest="num_epochs", type=int, default=200)
    p.add_argument("--lr-calibnet", type=float, default=2e-4)
    p.add_argument("--lr-locnet", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=4.56e-3)
    p.add_argument("--lambda-calib", "--lambda-mod", dest="lambda_calib", type=float, default=0.1)
    p.add_argument("--calib-delta", "--mod-delta", dest="calib_delta", type=float, default=0.05)
    p.add_argument("--lambda-ori", type=float, default=1.0)
    p.add_argument("--delta-xyz", type=float, default=0.061)
    p.add_argument("--lambda-pos", type=float, default=1.0)
    p.add_argument("--lambda-physics", type=float, default=1e-4)
    p.add_argument("--physics-delta", type=float, default=0.002)
    p.add_argument("--no-physics", action="store_true")
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--patience", type=int, default=45)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--samples-per-epoch", type=int, default=0)
    p.add_argument("--benchmark-samples", type=int, default=0,
                   help="Optional post-training single-sample inference benchmark (0 disables it).")
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def _read(path: str, width: int, name: str) -> np.ndarray:
    frame = pd.read_csv(path).apply(pd.to_numeric, errors="coerce")
    if frame.shape[1] != width:
        raise ValueError(f"{name} must have {width} columns, got {frame.shape[1]}: {path}")
    values = frame.to_numpy(np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values: {path}")
    return values


def _normalize_mvec(labels: np.ndarray) -> np.ndarray:
    labels = labels.copy(); norm = np.linalg.norm(labels[:, 3:], axis=1)
    if (norm < 1e-12).any(): raise ValueError("Pose labels contain a zero magnetic-moment vector")
    labels[:, 3:] /= norm[:, None]
    return labels


def _source_info(path: str) -> dict:
    p = Path(path).resolve(); s = p.stat()
    return {"path": str(p), "size": s.st_size, "mtime_ns": s.st_mtime_ns}


def _scaler_signature(scalers: dict) -> dict:
    return {"volt_min": np.asarray(scalers["volt"].data_min_).round(9).tolist(),
            "volt_max": np.asarray(scalers["volt"].data_max_).round(9).tolist(),
            "xyz_mean": np.asarray(scalers["label"].xyz_scaler.mean_).round(9).tolist()}


class MultiMemmapDataset(Dataset):
    """Lazy NPY-backed dataset for (x), (x,y), or (noisy,clean,pose)."""
    def __init__(self, files: list[Path], indices: np.ndarray):
        self.files = [str(x) for x in files]; self.indices = np.asarray(indices, dtype=np.int64); self._arrays = None
    def __len__(self): return len(self.indices)
    def _open(self):
        if self._arrays is None: self._arrays = [np.load(p, mmap_mode="r") for p in self.files]
    def __getitem__(self, i):
        self._open(); index = self.indices[i]
        return tuple(torch.from_numpy(a[index].copy()) for a in self._arrays)
    def __getstate__(self):
        state = self.__dict__.copy(); state["_arrays"] = None; return state


def _save_json(path: Path, data: object):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def prepare_cache(cfg, cache_dir: Path, scaler_file: str | None):
    """Fit/reuse scalers and prepare cached raw, clean, synthetic arrays."""
    sources = {key: _source_info(value) for key, value in {
        "raw": cfg.raw_voltage, "clean": cfg.clean_voltage, "raw_label": cfg.raw_label,
        "synthetic": cfg.synthetic_voltage, "synthetic_label": cfg.synthetic_label}.items()}
    cache_dir.mkdir(parents=True, exist_ok=True); manifest_path = cache_dir / "three_phase_manifest.json"
    external_scalers = scaler_file is not None and Path(scaler_file).is_file()
    if external_scalers:
        with open(scaler_file, "rb") as f: scalers = pickle.load(f)
    else: scalers = None
    expected = {"version": 1, "sources": sources, "val_ratio": cfg.val_ratio, "seed": cfg.seed,
                "external_scaler": _scaler_signature(scalers) if scalers else None}
    arrays = [cache_dir / x for x in ("raw.npy", "clean.npy", "pose.npy", "synthetic.npy", "synthetic_pose.npy")]
    index_path = cache_dir / "splits.npz"
    try:
        valid = (not cfg.rebuild_cache and json.loads(manifest_path.read_text()) == expected and index_path.is_file()
                 and all(x.is_file() for x in arrays) and (cache_dir / "scalers.pkl").is_file())
    except (OSError, json.JSONDecodeError): valid = False
    if valid:
        with open(cache_dir / "scalers.pkl", "rb") as f: return [*arrays], np.load(index_path), pickle.load(f)

    raw, clean = _read(cfg.raw_voltage, 64, "Raw voltage"), _read(cfg.clean_voltage, 64, "Clean voltage")
    pose = _normalize_mvec(_read(cfg.raw_label, 6, "Raw pose labels"))
    synth, synth_pose = _read(cfg.synthetic_voltage, 64, "Synthetic voltage"), _normalize_mvec(_read(cfg.synthetic_label, 6, "Synthetic pose labels"))
    if not (len(raw) == len(clean) == len(pose)): raise ValueError("Raw, clean, and raw-label CSV row counts must match")
    if len(synth) != len(synth_pose): raise ValueError("Synthetic voltage/label CSV row counts must match")
    rng = np.random.default_rng(cfg.seed); ia = rng.permutation(len(raw)); ib = np.random.default_rng(cfg.seed + 1).permutation(len(synth))
    na, nb = max(1, int(len(ia) * cfg.val_ratio)), max(1, int(len(ib) * cfg.val_ratio))
    train_a, val_a, train_b, val_b = ia[na:], ia[:na], ib[nb:], ib[:nb]
    if scalers is None:
        volt = StreamingMinMaxScaler((0, 1)); [volt.partial_fit(x) for x in (raw[train_a], clean[train_a], synth[train_b])]
        label = PosLabelScaler(); label.partial_fit(np.concatenate((pose[train_a], synth_pose[train_b]), axis=0))
        scalers = {"volt": volt, "label": label, "label_format": "mvec", "voltage_space": "minmax_[0,1]"}
    vscale, lscale = scalers["volt"], scalers["label"]
    scaled = [vscale.transform(x).reshape(-1, 1, 8, 8).astype(np.float32) for x in (raw, clean, synth)]
    output = [scaled[0], scaled[1], lscale.transform(pose), scaled[2], lscale.transform(synth_pose)]
    for path, value in zip(arrays, output): np.save(path, value)
    np.savez(index_path, train_a=train_a, val_a=val_a, train_b=train_b, val_b=val_b)
    with open(cache_dir / "scalers.pkl", "wb") as f: pickle.dump(scalers, f)
    # `external_scaler` is a cache identity only when the caller explicitly
    # supplied one.  For a locally fitted scaler, its persisted cache copy is
    # the authoritative object on subsequent runs.
    _save_json(manifest_path, expected)
    return arrays, np.load(index_path), scalers


def _unwrap(model): return model._orig_mod if hasattr(model, "_orig_mod") else model
def _strip(state): return {key.replace("_orig_mod.", ""): value for key, value in state.items()}


def load_component(module, path, name, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = _strip(checkpoint.get(f"{name}_state_dict", checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))))
    if name == "locnet" and any(key.startswith("locnet.") for key in state):
        state = {key[7:]: value for key, value in state.items() if key.startswith("locnet.")}
    missing, unexpected = module.load_state_dict(state, strict=False)
    if missing or unexpected: print(f"Warning loading {name}: missing={list(missing)}, unexpected={list(unexpected)}")
    return checkpoint


def save_checkpoint(path, phase, epoch, model, optimizer, scheduler, best_val, metrics, scalers):
    model = _unwrap(model)
    data = {"phase": phase, "epoch": epoch, "best_val": best_val, "metrics": metrics,
            "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
            "preprocessing": {"voltage_space": "minmax_[0,1]", "sensor_order": "sensor_1..sensor_64 row-major 8x8", "scaler_file": "scalers.pkl"},
            "scaler_metadata": _scaler_signature(scalers)}
    if phase == "calibnet": data["calibnet_state_dict"] = model.calibnet.state_dict()
    elif phase == "locnet": data["locnet_state_dict"] = model.locnet.state_dict()
    else: data.update({"model_state_dict": model.state_dict(), "calibnet_state_dict": model.calibnet.state_dict(), "locnet_state_dict": model.locnet.state_dict()})
    torch.save(data, path)


def append_log(path: Path, entry: dict):
    try: history = json.loads(path.read_text()) if path.is_file() else []
    except json.JSONDecodeError: history = []
    history.append(entry); _save_json(path, history)


def plot_losses(history: list[dict], path: Path):
    if not history: return
    epoch = [x["epoch"] for x in history]; fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for key, label in (("total", "Total"), ("loc", "Localization"), ("calib", "Calibration")):
        axes[0].plot(epoch, [x["train"][key] for x in history], label=f"Train {label}")
        axes[0].plot(epoch, [x["val"][key] for x in history], linestyle="--", label=f"Val {label}")
    axes[0].set_ylabel("Loss"); axes[0].grid(alpha=.3); axes[0].legend(ncol=2)
    axes[1].plot(epoch, [x["train"]["mae"] for x in history], label="Train correction MAE")
    axes[1].plot(epoch, [x["val"]["mae"] for x in history], label="Val correction MAE")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Scaled voltage MAE"); axes[1].grid(alpha=.3); axes[1].legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def benchmark(model, device, samples: int) -> dict:
    model.eval(); x = torch.randn(1, 1, 8, 8, device=device)
    with torch.no_grad():
        for _ in range(20): model(x)
        if device.type == "cuda": torch.cuda.synchronize()
        timings = []
        for _ in range(samples):
            if device.type == "cuda":
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record(); model(x); end.record(); torch.cuda.synchronize(); timings.append(start.elapsed_time(end))
            else:
                start = time.perf_counter(); model(x); timings.append((time.perf_counter() - start) * 1000)
    values = np.asarray(timings)
    return {"samples": samples, "mean_ms": float(values.mean()), "std_ms": float(values.std()), "p95_ms": float(np.percentile(values, 95))}


def main():
    cfg = get_config(); random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    if not 0 < cfg.val_ratio < 1: raise ValueError("--val-ratio must be in (0, 1)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(cfg.ckpt_dir); run_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cfg.cache_dir or run_dir / "data_cache")
    if cfg.num_workers is None: cfg.num_workers = default_num_workers(device)
    arrays, split, scalers = prepare_cache(cfg, cache_dir, cfg.scaler_file)
    with open(run_dir / "scalers.pkl", "wb") as f: pickle.dump(scalers, f)
    np.savez(run_dir / "split_info_three_phase.npz", **{key: split[key] for key in split.files})
    if cfg.phase == "calibnet": train_ds, val_ds = MultiMemmapDataset(arrays[:2], split["train_a"]), MultiMemmapDataset(arrays[:2], split["val_a"])
    elif cfg.phase == "locnet": train_ds, val_ds = MultiMemmapDataset([arrays[3], arrays[4]], split["train_b"]), MultiMemmapDataset([arrays[3], arrays[4]], split["val_b"])
    else: train_ds, val_ds = MultiMemmapDataset(arrays[:3], split["train_a"]), MultiMemmapDataset(arrays[:3], split["val_a"])
    train_loader, val_loader = make_loaders(train_ds, val_ds, cfg.batch_size, cfg.num_workers, device, cfg.prefetch_factor, cfg.samples_per_epoch)
    base_model = Model(use_calibnet=cfg.phase != "locnet").to(device)
    if cfg.phase == "finetune":
        if not (cfg.calibnet_checkpoint and cfg.locnet_checkpoint): raise ValueError("Phase finetune requires both pretrained checkpoint paths")
        load_component(base_model.calibnet, cfg.calibnet_checkpoint, "calibnet", device); load_component(base_model.locnet, cfg.locnet_checkpoint, "locnet", device)
    params = ([{"params": base_model.calibnet.parameters(), "lr": cfg.lr_calibnet}] if cfg.phase == "calibnet" else
              [{"params": base_model.locnet.parameters(), "lr": cfg.lr_locnet}] if cfg.phase == "locnet" else
              [{"params": base_model.calibnet.parameters(), "lr": cfg.lr_calibnet}, {"params": base_model.locnet.parameters(), "lr": cfg.lr_locnet}])
    optimizer = torch.optim.AdamW(params, weight_decay=cfg.weight_decay)
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=.1, end_factor=1., total_iters=max(1, cfg.warmup_epochs))
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg.num_epochs - cfg.warmup_epochs), eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[cfg.warmup_epochs])
    physics = 0.0 if cfg.no_physics else cfg.lambda_physics
    criterion = HuberPoseLossMVec(cfg.lambda_ori, cfg.delta_xyz, cfg.lambda_pos, physics, cfg.physics_delta,
        cfg.calib_physical_csv if physics else None, cfg.calib_alpha_csv if physics else None,
        scalers["volt"] if physics else None, scalers["label"] if physics else None).to(device)
    pipeline_criterion = CalibLocLoss(criterion, cfg.lambda_calib, cfg.calib_delta).to(device)
    model = base_model
    if cfg.compile and device.type == "cuda" and platform.system() != "Windows":
        try: model = torch.compile(base_model); print("torch.compile enabled")
        except Exception as exc: print(f"torch.compile unavailable: {exc}")
    use_amp = device.type == "cuda"; amp = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_epoch, best = 1, float("inf"); log_path = run_dir / "train_log.json"
    if cfg.resume:
        saved = torch.load(cfg.resume, map_location=device, weights_only=False)
        if cfg.phase == "calibnet": load_component(base_model.calibnet, cfg.resume, "calibnet", device)
        elif cfg.phase == "locnet": load_component(base_model.locnet, cfg.resume, "locnet", device)
        else: base_model.load_state_dict(_strip(saved["model_state_dict"]), strict=False)
        optimizer.load_state_dict(saved["optimizer_state_dict"]); scheduler.load_state_dict(saved["scheduler_state_dict"])
        start_epoch, best = saved["epoch"] + 1, saved["best_val"]
    names = {"calibnet": ("calibnet_pretrained.pt", "calibnet_last.pt"), "locnet": ("locnet_pretrained.pt", "locnet_last.pt"), "finetune": ("full_model_best.pt", "full_model_last.pt")}
    best_path, last_path = (run_dir / name for name in names[cfg.phase]); no_improve = 0
    def run_epoch(loader, training):
        (model.train() if training else model.eval()); totals = {k: 0. for k in ("total", "loc", "calib", "mae", "physics")}; seen = 0
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for batch in loader:
                batch = [x.to(device, non_blocking=True) for x in batch]
                if training: optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    if cfg.phase == "calibnet":
                        corrected = base_model.calibnet(batch[0])
                        calib = pipeline_criterion.calibration_term(corrected, batch[1])
                        loc = calib * 0
                        loss = calib
                    elif cfg.phase == "locnet":
                        corrected = batch[0]
                        pred = base_model.locnet(corrected)
                        loc, _, _ = criterion(pred, batch[1], X_b=corrected)
                        calib = loc * 0
                        loss = loc
                    else:
                        pred, aux = model(batch[0], return_features=True)
                        corrected = aux["corrected"]
                        loss, calib, loc, _, _ = pipeline_criterion(pred, batch[2], corrected, batch[1])
                if not torch.isfinite(loss): continue
                if training:
                    amp.scale(loss).backward(); amp.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(base_model.parameters(), 1.0); amp.step(optimizer); amp.update()
                n = len(batch[0]); totals["total"] += loss.item()*n; totals["loc"] += loc.item()*n; totals["calib"] += calib.item()*n; totals["mae"] += (corrected-batch[1 if cfg.phase != "locnet" else 0]).abs().mean().item()*n; totals["physics"] += criterion.latest_loss_physics.item()*n; seen += n
        return {k: v / max(1, seen) for k, v in totals.items()}
    print(f"Training {cfg.phase}: {len(train_ds):,} train / {len(val_ds):,} val, device={device}, cache={cache_dir}")
    for epoch in range(start_epoch, cfg.num_epochs + 1):
        t0 = time.time(); train_metrics, val_metrics = run_epoch(train_loader, True), run_epoch(val_loader, False); scheduler.step()
        entry = {"epoch": epoch, "train": train_metrics, "val": val_metrics, "lr": [g["lr"] for g in optimizer.param_groups], "seconds": time.time()-t0}
        append_log(log_path, entry); print(f"{epoch:03d} train={train_metrics['total']:.6f} val={val_metrics['total']:.6f} loc={val_metrics['loc']:.6f} calib={val_metrics['calib']:.6f} phys={val_metrics['physics']:.6f} ({entry['seconds']:.1f}s)")
        save_checkpoint(last_path, cfg.phase, epoch, base_model, optimizer, scheduler, best, entry, scalers)
        if val_metrics["total"] < best:
            best, no_improve = val_metrics["total"], 0; save_checkpoint(best_path, cfg.phase, epoch, base_model, optimizer, scheduler, best, entry, scalers)
            print(f"  -> best saved: {best:.6f}")
        else: no_improve += 1
        if epoch % cfg.save_every == 0: save_checkpoint(run_dir / f"{cfg.phase}_epoch_{epoch:04d}.pt", cfg.phase, epoch, base_model, optimizer, scheduler, best, entry, scalers)
        if no_improve >= cfg.patience: print(f"Early stopping after {cfg.patience} unimproved epochs."); break
    try: history = json.loads(log_path.read_text())
    except json.JSONDecodeError: history = []
    plot_losses(history, run_dir / "loss_plot.png")
    if cfg.benchmark_samples > 0 and best_path.is_file():
        saved = torch.load(best_path, map_location=device, weights_only=False)
        if cfg.phase == "calibnet": load_component(base_model.calibnet, best_path, "calibnet", device)
        elif cfg.phase == "locnet": load_component(base_model.locnet, best_path, "locnet", device)
        else: base_model.load_state_dict(_strip(saved["model_state_dict"]), strict=False)
        results = benchmark(model, device, cfg.benchmark_samples); _save_json(run_dir / "inference_time.json", results)
        print(f"Inference: {results['mean_ms']:.3f} ms/sample (p95 {results['p95_ms']:.3f} ms)")
    print(f"Done. Best validation loss: {best:.6f}; checkpoint: {best_path}")


if __name__ == "__main__": main()
