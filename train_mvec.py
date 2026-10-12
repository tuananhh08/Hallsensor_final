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
from torch.utils.data import Dataset

from data_mvec import PosLabelScaler, StreamingMinMaxScaler
from loss_mvec import CalibLocLoss, HuberPoseLossMVec
from model import Model
from training_loop import default_num_workers, make_loaders


def get_config():
    root = Path(__file__).resolve().parent.parent / "Dataset" / "Dataset"
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # --- Required ---
    p.add_argument("--phase", choices=("calibnet", "locnet", "finetune"), required=True,
                   help="Training phase to run.")

    # --- Data paths ---
    p.add_argument("--raw-voltage",       default=str(root / "Grid_data.csv"))
    p.add_argument("--clean-voltage",     default=str(root / "Grid_data_computed.csv"))
    p.add_argument("--raw-label",         default=str(root / "Grid_points_coordinates.csv"))
    p.add_argument("--synthetic-voltage", default=str(root / "synthetic_grid_data2.csv"))
    p.add_argument("--synthetic-label",   default=str(root / "synthetic_grid_coordinates2.csv"))
    p.add_argument("--calib-physical-csv",default=str(root / "Calibration_Physical_new.csv"))
    p.add_argument("--calib-alpha-csv",   default=str(root / "Calibration_Alpha_new.csv"))

    # --- Directories and cache ---
    p.add_argument("--ckpt-dir",       default="./ckpt_mvec")
    p.add_argument("--cache-dir",      default=None)
    p.add_argument("--rebuild-cache",  action="store_true")
    p.add_argument("--scaler-file",    default=None,
                   help="Path to an existing scalers.pkl to reuse (skip refitting).")

    # --- Checkpoint loading ---
    p.add_argument("--calibnet-checkpoint", default=None,
                   help="Pre-trained CalibNet checkpoint (required for --phase finetune).")
    p.add_argument("--locnet-checkpoint",   default=None,
                   help="Pre-trained LocNet checkpoint (required for --phase finetune).")
    p.add_argument("--resume",              default=None,
                   help="Resume training from this checkpoint.")

    # --- Training hyper-parameters ---
    p.add_argument("--batch-size",     type=int,   default=256)
    p.add_argument("--num-epochs", "--epochs", dest="num_epochs", type=int, default=200)
    p.add_argument("--lr-calibnet",    type=float, default=2e-4)
    p.add_argument("--lr-locnet",      type=float, default=1e-3)
    p.add_argument("--weight-decay",   type=float, default=4.56e-3)
    p.add_argument("--warmup-epochs",  type=int,   default=5)
    p.add_argument("--save-every",     type=int,   default=5)
    p.add_argument("--patience",       type=int,   default=45)
    p.add_argument("--val-ratio",      type=float, default=0.2)
    p.add_argument("--seed",           type=int,   default=42)

    # --- Loss weights ---
    p.add_argument("--lambda-calib",   type=float, default=0.1,
                   help="Weight for CalibNet Huber loss in finetune phase.")
    p.add_argument("--calib-delta",    type=float, default=0.05,
                   help="Huber delta for CalibNet loss.")
    p.add_argument("--lambda-ori",     type=float, default=1.0)
    p.add_argument("--delta-xyz",      type=float, default=0.061)
    p.add_argument("--lambda-pos",     type=float, default=1.0)
    p.add_argument("--lambda-physics", type=float, default=1e-4)
    p.add_argument("--physics-delta",  type=float, default=0.002)
    p.add_argument("--no-physics",     action="store_true",
                   help="Disable physics-informed loss term (saves memory in locnet/finetune).")

    # --- DataLoader / misc ---
    p.add_argument("--num-workers",        type=int, default=None)
    p.add_argument("--prefetch-factor",    type=int, default=2)
    p.add_argument("--samples-per-epoch",  type=int, default=0,
                   help="Subsample N training examples per epoch (0 = full dataset).")
    p.add_argument("--benchmark-samples",  type=int, default=0,
                   help="Run latency benchmark after training (0 = disabled).")
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True,
                   help="Use torch.compile on CUDA (default: True).")
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
    """Normalise the magnetic-moment columns (indices 3:6) to unit vectors in-place copy."""
    labels = labels.copy()
    norm = np.linalg.norm(labels[:, 3:], axis=1)
    if (norm < 1e-12).any():
        raise ValueError("Pose labels contain a zero magnetic-moment vector")
    labels[:, 3:] /= norm[:, None]
    return labels


def _source_info(path: str) -> dict:
    p = Path(path).resolve()
    s = p.stat()
    return {"path": str(p), "size": s.st_size, "mtime_ns": s.st_mtime_ns}


def _scaler_signature(scalers: dict) -> dict:
    return {
        "volt_min":  np.asarray(scalers["volt"].data_min_).round(9).tolist(),
        "volt_max":  np.asarray(scalers["volt"].data_max_).round(9).tolist(),
        "xyz_mean": np.asarray(scalers["label"].xyz_scaler.mean_).round(9).tolist(),
    }


class MultiMemmapDataset(Dataset):
    """Lazy memory-mapped dataset backing one or more NPY arrays.

    Supports datasets with 2 arrays (voltage, label) as well as the
    3-array variant (raw_voltage, clean_voltage, pose_label) used by
    the finetune phase.
    """

    def __init__(self, files: list[Path], indices: np.ndarray):
        self.files = [str(x) for x in files]
        self.indices = np.asarray(indices, dtype=np.int64)
        self._arrays = None

    def __len__(self):
        return len(self.indices)

    def _open(self):
        if self._arrays is None:
            self._arrays = [np.load(p, mmap_mode="r") for p in self.files]

    def __getitem__(self, i):
        self._open()
        idx = self.indices[i]
        return tuple(torch.from_numpy(a[idx].copy()) for a in self._arrays)

    def __getstate__(self):
        # Reset mmaps so the object can be pickled by DataLoader workers.
        state = self.__dict__.copy()
        state["_arrays"] = None
        return state


def _save_json(path: Path, data: object):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def prepare_cache(cfg, cache_dir: Path, scaler_file: str | None):
    """Fit/reuse scalers and build the five cached NPY arrays for all phases.

    Array layout (shared across all phases):
      arrays[0] = raw.npy           – raw voltage,  shape (N_raw,  1, 8, 8), scaled
      arrays[1] = clean.npy         – clean voltage, shape (N_raw,  1, 8, 8), scaled
      arrays[2] = pose.npy          – raw pose label,shape (N_raw,  6)
      arrays[3] = synthetic.npy     – synth voltage, shape (N_synth, 1, 8, 8), scaled
      arrays[4] = synthetic_pose.npy– synth label,   shape (N_synth, 6)

    split keys:
      train_a / val_a  – indices into arrays[0..2]  (raw data)
      train_b / val_b  – indices into arrays[3..4]  (synthetic data)
    """
    sources = {
        "raw":             _source_info(cfg.raw_voltage),
        "clean":           _source_info(cfg.clean_voltage),
        "raw_label":       _source_info(cfg.raw_label),
        "synthetic":       _source_info(cfg.synthetic_voltage),
        "synthetic_label": _source_info(cfg.synthetic_label),
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "three_phase_manifest.json"

    # Load external scalers if provided, otherwise they will be fitted below.
    if scaler_file is not None and Path(scaler_file).is_file():
        with open(scaler_file, "rb") as f:
            scalers = pickle.load(f)
    else:
        scalers = None

    expected = {
        "version": 1,
        "sources": sources,
        "val_ratio": cfg.val_ratio,
        "seed": cfg.seed,
        "external_scaler": _scaler_signature(scalers) if scalers else None,
    }
    arrays = [cache_dir / name for name in
              ("raw.npy", "clean.npy", "pose.npy", "synthetic.npy", "synthetic_pose.npy")]
    index_path = cache_dir / "splits.npz"

    # Return early if a valid cache already exists.
    try:
        cache_ok = (
            not cfg.rebuild_cache
            and json.loads(manifest_path.read_text()) == expected
            and index_path.is_file()
            and all(a.is_file() for a in arrays)
            and (cache_dir / "scalers.pkl").is_file()
        )
    except (OSError, json.JSONDecodeError):
        cache_ok = False

    if cache_ok:
        with open(cache_dir / "scalers.pkl", "rb") as f:
            cached_scalers = pickle.load(f)
        return list(arrays), np.load(index_path), cached_scalers

    # ------------------------------------------------------------------ #
    # Cache is stale or missing — rebuild from CSV.                        #
    # ------------------------------------------------------------------ #
    print("[cache] Loading CSVs ...")
    raw   = _read(cfg.raw_voltage,       64, "Raw voltage")
    clean = _read(cfg.clean_voltage,     64, "Clean voltage")
    pose  = _normalize_mvec(_read(cfg.raw_label, 6, "Raw pose labels"))
    synth = _read(cfg.synthetic_voltage, 64, "Synthetic voltage")
    synth_pose = _normalize_mvec(_read(cfg.synthetic_label, 6, "Synthetic pose labels"))

    if not (len(raw) == len(clean) == len(pose)):
        raise ValueError("Raw voltage, clean voltage, and raw-label CSV row counts must match.")
    if len(synth) != len(synth_pose):
        raise ValueError("Synthetic voltage and label CSV row counts must match.")

    # Shuffle indices for each dataset independently so the two splits
    # have different random seeds (avoiding accidental correlation).
    rng_a = np.random.default_rng(cfg.seed)
    rng_b = np.random.default_rng(cfg.seed + 1)
    ia = rng_a.permutation(len(raw))
    ib = rng_b.permutation(len(synth))
    na = max(1, int(len(ia) * cfg.val_ratio))
    nb = max(1, int(len(ib) * cfg.val_ratio))
    val_a,   train_a   = ia[:na],  ia[na:]
    val_b,   train_b   = ib[:nb],  ib[nb:]

    # Fit scalers on training splits only to avoid target leakage.
    if scalers is None:
        print("[cache] Fitting scalers on training splits ...")
        volt = StreamingMinMaxScaler((0, 1))
        for chunk in (raw[train_a], clean[train_a], synth[train_b]):
            volt.partial_fit(chunk)
        label = PosLabelScaler()
        label.partial_fit(np.concatenate((pose[train_a], synth_pose[train_b]), axis=0))
        scalers = {
            "volt":          volt,
            "label":         label,
            "label_format":  "mvec",
            "voltage_space": "minmax_[0,1]",
        }

    vscale, lscale = scalers["volt"], scalers["label"]

    print("[cache] Writing NPY arrays ...")
    raw_s   = vscale.transform(raw).reshape(-1, 1, 8, 8).astype(np.float32)
    clean_s = vscale.transform(clean).reshape(-1, 1, 8, 8).astype(np.float32)
    synth_s = vscale.transform(synth).reshape(-1, 1, 8, 8).astype(np.float32)
    outputs = [raw_s, clean_s, lscale.transform(pose),
               synth_s, lscale.transform(synth_pose)]
    for arr_path, arr_data in zip(arrays, outputs):
        np.save(arr_path, arr_data)

    np.savez(index_path, train_a=train_a, val_a=val_a, train_b=train_b, val_b=val_b)
    with open(cache_dir / "scalers.pkl", "wb") as f:
        pickle.dump(scalers, f)
    # The manifest acts as a commit marker — write it last so a partial
    # rebuild is detected as invalid on the next run.
    _save_json(manifest_path, expected)
    print(f"[cache] Done. raw={len(raw):,}  synth={len(synth):,}  cache={cache_dir}")
    return list(arrays), np.load(index_path), scalers


def _unwrap(model):
    """Return the original nn.Module, unwrapping torch.compile if needed."""
    return getattr(model, "_orig_mod", model)


def _strip(state):
    """Remove the '_orig_mod.' prefix that torch.compile adds to state dict keys."""
    return {key.replace("_orig_mod.", ""): value for key, value in state.items()}


def load_component(module, path, name, device):
    """Load weights for a single sub-module (calibnet or locnet) from a checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    raw_state = checkpoint.get(
        f"{name}_state_dict",
        checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint)),
    )
    state = _strip(raw_state)
    # Checkpoints saved from the finetune phase prefix locnet keys with "locnet."
    if name == "locnet" and any(key.startswith("locnet.") for key in state):
        state = {key[7:]: value for key, value in state.items() if key.startswith("locnet.")}
    missing, unexpected = module.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  [warn] Loading {name}: missing={list(missing)}, unexpected={list(unexpected)}")
    return checkpoint


def save_checkpoint(path, phase, epoch, model, optimizer, scheduler, best_val, metrics, scalers):
    model = _unwrap(model)
    data = {
        "phase":     phase,
        "epoch":     epoch,
        "best_val":  best_val,
        "metrics":   metrics,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "preprocessing": {
            "voltage_space": "minmax_[0,1]",
            "sensor_order":  "sensor_1..sensor_64 row-major 8x8",
            "scaler_file":   "scalers.pkl",
        },
        "scaler_metadata": _scaler_signature(scalers),
    }
    if phase == "calibnet":
        data["calibnet_state_dict"] = model.calibnet.state_dict()
    elif phase == "locnet":
        data["locnet_state_dict"] = model.locnet.state_dict()
    else:  # finetune — save everything
        data["model_state_dict"]    = model.state_dict()
        data["calibnet_state_dict"] = model.calibnet.state_dict()
        data["locnet_state_dict"]   = model.locnet.state_dict()
    torch.save(data, path)


def append_log(path: Path, entry: dict):
    try:
        history = json.loads(path.read_text()) if path.is_file() else []
    except json.JSONDecodeError:
        history = []
    history.append(entry)
    _save_json(path, history)


def plot_losses(history: list[dict], path: Path):
    """Save a 2-panel loss plot (total/loc/calib + voltage MAE) to *path*."""
    if not history:
        return
    epochs = [x["epoch"] for x in history]
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for key, label in (("total", "Total"), ("loc", "Localization"), ("calib", "Calibration")):
        axes[0].plot(epochs, [x["train"][key] for x in history], label=f"Train {label}")
        axes[0].plot(epochs, [x["val"][key]   for x in history], linestyle="--", label=f"Val {label}")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend(ncol=2)

    # MAE is only meaningful for calibnet / finetune phases.
    axes[1].plot(epochs, [x["train"]["mae"] for x in history], label="Train CalibNet correction MAE")
    axes[1].plot(epochs, [x["val"]["mae"]   for x in history], label="Val   CalibNet correction MAE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Scaled voltage MAE")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


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
    cfg = get_config()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if not 0 < cfg.val_ratio < 1:
        raise ValueError("--val-ratio must be in (0, 1)")

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir  = Path(cfg.ckpt_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cfg.cache_dir) if cfg.cache_dir else run_dir / "data_cache"
    if cfg.num_workers is None:
        cfg.num_workers = default_num_workers(device)

    # ------------------------------------------------------------------ #
    # 1. Prepare data cache and scalers                                    #
    # ------------------------------------------------------------------ #
    arrays, split, scalers = prepare_cache(cfg, cache_dir, cfg.scaler_file)
    # Persist scalers alongside checkpoints so test_mvec.py can load them.
    with open(run_dir / "scalers.pkl", "wb") as f:
        pickle.dump(scalers, f)
    np.savez(run_dir / "split_info_three_phase.npz",
             **{key: split[key] for key in split.files})

    # ------------------------------------------------------------------ #
    # 2. Build datasets for the active phase                               #
    # ------------------------------------------------------------------ #
    # Phase      | Input arrays         | Split keys
    # -----------|----------------------|-----------
    # calibnet   | raw, clean           | train_a / val_a
    # locnet     | synthetic, synth_pose| train_b / val_b
    # finetune   | raw, clean, pose     | train_a / val_a
    if cfg.phase == "calibnet":
        train_ds = MultiMemmapDataset(arrays[:2], split["train_a"])
        val_ds   = MultiMemmapDataset(arrays[:2], split["val_a"])
    elif cfg.phase == "locnet":
        train_ds = MultiMemmapDataset([arrays[3], arrays[4]], split["train_b"])
        val_ds   = MultiMemmapDataset([arrays[3], arrays[4]], split["val_b"])
    else:  # finetune
        train_ds = MultiMemmapDataset(arrays[:3], split["train_a"])
        val_ds   = MultiMemmapDataset(arrays[:3], split["val_a"])

    train_loader, val_loader = make_loaders(
        train_ds, val_ds, cfg.batch_size, cfg.num_workers,
        device, cfg.prefetch_factor, cfg.samples_per_epoch,
    )

    # ------------------------------------------------------------------ #
    # 3. Build model                                                        #
    # ------------------------------------------------------------------ #
    # LocNet-only phase: CalibNet is not needed (keeps separate for clarity)
    base_model = Model(use_calibnet=(cfg.phase != "locnet")).to(device)

    if cfg.phase == "finetune":
        if not (cfg.calibnet_checkpoint and cfg.locnet_checkpoint):
            raise ValueError("--phase finetune requires both --calibnet-checkpoint and --locnet-checkpoint")
        load_component(base_model.calibnet, cfg.calibnet_checkpoint, "calibnet", device)
        load_component(base_model.locnet,   cfg.locnet_checkpoint,   "locnet",   device)

    # ------------------------------------------------------------------ #
    # 4. Optimizer and scheduler                                           #
    # ------------------------------------------------------------------ #
    if cfg.phase == "calibnet":
        params = [{"params": base_model.calibnet.parameters(), "lr": cfg.lr_calibnet}]
    elif cfg.phase == "locnet":
        params = [{"params": base_model.locnet.parameters(), "lr": cfg.lr_locnet}]
    else:  # finetune — separate LRs for the two sub-networks
        params = [
            {"params": base_model.calibnet.parameters(), "lr": cfg.lr_calibnet},
            {"params": base_model.locnet.parameters(),   "lr": cfg.lr_locnet},
        ]

    optimizer = torch.optim.AdamW(params, weight_decay=cfg.weight_decay)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0,
        total_iters=max(1, cfg.warmup_epochs),
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, cfg.num_epochs - cfg.warmup_epochs), eta_min=1e-6,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], milestones=[cfg.warmup_epochs],
    )

    # ------------------------------------------------------------------ #
    # 5. Loss functions                                                     #
    # ------------------------------------------------------------------ #
    # Physics loss is disabled for calibnet (it does not predict poses)
    # and can be disabled globally with --no-physics.
    physics = 0.0 if (cfg.no_physics or cfg.phase == "calibnet") else cfg.lambda_physics
    criterion = HuberPoseLossMVec(
        cfg.lambda_ori, cfg.delta_xyz, cfg.lambda_pos,
        physics, cfg.physics_delta,
        cfg.calib_physical_csv if physics else None,
        cfg.calib_alpha_csv    if physics else None,
        scalers["volt"]        if physics else None,
        scalers["label"]       if physics else None,
    ).to(device)
    pipeline_criterion = CalibLocLoss(
        criterion, cfg.lambda_calib, cfg.calib_delta,
    ).to(device)

    # ------------------------------------------------------------------ #
    # 6. Optional torch.compile                                            #
    # ------------------------------------------------------------------ #
    model = base_model
    if cfg.compile and device.type == "cuda" and platform.system() != "Windows":
        try:
            model = torch.compile(base_model)
            print("torch.compile enabled")
        except Exception as exc:
            print(f"torch.compile unavailable: {exc}")

    use_amp = device.type == "cuda"
    amp_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ------------------------------------------------------------------ #
    # 7. Optional resume                                                   #
    start_epoch, best = 1, float("inf")
    log_path = run_dir / f"train_log_{cfg.phase}.json"
    if not cfg.resume:
        log_path.unlink(missing_ok=True)
    if cfg.resume:
        saved = torch.load(cfg.resume, map_location=device, weights_only=False)
        if cfg.phase == "calibnet":
            load_component(base_model.calibnet, cfg.resume, "calibnet", device)
        elif cfg.phase == "locnet":
            load_component(base_model.locnet, cfg.resume, "locnet", device)
        else:
            base_model.load_state_dict(_strip(saved["model_state_dict"]), strict=False)
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        scheduler.load_state_dict(saved["scheduler_state_dict"])
        start_epoch = saved["epoch"] + 1
        best        = saved["best_val"]

    ckpt_names = {
        "calibnet": ("calibnet_pretrained.pt", "calibnet_last.pt"),
        "locnet":   ("locnet_pretrained.pt",   "locnet_last.pt"),
        "finetune": ("full_model_best.pt",     "full_model_last.pt"),
    }
    best_path, last_path = (run_dir / n for n in ckpt_names[cfg.phase])
    no_improve = 0

    # ------------------------------------------------------------------ #
    # 8. Per-epoch training loop                                           #
    # ------------------------------------------------------------------ #
    def run_epoch(loader, training: bool) -> dict:
        model.train(training) if training else model.eval()
        totals = {k: 0.0 for k in ("total", "loc", "calib", "mae", "physics")}
        seen   = 0
        ctx    = torch.enable_grad() if training else torch.no_grad()

        with ctx:
            for batch in loader:
                batch = [x.to(device, non_blocking=True) for x in batch]
                if training:
                    optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    # ── Phase 1: CalibNet only ───────────────────────────
                    if cfg.phase == "calibnet":
                        corrected = base_model.calibnet(batch[0])   # batch: [raw, clean]
                        calib     = pipeline_criterion.calibration_term(corrected, batch[1])
                        loc       = calib * 0
                        loss      = calib

                    # ── Phase 2: LocNet only (synthetic data) ────────────
                    elif cfg.phase == "locnet":
                        corrected = batch[0]                         # batch: [synth_volt, synth_label]
                        pred      = base_model.locnet(corrected)
                        loc, _, _ = criterion(pred, batch[1], X_b=corrected)
                        calib     = loc * 0
                        loss      = loc

                    # ── Phase 3: End-to-end finetune ─────────────────────
                    else:
                        pred, aux = model(batch[0], return_features=True)  # batch: [raw, clean, pose]
                        corrected = aux["corrected"]
                        loss, calib, loc, _, _ = pipeline_criterion(
                            pred, batch[2], corrected, batch[1]
                        )

                if not torch.isfinite(loss):
                    continue

                if training:
                    amp_scaler.scale(loss).backward()
                    amp_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(base_model.parameters(), 1.0)
                    amp_scaler.step(optimizer)
                    amp_scaler.update()

                n = len(batch[0])
                totals["total"]   += loss.item()  * n
                totals["loc"]     += loc.item()   * n
                totals["calib"]   += calib.item() * n
                totals["physics"] += criterion.latest_loss_physics.item() * n
                # CalibNet correction MAE: only valid when CalibNet actually ran.
                # In locnet phase corrected == batch[0], so the difference is always
                # 0 and would silently mislead the loss plot.
                if cfg.phase != "locnet":
                    totals["mae"] += (corrected - batch[1]).abs().mean().item() * n
                seen += n

        metrics = {k: v / max(1, seen) for k, v in totals.items()}
        if cfg.phase == "locnet":
            metrics["mae"] = float("nan")  # not applicable
        return metrics

    # ------------------------------------------------------------------ #
    # 9. Epoch loop                                                        #
    # ------------------------------------------------------------------ #
    print(f"\n{'='*65}")
    print(f"  Phase : {cfg.phase}")
    print(f"  Train : {len(train_ds):,} samples")
    print(f"  Val   : {len(val_ds):,} samples")
    print(f"  Device: {device}   Cache: {cache_dir}")
    print(f"{'='*65}\n")

    for epoch in range(start_epoch, cfg.num_epochs + 1):
        t0           = time.time()
        train_m      = run_epoch(train_loader, training=True)
        val_m        = run_epoch(val_loader,   training=False)
        scheduler.step()

        entry = {
            "epoch":   epoch,
            "train":   train_m,
            "val":     val_m,
            "lr":      [g["lr"] for g in optimizer.param_groups],
            "seconds": time.time() - t0,
        }
        append_log(log_path, entry)
        print(
            f"{epoch:03d} "
            f"train={train_m['total']:.6f}  val={val_m['total']:.6f}  "
            f"loc={val_m['loc']:.6f}  calib={val_m['calib']:.6f}  "
            f"phys={val_m['physics']:.6f}  ({entry['seconds']:.1f}s)"
        )

        # Always save the last checkpoint so training can be resumed.
        save_checkpoint(last_path, cfg.phase, epoch, base_model,
                        optimizer, scheduler, best, entry, scalers)

        if val_m["total"] < best:
            best, no_improve = val_m["total"], 0
            save_checkpoint(best_path, cfg.phase, epoch, base_model,
                            optimizer, scheduler, best, entry, scalers)
            print(f"  -> best saved: {best:.6f}")
        else:
            no_improve += 1

        if epoch % cfg.save_every == 0:
            save_checkpoint(
                run_dir / f"{cfg.phase}_epoch_{epoch:04d}.pt",
                cfg.phase, epoch, base_model, optimizer, scheduler, best, entry, scalers,
            )

        if no_improve >= cfg.patience:
            print(f"Early stopping after {cfg.patience} unimproved epochs.")
            break

    # ------------------------------------------------------------------ #
    # 10. Post-training: plot and optional benchmark                       #
    # ------------------------------------------------------------------ #
    plot_losses(history, run_dir / f"loss_plot_{cfg.phase}.png")
    plot_losses(history, run_dir / "loss_plot.png")

    if cfg.benchmark_samples > 0 and best_path.is_file():
        saved = torch.load(best_path, map_location=device, weights_only=False)
        if cfg.phase == "calibnet":
            load_component(base_model.calibnet, best_path, "calibnet", device)
        elif cfg.phase == "locnet":
            load_component(base_model.locnet, best_path, "locnet", device)
        else:
            base_model.load_state_dict(_strip(saved["model_state_dict"]), strict=False)
        results = benchmark(model, device, cfg.benchmark_samples)
        _save_json(run_dir / "inference_time.json", results)
        print(f"Inference: {results['mean_ms']:.3f} ms/sample  (p95 {results['p95_ms']:.3f} ms)")

    print(f"\nDone. Best validation loss: {best:.6f}  |  checkpoint: {best_path}")


if __name__ == "__main__":
    main()