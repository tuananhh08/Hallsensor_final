"""Optuna HPO for the CalibNet -> LocNet three-phase pipeline.

Use ``--phase finetune`` after training both subnetworks.  That objective tunes
the learning rates of CalibNet and LocNet independently and optimizes their
joint calibration + localization + optional physics loss.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import optuna
import torch
import torch.nn.functional as F

from loss_mvec import CalibLocLoss, HuberPoseLossMVec
from model import Model
from train_mvec import (MultiMemmapDataset, load_component, prepare_cache)
from training_loop import default_num_workers, make_loaders


def get_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent / "Dataset" / "Dataset"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("calibnet", "locnet", "finetune"), default="finetune")
    parser.add_argument("--raw-voltage", default=str(root / "Grid_data.csv"))
    parser.add_argument("--clean-voltage", default=str(root / "Grid_data_computed.csv"))
    parser.add_argument("--raw-label", default=str(root / "Grid_points_coordinates.csv"))
    parser.add_argument("--synthetic-voltage", default=str(root / "synthetic_grid_data.csv"))
    parser.add_argument("--synthetic-label", default=str(root / "synthetic_grid_coordinates.csv"))
    parser.add_argument("--calib-physical-csv", default=str(root / "Calibration_Physical_new.csv"))
    parser.add_argument("--calib-alpha-csv", default=str(root / "Calibration_Alpha_new.csv"))
    parser.add_argument("--calibnet-checkpoint", default=None,
                        help="Required for --phase finetune; checkpoint from phase calibnet.")
    parser.add_argument("--locnet-checkpoint", default=None,
                        help="Required for --phase finetune; checkpoint from phase locnet.")
    parser.add_argument("--ckpt-dir", default="./ckpt_mvec_hpo")
    parser.add_argument("--local-dir", default="/content/tune_hyperparams")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--scaler-file", default=None)
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--trial-patience", type=int, default=6)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--no-physics", action="store_true")
    parser.add_argument("--samples-per-epoch", type=int, default=0,
                        help="Randomly subsample this many training examples per epoch during "
                             "the search (0 = use the full training set every epoch, like before). "
                             "HPO only needs a relative ranking between configs, so a subsample "
                             "(e.g. 50000-100000) speeds up every trial x every epoch without "
                             "changing the search space. The held-out validation set is always "
                             "used in full so trial comparisons stay fair.")
    return parser.parse_args()


def sync_from_drive(local_path: Path, drive_path: Path) -> None:
    if drive_path.is_file() and not local_path.is_file():
        shutil.copy2(drive_path, local_path)


def sync_to_drive(local_path: Path, drive_path: Path) -> None:
    if local_path.is_file():
        shutil.copy2(local_path, drive_path)


def make_pose_loss(args, scalers, device, trial) -> HuberPoseLossMVec:
    if args.phase == "calibnet":
        return HuberPoseLossMVec(lambda_physics=0.0).to(device)
    physics_weight = 0.0 if args.no_physics or args.phase == "calibnet" else trial.suggest_float(
        "lambda_physics", 1e-5, 1e-2, log=True)
    return HuberPoseLossMVec(
        lambda_ori=trial.suggest_float("lambda_ori", 0.03, 3.0, log=True),
        delta_xyz=trial.suggest_float("delta_xyz", 0.01, 0.25),
        lambda_pos=trial.suggest_float("lambda_pos", 0.1, 3.0, log=True),
        lambda_physics=physics_weight,
        physics_delta=trial.suggest_float("physics_delta", 1e-4, 1e-2, log=True),
        calib_physical_csv=args.calib_physical_csv if physics_weight else None,
        calib_alpha_csv=args.calib_alpha_csv if physics_weight else None,
        volt_scaler=scalers["volt"] if physics_weight else None,
        label_scaler=scalers["label"] if physics_weight else None,
    ).to(device)


def run_epoch(model, loader, phase, pose_loss, pipeline_loss, optimizer, device) -> tuple[float, int]:
    training = optimizer is not None
    model.train(training)
    total, seen = 0.0, 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            batch = [value.to(device, non_blocking=True) for value in batch]
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                if phase == "calibnet":
                    corrected = model.modnet(batch[0])
                    loss = pipeline_loss.calibration_term(corrected, batch[1])
                elif phase == "locnet":
                    pred = model.locnet(batch[0])
                    loss, _, _ = pose_loss(pred, batch[1], X_b=batch[0])
                else:
                    pred, features = model(batch[0], return_features=True)
                    loss, _, _, _, _ = pipeline_loss(pred, batch[2], features["corrected"], batch[1])
            if not torch.isfinite(loss):
                continue
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            batch_size = len(batch[0])
            total += loss.detach().item() * batch_size
            seen += batch_size
    return (total / seen if seen else float("inf")), seen


def main() -> None:
    args = get_args()
    if args.phase == "finetune" and not (args.calibnet_checkpoint and args.locnet_checkpoint):
        raise ValueError("--phase finetune requires --calibnet-checkpoint and --locnet-checkpoint")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.num_workers = default_num_workers(device) if args.num_workers is None else args.num_workers
    ckpt_dir, local_dir = Path(args.ckpt_dir), Path(args.local_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True); local_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else local_dir / "data_cache"
    arrays, split, scalers = prepare_cache(args, cache_dir, args.scaler_file)
    if args.phase == "calibnet":
        train_ds, val_ds = MultiMemmapDataset(arrays[:2], split["train_a"]), MultiMemmapDataset(arrays[:2], split["val_a"])
    elif args.phase == "locnet":
        train_ds, val_ds = MultiMemmapDataset(arrays[3:], split["train_b"]), MultiMemmapDataset(arrays[3:], split["val_b"])
    else:
        train_ds, val_ds = MultiMemmapDataset(arrays[:3], split["train_a"]), MultiMemmapDataset(arrays[:3], split["val_a"])
    local_db, drive_db = local_dir / f"study_{args.phase}.db", ckpt_dir / f"study_{args.phase}.db"
    sync_from_drive(local_db, drive_db)

    def objective(trial: optuna.Trial) -> float:
        # Make each trial reproducible while still giving separate trials a
        # deterministic initialization and loader shuffle sequence.
        trial_seed = args.seed + trial.number
        torch.manual_seed(trial_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(trial_seed)
        batch_size = trial.suggest_categorical("batch_size", [256, 512])
        train_loader, val_loader = make_loaders(
            train_ds, val_ds, batch_size, args.num_workers, device, args.prefetch_factor,
            args.samples_per_epoch,
        )
        model = Model(use_modnet=args.phase != "locnet").to(device)
        if args.phase == "finetune":
            load_component(model.modnet, args.calibnet_checkpoint, "modnet", device)
            load_component(model.locnet, args.locnet_checkpoint, "locnet", device)
        pose_loss = make_pose_loss(args, scalers, device, trial)
        if args.phase == "finetune":
            lambda_calib = trial.suggest_float("lambda_calib", 1e-3, 1.0, log=True)
            calib_delta = trial.suggest_float("calib_delta", 0.005, 0.2)
        else:
            lambda_calib, calib_delta = 1.0, 0.05
        pipeline_loss = CalibLocLoss(pose_loss, lambda_calib=lambda_calib, calib_delta=calib_delta).to(device)
        if args.phase == "calibnet":
            parameters = [{"params": model.modnet.parameters(), "lr": trial.suggest_float("lr_calibnet", 1e-5, 3e-3, log=True)}]
        elif args.phase == "locnet":
            parameters = [{"params": model.locnet.parameters(), "lr": trial.suggest_float("lr_locnet", 1e-5, 3e-3, log=True)}]
        else:
            parameters = [
                {"params": model.modnet.parameters(), "lr": trial.suggest_float("lr_calibnet", 1e-5, 1e-3, log=True)},
                {"params": model.locnet.parameters(), "lr": trial.suggest_float("lr_locnet", 1e-5, 3e-3, log=True)},
            ]
        optimizer = torch.optim.AdamW(parameters, weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.max_epochs), eta_min=1e-6)
        best, stagnant = float("inf"), 0
        try:
            for epoch in range(args.max_epochs):
                run_epoch(model, train_loader, args.phase, pose_loss, pipeline_loss, optimizer, device)
                val_loss, seen = run_epoch(model, val_loader, args.phase, pose_loss, pipeline_loss, None, device)
                if not seen:
                    raise FloatingPointError("No finite validation samples")
                scheduler.step(); trial.report(val_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
                if val_loss < best:
                    best, stagnant = val_loss, 0
                else:
                    stagnant += 1
                    if stagnant >= args.trial_patience:
                        break
            return best
        finally:
            del train_loader, val_loader, model, pose_loss, pipeline_loss, optimizer
            if device.type == "cuda": torch.cuda.empty_cache()

    study = optuna.create_study(
        direction="minimize", study_name=f"calibloc_{args.phase}",
        storage=f"sqlite:///{local_db.resolve()}", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=4),
    )
    started = time.time()
    try:
        study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)
    finally:
        sync_to_drive(local_db, drive_db)
    if not study.best_trials:
        raise RuntimeError("No completed Optuna trials")
    result = {"phase": args.phase, "val_loss": study.best_value, "params": study.best_params,
              "cache_dir": str(cache_dir), "n_train": len(train_ds), "n_val": len(val_ds)}
    (ckpt_dir / f"best_hparams_{args.phase}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Best validation loss: {study.best_value:.6f}; elapsed {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()