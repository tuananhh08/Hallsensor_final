"""Optuna tuning for the m-vector model using the shared on-disk cache."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time

import optuna
import torch

from data_mvec import build_datasets, copy_cache_metadata, load_scalers
from model import Model
from training_loop import (default_num_workers, make_criterion, make_loaders,
                           train_one_epoch, validate)

DEFAULT_CKPT_DIR = "/content/drive/MyDrive/training_ckpt"
DEFAULT_LOCAL_DIR = "/content/tune_hyperparams"


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--voltage", "--VOLTAGE", dest="voltage", default="../Data set 18.6/synthetic_grid_data.csv")
    p.add_argument("--label", "--LABEL", dest="label", default="../Data set 18.6/synthetic_grid_coordinates.csv")
    p.add_argument("--calib_physical_csv", default="../Data set 18.6/Calibration_Physical_new.csv")
    p.add_argument("--calib_alpha_csv", default="../Data set 18.6/Calibration_Alpha_new.csv")
    p.add_argument("--ckpt_dir", default=DEFAULT_CKPT_DIR)
    p.add_argument("--local_dir", default=DEFAULT_LOCAL_DIR)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--rebuild_cache", action="store_true")
    p.add_argument("--n_trials", type=int, default=40)
    p.add_argument("--max_epochs", type=int, default=20)
    p.add_argument("--trial_patience", type=int, default=8)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--prefetch_factor", type=int, default=2)
    p.add_argument("--val_every_n_epochs", type=int, default=1)
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    return p.parse_args()


def sync_from_drive(local_db, drive_db):
    if os.path.exists(drive_db) and not os.path.exists(local_db):
        shutil.copy2(drive_db, local_db)


def sync_to_drive(local_db, drive_db):
    if os.path.exists(local_db):
        shutil.copy2(local_db, drive_db)


def main():
    args = get_args()
    if args.val_every_n_epochs < 1:
        raise ValueError("val_every_n_epochs must be >= 1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.num_workers = default_num_workers(device) if args.num_workers is None else args.num_workers
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.local_dir, exist_ok=True)
    cache_dir = args.cache_dir or os.path.join(args.local_dir, "data_cache")
    local_db, drive_db = os.path.join(args.local_dir, "study_mvec.db"), os.path.join(args.ckpt_dir, "study_mvec.db")
    sync_from_drive(local_db, drive_db)

    train_ds, val_ds, n_train, n_val, cache_dir = build_datasets(
        args.voltage, args.label, args.val_ratio, args.seed, cache_dir, args.rebuild_cache)
    scalers = load_scalers(cache_dir)
    copy_cache_metadata(cache_dir, args.ckpt_dir)
    print(f"[HPO] cache ready: train={n_train:,}, val={n_val:,}, workers={args.num_workers}")

    def objective(trial):
        args.lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        args.weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
        args.lambda_ori = trial.suggest_float("lambda_ori", 0.01, 3.0, log=True)
        args.delta_xyz = trial.suggest_float("delta_xyz", 0.01, 1.5)
        args.lambda_pos = trial.suggest_float("lambda_pos", 0.1, 3.0, log=True)
        args.lambda_physics = trial.suggest_float("lambda_physics", 1e-4, 1e-2, log=True)
        args.physics_delta = trial.suggest_float("physics_delta", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024, 2048])
        train_loader, val_loader = make_loaders(train_ds, val_ds, batch_size, args.num_workers,
                                                device, args.prefetch_factor)
        model = Model(out_dim=6).to(device)
        if args.compile and device.type == "cuda":
            try: model = torch.compile(model)
            except Exception as exc: print(f"[HPO] compile skipped: {exc}")
        criterion = make_criterion(args, scalers, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                                                   total_iters=max(1, args.warmup_epochs))
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.max_epochs - args.warmup_epochs), eta_min=1e-6)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, [warmup, cosine], milestones=[args.warmup_epochs])
        amp_scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        best, stagnant, last_val = float("inf"), 0, None
        try:
            for epoch in range(args.max_epochs):
                train_one_epoch(model, train_loader, criterion, optimizer, amp_scaler, device)
                scheduler.step()
                must_validate = ((epoch + 1) % args.val_every_n_epochs == 0 or epoch + 1 == args.max_epochs)
                if not must_validate:
                    continue
                metrics, seen = validate(model, val_loader, criterion, device)
                last_val = metrics["loss"]
                if seen == 0: raise FloatingPointError("No finite validation samples")
                trial.report(last_val, epoch)
                if trial.should_prune(): raise optuna.exceptions.TrialPruned()
                if last_val < best:
                    best, stagnant = last_val, 0
                else:
                    stagnant += 1
                if stagnant >= args.trial_patience:
                    break
            # In case an early-control change ever skips validation, always return a full validation result.
            if last_val is None:
                last_val = validate(model, val_loader, criterion, device)[0]["loss"]
                best = min(best, last_val)
            return best
        finally:
            # Explicit release prevents persistent workers accumulating across trials.
            del train_loader, val_loader, model, criterion, optimizer
            if device.type == "cuda": torch.cuda.empty_cache()

    storage = f"sqlite:///{os.path.abspath(local_db)}"
    study = optuna.create_study(direction="minimize", study_name="pose_hpo_mvec", storage=storage,
                                load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=args.seed),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5))
    started = time.time()
    try:
        study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)
    finally:
        sync_to_drive(local_db, drive_db)
    if not study.best_trials:
        raise RuntimeError("No completed Optuna trials")
    result = {"val_loss": study.best_trial.value, "params": study.best_trial.params,
              "cache_dir": cache_dir, "n_train": n_train, "n_val": n_val}
    with open(os.path.join(args.ckpt_dir, "best_hparams.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"Best validation loss: {study.best_trial.value:.6f}; elapsed {(time.time()-started)/60:.1f} min")


if __name__ == "__main__":
    main()
