import optuna
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import json, os, argparse, pickle
from train_mvec import build_datasets
from model import Model
from loss_mvec import HuberPoseLossMVec
from torch.utils.data import DataLoader

# =============================================================================
# CONFIG
# =============================================================================

CKPT_DIR = "/content/drive/MyDrive/training_ckpt"

p = argparse.ArgumentParser()
p.add_argument("--VOLTAGE",       type=str, default="grid_calib_data.csv")
p.add_argument("--LABEL",         type=str, default="Grid_points_coordinates.csv")
p.add_argument("--calib_csv",     type=str, default="Calibration_PARAM.csv")
p.add_argument("--ckpt_dir",      type=str, default=CKPT_DIR)
p.add_argument("--n_trials",      type=int, default=50)
p.add_argument("--max_epochs",    type=int, default=80,     # [FIX-3] giảm từ 200 xuống 80 cho HPO
               help="Epochs per trial (dùng giá trị nhỏ hơn khi tuning)")
p.add_argument("--warmup_epochs", type=int, default=5)
p.add_argument("--num_workers",   type=int, default=2,      # [FIX-4] prefetch parallel
               help="DataLoader num_workers")

args = p.parse_args()

CKPT_DIR = args.ckpt_dir
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP  = (DEVICE.type == "cuda")                           # [FIX-2] flag AMP toàn cục

os.makedirs(CKPT_DIR, exist_ok=True)

STUDY_DB   = os.path.join(CKPT_DIR, "study_mvec.db")
STUDY_NAME = "pose_hpo_mvec"


# =============================================================================
# [FIX-1 + FIX-5 + FIX-6]
# Các phần cố định (dataset, scalers, calib) được chuẩn bị 1 lần duy nhất
# trước khi Optuna chạy, rồi truyền vào objective() qua closure.
# =============================================================================

def prepare_shared_resources():
    """Chạy 1 lần: đọc CSV, fit scaler, tạo dataset."""
    scaler_file = os.path.join(CKPT_DIR, "scalers.pkl")

    print("\n[HPO] Preparing shared dataset (once for all trials)...")
    train_ds, val_ds, n_train, n_val = build_datasets(
        args.VOLTAGE,
        args.LABEL,
        val_ratio=0.2,
        scaler_file=scaler_file,
        seed=42,
    )

    # [FIX-6] đọc scalers 1 lần
    with open(scaler_file, "rb") as f:
        sc = pickle.load(f)
    volt_scaler  = sc["volt"]
    label_scaler = sc["label"]

    print(f"[HPO] Dataset ready — train: {n_train:,}  val: {n_val:,}")
    return train_ds, val_ds, n_train, n_val, volt_scaler, label_scaler


# =============================================================================
# OBJECTIVE
# =============================================================================

def make_objective(train_ds, val_ds, n_train, n_val, volt_scaler, label_scaler):
    """Trả về closure đã capture shared resources — không đọc lại CSV."""

    def objective(trial):
        lr             = trial.suggest_float("lr",             1e-5, 1e-2,  log=True)
        weight_decay   = trial.suggest_float("weight_decay",   1e-5, 1e-2,  log=True)
        lambda_ori     = trial.suggest_float("lambda_ori",     0.01, 3.0,   log=True)
        delta_xyz      = trial.suggest_float("delta_xyz",      0.01, 1.5)
        lambda_pos     = trial.suggest_float("lambda_pos",     0.1,  3.0,   log=True)
        lambda_physics = trial.suggest_float("lambda_physics", 1e-3, 0.5,   log=True)
        physics_delta  = trial.suggest_float("physics_delta",  1e-3, 0.2,   log=True)
        batch_size     = trial.suggest_categorical("batch_size", [32, 64])

        pin = (DEVICE.type == "cuda")
        # [FIX-4] pin_memory + num_workers
        train_loader = DataLoader(
            train_ds, batch_size=batch_size,
            shuffle=True, drop_last=True,
            pin_memory=pin, num_workers=args.num_workers,
            persistent_workers=(args.num_workers > 0),
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size,
            shuffle=False,
            pin_memory=pin, num_workers=args.num_workers,
            persistent_workers=(args.num_workers > 0),
        )

        model = Model(out_dim=6).to(DEVICE)

        # [FIX-5] calib_csv / scalers chỉ dùng để tạo buffers cố định;
        #          chỉ lambda/delta thay đổi theo trial
        criterion = HuberPoseLossMVec(
            lambda_ori     = lambda_ori,
            delta_xyz      = delta_xyz,
            lambda_pos     = lambda_pos,
            lambda_physics = lambda_physics,
            physics_delta  = physics_delta,
            calib_csv      = args.calib_csv,
            volt_scaler    = volt_scaler,
            label_scaler   = label_scaler,
        ).to(DEVICE)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay)

        warmup_sch = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0,
            total_iters=args.warmup_epochs)
        cosine_sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, args.max_epochs - args.warmup_epochs),
            eta_min=1e-6)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_sch, cosine_sch],
            milestones=[args.warmup_epochs])

        # [FIX-2] AMP GradScaler
        amp_scaler = GradScaler(enabled=USE_AMP)

        best_val   = float("inf")
        no_improve = 0

        for epoch in range(args.max_epochs):

            # ── Train ────────────────────────────────────────────────────────
            model.train()
            for X_b, Y_b in train_loader:
                X_b = X_b.to(DEVICE, non_blocking=True)
                Y_b = Y_b.to(DEVICE, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                # [FIX-2] autocast bao quanh forward pass
                with autocast(enabled=USE_AMP):
                    pred = model(X_b)
                    loss, _, _ = criterion(pred, Y_b, X_b=X_b)

                if not torch.isfinite(loss):
                    optimizer.zero_grad(set_to_none=True)
                    continue

                amp_scaler.scale(loss).backward()
                amp_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                amp_scaler.step(optimizer)
                amp_scaler.update()

            scheduler.step()

            # ── Validation ───────────────────────────────────────────────────
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_b, Y_b in val_loader:
                    X_b = X_b.to(DEVICE, non_blocking=True)
                    Y_b = Y_b.to(DEVICE, non_blocking=True)
                    with autocast(enabled=USE_AMP):         # [FIX-2]
                        pred = model(X_b)
                        loss, _, _ = criterion(pred, Y_b, X_b=X_b)
                    if torch.isfinite(loss):
                        val_loss += loss.item() * len(X_b)
            val_loss /= n_val

            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            if val_loss < best_val:
                best_val   = val_loss
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= 10:
                break

        return best_val

    return objective


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    # [FIX-1 + FIX-6] chuẩn bị tài nguyên dùng chung — chỉ chạy 1 lần
    train_ds, val_ds, n_train, n_val, volt_scaler, label_scaler = \
        prepare_shared_resources()

    # Tạo objective closure đã capture dataset + scalers
    objective = make_objective(
        train_ds, val_ds, n_train, n_val, volt_scaler, label_scaler)

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=10,
    )
    sampler = optuna.samplers.TPESampler(seed=42)
    storage = f"sqlite:///{os.path.abspath(STUDY_DB)}"

    try:
        loaded_study = optuna.load_study(study_name=STUDY_NAME, storage=storage)
        print("\n" + "=" * 60)
        print("FOUND PREVIOUS BEST HYPERPARAMETERS")
        print("=" * 60)
        print(f"Best trial value (loss): {loaded_study.best_trial.value:.6f}")
        print("Best parameters:")
        for k, v in loaded_study.best_params.items():
            print(f"  {k:20s}: {v}")
        print("=" * 60 + "\n")
    except KeyError:
        print(f"\nNo previous study '{STUDY_NAME}' found. Starting fresh.\n")
    except Exception as e:
        print(f"\nNo previous trials loaded: {e}. Starting fresh.\n")

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        study_name=STUDY_NAME,
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    best = study.best_trial

    print("\n" + "=" * 60)
    print("BEST RESULT")
    print("=" * 60)
    print(f"Best validation loss: {best.value:.6f}\n")
    print("Best hyperparameters:")
    for k, v in best.params.items():
        print(f"  {k:20s}: {v}")
    print("=" * 60)

    result_path = os.path.join(CKPT_DIR, "best_hparams.json")
    with open(result_path, "w") as f:
        json.dump({"val_loss": best.value, "params": best.params}, f, indent=2)
    print(f"\nSaved best params to: {result_path}")