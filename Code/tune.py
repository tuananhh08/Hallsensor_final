import optuna
import torch
import json, os
from train import build_datasets
from model import Model
from loss import HuberPoseLoss
from torch.utils.data import DataLoader

# ================= CONFIG =================
CKPT_DIR    = "/content/drive/MyDrive/training_ckpt"
VOLTAGE     = "/content/drive/MyDrive/Dataset/grid_calib_data.csv"
LABEL       = "/content/drive/MyDrive/Dataset/Grid_points_coordinates.csv"
N_TRIALS    = 60
MAX_EPOCHS  = 50
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(CKPT_DIR, exist_ok=True)


# ================= OBJECTIVE =================
def objective(trial):

    # ---- Search space ----
    lr           = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    ang_weight   = trial.suggest_float("ang_weight", 0.5, 3.0)
    delta_xyz    = trial.suggest_float("delta_xyz", 0.05, 0.2)
    delta_ang    = trial.suggest_float("delta_ang", 0.05, 0.3)
    batch_size   = trial.suggest_categorical("batch_size", [32, 64, 128])

    # ---- Dataset ----
    scaler_file = os.path.join(CKPT_DIR, "scalers.pkl")

    # ⚠️ FIX: bỏ split_block_path vì build_datasets không support
    train_ds, val_ds, n_train, n_val = build_datasets(
        VOLTAGE,
        LABEL,
        val_ratio=0.2,
        scaler_file=scaler_file,
        seed=42
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size,
                              shuffle=False)

    # ---- Model ----
    model = Model(out_dim=5).to(DEVICE)

    criterion = HuberPoseLoss(
        ang_weight=ang_weight,
        delta_xyz=delta_xyz,
        delta_ang=delta_ang
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=1e-6
    )

    # ---- Training ----
    best_val = float("inf")
    no_improve = 0

    for epoch in range(MAX_EPOCHS):

        model.train()
        for X_b, Y_b in train_loader:
            X_b, Y_b = X_b.to(DEVICE), Y_b.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = criterion(model(X_b), Y_b)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        # ---- Validation ----
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for X_b, Y_b in val_loader:
                loss, _, _ = criterion(
                    model(X_b.to(DEVICE)),
                    Y_b.to(DEVICE)
                )
                val_loss += loss.item() * len(X_b)

        val_loss /= n_val

        # ---- Optuna tracking ----
        trial.report(val_loss, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        # ---- Early stopping ----
        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= 10:
            break

    return best_val


# ================= MAIN =================
if __name__ == "__main__":

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=10
    )

    sampler = optuna.samplers.TPESampler(seed=42)

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage="sqlite:////content/drive/MyDrive/training_ckpt/study.db",
        study_name="pose_hpo",
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    # ================= RESULT =================
    best = study.best_trial

    print("\n" + "="*60)
    print("🎯 BEST RESULT")
    print("="*60)

    print(f"Best validation loss: {best.value:.6f}\n")

    print("Best hyperparameters:")
    for k, v in best.params.items():
        print(f"{k:15s}: {v}")

    print("="*60)

    # ---- Save ----
    result_path = os.path.join(CKPT_DIR, "best_hparams.json")

    with open(result_path, "w") as f:
        json.dump({
            "val_loss": best.value,
            "params": best.params
        }, f, indent=2)

    print(f"\nSaved best params to: {result_path}")