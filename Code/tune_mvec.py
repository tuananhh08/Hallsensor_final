import optuna
import torch
import json, os, argparse, pickle
from train_mvec import build_datasets
from model import Model
from loss_mvec import HuberPoseLossMVec
from torch.utils.data import DataLoader

# ================= CONFIG =================

CKPT_DIR = "./ckpt_mvec"

p = argparse.ArgumentParser()
p.add_argument("--VOLTAGE",    type=str, default="grid_calib_data.csv")
p.add_argument("--LABEL",      type=str, default="Grid_points_coordinates.csv")
p.add_argument("--calib_csv",  type=str, default="Calibration_GRID_NEW_PARAM_results.csv",
               help="Path to Calibration_GRID_NEW_PARAM_results.csv")
p.add_argument("--ckpt_dir",   type=str, default=CKPT_DIR)
p.add_argument("--n_trials",   type=int, default=50)
p.add_argument("--max_epochs", type=int, default=200)
p.add_argument("--warmup_epochs", type=int, default=5,
               help="Warmup epochs")

args = p.parse_args()

CKPT_DIR = args.ckpt_dir
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(CKPT_DIR, exist_ok=True)

STUDY_DB   = os.path.join(CKPT_DIR, "study_mvec.db")
STUDY_NAME = "pose_hpo_mvec"


# ================= OBJECTIVE =================
def objective(trial):

    lr             = trial.suggest_float("lr",             1e-5, 1e-2,  log=True)
    weight_decay   = trial.suggest_float("weight_decay",   1e-5, 1e-2,  log=True)
    lambda_ori     = trial.suggest_float("lambda_ori",     0.01, 3.0,   log=True)
    delta_xyz      = trial.suggest_float("delta_xyz",      0.01, 1.5)
    lambda_pos     = trial.suggest_float("lambda_pos",     0.1,  3.0,   log=True)
    lambda_physics = trial.suggest_float("lambda_physics", 1e-5, 1e-2,  log=True)
    batch_size     = trial.suggest_categorical("batch_size", [256,512])

    scaler_file = os.path.join(CKPT_DIR, "scalers.pkl")

    train_ds, val_ds, n_train, n_val = build_datasets(
        args.VOLTAGE,
        args.LABEL,
        val_ratio=0.2,
        scaler_file=scaler_file,
        seed=42,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False)

    with open(scaler_file, "rb") as f:
        sc = pickle.load(f)
    volt_scaler  = sc["volt"]
    label_scaler = sc["label"]

    model = Model(out_dim=6)
    model = model.to(DEVICE)

    criterion = HuberPoseLossMVec(
        lambda_ori     = lambda_ori,
        delta_xyz      = delta_xyz,
        lambda_pos     = lambda_pos,
        lambda_physics = lambda_physics,
        calib_csv      = args.calib_csv,
        volt_scaler    = volt_scaler,
        label_scaler   = label_scaler,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    warmup_sch = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0,
        total_iters=args.warmup_epochs)
    cosine_sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.max_epochs - args.warmup_epochs,
        eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_sch, cosine_sch],
        milestones=[args.warmup_epochs])

    best_val   = float("inf")
    no_improve = 0

    for epoch in range(args.max_epochs):

        model.train()
        for X_b, Y_b in train_loader:
            X_b, Y_b = X_b.to(DEVICE), Y_b.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = criterion(model(X_b), Y_b, X_b=X_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_b, Y_b in val_loader:
                loss, _, _ = criterion(
                    model(X_b.to(DEVICE)),
                    Y_b.to(DEVICE),
                    X_b=X_b.to(DEVICE),
                )
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


# ================= MAIN =================
if __name__ == "__main__":

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=10,
    )
    sampler = optuna.samplers.TPESampler(seed=42)

    storage = f"sqlite:///{os.path.abspath(STUDY_DB)}"

    try:
        loaded_study = optuna.load_study(
            study_name=STUDY_NAME,
            storage=storage,
        )
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
