# =============================================================================
# SPLIT TRAIN/VAL
# =============================================================================
import os, sys, json, pickle, argparse, time
import uuid
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Model
from loss_calib import HuberPoseLossCalib


# =============================================================================
# CONFIG
# =============================================================================

def get_config():
    p = argparse.ArgumentParser()
    p.add_argument("--voltage",        type=str,   default="grid_calib_data.csv")
    p.add_argument("--label",          type=str,   default="Grid_points_coordinates.csv")
    p.add_argument("--calib_csv",      type=str,   default="Calibration_GRID_NEW_PARAM_results.csv")
    p.add_argument("--ckpt_dir",       type=str,   default="./ckpt2")
    p.add_argument("--val_ratio",      type=float, default=0.2)
    p.add_argument("--batch_size",     type=int,   default=64)
    p.add_argument("--num_epochs",     type=int,   default=200)
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--weight_decay",   type=float, default=0.00456)
    p.add_argument("--ang_weight",     type=float, default=1.0)
    p.add_argument("--delta_xyz",      type=float, default=0.061)
    p.add_argument("--delta_ang",      type=float, default=0.21)
    p.add_argument("--lambda_pos",     type=float, default=1.0)
    p.add_argument("--lambda_physics", type=float, default=1e-4)
    p.add_argument("--warmup_epochs",  type=int,   default=5)
    p.add_argument("--save_every",     type=int,   default=5)
    p.add_argument("--patience",       type=int,   default=45)
    p.add_argument("--seed",           type=int,   default=42)
    return p.parse_args()


# =============================================================================
# DATASET
# =============================================================================

# region agent log helpers
_DBG_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "debug-eb2a46.log")
_DBG_SESSION_ID = "eb2a46"
def _dbg_log(hypothesisId: str, location: str, message: str, data: dict, runId: str = "pre-fix"):
    try:
        payload = {
            "sessionId": _DBG_SESSION_ID,
            "runId": runId,
            "hypothesisId": hypothesisId,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
            "id": f"log_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}",
        }
        with open(_DBG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass
# endregion


class PoseDataset(Dataset):
    def __init__(self, voltages, labels):
        self.X = torch.tensor(voltages, dtype=torch.float32).view(-1, 1, 8, 8)
        self.Y = torch.tensor(labels,   dtype=torch.float32)
    def __len__(self):          return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.Y[idx]


def build_datasets(voltage_path, label_path, val_ratio, scaler_file, seed=42):
    """Split train / val. Trả về (train_ds, val_ds, n_train, n_val)."""

    def _read(path):
        df = pd.read_csv(path, header=None)
        try:
            df.iloc[0].astype(float)
            has_header = False
        except (ValueError, TypeError):
            has_header = True
        if has_header:
            df = pd.read_csv(path, header=0)
        df_num = df.apply(pd.to_numeric, errors="coerce")
        before = (len(df_num), df_num.shape[1])
        df_num = df_num.dropna().reset_index(drop=True)
        after  = (len(df_num), df_num.shape[1])
        _dbg_log("H1", "train.py:_read", "read_csv_auto",
                 {"path": str(path), "has_header": bool(has_header),
                  "shape_before_dropna": list(before),
                  "shape_after_dropna":  list(after)})
        return df_num

    volt_df  = _read(voltage_path)
    label_df = _read(label_path)
    _dbg_log("H1", "train.py:build_datasets", "loaded_dataframes",
             {"voltage_path": str(voltage_path), "label_path": str(label_path),
              "volt_shape": list(volt_df.shape), "label_shape": list(label_df.shape)})

    assert volt_df.shape[1]  == 64, f"Voltage can 64 cols, co {volt_df.shape[1]}"
    assert label_df.shape[1] == 5,  f"Label can 5 cols, co {label_df.shape[1]}"

    voltages = volt_df.values.astype(np.float32)
    labels   = label_df.values.astype(np.float32)
    N        = min(len(voltages), len(labels))
    voltages, labels = voltages[:N], labels[:N]
    print(f"  Total samples: {N:,}")
    _dbg_log("H2", "train.py:build_datasets", "aligned_arrays",
             {"N_after_min": int(N),
              "volt_min": float(np.min(voltages)) if N else None,
              "volt_max": float(np.max(voltages)) if N else None})

    rng       = np.random.default_rng(seed)
    idx       = rng.permutation(N)
    n_val     = int(N * val_ratio)
    n_train   = N - n_val
    train_idx = idx[:n_train]
    val_idx   = idx[n_train:]
    print(f"  Train: {n_train:,}  |  Val: {n_val:,}")
    _dbg_log("H3", "train.py:build_datasets", "split_indices",
             {"seed": int(seed), "val_ratio": float(val_ratio),
              "n_train": int(n_train), "n_val": int(n_val),
              "overlap_train_val": int(len(np.intersect1d(train_idx, val_idx)))})

    if os.path.exists(scaler_file):
        with open(scaler_file, "rb") as f:
            sc = pickle.load(f)
        volt_scaler  = sc["volt"]
        label_scaler = sc["label"]
        print(f"  Loaded scalers from {scaler_file}")
        _dbg_log("H4", "train.py:build_datasets", "loaded_existing_scalers",
                 {"scaler_file": str(scaler_file)})
    else:
        volt_scaler  = MinMaxScaler(feature_range=(0, 1)).fit(voltages[train_idx])
        label_scaler = StandardScaler().fit(labels[train_idx])
        with open(scaler_file, "wb") as f:
            pickle.dump({"volt": volt_scaler, "label": label_scaler}, f)
        print(f"  Fitted & saved scalers -> {scaler_file}")
        _dbg_log("H4", "train.py:build_datasets", "fitted_new_scalers",
                 {"scaler_file": str(scaler_file)})

    split_path = os.path.join(os.path.dirname(scaler_file), "split_info2.json")
    if not os.path.exists(split_path):
        with open(split_path, "w") as f:
            json.dump({"train": train_idx.tolist(),
                       "val":   val_idx.tolist(),
                       "seed":  seed}, f)
        print(f"  Split info saved -> {split_path}")

    v_scaled = volt_scaler.transform(voltages)
    l_scaled = label_scaler.transform(labels)
    _dbg_log("H5", "train.py:build_datasets", "scaled_stats",
             {"v_scaled_min": float(np.min(v_scaled)) if N else None,
              "v_scaled_max": float(np.max(v_scaled)) if N else None,
              "l_scaled_mean": [float(x) for x in np.mean(l_scaled, axis=0)] if N else None,
              "l_scaled_std":  [float(x) for x in np.std(l_scaled,  axis=0)] if N else None})

    train_ds = PoseDataset(v_scaled[train_idx], l_scaled[train_idx])
    val_ds   = PoseDataset(v_scaled[val_idx],   l_scaled[val_idx])
    return train_ds, val_ds, n_train, n_val


# =============================================================================
# CHECKPOINT HELPERS
# =============================================================================

def save_checkpoint(path, epoch, model, optimizer, scheduler, val_loss, best_val):
    torch.save({
        "epoch":     epoch,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "val_loss":  val_loss,
        "best_val":  best_val,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, device):
    ckpt      = torch.load(path, map_location=device, weights_only=False)
    raw_state = ckpt["model"]
    is_compiled = hasattr(model, "_orig_mod")
    if is_compiled:
        state = (raw_state if any(k.startswith("_orig_mod.") for k in raw_state)
                 else {"_orig_mod." + k: v for k, v in raw_state.items()})
    else:
        state = {k.replace("_orig_mod.", ""): v for k, v in raw_state.items()}
    model.load_state_dict(state)
    try:
        optimizer.load_state_dict(ckpt["optimizer"])
    except Exception as e:
        print(f"Warning: Could not load optimizer state: {e}. Re-initializing.")
    try:
        scheduler.load_state_dict(ckpt["scheduler"])
    except Exception as e:
        print(f"Warning: Could not load scheduler state: {e}. Re-initializing.")
    return ckpt["epoch"], ckpt["best_val"]


def append_log(log_file, entry):
    log = []
    if os.path.exists(log_file):
        with open(log_file) as f:
            try:   log = json.load(f)
            except json.JSONDecodeError: log = []
    log.append(entry)
    with open(log_file, "w") as f:
        json.dump(log, f, indent=2)


# =============================================================================
# PLOT (Train/Val/Physics losses)
# =============================================================================

def plot_losses(train_losses, val_losses,
                train_physics, val_physics,
                save_path: str, lambda_physics: float):

    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10),
                                    sharex=True,
                                    gridspec_kw={"hspace": 0.35})

    # ── Subplot 1: Total loss ─────────────────────────────────────────────────
    ax1.plot(epochs, train_losses, label="Train Loss",
             color="steelblue",  linewidth=1.5)
    ax1.plot(epochs, val_losses,   label="Val Loss",
             color="tomato",     linewidth=1.5)
    ax1.set_title("Training loss)",
                  fontsize=12, fontweight="bold")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # ── Subplot 2: Physics loss (raw, không nhân lambda) ─────────────────────
    ax2.plot(epochs, train_physics, label="Train Physics Loss",
             color="darkorange",  linewidth=1.5, linestyle="--")
    ax2.plot(epochs, val_physics,   label="Val Physics Loss",
             color="mediumorchid", linewidth=1.5, linestyle="--")
    ax2.set_title(
        f"Physics Loss)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("MAE  [V]")
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Loss plot saved -> {save_path}")


# =============================================================================
# INFERENCE TIME BENCHMARK
# =============================================================================

def measure_inference_time(model, device, n_samples=500, n_warmup=50):
    model.eval()
    dummy = torch.randn(1, 1, 8, 8, device=device)
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    timings = []
    with torch.no_grad():
        for _ in range(n_samples):
            if device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end   = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = model(dummy)
                end.record()
                torch.cuda.synchronize()
                timings.append(start.elapsed_time(end))
            else:
                t0 = time.perf_counter()
                _ = model(dummy)
                timings.append((time.perf_counter() - t0) * 1000)
    timings = np.array(timings)
    return timings.mean(), timings.std(), np.percentile(timings, 95)


# =============================================================================
# MAIN
# =============================================================================

def main():
    cfg    = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 65)
    print("  Model Training  (train / val split)")
    print("=" * 65)
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"  Device      : {device} ({gpu_name})")
    print(f"  Voltage     : {cfg.voltage}")
    print(f"  Label       : {cfg.label}")
    print(f"  Sensor calibaration  : {cfg.calib_csv}")
    print(f"  Split       : Train {(1-cfg.val_ratio)*100:.0f}% / Val {cfg.val_ratio*100:.0f}%")
    print(f"  Test        : separate file (run test2.py)")
    print(f"  Epochs      : {cfg.num_epochs}  |  Batch: {cfg.batch_size}  |  LR: {cfg.lr}")
    print(f"  Optimizer   : AdamW  weight_decay={cfg.weight_decay}")
    print(f"  Scheduler   : LinearLR warmup ({cfg.warmup_epochs} ep) -> CosineAnnealingLR")
    print(f"  Loss        : HuberPoseLossCalib  ang_weight={cfg.ang_weight}"
          f"  delta_xyz={cfg.delta_xyz}  delta_ang={cfg.delta_ang}")
    print(f"  lambda_pos  : {cfg.lambda_pos}  |  lambda_physics: {cfg.lambda_physics}")
    print(f"  Ckpt dir    : {cfg.ckpt_dir}")
    print("=" * 65 + "\n")

    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    ckpt_latest = os.path.join(cfg.ckpt_dir, "latest.pt")
    ckpt_best   = os.path.join(cfg.ckpt_dir, "best.pt")
    log_file    = os.path.join(cfg.ckpt_dir, "train_log.json")
    scaler_file = os.path.join(cfg.ckpt_dir, "scalers.pkl")

    # ── Dataset ───────────────────────────────────────────────────────────────
    print("Loading dataset ...")
    train_ds, val_ds, n_train, n_val = build_datasets(
        cfg.voltage, cfg.label, cfg.val_ratio, scaler_file, seed=cfg.seed)

    pin = (device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True,  pin_memory=pin, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size,
                              shuffle=False, pin_memory=pin)

    # ── Model + Loss ──────────────────────────────────────────────────────────
    with open(scaler_file, "rb") as f:
        sc = pickle.load(f)
    volt_scaler  = sc["volt"]
    label_scaler = sc["label"]

    model = Model(out_dim=5)
    model = model.to(device)

    criterion = HuberPoseLossCalib(
        ang_weight     = cfg.ang_weight,
        delta_xyz      = cfg.delta_xyz,
        delta_ang      = cfg.delta_ang,
        lambda_pos     = cfg.lambda_pos,
        lambda_physics = cfg.lambda_physics,
        calib_csv      = cfg.calib_csv,
        volt_scaler    = volt_scaler,
        label_scaler   = label_scaler,
    ).to(device)

    # ── Optimizer + Scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    warmup_sch = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0,
        total_iters=cfg.warmup_epochs)
    cosine_sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.num_epochs - cfg.warmup_epochs, eta_min=1e-6)
    scheduler  = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sch, cosine_sch],
        milestones=[cfg.warmup_epochs])

    import platform
    if platform.system() != "Windows":
        try:
            model = torch.compile(model)
            print("torch.compile enabled")
        except Exception:
            print("torch.compile not available - skipping")
    else:
        print("torch.compile disabled (Windows)")

    # ── Resume & History ──────────────────────────────────────────────────────
    start_epoch, best_val = 1, float("inf")
    train_losses_history   = []
    val_losses_history     = []
    train_physics_history  = []   
    val_physics_history    = []   

    if os.path.exists(ckpt_latest):
        print(f"Resuming from {ckpt_latest} ...")
        start_epoch, best_val = load_checkpoint(
            ckpt_latest, model, optimizer, scheduler, device)
        if os.path.exists(log_file):
            with open(log_file) as f:
                try:
                    history = json.load(f)
                    train_losses_history  = [x["train"]         for x in history]
                    val_losses_history    = [x["val"]           for x in history]
                    train_physics_history = [x["train_physics"] for x in history]  # ← THÊM MỚI
                    val_physics_history   = [x["val_physics"]   for x in history]  # ← THÊM MỚI
                except: pass
        start_epoch += 1
        print(f"  -> Epoch {start_epoch}  best_val={best_val:.6f}\n")
    else:
        print("Training from scratch\n")

    # ── AMP ───────────────────────────────────────────────────────────────────
    use_amp    = (device.type == "cuda")
    amp_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ── Training loop ─────────────────────────────────────────────────────────
    no_improve = 0
    hdr = (f"{'Epoch':>6}  {'Train':>9}  {'Val':>9}  "
           f"{'Val_xyz':>9}  {'Val_ang':>9}  {'Val_phys':>9}  {'LR':>8}  {'Time':>7}")
    print(hdr)
    print("-" * len(hdr))

    for epoch in range(start_epoch, cfg.num_epochs + 1):
        t0 = time.time()

        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        train_loss = train_xyz = train_ang = train_physics = 0.0
        for X_b, Y_b in train_loader:
            X_b = X_b.to(device, non_blocking=True)
            Y_b = Y_b.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(X_b)
            loss, loss_xyz, loss_ang = criterion(pred, Y_b, X_b=X_b)
            loss_phys_batch = criterion.latest_loss_physics
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            amp_scaler.step(optimizer)
            amp_scaler.update()
            n = len(X_b)
            train_loss    += loss.item()            * n
            train_xyz     += loss_xyz.item()        * n
            train_ang     += loss_ang.item()        * n
            train_physics += loss_phys_batch.item() * n
        train_loss    /= n_train
        train_xyz     /= n_train
        train_ang     /= n_train
        train_physics /= n_train
        scheduler.step()

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        val_loss = val_xyz = val_ang = val_physics = 0.0
        with torch.no_grad():
            for X_b, Y_b in val_loader:
                X_b = X_b.to(device, non_blocking=True)
                Y_b = Y_b.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred = model(X_b)
                loss, loss_xyz, loss_ang = criterion(pred, Y_b, X_b=X_b)
                loss_phys_batch = criterion.latest_loss_physics
                if not torch.isfinite(loss):
                    continue
                n = len(X_b)
                val_loss    += loss.item()            * n
                val_xyz     += loss_xyz.item()        * n
                val_ang     += loss_ang.item()        * n
                val_physics += loss_phys_batch.item() * n
        val_loss    /= n_val
        val_xyz     /= n_val
        val_ang     /= n_val
        val_physics /= n_val

        # ── History ───────────────────────────────────────────────────────────
        train_losses_history.append(train_loss)
        val_losses_history.append(val_loss)
        train_physics_history.append(train_physics)   
        val_physics_history.append(val_physics)       

        lr_now  = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0
        print(f"{epoch:>6}  {train_loss:>9.5f}  {val_loss:>9.5f}  "
              f"{val_xyz:>9.5f}  {val_ang:>9.5f}  {val_physics:>9.5f}  "
              f"{lr_now:>8.2e}  {elapsed:>6.1f}s", flush=True)

        append_log(log_file, {
            "epoch": epoch, "train": train_loss,
            "train_xyz": train_xyz, "train_ang": train_ang,
            "train_physics": train_physics,
            "val": val_loss, "val_xyz": val_xyz, "val_ang": val_ang,
            "val_physics": val_physics, "lr": lr_now,
        })

        save_checkpoint(ckpt_latest, epoch, model, optimizer, scheduler,
                        val_loss, best_val)

        if val_loss < best_val:
            best_val, no_improve = val_loss, 0
            save_checkpoint(ckpt_best, epoch, model, optimizer, scheduler,
                            val_loss, best_val)
            print(f"          >> Best saved  val={best_val:.6f} "
                  f"(xyz={val_xyz:.5f}  ang={val_ang:.5f}  phys={val_physics:.5f})",
                  flush=True)
        else:
            no_improve += 1

        if epoch % cfg.save_every == 0:
            save_checkpoint(os.path.join(cfg.ckpt_dir, f"epoch_{epoch:04d}.pt"),
                            epoch, model, optimizer, scheduler, val_loss, best_val)

        if no_improve >= cfg.patience:
            print(f"\nEarly stopping (no improvement for {cfg.patience} epochs)")
            break

    # ── Loss plot ───────────────────────────────────────────────
    plot_losses(
        train_losses   = train_losses_history,
        val_losses     = val_losses_history,
        train_physics  = train_physics_history,
        val_physics    = val_physics_history,
        save_path      = os.path.join(cfg.ckpt_dir, "loss_plot.png"),
        lambda_physics = cfg.lambda_physics,
    )

    # ── Inference time benchmark ──────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Inference Time Benchmark  (single sample, best checkpoint)")
    print("=" * 65)

    if os.path.exists(ckpt_best):
        best_ckpt = torch.load(ckpt_best, map_location=device, weights_only=False)
        raw_state = best_ckpt["model"]
        is_compiled = hasattr(model, "_orig_mod")
        if is_compiled:
            state = (raw_state if any(k.startswith("_orig_mod.") for k in raw_state)
                     else {"_orig_mod." + k: v for k, v in raw_state.items()})
        else:
            state = {k.replace("_orig_mod.", ""): v for k, v in raw_state.items()}
        model.load_state_dict(state)
        print(f"  Loaded best checkpoint from {ckpt_best}")

    mean_ms, std_ms, p95_ms = measure_inference_time(
        model, device, n_samples=500, n_warmup=50)

    print(f"  Mean latency : {mean_ms:.3f} ms/sample")
    print(f"  Std          : {std_ms:.3f} ms")
    print(f"  P95 latency  : {p95_ms:.3f} ms/sample")
    print(f"  Throughput   : ~{1000/mean_ms:.0f} samples/sec")
    print("=" * 65)

    infer_path = os.path.join(cfg.ckpt_dir, "inference_time.json")
    with open(infer_path, "w") as f:
        json.dump({"device": str(device),
                   "mean_ms": round(mean_ms, 4), "std_ms": round(std_ms, 4),
                   "p95_ms":  round(p95_ms,  4),
                   "throughput_sps": round(1000 / mean_ms, 1),
                   "n_warmup": 50, "n_samples": 500}, f, indent=2)
    print(f"  Inference time saved -> {infer_path}")

    print(f"\nDone! Best val loss = {best_val:.6f}")
    print(f"Checkpoints & Plot -> {cfg.ckpt_dir}")


if __name__ == "__main__":
    main()


# =============================================================================
# split train / val / test  
# =============================================================================

# import os, sys, json, pickle, argparse, time
# import uuid
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# import matplotlib.pyplot as plt

# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# from model import Model
# from loss_calib import HuberPoseLossCalib


# def get_config():
#     p = argparse.ArgumentParser()
#     p.add_argument("--voltage",        type=str,   default="grid_calib_data.csv")
#     p.add_argument("--label",          type=str,   default="Grid_points_coordinates.csv")
#     p.add_argument("--calib_csv",     type=str,   default="Hall_sensor_positions.csv")
#     p.add_argument("--ckpt_dir",       type=str,   default="./ckpt")
#     p.add_argument("--val_ratio",      type=float, default=0.2)
#     p.add_argument("--test_ratio",     type=float, default=0.03)
#     p.add_argument("--batch_size",     type=int,   default=64)
#     p.add_argument("--num_epochs",     type=int,   default=200)
#     p.add_argument("--lr",             type=float, default=1e-3)
#     p.add_argument("--weight_decay",   type=float, default=0.00456)
#     p.add_argument("--ang_weight",     type=float, default=1.0)
#     p.add_argument("--delta_xyz",      type=float, default=0.061)
#     p.add_argument("--delta_ang",      type=float, default=0.21)
#     p.add_argument("--lambda_pos",     type=float, default=1.0)
#     p.add_argument("--lambda_physics", type=float, default=1e-4)
#     p.add_argument("--warmup_epochs",  type=int,   default=5)
#     p.add_argument("--save_every",     type=int,   default=5)
#     p.add_argument("--patience",       type=int,   default=45)
#     p.add_argument("--seed",           type=int,   default=42)
#     return p.parse_args()


# class PoseDataset(Dataset):
#     def __init__(self, voltages, labels):
#         self.X = torch.tensor(voltages, dtype=torch.float32).view(-1, 1, 8, 8)
#         self.Y = torch.tensor(labels,   dtype=torch.float32)
#     def __len__(self):          return len(self.X)
#     def __getitem__(self, idx): return self.X[idx], self.Y[idx]


# def build_datasets(voltage_path, label_path, val_ratio, test_ratio,
#                    scaler_file, seed=42):
#     """Split train / val / test. Trả về (train_ds, val_ds, n_train, n_val)."""

#     def _read(path):
#         df = pd.read_csv(path, header=None)
#         try:
#             df.iloc[0].astype(float); has_header = False
#         except (ValueError, TypeError):
#             has_header = True
#         if has_header:
#             df = pd.read_csv(path, header=0)
#         return df.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)

#     volt_df  = _read(voltage_path)
#     label_df = _read(label_path)

#     assert volt_df.shape[1]  == 64, f"Voltage can 64 cols, co {volt_df.shape[1]}"
#     assert label_df.shape[1] == 5,  f"Label can 5 cols, co {label_df.shape[1]}"

#     voltages = volt_df.values.astype(np.float32)
#     labels   = label_df.values.astype(np.float32)
#     N        = min(len(voltages), len(labels))
#     voltages, labels = voltages[:N], labels[:N]
#     print(f"  Total samples: {N:,}")

#     rng      = np.random.default_rng(seed)
#     idx      = rng.permutation(N)
#     n_test   = int(N * test_ratio)
#     n_val    = int(N * val_ratio)
#     n_train  = N - n_val - n_test
#     train_idx = idx[:n_train]
#     val_idx   = idx[n_train:n_train + n_val]
#     test_idx  = idx[n_train + n_val:]
#     print(f"  Train: {n_train:,}  |  Val: {n_val:,}  |  Test: {n_test:,}")

#     if os.path.exists(scaler_file):
#         with open(scaler_file, "rb") as f:
#             sc = pickle.load(f)
#         volt_scaler  = sc["volt"]
#         label_scaler = sc["label"]
#         print(f"  Loaded scalers from {scaler_file}")
#     else:
#         volt_scaler  = MinMaxScaler(feature_range=(0, 1)).fit(voltages[train_idx])
#         label_scaler = StandardScaler().fit(labels[train_idx])
#         with open(scaler_file, "wb") as f:
#             pickle.dump({"volt": volt_scaler, "label": label_scaler}, f)
#         print(f"  Fitted & saved scalers -> {scaler_file}")

#     split_path = os.path.join(os.path.dirname(scaler_file), "split_info.json")
#     if not os.path.exists(split_path):
#         with open(split_path, "w") as f:
#             json.dump({"train": train_idx.tolist(), "val": val_idx.tolist(),
#                        "test":  test_idx.tolist(),  "seed": seed}, f)
#         print(f"  Split info saved -> {split_path}")

#     v_scaled = volt_scaler.transform(voltages)
#     l_scaled = label_scaler.transform(labels)

#     train_ds = PoseDataset(v_scaled[train_idx], l_scaled[train_idx])
#     val_ds   = PoseDataset(v_scaled[val_idx],   l_scaled[val_idx])
#     return train_ds, val_ds, n_train, n_val


# def save_checkpoint(path, epoch, model, optimizer, scheduler, val_loss, best_val):
#     torch.save({"epoch": epoch, "model": model.state_dict(),
#                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
#                 "val_loss": val_loss, "best_val": best_val}, path)


# def load_checkpoint(path, model, optimizer, scheduler, device):
#     ckpt = torch.load(path, map_location=device, weights_only=False)
#     raw_state = ckpt["model"]
#     is_compiled = hasattr(model, "_orig_mod")
#     if is_compiled:
#         state = (raw_state if any(k.startswith("_orig_mod.") for k in raw_state)
#                  else {"_orig_mod." + k: v for k, v in raw_state.items()})
#     else:
#         state = {k.replace("_orig_mod.", ""): v for k, v in raw_state.items()}
#     model.load_state_dict(state)
#     try:    optimizer.load_state_dict(ckpt["optimizer"])
#     except Exception as e: print(f"Warning optimizer: {e}")
#     try:    scheduler.load_state_dict(ckpt["scheduler"])
#     except Exception as e: print(f"Warning scheduler: {e}")
#     return ckpt["epoch"], ckpt["best_val"]


# def append_log(log_file, entry):
#     log = []
#     if os.path.exists(log_file):
#         with open(log_file) as f:
#             try:   log = json.load(f)
#             except json.JSONDecodeError: log = []
#     log.append(entry)
#     with open(log_file, "w") as f:
#         json.dump(log, f, indent=2)


# def measure_inference_time(model, device, n_samples=500, n_warmup=50):
#     model.eval()
#     dummy = torch.randn(1, 1, 8, 8, device=device)
#     with torch.no_grad():
#         for _ in range(n_warmup): _ = model(dummy)
#     if device.type == "cuda": torch.cuda.synchronize()
#     timings = []
#     with torch.no_grad():
#         for _ in range(n_samples):
#             if device.type == "cuda":
#                 start = torch.cuda.Event(enable_timing=True)
#                 end   = torch.cuda.Event(enable_timing=True)
#                 start.record(); _ = model(dummy); end.record()
#                 torch.cuda.synchronize()
#                 timings.append(start.elapsed_time(end))
#             else:
#                 t0 = time.perf_counter(); _ = model(dummy)
#                 timings.append((time.perf_counter() - t0) * 1000)
#     timings = np.array(timings)
#     return timings.mean(), timings.std(), np.percentile(timings, 95)


# def main():
#     cfg    = get_config()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     print("\n" + "=" * 65)
#     print("  Model Training  (train / val / test split)")
#     print("=" * 65)
#     gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
#     print(f"  Device      : {device} ({gpu_name})")
#     print(f"  Voltage     : {cfg.voltage}")
#     print(f"  Label       : {cfg.label}")
#     print(f"  Sensor pos  : {cfg.sensor_pos}")
#     print(f"  Split       : Train {(1-cfg.val_ratio-cfg.test_ratio)*100:.0f}%"
#           f" / Val {cfg.val_ratio*100:.0f}% / Test {cfg.test_ratio*100:.0f}%")
#     print(f"  Epochs      : {cfg.num_epochs}  |  Batch: {cfg.batch_size}  |  LR: {cfg.lr}")
#     print(f"  Optimizer   : AdamW  weight_decay={cfg.weight_decay}")
#     print(f"  Scheduler   : LinearLR warmup ({cfg.warmup_epochs} ep) -> CosineAnnealingLR")
#     print(f"  Loss        : HuberPoseLoss  ang_weight={cfg.ang_weight}"
#           f"  delta_xyz={cfg.delta_xyz}  delta_ang={cfg.delta_ang}")
#     print(f"  lambda_pos  : {cfg.lambda_pos}  |  lambda_physics: {cfg.lambda_physics}")
#     print(f"  Ckpt dir    : {cfg.ckpt_dir}")
#     print("=" * 65 + "\n")

#     os.makedirs(cfg.ckpt_dir, exist_ok=True)
#     torch.manual_seed(cfg.seed)
#     np.random.seed(cfg.seed)

#     ckpt_latest = os.path.join(cfg.ckpt_dir, "latest.pt")
#     ckpt_best   = os.path.join(cfg.ckpt_dir, "best.pt")
#     log_file    = os.path.join(cfg.ckpt_dir, "train_log.json")
#     scaler_file = os.path.join(cfg.ckpt_dir, "scalers.pkl")

#     print("Loading dataset ...")
#     train_ds, val_ds, n_train, n_val = build_datasets(
#         cfg.voltage, cfg.label, cfg.val_ratio, cfg.test_ratio,
#         scaler_file, seed=cfg.seed)

#     pin = (device.type == "cuda")
#     train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
#                               shuffle=True,  pin_memory=pin, drop_last=True)
#     val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size,
#                               shuffle=False, pin_memory=pin)

#     with open(scaler_file, "rb") as f:
#         sc = pickle.load(f)
#     volt_scaler  = sc["volt"]
#     label_scaler = sc["label"]

#     sensor_pos_df     = pd.read_csv(cfg.sensor_pos)
#     sensor_pos_tensor = torch.tensor(sensor_pos_df.values, dtype=torch.float32)

#     model = Model(out_dim=5)
#     model.register_buffer("sensor_pos", sensor_pos_tensor)
#     model = model.to(device)

#     criterion = HuberPoseLoss(
#         ang_weight     = cfg.ang_weight,
#         delta_xyz      = cfg.delta_xyz,
#         delta_ang      = cfg.delta_ang,
#         lambda_pos     = cfg.lambda_pos,
#         lambda_physics = cfg.lambda_physics,
#         sensor_pos     = sensor_pos_tensor,
#         volt_scaler    = volt_scaler,
#         label_scaler   = label_scaler,
#     ).to(device)

#     # ── Optimizer: AdamW với weight_decay ─────────────────────────────────────
#     optimizer = torch.optim.AdamW(
#         model.parameters(),
#         lr=cfg.lr,
#         weight_decay=cfg.weight_decay,
#     )

#     # ── Scheduler: LinearLR warmup → CosineAnnealingLR ────────────────────────
#     warmup_sch = torch.optim.lr_scheduler.LinearLR(
#         optimizer, start_factor=0.1, end_factor=1.0,
#         total_iters=cfg.warmup_epochs)
#     cosine_sch = torch.optim.lr_scheduler.CosineAnnealingLR(
#         optimizer,
#         T_max=cfg.num_epochs - cfg.warmup_epochs,
#         eta_min=1e-6)
#     scheduler = torch.optim.lr_scheduler.SequentialLR(
#         optimizer,
#         schedulers=[warmup_sch, cosine_sch],
#         milestones=[cfg.warmup_epochs])

#     import platform
#     if platform.system() != "Windows":
#         try:    model = torch.compile(model); print("torch.compile enabled")
#         except: print("torch.compile not available - skipping")
#     else:
#         print("torch.compile disabled (Windows)")

#     start_epoch, best_val = 1, float("inf")
#     train_losses_history  = []
#     val_losses_history    = []

#     if os.path.exists(ckpt_latest):
#         print(f"Resuming from {ckpt_latest} ...")
#         start_epoch, best_val = load_checkpoint(
#             ckpt_latest, model, optimizer, scheduler, device)
#         if os.path.exists(log_file):
#             with open(log_file) as f:
#                 try:
#                     history = json.load(f)
#                     train_losses_history = [x["train"] for x in history]
#                     val_losses_history   = [x["val"]   for x in history]
#                 except: pass
#         start_epoch += 1
#         print(f"  -> Epoch {start_epoch}  best_val={best_val:.6f}\n")
#     else:
#         print("Training from scratch\n")

#     use_amp    = (device.type == "cuda")
#     amp_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

#     no_improve = 0
#     hdr = (f"{'Epoch':>6}  {'Train':>9}  {'Val':>9}  "
#            f"{'Val_xyz':>9}  {'Val_ang':>9}  {'Val_phys':>9}  {'LR':>8}  {'Time':>7}")
#     print(hdr); print("-" * len(hdr))

#     for epoch in range(start_epoch, cfg.num_epochs + 1):
#         t0 = time.time()

#         model.train()
#         train_loss = train_xyz = train_ang = train_physics = 0.0
#         for X_b, Y_b in train_loader:
#             X_b = X_b.to(device, non_blocking=True)
#             Y_b = Y_b.to(device, non_blocking=True)
#             optimizer.zero_grad(set_to_none=True)
#             # FIX 1: chỉ autocast phần forward model
#             with torch.amp.autocast("cuda", enabled=use_amp):
#                 pred = model(X_b)
#             loss, loss_xyz, loss_ang = criterion(pred, Y_b, X_b=X_b)
#             # FIX 2: đọc loss_phys SAU khi criterion xong (ngoài autocast)
#             loss_phys_batch = criterion.latest_loss_physics
#             # FIX 3: bỏ qua batch nếu loss NaN/Inf
#             if not torch.isfinite(loss):
#                 optimizer.zero_grad(set_to_none=True)
#                 continue
#             amp_scaler.scale(loss).backward()
#             amp_scaler.unscale_(optimizer)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#             amp_scaler.step(optimizer); amp_scaler.update()
#             n = len(X_b)
#             train_loss    += loss.item()            * n
#             train_xyz     += loss_xyz.item()        * n
#             train_ang     += loss_ang.item()        * n
#             train_physics += loss_phys_batch.item() * n
#         train_loss /= n_train; train_xyz /= n_train
#         train_ang  /= n_train; train_physics /= n_train
#         scheduler.step()

#         model.eval()
#         val_loss = val_xyz = val_ang = val_physics = 0.0
#         with torch.no_grad():
#             for X_b, Y_b in val_loader:
#                 X_b = X_b.to(device, non_blocking=True)
#                 Y_b = Y_b.to(device, non_blocking=True)
#                 # FIX 1 (val): tương tự train
#                 with torch.amp.autocast("cuda", enabled=use_amp):
#                     pred = model(X_b)
#                 loss, loss_xyz, loss_ang = criterion(pred, Y_b, X_b=X_b)
#                 loss_phys_batch = criterion.latest_loss_physics
#                 if not torch.isfinite(loss):
#                     continue
#                 n = len(X_b)
#                 val_loss    += loss.item()            * n
#                 val_xyz     += loss_xyz.item()        * n
#                 val_ang     += loss_ang.item()        * n
#                 val_physics += loss_phys_batch.item() * n
#         val_loss /= n_val; val_xyz /= n_val
#         val_ang  /= n_val; val_physics /= n_val

#         train_losses_history.append(train_loss)
#         val_losses_history.append(val_loss)

#         lr_now  = optimizer.param_groups[0]["lr"]
#         elapsed = time.time() - t0
#         print(f"{epoch:>6}  {train_loss:>9.5f}  {val_loss:>9.5f}  "
#               f"{val_xyz:>9.5f}  {val_ang:>9.5f}  {val_physics:>9.5f}  "
#               f"{lr_now:>8.2e}  {elapsed:>6.1f}s", flush=True)

#         append_log(log_file, {
#             "epoch": epoch, "train": train_loss,
#             "train_xyz": train_xyz, "train_ang": train_ang, "train_physics": train_physics,
#             "val": val_loss, "val_xyz": val_xyz, "val_ang": val_ang, "val_physics": val_physics,
#             "lr": lr_now,
#         })

#         save_checkpoint(ckpt_latest, epoch, model, optimizer, scheduler,
#                         val_loss, best_val)

#         if val_loss < best_val:
#             best_val, no_improve = val_loss, 0
#             save_checkpoint(ckpt_best, epoch, model, optimizer, scheduler,
#                             val_loss, best_val)
#             print(f"          >> Best saved  val={best_val:.6f} "
#                   f"(xyz={val_xyz:.5f}  ang={val_ang:.5f}  phys={val_physics:.5f})",
#                   flush=True)
#         else:
#             no_improve += 1

#         if epoch % cfg.save_every == 0:
#             save_checkpoint(os.path.join(cfg.ckpt_dir, f"epoch_{epoch:04d}.pt"),
#                             epoch, model, optimizer, scheduler, val_loss, best_val)

#         if no_improve >= cfg.patience:
#             print(f"\nEarly stopping (no improvement for {cfg.patience} epochs)")
#             break

#     plt.figure(figsize=(10, 6))
#     plt.plot(train_losses_history, label="Train Loss", color="blue", linewidth=1.5)
#     plt.plot(val_losses_history,   label="Val Loss",   color="red",  linewidth=1.5)
#     plt.title("Training and Validation Loss"); plt.xlabel("Epochs"); plt.ylabel("Loss")
#     plt.legend(); plt.grid(True, linestyle="--", alpha=0.7)
#     plt.savefig(os.path.join(cfg.ckpt_dir, "loss_plot.png")); plt.close()

#     print("\n" + "=" * 65)
#     print("  Inference Time Benchmark  (single sample, best checkpoint)")
#     print("=" * 65)

#     if os.path.exists(ckpt_best):
#         best_ckpt = torch.load(ckpt_best, map_location=device, weights_only=False)
#         raw_state = best_ckpt["model"]
#         is_compiled = hasattr(model, "_orig_mod")
#         if is_compiled:
#             state = (raw_state if any(k.startswith("_orig_mod.") for k in raw_state)
#                      else {"_orig_mod." + k: v for k, v in raw_state.items()})
#         else:
#             state = {k.replace("_orig_mod.", ""): v for k, v in raw_state.items()}
#         model.load_state_dict(state)
#         print(f"  Loaded best checkpoint from {ckpt_best}")

#     mean_ms, std_ms, p95_ms = measure_inference_time(
#         model, device, n_samples=500, n_warmup=50)
#     print(f"  Mean latency : {mean_ms:.3f} ms/sample")
#     print(f"  Std          : {std_ms:.3f} ms")
#     print(f"  P95 latency  : {p95_ms:.3f} ms/sample")
#     print(f"  Throughput   : ~{1000/mean_ms:.0f} samples/sec")
#     print("=" * 65)

#     with open(os.path.join(cfg.ckpt_dir, "inference_time.json"), "w") as f:
#         json.dump({"device": str(device),
#                    "mean_ms": round(mean_ms, 4), "std_ms": round(std_ms, 4),
#                    "p95_ms":  round(p95_ms,  4),
#                    "throughput_sps": round(1000 / mean_ms, 1),
#                    "n_warmup": 50, "n_samples": 500}, f, indent=2)

#     print(f"\nDone! Best val loss = {best_val:.6f}")
#     print(f"Checkpoints & Plot -> {cfg.ckpt_dir}")


# if __name__ == "__main__":
#     main()