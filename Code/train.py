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
from loss import HuberPoseLoss


# =============================================================================
# CONFIG
# =============================================================================

def get_config():
    p = argparse.ArgumentParser()
    p.add_argument("--voltage",       type=str,   default="grid_calib_data.csv")
    p.add_argument("--label",         type=str,   default="Grid_points_coordinates.csv")
    p.add_argument("--ckpt_dir",      type=str,   default="./ckpt2")
    p.add_argument("--val_ratio",     type=float, default=0.2,
                   help="Ti le validation (default=0.2)")
    p.add_argument("--batch_size",    type=int,   default=64)
    p.add_argument("--num_epochs",    type=int,   default=200)
    p.add_argument("--lr",            type=float, default=0.003839)
    p.add_argument("--weight_decay",  type=float, default=0.00456)
    p.add_argument("--ang_weight",    type=float, default=1.0)
    p.add_argument("--delta_xyz",     type=float, default=0.061)
    p.add_argument("--delta_ang",     type=float, default=0.21)
    p.add_argument("--warmup_epochs", type=int,   default=5)
    p.add_argument("--save_every",    type=int,   default=5)
    p.add_argument("--patience",      type=int,   default=45)
    p.add_argument("--seed",          type=int,   default=42)
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
        after = (len(df_num), df_num.shape[1])
        _dbg_log(
            hypothesisId="H1",
            location="train.py:_read",
            message="read_csv_auto",
            data={
                "path": str(path),
                "has_header": bool(has_header),
                "shape_before_dropna": list(before),
                "shape_after_dropna": list(after),
            },
        )
        return df_num

    volt_df  = _read(voltage_path)
    label_df = _read(label_path)
    _dbg_log(
        hypothesisId="H1",
        location="train.py:build_datasets",
        message="loaded_dataframes",
        data={
            "voltage_path": str(voltage_path),
            "label_path": str(label_path),
            "volt_shape": list(volt_df.shape),
            "label_shape": list(label_df.shape),
        },
    )

    assert volt_df.shape[1]  == 64, f"Voltage can 64 cols, co {volt_df.shape[1]}"
    assert label_df.shape[1] == 5,  f"Label can 5 cols, co {label_df.shape[1]}"

    voltages = volt_df.values.astype(np.float32)
    labels   = label_df.values.astype(np.float32)
    N        = min(len(voltages), len(labels))
    voltages, labels = voltages[:N], labels[:N]
    print(f"  Total samples: {N:,}")
    _dbg_log(
        hypothesisId="H2",
        location="train.py:build_datasets",
        message="aligned_arrays",
        data={
            "len_voltages_raw": int(len(volt_df)),
            "len_labels_raw": int(len(label_df)),
            "N_after_min": int(N),
            "volt_min": float(np.min(voltages)) if N else None,
            "volt_max": float(np.max(voltages)) if N else None,
            "xyz_min": [float(x) for x in np.min(labels[:, :3], axis=0)] if N else None,
            "xyz_max": [float(x) for x in np.max(labels[:, :3], axis=0)] if N else None,
            "cos_min": [float(x) for x in np.min(labels[:, 3:], axis=0)] if N else None,
            "cos_max": [float(x) for x in np.max(labels[:, 3:], axis=0)] if N else None,
        },
    )

    # Split train / val 
    rng      = np.random.default_rng(seed)
    idx      = rng.permutation(N)
    n_val    = int(N * val_ratio)
    n_train  = N - n_val
    train_idx = idx[:n_train]
    val_idx   = idx[n_train:]
    print(f"  Train: {n_train:,}  |  Val: {n_val:,}")
    _dbg_log(
        hypothesisId="H3",
        location="train.py:build_datasets",
        message="split_indices",
        data={
            "seed": int(seed),
            "val_ratio": float(val_ratio),
            "n_train": int(n_train),
            "n_val": int(n_val),
            "idx_unique": int(len(np.unique(idx))),
            "train_unique": int(len(np.unique(train_idx))),
            "val_unique": int(len(np.unique(val_idx))),
            "overlap_train_val": int(len(np.intersect1d(train_idx, val_idx))),
            "train_min": int(np.min(train_idx)) if n_train else None,
            "train_max": int(np.max(train_idx)) if n_train else None,
            "val_min": int(np.min(val_idx)) if n_val else None,
            "val_max": int(np.max(val_idx)) if n_val else None,
        },
    )

    # Fit scaler CHỈ trên train
    if os.path.exists(scaler_file):
        with open(scaler_file, "rb") as f:
            sc = pickle.load(f)
        volt_scaler  = sc["volt"]
        label_scaler = sc["label"]
        print(f"  Loaded scalers from {scaler_file}")
        _dbg_log(
            hypothesisId="H4",
            location="train.py:build_datasets",
            message="loaded_existing_scalers",
            data={"scaler_file": str(scaler_file)},
        )
    else:
        volt_scaler  = MinMaxScaler(feature_range=(0, 1)).fit(voltages[train_idx])
        label_scaler = StandardScaler().fit(labels[train_idx])
        with open(scaler_file, "wb") as f:
            pickle.dump({"volt": volt_scaler, "label": label_scaler}, f)
        print(f"  Fitted & saved scalers -> {scaler_file}")
        _dbg_log(
            hypothesisId="H4",
            location="train.py:build_datasets",
            message="fitted_new_scalers",
            data={"scaler_file": str(scaler_file)},
        )

    # Lưu split info (chỉ train/val)
    split_path = os.path.join(os.path.dirname(scaler_file), "split_info2.json")
    if not os.path.exists(split_path):
        with open(split_path, "w") as f:
            json.dump({"train": train_idx.tolist(),
                       "val":   val_idx.tolist(),
                       "seed":  seed}, f)
        print(f"  Split info saved -> {split_path}")

    v_scaled = volt_scaler.transform(voltages)
    l_scaled = label_scaler.transform(labels)
    _dbg_log(
        hypothesisId="H5",
        location="train.py:build_datasets",
        message="scaled_stats",
        data={
            "v_scaled_min": float(np.min(v_scaled)) if N else None,
            "v_scaled_max": float(np.max(v_scaled)) if N else None,
            "l_scaled_mean": [float(x) for x in np.mean(l_scaled, axis=0)] if N else None,
            "l_scaled_std": [float(x) for x in np.std(l_scaled, axis=0)] if N else None,
        },
    )

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
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
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
    print("  Model Training")
    print("=" * 65)
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"  Device      : {device} ({gpu_name})")
    print(f"  Voltage     : {cfg.voltage}")
    print(f"  Label       : {cfg.label}")
    print(f"  Split       : Train {(1-cfg.val_ratio)*100:.0f}% / Val {cfg.val_ratio*100:.0f}%")
    print(f"  Test        : separate file (run test2.py)")
    print(f"  Epochs      : {cfg.num_epochs}  |  Batch: {cfg.batch_size}  |  LR: {cfg.lr}")
    print(f"  Loss        : HuberPoseLoss  ang_weight={cfg.ang_weight}"
          f"  delta_xyz={cfg.delta_xyz}  delta_ang={cfg.delta_ang}")
    print(f"  Weight decay: {cfg.weight_decay}  |  Warmup: {cfg.warmup_epochs} epochs")
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
    model     = Model(out_dim=5).to(device)
    criterion = HuberPoseLoss(ang_weight=cfg.ang_weight,
                              delta_xyz=cfg.delta_xyz,
                              delta_ang=cfg.delta_ang)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg.lr, weight_decay=cfg.weight_decay)

    warmup_sch = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=cfg.warmup_epochs)
    cosine_sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.num_epochs - cfg.warmup_epochs, eta_min=1e-6)
    scheduler  = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sch, cosine_sch], milestones=[cfg.warmup_epochs])

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
    start_epoch, best_val   = 1, float("inf")
    train_losses_history    = []
    val_losses_history      = []

    if os.path.exists(ckpt_latest):
        print(f"Resuming from {ckpt_latest} ...")
        start_epoch, best_val = load_checkpoint(
            ckpt_latest, model, optimizer, scheduler, device)
        if os.path.exists(log_file):
            with open(log_file) as f:
                try:
                    history = json.load(f)
                    train_losses_history = [x['train'] for x in history]
                    val_losses_history   = [x['val']   for x in history]
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
           f"{'Huber_xyz':>10}  {'Huber_ang':>10}  {'LR':>8}  {'Time':>7}")
    print(hdr)
    print("-" * len(hdr))

    for epoch in range(start_epoch, cfg.num_epochs + 1):
        t0 = time.time()

        # Train
        model.train()
        train_loss = 0.0
        for X_b, Y_b in train_loader:
            X_b = X_b.to(device, non_blocking=True)
            Y_b = Y_b.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss, _, _ = criterion(model(X_b), Y_b)
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            amp_scaler.step(optimizer)
            amp_scaler.update()
            train_loss += loss.item() * len(X_b)
        train_loss /= n_train
        scheduler.step()

        # Validate
        model.eval()
        val_loss = val_xyz = val_ang = 0.0
        with torch.no_grad():
            for X_b, Y_b in val_loader:
                X_b = X_b.to(device, non_blocking=True)
                Y_b = Y_b.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    loss, loss_xyz, loss_ang = criterion(model(X_b), Y_b)
                n = len(X_b)
                val_loss += loss.item()     * n
                val_xyz  += loss_xyz.item() * n
                val_ang  += loss_ang.item() * n
        val_loss /= n_val
        val_xyz  /= n_val
        val_ang  /= n_val

        train_losses_history.append(train_loss)
        val_losses_history.append(val_loss)

        lr_now  = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0
        print(f"{epoch:>6}  {train_loss:>9.5f}  {val_loss:>9.5f}  "
              f"{val_xyz:>10.5f}  {val_ang:>10.5f}  "
              f"{lr_now:>8.2e}  {elapsed:>6.1f}s", flush=True)

        append_log(log_file, {"epoch": epoch, "train": train_loss, "val": val_loss,
                               "val_xyz": val_xyz, "val_ang": val_ang, "lr": lr_now})

        save_checkpoint(ckpt_latest, epoch, model, optimizer, scheduler,
                        val_loss, best_val)

        if val_loss < best_val:
            best_val, no_improve = val_loss, 0
            save_checkpoint(ckpt_best, epoch, model, optimizer, scheduler,
                            val_loss, best_val)
            print(f"          >> Best saved  val={best_val:.6f} "
                  f"(xyz={val_xyz:.5f}  ang={val_ang:.5f})", flush=True)
        else:
            no_improve += 1

        if epoch % cfg.save_every == 0:
            save_checkpoint(os.path.join(cfg.ckpt_dir, f"epoch_{epoch:04d}.pt"),
                            epoch, model, optimizer, scheduler, val_loss, best_val)

        if no_improve >= cfg.patience:
            print(f"\nEarly stopping (no improvement for {cfg.patience} epochs)")
            break

    # ── Loss plot ─────────────────────────────────────────────────────────────
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses_history, label='Train Loss', color='blue', linewidth=1.5)
    plt.plot(val_losses_history,   label='Val Loss',   color='red',  linewidth=1.5)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plot_path = os.path.join(cfg.ckpt_dir, "loss_plot.png")
    plt.savefig(plot_path)
    plt.close()

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
        json.dump({
            "device": str(device),
            "mean_ms": round(mean_ms, 4), "std_ms": round(std_ms, 4),
            "p95_ms":  round(p95_ms,  4),
            "throughput_sps": round(1000 / mean_ms, 1),
            "n_warmup": 50, "n_samples": 500,
        }, f, indent=2)
    print(f"  Inference time saved -> {infer_path}")

    print(f"\nDone! Best val loss = {best_val:.6f}")
    print(f"Checkpoints & Plot -> {cfg.ckpt_dir}")


if __name__ == "__main__":
    main()