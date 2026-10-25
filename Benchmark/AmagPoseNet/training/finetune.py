"""AMagPoseNet – training/finetune.py

Stage 3: End-to-end fine-tuning of CaliNet + PoseNet on real sensor data.

Strategy (matching train_mvec.py finetune phase):
    - Both CaliNet AND PoseNet are trained jointly.
    - Separate learning rates: LR_calinet (slower) vs LR_posenet (faster).
    - Loss: lambda_calib * CaliNetLoss + AMagPoseLoss(PoseNet(CaliNet(raw)), pose)
    - Dataset: (raw, clean, pose) triples from train_a split.

Paper variant (frozen PoseNet) is NOT used here — the benchmark aligns
with train_mvec.py's approach for a fair framework-wide comparison.

Checkpoints saved as "full_model_best.pt" and "full_model_last.pt" with the
same key structure as train_mvec.py for compatibility with evaluate.py.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..models.amagposenet import AMagPoseNet
from ..losses.pose_loss import CalibPoseLoss


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _strip(state: dict) -> dict:
    return {k.replace("_orig_mod.", ""): v for k, v in state.items()}


def _load_component(module: nn.Module, path: str, key: str, device) -> None:
    """Load weights for a single sub-module from a checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    raw_state = ckpt.get(key, ckpt.get("model_state_dict", ckpt))
    state = _strip(raw_state)
    missing, unexpected = module.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  [warn] Loading {key}: missing={list(missing)}, unexpected={list(unexpected)}")


def save_checkpoint(path, phase, epoch, model, optimizer, scheduler, best_val, metrics):
    raw = getattr(model, "_orig_mod", model)
    torch.save(
        {
            "phase":                phase,
            "epoch":                epoch,
            "best_val":             best_val,
            "metrics":              metrics,
            "model_state_dict":     raw.state_dict(),
            "calinet_state_dict":   raw.calinet.state_dict(),
            "locnet_state_dict":    raw.posenet.state_dict(),   # "locnet" key for eval compat
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        path,
    )


def append_log(path: Path, entry: dict) -> None:
    history = json.loads(path.read_text()) if path.is_file() else []
    path.write_text(json.dumps(history + [entry], indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def finetune(
    train_loader,
    val_loader,
    *,
    ckpt_dir: Path,
    calinet_checkpoint: str,
    posenet_checkpoint: str,
    num_epochs: int      = 200,
    lr_calinet: float    = 2e-4,
    lr_posenet: float    = 1e-3,
    weight_decay: float  = 4.56e-3,
    warmup_epochs: int   = 5,
    eta_min: float       = 1e-6,
    beta: float          = 1.0,
    lambda_pos: float    = 1.0,
    lambda_calib: float  = 0.1,
    calib_reduction: str = "mean",
    grad_clip_norm: float = 1.0,
    patience: int        = 45,
    save_every: int      = 5,
    device: torch.device | None = None,
    compile_model: bool  = True,
    resume: str | None   = None,
) -> Path:
    """Fine-tune CaliNet + PoseNet end-to-end on real (raw, clean, pose) triples.

    Args:
        train_loader:       yields (raw [B,1,8,8], clean [B,1,8,8], pose [B,6])
        val_loader:         same format
        ckpt_dir:           output directory for checkpoints
        calinet_checkpoint: path to Stage 1 best checkpoint
        posenet_checkpoint: path to Stage 2 best checkpoint
        beta:               orientation loss weight
        lambda_calib:       calibration loss weight
        …                   other hyper-parameters

    Returns:
        Path to best checkpoint ("full_model_best.pt").
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Model ──────────────────────────────────────────────────────────────
    base_model = AMagPoseNet(use_calinet=True).to(device)
    _load_component(base_model.calinet, calinet_checkpoint, "calinet_state_dict", device)
    _load_component(base_model.posenet, posenet_checkpoint, "locnet_state_dict",  device)
    print(f"[Stage3] Loaded CaliNet  from: {calinet_checkpoint}")
    print(f"[Stage3] Loaded PoseNet  from: {posenet_checkpoint}")

    # ── Optimizer: separate LRs for the two sub-networks (train_mvec.py style)
    optimizer = torch.optim.AdamW(
        [
            {"params": base_model.calinet.parameters(), "lr": lr_calinet},
            {"params": base_model.posenet.parameters(), "lr": lr_posenet},
        ],
        weight_decay=weight_decay,
    )
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0,
        total_iters=max(1, warmup_epochs),
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, num_epochs - warmup_epochs), eta_min=eta_min,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], milestones=[warmup_epochs],
    )

    # ── Loss ──────────────────────────────────────────────────────────────
    criterion = CalibPoseLoss(
        beta=beta, lambda_pos=lambda_pos, lambda_calib=lambda_calib,
        calib_reduction=calib_reduction,
    )

    # ── Optional torch.compile ─────────────────────────────────────────────
    model = base_model
    if compile_model and device.type == "cuda":
        try:
            model = torch.compile(base_model)
            print("[Stage3] torch.compile enabled")
        except Exception as exc:
            print(f"[Stage3] torch.compile unavailable: {exc}")

    use_amp    = device.type == "cuda"
    amp_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ── Optional resume ────────────────────────────────────────────────────
    start_epoch, best = 1, float("inf")
    log_path = ckpt_dir / "train_log_finetune.json"
    if resume:
        saved = torch.load(resume, map_location=device, weights_only=False)
        base_model.load_state_dict(_strip(saved["model_state_dict"]), strict=False)
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        scheduler.load_state_dict(saved["scheduler_state_dict"])
        start_epoch = saved["epoch"] + 1
        best        = saved["best_val"]
        print(f"[Stage3] Resumed from epoch {saved['epoch']}, best_val={best:.6f}")

    best_path = ckpt_dir / "full_model_best.pt"
    last_path = ckpt_dir / "full_model_last.pt"
    no_improve = 0

    print(f"\n{'='*65}")
    print(f"  Stage 3 – End-to-end Fine-tuning")
    print(f"  Train : {len(train_loader.dataset):,}  Val: {len(val_loader.dataset):,}")
    print(f"  Device: {device}  LR_cal={lr_calinet}  LR_pose={lr_posenet}")
    print(f"  beta={beta}  lambda_calib={lambda_calib}  Epochs={num_epochs}")
    print(f"{'='*65}\n")

    def run_epoch(loader, training: bool) -> dict:
        model.train(training) if training else model.eval()
        totals = {"total": 0.0, "calib": 0.0, "pos": 0.0, "ori": 0.0, "mae": 0.0}
        seen   = 0
        ctx    = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for batch in loader:
                raw, clean, pose = [t.to(device, non_blocking=True) for t in batch]
                if training:
                    optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred, aux = model(raw, return_features=True)
                    corrected = aux["corrected"]
                    loss, l_calib, l_pos, l_ori = criterion(pred, pose, corrected, clean)
                if not torch.isfinite(loss):
                    continue
                if training:
                    amp_scaler.scale(loss).backward()
                    amp_scaler.unscale_(optimizer)
                    if grad_clip_norm > 0:
                        nn.utils.clip_grad_norm_(base_model.parameters(), grad_clip_norm)
                    amp_scaler.step(optimizer)
                    amp_scaler.update()
                n = len(raw)
                totals["total"] += loss.item()    * n
                totals["calib"] += l_calib.item() * n
                totals["pos"]   += l_pos.item()   * n
                totals["ori"]   += l_ori.item()   * n
                # CalibNet correction MAE (voltage-space)
                totals["mae"]   += (corrected - clean).abs().mean().item() * n
                seen            += n
        return {k: v / max(1, seen) for k, v in totals.items()}

    for epoch in range(start_epoch, num_epochs + 1):
        t0      = time.time()
        train_m = run_epoch(train_loader, training=True)
        val_m   = run_epoch(val_loader,   training=False)
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
            f"[S3] {epoch:03d}  total={val_m['total']:.6f}  "
            f"pos={val_m['pos']:.6f}  ori={val_m['ori']:.6f}  "
            f"calib={val_m['calib']:.6f}  ({entry['seconds']:.1f}s)"
        )

        save_checkpoint(last_path, "finetune", epoch, base_model,
                        optimizer, scheduler, best, entry)

        if val_m["total"] < best:
            best, no_improve = val_m["total"], 0
            save_checkpoint(best_path, "finetune", epoch, base_model,
                            optimizer, scheduler, best, entry)
            print(f"  -> best saved: {best:.6f}")
        else:
            no_improve += 1

        if epoch % save_every == 0:
            save_checkpoint(
                ckpt_dir / f"finetune_epoch_{epoch:04d}.pt",
                "finetune", epoch, base_model, optimizer, scheduler, best, entry,
            )

        if no_improve >= patience:
            print(f"[Stage3] Early stopping after {patience} unimproved epochs.")
            break

    print(f"\n[Stage3] Done. Best val loss: {best:.6f}  |  {best_path}")
    return best_path
