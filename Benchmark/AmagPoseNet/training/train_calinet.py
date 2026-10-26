"""AMagPoseNet – training/train_calinet.py

Stage 1: Train CaliNet on (raw, clean) Hall-sensor image pairs.

Paper Sec. III-C step 2:
    "pretrain CaliNet with the generated dataset … Mmodel is used as the
     desired output; as the input data after adding Gaussian noise."

Adaptation:
    My framework provides actual (raw, clean) pairs directly from the hardware
    (raw = measured Hall voltages; clean = computed from physical model).
    This is equivalent to the paper's intent: CaliNet learns to map imperfect
    real measurements to the ideal model-domain representation.

Dataset used: train_a / val_a split (same as train_mvec.py calibnet phase).
Loss: L1(CaliNet(raw), clean)

Usage (via scripts/train_stage1.py):
    python scripts/train_stage1.py --config configs/benchmark_config.yaml [overrides]
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..models.calinet import CaliNet
from ..losses.calinet_loss import CaliNetLoss
from .artifacts import save_loss_plot


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _strip(state: dict) -> dict:
    """Remove '_orig_mod.' prefix added by torch.compile."""
    return {k.replace("_orig_mod.", ""): v for k, v in state.items()}


def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer,
    scheduler,
    best_val: float,
    metrics: dict,
) -> None:
    raw = getattr(model, "_orig_mod", model)
    torch.save(
        {
            "phase":                "calinet",
            "epoch":                epoch,
            "best_val":             best_val,
            "metrics":              metrics,
            "calinet_state_dict":   raw.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        path,
    )


def append_log(path: Path, entry: dict) -> None:
    history = json.loads(path.read_text()) if path.is_file() else []
    path.write_text(json.dumps(history + [entry], indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train_calinet(
    train_loader,
    val_loader,
    *,
    ckpt_dir: Path,
    num_epochs: int      = 200,
    lr: float            = 2e-4,
    weight_decay: float  = 4.56e-3,
    warmup_epochs: int   = 5,
    eta_min: float       = 1e-6,
    calib_reduction: str = "mean",
    grad_clip_norm: float = 1.0,
    patience: int        = 45,
    save_every: int      = 5,
    device: torch.device | None = None,
    compile_model: bool  = True,
    resume: str | None   = None,
) -> Path:
    """Train CaliNet and save checkpoints.

    Args:
        train_loader: yields (raw [B,1,8,8], clean [B,1,8,8])
        val_loader:   same format
        ckpt_dir:     directory to save checkpoints
        …             hyper-parameters

    Returns:
        Path to best checkpoint.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Model ──────────────────────────────────────────────────────────────
    base_model = CaliNet().to(device)

    # ── Optimizer / Scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        base_model.parameters(), lr=lr, weight_decay=weight_decay,
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
    criterion = CaliNetLoss(reduction=calib_reduction)

    # ── Optional torch.compile ─────────────────────────────────────────────
    model = base_model
    if compile_model and device.type == "cuda":
        try:
            model = torch.compile(base_model)
            print("[Stage1] torch.compile enabled")
        except Exception as exc:
            print(f"[Stage1] torch.compile unavailable: {exc}")

    use_amp   = device.type == "cuda"
    amp_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ── Resume ─────────────────────────────────────────────────────────────
    start_epoch, best = 1, float("inf")
    log_path = ckpt_dir / "train_log_calinet.json"
    if resume:
        saved = torch.load(resume, map_location=device, weights_only=False)
        base_model.load_state_dict(_strip(saved["calinet_state_dict"]))
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        scheduler.load_state_dict(saved["scheduler_state_dict"])
        start_epoch = saved["epoch"] + 1
        best        = saved["best_val"]
        print(f"[Stage1] Resumed from epoch {saved['epoch']}, best_val={best:.6f}")

    best_path = ckpt_dir / "calinet_pretrained.pt"
    last_path = ckpt_dir / "calinet_last.pt"
    no_improve = 0

    print(f"\n{'='*65}")
    print(f"  Stage 1 – CaliNet Training")
    print(f"  Train : {len(train_loader.dataset):,}  Val: {len(val_loader.dataset):,}")
    print(f"  Device: {device}  LR={lr}  Epochs={num_epochs}")
    print(f"{'='*65}\n")

    def run_epoch(loader, training: bool) -> dict:
        model.train(training) if training else model.eval()
        total_loss, seen = 0.0, 0
        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for raw, clean in loader:
                raw, clean = raw.to(device, non_blocking=True), clean.to(device, non_blocking=True)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    corrected = base_model(raw)
                    loss      = criterion(corrected, clean)
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
                total_loss += loss.item() * n
                seen       += n
        return {"loss": total_loss / max(1, seen)}

    for epoch in range(start_epoch, num_epochs + 1):
        t0      = time.time()
        train_m = run_epoch(train_loader, training=True)
        val_m   = run_epoch(val_loader,   training=False)
        scheduler.step()

        entry = {
            "epoch":   epoch,
            "train":   train_m,
            "val":     val_m,
            "lr":      optimizer.param_groups[0]["lr"],
            "seconds": time.time() - t0,
        }
        append_log(log_path, entry)
        print(
            f"[S1] {epoch:03d}  train={train_m['loss']:.6f}  val={val_m['loss']:.6f}"
            f"  ({entry['seconds']:.1f}s)"
        )

        save_checkpoint(last_path, epoch, base_model, optimizer, scheduler,
                        best, entry)

        if val_m["loss"] < best:
            best, no_improve = val_m["loss"], 0
            save_checkpoint(best_path, epoch, base_model, optimizer, scheduler,
                            best, entry)
            print(f"  -> best saved: {best:.6f}")
        else:
            no_improve += 1

        if epoch % save_every == 0:
            save_checkpoint(
                ckpt_dir / f"calinet_epoch_{epoch:04d}.pt",
                epoch, base_model, optimizer, scheduler, best, entry,
            )

        if no_improve >= patience:
            print(f"[Stage1] Early stopping after {patience} unimproved epochs.")
            break

    save_loss_plot(log_path, ckpt_dir / "loss_plot_calinet.png", "AMagPoseNet CaliNet training loss")
    print(f"\n[Stage1] Done. Best val loss: {best:.6f}  |  {best_path}")
    return best_path
