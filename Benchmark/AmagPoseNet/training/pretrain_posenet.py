"""AMagPoseNet – training/pretrain_posenet.py

Stage 2: Pre-train PoseNet on synthetic (model-domain) data.

Paper Sec. III-C step 1:
    "Training PoseNet based on the prior mathematical model.
     A generated dataset Dmodel(Mmodel, vmodel) is provided to train PoseNet."

In my framework: synthetic_grid_data.csv / synthetic_grid_coordinates.csv
play the role of the paper's Dmodel.

Dataset used: train_b / val_b split (same as train_mvec.py locnet phase).
Loss: AMagPoseLoss (L1_pos + beta * L1_ori)

CaliNet is NOT instantiated in this phase (same as train_mvec.py locnet phase).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..models.posenet import PoseNet
from ..losses.pose_loss import AMagPoseLoss


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _strip(state: dict) -> dict:
    return {k.replace("_orig_mod.", ""): v for k, v in state.items()}


def save_checkpoint(path, epoch, model, optimizer, scheduler, best_val, metrics):
    raw = getattr(model, "_orig_mod", model)
    torch.save(
        {
            "phase":                "locnet",
            "epoch":                epoch,
            "best_val":             best_val,
            "metrics":              metrics,
            "locnet_state_dict":    raw.state_dict(),
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

def pretrain_posenet(
    train_loader,
    val_loader,
    *,
    ckpt_dir: Path,
    num_epochs: int      = 200,
    lr: float            = 1e-3,
    weight_decay: float  = 4.56e-3,
    warmup_epochs: int   = 5,
    beta: float          = 1.0,
    lambda_pos: float    = 1.0,
    patience: int        = 45,
    save_every: int      = 5,
    device: torch.device | None = None,
    compile_model: bool  = True,
    resume: str | None   = None,
) -> Path:
    """Pre-train PoseNet on synthetic data.

    Args:
        train_loader: yields (synthetic_volt [B,1,8,8], synthetic_pose [B,6])
        val_loader:   same format
        ckpt_dir:     directory for checkpoints
        beta:         orientation loss weight (paper: not reported; default 1.0)
        …             other hyper-parameters

    Returns:
        Path to best checkpoint.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    base_model = PoseNet().to(device)

    optimizer = torch.optim.AdamW(
        base_model.parameters(), lr=lr, weight_decay=weight_decay,
    )
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0,
        total_iters=max(1, warmup_epochs),
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, num_epochs - warmup_epochs), eta_min=1e-6,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], milestones=[warmup_epochs],
    )

    criterion = AMagPoseLoss(beta=beta, lambda_pos=lambda_pos)

    model = base_model
    if compile_model and device.type == "cuda":
        try:
            model = torch.compile(base_model)
            print("[Stage2] torch.compile enabled")
        except Exception as exc:
            print(f"[Stage2] torch.compile unavailable: {exc}")

    use_amp    = device.type == "cuda"
    amp_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch, best = 1, float("inf")
    log_path = ckpt_dir / "train_log_locnet.json"
    if resume:
        saved = torch.load(resume, map_location=device, weights_only=False)
        base_model.load_state_dict(_strip(saved["locnet_state_dict"]))
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        scheduler.load_state_dict(saved["scheduler_state_dict"])
        start_epoch = saved["epoch"] + 1
        best        = saved["best_val"]
        print(f"[Stage2] Resumed from epoch {saved['epoch']}, best_val={best:.6f}")

    best_path = ckpt_dir / "posenet_pretrained.pt"
    last_path = ckpt_dir / "posenet_last.pt"
    no_improve = 0

    print(f"\n{'='*65}")
    print(f"  Stage 2 – PoseNet Pre-training (synthetic data)")
    print(f"  Train : {len(train_loader.dataset):,}  Val: {len(val_loader.dataset):,}")
    print(f"  Device: {device}  LR={lr}  beta={beta}  Epochs={num_epochs}")
    print(f"{'='*65}\n")

    def run_epoch(loader, training: bool) -> dict:
        model.train(training) if training else model.eval()
        totals = {"total": 0.0, "pos": 0.0, "ori": 0.0}
        seen   = 0
        ctx    = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for volt, pose in loader:
                volt = volt.to(device, non_blocking=True)
                pose = pose.to(device, non_blocking=True)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred               = base_model(volt)
                    loss, l_pos, l_ori = criterion(pred, pose)
                if not torch.isfinite(loss):
                    continue
                if training:
                    amp_scaler.scale(loss).backward()
                    amp_scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(base_model.parameters(), 1.0)
                    amp_scaler.step(optimizer)
                    amp_scaler.update()
                n = len(volt)
                totals["total"] += loss.item()  * n
                totals["pos"]   += l_pos.item() * n
                totals["ori"]   += l_ori.item() * n
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
            "lr":      optimizer.param_groups[0]["lr"],
            "seconds": time.time() - t0,
        }
        append_log(log_path, entry)
        print(
            f"[S2] {epoch:03d}  train={train_m['total']:.6f}  val={val_m['total']:.6f}"
            f"  pos={val_m['pos']:.6f}  ori={val_m['ori']:.6f}  ({entry['seconds']:.1f}s)"
        )

        save_checkpoint(last_path, epoch, base_model, optimizer, scheduler,
                        best, entry)

        if val_m["total"] < best:
            best, no_improve = val_m["total"], 0
            save_checkpoint(best_path, epoch, base_model, optimizer, scheduler,
                            best, entry)
            print(f"  -> best saved: {best:.6f}")
        else:
            no_improve += 1

        if epoch % save_every == 0:
            save_checkpoint(
                ckpt_dir / f"posenet_epoch_{epoch:04d}.pt",
                epoch, base_model, optimizer, scheduler, best, entry,
            )

        if no_improve >= patience:
            print(f"[Stage2] Early stopping after {patience} unimproved epochs.")
            break

    print(f"\n[Stage2] Done. Best val loss: {best:.6f}  |  {best_path}")
    return best_path
