"""Single-stage theoretical-data training prescribed by the MobilePoseNet paper."""
from __future__ import annotations
import json
import time
from pathlib import Path
import torch
from torch import nn
from ..losses import MobilePoseLoss
from ..models import MobilePoseNet


def _save(path, epoch, model, optimizer, scheduler, best, metrics):
    raw = getattr(model, "_orig_mod", model)
    torch.save({"phase": "theoretical", "epoch": epoch, "best_val": best, "metrics": metrics, "model_state_dict": raw.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict()}, path)


def train_theoretical(train_loader, val_loader, *, ckpt_dir: Path, epochs: int = 256, lr: float = 1e-4, beta: float = 1.0, weight_decay: float = 0., grad_clip_norm: float = 1., device: torch.device | None = None, compile_model: bool = True) -> Path:
    """Train only on theoretical data; noise injection belongs in the dataset."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    base = MobilePoseNet().to(device); model = base
    if compile_model and device.type == "cuda":
        try: model = torch.compile(base)
        except Exception: pass
    optimizer = torch.optim.Adam(base.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.)
    criterion = MobilePoseLoss(beta); amp = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best, history, best_path = float("inf"), [], ckpt_dir / "mobileposenet_best.pt"

    def run(loader, training):
        model.train(training); sums = {"total": 0., "position": 0., "direction": 0.}; seen = 0
        with torch.set_grad_enabled(training):
            for x, pose in loader:
                x, pose = x.to(device, non_blocking=True), pose.to(device, non_blocking=True)
                if training: optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    total, position, direction = criterion(model(x), pose)
                if training:
                    amp.scale(total).backward(); amp.unscale_(optimizer)
                    if grad_clip_norm > 0: nn.utils.clip_grad_norm_(base.parameters(), grad_clip_norm)
                    amp.step(optimizer); amp.update()
                for key, value in (("total", total), ("position", position), ("direction", direction)): sums[key] += value.item() * len(x)
                seen += len(x)
        return {key: value / max(seen, 1) for key, value in sums.items()}

    for epoch in range(1, epochs + 1):
        start = time.time(); train, val = run(train_loader, True), run(val_loader, False); scheduler.step()
        entry = {"epoch": epoch, "train": train, "val": val, "lr": optimizer.param_groups[0]["lr"], "seconds": time.time() - start}; history.append(entry)
        (ckpt_dir / "train_log.json").write_text(json.dumps(history, indent=2))
        _save(ckpt_dir / "mobileposenet_last.pt", epoch, base, optimizer, scheduler, best, entry)
        if val["total"] < best: best = val["total"]; _save(best_path, epoch, base, optimizer, scheduler, best, entry)
        print(f"[MobilePoseNet] {epoch:03d}/{epochs}: val={val['total']:.6f} pos={val['position']:.6f} dir={val['direction']:.6f} ({entry['seconds']:.1f}s)")
    return best_path
