"""Reusable loaders and epoch functions for m-vector train and HPO scripts."""
from __future__ import annotations

import platform
import torch
from torch.utils.data import DataLoader, RandomSampler

from loss_mvec import HuberPoseLossMVec


def default_num_workers(device=None) -> int:
    return 4 if (device is not None and device.type == "cuda" and platform.system() != "Darwin") else 0


def make_loaders(train_ds, val_ds, batch_size, num_workers, device, prefetch_factor=2,
                 samples_per_epoch=0):
    common = {"pin_memory": device.type == "cuda", "num_workers": num_workers,
              "persistent_workers": num_workers > 0}
    if num_workers > 0:
        common["prefetch_factor"] = prefetch_factor
    if samples_per_epoch > 0:
        sampler = RandomSampler(train_ds, replacement=False,
                                num_samples=min(samples_per_epoch, len(train_ds)))
        train = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, drop_last=True, **common)
    else:
        train = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **common)
    val = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common)
    return train, val


def make_criterion(cfg, scalers, device):
    physics = cfg.lambda_physics > 0
    return HuberPoseLossMVec(
        lambda_ori=cfg.lambda_ori, delta_xyz=cfg.delta_xyz, lambda_pos=cfg.lambda_pos,
        lambda_physics=cfg.lambda_physics, physics_delta=cfg.physics_delta,
        calib_physical_csv=cfg.calib_physical_csv if physics else None,
        calib_alpha_csv=cfg.calib_alpha_csv if physics else None,
        volt_scaler=scalers["volt"] if physics else None,
        label_scaler=scalers["label"] if physics else None,
    ).to(device)


def _run_epoch(model, loader, criterion, device, optimizer=None, amp_scaler=None):
    is_train, use_amp = optimizer is not None, device.type == "cuda"
    model.train(is_train)
    totals = {"loss": 0.0, "xyz": 0.0, "ori": 0.0, "physics": 0.0}
    n_seen = 0
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            if is_train: optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(x_batch)
                loss, loss_xyz, loss_ori = criterion(pred, y_batch, X_b=x_batch)
            if not torch.isfinite(loss):
                continue
            if is_train:
                amp_scaler.scale(loss).backward()
                amp_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                amp_scaler.step(optimizer)
                amp_scaler.update()
            n = len(x_batch)
            totals["loss"] += loss.item() * n
            totals["xyz"] += loss_xyz.item() * n
            totals["ori"] += loss_ori.item() * n
            totals["physics"] += criterion.latest_loss_physics.item() * n
            n_seen += n
    if n_seen == 0:
        return {key: float("inf") for key in totals}, 0
    return {key: value / n_seen for key, value in totals.items()}, n_seen


def train_one_epoch(model, loader, criterion, optimizer, amp_scaler, device):
    return _run_epoch(model, loader, criterion, device, optimizer, amp_scaler)


def validate(model, loader, criterion, device):
    return _run_epoch(model, loader, criterion, device)
