from __future__ import annotations

import platform
import torch
from torch.utils.data import DataLoader, RandomSampler


def default_num_workers(device=None) -> int:
    """Return a sensible default worker count for the current platform.

    Uses 4 workers on non-macOS CUDA setups; falls back to 0 elsewhere
    (macOS multiprocessing with DataLoader can be slow/buggy).
    """
    return 4 if (device is not None and device.type == "cuda" and platform.system() != "Darwin") else 0


def make_loaders(train_ds, val_ds, batch_size, num_workers, device,
                 prefetch_factor=2, samples_per_epoch=0):
    """Build train and validation DataLoaders with consistent settings.

    Args:
        train_ds:         Training dataset.
        val_ds:           Validation dataset.
        batch_size:       Batch size for both loaders.
        num_workers:      Number of DataLoader worker processes.
        device:           torch.device (used to decide pin_memory).
        prefetch_factor:  Per-worker prefetch batches (ignored when num_workers=0).
        samples_per_epoch: If > 0, randomly subsample this many training examples
                           per epoch (useful for HPO speed-ups).
    """
    common = {
        "pin_memory":          device.type == "cuda",
        "num_workers":         num_workers,
        "persistent_workers":  num_workers > 0,
    }
    if num_workers > 0:
        common["prefetch_factor"] = prefetch_factor

    if samples_per_epoch > 0:
        sampler = RandomSampler(
            train_ds, replacement=False,
            num_samples=min(samples_per_epoch, len(train_ds)),
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  sampler=sampler, drop_last=True, **common)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True, drop_last=True, **common)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common)
    return train_loader, val_loader
