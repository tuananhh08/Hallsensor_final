"""Colab-friendly CLI for the three AMagPoseNet training stages.

Run from the repository root, for example:

    python -m Benchmark.AmagPoseNet.run --stage calinet ...

All dataset locations, optimizer settings, scheduler settings, and loss weights
are CLI arguments so one launcher can be used for controlled benchmarks.
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
for module_dir in (REPO_ROOT, REPO_ROOT / "Network Architecture", REPO_ROOT / "Losses"):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

from train_mvec import MultiMemmapDataset, prepare_cache  # noqa: E402
from training_loop import default_num_workers, make_loaders  # noqa: E402
from Benchmark.AmagPoseNet.training.finetune import finetune  # noqa: E402
from Benchmark.AmagPoseNet.training.pretrain_posenet import pretrain_posenet  # noqa: E402
from Benchmark.AmagPoseNet.training.train_calinet import train_calinet  # noqa: E402


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def parse_args() -> argparse.Namespace:
    data = REPO_ROOT / "Dataset"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("calinet", "posenet", "finetune"))

    dataset = parser.add_argument_group("datasets")
    dataset.add_argument("--raw-voltage", type=_path, default=data / "Grid_add_Random_data.csv")
    dataset.add_argument("--clean-voltage", type=_path, default=data / "Grid_add_Random_data_computed.csv")
    dataset.add_argument("--raw-label", type=_path, default=data / "Grid_add_Random_points_coordinates.csv")
    dataset.add_argument("--synthetic-voltage", type=_path, default=data / "synthetic_grid_data.csv")
    dataset.add_argument("--synthetic-label", type=_path, default=data / "synthetic_grid_coordinates.csv")

    io = parser.add_argument_group("output and cache")
    io.add_argument("--ckpt-dir", required=True, type=_path)
    io.add_argument("--cache-dir", required=True, type=_path)
    io.add_argument("--scaler-file", type=_path, default=None,
                    help="Optional fitted scalers.pkl to reuse instead of fitting from supplied datasets.")
    io.add_argument("--rebuild-cache", action="store_true")
    io.add_argument("--resume", type=_path, default=None)

    split = parser.add_argument_group("split and loader")
    split.add_argument("--val-ratio", type=float, default=0.2)
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--batch-size", type=int, default=256)
    split.add_argument("--num-workers", type=int, default=None)
    split.add_argument("--prefetch-factor", type=int, default=2)
    split.add_argument("--samples-per-epoch", type=int, default=0)

    optimisation = parser.add_argument_group("optimisation")
    optimisation.add_argument("--num-epochs", type=int, default=200)
    optimisation.add_argument("--lr", type=float, default=None,
                              help="Learning rate for calinet or posenet stage.")
    optimisation.add_argument("--lr-calinet", type=float, default=2e-4)
    optimisation.add_argument("--lr-posenet", type=float, default=1e-3)
    optimisation.add_argument("--weight-decay", type=float, default=4.56e-3)
    optimisation.add_argument("--warmup-epochs", type=int, default=5)
    optimisation.add_argument("--eta-min", type=float, default=1e-6)
    optimisation.add_argument("--patience", type=int, default=45)
    optimisation.add_argument("--save-every", type=int, default=5)
    optimisation.add_argument("--grad-clip-norm", type=float, default=1.0,
                              help="Set <= 0 to disable gradient clipping.")
    optimisation.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    optimisation.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")

    loss = parser.add_argument_group("loss")
    loss.add_argument("--beta", type=float, default=1.0,
                      help="Orientation-loss weight for AMagPoseLoss.")
    loss.add_argument("--lambda-pos", type=float, default=1.0)
    loss.add_argument("--lambda-calib", type=float, default=0.1,
                      help="CaliNet loss weight used only by finetune.")
    loss.add_argument("--calib-reduction", choices=("mean", "sum"), default="mean")

    checkpoints = parser.add_argument_group("stage checkpoints")
    checkpoints.add_argument("--calinet-checkpoint", type=_path, default=None,
                             help="Required by --stage finetune.")
    checkpoints.add_argument("--posenet-checkpoint", type=_path, default=None,
                             help="Required by --stage finetune.")
    return parser.parse_args()


def get_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(choice)
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable.")
    return device


def build_loaders(args: argparse.Namespace, device: torch.device):
    cache_cfg = SimpleNamespace(
        raw_voltage=str(args.raw_voltage), clean_voltage=str(args.clean_voltage),
        raw_label=str(args.raw_label), synthetic_voltage=str(args.synthetic_voltage),
        synthetic_label=str(args.synthetic_label), val_ratio=args.val_ratio, seed=args.seed,
        rebuild_cache=args.rebuild_cache,
    )
    arrays, split, scalers = prepare_cache(cache_cfg, args.cache_dir, str(args.scaler_file) if args.scaler_file else None)
    if args.stage == "calinet":
        train_ds = MultiMemmapDataset(arrays[:2], split["train_a"])
        val_ds = MultiMemmapDataset(arrays[:2], split["val_a"])
    elif args.stage == "posenet":
        train_ds = MultiMemmapDataset([arrays[3], arrays[4]], split["train_b"])
        val_ds = MultiMemmapDataset([arrays[3], arrays[4]], split["val_b"])
    else:
        train_ds = MultiMemmapDataset(arrays[:3], split["train_a"])
        val_ds = MultiMemmapDataset(arrays[:3], split["val_a"])

    workers = default_num_workers(device) if args.num_workers is None else args.num_workers
    loaders = make_loaders(train_ds, val_ds, args.batch_size, workers, device,
                           args.prefetch_factor, args.samples_per_epoch)
    return (*loaders, scalers)


def main() -> None:
    args = parse_args()
    if not 0 < args.val_ratio < 1:
        raise ValueError("--val-ratio must be in (0, 1).")
    if args.stage == "finetune" and not (args.calinet_checkpoint and args.posenet_checkpoint):
        raise ValueError("finetune requires --calinet-checkpoint and --posenet-checkpoint.")
    for path in (args.raw_voltage, args.clean_voltage, args.raw_label,
                 args.synthetic_voltage, args.synthetic_label):
        if not path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {path}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = get_device(args.device)
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, scalers = build_loaders(args, device)
    with open(args.ckpt_dir / "scalers.pkl", "wb") as handle:
        pickle.dump(scalers, handle)
    (args.ckpt_dir / "run_config.json").write_text(
        json.dumps({key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()}, indent=2), encoding="utf-8"
    )

    common = dict(
        ckpt_dir=args.ckpt_dir, num_epochs=args.num_epochs, weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs, eta_min=args.eta_min, patience=args.patience,
        save_every=args.save_every, device=device, compile_model=args.compile,
        resume=str(args.resume) if args.resume else None, grad_clip_norm=args.grad_clip_norm,
    )
    if args.stage == "calinet":
        best = train_calinet(train_loader, val_loader,
                             lr=args.lr if args.lr is not None else args.lr_calinet,
                             calib_reduction=args.calib_reduction, **common)
    elif args.stage == "posenet":
        best = pretrain_posenet(train_loader, val_loader,
                                lr=args.lr if args.lr is not None else args.lr_posenet,
                                beta=args.beta, lambda_pos=args.lambda_pos, **common)
    else:
        best = finetune(train_loader, val_loader,
                        calinet_checkpoint=str(args.calinet_checkpoint),
                        posenet_checkpoint=str(args.posenet_checkpoint),
                        lr_calinet=args.lr_calinet, lr_posenet=args.lr_posenet,
                        beta=args.beta, lambda_pos=args.lambda_pos,
                        lambda_calib=args.lambda_calib, calib_reduction=args.calib_reduction,
                        **common)
    print(f"Best checkpoint: {best}")


if __name__ == "__main__":
    main()
