"""Train MobilePoseNet on this repository's synthetic scalar 8×8 Hall data."""
from __future__ import annotations
import argparse, pickle, random, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "Network_Architecture"))
from train_mvec import MultiMemmapDataset, prepare_cache
from Benchmark.MobilePosenet.training import train_theoretical

class NoiseDataset(Dataset):
    def __init__(self, base, noise_std=0.): self.base, self.noise_std = base, noise_std
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        x, pose = self.base[i]
        return (x + torch.randn_like(x) * self.noise_std if self.noise_std else x), pose

def main():
    data = ROOT / "Dataset"; p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--synthetic-voltage", type=Path, default=data / "synthetic_grid_data.csv"); p.add_argument("--synthetic-label", type=Path, default=data / "synthetic_grid_coordinates.csv")
    p.add_argument("--raw-voltage", type=Path, default=data / "Grid_data.csv"); p.add_argument("--clean-voltage", type=Path, default=data / "Grid_data_computed.csv"); p.add_argument("--raw-label", type=Path, default=data / "Grid_points_coordinates.csv")
    p.add_argument("--ckpt-dir", type=Path, required=True); p.add_argument("--cache-dir", type=Path, required=True); p.add_argument("--epochs", type=int, default=256); p.add_argument("--batch-size", type=int, default=3200); p.add_argument("--lr", type=float, default=1e-4); p.add_argument("--beta", type=float, default=1.); p.add_argument("--save-every", type=int, default=5, help="Save a resumable checkpoint every N epochs."); p.add_argument("--noise-std", type=float, default=0., help="Gaussian std in MinMax-scaled input space; 0 disables augmentation.")
    p.add_argument("--val-ratio", type=float, default=.02); p.add_argument("--seed", type=int, default=42); p.add_argument("--num-workers", type=int, default=0); p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto"); p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True); a = p.parse_args()
    if not 0 < a.val_ratio < 1: raise ValueError("--val-ratio must be in (0,1)")
    for path in (a.synthetic_voltage, a.synthetic_label, a.raw_voltage, a.clean_voltage, a.raw_label):
        if not path.is_file(): raise FileNotFoundError(path)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    cfg = SimpleNamespace(raw_voltage=str(a.raw_voltage), clean_voltage=str(a.clean_voltage), raw_label=str(a.raw_label), synthetic_voltage=str(a.synthetic_voltage), synthetic_label=str(a.synthetic_label), val_ratio=a.val_ratio, seed=a.seed, rebuild_cache=False)
    arrays, split, scalers = prepare_cache(cfg, a.cache_dir, None)
    train = NoiseDataset(MultiMemmapDataset([arrays[3], arrays[4]], split["train_b"]), a.noise_std)
    val = NoiseDataset(MultiMemmapDataset([arrays[3], arrays[4]], split["val_b"]))
    device = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available() else ("cpu" if a.device == "auto" else a.device)); a.ckpt_dir.mkdir(parents=True, exist_ok=True)
    with (a.ckpt_dir / "scalers.pkl").open("wb") as f: pickle.dump(scalers, f)
    loaders = DataLoader(train, batch_size=a.batch_size, shuffle=True, num_workers=a.num_workers, pin_memory=device.type == "cuda"), DataLoader(val, batch_size=a.batch_size, num_workers=a.num_workers, pin_memory=device.type == "cuda")
    print(train_theoretical(*loaders, ckpt_dir=a.ckpt_dir, epochs=a.epochs, lr=a.lr, beta=a.beta, save_every=a.save_every, device=device, compile_model=a.compile))
if __name__ == "__main__": main()
