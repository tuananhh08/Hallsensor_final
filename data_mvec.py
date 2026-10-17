from __future__ import annotations

import itertools
import json
import os
import pickle
import shutil
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

CACHE_VERSION = 1
VOLT_FILE = "volt_scaled.npy"
LABEL_FILE = "label_scaled.npy"
SCALER_FILE = "scalers.pkl"
SPLIT_FILE = "split_info2.json"
MANIFEST_FILE = "manifest.json"


class StreamingMinMaxScaler:
    """Small pickle-friendly MinMaxScaler subset; avoids a sklearn runtime dependency."""
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.data_min_ = self.data_max_ = None

    def partial_fit(self, values):
        values = np.asarray(values, dtype=np.float32)
        current_min, current_max = values.min(axis=0), values.max(axis=0)
        self.data_min_ = current_min if self.data_min_ is None else np.minimum(self.data_min_, current_min)
        self.data_max_ = current_max if self.data_max_ is None else np.maximum(self.data_max_, current_max)
        return self

    def transform(self, values):
        scale = self.data_max_ - self.data_min_
        safe_scale = np.where(scale == 0, 1.0, scale)
        low, high = self.feature_range
        return ((np.asarray(values, dtype=np.float32) - self.data_min_) / safe_scale * (high - low) + low)


class PosLabelScaler:
    """Scale xyz only; the unit magnetic-moment vector is unchanged."""

    def __init__(self):
        self.xyz_scaler = StreamingStandardScaler()

    def fit(self, labels):
        self.xyz_scaler.fit(labels[:, :3])
        return self

    def partial_fit(self, labels):
        self.xyz_scaler.partial_fit(labels[:, :3])
        return self

    def transform(self, labels):
        out = np.asarray(labels, dtype=np.float32).copy()
        out[:, :3] = self.xyz_scaler.transform(out[:, :3]).astype(np.float32)
        return out

    def inverse_transform(self, labels_scaled):
        out = np.asarray(labels_scaled, dtype=np.float32).copy()
        out[:, :3] = self.xyz_scaler.inverse_transform(out[:, :3]).astype(np.float32)
        return out


class StreamingStandardScaler:
    """Streaming StandardScaler subset with sklearn-compatible fitted attributes."""
    def __init__(self):
        self.n_samples_seen_ = 0
        self.mean_ = self.var_ = self.scale_ = None

    def partial_fit(self, values):
        values = np.asarray(values, dtype=np.float64)
        n = len(values)
        if n == 0: return self
        mean, var = values.mean(axis=0), values.var(axis=0)
        if self.n_samples_seen_ == 0:
            total_mean, total_var = mean, var
        else:
            total = self.n_samples_seen_ + n
            delta = mean - self.mean_
            total_mean = self.mean_ + delta * n / total
            total_var = (self.n_samples_seen_ * self.var_ + n * var + delta ** 2 * self.n_samples_seen_ * n / total) / total
        self.n_samples_seen_ += n
        self.mean_, self.var_ = total_mean, total_var
        self.scale_ = np.sqrt(np.maximum(total_var, 0.0))
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def fit(self, values):
        self.n_samples_seen_, self.mean_, self.var_, self.scale_ = 0, None, None, None
        return self.partial_fit(values)

    def transform(self, values):
        return (np.asarray(values, dtype=np.float32) - self.mean_) / self.scale_

    def inverse_transform(self, values):
        return np.asarray(values, dtype=np.float32) * self.scale_ + self.mean_


def _source_info(path: str) -> dict:
    resolved = Path(path).expanduser().resolve()
    stat = resolved.stat()
    return {"path": str(resolved), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _expected_manifest(voltage_path, label_path, val_ratio, seed):
    v, y = _source_info(voltage_path), _source_info(label_path)
    return {
        "cache_version": CACHE_VERSION,
        "voltage_path": v["path"], "voltage_size": v["size"], "voltage_mtime_ns": v["mtime_ns"],
        "label_path": y["path"], "label_size": y["size"], "label_mtime_ns": y["mtime_ns"],
        "val_ratio": float(val_ratio), "seed": int(seed), "dtype": "float32",
    }


def _cache_is_valid(cache_dir: Path, expected: dict) -> bool:
    try:
        manifest = json.loads((cache_dir / MANIFEST_FILE).read_text())
        if any(manifest.get(k) != v for k, v in expected.items()):
            return False
        n = int(manifest["n_total"])
        return (np.load(cache_dir / VOLT_FILE, mmap_mode="r").shape == (n, 64)
                and np.load(cache_dir / LABEL_FILE, mmap_mode="r").shape == (n, 6)
                and (cache_dir / SCALER_FILE).is_file()
                and (cache_dir / SPLIT_FILE).is_file())
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def _numeric_frame(chunk: pd.DataFrame, width: int, name: str) -> np.ndarray:
    if chunk.shape[1] != width:
        raise ValueError(f"{name} must contain {width} columns; found {chunk.shape[1]}")
    return chunk.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32, na_value=np.nan)


def _valid_chunks(voltage_path, label_path, chunksize):
    """Yield paired valid rows. A row is retained only if both inputs are valid."""
    v_iter = pd.read_csv(voltage_path, header=0, chunksize=chunksize)
    y_iter = pd.read_csv(label_path, header=0, chunksize=chunksize)
    for part, pair in enumerate(itertools.zip_longest(v_iter, y_iter)):
        if pair[0] is None or pair[1] is None:
            raise ValueError("Voltage and label CSV files have different numbers of rows/chunks")
        volt = _numeric_frame(pair[0], 64, "Voltage CSV")
        label = _numeric_frame(pair[1], 6, "Label CSV")
        if len(volt) != len(label):
            raise ValueError(f"CSV row mismatch in chunk {part}: {len(volt)} voltage vs {len(label)} label rows")
        norms = np.linalg.norm(label[:, 3:6], axis=1)
        mask = np.isfinite(volt).all(axis=1) & np.isfinite(label).all(axis=1) & (norms > 1e-12)
        volt, label, norms = volt[mask], label[mask], norms[mask]
        label[:, 3:6] /= norms[:, None]
        yield volt, label


def _atomic_json(path: Path, payload: dict) -> None:
    temp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    temp.write_text(json.dumps(payload, indent=2))
    os.replace(temp, path)


def prepare_data_cache(voltage_path: str, label_path: str, cache_dir: str,
                       val_ratio: float = 0.2, seed: int = 42, force: bool = False,
                       chunksize: int = 100_000) -> dict:
    """Create (or validate) a canonical cache without loading the full CSV into RAM."""
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be strictly between 0 and 1")
    if chunksize <= 0:
        raise ValueError("chunksize must be positive")
    cache = Path(cache_dir).expanduser().resolve()
    expected = _expected_manifest(voltage_path, label_path, val_ratio, seed)
    if not force and _cache_is_valid(cache, expected):
        return json.loads((cache / MANIFEST_FILE).read_text())

    cache.mkdir(parents=True, exist_ok=True)
    # The manifest is the commit marker. Removing it first makes partial rebuilds unusable.
    (cache / MANIFEST_FILE).unlink(missing_ok=True)
    for name in (VOLT_FILE, LABEL_FILE, SCALER_FILE, SPLIT_FILE):
        (cache / name).unlink(missing_ok=True)

    print("[cache] Pass 1/3: validating paired CSV rows and counting samples...")
    n_total = sum(len(volt) for volt, _ in _valid_chunks(voltage_path, label_path, chunksize))
    if n_total == 0:
        raise ValueError("No paired finite samples found in the input CSV files")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_total)
    n_val = int(n_total * val_ratio)
    train_idx, val_idx = order[n_val:], order[:n_val]
    train_mask = np.zeros(n_total, dtype=bool)
    train_mask[train_idx] = True

    print("[cache] Pass 2/3: fitting scalers on training rows only...")
    volt_scaler, label_scaler = StreamingMinMaxScaler(feature_range=(0, 1)), PosLabelScaler()
    offset = 0
    for volt, label in _valid_chunks(voltage_path, label_path, chunksize):
        select = train_mask[offset:offset + len(volt)]
        if select.any():
            volt_scaler.partial_fit(volt[select])
            label_scaler.partial_fit(label[select])
        offset += len(volt)

    print("[cache] Pass 3/3: writing scaled memmaps...")
    volt_mm = np.lib.format.open_memmap(cache / VOLT_FILE, mode="w+", dtype=np.float32, shape=(n_total, 64))
    label_mm = np.lib.format.open_memmap(cache / LABEL_FILE, mode="w+", dtype=np.float32, shape=(n_total, 6))
    offset = 0
    for volt, label in _valid_chunks(voltage_path, label_path, chunksize):
        end = offset + len(volt)
        volt_mm[offset:end] = volt_scaler.transform(volt).astype(np.float32)
        label_mm[offset:end] = label_scaler.transform(label)
        offset = end
    volt_mm.flush(); label_mm.flush()
    del volt_mm, label_mm

    with (cache / SCALER_FILE).open("wb") as f:
        pickle.dump({"volt": volt_scaler, "label": label_scaler, "label_format": "mvec"}, f)
    _atomic_json(cache / SPLIT_FILE, {"train": train_idx.tolist(), "val": val_idx.tolist(), "seed": seed})
    manifest = {**expected, "n_total": n_total, "n_train": len(train_idx), "n_val": len(val_idx),
                "volt_shape": [n_total, 64], "label_shape": [n_total, 6]}
    _atomic_json(cache / MANIFEST_FILE, manifest)
    print(f"[cache] Ready: {n_total:,} samples in {cache}")
    return manifest


class MemmapPoseDataset(Dataset):
    """Dataset that opens NPY memmaps lazily in each DataLoader process."""
    def __init__(self, volt_path: str, label_path: str, indices: np.ndarray):
        self.volt_path, self.label_path = str(volt_path), str(label_path)
        self.indices = np.asarray(indices, dtype=np.int64)
        self._volt = self._label = None

    def __len__(self): return len(self.indices)

    def _ensure_open(self):
        if self._volt is None:
            self._volt = np.load(self.volt_path, mmap_mode="r")
            self._label = np.load(self.label_path, mmap_mode="r")

    def __getitem__(self, item):
        self._ensure_open()
        idx = self.indices[item]
        return (torch.from_numpy(self._volt[idx].reshape(1, 8, 8).copy()),
                torch.from_numpy(self._label[idx].copy()))

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_volt"] = state["_label"] = None
        return state


def build_datasets(voltage_path: str, label_path: str, val_ratio: float = 0.2,
                   seed: int = 42, cache_dir: str = "./data_cache",
                   rebuild_cache: bool = False):
    """Return train/validation memmap datasets and their sample counts."""
    manifest = prepare_data_cache(voltage_path, label_path, cache_dir, val_ratio, seed, rebuild_cache)
    cache = Path(cache_dir).expanduser().resolve()
    split = json.loads((cache / SPLIT_FILE).read_text())
    train = MemmapPoseDataset(cache / VOLT_FILE, cache / LABEL_FILE, np.asarray(split["train"]))
    val = MemmapPoseDataset(cache / VOLT_FILE, cache / LABEL_FILE, np.asarray(split["val"]))
    return train, val, manifest["n_train"], manifest["n_val"], str(cache)


def load_scalers(cache_dir: str) -> dict:
    with (Path(cache_dir) / SCALER_FILE).open("rb") as f:
        return pickle.load(f)


def copy_cache_metadata(cache_dir: str, run_dir: str) -> None:
    """Preserve the established test_mvec.py scaler contract for each run."""
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    for name in (SCALER_FILE, SPLIT_FILE):
        shutil.copy2(Path(cache_dir) / name, Path(run_dir) / name)
