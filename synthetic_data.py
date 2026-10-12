"""
synthetic_data.py

Generate synthetic Hall-sensor grid data: spatial poses × magnet orientations,
dipole Bz model + alpha(h) calibration, chunked CSV output compatible with
train_mvec.py (6-col labels, 64-col voltages).
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

from loss_mvec import load_alpha_calib_csv, load_physical_calib_csv

# ─── Constants ────────────────────────────────────────────────────────────────
MU_0_4PI = 1e-7
NUM_SENSORS = 64

REFERENCE_QUAT_ABB = (0.0, 0.0, 1.0, 0.0)  # ABB order (qw, qx, qy, qz)
MAGNET_DIR_TOOL = np.array([1.0, 0.0, 0.0])

COORD_COLUMNS = ["x", "y", "z", "mx", "my", "mz"]
VOLT_COLUMNS = [f"sensor_{i + 1}" for i in range(NUM_SENSORS)]


@dataclass
class ROIConfig:
    x_min: float = -0.045
    x_max: float = 0.095
    
    y_min: float = 0.635
    y_max: float = 0.775
    
    z_min: float = -0.0635
    z_max: float = -0.0335
    
    num_xy: int = 16 # -> 21, step hien tai la 17.5 mm
    num_z: int = 12  # -> 16, step hien tai la 3.3 mm
    
    pitch_min: float = -20.0
    pitch_max: float = 20.0
    
    yaw_min: float = -20.0
    yaw_max: float = 20.0
    
    num_angles: int = 15 

    @property
    def num_spatial(self) -> int:
        return self.num_xy * self.num_xy * self.num_z

    @property
    def num_orientations(self) -> int:
        return self.num_angles * self.num_angles

    def total_rows(self) -> int:
        return self.num_spatial * self.num_orientations


# ─── Rotation ─────────────────────────────────────
def angles_to_quaternion(
    pitch_deg: float,
    yaw_deg: float,
    ref_quat: tuple[float, float, float, float] = REFERENCE_QUAT_ABB,
) -> tuple[float, float, float, float]:
    """Compose r_ref * R_y(pitch) * R_z(yaw); return ABB quaternion (qw, qx, qy, qz)."""
    w, x, y, z = ref_quat
    r_ref = R.from_quat([x, y, z, w])
    r_final = (
        r_ref
        * R.from_euler("y", pitch_deg, degrees=True)
        * R.from_euler("z", yaw_deg, degrees=True)
    )
    q = r_final.as_quat()  # scipy [qx, qy, qz, qw]
    return (q[3], q[0], q[1], q[2])


def quat_to_rot_matrix(q: tuple[float, float, float, float]) -> list[list[float]]:
    """Convert ABB quaternion (qw, qx, qy, qz) to 3×3 rotation matrix."""
    qw, qx, qy, qz = q
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
    return [
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ]


def magnet_vector_from_quat(q: tuple[float, float, float, float]) -> np.ndarray:
    """Apply MAGNET_DIR_TOOL in world frame; return unit magnet vector."""
    qw, qx, qy, qz = q
    m_world = R.from_quat([qx, qy, qz, qw]).apply(MAGNET_DIR_TOOL)
    norm = np.linalg.norm(m_world)
    return m_world / norm if norm > 1e-12 else m_world


def build_angle_grid(cfg: ROIConfig) -> tuple[np.ndarray, np.ndarray]:
    pitch_vals = np.linspace(cfg.pitch_min, cfg.pitch_max, cfg.num_angles)
    yaw_vals = np.linspace(cfg.yaw_min, cfg.yaw_max, cfg.num_angles)
    pitch_grid, yaw_grid = np.meshgrid(pitch_vals, yaw_vals, indexing="ij")
    return pitch_grid.ravel(), yaw_grid.ravel()


def precompute_magnet_vectors(cfg: ROIConfig) -> np.ndarray:
    """Precompute all orientation magnet vectors once; shape (num_orientations, 3)."""
    pitch_flat, yaw_flat = build_angle_grid(cfg)
    m_vecs = np.empty((len(pitch_flat), 3), dtype=np.float64)
    for i, (pitch, yaw) in enumerate(zip(pitch_flat, yaw_flat)):
        q = angles_to_quaternion(float(pitch), float(yaw))
        m_vecs[i] = magnet_vector_from_quat(q)
    return m_vecs


# ─── Spatial grid ─────────────────────────────────────────────────────────────
def build_spatial_grid(cfg: ROIConfig) -> np.ndarray:
    """
    Build (num_spatial, 3) ROI positions.

    Row order with indexing='ij': x varies slowest, y middle, z fastest.
    This differs from robot_ABB_grid (z outer → x → y) but coord/voltage
    files stay internally consistent.
    """
    x_vals = np.linspace(cfg.x_min, cfg.x_max, cfg.num_xy)
    y_vals = np.linspace(cfg.y_min, cfg.y_max, cfg.num_xy)
    z_vals = np.linspace(cfg.z_min, cfg.z_max, cfg.num_z)
    xx, yy, zz = np.meshgrid(x_vals, y_vals, z_vals, indexing="ij")
    return np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)


# ─── Physics ──────────────────────────────────────────────────────────────────
def compute_Bz(
    roi_xyz: np.ndarray,
    sensor_pos: np.ndarray,
    m_vecs: np.ndarray,
) -> np.ndarray:
    """Dipole Bz at all sensors; shapes (N,3), (64,3), (N,3) → (N,64)."""
    r_vec = sensor_pos[None, :, :] - roi_xyz[:, None, :]
    r_norm = np.linalg.norm(r_vec, axis=2, keepdims=True)
    r_norm = np.clip(r_norm, 1e-4, None)
    m_dot_r = np.sum(m_vecs[:, None, :] * r_vec, axis=2, keepdims=True)
    term1 = 3 * m_dot_r * r_vec / (r_norm**5)
    term2 = m_vecs[:, None, :] / (r_norm**3)
    B_vec = MU_0_4PI * (term1 - term2)
    return B_vec[:, :, 2]


def compute_voltage(
    roi_xyz: np.ndarray,
    bz_raw: np.ndarray,
    sensor_pos: np.ndarray,
    offset: np.ndarray,
    gain: np.ndarray,
    c0: float,
    c1: float,
) -> np.ndarray:
    """V = offset + gain * Bz_raw * (c0 + c1 * h); h = z_pose - z_sensor."""
    h = roi_xyz[:, 2:3] - sensor_pos[:, 2]
    alpha = c0 + c1 * h
    return offset[None, :] + gain[None, :] * bz_raw * alpha


# ─── IO ───────────────────────────────────────────────────────────────────────
def load_calibration(physical_path: Path, alpha_path: Path) -> dict:
    physical = load_physical_calib_csv(str(physical_path))
    alpha = load_alpha_calib_csv(str(alpha_path))
    return {
        "sensor_pos": physical["sensor_pos"],
        "offset": physical["offset"],
        "gain": physical["gain"],
        "c0": alpha["c0"],
        "c1": alpha["c1"],
    }


def write_coordinates_chunk(
    path: Path,
    roi_xyz: np.ndarray,
    m_vecs: np.ndarray,
    *,
    write_header: bool,
) -> None:
    df = pd.DataFrame(
        {
            "x": roi_xyz[:, 0],
            "y": roi_xyz[:, 1],
            "z": roi_xyz[:, 2],
            "mx": m_vecs[:, 0],
            "my": m_vecs[:, 1],
            "mz": m_vecs[:, 2],
        }
    )
    df.to_csv(path, mode="w" if write_header else "a", header=write_header, index=False)


def write_voltage_chunk(
    path: Path,
    voltages: np.ndarray,
    *,
    write_header: bool,
) -> None:
    """Write voltages rounded and formatted to seven decimal places."""
    df = pd.DataFrame(voltages, columns=VOLT_COLUMNS)
    df.to_csv(
        path,
        mode="w" if write_header else "a",
        header=write_header,
        index=False,
        float_format=None,
    )


def expand_spatial_chunk(
    xyz_chunk: np.ndarray,
    m_precomputed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Repeat each spatial point for all orientations; return (Cs*256, 3) arrays."""
    n_angles = len(m_precomputed)
    roi_xyz = np.repeat(xyz_chunk, n_angles, axis=0)
    m_vecs = np.tile(m_precomputed, (len(xyz_chunk), 1))
    return roi_xyz, m_vecs


def validate_outputs(
    coord_path: Path,
    volt_path: Path,
    expected_rows: int,
    *,
    seed: int = 42,
) -> None:
    coord_df = pd.read_csv(coord_path)
    volt_df = pd.read_csv(volt_path)

    assert len(coord_df) == expected_rows, (
        f"coord rows {len(coord_df):,} != expected {expected_rows:,}"
    )
    assert len(volt_df) == expected_rows, (
        f"voltage rows {len(volt_df):,} != expected {expected_rows:,}"
    )
    assert list(coord_df.columns) == COORD_COLUMNS, f"coord columns: {list(coord_df.columns)}"
    assert volt_df.shape[1] == NUM_SENSORS, f"voltage cols {volt_df.shape[1]} != {NUM_SENSORS}"

    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(len(coord_df), size=min(10, len(coord_df)), replace=False)
    m = coord_df.iloc[sample_idx][["mx", "my", "mz"]].to_numpy()
    norms = np.linalg.norm(m, axis=1)
    max_dev = np.max(np.abs(norms - 1.0))
    assert max_dev < 1e-5, f"||m|| deviates from 1 by {max_dev:.2e}"

    volt_num = volt_df.apply(pd.to_numeric, errors="coerce")
    assert volt_num.notna().all().all(), "voltage CSV contains non-numeric values"

    print(f"  Validation OK: {expected_rows:,} rows, ||m|| max dev {max_dev:.2e}")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_data_dir = script_dir.parent / "Data_8_2026"

    parser = argparse.ArgumentParser(description="Generate synthetic grid data for train_mvec.py")
    parser.add_argument("--data_dir", type=Path, default=default_data_dir)
    parser.add_argument("--calib_physical", type=str, default="Calibration_Physical_new.csv")
    parser.add_argument("--calib_alpha", type=str, default="Calibration_Alpha_new.csv")
    parser.add_argument("--out_coord", type=str, default="synthetic_grid_coordinates2.csv")
    parser.add_argument("--out_volt", type=str, default="synthetic_grid_data2.csv")
    parser.add_argument("--chunk_size", type=int, default=500, help="Spatial points per chunk")
    parser.add_argument(
        "--max_spatial",
        type=int,
        default=None,
        help="Limit spatial points (debug/smoke test)",
    )
    parser.add_argument("--skip_validate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ROIConfig()

    data_dir = args.data_dir.resolve()
    calib_physical = data_dir / args.calib_physical
    calib_alpha = data_dir / args.calib_alpha
    coord_path = data_dir / args.out_coord
    volt_path = data_dir / args.out_volt

    print("Loading calibration...")
    calib = load_calibration(calib_physical, calib_alpha)
    sensor_pos = calib["sensor_pos"]
    offset = calib["offset"]
    gain = calib["gain"]
    c0, c1 = calib["c0"], calib["c1"]
    print(f"  alpha(h) = {c0:.8f} + {c1:.8f} * h")

    print("Precomputing magnet vectors...")
    m_precomputed = precompute_magnet_vectors(cfg)
    print(f"  Orientations: {cfg.num_orientations}")

    print("Building spatial grid...")
    xyz = build_spatial_grid(cfg)
    if args.max_spatial is not None:
        xyz = xyz[: args.max_spatial]
        print(f"  Limited to {len(xyz):,} spatial points (--max_spatial)")
    n_spatial = len(xyz)
    expected_rows = n_spatial * cfg.num_orientations
    print(f"  Spatial points: {n_spatial:,}")
    print(f"  Total rows    : {expected_rows:,}")

    chunk_size = max(1, args.chunk_size)
    n_chunks = math.ceil(n_spatial / chunk_size)

    coord_path.parent.mkdir(parents=True, exist_ok=True)
    if coord_path.exists():
        coord_path.unlink()
    if volt_path.exists():
        volt_path.unlink()

    print(f"\nWriting chunks ({n_chunks} spatial chunks, chunk_size={chunk_size})...")
    t0 = time.perf_counter()
    rows_written = 0

    for chunk_idx in range(n_chunks):
        s0 = chunk_idx * chunk_size
        s1 = min(s0 + chunk_size, n_spatial)
        xyz_chunk = xyz[s0:s1]

        roi_xyz, m_vecs = expand_spatial_chunk(xyz_chunk, m_precomputed)
        write_header = chunk_idx == 0

        write_coordinates_chunk(coord_path, roi_xyz, m_vecs, write_header=write_header)
        bz_raw = compute_Bz(roi_xyz, sensor_pos, m_vecs)
        voltages = compute_voltage(roi_xyz, bz_raw, sensor_pos, offset, gain, c0, c1)
        write_voltage_chunk(volt_path, voltages, write_header=write_header)

        rows_written += len(roi_xyz)
        elapsed = time.perf_counter() - t0
        rate = rows_written / elapsed if elapsed > 0 else 0.0
        eta = (expected_rows - rows_written) / rate if rate > 0 else float("inf")
        print(
            f"  spatial chunk {chunk_idx + 1}/{n_chunks} | "
            f"rows {rows_written:,}/{expected_rows:,} | "
            f"elapsed {elapsed:.1f}s | ETA {eta:.1f}s"
        )

    total_elapsed = time.perf_counter() - t0
    print(f"\nDone in {total_elapsed:.1f}s")
    print(f"  {coord_path}")
    print(f"  {volt_path}")

    assert rows_written == expected_rows, f"wrote {rows_written:,} != expected {expected_rows:,}"

    if not args.skip_validate:
        print("\nValidating outputs...")
        validate_outputs(coord_path, volt_path, expected_rows)


if __name__ == "__main__":
    main()
