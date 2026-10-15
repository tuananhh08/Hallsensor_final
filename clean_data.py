"""Generate model-domain (clean) Hall voltages for CaliBLocNet training.

The generated CSV is paired row-by-row with the measured/raw voltage CSV:

    raw Hall voltages  ->  CaliBLocNet  ->  clean voltages

For each pose and sensor, the clean target uses the calibration produced by
``Calibration/calib_linear_r_per_sensor.py``:

    V_clean = offset_s + gain_s * Bz_dipole * (c0_s + c1_s * r_s)

where ``r_s`` is the magnet-to-sensor distance and ``Bz_dipole`` is the
dipole field projected onto the fixed sensor direction ``[0, 0, 1]``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MU0_OVER_4PI = 1e-7
POSE_COLUMNS = ("x", "y", "z", "mx", "my", "mz")
PHYSICAL_COLUMNS = ("sensor_index", "x", "y", "z", "offset", "gain")
ALPHA_COLUMNS = ("sensor_index", "c0", "c1")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Compute clean Hall-voltage targets using per-sensor linear-r calibration."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "Dataset" / "Grid_points_coordinates.csv",
        help="Pose CSV containing x,y,z,mx,my,mz.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "Dataset" / "Grid_data_computed.csv",
        help="Output clean-voltage CSV (one sensor column per calibration row).",
    )
    parser.add_argument(
        "--calib-physical",
        type=Path,
        default=(root / "Calibration" / "Parameters" / "Linear_r_per_sensor" /
                 "Calibration_Parameters_r_per_sensor.csv"),
        help="Stage-1 per-sensor physical calibration CSV.",
    )
    parser.add_argument(
        "--calib-alpha",
        type=Path,
        default=(root / "Calibration" / "Parameters" / "Linear_r_per_sensor" /
                 "Calibration_Alpha_r_per_sensor.csv"),
        help="Stage-2 per-sensor alpha(r) calibration CSV.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=10_000,
        help="Number of pose rows to process at a time; use 0 to load all rows.",
    )
    return parser.parse_args()


def _numeric(frame: pd.DataFrame, columns: tuple[str, ...], source: Path) -> np.ndarray:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing columns: {sorted(missing)}")
    values = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{source} contains non-numeric or non-finite values in {list(columns)}")
    return values


def load_calibration(physical_path: Path, alpha_path: Path) -> tuple[np.ndarray, ...]:
    """Load and align Stage-1 and per-sensor Stage-2 calibration tables."""
    if not physical_path.is_file():
        raise FileNotFoundError(f"Physical calibration file not found: {physical_path}")
    if not alpha_path.is_file():
        raise FileNotFoundError(f"Alpha calibration file not found: {alpha_path}")

    physical = pd.read_csv(physical_path)
    alpha = pd.read_csv(alpha_path)
    _numeric(physical, PHYSICAL_COLUMNS, physical_path)
    _numeric(alpha, ALPHA_COLUMNS, alpha_path)

    if physical["sensor_index"].duplicated().any() or alpha["sensor_index"].duplicated().any():
        raise ValueError("Calibration files must not contain duplicate sensor_index values.")

    physical = physical.sort_values("sensor_index").reset_index(drop=True)
    alpha = alpha.sort_values("sensor_index").reset_index(drop=True)
    physical_indices = physical["sensor_index"].to_numpy(dtype=int)
    alpha_indices = alpha["sensor_index"].to_numpy(dtype=int)
    if not np.array_equal(physical_indices, alpha_indices):
        raise ValueError("Physical and alpha calibration files have different sensor_index values.")
    if not np.array_equal(physical_indices, np.arange(len(physical_indices))):
        raise ValueError("sensor_index must be consecutive and start at 0.")

    sensor_positions = physical[["x", "y", "z"]].to_numpy(dtype=np.float64)
    offsets = physical["offset"].to_numpy(dtype=np.float64)
    gains = physical["gain"].to_numpy(dtype=np.float64)
    c0 = alpha["c0"].to_numpy(dtype=np.float64)
    c1 = alpha["c1"].to_numpy(dtype=np.float64)
    return sensor_positions, offsets, gains, c0, c1


def compute_clean_voltage(
    positions: np.ndarray,
    moments: np.ndarray,
    sensor_positions: np.ndarray,
    offsets: np.ndarray,
    gains: np.ndarray,
    c0: np.ndarray,
    c1: np.ndarray,
) -> np.ndarray:
    """Return clean voltages with shape ``(n_poses, n_sensors)``."""
    moment_norm = np.linalg.norm(moments, axis=1, keepdims=True)
    if np.any(moment_norm <= 1e-12):
        raise ValueError("Input contains a zero-length magnetic-moment vector.")
    unit_moments = moments / moment_norm

    # r points from the magnet to each sensor: (N poses, S sensors, 3).
    r_vec = sensor_positions[None, :, :] - positions[:, None, :]
    distance = np.linalg.norm(r_vec, axis=2)
    if np.any(distance <= 1e-12):
        raise ValueError("A magnet position coincides with a sensor position.")

    m_dot_r = np.sum(unit_moments[:, None, :] * r_vec, axis=2)
    distance3 = distance ** 3
    distance5 = distance ** 5
    # Only the fixed sensor z-axis is measured, so take B[..., 2] directly.
    bz = MU0_OVER_4PI * (
        3.0 * r_vec[..., 2] * m_dot_r / distance5
        - unit_moments[:, None, 2] / distance3
    )
    alpha_r = c0[None, :] + c1[None, :] * distance
    return offsets[None, :] + gains[None, :] * bz * alpha_r


def pose_arrays(frame: pd.DataFrame, source: Path) -> tuple[np.ndarray, np.ndarray]:
    values = _numeric(frame, POSE_COLUMNS, source)
    return values[:, :3], values[:, 3:]


def main() -> None:
    args = parse_args()
    if args.chunksize < 0:
        raise ValueError("--chunksize must be >= 0")
    if not args.input.is_file():
        raise FileNotFoundError(f"Input pose file not found: {args.input}")
    if args.input.resolve() == args.output.resolve():
        raise ValueError("--input and --output must be different files.")

    sensor_positions, offsets, gains, c0, c1 = load_calibration(
        args.calib_physical, args.calib_alpha
    )
    output_columns = [f"sensor_{index}" for index in range(1, len(sensor_positions) + 1)]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    chunks = (
        [pd.read_csv(args.input)]
        if args.chunksize == 0
        else pd.read_csv(args.input, chunksize=args.chunksize)
    )
    rows_written = 0
    first_chunk = True
    for frame in chunks:
        positions, moments = pose_arrays(frame, args.input)
        voltage = compute_clean_voltage(
            positions, moments, sensor_positions, offsets, gains, c0, c1
        )
        pd.DataFrame(voltage, columns=output_columns).to_csv(
            args.output,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
            float_format="%.9g",
        )
        rows_written += len(frame)
        first_chunk = False

    print(f"Saved {rows_written:,} clean rows x {len(sensor_positions)} sensors -> {args.output}")


if __name__ == "__main__":
    main()
