from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MU0_OVER_4PI = 1e-7
NUM_SENSORS = 64


def parse_args():
    data_dir = Path(__file__).resolve().parent.parent / "Data_8_2026"
    parser = argparse.ArgumentParser(
        description=("Compute Hall voltages using V = offset + alpha(h) * gain * Bz. "
                     "The default files are the Helix trajectory and its calibrations."))
    parser.add_argument("--input", type=Path,
                        default=data_dir / "Grid_points_coordinates.csv",
                        help="Input pose CSV with columns x,y,z,mx,my,mz")
    parser.add_argument("--out", type=Path,
                        default=data_dir / "Grid_data_computed.csv",
                        help="Output voltage CSV with columns sensor_1,...,sensor_64")
    parser.add_argument("--calib_physical_csv", type=Path,
                        default=data_dir / "Calibration_Physical_new.csv",
                        help="Physical calibration CSV (sensor positions, offset, gain)")
    parser.add_argument("--calib_alpha_csv", type=Path,
                        default=data_dir / "Calibration_Alpha_new.csv",
                        help="Alpha calibration CSV containing c0 and c1")
    parser.add_argument("--chunksize", type=int, default=1000,
                        help="Rows processed at once; use 0 to load all rows")
    return parser.parse_args()


def load_calibration(physical_path: Path, alpha_path: Path):
    physical = pd.read_csv(physical_path)
    required = {"sensor_index", "x", "y", "z", "offset", "gain"}
    missing = required.difference(physical.columns)
    if missing:
        raise ValueError(f"{physical_path} is missing columns: {sorted(missing)}")
    physical = physical.sort_values("sensor_index").reset_index(drop=True)
    expected_indices = np.arange(NUM_SENSORS)
    if len(physical) != NUM_SENSORS or not np.array_equal(physical["sensor_index"].to_numpy(), expected_indices):
        raise ValueError("Physical calibration must contain sensor_index 0 through 63 exactly once")

    alpha_df = pd.read_csv(alpha_path)
    if not {"coefficient", "value"}.issubset(alpha_df.columns):
        raise ValueError(f"{alpha_path} must contain coefficient,value columns")
    alpha = dict(zip(alpha_df["coefficient"].astype(str).str.strip(), alpha_df["value"].astype(float)))
    if not {"c0", "c1"}.issubset(alpha):
        raise ValueError(f"{alpha_path} must define c0 and c1")

    return (
        physical[["x", "y", "z"]].to_numpy(dtype=np.float64),
        physical["offset"].to_numpy(dtype=np.float64),
        physical["gain"].to_numpy(dtype=np.float64),
        float(alpha["c0"]), float(alpha["c1"]),
    )


def compute_voltage(positions, moments, sensor_pos, offset, gain, c0, c1):
    """Vectorized forward model; returns an ``(N, 64)`` voltage array."""
    norms = np.linalg.norm(moments, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("Input contains a zero-length magnetic moment vector")
    moments = moments / norms

    r_vec = sensor_pos[None, :, :] - positions[:, None, :]       # (N,64,3)
    r_norm = np.linalg.norm(r_vec, axis=2, keepdims=True)
    if np.any(r_norm <= 1e-12):
        raise ValueError("A magnet position coincides with a sensor position")
    m_dot_r = np.sum(moments[:, None, :] * r_vec, axis=2, keepdims=True)
    b_vec = MU0_OVER_4PI * (
        3.0 * m_dot_r * r_vec / r_norm**5 - moments[:, None, :] / r_norm**3)
    bz = b_vec[..., 2]                                             # fixed sensor direction [0,0,1]

    h = positions[:, 2:3] - sensor_pos[None, :, 2]                # (N,64)
    alpha_h = c0 + c1 * h
    return offset[None, :] + gain[None, :] * bz * alpha_h


def pose_arrays(frame: pd.DataFrame):
    required = ["x", "y", "z", "mx", "my", "mz"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Input pose CSV is missing columns: {missing}")
    values = frame[required].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Input pose CSV contains non-numeric or non-finite values")
    return values[:, :3], values[:, 3:]


def main():
    args = parse_args()
    if args.chunksize < 0:
        raise ValueError("chunksize must be >= 0")
    input_path, output_path = args.input, args.out
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sensor_pos, offset, gain, c0, c1 = load_calibration(args.calib_physical_csv, args.calib_alpha_csv)
    columns = [f"sensor_{index}" for index in range(1, NUM_SENSORS + 1)]
    print(f"Calibration: {args.calib_physical_csv}")
    print(f"alpha(h) = {c0:.8f} + {c1:.8f} * h")

    chunks = [pd.read_csv(input_path)] if args.chunksize == 0 else pd.read_csv(input_path, chunksize=args.chunksize)
    n_rows, first = 0, True
    for frame in chunks:
        positions, moments = pose_arrays(frame)
        voltage = compute_voltage(positions, moments, sensor_pos, offset, gain, c0, c1)
        pd.DataFrame(voltage, columns=columns).to_csv(output_path, mode="w" if first else "a",
                                                        header=first, index=False,float_format="%.6f")
        n_rows += len(frame)
        first = False

    print(f"Saved {n_rows:,} rows x {NUM_SENSORS} sensors -> {output_path}")


if __name__ == "__main__":
    main()
