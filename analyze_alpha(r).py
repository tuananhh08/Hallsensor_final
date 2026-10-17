from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MU0_OVER_4PI = 1e-7


def dipole_field(r_vec: np.ndarray, m_vec: np.ndarray) -> np.ndarray:
    """Return the dipole field (Tesla) for sample-wise vectors."""
    r = np.linalg.norm(r_vec, axis=1, keepdims=True)
    r3 = np.maximum(r**3, 1e-12)
    r5 = np.maximum(r**5, 1e-12)
    m_dot_r = np.sum(m_vec * r_vec, axis=1, keepdims=True)
    return MU0_OVER_4PI * (3.0 * r_vec * m_dot_r / r5 - m_vec / r3)


def read_calibration(path: Path, required_columns: set[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    if df["sensor_index"].duplicated().any():
        raise ValueError(f"{path.name} contains duplicate sensor_index values.")
    return df.sort_values("sensor_index").reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    dataset_dir = project_dir / "Dataset"
    parser = argparse.ArgumentParser(
        description="Create one alpha(r) scatter/linear-fit plot for each Helix-2 sensor."
    )
    parser.add_argument("--coordinates", type=Path,
                        default=dataset_dir / "Grid_add_Random_points_coordinates.csv")
    parser.add_argument("--voltages", type=Path,
                        default=dataset_dir / "Grid_add_Random_data.csv")
    parser.add_argument("--parameters", type=Path,
                        default=dataset_dir / "Calibration_Parameters_r_per_sensor.csv")
    parser.add_argument("--alpha-parameters", type=Path,
                        default=dataset_dir / "Calibration_Alpha_r_per_sensor.csv")
    parser.add_argument("--output-dir", type=Path,
                        default=project_dir / "Reporting" / "analyze_alpha(r)_Grid_add_Random")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.coordinates, args.voltages, args.parameters, args.alpha_parameters):
        if not path.is_file():
            raise FileNotFoundError(f"Input file not found: {path}")

    pose_df = pd.read_csv(args.coordinates)
    pose_columns = {"x", "y", "z", "mx", "my", "mz"}
    missing_pose = pose_columns - set(pose_df.columns)
    if missing_pose:
        raise ValueError(f"{args.coordinates.name} is missing columns: {sorted(missing_pose)}")

    voltage_df = pd.read_csv(args.voltages)
    physical = read_calibration(
        args.parameters, {"sensor_index", "x", "y", "z", "offset", "gain"}
    )
    alpha_fit = read_calibration(args.alpha_parameters, {"sensor_index", "c0", "c1"})

    n_sensors = len(physical)
    expected_indices = np.arange(n_sensors)
    if not np.array_equal(physical["sensor_index"].to_numpy(), expected_indices):
        raise ValueError("Physical calibration sensor_index must be consecutive from 0.")
    if not np.array_equal(alpha_fit["sensor_index"].to_numpy(), expected_indices):
        raise ValueError("Alpha calibration sensor_index must match physical calibration.")
    if voltage_df.shape[1] != n_sensors:
        raise ValueError(
            f"Voltage file has {voltage_df.shape[1]} columns but calibration has {n_sensors} sensors."
        )
    if len(voltage_df) != len(pose_df):
        raise ValueError(
            f"Voltage file has {len(voltage_df)} rows but coordinate file has {len(pose_df)} rows."
        )

    magnet_pos = pose_df[["x", "y", "z"]].to_numpy(dtype=float)
    magnet_m = pose_df[["mx", "my", "mz"]].to_numpy(dtype=float)
    m_norm = np.linalg.norm(magnet_m, axis=1, keepdims=True)
    if np.any(m_norm == 0):
        raise ValueError("Coordinate file contains a zero magnetic-moment vector.")
    magnet_m /= m_norm
    voltages = voltage_df.to_numpy(dtype=float)

    sensor_pos = physical[["x", "y", "z"]].to_numpy(dtype=float)
    offsets = physical["offset"].to_numpy(dtype=float)
    gains = physical["gain"].to_numpy(dtype=float)
    c0 = alpha_fit["c0"].to_numpy(dtype=float)
    c1 = alpha_fit["c1"].to_numpy(dtype=float)
    calibration_values = np.r_[sensor_pos.ravel(), offsets, gains, c0, c1]
    if not np.isfinite(calibration_values).all():
        raise ValueError("Calibration files contain missing or non-finite values.")
    if np.any(gains == 0):
        raise ValueError("Calibration file contains a zero gain.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for sensor_index in range(n_sensors):
        # Identical r-vector convention to calib_linear_r_per_sensor.py.
        r_vec = sensor_pos[sensor_index] - magnet_pos
        r_distance = np.linalg.norm(r_vec, axis=1)
        bz = dipole_field(r_vec, magnet_m)[:, 2]
        denominator = bz * gains[sensor_index]
        alpha = (voltages[:, sensor_index] - offsets[sensor_index]) / denominator

        valid = np.isfinite(r_distance) & np.isfinite(alpha) & (np.abs(denominator) > 0)
        if not valid.any():
            print(f"Sensor {sensor_index + 1:02d}: no finite alpha values; skipped.")
            continue

        r_valid = r_distance[valid]
        alpha_valid = alpha[valid]
        r_line = np.linspace(r_valid.min(), r_valid.max(), 300)
        alpha_pred = c0[sensor_index] + c1[sensor_index] * r_line

        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.scatter(r_valid, alpha_valid, s=20, alpha=0.72, color="tab:blue",
                   edgecolors="none", label="Empirical alpha")
        ax.plot(r_line, alpha_pred, color="tab:red", linewidth=2.2,
                label=f"Predicted: {c0[sensor_index]:.5f} + {c1[sensor_index]:.5f} r")
        ax.set_xlabel("r = ||sensor position - magnet position|| (m)")
        ax.set_ylabel(r"$\alpha = (V - offset)/(B_z \cdot gain)$")
        ax.set_title(f"Sensor {sensor_index + 1:02d}: alpha vs r (Grid_add_Random)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        fig.tight_layout()
        output_path = args.output_dir / f"sensor_{sensor_index + 1:02d}_alpha_vs_r.png"
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        saved += 1

    print(f"Saved {saved} plots to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
