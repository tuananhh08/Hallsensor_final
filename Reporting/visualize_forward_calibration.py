"""Create 64 measured-vs-model plots for grid_forward_new_param calibration."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MU0_OVER_4PI = 1e-7


def dipole_field(r_vec, m_vec):
    r = np.linalg.norm(r_vec, axis=1, keepdims=True)
    r3 = np.maximum(r**3, 1e-12)
    r5 = np.maximum(r**5, 1e-12)
    mdotr = np.sum(m_vec * r_vec, axis=1, keepdims=True)
    return MU0_OVER_4PI * (3.0 * r_vec * mdotr / r5 - m_vec / r3)


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", type=Path,
                        default=(root / "Calibration" / "Parameters" / "forward_calibration" /
                                 "Calibration_Parameters_forward2.csv"))
    parser.add_argument("--robot-positions", type=Path,
                        default=root / "Dataset" / "Helix_points_coordinates_2.csv")
    parser.add_argument("--voltage-data", type=Path,
                        default=root / "Dataset" / "Helix_data_2.csv")
    parser.add_argument("--output-dir", type=Path,
                        default=root / "Reporting" / "forward_calibration")
    parser.add_argument("--moment", type=float, nargs=3, default=(1.0, 0.0, 0.0),
                        metavar=("MX", "MY", "MZ"),
                        help="The fixed dipole moment used during calibration.")
    return parser.parse_args()


def load_parameters(path):
    frame = pd.read_csv(path)
    required = ["sensor_index", "x", "y", "z", "offset", "gain"]
    missing = set(required).difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame = frame.sort_values("sensor_index").reset_index(drop=True)
    if not np.array_equal(frame["sensor_index"].to_numpy(), np.arange(len(frame))):
        raise ValueError("Parameter sensor_index values must start at 0 and be consecutive.")
    values = frame[required].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{path} contains non-numeric or non-finite parameter values.")
    return frame


def predict_voltage(parameters, robot_positions, moment):
    predicted = np.empty((len(robot_positions), len(parameters)), dtype=float)
    sensor_direction = np.array([0.0, 0.0, 1.0])
    for sensor_index, row in parameters.iterrows():
        sensor_position = row[["x", "y", "z"]].to_numpy(dtype=float)
        field = dipole_field(sensor_position - robot_positions, moment)
        # Sensor direction is fixed to [0, 0, 1], therefore use Bz directly.
        predicted[:, sensor_index] = row["offset"] + row["gain"] * field[:, 2]
    return predicted


def main():
    args = parse_args()
    parameters = load_parameters(args.parameters)
    robot_frame = pd.read_csv(args.robot_positions)
    robot_positions = robot_frame[["x", "y", "z"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    voltage_measured = pd.read_csv(args.voltage_data).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    moment = np.asarray(args.moment, dtype=float)
    if (not np.isfinite(robot_positions).all() or not np.isfinite(voltage_measured).all()
            or not np.isfinite(moment).all() or np.linalg.norm(moment) <= 1e-12):
        raise ValueError("Inputs and --moment must contain finite values; --moment must be non-zero.")
    if len(robot_positions) != len(voltage_measured) or voltage_measured.shape[1] != len(parameters):
        raise ValueError("Position, voltage, and parameter sensor dimensions do not match.")

    voltage_predicted = predict_voltage(parameters, robot_positions, moment)
    residuals = voltage_measured - voltage_predicted
    rmse = np.sqrt(np.mean(residuals**2, axis=0))
    mean_abs_measured = np.mean(np.abs(voltage_measured), axis=0)
    relative_rmse_pct = np.divide(rmse, mean_abs_measured, out=np.full_like(rmse, np.nan),
                                  where=mean_abs_measured > 0) * 100.0
    overall_rmse = float(np.sqrt(np.mean(residuals**2)))
    overall_mean_abs = float(np.mean(np.abs(voltage_measured)))
    overall_relative_pct = overall_rmse / overall_mean_abs * 100.0 if overall_mean_abs else float("nan")

    image_dir = args.output_dir / "Visualization"
    image_dir.mkdir(parents=True, exist_ok=True)
    samples = np.arange(len(voltage_measured))
    for sensor_index in range(len(parameters)):
        fig, axis = plt.subplots(figsize=(10, 4))
        axis.plot(samples, voltage_measured[:, sensor_index], label="V measured", linewidth=1.0)
        axis.plot(samples, voltage_predicted[:, sensor_index], label="V computed", linewidth=1.0)
        axis.set(xlabel="Sample index", ylabel="Voltage (V)",
                 title=(f"Sensor {sensor_index + 1:02d} | RMSE = {rmse[sensor_index]:.6f} V | "
                        f"Relative RMSE = {relative_rmse_pct[sensor_index]:.2f}%"))
        axis.legend()
        axis.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(image_dir / f"sensor_{sensor_index + 1:02d}.png", dpi=120)
        plt.close(fig)

    summary = pd.DataFrame({"sensor_index": np.arange(1, len(parameters) + 1),
                            "rmse_v": rmse, "relative_rmse_pct": relative_rmse_pct})
    summary.loc[len(summary)] = ["OVERALL", overall_rmse, overall_relative_pct]
    summary_path = args.output_dir / "rmse_summary.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    for sensor_index, (sensor_rmse, sensor_relative) in enumerate(zip(rmse, relative_rmse_pct), start=1):
        print(f"Sensor {sensor_index:02d} | RMSE = {sensor_rmse:.6f} V | Relative RMSE = {sensor_relative:.2f}%")
    print(f"OVERALL | RMSE = {overall_rmse:.6f} V | Relative RMSE = {overall_relative_pct:.2f}%")
    print(f"Saved {len(parameters)} images -> {image_dir}")
    print(f"Saved RMSE summary -> {summary_path}")


if __name__ == "__main__":
    main()
