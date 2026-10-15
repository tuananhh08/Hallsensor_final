"""Joint 5-parameter Hall-sensor calibration on grid measurements.

The optimisation remains the original joint forward-model fit: each sensor has
``(x, y, z, offset, gain)`` and all parameters are fitted in one least-squares
problem. Its bounds match ``calib_linear_r_per_sensor.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


MU0_OVER_4PI = 1e-7
POSITION_TOLERANCE_M = 0.0011
OFFSET_TOLERANCE_V = 0.001
GAIN_INITIAL = 7.5
GAIN_BOUNDS = (7.0, 8.0)


def dipole_field(r_vec: np.ndarray, m_vec: np.ndarray) -> np.ndarray:
    """Magnetic field of a fixed dipole at each vector from dipole to sensor."""
    r = np.linalg.norm(r_vec, axis=-1, keepdims=True)
    r3 = np.maximum(r**3, 1e-12)
    r5 = np.maximum(r**5, 1e-12)
    mdotr = np.sum(m_vec * r_vec, axis=-1, keepdims=True)
    return MU0_OVER_4PI * (3.0 * r_vec * mdotr / r5 - m_vec / r3)


class GridCalibrationModel:
    """Original joint forward model with fixed sensor directions [0, 0, 1]."""

    def __init__(self, positions_world, voltage_measured, sensor_positions, fixed_moment):
        self.positions = np.asarray(positions_world, dtype=float)
        self.voltage = np.asarray(voltage_measured, dtype=float)
        self.sensor_positions_initial = np.asarray(sensor_positions, dtype=float)
        self.fixed_moment = np.asarray(fixed_moment, dtype=float)
        self.n_sensors = self.sensor_positions_initial.shape[0]
        self.sensor_directions = np.tile(np.array([0.0, 0.0, 1.0]), (self.n_sensors, 1))

    def pack_params(self, offsets, gains, positions):
        return np.concatenate([offsets, gains, positions.ravel()])

    def unpack_params(self, params):
        offsets = params[:self.n_sensors]
        gains = params[self.n_sensors:2 * self.n_sensors]
        positions = params[2 * self.n_sensors:].reshape(self.n_sensors, 3)
        return offsets, gains, positions

    def residuals(self, params):
        offsets, gains, sensor_positions = self.unpack_params(params)
        voltage_predicted = np.empty_like(self.voltage)
        for sensor_index in range(self.n_sensors):
            r_vec = sensor_positions[sensor_index] - self.positions
            field = dipole_field(r_vec, self.fixed_moment)
            # Sensor direction is fixed to [0, 0, 1], therefore this is Bz.
            bz = field[:, 2]
            voltage_predicted[:, sensor_index] = offsets[sensor_index] + gains[sensor_index] * bz
        return (self.voltage - voltage_predicted).ravel()


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensor-positions", type=Path,
                        default=root / "Dataset" / "Hall_sensor_positions.csv")
    parser.add_argument("--robot-positions", type=Path,
                        default=root / "Dataset" / "Grid_add_Random_points_coordinates.csv")
    parser.add_argument("--voltage-data", type=Path,
                        default=root / "Dataset" / "Grid_add_Random_data.csv")
    parser.add_argument("--offset-initial", type=Path,
                        default=root / "Dataset" / "Offset_Sens.csv")
    parser.add_argument("--output", type=Path,
                        default=(root / "Calibration" / "Parameters" / "forward_calibration" /
                                 "Calibration_Parameters_forward2.csv"))
    parser.add_argument("--moment", type=float, nargs=3, default=(1.0, 0.0, 0.0),
                        metavar=("MX", "MY", "MZ"),
                        help="Fixed dipole moment used by the original algorithm.")
    parser.add_argument("--max-nfev", type=int, default=200)
    return parser.parse_args()


def _load_numeric(path, columns):
    frame = pd.read_csv(path)
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    values = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{path} contains non-numeric or non-finite values.")
    return values


def load_initial_offsets(path, n_sensors):
    frame = pd.read_csv(path)
    required = {"sensor_index", "offset_a_V"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame = frame.sort_values("sensor_index").reset_index(drop=True)
    expected = np.arange(n_sensors)
    if len(frame) != n_sensors or not np.array_equal(frame["sensor_index"].to_numpy(), expected):
        raise ValueError(f"{path} must contain sensor_index values 0 to {n_sensors - 1} exactly once.")
    offsets = pd.to_numeric(frame["offset_a_V"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(offsets).all():
        raise ValueError(f"{path} contains non-numeric or non-finite offsets.")
    return offsets


def run_grid_calibration(sensor_positions, voltage_data, robot_positions, initial_offsets, fixed_moment, max_nfev):
    """Run the unchanged joint least-squares algorithm with matched bounds."""
    model = GridCalibrationModel(robot_positions, voltage_data, sensor_positions, fixed_moment)
    initial_gains = np.full(model.n_sensors, GAIN_INITIAL)
    x0 = model.pack_params(initial_offsets, initial_gains, sensor_positions)
    lower = model.pack_params(initial_offsets - OFFSET_TOLERANCE_V,
                              np.full(model.n_sensors, GAIN_BOUNDS[0]),
                              sensor_positions - POSITION_TOLERANCE_M)
    upper = model.pack_params(initial_offsets + OFFSET_TOLERANCE_V,
                              np.full(model.n_sensors, GAIN_BOUNDS[1]),
                              sensor_positions + POSITION_TOLERANCE_M)
    print(f"Running joint calibration: {len(x0)} parameters, {len(robot_positions)} samples")
    print("Bounds: position ±0.001 m, offset ±0.0011 V, gain [7.0, 8.0].")
    result = least_squares(model.residuals, x0, bounds=(lower, upper), method="trf",
                           max_nfev=max_nfev, verbose=1)
    print(f"Finished: success={result.success}, cost={result.cost:.6g}, nfev={result.nfev}")
    return result.x


def save_calibration(params, n_sensors, output_path):
    offsets = params[:n_sensors]
    gains = params[n_sensors:2 * n_sensors]
    positions = params[2 * n_sensors:].reshape(n_sensors, 3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sensor_index": np.arange(n_sensors), "x": positions[:, 0],
                  "y": positions[:, 1], "z": positions[:, 2],
                  "offset": offsets, "gain": gains}).to_csv(output_path, index=False)
    print(f"Saved calibrated parameters -> {output_path}")


def main():
    args = parse_args()
    sensor_positions = _load_numeric(args.sensor_positions, ["x", "y", "z"])
    robot_positions = _load_numeric(args.robot_positions, ["x", "y", "z"])
    voltage_data = pd.read_csv(args.voltage_data).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(voltage_data).all() or voltage_data.shape[1] != len(sensor_positions):
        raise ValueError("Voltage CSV must be finite and have one column per sensor.")
    if len(robot_positions) != len(voltage_data):
        raise ValueError("Robot-position and voltage CSV files must have the same number of rows.")
    initial_offsets = load_initial_offsets(args.offset_initial, len(sensor_positions))
    moment = np.asarray(args.moment, dtype=float)
    if not np.isfinite(moment).all() or np.linalg.norm(moment) <= 1e-12:
        raise ValueError("--moment must be a non-zero finite vector.")
    params = run_grid_calibration(sensor_positions, voltage_data, robot_positions, initial_offsets,
                                  moment, args.max_nfev)
    # This calibration script deliberately writes only this parameter CSV.
    save_calibration(params, len(sensor_positions), args.output)


if __name__ == "__main__":
    main()
