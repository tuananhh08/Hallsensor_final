import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import least_squares, lsq_linear

# =============================================================================
# FILE PATHS
# =============================================================================
# BASE_DIR = Path(r"/Users/tuananhnguyen/Downloads/Hallsensor_final/Data_8_2026")  # MAC
BASE_DIR = Path(r"D:\Downloads\Hallsensor_final\Data_8_2026")  # WINDOWS

SENSOR_POSITIONS_PATH = BASE_DIR / "Hall_sensor_positions.csv"
ROBOT_POSE_PATH = BASE_DIR / "Grid_points_coordinates.csv"
VOLTAGE_DATA_PATH = BASE_DIR / "Grid_data.csv"
OFFSET_INIT_PATH = BASE_DIR / "Offset_Sens.csv"

PHYSICAL_OUTPUT_PATH = BASE_DIR / "Calibration_Physical_r.csv"
ALPHA_OUTPUT_PATH = BASE_DIR / "Calibration_Alpha_r.csv"
RMSE_OUTPUT_PATH = BASE_DIR / "Calibration_RMSE_r.png"

# =============================================================================
# CONSTANTS / SPLIT
# =============================================================================
MU0_OVER_4PI = 1e-7
N_TOTAL_CALIB_SAMPLES = 600
N_STAGE1_SAMPLES = 300
N_STAGE2_SAMPLES = N_TOTAL_CALIB_SAMPLES - N_STAGE1_SAMPLES

# =============================================================================
# STAGE 1 REGULARIZATION
# =============================================================================
LAMBDA_POS = 2000
LAMBDA_GAIN = 9e-3
LAMBDA_OFFSET = 750

# =============================================================================
# STAGE 2: GLOBAL LINEAR ALPHA(R) = C0 + C1 * R
# Same ridge + bounded least-squares algorithm as the original Stage 2.
# =============================================================================
ALPHA_C0_PRIOR = 0.2
ALPHA_C1_PRIOR = 6.7
LAMBDA_ALPHA_C0 = 1e-3
LAMBDA_ALPHA_C1 = 1e-5
ALPHA_C0_BOUNDS = (-0.3, 1.3)
ALPHA_C1_BOUNDS = (-10, 10)

# =============================================================================
# DIPOLE MODEL
# =============================================================================
def dipole_field(r_vec, m_vec):
    r = np.linalg.norm(r_vec, axis=1, keepdims=True)
    r3 = np.maximum(r**3, 1e-12)
    r5 = np.maximum(r**5, 1e-12)
    mdotr = np.sum(m_vec * r_vec, axis=1, keepdims=True)
    return MU0_OVER_4PI * (
        3.0 * r_vec * mdotr / r5 - m_vec / r3
    )

# =============================================================================
# LOADERS
# =============================================================================
def load_sensor_positions(file_path):
    df = pd.read_csv(file_path)
    sensor_positions = df.values.astype(float)
    print(f"Loaded sensor positions: {sensor_positions.shape}")
    return sensor_positions


def load_robot_pose(file_path):
    df = pd.read_csv(file_path)
    required_cols = ["x", "y", "z", "mx", "my", "mz"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    positions = df[["x", "y", "z"]].values.astype(float)
    m_world = df[["mx", "my", "mz"]].values.astype(float)
    norm = np.linalg.norm(m_world, axis=1, keepdims=True)
    if np.any(norm <= 0):
        raise ValueError("Found zero-norm magnetic orientation vector.")
    m_world = m_world / norm
    print(f"Loaded robot positions: {positions.shape}")
    print(f"Loaded magnetic orientations: {m_world.shape}")
    return positions, m_world


def load_voltage_data(file_path):
    df = pd.read_csv(file_path)
    voltage = df.values.astype(float)
    print(f"Loaded voltage data: {voltage.shape}")
    return voltage


def load_offset_initial_values(file_path, n_sensors):
    df = pd.read_csv(file_path)
    required_columns = {"sensor_index", "offset_a_V"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing column(s) in {file_path.name}: {sorted(missing_columns)}"
        )
    if df["sensor_index"].duplicated().any():
        raise ValueError("Offset_Sens.csv contains duplicate sensor_index values.")
    df = df.sort_values("sensor_index").reset_index(drop=True)
    expected_indices = np.arange(n_sensors)
    actual_indices = df["sensor_index"].to_numpy()
    if not np.array_equal(actual_indices, expected_indices):
        raise ValueError(
            "Offset_Sens.csv must contain exactly sensor_index values "
            f"0 to {n_sensors - 1}."
        )
    offsets = df["offset_a_V"].to_numpy(dtype=float)
    if not np.isfinite(offsets).all():
        raise ValueError("Column offset_a_V contains missing or non-finite values.")
    print(f"Loaded offset initial values: {offsets.shape}")
    return offsets

# =============================================================================
# STAGE 1 RESIDUAL
# =============================================================================
def sensor_residuals(
    params,
    robot_positions,
    m_world,
    voltage_sensor,
    pos_prior=None,
    offset_prior=None,
    g0=7.5,
    lambda_pos=LAMBDA_POS,
    lambda_gain=LAMBDA_GAIN,
    lambda_offset=LAMBDA_OFFSET,
):
    x, y, z, a, g = params
    sensor_dir = np.array([0.0, 0.0, 1.0])
    sensor_pos = np.array([x, y, z])
    r_vec = sensor_pos - robot_positions
    B = dipole_field(r_vec, m_world)
    B_proj = B @ sensor_dir
    voltage_pred = a + g * B_proj
    r_voltage = voltage_sensor - voltage_pred
    if not np.all(np.isfinite(r_voltage)):
        raise ValueError("Non-finite Stage 1 voltage residual detected.")

    if pos_prior is not None:
        x0, y0, z0 = pos_prior
        r_pos = np.sqrt(lambda_pos) * np.array([x - x0, y - y0, z - z0])
    else:
        r_pos = np.array([])

    r_gain = np.sqrt(lambda_gain) * np.array([g - g0])

    if offset_prior is not None:
        r_offset = np.sqrt(lambda_offset) * np.array([a - offset_prior])
    else:
        r_offset = np.array([])

    return np.concatenate([r_voltage, r_pos, r_offset, r_gain])

# =============================================================================
# STAGE 1 SINGLE SENSOR
# =============================================================================
def calibrate_single_sensor(
    sensor_index,
    sensor_pos_init,
    robot_positions,
    m_world,
    voltage_sensor,
    offset_init,
):
    g0 = 7.5
    x0 = np.array(
        [
            sensor_pos_init[0],
            sensor_pos_init[1],
            sensor_pos_init[2],
            offset_init,
            g0,
        ],
        dtype=float,
    )
    pos_tol = 0.001
    lower = [
        sensor_pos_init[0] - pos_tol,
        sensor_pos_init[1] - pos_tol,
        sensor_pos_init[2] - pos_tol,
        offset_init - 0.0011,
        7.0,
    ]
    upper = [
        sensor_pos_init[0] + pos_tol,
        sensor_pos_init[1] + pos_tol,
        sensor_pos_init[2] + pos_tol,
        offset_init + 0.0011,
        8.0,
    ]
    result = least_squares(
        sensor_residuals,
        x0,
        bounds=(lower, upper),
        args=(robot_positions, m_world, voltage_sensor),
        kwargs=dict(
            pos_prior=(sensor_pos_init[0], sensor_pos_init[1], sensor_pos_init[2]),
            offset_prior=offset_init,
            g0=g0,
        ),
        method="trf",
        max_nfev=250,
    )
    params_opt = result.x
    nx, ny, nz = 0.0, 0.0, 1.0
    theta_opt, phi_opt = 0.0, 0.0
    params_extended = np.array(
        [
            params_opt[0], params_opt[1], params_opt[2], params_opt[3], params_opt[4],
            nx, ny, nz, theta_opt, phi_opt,
        ]
    )
    n_voltage = voltage_sensor.shape[0]
    rmse = np.sqrt(np.mean(result.fun[:n_voltage] ** 2))
    print(
        f"Sensor {sensor_index + 1:02d} | RMSE = {rmse:.6f} V | "
        f"x={params_opt[0]:.6f}, y={params_opt[1]:.6f}, "
        f"z={params_opt[2]:.6f}, a={params_opt[3]:.6f}, "
        f"g={params_opt[4]:.6f}"
    )
    return params_extended, rmse

# =============================================================================
# STAGE 1 ALL SENSORS
# =============================================================================
def run_calibration(
    sensor_positions,
    robot_positions,
    m_world,
    voltage_data,
    offset_initial_values,
):
    n_sensors = sensor_positions.shape[0]
    if len(offset_initial_values) != n_sensors:
        raise ValueError(
            f"Expected {n_sensors} initial offsets, got {len(offset_initial_values)}."
        )
    results, rmses = [], []
    for i in range(n_sensors):
        params, rmse = calibrate_single_sensor(
            sensor_index=i,
            sensor_pos_init=sensor_positions[i],
            robot_positions=robot_positions,
            m_world=m_world,
            voltage_sensor=voltage_data[:, i],
            offset_init=offset_initial_values[i],
        )
        results.append(params)
        rmses.append(rmse)
    return np.array(results), np.array(rmses)

# =============================================================================
# RANDOM SPLIT
# =============================================================================
def select_stage1_stage2_split(
    robot_positions,
    m_world,
    voltage_data,
    n_total=N_TOTAL_CALIB_SAMPLES,
    n_stage1=N_STAGE1_SAMPLES,
):
    n_samples = robot_positions.shape[0]
    if n_total > n_samples:
        raise ValueError(
            f"Requested {n_total} calibration points but dataset only has {n_samples} samples."
        )
    all_idx = np.random.choice(n_samples, size=n_total, replace=False)
    stage1_idx = all_idx[:n_stage1]
    stage2_idx = all_idx[n_stage1:]
    print(f"\n[Sampling] Drew {n_total} random points out of {n_samples}.")
    print(f"  Stage 1: {len(stage1_idx)} points")
    print(f"  Stage 2: {len(stage2_idx)} points")
    stage1_data = (
        stage1_idx,
        robot_positions[stage1_idx],
        m_world[stage1_idx],
        voltage_data[stage1_idx],
    )
    stage2_data = (
        stage2_idx,
        robot_positions[stage2_idx],
        m_world[stage2_idx],
        voltage_data[stage2_idx],
    )
    return stage1_data, stage2_data

# =============================================================================
# STAGE 2 GLOBAL LINEAR ALPHA(R)
# =============================================================================
def calibrate_alpha_linear_r(
    physical_results,
    rp_calib2,
    mw_calib2,
    vd_calib2,
    c0_prior=ALPHA_C0_PRIOR,
    c1_prior=ALPHA_C1_PRIOR,
    lambda_c0=LAMBDA_ALPHA_C0,
    lambda_c1=LAMBDA_ALPHA_C1,
):
    n_samples = rp_calib2.shape[0]
    n_sensors = physical_results.shape[0]
    sensor_pos = physical_results[:, 0:3]
    a = physical_results[:, 3]
    g = physical_results[:, 4]
    sensor_dir = physical_results[:, 5:8]

    r_distance = np.zeros((n_samples, n_sensors))
    B_proj = np.zeros((n_samples, n_sensors))

    for s in range(n_sensors):
        r_vec = sensor_pos[s] - rp_calib2
        r_distance[:, s] = np.linalg.norm(r_vec, axis=1)
        B = dipole_field(r_vec, mw_calib2)
        B_proj[:, s] = B @ sensor_dir[s]

    gB = g[None, :] * B_proj
    v_minus_a = vd_calib2 - a[None, :]

    x_c0 = gB.ravel()
    x_c1 = (gB * r_distance).ravel()
    y = v_minus_a.ravel()
    X = np.column_stack([x_c0, x_c1])

    X_aug = np.vstack([
        X,
        [np.sqrt(lambda_c0), 0.0],
        [0.0, np.sqrt(lambda_c1)],
    ])
    y_aug = np.concatenate([
        y,
        [np.sqrt(lambda_c0) * c0_prior],
        [np.sqrt(lambda_c1) * c1_prior],
    ])
    bounds = (
        [ALPHA_C0_BOUNDS[0], ALPHA_C1_BOUNDS[0]],
        [ALPHA_C0_BOUNDS[1], ALPHA_C1_BOUNDS[1]],
    )
    fit = lsq_linear(X_aug, y_aug, bounds=bounds)
    c0, c1 = fit.x
    resid = y - X @ fit.x
    rmse = np.sqrt(np.mean(resid ** 2))

    print("\n[Stage 2] GLOBAL LINEAR CORRECTION")
    print(f"  alpha(r) = {c0:.6f} + ({c1:.6f}) * r")
    print(f"  Fit pairs = {len(y)}")
    print(f"  RMSE = {rmse:.6f} V")
    print(f"  r range = [{r_distance.min():.6f}, {r_distance.max():.6f}] m")

    return {
        "c0": c0,
        "c1": c1,
        "rmse": rmse,
        "r_min": r_distance.min(),
        "r_max": r_distance.max(),
    }

# =============================================================================
# SAVE
# =============================================================================
def save_physical_results(results, rmses, output_file):
    df = pd.DataFrame({
        "sensor_index": np.arange(len(results)),
        "x": results[:, 0],
        "y": results[:, 1],
        "z": results[:, 2],
        "offset": results[:, 3],
        "gain": results[:, 4],
    })
    df.to_csv(output_file, index=False)
    print(f"\nSaved Stage 1: {output_file}")


def save_alpha_results(alpha_params, output_file):
    df = pd.DataFrame({
        "coefficient": ["c0", "c1"],
        "value": [alpha_params["c0"], alpha_params["c1"]],
    })
    df.to_csv(output_file, index=False)
    print(f"Saved Stage 2: {output_file}")


def plot_rmse(rmses, output_file):
    plt.figure(figsize=(10, 5))
    plt.bar(np.arange(len(rmses)), rmses)
    plt.xlabel("Sensor Index")
    plt.ylabel("RMSE (V)")
    plt.title("Stage 1 Calibration RMSE")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\n===================================")
    print("2-STAGE CALIBRATION")
    print("Version 1: GLOBAL LINEAR ALPHA(R)")
    print("===================================\n")

    sensor_positions = load_sensor_positions(SENSOR_POSITIONS_PATH)
    robot_positions, m_world = load_robot_pose(ROBOT_POSE_PATH)
    voltage_data = load_voltage_data(VOLTAGE_DATA_PATH)
    offsets = load_offset_initial_values(
        OFFSET_INIT_PATH,
        n_sensors=sensor_positions.shape[0],
    )

    n_samples = min(len(robot_positions), len(voltage_data))
    robot_positions = robot_positions[:n_samples]
    m_world = m_world[:n_samples]
    voltage_data = voltage_data[:n_samples]

    if sensor_positions.shape[0] != voltage_data.shape[1]:
        raise ValueError(
            f"Sensor count mismatch: {sensor_positions.shape[0]} calibration sensors vs "
            f"{voltage_data.shape[1]} voltage columns."
        )

    stage1_data, stage2_data = select_stage1_stage2_split(
        robot_positions,
        m_world,
        voltage_data,
    )

    _, rp1, mw1, vd1 = stage1_data
    _, rp2, mw2, vd2 = stage2_data

    print("\n===================================")
    print("STAGE 1: PHYSICAL PARAMETER FIT")
    print("===================================\n")

    results, rmses = run_calibration(
        sensor_positions,
        rp1,
        mw1,
        vd1,
        offsets,
    )

    print("\nStage 1 statistics:")
    print(f"Mean RMSE = {np.mean(rmses):.6f} V")
    print(f"Min RMSE  = {np.min(rmses):.6f} V")
    print(f"Max RMSE  = {np.max(rmses):.6f} V")

    save_physical_results(
        results,
        rmses,
        PHYSICAL_OUTPUT_PATH,
    )
    plot_rmse(rmses, RMSE_OUTPUT_PATH)

    print("\n===================================")
    print("STAGE 2: GLOBAL LINEAR ALPHA(R)")
    print("===================================\n")

    alpha_params = calibrate_alpha_linear_r(
        results,
        rp2,
        mw2,
        vd2,
    )

    save_alpha_results(
        alpha_params,
        ALPHA_OUTPUT_PATH,
    )

    print("\n===================================")
    print("ALL STAGES FINISHED")
    print("===================================")


if __name__ == "__main__":
    main()
