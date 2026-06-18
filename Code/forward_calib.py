import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import least_squares

# =============================================================================
# FILE PATHS
# =============================================================================
BASE_DIR = Path(r"D:\Downloads\Hallsensor_final\Data set 18.6")

SENSOR_POSITIONS_PATH = BASE_DIR / "Hall_sensor_positions.csv"

ROBOT_POSE_PATH = BASE_DIR / "grid_add_random_coordinates.csv"

VOLTAGE_DATA_PATH = BASE_DIR / "grid_add_random_data.csv"

OFFSET_FILE_PATH = BASE_DIR / "Offset_Sens.csv"

OUTPUT_PATH = BASE_DIR / "Calibration_GRID_results.csv"

# =============================================================================
# CONSTANTS
# =============================================================================

MU0_OVER_4PI = 1e-7

SENSOR_DIR = np.array([0.0, 0.0, 1.0])

# =============================================================================
# DIPOLE MODEL
# =============================================================================

def dipole_field(r_vec, m_vec):

    r = np.linalg.norm(r_vec, axis=1, keepdims=True)

    r3 = np.maximum(r**3, 1e-12)

    r5 = np.maximum(r**5, 1e-12)

    mdotr = np.sum(m_vec * r_vec, axis=1, keepdims=True)

    B = MU0_OVER_4PI * (
        3.0 * r_vec * mdotr / r5
        - m_vec / r3
    )

    return B


# =============================================================================
# LOAD SENSOR POSITIONS
# =============================================================================

def load_sensor_positions(file_path):

    df = pd.read_csv(file_path)

    sensor_positions = df.values

    print(
        f"Loaded sensor positions: {sensor_positions.shape}"
    )

    return sensor_positions


# =============================================================================
# LOAD ROBOT POSE
# =============================================================================

def load_robot_pose(file_path):

    df = pd.read_csv(file_path)

    required_cols = [
        'x',
        'y',
        'z',
        'mx',
        'my',
        'mz'
    ]

    for c in required_cols:

        if c not in df.columns:

            raise ValueError(
                f"Missing column: {c}"
            )

    positions = df[
        ['x', 'y', 'z']
    ].values

    m_world = df[
        ['mx', 'my', 'mz']
    ].values

    # normalize magnetic orientation

    norm = np.linalg.norm(
        m_world,
        axis=1,
        keepdims=True
    )

    m_world = m_world / norm

    print(
        f"Loaded robot positions: {positions.shape}"
    )

    print(
        f"Loaded magnetic orientations: {m_world.shape}"
    )

    return positions, m_world


# =============================================================================
# LOAD VOLTAGE DATA
# =============================================================================

def load_voltage_data(file_path):

    df = pd.read_csv(file_path)

    voltage = df.values

    print(
        f"Loaded voltage data: {voltage.shape}"
    )

    return voltage


# =============================================================================
# LOAD OFFSET FILE
# =============================================================================

def load_sensor_offsets(file_path):

    df = pd.read_csv(
        file_path,
        header=0
    )

    offsets = df.iloc[:, 1].values

    print(
        f"Loaded offsets: {offsets.shape}"
    )

    return offsets


# =============================================================================
# RESIDUAL FUNCTION
# =============================================================================

def sensor_residuals(
        params,
        robot_positions,
        m_world,
        voltage_sensor):

    x, y, z, a, g = params

    sensor_pos = np.array([
        x,
        y,
        z
    ])

    r_vec = (
        sensor_pos
        - robot_positions
    )

    B = dipole_field(
        r_vec,
        m_world
    )

    B_proj = B @ SENSOR_DIR

    voltage_pred = (
        a
        + g * B_proj
    )

    residual = (
        voltage_sensor
        - voltage_pred
    )

    return residual


# =============================================================================
# SINGLE SENSOR CALIBRATION
# =============================================================================

def calibrate_single_sensor(
        sensor_index,
        sensor_pos_init,
        offset_init,
        robot_positions,
        m_world,
        voltage_sensor):
    # ----------------------------------------------------
    # KHỞI TẠO GAIN CỐ ĐỊNH = 7.5 V/T
    # ----------------------------------------------------
    g0 = 7.5  # <--- Initial gain value
    # ----------------------------------------------------
    x0 = np.array([
        sensor_pos_init[0],
        sensor_pos_init[1],
        sensor_pos_init[2],
        offset_init,
        g0
    ])
    pos_tol = 0.005 # 5mm tolerance for sensor position
    lower = [
        sensor_pos_init[0] - pos_tol,
        sensor_pos_init[1] - pos_tol,
        sensor_pos_init[2] - pos_tol,
        offset_init - 0.005, # 5mV tolerance for offset
        -1e6  # bound cho gain 
    ]
    upper = [

        sensor_pos_init[0] + pos_tol,
        sensor_pos_init[1] + pos_tol,
        sensor_pos_init[2] + pos_tol,
        offset_init + 0.005, # 5mV tolerance for offset
        1e6  # bound cho gain
    ]
    result = least_squares(
        sensor_residuals,
        x0,
        bounds=(lower, upper),
        args=(
            robot_positions,
            m_world,
            voltage_sensor
        ),
        method='trf',
        max_nfev=500
    )
    rmse = np.sqrt(
        np.mean(
            result.fun**2
        )
    )

    print(
        f"Sensor {sensor_index+1:02d}"
        f" | RMSE = {rmse:.6f}"
    )

    return result.x, rmse


# =============================================================================
# FULL CALIBRATION
# =============================================================================

def run_calibration(
        sensor_positions,
        offsets,
        robot_positions,
        m_world,
        voltage_data):

    n_sensors = sensor_positions.shape[0]

    results = []

    rmses = []

    for i in range(n_sensors):

        params, rmse = calibrate_single_sensor(

            sensor_index=i,

            sensor_pos_init=sensor_positions[i],

            offset_init=offsets[i],

            robot_positions=robot_positions,

            m_world=m_world,

            voltage_sensor=voltage_data[:, i]
        )

        results.append(
            params
        )

        rmses.append(
            rmse
        )

    return (
        np.array(results),
        np.array(rmses)
    )


# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_results(
        results,
        rmses,
        output_file):

    df = pd.DataFrame({

        "sensor_id":
            np.arange(
                len(results)
            ),

        "x":
            results[:, 0],

        "y":
            results[:, 1],

        "z":
            results[:, 2],

        "offset_a":
            results[:, 3],

        "gain_g":
            results[:, 4],

    })

    df.to_csv(
        output_file,
        index=False
    )

    print(
        f"\nSaved: {output_file}"
    )


# =============================================================================
# PLOT RMSE
# =============================================================================

def plot_rmse(rmses):

    plt.figure(
        figsize=(10, 5)
    )

    plt.bar(
        np.arange(
            len(rmses)
        ),
        rmses
    )

    plt.xlabel(
        "Sensor Index"
    )

    plt.ylabel(
        "RMSE"
    )

    plt.title(
        "Calibration RMSE"
    )

    plt.grid(True)

    plt.show()


# =============================================================================
# MAIN
# =============================================================================

def main():

    sensor_positions = load_sensor_positions(
        SENSOR_POSITIONS_PATH
    )

    robot_positions, m_world = load_robot_pose(
        ROBOT_POSE_PATH
    )

    voltage_data = load_voltage_data(
        VOLTAGE_DATA_PATH
    )

    offsets = load_sensor_offsets(
        OFFSET_FILE_PATH
    )

    n_samples = min(

        len(robot_positions),

        len(voltage_data)
    )

    robot_positions = (
        robot_positions[:n_samples]
    )

    m_world = (
        m_world[:n_samples]
    )

    voltage_data = (
        voltage_data[:n_samples]
    )

    results, rmses = run_calibration(

        sensor_positions,

        offsets,

        robot_positions,

        m_world,

        voltage_data
    )

    print("\n========================")

    print(
        f"Mean RMSE = {np.mean(rmses):.6f}"
    )

    print(
        f"Max RMSE  = {np.max(rmses):.6f}"
    )

    print(
        f"Min RMSE  = {np.min(rmses):.6f}"
    )

    print("========================")

    save_results(
        results,
        rmses,
        OUTPUT_PATH
    )

    plot_rmse(
        rmses
    )


if __name__ == "__main__":

    main()
