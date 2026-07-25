import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import least_squares

# =============================================================================
# FILE PATHS
# =============================================================================
BASE_DIR = Path(r"/Users/tuananhnguyen/Downloads/Hallsensor_final/Data set 18.6") #MAC

SENSOR_POSITIONS_PATH = BASE_DIR / "Hall_sensor_positions.csv"   #tọa độ sensors gốc

ROBOT_POSE_PATH = BASE_DIR / "grid_points_coordinates.csv"

VOLTAGE_DATA_PATH = BASE_DIR / "grid_data.csv"

OFFSET_FILE_PATH = BASE_DIR / "Offset_Sens.csv"

# ---- output (KHONG con Calibration_Alpha.csv vi khong con Stage 2) ----
PHYSICAL_OUTPUT_PATH = BASE_DIR / "Calibration_Physical_no_alpha.csv"


# =============================================================================
# CONSTANTS
# =============================================================================

MU0_OVER_4PI = 1e-7

# SO DIEM DUOC CHON NGAU NHIEN TU TOAN BO tap du lieu (KHONG con chia region)
N_RANDOM_SAMPLES = 200

# ----  Stage 1 regularization weights (physical priors)  ----
# These penalize deviation from the design/nominal sensor pose so the
# optimizer only moves a parameter away from its nominal value when the
# voltage data actually demands it. Offset is intentionally NOT regularized.
# NOTE: theta/phi (orientation) removed entirely -- sensor direction is
# fixed to straight-up [0, 0, 1], so there is no orientation prior/lambda.
LAMBDA_POS = 2000   # position prior weight (x, y, z)   [1/m^2 scale]
LAMBDA_GAIN = 2e-3   # gain prior weight                  [1/(V/T)^2 scale]

# =============================================================================
# DIPOLE MODEL
# =============================================================================

def dipole_field(r_vec, m_vec):
    """Calculate magnetic field from dipole model"""
    r = np.linalg.norm(r_vec, axis=1, keepdims=True)
    
    r3 = np.maximum(r**3, 1e-12)
    r5 = np.maximum(r**5, 1e-12)
    
    mdotr = np.sum(m_vec * r_vec, axis=1, keepdims=True)
    
    B = MU0_OVER_4PI * (
        3.0 * r_vec * mdotr / r5 - m_vec / r3
    )
    
    return B


# =============================================================================
# LOAD SENSOR POSITIONS
# =============================================================================

def load_sensor_positions(file_path):
    """Load sensor positions from CSV"""
    df = pd.read_csv(file_path)
    sensor_positions = df.values
    print(f"Loaded sensor positions: {sensor_positions.shape}")
    return sensor_positions

# =============================================================================
# LOAD ROBOT POSE
# =============================================================================

def load_robot_pose(file_path):
    """Load robot positions and magnetic orientations"""
    df = pd.read_csv(file_path)
    
    required_cols = ['x', 'y', 'z', 'mx', 'my', 'mz']
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    positions = df[['x', 'y', 'z']].values
    m_world = df[['mx', 'my', 'mz']].values
    
    # Normalize magnetic orientation
    norm = np.linalg.norm(m_world, axis=1, keepdims=True)
    m_world = m_world / norm
    
    print(f"Loaded robot positions: {positions.shape}")
    print(f"Loaded magnetic orientations: {m_world.shape}")
    
    return positions, m_world


# =============================================================================
# LOAD VOLTAGE DATA
# =============================================================================

def load_voltage_data(file_path):
    """Load voltage measurements"""
    df = pd.read_csv(file_path)
    voltage = df.values
    print(f"Loaded voltage data: {voltage.shape}")
    return voltage


# =============================================================================
# LOAD OFFSET FILE
# =============================================================================

def load_sensor_offsets(file_path):
    """Load sensor offsets"""
    df = pd.read_csv(file_path, header=0)
    offsets = df.iloc[:, 1].values
    print(f"Loaded offsets: {offsets.shape}")
    return offsets


# =============================================================================
# RESIDUAL FUNCTION (KHONG DOI so voi ban goc)
# =============================================================================

def sensor_residuals(params, robot_positions, m_world, voltage_sensor,
                      pos_prior=None, g0=7.5,
                      lambda_pos=LAMBDA_POS, lambda_gain=LAMBDA_GAIN):
    """
    Calculate residuals between measured and predicted voltages, PLUS
    regularization residuals that pull (x,y,z,gain) toward their
    nominal/physical prior values. Offset 'a' is left unregularized.

    params: [x, y, z, a, g]
    Sensor direction is FIXED to straight up [0, 0, 1] -- theta/phi are no
    longer free parameters (sensors are assumed soldered perpendicular to
    the PCB with negligible tilt).

    pos_prior: (x0, y0, z0) nominal sensor position from
               Hall_sensor_positions.csv (design position). If None, no
               position regularization terms are appended (kept for safety;
               in normal use this is always provided by the caller).
    """
    x, y, z, a, g = params

    # Fixed sensor direction (straight up, perpendicular to PCB)
    sensor_dir = np.array([0.0, 0.0, 1.0])

    # Sensor position
    sensor_pos = np.array([x, y, z])

    # Vector from robot to sensor
    r_vec = sensor_pos - robot_positions

    # Calculate magnetic field at sensor position
    B = dipole_field(r_vec, m_world)

    # Project magnetic field onto sensor direction
    B_proj = B @ sensor_dir

    # Predicted voltage
    voltage_pred = a + g * B_proj

    # ---- Voltage residual (unchanged) ----
    r_voltage = voltage_sensor - voltage_pred

    if not np.all(np.isfinite(r_voltage)):
        print(f"WARNING: Non-finite residuals detected at params: {params}")
        print(f"Max robot distance norm: {np.max(np.linalg.norm(r_vec, axis=1))}")

    # ---- Regularization residuals (physical priors) ----
    # NOTE: concatenated onto the residual vector, NOT added to the scalar
    # cost -- appending sqrt(lambda) * (param - prior) as extra residual
    # entries is exactly equivalent to adding lambda * (param - prior)^2
    # to the cost, keeping us fully inside least_squares()/'trf'.
    if pos_prior is not None:
        x0, y0, z0 = pos_prior
        r_pos = np.sqrt(lambda_pos) * np.array([x - x0, y - y0, z - z0])
    else:
        r_pos = np.array([])

    r_gain = np.sqrt(lambda_gain) * np.array([g - g0])

    residual = np.concatenate([r_voltage, r_pos, r_gain])

    return residual


# =============================================================================
# SINGLE SENSOR CALIBRATION (KHONG DOI so voi ban goc)
# =============================================================================

def calibrate_single_sensor(
        sensor_index,
        sensor_pos_init,
        robot_positions,
        m_world,
        voltage_sensor,
        offset_init=1.618):
    """
    Calibrate a single sensor -- position + offset + gain only.
    Sensor direction is fixed straight up (perpendicular to PCB);
    theta/phi are no longer free parameters.
    """
    # ----------------------------------------------------
    # INITIAL VALUES
    # ----------------------------------------------------
    g0 = 7.5  # Initial gain (V/T)
    offset_init = 1.618         # a (offset)

    # ----------------------------------------------------
    # INITIAL PARAMETERS
    # ----------------------------------------------------
    x0 = np.array([
        sensor_pos_init[0],  # x
        sensor_pos_init[1],  # y
        sensor_pos_init[2],  # z
        offset_init,         # a (offset)
        g0                   # g (gain)
    ])
    
    # ----------------------------------------------------
    # BOUNDS
    # ----------------------------------------------------
    pos_tol = 0.0013  # 1.3mm tolerance for sensor position
    lower = [
        sensor_pos_init[0] - pos_tol,    # x min
        sensor_pos_init[1] - pos_tol,    # y min
        sensor_pos_init[2] - pos_tol,    # z min
        offset_init - 0.02,             # a min (20mV)
        6.9                               # g min
    ]
    
    upper = [
        sensor_pos_init[0] + pos_tol,    # x max
        sensor_pos_init[1] + pos_tol,    # y max
        sensor_pos_init[2] + pos_tol,    # z max
        offset_init + 0.02,             # a max (20mV)
        8                                # g max
    ]
    
    # ----------------------------------------------------
    # OPTIMIZATION
    # ----------------------------------------------------
    result = least_squares(
        sensor_residuals,
        x0,
        bounds=(lower, upper),
        args=(
            robot_positions,
            m_world,
            voltage_sensor
        ),
        kwargs=dict(
            pos_prior=(sensor_pos_init[0], sensor_pos_init[1], sensor_pos_init[2]),
            g0=g0,
        ),
        method='trf',
        max_nfev=200           
    )
    
    # ----------------------------------------------------
    # EXTRACT OPTIMIZED PARAMETERS
    # ----------------------------------------------------
    params_opt = result.x
    
    # Direction is fixed straight up -- theta/phi kept at 0 rad so the
    # OUTPUT FORMAT (10 columns) stays identical to the previous script
    # for downstream compatibility.
    theta_opt, phi_opt = 0.0, 0.0
    nx, ny, nz = 0.0, 0.0, 1.0
    
    # Create extended parameter array for saving
    params_extended = np.array([
        params_opt[0],  # x
        params_opt[1],  # y
        params_opt[2],  # z
        params_opt[3],  # a
        params_opt[4],  # g
        nx, ny, nz,     # direction vector components (fixed straight up)
        theta_opt,      # theta (radians, fixed = 0)
        phi_opt         # phi (radians, fixed = 0)
    ])
    
    # ----------------------------------------------------
    # CALCULATE RMSE
    # (result.fun = [voltage residuals, pos reg, gain reg];
    #  RMSE must be computed from the voltage part only, so it still
    #  means "V measured vs V predicted" and stays comparable to before)
    # ----------------------------------------------------
    n_voltage = voltage_sensor.shape[0]
    rmse = np.sqrt(np.mean(result.fun[:n_voltage]**2))
    
    angle_from_vertical = np.rad2deg(abs(theta_opt))
    print(f"Sensor {sensor_index+1:02d} | RMSE = {rmse:.6f} | "
          f"Angle from Z = {angle_from_vertical:.2f}° | "
          f"Dir = [{nx:.3f}, {ny:.3f}, {nz:.3f}]")
    
    return params_extended, rmse


# =============================================================================
# FULL CALIBRATION (KHONG DOI so voi ban goc)
# =============================================================================

def run_calibration(
        sensor_positions,
        offsets,
        robot_positions,
        m_world,
        voltage_data):
    """
    Calibrate all sensors
    """
    n_sensors = sensor_positions.shape[0]
    results = []
    rmses = []
    
    for i in range(n_sensors):
        params, rmse = calibrate_single_sensor(
            sensor_index=i,
            sensor_pos_init=sensor_positions[i],
            robot_positions=robot_positions,
            m_world=m_world,
            voltage_sensor=voltage_data[:, i]
        )
        
        results.append(params)
        rmses.append(rmse)
    
    return np.array(results), np.array(rmses)


# =============================================================================
# MOI: RANDOM SAMPLE SELECTION (thay cho select_stage1_calibration_set theo region)
# =============================================================================

def select_random_calibration_set(
        robot_positions,
        m_world,
        voltage_data,
        n_samples=N_RANDOM_SAMPLES):
    """
    Chon ngau nhien n_samples diem tu TOAN BO tap du lieu, KHONG chia theo
    region/h nua (khac voi select_stage1_calibration_set ban goc). Dung
    chung 1 tap ngau nhien nay cho ca 64 sensor (giong cach lam cu, chi la
    khong con stratify theo h).
    """
    n_total = robot_positions.shape[0]

    if n_samples > n_total:
        raise ValueError(
            f"n_samples ({n_samples}) > so mau co san ({n_total})"
        )

    # NOTE: khong dat random seed, giu dung tinh chat "khong seed" nhu
    # select_region_samples() ban goc.
    calib_indices = np.random.choice(n_total, size=n_samples, replace=False)

    rp_calib = robot_positions[calib_indices]
    mw_calib = m_world[calib_indices]
    vd_calib = voltage_data[calib_indices]

    print(f"\n[Calibration set] Chon ngau nhien {n_samples} / {n_total} "
          f"mau (khong chia region)")

    return calib_indices, rp_calib, mw_calib, vd_calib


# =============================================================================
# SAVE STAGE 1 RESULT (KHONG DOI so voi ban goc)
# =============================================================================

def save_physical_results(results, output_file):
    """
    Save calibrated physical parameters to Calibration_Physical.csv
    with columns: sensor_index, x, y, z, offset, gain, theta, phi
    """
    df = pd.DataFrame({
        "sensor_index": np.arange(len(results)),
        "x": results[:, 0],
        "y": results[:, 1],
        "z": results[:, 2],
        "offset": results[:, 3],
        "gain": results[:, 4],
        "theta": results[:, 8],
        "phi": results[:, 9],
    })
    df.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")


# =============================================================================
# PLOT RMSE (KHONG DOI so voi ban goc)
# =============================================================================

def plot_rmse(rmses):
    """Plot RMSE for all sensors"""
    plt.figure(figsize=(10, 5))
    plt.bar(np.arange(len(rmses)), rmses)
    plt.xlabel("Sensor Index")
    plt.ylabel("RMSE")
    plt.title("Calibration RMSE")
    plt.grid(True)
    plt.show()


# =============================================================================
# MAIN (rut gon: chi 1 stage, khong con alpha/Stage 2)
# =============================================================================

def main():
    """Main execution function -- single-stage calibration, no alpha"""

    # Load data
    sensor_positions = load_sensor_positions(SENSOR_POSITIONS_PATH)
    robot_positions, m_world = load_robot_pose(ROBOT_POSE_PATH)
    voltage_data = load_voltage_data(VOLTAGE_DATA_PATH)
    offsets = load_sensor_offsets(OFFSET_FILE_PATH)

    # Ensure consistent number of samples
    n_samples = min(len(robot_positions), len(voltage_data))
    robot_positions = robot_positions[:n_samples]
    m_world = m_world[:n_samples]
    voltage_data = voltage_data[:n_samples]

    # ==================================================================
    # SAMPLE SELECTION: 200 diem NGAU NHIEN tu toan bo tap du lieu
    # (khong con chia theo region nhu ban goc)
    # ==================================================================
    print("\n===================================")
    print("SAMPLE SELECTION (random, no region)")
    print("===================================")
    calib_indices, rp_calib, mw_calib, vd_calib = select_random_calibration_set(
        robot_positions, m_world, voltage_data,
        n_samples=N_RANDOM_SAMPLES
    )

    # ==================================================================
    # PHYSICAL PARAMETER CALIBRATION
    # (unchanged residual function, dipole model, initial values, bounds)
    # ==================================================================
    print("\n===================================")
    print("PHYSICAL PARAMETER FIT")
    print("===================================")
    results, rmses = run_calibration(
        sensor_positions, offsets, rp_calib, mw_calib, vd_calib
    )

    print("\n========================")
    print(f"Mean RMSE = {np.mean(rmses):.6f}")
    print(f"Max RMSE  = {np.max(rmses):.6f}")
    print(f"Min RMSE  = {np.min(rmses):.6f}")
    print("========================")

    # Save physical parameters (KHONG con Stage 2 / alpha)
    save_physical_results(results, PHYSICAL_OUTPUT_PATH)

    # Plots
    plot_rmse(rmses)

    print("\n===================================")
    print("FINISHED (no alpha / no Stage 2)")
    print("===================================")


if __name__ == "__main__":
    main()