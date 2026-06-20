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

OUTPUT_REGION1 = BASE_DIR / "calibration_region1.csv"
OUTPUT_REGION2 = BASE_DIR / "calibration_region2.csv"
OUTPUT_REGION3 = BASE_DIR / "calibration_region3.csv"
# =============================================================================
# CONSTANTS
# =============================================================================

MU0_OVER_4PI = 1e-7

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
        3.0 * r_vec * mdotr / r5
        - m_vec / r3
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
# RESIDUAL FUNCTION
# =============================================================================

def sensor_residuals(params, robot_positions, m_world, voltage_sensor):
    """
    Calculate residuals between measured and predicted voltages
    params: [x, y, z, a, g, theta, phi]
    theta: angle from z-axis (0 = straight up)
    phi: azimuthal angle in xy-plane
    """
    x, y, z, a, g, theta, phi = params
    
    # Convert angles to direction vector
    sensor_dir = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])
    
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
    
    # Residual
    residual = voltage_sensor - voltage_pred
    
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
    """
    Calibrate a single sensor with direction angles
    """
    # ----------------------------------------------------
    # INITIAL VALUES
    # ----------------------------------------------------
    g0 = 7.5  # Initial gain (V/T)
    theta0 = 0.0  # Initial theta (pointing straight up)
    phi0 = 0.0    # Initial phi
    
    # ----------------------------------------------------
    # INITIAL PARAMETERS
    # ----------------------------------------------------
    x0 = np.array([
        sensor_pos_init[0],  # x
        sensor_pos_init[1],  # y
        sensor_pos_init[2],  # z
        offset_init,         # a (offset)
        g0,                  # g (gain)
        theta0,              # theta
        phi0                 # phi
    ])
    
    # ----------------------------------------------------
    # BOUNDS
    # ----------------------------------------------------
    pos_tol = 0.005  # 5mm tolerance for sensor position
    theta_max = np.deg2rad(30)  # Max 30 degrees from vertical
    
    lower = [
        sensor_pos_init[0] - pos_tol,    # x min
        sensor_pos_init[1] - pos_tol,    # y min
        sensor_pos_init[2] - pos_tol,    # z min
        offset_init - 0.005,             # a min (5mV)
        -1e6,                            # g min
        -theta_max,                      # theta min
        -np.pi                           # phi min (full rotation)
    ]
    
    upper = [
        sensor_pos_init[0] + pos_tol,    # x max
        sensor_pos_init[1] + pos_tol,    # y max
        sensor_pos_init[2] + pos_tol,    # z max
        offset_init + 0.005,             # a max (5mV)
        1e6,                             # g max
        theta_max,                       # theta max
        np.pi                            # phi max (full rotation)
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
        method='trf',
        max_nfev=500
    )
    
    # ----------------------------------------------------
    # EXTRACT OPTIMIZED PARAMETERS
    # ----------------------------------------------------
    params_opt = result.x
    
    # Convert angles to direction vector for output
    theta_opt, phi_opt = params_opt[5], params_opt[6]
    nx = np.sin(theta_opt) * np.cos(phi_opt)
    ny = np.sin(theta_opt) * np.sin(phi_opt)
    nz = np.cos(theta_opt)
    
    # Create extended parameter array for saving
    params_extended = np.array([
        params_opt[0],  # x
        params_opt[1],  # y
        params_opt[2],  # z
        params_opt[3],  # a
        params_opt[4],  # g
        nx, ny, nz,     # direction vector components
        theta_opt,      # theta (radians)
        phi_opt         # phi (radians)
    ])
    
    # ----------------------------------------------------
    # CALCULATE RMSE
    # ----------------------------------------------------
    rmse = np.sqrt(np.mean(result.fun**2))
    
    angle_from_vertical = np.rad2deg(abs(theta_opt))
    print(f"Sensor {sensor_index+1:02d} | RMSE = {rmse:.6f} | "
          f"Angle from Z = {angle_from_vertical:.2f}° | "
          f"Dir = [{nx:.3f}, {ny:.3f}, {nz:.3f}]")
    
    return params_extended, rmse


# =============================================================================
# FULL CALIBRATION
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
            offset_init=offsets[i],
            robot_positions=robot_positions,
            m_world=m_world,
            voltage_sensor=voltage_data[:, i]
        )
        
        results.append(params)
        rmses.append(rmse)
    
    return np.array(results), np.array(rmses)


# =============================================================================
# REGION SELECTION
# =============================================================================

def select_region_samples(
        robot_positions,
        m_world,
        voltage_data,
        sensor_z,
        h_min,
        h_max=None,
        max_samples=2000):
    """
    Select samples based on height range
    """
    h = robot_positions[:, 2] - sensor_z
    
    if h_max is None:
        region_idx = np.where(h >= h_min)[0]
    else:
        region_idx = np.where((h >= h_min) & (h < h_max))[0]
    
    print(f"\nRegion [{h_min}, {h_max}] contains {len(region_idx)} samples")
    
    if len(region_idx) > max_samples:
        region_idx = np.random.choice(
            region_idx,
            size=max_samples,
            replace=False
        )
    
    print(f"Using {len(region_idx)} samples")
    
    return (
        robot_positions[region_idx],
        m_world[region_idx],
        voltage_data[region_idx]
    )


# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_results(results, rmses, output_file):
    """
    Save calibration results to CSV
    """
    df = pd.DataFrame({
        "sensor_id": np.arange(len(results)),
        "x": results[:, 0],
        "y": results[:, 1],
        "z": results[:, 2],
        "offset_a": results[:, 3],
        "gain_g": results[:, 4],
        "nx": results[:, 5],
        "ny": results[:, 6],
        "nz": results[:, 7],
        "theta_rad": results[:, 8],
        "phi_rad": results[:, 9],
        "angle_from_z_deg": np.rad2deg(np.abs(results[:, 8])),
        "rmse": rmses
    })
    
    df.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")


# =============================================================================
# PLOT RMSE
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
# PLOT DIRECTION VECTORS
# =============================================================================

def plot_direction_vectors(results):
    """Plot direction vectors of all sensors"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    sensor_ids = np.arange(len(results))
    
    # Subplot 1: Direction components
    ax1 = axes[0]
    ax1.plot(sensor_ids, results[:, 5], 'r.-', label='nx', markersize=8)
    ax1.plot(sensor_ids, results[:, 6], 'g.-', label='ny', markersize=8)
    ax1.plot(sensor_ids, results[:, 7], 'b.-', label='nz', markersize=8)
    ax1.set_xlabel("Sensor Index")
    ax1.set_ylabel("Direction Component")
    ax1.set_title("Sensor Direction Components")
    ax1.legend()
    ax1.grid(True)
    
    # Subplot 2: Angular deviation from vertical
    ax2 = axes[1]
    angle_from_vertical = np.rad2deg(np.abs(results[:, 8]))
    ax2.bar(sensor_ids, angle_from_vertical)
    ax2.set_xlabel("Sensor Index")
    ax2.set_ylabel("Angle (degrees)")
    ax2.set_title("Angular Deviation from Z-axis")
    ax2.grid(True)
    
    # Subplot 3: Azimuthal angle
    ax3 = axes[2]
    phi_deg = np.rad2deg(results[:, 9])
    ax3.bar(sensor_ids, phi_deg)
    ax3.set_xlabel("Sensor Index")
    ax3.set_ylabel("Phi (degrees)")
    ax3.set_title("Azimuthal Angle (XY-plane)")
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main execution function"""
    
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
    
    # Sensor reference height
    sensor_z = sensor_positions[0, 2]
    print(f"\nSensor reference z = {sensor_z:.6f} m")
    
    # Define regions
    regions = [
        ("REGION 1", 0.025, 0.045, OUTPUT_REGION1),
        ("REGION 2", 0.045, 0.065, OUTPUT_REGION2),
        ("REGION 3", 0.065, None, OUTPUT_REGION3)
    ]
    
    # Process each region
    for region_name, h_min, h_max, output_file in regions:
        print("\n===================================")
        print(region_name)
        print("===================================")
        
        # Select samples for this region
        rp, mw, vd = select_region_samples(
            robot_positions, m_world, voltage_data,
            sensor_z, h_min, h_max, max_samples=2000
        )
        
        if len(rp) < 20:
            print(f"{region_name}: too few samples ({len(rp)}), skipping...")
            continue
        
        # Run calibration
        results, rmses = run_calibration(
            sensor_positions, offsets, rp, mw, vd
        )
        
        # Print statistics
        print("\n========================")
        print(f"Mean RMSE = {np.mean(rmses):.6f}")
        print(f"Max RMSE  = {np.max(rmses):.6f}")
        print(f"Min RMSE  = {np.min(rmses):.6f}")
        print("========================")
        
        # Save results
        save_results(results, rmses, output_file)
        
        # Plot RMSE
        plot_rmse(rmses)
        
        # Plot direction vectors
        plot_direction_vectors(results)
    
    print("\n===================================")
    print("ALL REGIONS FINISHED")
    print("===================================")


if __name__ == "__main__":
    main()