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

# ---- outputs for the 2-stage calibration framework ----
PHYSICAL_OUTPUT_PATH = BASE_DIR / "Calibration_Physical.csv"
ALPHA_OUTPUT_PATH = BASE_DIR / "Calibration_Alpha.csv"


# =============================================================================
# CONSTANTS
# =============================================================================

MU0_OVER_4PI = 1e-7

# ----  region boundaries  ----
REGION1_H_MAX = 0.040   # Region 1: h < 0.040
REGION2_H_MAX = 0.055   # Region 2: 0.040 <= h < 0.055 ; Region 3: h >= 0.055
SAMPLES_PER_REGION = 80  # Stage 1: up to 80 points sampled per region

# ----  Stage 1 regularization weights (physical priors)  ----
# These penalize deviation from the design/nominal sensor pose so the
# optimizer only moves a parameter away from its nominal value when the
# voltage data actually demands it. Offset is intentionally NOT regularized.
# NOTE: theta/phi (orientation) removed entirely -- sensor direction is
# fixed to straight-up [0, 0, 1], so there is no orientation prior/lambda.
LAMBDA_POS = 550   # position prior weight (x, y, z)   [1/m^2 scale]
LAMBDA_GAIN = 8e-4   # gain prior weight                  [1/(V/T)^2 scale]

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
# RESIDUAL FUNCTION
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
# SINGLE SENSOR CALIBRATION
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
        offset_init - 0.02,             # a min (15mV)
        6.9                               # g min
    ]
    
    upper = [
        sensor_pos_init[0] + pos_tol,    # x max
        sensor_pos_init[1] + pos_tol,    # y max
        sensor_pos_init[2] + pos_tol,    # z max
        offset_init + 0.02,             # a max (15mV)
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
    # for downstream compatibility (save_results, save_physical_results,
    # plot_direction_vectors, Stage 2 alpha).
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
        max_samples=240,
        return_indices=False):

    h = robot_positions[:, 2] - sensor_z

    lower_ok = np.ones_like(h, dtype=bool) if h_min is None else (h >= h_min)
    upper_ok = np.ones_like(h, dtype=bool) if h_max is None else (h < h_max)
    region_idx = np.where(lower_ok & upper_ok)[0]

    print(f"\nRegion [{h_min}, {h_max}] contains {len(region_idx)} samples")

    if len(region_idx) > max_samples:
        # NOTE: no random seed is set here, per the calibration spec.
        region_idx = np.random.choice(
            region_idx,
            size=max_samples,
            replace=False
        )

    print(f"Using {len(region_idx)} samples")

    if return_indices:
        return (
            robot_positions[region_idx],
            m_world[region_idx],
            voltage_data[region_idx],
            region_idx
        )

    return (
        robot_positions[region_idx],
        m_world[region_idx],
        voltage_data[region_idx]
    )


# =============================================================================
# NEW: STAGE 1 - STRATIFIED CALIBRATION SET (fixed, reused in Stage 2)
# =============================================================================

def select_stage1_calibration_set(
        robot_positions,
        m_world,
        voltage_data,
        sensor_z_ref,
        per_region=SAMPLES_PER_REGION):

    rp1, mw1, vd1, idx1 = select_region_samples(
        robot_positions, m_world, voltage_data, sensor_z_ref,
        h_min=None, h_max=REGION1_H_MAX,
        max_samples=per_region, return_indices=True
    )
    rp2, mw2, vd2, idx2 = select_region_samples(
        robot_positions, m_world, voltage_data, sensor_z_ref,
        h_min=REGION1_H_MAX, h_max=REGION2_H_MAX,
        max_samples=per_region, return_indices=True
    )
    rp3, mw3, vd3, idx3 = select_region_samples(
        robot_positions, m_world, voltage_data, sensor_z_ref,
        h_min=REGION2_H_MAX, h_max=None,
        max_samples=per_region, return_indices=True
    )

    calib_indices = np.concatenate([idx1, idx2, idx3])
    rp_calib = np.concatenate([rp1, rp2, rp3], axis=0)
    mw_calib = np.concatenate([mw1, mw2, mw3], axis=0)
    vd_calib = np.concatenate([vd1, vd2, vd3], axis=0)

    print(f"\n[Stage 1] Combined calibration set: {len(calib_indices)} "
          f"points (Region1={len(idx1)}, Region2={len(idx2)}, "
          f"Region3={len(idx3)})")

    return calib_indices, rp_calib, mw_calib, vd_calib


# =============================================================================
# NEW: STAGE 2 - CLOSED-FORM ALPHA PER REGION
# =============================================================================

def calibrate_alpha_by_region(physical_results, rp_calib, mw_calib, vd_calib):

    n_samples = rp_calib.shape[0]
    n_sensors = physical_results.shape[0]

    sensor_pos = physical_results[:, 0:3]      # (n_sensors, 3): x, y, z
    a = physical_results[:, 3]                  # (n_sensors,)  offset
    g = physical_results[:, 4]                  # (n_sensors,)  gain
    sensor_dir = physical_results[:, 5:8]       # (n_sensors, 3): nx, ny, nz

    # h[i, s] = z_capsule_i - z_sensor_calibrated_s  (uses Stage-1 result)
    z_calibrated = sensor_pos[:, 2]
    h = rp_calib[:, 2][:, None] - z_calibrated[None, :]   # (n_samples, n_sensors)

    # Raw dipole projection B_proj[i, s] using the FROZEN calibrated pose
    # (position + direction) of each sensor -- gain/offset are NOT
    # reapplied here, they're multiplied in separately below.
    B_proj = np.zeros((n_samples, n_sensors))
    for s in range(n_sensors):
        r_vec = sensor_pos[s] - rp_calib               # (n_samples, 3)
        B = dipole_field(r_vec, mw_calib)               # (n_samples, 3)
        B_proj[:, s] = B @ sensor_dir[s]

    gB = g[None, :] * B_proj                            # (n_samples, n_sensors)
    v_minus_a = vd_calib - a[None, :]                    # V_measured - a

    region_masks = {
        1: h < REGION1_H_MAX,
        2: (h >= REGION1_H_MAX) & (h < REGION2_H_MAX),
        3: h >= REGION2_H_MAX
    }

    alphas = {}
    for region, mask in region_masks.items():
        numerator = np.sum(gB[mask] * v_minus_a[mask])
        denominator = np.sum(gB[mask] ** 2)
        alpha = numerator / denominator if denominator > 0 else np.nan
        print(f"Region {region}: alpha = {alpha:.6f} "
              f"(from {np.sum(mask)} sample-sensor pairs)")
        alphas[region] = alpha

    return alphas


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
# NEW: SAVE STAGE 1 / STAGE 2 RESULTS (required output files)
# =============================================================================

def save_physical_results(results, output_file):
    """
    Save Stage 1 frozen physical parameters to Calibration_Physical.csv
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


def save_alpha_results(alphas, output_file):
    """
    Save Stage 2 region correction coefficients to Calibration_Alpha.csv
    with exactly 3 rows: Region 1, Region 2, Region 3.
    """
    regions_sorted = sorted(alphas.keys())
    df = pd.DataFrame({
        "Region": [f"Region {r}" for r in regions_sorted],
        "Alpha": [alphas[r] for r in regions_sorted],
    })
    df.to_csv(output_file, index=False)
    print(f"Saved: {output_file}")


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

# def plot_direction_vectors(results):
#     """Plot direction vectors of all sensors"""
#     fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
#     sensor_ids = np.arange(len(results))
    
#     # Subplot 1: Direction components
#     ax1 = axes[0]
#     ax1.plot(sensor_ids, results[:, 5], 'r.-', label='nx', markersize=8)
#     ax1.plot(sensor_ids, results[:, 6], 'g.-', label='ny', markersize=8)
#     ax1.plot(sensor_ids, results[:, 7], 'b.-', label='nz', markersize=8)
#     ax1.set_xlabel("Sensor Index")
#     ax1.set_ylabel("Direction Component")
#     ax1.set_title("Sensor Direction Components")
#     ax1.legend()
#     ax1.grid(True)
    
#     # Subplot 2: Angular deviation from vertical
#     ax2 = axes[1]
#     angle_from_vertical = np.rad2deg(np.abs(results[:, 8]))
#     ax2.bar(sensor_ids, angle_from_vertical)
#     ax2.set_xlabel("Sensor Index")
#     ax2.set_ylabel("Angle (degrees)")
#     ax2.set_title("Angular Deviation from Z-axis")
#     ax2.grid(True)
    
#     # Subplot 3: Azimuthal angle
#     ax3 = axes[2]
#     phi_deg = np.rad2deg(results[:, 9])
#     ax3.bar(sensor_ids, phi_deg)
#     ax3.set_xlabel("Sensor Index")
#     ax3.set_ylabel("Phi (degrees)")
#     ax3.set_title("Azimuthal Angle (XY-plane)")
#     ax3.grid(True)
    
#     plt.tight_layout()
#     plt.show()


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main execution function -- 2-stage calibration framework"""

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

    # ------------------------------------------------------------------
    # NEW: Sensor reference height for Stage 1 region assignment.
    # Must come from the ORIGINAL (uncalibrated) sensor positions file.
    # ------------------------------------------------------------------
    sensor_z_initial_ref = sensor_positions[:, 2].mean()
    print(f"\nSensor reference z (initial, for Stage 1) = "
          f"{sensor_z_initial_ref:.6f} m")

    # ==================================================================
    # STAGE 1a: Stratified, fixed calibration set (<=150 points)
    # ==================================================================
    print("\n===================================")
    print("STAGE 1: SAMPLE SELECTION")
    print("===================================")
    calib_indices, rp_calib, mw_calib, vd_calib = select_stage1_calibration_set(
        robot_positions, m_world, voltage_data, sensor_z_initial_ref,
        per_region=SAMPLES_PER_REGION
    )
    # calib_indices is frozen here and reused as-is in Stage 2 below --
    # no re-sampling happens after this point.

    # ==================================================================
    # STAGE 1b: Per-sensor physical parameter calibration
    # (unchanged residual function, dipole model, initial values, bounds)
    # ==================================================================
    print("\n===================================")
    print("STAGE 1: PHYSICAL PARAMETER FIT")
    print("===================================")
    results, rmses = run_calibration(
        sensor_positions, offsets, rp_calib, mw_calib, vd_calib
    )

    print("\n========================")
    print(f"Mean RMSE = {np.mean(rmses):.6f}")
    print(f"Max RMSE  = {np.max(rmses):.6f}")
    print(f"Min RMSE  = {np.min(rmses):.6f}")
    print("========================")

    # Save Stage 1 physical parameters (frozen going into Stage 2)
    save_physical_results(results, PHYSICAL_OUTPUT_PATH)

    # Plots
    plot_rmse(rmses)
    # plot_direction_vectors(results)

    # ==================================================================
    # STAGE 2: Closed-form alpha per region (physical params frozen)
    # ==================================================================
    print("\n===================================")
    print("STAGE 2: ALPHA CORRECTION (closed-form)")
    print("===================================")
    alphas = calibrate_alpha_by_region(results, rp_calib, mw_calib, vd_calib)
    save_alpha_results(alphas, ALPHA_OUTPUT_PATH)

    print("\n===================================")
    print("ALL STAGES FINISHED")
    print("===================================")


if __name__ == "__main__":
    main()
    
    
    
# # # """
# # # Sweep lambda_pos va lambda_gain quanh vung uoc luong co co so, ve L-curve
# # # (RMSE vs do lech tham so khoi prior) de chon lambda can bang.

# # # CACH DUNG: dan doan code nay vao CUOI file calib_region_based.py (ban da
# # # co san cac ham dipole_field, sensor_residuals, calibrate_single_sensor,
# # # select_stage1_calibration_set, load_sensor_positions, load_robot_pose,
# # # load_voltage_data, v.v. trong file goc). Script nay KHONG doi thuat toan
# # # least_squares/'trf', chi goi lai calibrate_single_sensor() nhieu lan voi
# # # cac gia tri lambda khac nhau.

# # # Diem khoi dau duoc uoc luong tu cong thuc:
# # #     lambda = (RMSE_v_no_reg / sigma)^2
# # # voi RMSE_v_no_reg lay tu log ban da chay (0.000383 V), va sigma la do lech
# # # "chap nhan duoc" ban tu chon (o day: sigma_pos=0.5mm, sigma_gain=1.0 V/T).
# # # Day chi la DIEM KHOI DAU -- sweep xung quanh no de tim diem can bang that su.
# # # """

# # import numpy as np
# # import matplotlib.pyplot as plt

# # # =============================================================================
# # # CAU HINH SWEEP
# # # =============================================================================
# # RMSE_V_NO_REG = 0.000383  # tu log khong-regularize cua ban

# # # Diem khoi dau uoc luong (KHONG phai gia tri cuoi cung)
# # LAMBDA_POS_GUESS = (RMSE_V_NO_REG / 0.0005) ** 2   # sigma_pos = 0.5 mm
# # LAMBDA_GAIN_GUESS = (RMSE_V_NO_REG / 1.0) ** 2      # sigma_gain = 1.0 V/T

# # # Sweep tu 1/100x den 100x quanh diem khoi dau, log-spaced, 9 diem
# # N_POINTS = 9
# # lambda_pos_grid = np.geomspace(LAMBDA_POS_GUESS / 100, LAMBDA_POS_GUESS * 100, N_POINTS)
# # lambda_gain_grid = np.geomspace(LAMBDA_GAIN_GUESS / 100, LAMBDA_GAIN_GUESS * 100, N_POINTS)

# # # Sensor dai dien de sweep nhanh (thay vi chay het 64 sensor cho moi lambda).
# # # Nen chon vai sensor o cac vi tri khac nhau (giua/canh mang) de dai dien.
# # SENSOR_SAMPLE_IDX = [0, 15, 31, 47, 63]  # 5 sensor rai deu tren 64 sensor


# # def run_sweep_1d(param_name, lambda_grid, sensor_positions, robot_positions,
# #                   m_world, voltage_data, sensor_z_ref):
# #     """
# #     Sweep 1 chieu: giu lambda con lai = 0, chi thay doi lambda cua param_name
# #     ('pos' hoac 'gain'). Tra ve (rmse_list, deviation_list).

# #     Goi least_squares() TRUC TIEP (khong qua calibrate_single_sensor) vi
# #     lambda_pos/lambda_gain trong sensor_residuals la default-argument, bi
# #     "dong bang" luc dinh nghia ham -- doi bien global sau do KHONG co tac
# #     dung. Truyen tuong minh qua kwargs moi dam bao dung gia tri lambda.
# #     """
# #     rmse_list = []
# #     deviation_list = []  # do lech tuong doi TB khoi prior (chuan hoa theo bound)

# #     calib_indices, rp_calib, mw_calib, vd_calib = select_stage1_calibration_set(
# #         robot_positions, m_world, voltage_data, sensor_z_ref
# #     )

# #     g0 = 7.5
# #     pos_tol = 0.0015

# #     for lam in lambda_grid:
# #         lambda_pos = lam if param_name == "pos" else 0.0
# #         lambda_gain = lam if param_name == "gain" else 0.0

# #         rmses_this_lambda = []
# #         deviations_this_lambda = []

# #         for s in SENSOR_SAMPLE_IDX:
# #             sensor_pos_init = sensor_positions[s]

# #             x0 = np.array([
# #                 sensor_pos_init[0], sensor_pos_init[1], sensor_pos_init[2],
# #                 1.618,  # offset_init
# #                 g0
# #             ])
# #             lower = [
# #                 sensor_pos_init[0] - pos_tol, sensor_pos_init[1] - pos_tol,
# #                 sensor_pos_init[2] - pos_tol, 1.618 - 0.015, 6.9
# #             ]
# #             upper = [
# #                 sensor_pos_init[0] + pos_tol, sensor_pos_init[1] + pos_tol,
# #                 sensor_pos_init[2] + pos_tol, 1.618 + 0.015, 8.0
# #             ]

# #             result = least_squares(
# #                 sensor_residuals,
# #                 x0,
# #                 bounds=(lower, upper),
# #                 args=(rp_calib, mw_calib, vd_calib[:, s]),
# #                 kwargs=dict(
# #                     pos_prior=tuple(sensor_pos_init),
# #                     g0=g0,
# #                     lambda_pos=lambda_pos,
# #                     lambda_gain=lambda_gain,
# #                 ),
# #                 method="trf",
# #                 max_nfev=200
# #             )

# #             n_voltage = vd_calib.shape[0]
# #             rmse = np.sqrt(np.mean(result.fun[:n_voltage] ** 2))
# #             rmses_this_lambda.append(rmse)

# #             if param_name == "pos":
# #                 dev = np.linalg.norm(result.x[0:3] - sensor_pos_init) / pos_tol
# #             else:  # gain
# #                 dev = abs(result.x[4] - g0) / 1.1  # chuan hoa theo bien do bound gain [6.9, 8.0]

# #             deviations_this_lambda.append(dev)

# #         rmse_list.append(np.mean(rmses_this_lambda))
# #         deviation_list.append(np.mean(deviations_this_lambda))

# #         print(f"  lambda_{param_name} = {lam:.3e} | "
# #               f"mean RMSE = {np.mean(rmses_this_lambda):.6f} | "
# #               f"mean |dev|/bound = {np.mean(deviations_this_lambda):.4f}")

# #     return rmse_list, deviation_list


# # def plot_lcurve(lambda_grid, rmse_list, deviation_list, param_name, guess_value):
# #     fig, ax1 = plt.subplots(figsize=(7, 5))

# #     color1 = "tab:blue"
# #     ax1.set_xlabel(f"lambda_{param_name}")
# #     ax1.set_ylabel("Mean RMSE (V)", color=color1)
# #     ax1.plot(lambda_grid, rmse_list, "o-", color=color1, label="RMSE")
# #     ax1.set_xscale("log")
# #     ax1.tick_params(axis="y", labelcolor=color1)

# #     ax2 = ax1.twinx()
# #     color2 = "tab:red"
# #     ax2.set_ylabel("Mean |deviation| / bound", color=color2)
# #     ax2.plot(lambda_grid, deviation_list, "s--", color=color2, label="Deviation")
# #     ax2.tick_params(axis="y", labelcolor=color2)

# #     ax1.axvline(guess_value, color="gray", linestyle=":", alpha=0.7,
# #                 label=f"initial guess = {guess_value:.2e}")

# #     fig.suptitle(f"L-curve: RMSE vs Deviation ({param_name})")
# #     fig.tight_layout()
# #     output_path = BASE_DIR / f"lcurve_{param_name}.png"
# #     fig.savefig(output_path, dpi=120)
# #     plt.close(fig)
# #     print(f"  -> Da luu: {output_path}")


# # def main_sweep():
# #     sensor_positions = load_sensor_positions(SENSOR_POSITIONS_PATH)
# #     robot_positions, m_world = load_robot_pose(ROBOT_POSE_PATH)
# #     voltage_data = load_voltage_data(VOLTAGE_DATA_PATH)

# #     # QUAN TRONG: sensor_z_ref phai la 1 gia tri SCALAR (trung binh z cua
# #     # tat ca sensor), giong het cach goc dung trong run_calibration():
# #     #   sensor_z_initial_ref = sensor_positions[:, 2].mean()
# #     # KHONG duoc truyen mang 64 phan tu vao day.
# #     sensor_z_ref = sensor_positions[:, 2].mean()

# #     print("=" * 60)
# #     print(f"Diem khoi dau uoc luong: lambda_pos = {LAMBDA_POS_GUESS:.3e}, "
# #           f"lambda_gain = {LAMBDA_GAIN_GUESS:.3e}")
# #     print("=" * 60)

# #     print("\n--- SWEEP lambda_pos (lambda_gain = 0) ---")
# #     rmse_pos, dev_pos = run_sweep_1d(
# #         "pos", lambda_pos_grid, sensor_positions, robot_positions,
# #         m_world, voltage_data, sensor_z_ref
# #     )
# #     plot_lcurve(lambda_pos_grid, rmse_pos, dev_pos, "pos", LAMBDA_POS_GUESS)

# #     print("\n--- SWEEP lambda_gain (lambda_pos = 0) ---")
# #     rmse_gain, dev_gain = run_sweep_1d(
# #         "gain", lambda_gain_grid, sensor_positions, robot_positions,
# #         m_world, voltage_data, sensor_z_ref
# #     )
# #     plot_lcurve(lambda_gain_grid, rmse_gain, dev_gain, "gain", LAMBDA_GAIN_GUESS)

# #     print("\nDa luu 2 anh L-curve vao thu muc BASE_DIR (xem duong dan o tren)")
# #     print("Chon lambda tai 'diem khuyu' (elbow): noi RMSE bat dau tang nhanh")
# #     print("nhung deviation da giam ve gan 0 -- do la vung can bang tot nhat.")


# # if __name__ == "__main__":
# #     main_sweep()




# # import numpy as np
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # from pathlib import Path
# # from scipy.optimize import least_squares

# # # =============================================================================
# # # FILE PATHS
# # # =============================================================================
# # BASE_DIR = Path(r"/Users/tuananhnguyen/Downloads/Hallsensor_final/Data set 18.6")

# # SENSOR_POSITIONS_PATH = BASE_DIR / "Hall_sensor_positions.csv"

# # ROBOT_POSE_PATH = BASE_DIR / "grid_points_coordinates.csv"

# # VOLTAGE_DATA_PATH = BASE_DIR / "grid_data.csv"

# # OFFSET_FILE_PATH = BASE_DIR / "Offset_Sens.csv"

# # # ---- outputs for the 2-stage calibration framework ----
# # PHYSICAL_OUTPUT_PATH = BASE_DIR / "Calibration_Physical.csv"
# # ALPHA_OUTPUT_PATH = BASE_DIR / "Calibration_Alpha.csv"


# # # =============================================================================
# # # CONSTANTS
# # # =============================================================================

# # MU0_OVER_4PI = 1e-7

# # # ----  region boundaries  ----
# # REGION1_H_MAX = 0.040   # Region 1: h < 0.040
# # REGION2_H_MAX = 0.055   # Region 2: 0.040 <= h < 0.055 ; Region 3: h >= 0.055
# # SAMPLES_PER_REGION = 50  # Stage 1: up to 70 points sampled per region

# # # ----  Stage 1 regularization weights (physical priors)  ----
# # # These penalize deviation from the design/nominal sensor pose so the
# # # optimizer only moves a parameter away from its nominal value when the
# # # voltage data actually demands it. Offset is intentionally NOT regularized.
# # # NOTE: theta/phi (orientation) removed entirely -- sensor direction is
# # # fixed to straight-up [0, 0, 1], so there is no orientation prior/lambda.
# # LAMBDA_POS = 0    # position prior weight (x, y, z)   [1/m^2 scale]
# # LAMBDA_GAIN = 0   # gain prior weight                  [1/(V/T)^2 scale]

# # # =============================================================================
# # # DIPOLE MODEL
# # # =============================================================================

# # def dipole_field(r_vec, m_vec):
# #     """Calculate magnetic field from dipole model"""
# #     r = np.linalg.norm(r_vec, axis=1, keepdims=True)
    
# #     r3 = np.maximum(r**3, 1e-12)
# #     r5 = np.maximum(r**5, 1e-12)
    
# #     mdotr = np.sum(m_vec * r_vec, axis=1, keepdims=True)
    
# #     B = MU0_OVER_4PI * (
# #         3.0 * r_vec * mdotr / r5 - m_vec / r3
# #     )
    
# #     return B


# # # =============================================================================
# # # LOAD SENSOR POSITIONS
# # # =============================================================================

# # def load_sensor_positions(file_path):
# #     """Load sensor positions from CSV"""
# #     df = pd.read_csv(file_path)
# #     sensor_positions = df.values
# #     print(f"Loaded sensor positions: {sensor_positions.shape}")
# #     return sensor_positions

# # # =============================================================================
# # # LOAD ROBOT POSE
# # # =============================================================================

# # def load_robot_pose(file_path):
# #     """Load robot positions and magnetic orientations"""
# #     df = pd.read_csv(file_path)
    
# #     required_cols = ['x', 'y', 'z', 'mx', 'my', 'mz']
# #     for c in required_cols:
# #         if c not in df.columns:
# #             raise ValueError(f"Missing column: {c}")

# #     positions = df[['x', 'y', 'z']].values
# #     m_world = df[['mx', 'my', 'mz']].values
    
# #     # Normalize magnetic orientation
# #     norm = np.linalg.norm(m_world, axis=1, keepdims=True)
# #     m_world = m_world / norm
    
# #     print(f"Loaded robot positions: {positions.shape}")
# #     print(f"Loaded magnetic orientations: {m_world.shape}")
    
# #     return positions, m_world


# # # =============================================================================
# # # LOAD VOLTAGE DATA
# # # =============================================================================

# # def load_voltage_data(file_path):
# #     """Load voltage measurements"""
# #     df = pd.read_csv(file_path)
# #     voltage = df.values
# #     print(f"Loaded voltage data: {voltage.shape}")
# #     return voltage


# # # =============================================================================
# # # LOAD OFFSET FILE
# # # =============================================================================

# # def load_sensor_offsets(file_path):
# #     """Load sensor offsets"""
# #     df = pd.read_csv(file_path, header=0)
# #     offsets = df.iloc[:, 1].values
# #     print(f"Loaded offsets: {offsets.shape}")
# #     return offsets


# # # =============================================================================
# # # RESIDUAL FUNCTION
# # # =============================================================================

# # def sensor_residuals(params, robot_positions, m_world, voltage_sensor,
# #                       pos_prior=None, g0=7.5,
# #                       lambda_pos=LAMBDA_POS, lambda_gain=LAMBDA_GAIN):
# #     """
# #     Calculate residuals between measured and predicted voltages, PLUS
# #     regularization residuals that pull (x,y,z,gain) toward their
# #     nominal/physical prior values. Offset 'a' is left unregularized.

# #     params: [x, y, z, a, g]
# #     Sensor direction is FIXED to straight up [0, 0, 1] -- theta/phi are no
# #     longer free parameters (sensors are assumed soldered perpendicular to
# #     the PCB with negligible tilt).

# #     pos_prior: (x0, y0, z0) nominal sensor position from
# #                Hall_sensor_positions.csv (design position). If None, no
# #                position regularization terms are appended (kept for safety;
# #                in normal use this is always provided by the caller).
# #     """
# #     x, y, z, a, g = params

# #     # Fixed sensor direction (straight up, perpendicular to PCB)
# #     sensor_dir = np.array([0.0, 0.0, 1.0])

# #     # Sensor position
# #     sensor_pos = np.array([x, y, z])

# #     # Vector from robot to sensor
# #     r_vec = sensor_pos - robot_positions

# #     # Calculate magnetic field at sensor position
# #     B = dipole_field(r_vec, m_world)

# #     # Project magnetic field onto sensor direction
# #     B_proj = B @ sensor_dir

# #     # Predicted voltage
# #     voltage_pred = a + g * B_proj

# #     # ---- Voltage residual (unchanged) ----
# #     r_voltage = voltage_sensor - voltage_pred

# #     # ---- Regularization residuals (physical priors) ----
# #     # NOTE: concatenated onto the residual vector, NOT added to the scalar
# #     # cost -- appending sqrt(lambda) * (param - prior) as extra residual
# #     # entries is exactly equivalent to adding lambda * (param - prior)^2
# #     # to the cost, keeping us fully inside least_squares()/'trf'.
# #     if pos_prior is not None:
# #         x0, y0, z0 = pos_prior
# #         r_pos = np.sqrt(lambda_pos) * np.array([x - x0, y - y0, z - z0])
# #     else:
# #         r_pos = np.array([])

# #     r_gain = np.sqrt(lambda_gain) * np.array([g - g0])

# #     residual = np.concatenate([r_voltage, r_pos, r_gain])

# #     return residual


# # # =============================================================================
# # # SINGLE SENSOR CALIBRATION
# # # =============================================================================

# # # =============================================================================
# # # IDENTIFIABILITY / CORRELATION ANALYSIS (Stage 1 post-fit diagnostic)
# # # =============================================================================
# # # Muc dich: kiem tra xem gain co bi "nhap nhang" (correlated) voi cac tham so
# # # khac (dac biet la z, vi dipole suy giam theo 1/r^3 nen z va gain co the
# # # danh doi lan nhau) hay khong. Neu |corr| gan 1, gain KHONG duoc xac dinh
# # # duy nhat tu du lieu voltage -- tuc gain fit duoc co the chi la 1 nghiem
# # # trong vo so nghiem tuong duong, khong phai gia tri vat ly that cua sensor.
# # #
# # # Dung result.jac cua chinh least_squares() da tra ve -- KHONG chay lai toi
# # # uu hoa, khong doi thuat toan/method='trf'.
# # PARAM_NAMES = ["x", "y", "z", "offset", "gain"]
# # CORRELATION_LOG = []  # tich luy 1 dict/sensor, dung de in bang tong ket cuoi


# # def analyze_gain_identifiability(result, n_voltage, sensor_index, verbose=True):
# #     """
# #     Tinh ma tran covariance/correlation giua 5 tham so (x,y,z,offset,gain)
# #     tu Jacobian tai nghiem hoi tu, CHI dung phan voltage residual (bo phan
# #     regularization residual neu co, vi regularization lam gia tao giam
# #     correlation -- xem giai thich o tren).

# #     Tra ve dict {sensor_index, corr_x_gain, corr_y_gain, corr_z_gain,
# #     corr_offset_gain, n_voltage, n_params, is_rank_deficient}.
# #     """
# #     n_params = len(result.x)

# #     # Chi lay J ung voi voltage residual (n_voltage dong dau), bo residual
# #     # regularization (pos/gain prior) neu dang bat regularize.
# #     J_voltage = result.jac[:n_voltage, :]
# #     residual_voltage = result.fun[:n_voltage]

# #     dof = n_voltage - n_params
# #     if dof <= 0:
# #         row = {
# #             "sensor_index": sensor_index,
# #             "corr_x_gain": np.nan, "corr_y_gain": np.nan,
# #             "corr_z_gain": np.nan, "corr_offset_gain": np.nan,
# #             "n_voltage": n_voltage, "n_params": n_params,
# #             "is_rank_deficient": True,
# #         }
# #         CORRELATION_LOG.append(row)
# #         return row

# #     sigma2 = np.sum(residual_voltage ** 2) / dof

# #     JTJ = J_voltage.T @ J_voltage
# #     try:
# #         JTJ_inv = np.linalg.inv(JTJ)
# #         is_rank_deficient = False
# #     except np.linalg.LinAlgError:
# #         # JTJ suy bien (singular) -- day la dau hieu MANH NHAT cua non-
# #         # identifiability: mot so to hop tham so hoan toan khong the tach
# #         # biet tu du lieu nay. Dung pseudo-inverse de van tinh duoc con so
# #         # tham khao, nhung phai canh bao ro.
# #         JTJ_inv = np.linalg.pinv(JTJ)
# #         is_rank_deficient = True

# #     cov = sigma2 * JTJ_inv
# #     diag = np.diag(cov)

# #     diag_safe = np.where(diag > 0, diag, np.nan)
# #     denom = np.sqrt(np.outer(diag_safe, diag_safe))
# #     corr_matrix = cov / denom

# #     gain_idx = PARAM_NAMES.index("gain")
# #     row = {
# #         "sensor_index": sensor_index,
# #         "corr_x_gain": corr_matrix[PARAM_NAMES.index("x"), gain_idx],
# #         "corr_y_gain": corr_matrix[PARAM_NAMES.index("y"), gain_idx],
# #         "corr_z_gain": corr_matrix[PARAM_NAMES.index("z"), gain_idx],
# #         "corr_offset_gain": corr_matrix[PARAM_NAMES.index("offset"), gain_idx],
# #         "n_voltage": n_voltage, "n_params": n_params,
# #         "is_rank_deficient": is_rank_deficient,
# #     }
# #     CORRELATION_LOG.append(row)

# #     if verbose:
# #         flag = " [RANK-DEFICIENT!]" if is_rank_deficient else ""
# #         print(f"    corr(z,gain)={row['corr_z_gain']:+.3f}  "
# #               f"corr(offset,gain)={row['corr_offset_gain']:+.3f}  "
# #               f"corr(x,gain)={row['corr_x_gain']:+.3f}  "
# #               f"corr(y,gain)={row['corr_y_gain']:+.3f}{flag}")

# #     return row


# # def print_correlation_summary():
# #     """In bang tong ket |corr(*, gain)| trung binh/max qua 64 sensor, va
# #     liet ke cac sensor co correlation cao (|corr| > 0.8) -- nghi van manh
# #     ve identifiability."""
# #     if not CORRELATION_LOG:
# #         print("Chua co du lieu correlation nao duoc ghi lai.")
# #         return

# #     z_gain = np.array([r["corr_z_gain"] for r in CORRELATION_LOG])
# #     offset_gain = np.array([r["corr_offset_gain"] for r in CORRELATION_LOG])
# #     x_gain = np.array([r["corr_x_gain"] for r in CORRELATION_LOG])
# #     y_gain = np.array([r["corr_y_gain"] for r in CORRELATION_LOG])
# #     n_deficient = sum(r["is_rank_deficient"] for r in CORRELATION_LOG)

# #     print("\n" + "=" * 60)
# #     print("IDENTIFIABILITY SUMMARY (correlation with gain, 64 sensors)")
# #     print("=" * 60)
# #     print(f"{'pair':<18}{'mean |corr|':>14}{'max |corr|':>14}")
# #     print(f"{'z    - gain':<18}{np.nanmean(np.abs(z_gain)):>14.3f}"
# #           f"{np.nanmax(np.abs(z_gain)):>14.3f}")
# #     print(f"{'offset - gain':<18}{np.nanmean(np.abs(offset_gain)):>14.3f}"
# #           f"{np.nanmax(np.abs(offset_gain)):>14.3f}")
# #     print(f"{'x    - gain':<18}{np.nanmean(np.abs(x_gain)):>14.3f}"
# #           f"{np.nanmax(np.abs(x_gain)):>14.3f}")
# #     print(f"{'y    - gain':<18}{np.nanmean(np.abs(y_gain)):>14.3f}"
# #           f"{np.nanmax(np.abs(y_gain)):>14.3f}")
# #     print(f"\nSo sensor bi rank-deficient (JTJ singular): {n_deficient}/64")

# #     HIGH_CORR_THRESHOLD = 0.8
# #     print(f"\nSensor co |corr(z,gain)| > {HIGH_CORR_THRESHOLD}:")
# #     flagged = [r for r in CORRELATION_LOG
# #                if not np.isnan(r["corr_z_gain"])
# #                and abs(r["corr_z_gain"]) > HIGH_CORR_THRESHOLD]
# #     if flagged:
# #         for r in flagged:
# #             print(f"  Sensor {r['sensor_index']+1:02d}: "
# #                   f"corr(z,gain) = {r['corr_z_gain']:+.3f}")
# #     else:
# #         print("  (khong co)")

# #     print(f"\nSensor co |corr(offset,gain)| > {HIGH_CORR_THRESHOLD}:")
# #     flagged = [r for r in CORRELATION_LOG
# #                if not np.isnan(r["corr_offset_gain"])
# #                and abs(r["corr_offset_gain"]) > HIGH_CORR_THRESHOLD]
# #     if flagged:
# #         for r in flagged:
# #             print(f"  Sensor {r['sensor_index']+1:02d}: "
# #                   f"corr(offset,gain) = {r['corr_offset_gain']:+.3f}")
# #     else:
# #         print("  (khong co)")
# #     print("=" * 60)


# # def calibrate_single_sensor(
# #         sensor_index,
# #         sensor_pos_init,
# #         robot_positions,
# #         m_world,
# #         voltage_sensor,
# #         offset_init=1.618):
# #     """
# #     Calibrate a single sensor -- position + offset + gain only.
# #     Sensor direction is fixed straight up (perpendicular to PCB);
# #     theta/phi are no longer free parameters.
# #     """
# #     # ----------------------------------------------------
# #     # INITIAL VALUES
# #     # ----------------------------------------------------
# #     g0 = 7.5  # Initial gain (V/T)
# #     offset_init = 1.618         # a (offset)

# #     # ----------------------------------------------------
# #     # INITIAL PARAMETERS
# #     # ----------------------------------------------------
# #     x0 = np.array([
# #         sensor_pos_init[0],  # x
# #         sensor_pos_init[1],  # y
# #         sensor_pos_init[2],  # z
# #         offset_init,         # a (offset)
# #         g0                   # g (gain)
# #     ])
    
# # # ----------------------------------------------------
# #     # BOUNDS
# #     # ----------------------------------------------------
# #     pos_tol = 0.0012  # 1.5mm dung sai lap dat thuc te cua sensor tren PCB
# #     lower = [
# #         sensor_pos_init[0] - pos_tol,    # x min
# #         sensor_pos_init[1] - pos_tol,    # y min
# #         sensor_pos_init[2] - pos_tol,    # z min
# #         offset_init - 0.02,              # a min (20mV)
# #         -9                                # g min
# #     ]
    
# #     upper = [
# #         sensor_pos_init[0] + pos_tol,    # x max
# #         sensor_pos_init[1] + pos_tol,    # y max
# #         sensor_pos_init[2] + pos_tol,    # z max
# #         offset_init + 0.02,              # a max (20mV)
# #         9                                 # g max
# #     ]
    
# #     # ----------------------------------------------------
# #     # OPTIMIZATION
# #     # ----------------------------------------------------
# #     result = least_squares(
# #         sensor_residuals,
# #         x0,
# #         bounds=(lower, upper),
# #         args=(
# #             robot_positions,
# #             m_world,
# #             voltage_sensor
# #         ),
# #         kwargs=dict(
# #             pos_prior=(sensor_pos_init[0], sensor_pos_init[1], sensor_pos_init[2]),
# #             g0=g0,
# #         ),
# #         method='trf',
# #         max_nfev=250           
# #     )
    
# #     # ----------------------------------------------------
# #     # EXTRACT OPTIMIZED PARAMETERS
# #     # ----------------------------------------------------
# #     params_opt = result.x
    
# #     # Direction is fixed straight up -- theta/phi kept at 0 rad so the
# #     # OUTPUT FORMAT (10 columns) stays identical to the previous script
# #     # for downstream compatibility (save_results, save_physical_results,
# #     # plot_direction_vectors, Stage 2 alpha).
# #     theta_opt, phi_opt = 0.0, 0.0
# #     nx, ny, nz = 0.0, 0.0, 1.0
    
# #     # Create extended parameter array for saving
# #     params_extended = np.array([
# #         params_opt[0],  # x
# #         params_opt[1],  # y
# #         params_opt[2],  # z
# #         params_opt[3],  # a
# #         params_opt[4],  # g
# #         nx, ny, nz,     # direction vector components (fixed straight up)
# #         theta_opt,      # theta (radians, fixed = 0)
# #         phi_opt         # phi (radians, fixed = 0)
# #     ])
    
# #     # ----------------------------------------------------
# #     # CALCULATE RMSE
# #     # (result.fun = [voltage residuals, pos reg, gain reg];
# #     #  RMSE must be computed from the voltage part only, so it still
# #     #  means "V measured vs V predicted" and stays comparable to before)
# #     # ----------------------------------------------------
# #     n_voltage = voltage_sensor.shape[0]
# #     rmse = np.sqrt(np.mean(result.fun[:n_voltage]**2))

# #     # ---- IDENTIFIABILITY CHECK: gain co bi nhap nhang voi z/offset/x/y? ----
# #     # Dung result.jac da co san tu least_squares() o tren, khong chay lai
# #     # toi uu hoa, khong doi thuat toan.
# #     analyze_gain_identifiability(result, n_voltage, sensor_index)

# #     angle_from_vertical = np.rad2deg(abs(theta_opt))
# #     print(f"Sensor {sensor_index+1:02d} | RMSE = {rmse:.6f} | "
# #           f"Angle from Z = {angle_from_vertical:.2f}° | "
# #           f"Dir = [{nx:.3f}, {ny:.3f}, {nz:.3f}]")
    
# #     return params_extended, rmse


# # # =============================================================================
# # # FULL CALIBRATION
# # # =============================================================================

# # def run_calibration(
# #         sensor_positions,
# #         offsets,
# #         robot_positions,
# #         m_world,
# #         voltage_data):
# #     """
# #     Calibrate all sensors
# #     """
# #     n_sensors = sensor_positions.shape[0]
# #     results = []
# #     rmses = []
    
# #     for i in range(n_sensors):
# #         params, rmse = calibrate_single_sensor(
# #             sensor_index=i,
# #             sensor_pos_init=sensor_positions[i],
# #             robot_positions=robot_positions,
# #             m_world=m_world,
# #             voltage_sensor=voltage_data[:, i]
# #         )
        
# #         results.append(params)
# #         rmses.append(rmse)
    
# #     # ---- In bang tong ket identifiability sau khi calib het 64 sensor ----
# #     print_correlation_summary()
    
# #     return np.array(results), np.array(rmses)


# # # =============================================================================
# # # REGION SELECTION
# # # =============================================================================

# # def select_region_samples(
# #         robot_positions,
# #         m_world,
# #         voltage_data,
# #         sensor_z,
# #         h_min,
# #         h_max=None,
# #         max_samples=210,
# #         return_indices=False):

# #     h = robot_positions[:, 2] - sensor_z

# #     lower_ok = np.ones_like(h, dtype=bool) if h_min is None else (h >= h_min)
# #     upper_ok = np.ones_like(h, dtype=bool) if h_max is None else (h < h_max)
# #     region_idx = np.where(lower_ok & upper_ok)[0]

# #     print(f"\nRegion [{h_min}, {h_max}] contains {len(region_idx)} samples")

# #     if len(region_idx) > max_samples:
# #         # NOTE: no random seed is set here, per the calibration spec.
# #         region_idx = np.random.choice(
# #             region_idx,
# #             size=max_samples,
# #             replace=False
# #         )

# #     print(f"Using {len(region_idx)} samples")

# #     if return_indices:
# #         return (
# #             robot_positions[region_idx],
# #             m_world[region_idx],
# #             voltage_data[region_idx],
# #             region_idx
# #         )

# #     return (
# #         robot_positions[region_idx],
# #         m_world[region_idx],
# #         voltage_data[region_idx]
# #     )


# # # =============================================================================
# # # NEW: STAGE 1 - STRATIFIED CALIBRATION SET (fixed, reused in Stage 2)
# # # =============================================================================

# # def select_stage1_calibration_set(
# #         robot_positions,
# #         m_world,
# #         voltage_data,
# #         sensor_z_ref,
# #         per_region=SAMPLES_PER_REGION):

# #     rp1, mw1, vd1, idx1 = select_region_samples(
# #         robot_positions, m_world, voltage_data, sensor_z_ref,
# #         h_min=None, h_max=REGION1_H_MAX,
# #         max_samples=per_region, return_indices=True
# #     )
# #     rp2, mw2, vd2, idx2 = select_region_samples(
# #         robot_positions, m_world, voltage_data, sensor_z_ref,
# #         h_min=REGION1_H_MAX, h_max=REGION2_H_MAX,
# #         max_samples=per_region, return_indices=True
# #     )
# #     rp3, mw3, vd3, idx3 = select_region_samples(
# #         robot_positions, m_world, voltage_data, sensor_z_ref,
# #         h_min=REGION2_H_MAX, h_max=None,
# #         max_samples=per_region, return_indices=True
# #     )

# #     calib_indices = np.concatenate([idx1, idx2, idx3])
# #     rp_calib = np.concatenate([rp1, rp2, rp3], axis=0)
# #     mw_calib = np.concatenate([mw1, mw2, mw3], axis=0)
# #     vd_calib = np.concatenate([vd1, vd2, vd3], axis=0)

# #     print(f"\n[Stage 1] Combined calibration set: {len(calib_indices)} "
# #           f"points (Region1={len(idx1)}, Region2={len(idx2)}, "
# #           f"Region3={len(idx3)})")

# #     return calib_indices, rp_calib, mw_calib, vd_calib


# # # =============================================================================
# # # NEW: STAGE 2 - CLOSED-FORM ALPHA PER REGION
# # # =============================================================================

# # def calibrate_alpha_by_region(physical_results, rp_calib, mw_calib, vd_calib):

# #     n_samples = rp_calib.shape[0]
# #     n_sensors = physical_results.shape[0]

# #     sensor_pos = physical_results[:, 0:3]      # (n_sensors, 3): x, y, z
# #     a = physical_results[:, 3]                  # (n_sensors,)  offset
# #     g = physical_results[:, 4]                  # (n_sensors,)  gain
# #     sensor_dir = physical_results[:, 5:8]       # (n_sensors, 3): nx, ny, nz

# #     # h[i, s] = z_capsule_i - z_sensor_calibrated_s  (uses Stage-1 result)
# #     z_calibrated = sensor_pos[:, 2]
# #     h = rp_calib[:, 2][:, None] - z_calibrated[None, :]   # (n_samples, n_sensors)

# #     # Raw dipole projection B_proj[i, s] using the FROZEN calibrated pose
# #     # (position + direction) of each sensor -- gain/offset are NOT
# #     # reapplied here, they're multiplied in separately below.
# #     B_proj = np.zeros((n_samples, n_sensors))
# #     for s in range(n_sensors):
# #         r_vec = sensor_pos[s] - rp_calib               # (n_samples, 3)
# #         B = dipole_field(r_vec, mw_calib)               # (n_samples, 3)
# #         B_proj[:, s] = B @ sensor_dir[s]

# #     gB = g[None, :] * B_proj                            # (n_samples, n_sensors)
# #     v_minus_a = vd_calib - a[None, :]                    # V_measured - a

# #     region_masks = {
# #         1: h < REGION1_H_MAX,
# #         2: (h >= REGION1_H_MAX) & (h < REGION2_H_MAX),
# #         3: h >= REGION2_H_MAX
# #     }

# #     alphas = {}
# #     for region, mask in region_masks.items():
# #         numerator = np.sum(gB[mask] * v_minus_a[mask])
# #         denominator = np.sum(gB[mask] ** 2)
# #         alpha = numerator / denominator if denominator > 0 else np.nan
# #         print(f"Region {region}: alpha = {alpha:.6f} "
# #               f"(from {np.sum(mask)} sample-sensor pairs)")
# #         alphas[region] = alpha

# #     return alphas


# # # =============================================================================
# # # SAVE RESULTS
# # # =============================================================================

# # def save_results(results, rmses, output_file):
# #     """
# #     Save calibration results to CSV
# #     """
# #     df = pd.DataFrame({
# #         "sensor_id": np.arange(len(results)),
# #         "x": results[:, 0],
# #         "y": results[:, 1],
# #         "z": results[:, 2],
# #         "offset_a": results[:, 3],
# #         "gain_g": results[:, 4],
# #         "nx": results[:, 5],
# #         "ny": results[:, 6],
# #         "nz": results[:, 7],
# #         "theta_rad": results[:, 8],
# #         "phi_rad": results[:, 9],
# #         "angle_from_z_deg": np.rad2deg(np.abs(results[:, 8])),
# #         "rmse": rmses
# #     })
    
# #     df.to_csv(output_file, index=False)
# #     print(f"\nSaved: {output_file}")


# # # =============================================================================
# # # NEW: SAVE STAGE 1 / STAGE 2 RESULTS (required output files)
# # # =============================================================================

# # def save_physical_results(results, output_file):
# #     """
# #     Save Stage 1 frozen physical parameters to Calibration_Physical.csv
# #     with columns: sensor_index, x, y, z, offset, gain, theta, phi
# #     """
# #     df = pd.DataFrame({
# #         "sensor_index": np.arange(len(results)),
# #         "x": results[:, 0],
# #         "y": results[:, 1],
# #         "z": results[:, 2],
# #         "offset": results[:, 3],
# #         "gain": results[:, 4],
# #         "theta": results[:, 8],
# #         "phi": results[:, 9],
# #     })
# #     df.to_csv(output_file, index=False)
# #     print(f"\nSaved: {output_file}")


# # def save_alpha_results(alphas, output_file):
# #     """
# #     Save Stage 2 region correction coefficients to Calibration_Alpha.csv
# #     with exactly 3 rows: Region 1, Region 2, Region 3.
# #     """
# #     regions_sorted = sorted(alphas.keys())
# #     df = pd.DataFrame({
# #         "Region": [f"Region {r}" for r in regions_sorted],
# #         "Alpha": [alphas[r] for r in regions_sorted],
# #     })
# #     df.to_csv(output_file, index=False)
# #     print(f"Saved: {output_file}")


# # # =============================================================================
# # # PLOT RMSE
# # # =============================================================================

# # def plot_rmse(rmses):
# #     """Plot RMSE for all sensors"""
# #     plt.figure(figsize=(10, 5))
# #     plt.bar(np.arange(len(rmses)), rmses)
# #     plt.xlabel("Sensor Index")
# #     plt.ylabel("RMSE")
# #     plt.title("Calibration RMSE")
# #     plt.grid(True)
# #     plt.show()


# # # =============================================================================
# # # PLOT DIRECTION VECTORS
# # # =============================================================================

# # # def plot_direction_vectors(results):
# # #     """Plot direction vectors of all sensors"""
# # #     fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
# # #     sensor_ids = np.arange(len(results))
    
# # #     # Subplot 1: Direction components
# # #     ax1 = axes[0]
# # #     ax1.plot(sensor_ids, results[:, 5], 'r.-', label='nx', markersize=8)
# # #     ax1.plot(sensor_ids, results[:, 6], 'g.-', label='ny', markersize=8)
# # #     ax1.plot(sensor_ids, results[:, 7], 'b.-', label='nz', markersize=8)
# # #     ax1.set_xlabel("Sensor Index")
# # #     ax1.set_ylabel("Direction Component")
# # #     ax1.set_title("Sensor Direction Components")
# # #     ax1.legend()
# # #     ax1.grid(True)
    
# # #     # Subplot 2: Angular deviation from vertical
# # #     ax2 = axes[1]
# # #     angle_from_vertical = np.rad2deg(np.abs(results[:, 8]))
# # #     ax2.bar(sensor_ids, angle_from_vertical)
# # #     ax2.set_xlabel("Sensor Index")
# # #     ax2.set_ylabel("Angle (degrees)")
# # #     ax2.set_title("Angular Deviation from Z-axis")
# # #     ax2.grid(True)
    
# # #     # Subplot 3: Azimuthal angle
# # #     ax3 = axes[2]
# # #     phi_deg = np.rad2deg(results[:, 9])
# # #     ax3.bar(sensor_ids, phi_deg)
# # #     ax3.set_xlabel("Sensor Index")
# # #     ax3.set_ylabel("Phi (degrees)")
# # #     ax3.set_title("Azimuthal Angle (XY-plane)")
# # #     ax3.grid(True)
    
# # #     plt.tight_layout()
# # #     plt.show()


# # # =============================================================================
# # # MAIN
# # # =============================================================================

# # def main():
# #     """Main execution function -- 2-stage calibration framework"""

# #     # Load data
# #     sensor_positions = load_sensor_positions(SENSOR_POSITIONS_PATH)
# #     robot_positions, m_world = load_robot_pose(ROBOT_POSE_PATH)
# #     voltage_data = load_voltage_data(VOLTAGE_DATA_PATH)
# #     offsets = load_sensor_offsets(OFFSET_FILE_PATH)

# #     # Ensure consistent number of samples
# #     n_samples = min(len(robot_positions), len(voltage_data))
# #     robot_positions = robot_positions[:n_samples]
# #     m_world = m_world[:n_samples]
# #     voltage_data = voltage_data[:n_samples]

# #     # ------------------------------------------------------------------
# #     # NEW: Sensor reference height for Stage 1 region assignment.
# #     # Must come from the ORIGINAL (uncalibrated) sensor positions file.
# #     # ------------------------------------------------------------------
# #     sensor_z_initial_ref = sensor_positions[:, 2].mean()
# #     print(f"\nSensor reference z (initial, for Stage 1) = "
# #           f"{sensor_z_initial_ref:.6f} m")

# #     # ==================================================================
# #     # STAGE 1a: Stratified, fixed calibration set (<=150 points)
# #     # ==================================================================
# #     print("\n===================================")
# #     print("STAGE 1: SAMPLE SELECTION")
# #     print("===================================")
# #     calib_indices, rp_calib, mw_calib, vd_calib = select_stage1_calibration_set(
# #         robot_positions, m_world, voltage_data, sensor_z_initial_ref,
# #         per_region=SAMPLES_PER_REGION
# #     )
# #     # calib_indices is frozen here and reused as-is in Stage 2 below --
# #     # no re-sampling happens after this point.

# #     # ==================================================================
# #     # STAGE 1b: Per-sensor physical parameter calibration
# #     # (unchanged residual function, dipole model, initial values, bounds)
# #     # ==================================================================
# #     print("\n===================================")
# #     print("STAGE 1: PHYSICAL PARAMETER FIT")
# #     print("===================================")
# #     results, rmses = run_calibration(
# #         sensor_positions, offsets, rp_calib, mw_calib, vd_calib
# #     )

# #     print("\n========================")
# #     print(f"Mean RMSE = {np.mean(rmses):.6f}")
# #     print(f"Max RMSE  = {np.max(rmses):.6f}")
# #     print(f"Min RMSE  = {np.min(rmses):.6f}")
# #     print("========================")

# #     # Save Stage 1 physical parameters (frozen going into Stage 2)
# #     save_physical_results(results, PHYSICAL_OUTPUT_PATH)

# #     # Plots
# #     plot_rmse(rmses)
# #     # plot_direction_vectors(results)

# #     # ==================================================================
# #     # STAGE 2: Closed-form alpha per region (physical params frozen)
# #     # ==================================================================
# #     print("\n===================================")
# #     print("STAGE 2: ALPHA CORRECTION (closed-form)")
# #     print("===================================")
# #     alphas = calibrate_alpha_by_region(results, rp_calib, mw_calib, vd_calib)
# #     save_alpha_results(alphas, ALPHA_OUTPUT_PATH)

# #     print("\n===================================")
# #     print("ALL STAGES FINISHED")
# #     print("===================================")


# # if __name__ == "__main__":
# #     main()
    
# # # # """
# # # # Sweep lambda_pos va lambda_gain quanh vung uoc luong co co so, ve L-curve
# # # # (RMSE vs do lech tham so khoi prior) de chon lambda can bang.

# # # # CACH DUNG: dan doan code nay vao CUOI file calib_region_based.py (ban da
# # # # co san cac ham dipole_field, sensor_residuals, calibrate_single_sensor,
# # # # select_stage1_calibration_set, load_sensor_positions, load_robot_pose,
# # # # load_voltage_data, v.v. trong file goc). Script nay KHONG doi thuat toan
# # # # least_squares/'trf', chi goi lai calibrate_single_sensor() nhieu lan voi
# # # # cac gia tri lambda khac nhau.

# # # # Diem khoi dau duoc uoc luong tu cong thuc:
# # # #     lambda = (RMSE_v_no_reg / sigma)^2
# # # # voi RMSE_v_no_reg lay tu log ban da chay (0.000383 V), va sigma la do lech
# # # # "chap nhan duoc" ban tu chon (o day: sigma_pos=0.5mm, sigma_gain=1.0 V/T).
# # # # Day chi la DIEM KHOI DAU -- sweep xung quanh no de tim diem can bang that su.
# # # # """

# # # # import numpy as np
# # # # import matplotlib.pyplot as plt

# # # # # =============================================================================
# # # # # CAU HINH SWEEP
# # # # # =============================================================================
# # # # RMSE_V_NO_REG = 0.000383  # tu log khong-regularize cua ban

# # # # # Diem khoi dau uoc luong (KHONG phai gia tri cuoi cung)
# # # # LAMBDA_POS_GUESS = (RMSE_V_NO_REG / 0.0005) ** 2   # sigma_pos = 0.5 mm
# # # # LAMBDA_GAIN_GUESS = (RMSE_V_NO_REG / 1.0) ** 2      # sigma_gain = 1.0 V/T

# # # # # Sweep tu 1/100x den 100x quanh diem khoi dau, log-spaced, 9 diem
# # # # N_POINTS = 9
# # # # lambda_pos_grid = np.geomspace(LAMBDA_POS_GUESS / 100, LAMBDA_POS_GUESS * 100, N_POINTS)
# # # # lambda_gain_grid = np.geomspace(LAMBDA_GAIN_GUESS / 100, LAMBDA_GAIN_GUESS * 100, N_POINTS)

# # # # # Sensor dai dien de sweep nhanh (thay vi chay het 64 sensor cho moi lambda).
# # # # # Nen chon vai sensor o cac vi tri khac nhau (giua/canh mang) de dai dien.
# # # # SENSOR_SAMPLE_IDX = [0, 15, 31, 47, 63]  # 5 sensor rai deu tren 64 sensor


# # # # def run_sweep_1d(param_name, lambda_grid, sensor_positions, robot_positions,
# # # #                   m_world, voltage_data, sensor_z_ref):
# # # #     """
# # # #     Sweep 1 chieu: giu lambda con lai = 0, chi thay doi lambda cua param_name
# # # #     ('pos' hoac 'gain'). Tra ve (rmse_list, deviation_list).

# # # #     Goi least_squares() TRUC TIEP (khong qua calibrate_single_sensor) vi
# # # #     lambda_pos/lambda_gain trong sensor_residuals la default-argument, bi
# # # #     "dong bang" luc dinh nghia ham -- doi bien global sau do KHONG co tac
# # # #     dung. Truyen tuong minh qua kwargs moi dam bao dung gia tri lambda.
# # # #     """
# # # #     rmse_list = []
# # # #     deviation_list = []  # do lech tuong doi TB khoi prior (chuan hoa theo bound)

# # # #     calib_indices, rp_calib, mw_calib, vd_calib = select_stage1_calibration_set(
# # # #         robot_positions, m_world, voltage_data, sensor_z_ref
# # # #     )

# # # #     g0 = 7.5
# # # #     pos_tol = 0.0015

# # # #     for lam in lambda_grid:
# # # #         lambda_pos = lam if param_name == "pos" else 0.0
# # # #         lambda_gain = lam if param_name == "gain" else 0.0

# # # #         rmses_this_lambda = []
# # # #         deviations_this_lambda = []

# # # #         for s in SENSOR_SAMPLE_IDX:
# # # #             sensor_pos_init = sensor_positions[s]

# # # #             x0 = np.array([
# # # #                 sensor_pos_init[0], sensor_pos_init[1], sensor_pos_init[2],
# # # #                 1.618,  # offset_init
# # # #                 g0
# # # #             ])
# # # #             lower = [
# # # #                 sensor_pos_init[0] - pos_tol, sensor_pos_init[1] - pos_tol,
# # # #                 sensor_pos_init[2] - pos_tol, 1.618 - 0.015, 4
# # # #             ]
# # # #             upper = [
# # # #                 sensor_pos_init[0] + pos_tol, sensor_pos_init[1] + pos_tol,
# # # #                 sensor_pos_init[2] + pos_tol, 1.618 + 0.015, 9
# # # #             ]

# # # #             result = least_squares(
# # # #                 sensor_residuals,
# # # #                 x0,
# # # #                 bounds=(lower, upper),
# # # #                 args=(rp_calib, mw_calib, vd_calib[:, s]),
# # # #                 kwargs=dict(
# # # #                     pos_prior=tuple(sensor_pos_init),
# # # #                     g0=g0,
# # # #                     lambda_pos=lambda_pos,
# # # #                     lambda_gain=lambda_gain,
# # # #                 ),
# # # #                 method="trf",
# # # #                 max_nfev=250
# # # #             )

# # # #             n_voltage = vd_calib.shape[0]
# # # #             rmse = np.sqrt(np.mean(result.fun[:n_voltage] ** 2))
# # # #             rmses_this_lambda.append(rmse)

# # # #             if param_name == "pos":
# # # #                 dev = np.linalg.norm(result.x[0:3] - sensor_pos_init) / pos_tol
# # # #             else:  # gain
# # # #                 dev = abs(result.x[4] - g0) / 5.0  # chuan hoa theo bien do bound gain [4,9]

# # # #             deviations_this_lambda.append(dev)

# # # #         rmse_list.append(np.mean(rmses_this_lambda))
# # # #         deviation_list.append(np.mean(deviations_this_lambda))

# # # #         print(f"  lambda_{param_name} = {lam:.3e} | "
# # # #               f"mean RMSE = {np.mean(rmses_this_lambda):.6f} | "
# # # #               f"mean |dev|/bound = {np.mean(deviations_this_lambda):.4f}")

# # # #     return rmse_list, deviation_list


# # # # def plot_lcurve(lambda_grid, rmse_list, deviation_list, param_name, guess_value):
# # # #     fig, ax1 = plt.subplots(figsize=(7, 5))

# # # #     color1 = "tab:blue"
# # # #     ax1.set_xlabel(f"lambda_{param_name}")
# # # #     ax1.set_ylabel("Mean RMSE (V)", color=color1)
# # # #     ax1.plot(lambda_grid, rmse_list, "o-", color=color1, label="RMSE")
# # # #     ax1.set_xscale("log")
# # # #     ax1.tick_params(axis="y", labelcolor=color1)

# # # #     ax2 = ax1.twinx()
# # # #     color2 = "tab:red"
# # # #     ax2.set_ylabel("Mean |deviation| / bound", color=color2)
# # # #     ax2.plot(lambda_grid, deviation_list, "s--", color=color2, label="Deviation")
# # # #     ax2.tick_params(axis="y", labelcolor=color2)

# # # #     ax1.axvline(guess_value, color="gray", linestyle=":", alpha=0.7,
# # # #                 label=f"initial guess = {guess_value:.2e}")

# # # #     fig.suptitle(f"L-curve: RMSE vs Deviation ({param_name})")
# # # #     fig.tight_layout()
# # # #     output_path = BASE_DIR / f"lcurve_{param_name}.png"
# # # #     fig.savefig(output_path, dpi=120)
# # # #     plt.close(fig)
# # # #     print(f"  -> Da luu: {output_path}")


# # # # def main_sweep():
# # # #     sensor_positions = load_sensor_positions(SENSOR_POSITIONS_PATH)
# # # #     robot_positions, m_world = load_robot_pose(ROBOT_POSE_PATH)
# # # #     voltage_data = load_voltage_data(VOLTAGE_DATA_PATH)

# # # #     # QUAN TRONG: sensor_z_ref phai la 1 gia tri SCALAR (trung binh z cua
# # # #     # tat ca sensor), giong het cach goc dung trong run_calibration():
# # # #     #   sensor_z_initial_ref = sensor_positions[:, 2].mean()
# # # #     # KHONG duoc truyen mang 64 phan tu vao day.
# # # #     sensor_z_ref = sensor_positions[:, 2].mean()

# # # #     print("=" * 60)
# # # #     print(f"Diem khoi dau uoc luong: lambda_pos = {LAMBDA_POS_GUESS:.3e}, "
# # # #           f"lambda_gain = {LAMBDA_GAIN_GUESS:.3e}")
# # # #     print("=" * 60)

# # # #     print("\n--- SWEEP lambda_pos (lambda_gain = 0) ---")
# # # #     rmse_pos, dev_pos = run_sweep_1d(
# # # #         "pos", lambda_pos_grid, sensor_positions, robot_positions,
# # # #         m_world, voltage_data, sensor_z_ref
# # # #     )
# # # #     plot_lcurve(lambda_pos_grid, rmse_pos, dev_pos, "pos", LAMBDA_POS_GUESS)

# # # #     print("\n--- SWEEP lambda_gain (lambda_pos = 0) ---")
# # # #     rmse_gain, dev_gain = run_sweep_1d(
# # # #         "gain", lambda_gain_grid, sensor_positions, robot_positions,
# # # #         m_world, voltage_data, sensor_z_ref
# # # #     )
# # # #     plot_lcurve(lambda_gain_grid, rmse_gain, dev_gain, "gain", LAMBDA_GAIN_GUESS)

# # # #     print("\nDa luu 2 anh L-curve vao thu muc BASE_DIR (xem duong dan o tren)")
# # # #     print("Chon lambda tai 'diem khuyu' (elbow): noi RMSE bat dau tang nhanh")
# # # #     print("nhung deviation da giam ve gan 0 -- do la vung can bang tot nhat.")


# # # # if __name__ == "__main__":
# # # #     main_sweep()

# # import numpy as np
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # from pathlib import Path
# # from scipy.optimize import least_squares

# # # =============================================================================
# # # FILE PATHS
# # # =============================================================================
# # BASE_DIR = Path(r"/Users/tuananhnguyen/Downloads/Hallsensor_final/Data set 18.6")

# # SENSOR_POSITIONS_PATH = BASE_DIR / "Hall_sensor_positions.csv"

# # ROBOT_POSE_PATH = BASE_DIR / "grid_points_coordinates.csv"

# # VOLTAGE_DATA_PATH = BASE_DIR / "grid_data.csv"

# # OFFSET_FILE_PATH = BASE_DIR / "Offset_Sens.csv"

# # # ---- outputs for the 2-stage calibration framework ----
# # PHYSICAL_OUTPUT_PATH = BASE_DIR / "Calibration_Physical.csv"
# # ALPHA_OUTPUT_PATH = BASE_DIR / "Calibration_Alpha.csv"


# # # =============================================================================
# # # CONSTANTS
# # # =============================================================================

# # MU0_OVER_4PI = 1e-7

# # # ----  region boundaries  ----
# # REGION1_H_MAX = 0.040   # Region 1: h < 0.040
# # REGION2_H_MAX = 0.055   # Region 2: 0.040 <= h < 0.055 ; Region 3: h >= 0.055
# # SAMPLES_PER_REGION = 50  # Stage 1: up to 50 points sampled per region

# # # ----  Stage 1 regularization weights (physical priors)  ----
# # # These penalize deviation from the design/nominal sensor pose so the
# # # optimizer only moves a parameter away from its nominal value when the
# # # voltage data actually demands it. Offset is intentionally NOT regularized.
# # # NOTE: theta/phi (orientation) removed entirely -- sensor direction is
# # # fixed to straight-up [0, 0, 1], so there is no orientation prior/lambda.
# # LAMBDA_POS = 0    # position prior weight (x, y, z)   [1/m^2 scale]
# # LAMBDA_GAIN = 0   # gain prior weight                  [1/(V/T)^2 scale]

# # # =============================================================================
# # # DIPOLE MODEL
# # # =============================================================================

# # def dipole_field(r_vec, m_vec):
# #     """Calculate magnetic field from dipole model"""
# #     r = np.linalg.norm(r_vec, axis=1, keepdims=True)
    
# #     r3 = np.maximum(r**3, 1e-12)
# #     r5 = np.maximum(r**5, 1e-12)
    
# #     mdotr = np.sum(m_vec * r_vec, axis=1, keepdims=True)
    
# #     B = MU0_OVER_4PI * (
# #         3.0 * r_vec * mdotr / r5 - m_vec / r3
# #     )
    
# #     return B


# # # =============================================================================
# # # LOAD SENSOR POSITIONS
# # # =============================================================================

# # def load_sensor_positions(file_path):
# #     """Load sensor positions from CSV"""
# #     df = pd.read_csv(file_path)
# #     sensor_positions = df.values
# #     print(f"Loaded sensor positions: {sensor_positions.shape}")
# #     return sensor_positions

# # # =============================================================================
# # # LOAD ROBOT POSE
# # # =============================================================================

# # def load_robot_pose(file_path):
# #     """Load robot positions and magnetic orientations"""
# #     df = pd.read_csv(file_path)
    
# #     required_cols = ['x', 'y', 'z', 'mx', 'my', 'mz']
# #     for c in required_cols:
# #         if c not in df.columns:
# #             raise ValueError(f"Missing column: {c}")

# #     positions = df[['x', 'y', 'z']].values
# #     m_world = df[['mx', 'my', 'mz']].values
    
# #     # Normalize magnetic orientation
# #     norm = np.linalg.norm(m_world, axis=1, keepdims=True)
# #     m_world = m_world / norm
    
# #     print(f"Loaded robot positions: {positions.shape}")
# #     print(f"Loaded magnetic orientations: {m_world.shape}")
    
# #     return positions, m_world


# # # =============================================================================
# # # LOAD VOLTAGE DATA
# # # =============================================================================

# # def load_voltage_data(file_path):
# #     """Load voltage measurements"""
# #     df = pd.read_csv(file_path)
# #     voltage = df.values
# #     print(f"Loaded voltage data: {voltage.shape}")
# #     return voltage


# # # =============================================================================
# # # LOAD OFFSET FILE
# # # =============================================================================

# # def load_sensor_offsets(file_path):
# #     """Load sensor offsets"""
# #     df = pd.read_csv(file_path, header=0)
# #     offsets = df.iloc[:, 1].values
# #     print(f"Loaded offsets: {offsets.shape}")
# #     return offsets


# # # =============================================================================
# # # RESIDUAL FUNCTION
# # # =============================================================================

# # def sensor_residuals(params, robot_positions, m_world, voltage_sensor,
# #                       pos_prior=None, g0=7.5,
# #                       lambda_pos=LAMBDA_POS, lambda_gain=LAMBDA_GAIN):
# #     """
# #     Calculate residuals between measured and predicted voltages, PLUS
# #     regularization residuals that pull (x,y,z,gain) toward their
# #     nominal/physical prior values. Offset 'a' is left unregularized.

# #     params: [x, y, z, a, g]
# #     Sensor direction is FIXED to straight up [0, 0, 1] -- theta/phi are no
# #     longer free parameters (sensors are assumed soldered perpendicular to
# #     the PCB with negligible tilt).

# #     pos_prior: (x0, y0, z0) nominal sensor position from
# #                Hall_sensor_positions.csv (design position). If None, no
# #                position regularization terms are appended (kept for safety;
# #                in normal use this is always provided by the caller).
# #     """
# #     x, y, z, a, g = params

# #     # Fixed sensor direction (straight up, perpendicular to PCB)
# #     sensor_dir = np.array([0.0, 0.0, 1.0])

# #     # Sensor position
# #     sensor_pos = np.array([x, y, z])

# #     # Vector from robot to sensor
# #     r_vec = sensor_pos - robot_positions

# #     # Calculate magnetic field at sensor position
# #     B = dipole_field(r_vec, m_world)

# #     # Project magnetic field onto sensor direction
# #     B_proj = B @ sensor_dir

# #     # Predicted voltage
# #     voltage_pred = a + g * B_proj

# #     # ---- Voltage residual (unchanged) ----
# #     r_voltage = voltage_sensor - voltage_pred

# #     # ---- Regularization residuals (physical priors) ----
# #     # NOTE: concatenated onto the residual vector, NOT added to the scalar
# #     # cost -- appending sqrt(lambda) * (param - prior) as extra residual
# #     # entries is exactly equivalent to adding lambda * (param - prior)^2
# #     # to the cost, keeping us fully inside least_squares()/'trf'.
# #     if pos_prior is not None:
# #         x0, y0, z0 = pos_prior
# #         r_pos = np.sqrt(lambda_pos) * np.array([x - x0, y - y0, z - z0])
# #     else:
# #         r_pos = np.array([])

# #     r_gain = np.sqrt(lambda_gain) * np.array([g - g0])

# #     residual = np.concatenate([r_voltage, r_pos, r_gain])

# #     return residual


# # def sensor_residuals_xyz_frozen(params, robot_positions, m_world, voltage_sensor,
# #                                  pos_fixed, g0=7.5, lambda_gain=LAMBDA_GAIN):
# #     """
# #     Bien the CUA sensor_residuals VOI CA x,y,z DEU BI DONG BANG -- chi con
# #     2 tham so tu do: offset (a) va gain (g). Dung de kiem tra dut diem xem
# #     suy bien gain co bien mat hoan toan khi vi tri sensor bi khoa cung
# #     hoan toan hay khong (thi nghiem quyet dinh sau khi z-only-frozen van
# #     con corr(x,gain)/corr(y,gain) cao).

# #     params: [a, g]
# #     pos_fixed: (x0, y0, z0) vi tri sensor co dinh hoan toan (khong toi uu)
# #     """
# #     a, g = params

# #     sensor_dir = np.array([0.0, 0.0, 1.0])
# #     sensor_pos = np.array(pos_fixed)

# #     r_vec = sensor_pos - robot_positions
# #     B = dipole_field(r_vec, m_world)
# #     B_proj = B @ sensor_dir

# #     voltage_pred = a + g * B_proj
# #     r_voltage = voltage_sensor - voltage_pred

# #     r_gain = np.sqrt(lambda_gain) * np.array([g - g0])

# #     residual = np.concatenate([r_voltage, r_gain])

# #     return residual


# # def sensor_residuals_z_frozen(params, robot_positions, m_world, voltage_sensor,
# #                                z_fixed, pos_prior=None, g0=7.5,
# #                                lambda_pos=LAMBDA_POS, lambda_gain=LAMBDA_GAIN):
# #     """
# #     Bien the CUA sensor_residuals VOI z BI DONG BANG (khong toi uu).
# #     Dung de pha vo suy bien z-gain (corr(z,gain) ~ -0.9 da do duoc): z duoc
# #     lay thang tu ban ve co khi (sensor_pos_init[2]), khong con la tham so
# #     tu do de "danh doi" voi gain nua.

# #     params: [x, y, a, g]   (chi 4 tham so, thieu z so voi ban goc)
# #     z_fixed: gia tri z co dinh (thuong la sensor_pos_init[2])

# #     pos_prior: (x0, y0) -- CHI con x,y (z da bi dong bang, khong can
# #                regularize rieng vi no khong con la bien tu do).
# #     """
# #     x, y, a, g = params

# #     sensor_dir = np.array([0.0, 0.0, 1.0])
# #     sensor_pos = np.array([x, y, z_fixed])

# #     r_vec = sensor_pos - robot_positions
# #     B = dipole_field(r_vec, m_world)
# #     B_proj = B @ sensor_dir

# #     voltage_pred = a + g * B_proj
# #     r_voltage = voltage_sensor - voltage_pred

# #     if pos_prior is not None:
# #         x0, y0 = pos_prior
# #         r_pos = np.sqrt(lambda_pos) * np.array([x - x0, y - y0])
# #     else:
# #         r_pos = np.array([])

# #     r_gain = np.sqrt(lambda_gain) * np.array([g - g0])

# #     residual = np.concatenate([r_voltage, r_pos, r_gain])

# #     return residual


# # # =============================================================================
# # # SINGLE SENSOR CALIBRATION
# # # =============================================================================

# # # =============================================================================
# # # IDENTIFIABILITY / CORRELATION ANALYSIS (Stage 1 post-fit diagnostic)
# # # =============================================================================
# # # Muc dich: kiem tra xem gain co bi "nhap nhang" (correlated) voi cac tham so
# # # khac (dac biet la z, vi dipole suy giam theo 1/r^3 nen z va gain co the
# # # danh doi lan nhau) hay khong. Neu |corr| gan 1, gain KHONG duoc xac dinh
# # # duy nhat tu du lieu voltage -- tuc gain fit duoc co the chi la 1 nghiem
# # # trong vo so nghiem tuong duong, khong phai gia tri vat ly that cua sensor.
# # #
# # # Dung result.jac cua chinh least_squares() da tra ve -- KHONG chay lai toi
# # # uu hoa, khong doi thuat toan/method='trf'.
# # PARAM_NAMES = ["x", "y", "z", "offset", "gain"]
# # CORRELATION_LOG = []  # tich luy 1 dict/sensor, dung de in bang tong ket cuoi


# # def analyze_gain_identifiability(result, n_voltage, sensor_index, verbose=True):
# #     """
# #     Tinh ma tran covariance/correlation giua 5 tham so (x,y,z,offset,gain)
# #     tu Jacobian tai nghiem hoi tu, CHI dung phan voltage residual (bo phan
# #     regularization residual neu co, vi regularization lam gia tao giam
# #     correlation -- xem giai thich o tren).

# #     Tra ve dict {sensor_index, corr_x_gain, corr_y_gain, corr_z_gain,
# #     corr_offset_gain, n_voltage, n_params, is_rank_deficient}.
# #     """
# #     n_params = len(result.x)

# #     # Chi lay J ung voi voltage residual (n_voltage dong dau), bo residual
# #     # regularization (pos/gain prior) neu dang bat regularize.
# #     J_voltage = result.jac[:n_voltage, :]
# #     residual_voltage = result.fun[:n_voltage]

# #     dof = n_voltage - n_params
# #     if dof <= 0:
# #         row = {
# #             "sensor_index": sensor_index,
# #             "corr_x_gain": np.nan, "corr_y_gain": np.nan,
# #             "corr_z_gain": np.nan, "corr_offset_gain": np.nan,
# #             "n_voltage": n_voltage, "n_params": n_params,
# #             "is_rank_deficient": True,
# #         }
# #         CORRELATION_LOG.append(row)
# #         return row

# #     sigma2 = np.sum(residual_voltage ** 2) / dof

# #     JTJ = J_voltage.T @ J_voltage
# #     try:
# #         JTJ_inv = np.linalg.inv(JTJ)
# #         is_rank_deficient = False
# #     except np.linalg.LinAlgError:
# #         # JTJ suy bien (singular) -- day la dau hieu MANH NHAT cua non-
# #         # identifiability: mot so to hop tham so hoan toan khong the tach
# #         # biet tu du lieu nay. Dung pseudo-inverse de van tinh duoc con so
# #         # tham khao, nhung phai canh bao ro.
# #         JTJ_inv = np.linalg.pinv(JTJ)
# #         is_rank_deficient = True

# #     cov = sigma2 * JTJ_inv
# #     diag = np.diag(cov)

# #     diag_safe = np.where(diag > 0, diag, np.nan)
# #     denom = np.sqrt(np.outer(diag_safe, diag_safe))
# #     corr_matrix = cov / denom

# #     gain_idx = PARAM_NAMES.index("gain")
# #     row = {
# #         "sensor_index": sensor_index,
# #         "corr_x_gain": corr_matrix[PARAM_NAMES.index("x"), gain_idx],
# #         "corr_y_gain": corr_matrix[PARAM_NAMES.index("y"), gain_idx],
# #         "corr_z_gain": corr_matrix[PARAM_NAMES.index("z"), gain_idx],
# #         "corr_offset_gain": corr_matrix[PARAM_NAMES.index("offset"), gain_idx],
# #         "n_voltage": n_voltage, "n_params": n_params,
# #         "is_rank_deficient": is_rank_deficient,
# #     }
# #     CORRELATION_LOG.append(row)

# #     if verbose:
# #         flag = " [RANK-DEFICIENT!]" if is_rank_deficient else ""
# #         print(f"    corr(z,gain)={row['corr_z_gain']:+.3f}  "
# #               f"corr(offset,gain)={row['corr_offset_gain']:+.3f}  "
# #               f"corr(x,gain)={row['corr_x_gain']:+.3f}  "
# #               f"corr(y,gain)={row['corr_y_gain']:+.3f}{flag}")

# #     return row


# # def print_correlation_summary():
# #     """In bang tong ket |corr(*, gain)| trung binh/max qua 64 sensor, va
# #     liet ke cac sensor co correlation cao (|corr| > 0.8) -- nghi van manh
# #     ve identifiability."""
# #     if not CORRELATION_LOG:
# #         print("Chua co du lieu correlation nao duoc ghi lai.")
# #         return

# #     z_gain = np.array([r["corr_z_gain"] for r in CORRELATION_LOG])
# #     offset_gain = np.array([r["corr_offset_gain"] for r in CORRELATION_LOG])
# #     x_gain = np.array([r["corr_x_gain"] for r in CORRELATION_LOG])
# #     y_gain = np.array([r["corr_y_gain"] for r in CORRELATION_LOG])
# #     n_deficient = sum(r["is_rank_deficient"] for r in CORRELATION_LOG)

# #     print("\n" + "=" * 60)
# #     print("IDENTIFIABILITY SUMMARY (correlation with gain, 64 sensors)")
# #     print("=" * 60)
# #     print(f"{'pair':<18}{'mean |corr|':>14}{'max |corr|':>14}")
# #     print(f"{'z    - gain':<18}{np.nanmean(np.abs(z_gain)):>14.3f}"
# #           f"{np.nanmax(np.abs(z_gain)):>14.3f}")
# #     print(f"{'offset - gain':<18}{np.nanmean(np.abs(offset_gain)):>14.3f}"
# #           f"{np.nanmax(np.abs(offset_gain)):>14.3f}")
# #     print(f"{'x    - gain':<18}{np.nanmean(np.abs(x_gain)):>14.3f}"
# #           f"{np.nanmax(np.abs(x_gain)):>14.3f}")
# #     print(f"{'y    - gain':<18}{np.nanmean(np.abs(y_gain)):>14.3f}"
# #           f"{np.nanmax(np.abs(y_gain)):>14.3f}")
# #     print(f"\nSo sensor bi rank-deficient (JTJ singular): {n_deficient}/64")

# #     HIGH_CORR_THRESHOLD = 0.8
# #     print(f"\nSensor co |corr(z,gain)| > {HIGH_CORR_THRESHOLD}:")
# #     flagged = [r for r in CORRELATION_LOG
# #                if not np.isnan(r["corr_z_gain"])
# #                and abs(r["corr_z_gain"]) > HIGH_CORR_THRESHOLD]
# #     if flagged:
# #         for r in flagged:
# #             print(f"  Sensor {r['sensor_index']+1:02d}: "
# #                   f"corr(z,gain) = {r['corr_z_gain']:+.3f}")
# #     else:
# #         print("  (khong co)")

# #     print(f"\nSensor co |corr(offset,gain)| > {HIGH_CORR_THRESHOLD}:")
# #     flagged = [r for r in CORRELATION_LOG
# #                if not np.isnan(r["corr_offset_gain"])
# #                and abs(r["corr_offset_gain"]) > HIGH_CORR_THRESHOLD]
# #     if flagged:
# #         for r in flagged:
# #             print(f"  Sensor {r['sensor_index']+1:02d}: "
# #                   f"corr(offset,gain) = {r['corr_offset_gain']:+.3f}")
# #     else:
# #         print("  (khong co)")
# #     print("=" * 60)


# # # ---- BIEN THE CHO TRUONG HOP Z BI DONG BANG (chi con 4 tham so: x,y,a,g) ----
# # PARAM_NAMES_FROZEN = ["x", "y", "offset", "gain"]
# # CORRELATION_LOG_FROZEN = []


# # def analyze_gain_identifiability_frozen(result, n_voltage, sensor_index, verbose=True):
# #     """
# #     Ban sao cua analyze_gain_identifiability() nhung cho truong hop z bi
# #     dong bang -- chi con 4 tham so (x,y,offset,gain), khong con cot z
# #     trong Jacobian nen khong the/khong can tinh corr(z,gain) nua.
# #     """
# #     n_params = len(result.x)  # = 4
# #     J_voltage = result.jac[:n_voltage, :]
# #     residual_voltage = result.fun[:n_voltage]

# #     dof = n_voltage - n_params
# #     if dof <= 0:
# #         row = {
# #             "sensor_index": sensor_index,
# #             "corr_x_gain": np.nan, "corr_y_gain": np.nan,
# #             "corr_offset_gain": np.nan,
# #             "n_voltage": n_voltage, "n_params": n_params,
# #             "is_rank_deficient": True,
# #         }
# #         CORRELATION_LOG_FROZEN.append(row)
# #         return row

# #     sigma2 = np.sum(residual_voltage ** 2) / dof
# #     JTJ = J_voltage.T @ J_voltage
# #     try:
# #         JTJ_inv = np.linalg.inv(JTJ)
# #         is_rank_deficient = False
# #     except np.linalg.LinAlgError:
# #         JTJ_inv = np.linalg.pinv(JTJ)
# #         is_rank_deficient = True

# #     cov = sigma2 * JTJ_inv
# #     diag = np.diag(cov)
# #     diag_safe = np.where(diag > 0, diag, np.nan)
# #     denom = np.sqrt(np.outer(diag_safe, diag_safe))
# #     corr_matrix = cov / denom

# #     gain_idx = PARAM_NAMES_FROZEN.index("gain")
# #     row = {
# #         "sensor_index": sensor_index,
# #         "corr_x_gain": corr_matrix[PARAM_NAMES_FROZEN.index("x"), gain_idx],
# #         "corr_y_gain": corr_matrix[PARAM_NAMES_FROZEN.index("y"), gain_idx],
# #         "corr_offset_gain": corr_matrix[PARAM_NAMES_FROZEN.index("offset"), gain_idx],
# #         "n_voltage": n_voltage, "n_params": n_params,
# #         "is_rank_deficient": is_rank_deficient,
# #     }
# #     CORRELATION_LOG_FROZEN.append(row)

# #     if verbose:
# #         flag = " [RANK-DEFICIENT!]" if is_rank_deficient else ""
# #         print(f"    [Z FROZEN] corr(offset,gain)={row['corr_offset_gain']:+.3f}  "
# #               f"corr(x,gain)={row['corr_x_gain']:+.3f}  "
# #               f"corr(y,gain)={row['corr_y_gain']:+.3f}{flag}")

# #     return row


# # def print_correlation_summary_frozen():
# #     """Ban sao cua print_correlation_summary() cho truong hop z bi dong bang."""
# #     if not CORRELATION_LOG_FROZEN:
# #         print("Chua co du lieu correlation (z-frozen) nao duoc ghi lai.")
# #         return

# #     offset_gain = np.array([r["corr_offset_gain"] for r in CORRELATION_LOG_FROZEN])
# #     x_gain = np.array([r["corr_x_gain"] for r in CORRELATION_LOG_FROZEN])
# #     y_gain = np.array([r["corr_y_gain"] for r in CORRELATION_LOG_FROZEN])
# #     n_deficient = sum(r["is_rank_deficient"] for r in CORRELATION_LOG_FROZEN)

# #     print("\n" + "=" * 60)
# #     print("IDENTIFIABILITY SUMMARY -- Z FROZEN (correlation with gain, 64 sensors)")
# #     print("=" * 60)
# #     print(f"{'pair':<18}{'mean |corr|':>14}{'max |corr|':>14}")
# #     print(f"{'offset - gain':<18}{np.nanmean(np.abs(offset_gain)):>14.3f}"
# #           f"{np.nanmax(np.abs(offset_gain)):>14.3f}")
# #     print(f"{'x    - gain':<18}{np.nanmean(np.abs(x_gain)):>14.3f}"
# #           f"{np.nanmax(np.abs(x_gain)):>14.3f}")
# #     print(f"{'y    - gain':<18}{np.nanmean(np.abs(y_gain)):>14.3f}"
# #           f"{np.nanmax(np.abs(y_gain)):>14.3f}")
# #     print(f"\nSo sensor bi rank-deficient (JTJ singular): {n_deficient}/64")
# #     print("=" * 60)


# # # ---- BIEN THE CHO TRUONG HOP CA X,Y,Z DEU BI DONG BANG (chi con 2 tham so: a,g) ----
# # PARAM_NAMES_XYZ_FROZEN = ["offset", "gain"]
# # CORRELATION_LOG_XYZ_FROZEN = []


# # def analyze_gain_identifiability_xyz_frozen(result, n_voltage, sensor_index, verbose=True):
# #     """
# #     Ban sao cua analyze_gain_identifiability() cho truong hop x,y,z DEU bi
# #     dong bang -- chi con 2 tham so (offset, gain). Day la thi nghiem
# #     "quyet dinh": neu corr(offset,gain) van cao o day, suy bien khong lien
# #     quan gi den vi tri sensor ca -- ban than offset va gain von da nhap
# #     nhang voi nhau trong chinh cong thuc V = a + g*Bz (voi Bz co the mang
# #     ca gia tri gan 0, khien a va g kho tach biet).
# #     """
# #     n_params = len(result.x)  # = 2
# #     J_voltage = result.jac[:n_voltage, :]
# #     residual_voltage = result.fun[:n_voltage]

# #     dof = n_voltage - n_params
# #     if dof <= 0:
# #         row = {
# #             "sensor_index": sensor_index,
# #             "corr_offset_gain": np.nan,
# #             "n_voltage": n_voltage, "n_params": n_params,
# #             "is_rank_deficient": True,
# #         }
# #         CORRELATION_LOG_XYZ_FROZEN.append(row)
# #         return row

# #     sigma2 = np.sum(residual_voltage ** 2) / dof
# #     JTJ = J_voltage.T @ J_voltage
# #     try:
# #         JTJ_inv = np.linalg.inv(JTJ)
# #         is_rank_deficient = False
# #     except np.linalg.LinAlgError:
# #         JTJ_inv = np.linalg.pinv(JTJ)
# #         is_rank_deficient = True

# #     cov = sigma2 * JTJ_inv
# #     diag = np.diag(cov)
# #     diag_safe = np.where(diag > 0, diag, np.nan)
# #     denom = np.sqrt(np.outer(diag_safe, diag_safe))
# #     corr_matrix = cov / denom

# #     gain_idx = PARAM_NAMES_XYZ_FROZEN.index("gain")
# #     offset_idx = PARAM_NAMES_XYZ_FROZEN.index("offset")
# #     row = {
# #         "sensor_index": sensor_index,
# #         "corr_offset_gain": corr_matrix[offset_idx, gain_idx],
# #         "n_voltage": n_voltage, "n_params": n_params,
# #         "is_rank_deficient": is_rank_deficient,
# #     }
# #     CORRELATION_LOG_XYZ_FROZEN.append(row)

# #     if verbose:
# #         flag = " [RANK-DEFICIENT!]" if is_rank_deficient else ""
# #         print(f"    [XYZ FROZEN] corr(offset,gain)={row['corr_offset_gain']:+.3f}{flag}")

# #     return row


# # def print_correlation_summary_xyz_frozen():
# #     """Ban sao cua print_correlation_summary() cho truong hop x,y,z deu
# #     bi dong bang -- chi con corr(offset,gain) de xem xet."""
# #     if not CORRELATION_LOG_XYZ_FROZEN:
# #         print("Chua co du lieu correlation (xyz-frozen) nao duoc ghi lai.")
# #         return

# #     offset_gain = np.array([r["corr_offset_gain"] for r in CORRELATION_LOG_XYZ_FROZEN])
# #     n_deficient = sum(r["is_rank_deficient"] for r in CORRELATION_LOG_XYZ_FROZEN)

# #     print("\n" + "=" * 60)
# #     print("IDENTIFIABILITY SUMMARY -- X,Y,Z FROZEN (correlation with gain, 64 sensors)")
# #     print("=" * 60)
# #     print(f"{'pair':<18}{'mean |corr|':>14}{'max |corr|':>14}")
# #     print(f"{'offset - gain':<18}{np.nanmean(np.abs(offset_gain)):>14.3f}"
# #           f"{np.nanmax(np.abs(offset_gain)):>14.3f}")
# #     print(f"\nSo sensor bi rank-deficient (JTJ singular): {n_deficient}/64")
# #     print("=" * 60)

# # def calibrate_single_sensor(
# #         sensor_index,
# #         sensor_pos_init,
# #         robot_positions,
# #         m_world,
# #         voltage_sensor,
# #         offset_init=1.618):
# #     """
# #     Calibrate a single sensor -- CHI con offset + gain la tham so tu do.
# #     Sensor direction is fixed straight up (perpendicular to PCB);
# #     theta/phi are no longer free parameters.

# #     CA X,Y,Z DEU BI DONG BANG (khong toi uu) -- lay thang tu
# #     sensor_pos_init (ban ve co khi). Day la thi nghiem quyet dinh: sau khi
# #     z-only-frozen van con corr(x,gain)/corr(y,gain) cao (~0.4 mean, ~0.95
# #     max), ta kiem tra xem khoa CA vi tri sensor co xoa duoc suy bien gain
# #     hay khong, hay ban chat van la V=a+g*Bz von da nhap nhang offset-gain.
# #     """
# #     # ----------------------------------------------------
# #     # INITIAL VALUES
# #     # ----------------------------------------------------
# #     g0 = 7.5  # Initial gain (V/T)
# #     offset_init = 1.618         # a (offset)
# #     pos_fixed = tuple(sensor_pos_init)  # X,Y,Z DONG BANG HOAN TOAN

# #     # ----------------------------------------------------
# #     # INITIAL PARAMETERS (chi con 2: offset, gain)
# #     # ----------------------------------------------------
# #     x0 = np.array([
# #         offset_init,  # a (offset)
# #         g0            # g (gain)
# #     ])
    
# #     # ----------------------------------------------------
# #     # BOUNDS
# #     # ----------------------------------------------------
# #     lower = [
# #         offset_init - 0.02,   # a min (20mV)
# #         -9                     # g min
# #     ]
    
# #     upper = [
# #         offset_init + 0.02,   # a max (20mV)
# #         9                      # g max
# #     ]
    
# #     # ----------------------------------------------------
# #     # OPTIMIZATION
# #     # ----------------------------------------------------
# #     result = least_squares(
# #         sensor_residuals_xyz_frozen,
# #         x0,
# #         bounds=(lower, upper),
# #         args=(
# #             robot_positions,
# #             m_world,
# #             voltage_sensor,
# #             pos_fixed
# #         ),
# #         kwargs=dict(
# #             g0=g0,
# #         ),
# #         method='trf',
# #         max_nfev=250           
# #     )
    
# #     # ----------------------------------------------------
# #     # EXTRACT OPTIMIZED PARAMETERS
# #     # ----------------------------------------------------
# #     params_opt = result.x  # [a, g]
    
# #     # Direction is fixed straight up -- theta/phi kept at 0 rad so the
# #     # OUTPUT FORMAT (10 columns) stays identical to the previous script
# #     # for downstream compatibility (save_results, save_physical_results,
# #     # plot_direction_vectors, Stage 2 alpha).
# #     theta_opt, phi_opt = 0.0, 0.0
# #     nx, ny, nz = 0.0, 0.0, 1.0
    
# #     # Create extended parameter array for saving (x,y,z lay tu pos_fixed,
# #     # khong phai tu params_opt vi ca 3 deu khong con trong danh sach
# #     # tham so toi uu)
# #     params_extended = np.array([
# #         pos_fixed[0],   # x (DONG BANG, khong doi)
# #         pos_fixed[1],   # y (DONG BANG, khong doi)
# #         pos_fixed[2],   # z (DONG BANG, khong doi)
# #         params_opt[0],  # a
# #         params_opt[1],  # g
# #         nx, ny, nz,     # direction vector components (fixed straight up)
# #         theta_opt,      # theta (radians, fixed = 0)
# #         phi_opt         # phi (radians, fixed = 0)
# #     ])
    
# #     # ----------------------------------------------------
# #     # CALCULATE RMSE
# #     # (result.fun = [voltage residuals, gain reg];
# #     #  RMSE must be computed from the voltage part only, so it still
# #     #  means "V measured vs V predicted" and stays comparable to before)
# #     # ----------------------------------------------------
# #     n_voltage = voltage_sensor.shape[0]
# #     rmse = np.sqrt(np.mean(result.fun[:n_voltage]**2))

# #     # ---- IDENTIFIABILITY CHECK (bien the xyz-frozen, chi con 2 tham so) ----
# #     analyze_gain_identifiability_xyz_frozen(result, n_voltage, sensor_index)

# #     angle_from_vertical = np.rad2deg(abs(theta_opt))
# #     print(f"Sensor {sensor_index+1:02d} | RMSE = {rmse:.6f} | "
# #           f"Angle from Z = {angle_from_vertical:.2f}° | "
# #           f"Dir = [{nx:.3f}, {ny:.3f}, {nz:.3f}]")
    
# #     return params_extended, rmse


# # # =============================================================================
# # # FULL CALIBRATION
# # # =============================================================================

# # def run_calibration(
# #         sensor_positions,
# #         offsets,
# #         robot_positions,
# #         m_world,
# #         voltage_data):
# #     """
# #     Calibrate all sensors
# #     """
# #     n_sensors = sensor_positions.shape[0]
# #     results = []
# #     rmses = []
    
# #     for i in range(n_sensors):
# #         params, rmse = calibrate_single_sensor(
# #             sensor_index=i,
# #             sensor_pos_init=sensor_positions[i],
# #             robot_positions=robot_positions,
# #             m_world=m_world,
# #             voltage_sensor=voltage_data[:, i]
# #         )
        
# #         results.append(params)
# #         rmses.append(rmse)
    
# #     # ---- In bang tong ket identifiability sau khi calib het 64 sensor ----
# #     # (dung ban X,Y,Z-FROZEN vi calibrate_single_sensor gio dong bang ca 3 truc)
# #     print_correlation_summary_xyz_frozen()
    
# #     return np.array(results), np.array(rmses)


# # # =============================================================================
# # # REGION SELECTION
# # # =============================================================================

# # def select_region_samples(
# #         robot_positions,
# #         m_world,
# #         voltage_data,
# #         sensor_z,
# #         h_min,
# #         h_max=None,
# #         max_samples=210,
# #         return_indices=False):

# #     h = robot_positions[:, 2] - sensor_z

# #     lower_ok = np.ones_like(h, dtype=bool) if h_min is None else (h >= h_min)
# #     upper_ok = np.ones_like(h, dtype=bool) if h_max is None else (h < h_max)
# #     region_idx = np.where(lower_ok & upper_ok)[0]

# #     print(f"\nRegion [{h_min}, {h_max}] contains {len(region_idx)} samples")

# #     if len(region_idx) > max_samples:
# #         # NOTE: no random seed is set here, per the calibration spec.
# #         region_idx = np.random.choice(
# #             region_idx,
# #             size=max_samples,
# #             replace=False
# #         )

# #     print(f"Using {len(region_idx)} samples")

# #     if return_indices:
# #         return (
# #             robot_positions[region_idx],
# #             m_world[region_idx],
# #             voltage_data[region_idx],
# #             region_idx
# #         )

# #     return (
# #         robot_positions[region_idx],
# #         m_world[region_idx],
# #         voltage_data[region_idx]
# #     )


# # # =============================================================================
# # # NEW: STAGE 1 - STRATIFIED CALIBRATION SET (fixed, reused in Stage 2)
# # # =============================================================================

# # def select_stage1_calibration_set(
# #         robot_positions,
# #         m_world,
# #         voltage_data,
# #         sensor_z_ref,
# #         per_region=SAMPLES_PER_REGION):

# #     rp1, mw1, vd1, idx1 = select_region_samples(
# #         robot_positions, m_world, voltage_data, sensor_z_ref,
# #         h_min=None, h_max=REGION1_H_MAX,
# #         max_samples=per_region, return_indices=True
# #     )
# #     rp2, mw2, vd2, idx2 = select_region_samples(
# #         robot_positions, m_world, voltage_data, sensor_z_ref,
# #         h_min=REGION1_H_MAX, h_max=REGION2_H_MAX,
# #         max_samples=per_region, return_indices=True
# #     )
# #     rp3, mw3, vd3, idx3 = select_region_samples(
# #         robot_positions, m_world, voltage_data, sensor_z_ref,
# #         h_min=REGION2_H_MAX, h_max=None,
# #         max_samples=per_region, return_indices=True
# #     )

# #     calib_indices = np.concatenate([idx1, idx2, idx3])
# #     rp_calib = np.concatenate([rp1, rp2, rp3], axis=0)
# #     mw_calib = np.concatenate([mw1, mw2, mw3], axis=0)
# #     vd_calib = np.concatenate([vd1, vd2, vd3], axis=0)

# #     print(f"\n[Stage 1] Combined calibration set: {len(calib_indices)} "
# #           f"points (Region1={len(idx1)}, Region2={len(idx2)}, "
# #           f"Region3={len(idx3)})")

# #     return calib_indices, rp_calib, mw_calib, vd_calib


# # # =============================================================================
# # # NEW: STAGE 2 - CLOSED-FORM ALPHA PER REGION
# # # =============================================================================

# # def calibrate_alpha_by_region(physical_results, rp_calib, mw_calib, vd_calib):

# #     n_samples = rp_calib.shape[0]
# #     n_sensors = physical_results.shape[0]

# #     sensor_pos = physical_results[:, 0:3]      # (n_sensors, 3): x, y, z
# #     a = physical_results[:, 3]                  # (n_sensors,)  offset
# #     g = physical_results[:, 4]                  # (n_sensors,)  gain
# #     sensor_dir = physical_results[:, 5:8]       # (n_sensors, 3): nx, ny, nz

# #     # h[i, s] = z_capsule_i - z_sensor_calibrated_s  (uses Stage-1 result)
# #     z_calibrated = sensor_pos[:, 2]
# #     h = rp_calib[:, 2][:, None] - z_calibrated[None, :]   # (n_samples, n_sensors)

# #     # Raw dipole projection B_proj[i, s] using the FROZEN calibrated pose
# #     # (position + direction) of each sensor -- gain/offset are NOT
# #     # reapplied here, they're multiplied in separately below.
# #     B_proj = np.zeros((n_samples, n_sensors))
# #     for s in range(n_sensors):
# #         r_vec = sensor_pos[s] - rp_calib               # (n_samples, 3)
# #         B = dipole_field(r_vec, mw_calib)               # (n_samples, 3)
# #         B_proj[:, s] = B @ sensor_dir[s]

# #     gB = g[None, :] * B_proj                            # (n_samples, n_sensors)
# #     v_minus_a = vd_calib - a[None, :]                    # V_measured - a

# #     region_masks = {
# #         1: h < REGION1_H_MAX,
# #         2: (h >= REGION1_H_MAX) & (h < REGION2_H_MAX),
# #         3: h >= REGION2_H_MAX
# #     }

# #     alphas = {}
# #     for region, mask in region_masks.items():
# #         numerator = np.sum(gB[mask] * v_minus_a[mask])
# #         denominator = np.sum(gB[mask] ** 2)
# #         alpha = numerator / denominator if denominator > 0 else np.nan
# #         print(f"Region {region}: alpha = {alpha:.6f} "
# #               f"(from {np.sum(mask)} sample-sensor pairs)")
# #         alphas[region] = alpha

# #     return alphas


# # # =============================================================================
# # # SAVE RESULTS
# # # =============================================================================

# # def save_results(results, rmses, output_file):
# #     """
# #     Save calibration results to CSV
# #     """
# #     df = pd.DataFrame({
# #         "sensor_id": np.arange(len(results)),
# #         "x": results[:, 0],
# #         "y": results[:, 1],
# #         "z": results[:, 2],
# #         "offset_a": results[:, 3],
# #         "gain_g": results[:, 4],
# #         "nx": results[:, 5],
# #         "ny": results[:, 6],
# #         "nz": results[:, 7],
# #         "theta_rad": results[:, 8],
# #         "phi_rad": results[:, 9],
# #         "angle_from_z_deg": np.rad2deg(np.abs(results[:, 8])),
# #         "rmse": rmses
# #     })
    
# #     df.to_csv(output_file, index=False)
# #     print(f"\nSaved: {output_file}")


# # # =============================================================================
# # # NEW: SAVE STAGE 1 / STAGE 2 RESULTS (required output files)
# # # =============================================================================

# # def save_physical_results(results, output_file):
# #     """
# #     Save Stage 1 frozen physical parameters to Calibration_Physical.csv
# #     with columns: sensor_index, x, y, z, offset, gain, theta, phi
# #     """
# #     df = pd.DataFrame({
# #         "sensor_index": np.arange(len(results)),
# #         "x": results[:, 0],
# #         "y": results[:, 1],
# #         "z": results[:, 2],
# #         "offset": results[:, 3],
# #         "gain": results[:, 4],
# #         "theta": results[:, 8],
# #         "phi": results[:, 9],
# #     })
# #     df.to_csv(output_file, index=False)
# #     print(f"\nSaved: {output_file}")


# # def save_alpha_results(alphas, output_file):
# #     """
# #     Save Stage 2 region correction coefficients to Calibration_Alpha.csv
# #     with exactly 3 rows: Region 1, Region 2, Region 3.
# #     """
# #     regions_sorted = sorted(alphas.keys())
# #     df = pd.DataFrame({
# #         "Region": [f"Region {r}" for r in regions_sorted],
# #         "Alpha": [alphas[r] for r in regions_sorted],
# #     })
# #     df.to_csv(output_file, index=False)
# #     print(f"Saved: {output_file}")


# # # =============================================================================
# # # PLOT RMSE
# # # =============================================================================

# # def plot_rmse(rmses):
# #     """Plot RMSE for all sensors"""
# #     plt.figure(figsize=(10, 5))
# #     plt.bar(np.arange(len(rmses)), rmses)
# #     plt.xlabel("Sensor Index")
# #     plt.ylabel("RMSE")
# #     plt.title("Calibration RMSE")
# #     plt.grid(True)
# #     plt.show()


# # # =============================================================================
# # # PLOT DIRECTION VECTORS
# # # =============================================================================

# # # def plot_direction_vectors(results):
# # #     """Plot direction vectors of all sensors"""
# # #     fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
# # #     sensor_ids = np.arange(len(results))
    
# # #     # Subplot 1: Direction components
# # #     ax1 = axes[0]
# # #     ax1.plot(sensor_ids, results[:, 5], 'r.-', label='nx', markersize=8)
# # #     ax1.plot(sensor_ids, results[:, 6], 'g.-', label='ny', markersize=8)
# # #     ax1.plot(sensor_ids, results[:, 7], 'b.-', label='nz', markersize=8)
# # #     ax1.set_xlabel("Sensor Index")
# # #     ax1.set_ylabel("Direction Component")
# # #     ax1.set_title("Sensor Direction Components")
# # #     ax1.legend()
# # #     ax1.grid(True)
    
# # #     # Subplot 2: Angular deviation from vertical
# # #     ax2 = axes[1]
# # #     angle_from_vertical = np.rad2deg(np.abs(results[:, 8]))
# # #     ax2.bar(sensor_ids, angle_from_vertical)
# # #     ax2.set_xlabel("Sensor Index")
# # #     ax2.set_ylabel("Angle (degrees)")
# # #     ax2.set_title("Angular Deviation from Z-axis")
# # #     ax2.grid(True)
    
# # #     # Subplot 3: Azimuthal angle
# # #     ax3 = axes[2]
# # #     phi_deg = np.rad2deg(results[:, 9])
# # #     ax3.bar(sensor_ids, phi_deg)
# # #     ax3.set_xlabel("Sensor Index")
# # #     ax3.set_ylabel("Phi (degrees)")
# # #     ax3.set_title("Azimuthal Angle (XY-plane)")
# # #     ax3.grid(True)
    
# # #     plt.tight_layout()
# # #     plt.show()


# # # =============================================================================
# # # MAIN
# # # =============================================================================

# # def main():
# #     """Main execution function -- 2-stage calibration framework"""

# #     # Load data
# #     sensor_positions = load_sensor_positions(SENSOR_POSITIONS_PATH)
# #     robot_positions, m_world = load_robot_pose(ROBOT_POSE_PATH)
# #     voltage_data = load_voltage_data(VOLTAGE_DATA_PATH)
# #     offsets = load_sensor_offsets(OFFSET_FILE_PATH)

# #     # Ensure consistent number of samples
# #     n_samples = min(len(robot_positions), len(voltage_data))
# #     robot_positions = robot_positions[:n_samples]
# #     m_world = m_world[:n_samples]
# #     voltage_data = voltage_data[:n_samples]

# #     # ------------------------------------------------------------------
# #     # NEW: Sensor reference height for Stage 1 region assignment.
# #     # Must come from the ORIGINAL (uncalibrated) sensor positions file.
# #     # ------------------------------------------------------------------
# #     sensor_z_initial_ref = sensor_positions[:, 2].mean()
# #     print(f"\nSensor reference z (initial, for Stage 1) = "
# #           f"{sensor_z_initial_ref:.6f} m")

# #     # ==================================================================
# #     # STAGE 1a: Stratified, fixed calibration set (<=150 points)
# #     # ==================================================================
# #     print("\n===================================")
# #     print("STAGE 1: SAMPLE SELECTION")
# #     print("===================================")
# #     calib_indices, rp_calib, mw_calib, vd_calib = select_stage1_calibration_set(
# #         robot_positions, m_world, voltage_data, sensor_z_initial_ref,
# #         per_region=SAMPLES_PER_REGION
# #     )
# #     # calib_indices is frozen here and reused as-is in Stage 2 below --
# #     # no re-sampling happens after this point.

# #     # ==================================================================
# #     # STAGE 1b: Per-sensor physical parameter calibration
# #     # (unchanged residual function, dipole model, initial values, bounds)
# #     # ==================================================================
# #     print("\n===================================")
# #     print("STAGE 1: PHYSICAL PARAMETER FIT")
# #     print("===================================")
# #     results, rmses = run_calibration(
# #         sensor_positions, offsets, rp_calib, mw_calib, vd_calib
# #     )

# #     print("\n========================")
# #     print(f"Mean RMSE = {np.mean(rmses):.6f}")
# #     print(f"Max RMSE  = {np.max(rmses):.6f}")
# #     print(f"Min RMSE  = {np.min(rmses):.6f}")
# #     print("========================")

# #     # Save Stage 1 physical parameters (frozen going into Stage 2)
# #     save_physical_results(results, PHYSICAL_OUTPUT_PATH)

# #     # Plots
# #     plot_rmse(rmses)
# #     # plot_direction_vectors(results)

# #     # ==================================================================
# #     # STAGE 2: Closed-form alpha per region (physical params frozen)
# #     # ==================================================================
# #     print("\n===================================")
# #     print("STAGE 2: ALPHA CORRECTION (closed-form)")
# #     print("===================================")
# #     alphas = calibrate_alpha_by_region(results, rp_calib, mw_calib, vd_calib)
# #     save_alpha_results(alphas, ALPHA_OUTPUT_PATH)

# #     print("\n===================================")
# #     print("ALL STAGES FINISHED")
# #     print("===================================")


# # if __name__ == "__main__":
# #     main()
    
# # # """
# # # Sweep lambda_pos va lambda_gain quanh vung uoc luong co co so, ve L-curve
# # # (RMSE vs do lech tham so khoi prior) de chon lambda can bang.

# # # CACH DUNG: dan doan code nay vao CUOI file calib_region_based.py (ban da
# # # co san cac ham dipole_field, sensor_residuals, calibrate_single_sensor,
# # # select_stage1_calibration_set, load_sensor_positions, load_robot_pose,
# # # load_voltage_data, v.v. trong file goc). Script nay KHONG doi thuat toan
# # # least_squares/'trf', chi goi lai calibrate_single_sensor() nhieu lan voi
# # # cac gia tri lambda khac nhau.

# # # Diem khoi dau duoc uoc luong tu cong thuc:
# # #     lambda = (RMSE_v_no_reg / sigma)^2
# # # voi RMSE_v_no_reg lay tu log ban da chay (0.000383 V), va sigma la do lech
# # # "chap nhan duoc" ban tu chon (o day: sigma_pos=0.5mm, sigma_gain=1.0 V/T).
# # # Day chi la DIEM KHOI DAU -- sweep xung quanh no de tim diem can bang that su.
# # # """

# # # import numpy as np
# # # import matplotlib.pyplot as plt

# # # # =============================================================================
# # # # CAU HINH SWEEP
# # # # =============================================================================
# # # RMSE_V_NO_REG = 0.000383  # tu log khong-regularize cua ban

# # # # Diem khoi dau uoc luong (KHONG phai gia tri cuoi cung)
# # # LAMBDA_POS_GUESS = (RMSE_V_NO_REG / 0.0005) ** 2   # sigma_pos = 0.5 mm
# # # LAMBDA_GAIN_GUESS = (RMSE_V_NO_REG / 1.0) ** 2      # sigma_gain = 1.0 V/T

# # # # Sweep tu 1/100x den 100x quanh diem khoi dau, log-spaced, 9 diem
# # # N_POINTS = 9
# # # lambda_pos_grid = np.geomspace(LAMBDA_POS_GUESS / 100, LAMBDA_POS_GUESS * 100, N_POINTS)
# # # lambda_gain_grid = np.geomspace(LAMBDA_GAIN_GUESS / 100, LAMBDA_GAIN_GUESS * 100, N_POINTS)

# # # # Sensor dai dien de sweep nhanh (thay vi chay het 64 sensor cho moi lambda).
# # # # Nen chon vai sensor o cac vi tri khac nhau (giua/canh mang) de dai dien.
# # # SENSOR_SAMPLE_IDX = [0, 15, 31, 47, 63]  # 5 sensor rai deu tren 64 sensor


# # # def run_sweep_1d(param_name, lambda_grid, sensor_positions, robot_positions,
# # #                   m_world, voltage_data, sensor_z_ref):
# # #     """
# # #     Sweep 1 chieu: giu lambda con lai = 0, chi thay doi lambda cua param_name
# # #     ('pos' hoac 'gain'). Tra ve (rmse_list, deviation_list).

# # #     Goi least_squares() TRUC TIEP (khong qua calibrate_single_sensor) vi
# # #     lambda_pos/lambda_gain trong sensor_residuals la default-argument, bi
# # #     "dong bang" luc dinh nghia ham -- doi bien global sau do KHONG co tac
# # #     dung. Truyen tuong minh qua kwargs moi dam bao dung gia tri lambda.
# # #     """
# # #     rmse_list = []
# # #     deviation_list = []  # do lech tuong doi TB khoi prior (chuan hoa theo bound)

# # #     calib_indices, rp_calib, mw_calib, vd_calib = select_stage1_calibration_set(
# # #         robot_positions, m_world, voltage_data, sensor_z_ref
# # #     )

# # #     g0 = 7.5
# # #     pos_tol = 0.0015

# # #     for lam in lambda_grid:
# # #         lambda_pos = lam if param_name == "pos" else 0.0
# # #         lambda_gain = lam if param_name == "gain" else 0.0

# # #         rmses_this_lambda = []
# # #         deviations_this_lambda = []

# # #         for s in SENSOR_SAMPLE_IDX:
# # #             sensor_pos_init = sensor_positions[s]

# # #             x0 = np.array([
# # #                 sensor_pos_init[0], sensor_pos_init[1], sensor_pos_init[2],
# # #                 1.618,  # offset_init
# # #                 g0
# # #             ])
# # #             lower = [
# # #                 sensor_pos_init[0] - pos_tol, sensor_pos_init[1] - pos_tol,
# # #                 sensor_pos_init[2] - pos_tol, 1.618 - 0.015, 4
# # #             ]
# # #             upper = [
# # #                 sensor_pos_init[0] + pos_tol, sensor_pos_init[1] + pos_tol,
# # #                 sensor_pos_init[2] + pos_tol, 1.618 + 0.015, 9
# # #             ]

# # #             result = least_squares(
# # #                 sensor_residuals,
# # #                 x0,
# # #                 bounds=(lower, upper),
# # #                 args=(rp_calib, mw_calib, vd_calib[:, s]),
# # #                 kwargs=dict(
# # #                     pos_prior=tuple(sensor_pos_init),
# # #                     g0=g0,
# # #                     lambda_pos=lambda_pos,
# # #                     lambda_gain=lambda_gain,
# # #                 ),
# # #                 method="trf",
# # #                 max_nfev=250
# # #             )

# # #             n_voltage = vd_calib.shape[0]
# # #             rmse = np.sqrt(np.mean(result.fun[:n_voltage] ** 2))
# # #             rmses_this_lambda.append(rmse)

# # #             if param_name == "pos":
# # #                 dev = np.linalg.norm(result.x[0:3] - sensor_pos_init) / pos_tol
# # #             else:  # gain
# # #                 dev = abs(result.x[4] - g0) / 5.0  # chuan hoa theo bien do bound gain [4,9]

# # #             deviations_this_lambda.append(dev)

# # #         rmse_list.append(np.mean(rmses_this_lambda))
# # #         deviation_list.append(np.mean(deviations_this_lambda))

# # #         print(f"  lambda_{param_name} = {lam:.3e} | "
# # #               f"mean RMSE = {np.mean(rmses_this_lambda):.6f} | "
# # #               f"mean |dev|/bound = {np.mean(deviations_this_lambda):.4f}")

# # #     return rmse_list, deviation_list


# # # def plot_lcurve(lambda_grid, rmse_list, deviation_list, param_name, guess_value):
# # #     fig, ax1 = plt.subplots(figsize=(7, 5))

# # #     color1 = "tab:blue"
# # #     ax1.set_xlabel(f"lambda_{param_name}")
# # #     ax1.set_ylabel("Mean RMSE (V)", color=color1)
# # #     ax1.plot(lambda_grid, rmse_list, "o-", color=color1, label="RMSE")
# # #     ax1.set_xscale("log")
# # #     ax1.tick_params(axis="y", labelcolor=color1)

# # #     ax2 = ax1.twinx()
# # #     color2 = "tab:red"
# # #     ax2.set_ylabel("Mean |deviation| / bound", color=color2)
# # #     ax2.plot(lambda_grid, deviation_list, "s--", color=color2, label="Deviation")
# # #     ax2.tick_params(axis="y", labelcolor=color2)

# # #     ax1.axvline(guess_value, color="gray", linestyle=":", alpha=0.7,
# # #                 label=f"initial guess = {guess_value:.2e}")

# # #     fig.suptitle(f"L-curve: RMSE vs Deviation ({param_name})")
# # #     fig.tight_layout()
# # #     output_path = BASE_DIR / f"lcurve_{param_name}.png"
# # #     fig.savefig(output_path, dpi=120)
# # #     plt.close(fig)
# # #     print(f"  -> Da luu: {output_path}")


# # # def main_sweep():
# # #     sensor_positions = load_sensor_positions(SENSOR_POSITIONS_PATH)
# # #     robot_positions, m_world = load_robot_pose(ROBOT_POSE_PATH)
# # #     voltage_data = load_voltage_data(VOLTAGE_DATA_PATH)

# # #     # QUAN TRONG: sensor_z_ref phai la 1 gia tri SCALAR (trung binh z cua
# # #     # tat ca sensor), giong het cach goc dung trong run_calibration():
# # #     #   sensor_z_initial_ref = sensor_positions[:, 2].mean()
# # #     # KHONG duoc truyen mang 64 phan tu vao day.
# # #     sensor_z_ref = sensor_positions[:, 2].mean()

# # #     print("=" * 60)
# # #     print(f"Diem khoi dau uoc luong: lambda_pos = {LAMBDA_POS_GUESS:.3e}, "
# # #           f"lambda_gain = {LAMBDA_GAIN_GUESS:.3e}")
# # #     print("=" * 60)

# # #     print("\n--- SWEEP lambda_pos (lambda_gain = 0) ---")
# # #     rmse_pos, dev_pos = run_sweep_1d(
# # #         "pos", lambda_pos_grid, sensor_positions, robot_positions,
# # #         m_world, voltage_data, sensor_z_ref
# # #     )
# # #     plot_lcurve(lambda_pos_grid, rmse_pos, dev_pos, "pos", LAMBDA_POS_GUESS)

# # #     print("\n--- SWEEP lambda_gain (lambda_pos = 0) ---")
# # #     rmse_gain, dev_gain = run_sweep_1d(
# # #         "gain", lambda_gain_grid, sensor_positions, robot_positions,
# # #         m_world, voltage_data, sensor_z_ref
# # #     )
# # #     plot_lcurve(lambda_gain_grid, rmse_gain, dev_gain, "gain", LAMBDA_GAIN_GUESS)

# # #     print("\nDa luu 2 anh L-curve vao thu muc BASE_DIR (xem duong dan o tren)")
# # #     print("Chon lambda tai 'diem khuyu' (elbow): noi RMSE bat dau tang nhanh")
# # #     print("nhung deviation da giam ve gan 0 -- do la vung can bang tot nhat.")


# # # if __name__ == "__main__":
# # #     main_sweep()



# "khóa x,y,z"
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from pathlib import Path
# from scipy.optimize import least_squares

# # =============================================================================
# # FILE PATHS
# # =============================================================================
# BASE_DIR = Path(r"/Users/tuananhnguyen/Downloads/Hallsensor_final/Data set 18.6")

# SENSOR_POSITIONS_PATH = BASE_DIR / "Hall_sensor_positions.csv"

# ROBOT_POSE_PATH = BASE_DIR / "grid_points_coordinates.csv"

# VOLTAGE_DATA_PATH = BASE_DIR / "grid_data.csv"

# OFFSET_FILE_PATH = BASE_DIR / "Offset_Sens.csv"

# # ---- outputs for the 2-stage calibration framework ----
# PHYSICAL_OUTPUT_PATH = BASE_DIR / "Calibration_Physical.csv"
# ALPHA_OUTPUT_PATH = BASE_DIR / "Calibration_Alpha.csv"


# # =============================================================================
# # CONSTANTS
# # =============================================================================

# MU0_OVER_4PI = 1e-7

# # ----  region boundaries  ----
# REGION1_H_MAX = 0.040   # Region 1: h < 0.040
# REGION2_H_MAX = 0.055   # Region 2: 0.040 <= h < 0.055 ; Region 3: h >= 0.055
# SAMPLES_PER_REGION = 60  # Stage 1: up to 60 points sampled per region

# # ----  Stage 1 regularization weights (physical priors)  ----
# # These penalize deviation from the design/nominal sensor pose so the
# # optimizer only moves a parameter away from its nominal value when the
# # voltage data actually demands it. Offset is intentionally NOT regularized.
# # NOTE: theta/phi (orientation) removed entirely -- sensor direction is
# # fixed to straight-up [0, 0, 1], so there is no orientation prior/lambda.
# LAMBDA_POS = 0    # position prior weight (x, y, z)   [1/m^2 scale]
# LAMBDA_GAIN = 0   # gain prior weight                  [1/(V/T)^2 scale]

# # =============================================================================
# # DIPOLE MODEL
# # =============================================================================

# def dipole_field(r_vec, m_vec):
#     """Calculate magnetic field from dipole model"""
#     r = np.linalg.norm(r_vec, axis=1, keepdims=True)
    
#     r3 = np.maximum(r**3, 1e-12)
#     r5 = np.maximum(r**5, 1e-12)
    
#     mdotr = np.sum(m_vec * r_vec, axis=1, keepdims=True)
    
#     B = MU0_OVER_4PI * (
#         3.0 * r_vec * mdotr / r5 - m_vec / r3
#     )
    
#     return B


# # =============================================================================
# # LOAD SENSOR POSITIONS
# # =============================================================================

# def load_sensor_positions(file_path):
#     """Load sensor positions from CSV"""
#     df = pd.read_csv(file_path)
#     sensor_positions = df.values
#     print(f"Loaded sensor positions: {sensor_positions.shape}")
#     return sensor_positions

# # =============================================================================
# # LOAD ROBOT POSE
# # =============================================================================

# def load_robot_pose(file_path):
#     """Load robot positions and magnetic orientations"""
#     df = pd.read_csv(file_path)
    
#     required_cols = ['x', 'y', 'z', 'mx', 'my', 'mz']
#     for c in required_cols:
#         if c not in df.columns:
#             raise ValueError(f"Missing column: {c}")

#     positions = df[['x', 'y', 'z']].values
#     m_world = df[['mx', 'my', 'mz']].values
    
#     # Normalize magnetic orientation
#     norm = np.linalg.norm(m_world, axis=1, keepdims=True)
#     m_world = m_world / norm
    
#     print(f"Loaded robot positions: {positions.shape}")
#     print(f"Loaded magnetic orientations: {m_world.shape}")
    
#     return positions, m_world


# # =============================================================================
# # LOAD VOLTAGE DATA
# # =============================================================================

# def load_voltage_data(file_path):
#     """Load voltage measurements"""
#     df = pd.read_csv(file_path)
#     voltage = df.values
#     print(f"Loaded voltage data: {voltage.shape}")
#     return voltage


# # =============================================================================
# # LOAD OFFSET FILE
# # =============================================================================

# def load_sensor_offsets(file_path):
#     """Load sensor offsets"""
#     df = pd.read_csv(file_path, header=0)
#     offsets = df.iloc[:, 1].values
#     print(f"Loaded offsets: {offsets.shape}")
#     return offsets


# # =============================================================================
# # RESIDUAL FUNCTION
# # =============================================================================

# def sensor_residuals(params, robot_positions, m_world, voltage_sensor,
#                       pos_prior=None, g0=7.5,
#                       lambda_pos=LAMBDA_POS, lambda_gain=LAMBDA_GAIN):
#     """
#     Calculate residuals between measured and predicted voltages, PLUS
#     regularization residuals that pull (x,y,z,gain) toward their
#     nominal/physical prior values. Offset 'a' is left unregularized.

#     params: [x, y, z, a, g]
#     Sensor direction is FIXED to straight up [0, 0, 1] -- theta/phi are no
#     longer free parameters (sensors are assumed soldered perpendicular to
#     the PCB with negligible tilt).

#     pos_prior: (x0, y0, z0) nominal sensor position from
#                Hall_sensor_positions.csv (design position). If None, no
#                position regularization terms are appended (kept for safety;
#                in normal use this is always provided by the caller).
#     """
#     x, y, z, a, g = params

#     # Fixed sensor direction (straight up, perpendicular to PCB)
#     sensor_dir = np.array([0.0, 0.0, 1.0])

#     # Sensor position
#     sensor_pos = np.array([x, y, z])

#     # Vector from robot to sensor
#     r_vec = sensor_pos - robot_positions

#     # Calculate magnetic field at sensor position
#     B = dipole_field(r_vec, m_world)

#     # Project magnetic field onto sensor direction
#     B_proj = B @ sensor_dir

#     # Predicted voltage
#     voltage_pred = a + g * B_proj

#     # ---- Voltage residual (unchanged) ----
#     r_voltage = voltage_sensor - voltage_pred

#     # ---- Regularization residuals (physical priors) ----
#     # NOTE: concatenated onto the residual vector, NOT added to the scalar
#     # cost -- appending sqrt(lambda) * (param - prior) as extra residual
#     # entries is exactly equivalent to adding lambda * (param - prior)^2
#     # to the cost, keeping us fully inside least_squares()/'trf'.
#     if pos_prior is not None:
#         x0, y0, z0 = pos_prior
#         r_pos = np.sqrt(lambda_pos) * np.array([x - x0, y - y0, z - z0])
#     else:
#         r_pos = np.array([])

#     r_gain = np.sqrt(lambda_gain) * np.array([g - g0])

#     residual = np.concatenate([r_voltage, r_pos, r_gain])

#     return residual


# def sensor_residuals_xyz_frozen(params, robot_positions, m_world, voltage_sensor,
#                                  pos_fixed, g0=7.5, lambda_gain=LAMBDA_GAIN):
#     """
#     Bien the CUA sensor_residuals VOI CA x,y,z DEU BI DONG BANG -- chi con
#     2 tham so tu do: offset (a) va gain (g). Dung de kiem tra dut diem xem
#     suy bien gain co bien mat hoan toan khi vi tri sensor bi khoa cung
#     hoan toan hay khong (thi nghiem quyet dinh sau khi z-only-frozen van
#     con corr(x,gain)/corr(y,gain) cao).

#     params: [a, g]
#     pos_fixed: (x0, y0, z0) vi tri sensor co dinh hoan toan (khong toi uu)
#     """
#     a, g = params

#     sensor_dir = np.array([0.0, 0.0, 1.0])
#     sensor_pos = np.array(pos_fixed)

#     r_vec = sensor_pos - robot_positions
#     B = dipole_field(r_vec, m_world)
#     B_proj = B @ sensor_dir

#     voltage_pred = a + g * B_proj
#     r_voltage = voltage_sensor - voltage_pred

#     r_gain = np.sqrt(lambda_gain) * np.array([g - g0])

#     residual = np.concatenate([r_voltage, r_gain])

#     return residual


# def sensor_residuals_z_frozen(params, robot_positions, m_world, voltage_sensor,
#                                z_fixed, pos_prior=None, g0=7.5,
#                                lambda_pos=LAMBDA_POS, lambda_gain=LAMBDA_GAIN):
#     """
#     Bien the CUA sensor_residuals VOI z BI DONG BANG (khong toi uu).
#     Dung de pha vo suy bien z-gain (corr(z,gain) ~ -0.9 da do duoc): z duoc
#     lay thang tu ban ve co khi (sensor_pos_init[2]), khong con la tham so
#     tu do de "danh doi" voi gain nua.

#     params: [x, y, a, g]   (chi 4 tham so, thieu z so voi ban goc)
#     z_fixed: gia tri z co dinh (thuong la sensor_pos_init[2])

#     pos_prior: (x0, y0) -- CHI con x,y (z da bi dong bang, khong can
#                regularize rieng vi no khong con la bien tu do).
#     """
#     x, y, a, g = params

#     sensor_dir = np.array([0.0, 0.0, 1.0])
#     sensor_pos = np.array([x, y, z_fixed])

#     r_vec = sensor_pos - robot_positions
#     B = dipole_field(r_vec, m_world)
#     B_proj = B @ sensor_dir

#     voltage_pred = a + g * B_proj
#     r_voltage = voltage_sensor - voltage_pred

#     if pos_prior is not None:
#         x0, y0 = pos_prior
#         r_pos = np.sqrt(lambda_pos) * np.array([x - x0, y - y0])
#     else:
#         r_pos = np.array([])

#     r_gain = np.sqrt(lambda_gain) * np.array([g - g0])

#     residual = np.concatenate([r_voltage, r_pos, r_gain])

#     return residual


# # =============================================================================
# # SINGLE SENSOR CALIBRATION
# # =============================================================================

# # =============================================================================
# # IDENTIFIABILITY / CORRELATION ANALYSIS (Stage 1 post-fit diagnostic)
# # =============================================================================
# # Muc dich: kiem tra xem gain co bi "nhap nhang" (correlated) voi cac tham so
# # khac (dac biet la z, vi dipole suy giam theo 1/r^3 nen z va gain co the
# # danh doi lan nhau) hay khong. Neu |corr| gan 1, gain KHONG duoc xac dinh
# # duy nhat tu du lieu voltage -- tuc gain fit duoc co the chi la 1 nghiem
# # trong vo so nghiem tuong duong, khong phai gia tri vat ly that cua sensor.
# #
# # Dung result.jac cua chinh least_squares() da tra ve -- KHONG chay lai toi
# # uu hoa, khong doi thuat toan/method='trf'.
# PARAM_NAMES = ["x", "y", "z", "offset", "gain"]
# CORRELATION_LOG = []  # tich luy 1 dict/sensor, dung de in bang tong ket cuoi


# def analyze_gain_identifiability(result, n_voltage, sensor_index, verbose=True):
#     """
#     Tinh ma tran covariance/correlation giua 5 tham so (x,y,z,offset,gain)
#     tu Jacobian tai nghiem hoi tu, CHI dung phan voltage residual (bo phan
#     regularization residual neu co, vi regularization lam gia tao giam
#     correlation -- xem giai thich o tren).

#     Tra ve dict {sensor_index, corr_x_gain, corr_y_gain, corr_z_gain,
#     corr_offset_gain, n_voltage, n_params, is_rank_deficient}.
#     """
#     n_params = len(result.x)

#     # Chi lay J ung voi voltage residual (n_voltage dong dau), bo residual
#     # regularization (pos/gain prior) neu dang bat regularize.
#     J_voltage = result.jac[:n_voltage, :]
#     residual_voltage = result.fun[:n_voltage]

#     dof = n_voltage - n_params
#     if dof <= 0:
#         row = {
#             "sensor_index": sensor_index,
#             "corr_x_gain": np.nan, "corr_y_gain": np.nan,
#             "corr_z_gain": np.nan, "corr_offset_gain": np.nan,
#             "n_voltage": n_voltage, "n_params": n_params,
#             "is_rank_deficient": True,
#         }
#         CORRELATION_LOG.append(row)
#         return row

#     sigma2 = np.sum(residual_voltage ** 2) / dof

#     JTJ = J_voltage.T @ J_voltage
#     try:
#         JTJ_inv = np.linalg.inv(JTJ)
#         is_rank_deficient = False
#     except np.linalg.LinAlgError:
#         # JTJ suy bien (singular) -- day la dau hieu MANH NHAT cua non-
#         # identifiability: mot so to hop tham so hoan toan khong the tach
#         # biet tu du lieu nay. Dung pseudo-inverse de van tinh duoc con so
#         # tham khao, nhung phai canh bao ro.
#         JTJ_inv = np.linalg.pinv(JTJ)
#         is_rank_deficient = True

#     cov = sigma2 * JTJ_inv
#     diag = np.diag(cov)

#     diag_safe = np.where(diag > 0, diag, np.nan)
#     denom = np.sqrt(np.outer(diag_safe, diag_safe))
#     corr_matrix = cov / denom

#     gain_idx = PARAM_NAMES.index("gain")
#     row = {
#         "sensor_index": sensor_index,
#         "corr_x_gain": corr_matrix[PARAM_NAMES.index("x"), gain_idx],
#         "corr_y_gain": corr_matrix[PARAM_NAMES.index("y"), gain_idx],
#         "corr_z_gain": corr_matrix[PARAM_NAMES.index("z"), gain_idx],
#         "corr_offset_gain": corr_matrix[PARAM_NAMES.index("offset"), gain_idx],
#         "n_voltage": n_voltage, "n_params": n_params,
#         "is_rank_deficient": is_rank_deficient,
#     }
#     CORRELATION_LOG.append(row)

#     if verbose:
#         flag = " [RANK-DEFICIENT!]" if is_rank_deficient else ""
#         print(f"    corr(z,gain)={row['corr_z_gain']:+.3f}  "
#               f"corr(offset,gain)={row['corr_offset_gain']:+.3f}  "
#               f"corr(x,gain)={row['corr_x_gain']:+.3f}  "
#               f"corr(y,gain)={row['corr_y_gain']:+.3f}{flag}")

#     return row


# def print_correlation_summary():
#     """In bang tong ket |corr(*, gain)| trung binh/max qua 64 sensor, va
#     liet ke cac sensor co correlation cao (|corr| > 0.8) -- nghi van manh
#     ve identifiability."""
#     if not CORRELATION_LOG:
#         print("Chua co du lieu correlation nao duoc ghi lai.")
#         return

#     z_gain = np.array([r["corr_z_gain"] for r in CORRELATION_LOG])
#     offset_gain = np.array([r["corr_offset_gain"] for r in CORRELATION_LOG])
#     x_gain = np.array([r["corr_x_gain"] for r in CORRELATION_LOG])
#     y_gain = np.array([r["corr_y_gain"] for r in CORRELATION_LOG])
#     n_deficient = sum(r["is_rank_deficient"] for r in CORRELATION_LOG)

#     print("\n" + "=" * 60)
#     print("IDENTIFIABILITY SUMMARY (correlation with gain, 64 sensors)")
#     print("=" * 60)
#     print(f"{'pair':<18}{'mean |corr|':>14}{'max |corr|':>14}")
#     print(f"{'z    - gain':<18}{np.nanmean(np.abs(z_gain)):>14.3f}"
#           f"{np.nanmax(np.abs(z_gain)):>14.3f}")
#     print(f"{'offset - gain':<18}{np.nanmean(np.abs(offset_gain)):>14.3f}"
#           f"{np.nanmax(np.abs(offset_gain)):>14.3f}")
#     print(f"{'x    - gain':<18}{np.nanmean(np.abs(x_gain)):>14.3f}"
#           f"{np.nanmax(np.abs(x_gain)):>14.3f}")
#     print(f"{'y    - gain':<18}{np.nanmean(np.abs(y_gain)):>14.3f}"
#           f"{np.nanmax(np.abs(y_gain)):>14.3f}")
#     print(f"\nSo sensor bi rank-deficient (JTJ singular): {n_deficient}/64")

#     HIGH_CORR_THRESHOLD = 0.8
#     print(f"\nSensor co |corr(z,gain)| > {HIGH_CORR_THRESHOLD}:")
#     flagged = [r for r in CORRELATION_LOG
#                if not np.isnan(r["corr_z_gain"])
#                and abs(r["corr_z_gain"]) > HIGH_CORR_THRESHOLD]
#     if flagged:
#         for r in flagged:
#             print(f"  Sensor {r['sensor_index']+1:02d}: "
#                   f"corr(z,gain) = {r['corr_z_gain']:+.3f}")
#     else:
#         print("  (khong co)")

#     print(f"\nSensor co |corr(offset,gain)| > {HIGH_CORR_THRESHOLD}:")
#     flagged = [r for r in CORRELATION_LOG
#                if not np.isnan(r["corr_offset_gain"])
#                and abs(r["corr_offset_gain"]) > HIGH_CORR_THRESHOLD]
#     if flagged:
#         for r in flagged:
#             print(f"  Sensor {r['sensor_index']+1:02d}: "
#                   f"corr(offset,gain) = {r['corr_offset_gain']:+.3f}")
#     else:
#         print("  (khong co)")
#     print("=" * 60)


# # ---- BIEN THE CHO TRUONG HOP Z BI DONG BANG (chi con 4 tham so: x,y,a,g) ----
# PARAM_NAMES_FROZEN = ["x", "y", "offset", "gain"]
# CORRELATION_LOG_FROZEN = []


# def analyze_gain_identifiability_frozen(result, n_voltage, sensor_index, verbose=True):
#     """
#     Ban sao cua analyze_gain_identifiability() nhung cho truong hop z bi
#     dong bang -- chi con 4 tham so (x,y,offset,gain), khong con cot z
#     trong Jacobian nen khong the/khong can tinh corr(z,gain) nua.
#     """
#     n_params = len(result.x)  # = 4
#     J_voltage = result.jac[:n_voltage, :]
#     residual_voltage = result.fun[:n_voltage]

#     dof = n_voltage - n_params
#     if dof <= 0:
#         row = {
#             "sensor_index": sensor_index,
#             "corr_x_gain": np.nan, "corr_y_gain": np.nan,
#             "corr_offset_gain": np.nan,
#             "n_voltage": n_voltage, "n_params": n_params,
#             "is_rank_deficient": True,
#         }
#         CORRELATION_LOG_FROZEN.append(row)
#         return row

#     sigma2 = np.sum(residual_voltage ** 2) / dof
#     JTJ = J_voltage.T @ J_voltage
#     try:
#         JTJ_inv = np.linalg.inv(JTJ)
#         is_rank_deficient = False
#     except np.linalg.LinAlgError:
#         JTJ_inv = np.linalg.pinv(JTJ)
#         is_rank_deficient = True

#     cov = sigma2 * JTJ_inv
#     diag = np.diag(cov)
#     diag_safe = np.where(diag > 0, diag, np.nan)
#     denom = np.sqrt(np.outer(diag_safe, diag_safe))
#     corr_matrix = cov / denom

#     gain_idx = PARAM_NAMES_FROZEN.index("gain")
#     row = {
#         "sensor_index": sensor_index,
#         "corr_x_gain": corr_matrix[PARAM_NAMES_FROZEN.index("x"), gain_idx],
#         "corr_y_gain": corr_matrix[PARAM_NAMES_FROZEN.index("y"), gain_idx],
#         "corr_offset_gain": corr_matrix[PARAM_NAMES_FROZEN.index("offset"), gain_idx],
#         "n_voltage": n_voltage, "n_params": n_params,
#         "is_rank_deficient": is_rank_deficient,
#     }
#     CORRELATION_LOG_FROZEN.append(row)

#     if verbose:
#         flag = " [RANK-DEFICIENT!]" if is_rank_deficient else ""
#         print(f"    [Z FROZEN] corr(offset,gain)={row['corr_offset_gain']:+.3f}  "
#               f"corr(x,gain)={row['corr_x_gain']:+.3f}  "
#               f"corr(y,gain)={row['corr_y_gain']:+.3f}{flag}")

#     return row


# def print_correlation_summary_frozen():
#     """Ban sao cua print_correlation_summary() cho truong hop z bi dong bang."""
#     if not CORRELATION_LOG_FROZEN:
#         print("Chua co du lieu correlation (z-frozen) nao duoc ghi lai.")
#         return

#     offset_gain = np.array([r["corr_offset_gain"] for r in CORRELATION_LOG_FROZEN])
#     x_gain = np.array([r["corr_x_gain"] for r in CORRELATION_LOG_FROZEN])
#     y_gain = np.array([r["corr_y_gain"] for r in CORRELATION_LOG_FROZEN])
#     n_deficient = sum(r["is_rank_deficient"] for r in CORRELATION_LOG_FROZEN)

#     print("\n" + "=" * 60)
#     print("IDENTIFIABILITY SUMMARY -- Z FROZEN (correlation with gain, 64 sensors)")
#     print("=" * 60)
#     print(f"{'pair':<18}{'mean |corr|':>14}{'max |corr|':>14}")
#     print(f"{'offset - gain':<18}{np.nanmean(np.abs(offset_gain)):>14.3f}"
#           f"{np.nanmax(np.abs(offset_gain)):>14.3f}")
#     print(f"{'x    - gain':<18}{np.nanmean(np.abs(x_gain)):>14.3f}"
#           f"{np.nanmax(np.abs(x_gain)):>14.3f}")
#     print(f"{'y    - gain':<18}{np.nanmean(np.abs(y_gain)):>14.3f}"
#           f"{np.nanmax(np.abs(y_gain)):>14.3f}")
#     print(f"\nSo sensor bi rank-deficient (JTJ singular): {n_deficient}/64")
#     print("=" * 60)


# # ---- BIEN THE CHO TRUONG HOP CA X,Y,Z DEU BI DONG BANG (chi con 2 tham so: a,g) ----
# PARAM_NAMES_XYZ_FROZEN = ["offset", "gain"]
# CORRELATION_LOG_XYZ_FROZEN = []


# def analyze_gain_identifiability_xyz_frozen(result, n_voltage, sensor_index, verbose=True):
#     """
#     Ban sao cua analyze_gain_identifiability() cho truong hop x,y,z DEU bi
#     dong bang -- chi con 2 tham so (offset, gain). Day la thi nghiem
#     "quyet dinh": neu corr(offset,gain) van cao o day, suy bien khong lien
#     quan gi den vi tri sensor ca -- ban than offset va gain von da nhap
#     nhang voi nhau trong chinh cong thuc V = a + g*Bz (voi Bz co the mang
#     ca gia tri gan 0, khien a va g kho tach biet).
#     """
#     n_params = len(result.x)  # = 2
#     J_voltage = result.jac[:n_voltage, :]
#     residual_voltage = result.fun[:n_voltage]

#     dof = n_voltage - n_params
#     if dof <= 0:
#         row = {
#             "sensor_index": sensor_index,
#             "corr_offset_gain": np.nan,
#             "n_voltage": n_voltage, "n_params": n_params,
#             "is_rank_deficient": True,
#         }
#         CORRELATION_LOG_XYZ_FROZEN.append(row)
#         return row

#     sigma2 = np.sum(residual_voltage ** 2) / dof
#     JTJ = J_voltage.T @ J_voltage
#     try:
#         JTJ_inv = np.linalg.inv(JTJ)
#         is_rank_deficient = False
#     except np.linalg.LinAlgError:
#         JTJ_inv = np.linalg.pinv(JTJ)
#         is_rank_deficient = True

#     cov = sigma2 * JTJ_inv
#     diag = np.diag(cov)
#     diag_safe = np.where(diag > 0, diag, np.nan)
#     denom = np.sqrt(np.outer(diag_safe, diag_safe))
#     corr_matrix = cov / denom

#     gain_idx = PARAM_NAMES_XYZ_FROZEN.index("gain")
#     offset_idx = PARAM_NAMES_XYZ_FROZEN.index("offset")
#     row = {
#         "sensor_index": sensor_index,
#         "corr_offset_gain": corr_matrix[offset_idx, gain_idx],
#         "n_voltage": n_voltage, "n_params": n_params,
#         "is_rank_deficient": is_rank_deficient,
#     }
#     CORRELATION_LOG_XYZ_FROZEN.append(row)

#     if verbose:
#         flag = " [RANK-DEFICIENT!]" if is_rank_deficient else ""
#         print(f"    [XYZ FROZEN] corr(offset,gain)={row['corr_offset_gain']:+.3f}{flag}")

#     return row


# def print_correlation_summary_xyz_frozen():
#     """Ban sao cua print_correlation_summary() cho truong hop x,y,z deu
#     bi dong bang -- chi con corr(offset,gain) de xem xet."""
#     if not CORRELATION_LOG_XYZ_FROZEN:
#         print("Chua co du lieu correlation (xyz-frozen) nao duoc ghi lai.")
#         return

#     offset_gain = np.array([r["corr_offset_gain"] for r in CORRELATION_LOG_XYZ_FROZEN])
#     n_deficient = sum(r["is_rank_deficient"] for r in CORRELATION_LOG_XYZ_FROZEN)

#     print("\n" + "=" * 60)
#     print("IDENTIFIABILITY SUMMARY -- X,Y,Z FROZEN (correlation with gain, 64 sensors)")
#     print("=" * 60)
#     print(f"{'pair':<18}{'mean |corr|':>14}{'max |corr|':>14}")
#     print(f"{'offset - gain':<18}{np.nanmean(np.abs(offset_gain)):>14.3f}"
#           f"{np.nanmax(np.abs(offset_gain)):>14.3f}")
#     print(f"\nSo sensor bi rank-deficient (JTJ singular): {n_deficient}/64")
#     print("=" * 60)

# def calibrate_single_sensor(
#         sensor_index,
#         sensor_pos_init,
#         robot_positions,
#         m_world,
#         voltage_sensor,
#         offset_init=1.618):
#     """
#     Calibrate a single sensor -- CHI con offset + gain la tham so tu do.
#     Sensor direction is fixed straight up (perpendicular to PCB);
#     theta/phi are no longer free parameters.

#     CA X,Y,Z DEU BI DONG BANG (khong toi uu) -- lay thang tu
#     sensor_pos_init (ban ve co khi). Day la thi nghiem quyet dinh: sau khi
#     z-only-frozen van con corr(x,gain)/corr(y,gain) cao (~0.4 mean, ~0.95
#     max), ta kiem tra xem khoa CA vi tri sensor co xoa duoc suy bien gain
#     hay khong, hay ban chat van la V=a+g*Bz von da nhap nhang offset-gain.
#     """
#     # ----------------------------------------------------
#     # INITIAL VALUES
#     # ----------------------------------------------------
#     g0 = 7.5  # Initial gain (V/T)
#     offset_init = 1.618         # a (offset)
#     pos_fixed = tuple(sensor_pos_init)  # X,Y,Z DONG BANG HOAN TOAN

#     # ----------------------------------------------------
#     # INITIAL PARAMETERS (chi con 2: offset, gain)
#     # ----------------------------------------------------
#     x0 = np.array([
#         offset_init,  # a (offset)
#         g0            # g (gain)
#     ])
    
#     # ----------------------------------------------------
#     # BOUNDS
#     # ----------------------------------------------------
#     lower = [
#         offset_init - 0.02,   # a min (20mV)
#         6.9                     # g min
#     ]
    
#     upper = [
#         offset_init + 0.02,   # a max (20mV)
#         8                      # g max
#     ]
    
#     # ----------------------------------------------------
#     # OPTIMIZATION
#     # ----------------------------------------------------
#     result = least_squares(
#         sensor_residuals_xyz_frozen,
#         x0,
#         bounds=(lower, upper),
#         args=(
#             robot_positions,
#             m_world,
#             voltage_sensor,
#             pos_fixed
#         ),
#         kwargs=dict(
#             g0=g0,
#         ),
#         method='trf',
#         max_nfev=250           
#     )
    
#     # ----------------------------------------------------
#     # EXTRACT OPTIMIZED PARAMETERS
#     # ----------------------------------------------------
#     params_opt = result.x  # [a, g]
    
#     # Direction is fixed straight up -- theta/phi kept at 0 rad so the
#     # OUTPUT FORMAT (10 columns) stays identical to the previous script
#     # for downstream compatibility (save_results, save_physical_results,
#     # plot_direction_vectors, Stage 2 alpha).
#     theta_opt, phi_opt = 0.0, 0.0
#     nx, ny, nz = 0.0, 0.0, 1.0
    
#     # Create extended parameter array for saving (x,y,z lay tu pos_fixed,
#     # khong phai tu params_opt vi ca 3 deu khong con trong danh sach
#     # tham so toi uu)
#     params_extended = np.array([
#         pos_fixed[0],   # x (DONG BANG, khong doi)
#         pos_fixed[1],   # y (DONG BANG, khong doi)
#         pos_fixed[2],   # z (DONG BANG, khong doi)
#         params_opt[0],  # a
#         params_opt[1],  # g
#         nx, ny, nz,     # direction vector components (fixed straight up)
#         theta_opt,      # theta (radians, fixed = 0)
#         phi_opt         # phi (radians, fixed = 0)
#     ])
    
#     # ----------------------------------------------------
#     # CALCULATE RMSE
#     # (result.fun = [voltage residuals, gain reg];
#     #  RMSE must be computed from the voltage part only, so it still
#     #  means "V measured vs V predicted" and stays comparable to before)
#     # ----------------------------------------------------
#     n_voltage = voltage_sensor.shape[0]
#     rmse = np.sqrt(np.mean(result.fun[:n_voltage]**2))

#     # ---- IDENTIFIABILITY CHECK (bien the xyz-frozen, chi con 2 tham so) ----
#     analyze_gain_identifiability_xyz_frozen(result, n_voltage, sensor_index)

#     angle_from_vertical = np.rad2deg(abs(theta_opt))
#     print(f"Sensor {sensor_index+1:02d} | RMSE = {rmse:.6f} | "
#           f"Angle from Z = {angle_from_vertical:.2f}° | "
#           f"Dir = [{nx:.3f}, {ny:.3f}, {nz:.3f}]")
    
#     return params_extended, rmse


# # =============================================================================
# # FULL CALIBRATION
# # =============================================================================

# def run_calibration(
#         sensor_positions,
#         offsets,
#         robot_positions,
#         m_world,
#         voltage_data):
#     """
#     Calibrate all sensors
#     """
#     n_sensors = sensor_positions.shape[0]
#     results = []
#     rmses = []
    
#     for i in range(n_sensors):
#         params, rmse = calibrate_single_sensor(
#             sensor_index=i,
#             sensor_pos_init=sensor_positions[i],
#             robot_positions=robot_positions,
#             m_world=m_world,
#             voltage_sensor=voltage_data[:, i]
#         )
        
#         results.append(params)
#         rmses.append(rmse)
    
#     # ---- In bang tong ket identifiability sau khi calib het 64 sensor ----
#     # (dung ban X,Y,Z-FROZEN vi calibrate_single_sensor gio dong bang ca 3 truc)
#     print_correlation_summary_xyz_frozen()
    
#     return np.array(results), np.array(rmses)


# # =============================================================================
# # REGION SELECTION
# # =============================================================================

# def select_region_samples(
#         robot_positions,
#         m_world,
#         voltage_data,
#         sensor_z,
#         h_min,
#         h_max=None,
#         max_samples=180,
#         return_indices=False):

#     h = robot_positions[:, 2] - sensor_z

#     lower_ok = np.ones_like(h, dtype=bool) if h_min is None else (h >= h_min)
#     upper_ok = np.ones_like(h, dtype=bool) if h_max is None else (h < h_max)
#     region_idx = np.where(lower_ok & upper_ok)[0]

#     print(f"\nRegion [{h_min}, {h_max}] contains {len(region_idx)} samples")

#     if len(region_idx) > max_samples:
#         # NOTE: no random seed is set here, per the calibration spec.
#         region_idx = np.random.choice(
#             region_idx,
#             size=max_samples,
#             replace=False
#         )

#     print(f"Using {len(region_idx)} samples")

#     if return_indices:
#         return (
#             robot_positions[region_idx],
#             m_world[region_idx],
#             voltage_data[region_idx],
#             region_idx
#         )

#     return (
#         robot_positions[region_idx],
#         m_world[region_idx],
#         voltage_data[region_idx]
#     )


# # =============================================================================
# # NEW: STAGE 1 - STRATIFIED CALIBRATION SET (fixed, reused in Stage 2)
# # =============================================================================

# def select_stage1_calibration_set(
#         robot_positions,
#         m_world,
#         voltage_data,
#         sensor_z_ref,
#         per_region=SAMPLES_PER_REGION):

#     rp1, mw1, vd1, idx1 = select_region_samples(
#         robot_positions, m_world, voltage_data, sensor_z_ref,
#         h_min=None, h_max=REGION1_H_MAX,
#         max_samples=per_region, return_indices=True
#     )
#     rp2, mw2, vd2, idx2 = select_region_samples(
#         robot_positions, m_world, voltage_data, sensor_z_ref,
#         h_min=REGION1_H_MAX, h_max=REGION2_H_MAX,
#         max_samples=per_region, return_indices=True
#     )
#     rp3, mw3, vd3, idx3 = select_region_samples(
#         robot_positions, m_world, voltage_data, sensor_z_ref,
#         h_min=REGION2_H_MAX, h_max=None,
#         max_samples=per_region, return_indices=True
#     )

#     calib_indices = np.concatenate([idx1, idx2, idx3])
#     rp_calib = np.concatenate([rp1, rp2, rp3], axis=0)
#     mw_calib = np.concatenate([mw1, mw2, mw3], axis=0)
#     vd_calib = np.concatenate([vd1, vd2, vd3], axis=0)

#     print(f"\n[Stage 1] Combined calibration set: {len(calib_indices)} "
#           f"points (Region1={len(idx1)}, Region2={len(idx2)}, "
#           f"Region3={len(idx3)})")

#     return calib_indices, rp_calib, mw_calib, vd_calib


# # =============================================================================
# # NEW: STAGE 2 - CLOSED-FORM ALPHA PER REGION
# # =============================================================================

# def calibrate_alpha_by_region(physical_results, rp_calib, mw_calib, vd_calib):

#     n_samples = rp_calib.shape[0]
#     n_sensors = physical_results.shape[0]

#     sensor_pos = physical_results[:, 0:3]      # (n_sensors, 3): x, y, z
#     a = physical_results[:, 3]                  # (n_sensors,)  offset
#     g = physical_results[:, 4]                  # (n_sensors,)  gain
#     sensor_dir = physical_results[:, 5:8]       # (n_sensors, 3): nx, ny, nz

#     # h[i, s] = z_capsule_i - z_sensor_calibrated_s  (uses Stage-1 result)
#     z_calibrated = sensor_pos[:, 2]
#     h = rp_calib[:, 2][:, None] - z_calibrated[None, :]   # (n_samples, n_sensors)

#     # Raw dipole projection B_proj[i, s] using the FROZEN calibrated pose
#     # (position + direction) of each sensor -- gain/offset are NOT
#     # reapplied here, they're multiplied in separately below.
#     B_proj = np.zeros((n_samples, n_sensors))
#     for s in range(n_sensors):
#         r_vec = sensor_pos[s] - rp_calib               # (n_samples, 3)
#         B = dipole_field(r_vec, mw_calib)               # (n_samples, 3)
#         B_proj[:, s] = B @ sensor_dir[s]

#     gB = g[None, :] * B_proj                            # (n_samples, n_sensors)
#     v_minus_a = vd_calib - a[None, :]                    # V_measured - a

#     region_masks = {
#         1: h < REGION1_H_MAX,
#         2: (h >= REGION1_H_MAX) & (h < REGION2_H_MAX),
#         3: h >= REGION2_H_MAX
#     }

#     alphas = {}
#     for region, mask in region_masks.items():
#         numerator = np.sum(gB[mask] * v_minus_a[mask])
#         denominator = np.sum(gB[mask] ** 2)
#         alpha = numerator / denominator if denominator > 0 else np.nan
#         print(f"Region {region}: alpha = {alpha:.6f} "
#               f"(from {np.sum(mask)} sample-sensor pairs)")
#         alphas[region] = alpha

#     return alphas


# # =============================================================================
# # SAVE RESULTS
# # =============================================================================

# def save_results(results, rmses, output_file):
#     """
#     Save calibration results to CSV
#     """
#     df = pd.DataFrame({
#         "sensor_id": np.arange(len(results)),
#         "x": results[:, 0],
#         "y": results[:, 1],
#         "z": results[:, 2],
#         "offset_a": results[:, 3],
#         "gain_g": results[:, 4],
#         "nx": results[:, 5],
#         "ny": results[:, 6],
#         "nz": results[:, 7],
#         "theta_rad": results[:, 8],
#         "phi_rad": results[:, 9],
#         "angle_from_z_deg": np.rad2deg(np.abs(results[:, 8])),
#         "rmse": rmses
#     })
    
#     df.to_csv(output_file, index=False)
#     print(f"\nSaved: {output_file}")


# # =============================================================================
# # NEW: SAVE STAGE 1 / STAGE 2 RESULTS (required output files)
# # =============================================================================

# def save_physical_results(results, output_file):
#     """
#     Save Stage 1 frozen physical parameters to Calibration_Physical.csv
#     with columns: sensor_index, x, y, z, offset, gain, theta, phi
#     """
#     df = pd.DataFrame({
#         "sensor_index": np.arange(len(results)),
#         "x": results[:, 0],
#         "y": results[:, 1],
#         "z": results[:, 2],
#         "offset": results[:, 3],
#         "gain": results[:, 4],
#         "theta": results[:, 8],
#         "phi": results[:, 9],
#     })
#     df.to_csv(output_file, index=False)
#     print(f"\nSaved: {output_file}")


# def save_alpha_results(alphas, output_file):
#     """
#     Save Stage 2 region correction coefficients to Calibration_Alpha.csv
#     with exactly 3 rows: Region 1, Region 2, Region 3.
#     """
#     regions_sorted = sorted(alphas.keys())
#     df = pd.DataFrame({
#         "Region": [f"Region {r}" for r in regions_sorted],
#         "Alpha": [alphas[r] for r in regions_sorted],
#     })
#     df.to_csv(output_file, index=False)
#     print(f"Saved: {output_file}")


# # =============================================================================
# # PLOT RMSE
# # =============================================================================

# def plot_rmse(rmses):
#     """Plot RMSE for all sensors"""
#     plt.figure(figsize=(10, 5))
#     plt.bar(np.arange(len(rmses)), rmses)
#     plt.xlabel("Sensor Index")
#     plt.ylabel("RMSE")
#     plt.title("Calibration RMSE")
#     plt.grid(True)
#     plt.show()


# # =============================================================================
# # PLOT DIRECTION VECTORS
# # =============================================================================

# # def plot_direction_vectors(results):
# #     """Plot direction vectors of all sensors"""
# #     fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
# #     sensor_ids = np.arange(len(results))
    
# #     # Subplot 1: Direction components
# #     ax1 = axes[0]
# #     ax1.plot(sensor_ids, results[:, 5], 'r.-', label='nx', markersize=8)
# #     ax1.plot(sensor_ids, results[:, 6], 'g.-', label='ny', markersize=8)
# #     ax1.plot(sensor_ids, results[:, 7], 'b.-', label='nz', markersize=8)
# #     ax1.set_xlabel("Sensor Index")
# #     ax1.set_ylabel("Direction Component")
# #     ax1.set_title("Sensor Direction Components")
# #     ax1.legend()
# #     ax1.grid(True)
    
# #     # Subplot 2: Angular deviation from vertical
# #     ax2 = axes[1]
# #     angle_from_vertical = np.rad2deg(np.abs(results[:, 8]))
# #     ax2.bar(sensor_ids, angle_from_vertical)
# #     ax2.set_xlabel("Sensor Index")
# #     ax2.set_ylabel("Angle (degrees)")
# #     ax2.set_title("Angular Deviation from Z-axis")
# #     ax2.grid(True)
    
# #     # Subplot 3: Azimuthal angle
# #     ax3 = axes[2]
# #     phi_deg = np.rad2deg(results[:, 9])
# #     ax3.bar(sensor_ids, phi_deg)
# #     ax3.set_xlabel("Sensor Index")
# #     ax3.set_ylabel("Phi (degrees)")
# #     ax3.set_title("Azimuthal Angle (XY-plane)")
# #     ax3.grid(True)
    
# #     plt.tight_layout()
# #     plt.show()


# # =============================================================================
# # MAIN
# # =============================================================================

# def main():
#     """Main execution function -- 2-stage calibration framework"""

#     # Load data
#     sensor_positions = load_sensor_positions(SENSOR_POSITIONS_PATH)
#     robot_positions, m_world = load_robot_pose(ROBOT_POSE_PATH)
#     voltage_data = load_voltage_data(VOLTAGE_DATA_PATH)
#     offsets = load_sensor_offsets(OFFSET_FILE_PATH)

#     # Ensure consistent number of samples
#     n_samples = min(len(robot_positions), len(voltage_data))
#     robot_positions = robot_positions[:n_samples]
#     m_world = m_world[:n_samples]
#     voltage_data = voltage_data[:n_samples]

#     # ------------------------------------------------------------------
#     # NEW: Sensor reference height for Stage 1 region assignment.
#     # Must come from the ORIGINAL (uncalibrated) sensor positions file.
#     # ------------------------------------------------------------------
#     sensor_z_initial_ref = sensor_positions[:, 2].mean()
#     print(f"\nSensor reference z (initial, for Stage 1) = "
#           f"{sensor_z_initial_ref:.6f} m")

#     # ==================================================================
#     # STAGE 1a: Stratified, fixed calibration set (<=150 points)
#     # ==================================================================
#     print("\n===================================")
#     print("STAGE 1: SAMPLE SELECTION")
#     print("===================================")
#     calib_indices, rp_calib, mw_calib, vd_calib = select_stage1_calibration_set(
#         robot_positions, m_world, voltage_data, sensor_z_initial_ref,
#         per_region=SAMPLES_PER_REGION
#     )
#     # calib_indices is frozen here and reused as-is in Stage 2 below --
#     # no re-sampling happens after this point.

#     # ==================================================================
#     # STAGE 1b: Per-sensor physical parameter calibration
#     # (unchanged residual function, dipole model, initial values, bounds)
#     # ==================================================================
#     print("\n===================================")
#     print("STAGE 1: PHYSICAL PARAMETER FIT")
#     print("===================================")
#     results, rmses = run_calibration(
#         sensor_positions, offsets, rp_calib, mw_calib, vd_calib
#     )

#     print("\n========================")
#     print(f"Mean RMSE = {np.mean(rmses):.6f}")
#     print(f"Max RMSE  = {np.max(rmses):.6f}")
#     print(f"Min RMSE  = {np.min(rmses):.6f}")
#     print("========================")

#     # Save Stage 1 physical parameters (frozen going into Stage 2)
#     save_physical_results(results, PHYSICAL_OUTPUT_PATH)

#     # Plots
#     plot_rmse(rmses)
#     # plot_direction_vectors(results)

#     # ==================================================================
#     # STAGE 2: Closed-form alpha per region (physical params frozen)
#     # ==================================================================
#     print("\n===================================")
#     print("STAGE 2: ALPHA CORRECTION (closed-form)")
#     print("===================================")
#     alphas = calibrate_alpha_by_region(results, rp_calib, mw_calib, vd_calib)
#     save_alpha_results(alphas, ALPHA_OUTPUT_PATH)

#     print("\n===================================")
#     print("ALL STAGES FINISHED")
#     print("===================================")


# if __name__ == "__main__":
#     main()
    
# # """
# # Sweep lambda_pos va lambda_gain quanh vung uoc luong co co so, ve L-curve
# # (RMSE vs do lech tham so khoi prior) de chon lambda can bang.

# # CACH DUNG: dan doan code nay vao CUOI file calib_region_based.py (ban da
# # co san cac ham dipole_field, sensor_residuals, calibrate_single_sensor,
# # select_stage1_calibration_set, load_sensor_positions, load_robot_pose,
# # load_voltage_data, v.v. trong file goc). Script nay KHONG doi thuat toan
# # least_squares/'trf', chi goi lai calibrate_single_sensor() nhieu lan voi
# # cac gia tri lambda khac nhau.

# # Diem khoi dau duoc uoc luong tu cong thuc:
# #     lambda = (RMSE_v_no_reg / sigma)^2
# # voi RMSE_v_no_reg lay tu log ban da chay (0.000383 V), va sigma la do lech
# # "chap nhan duoc" ban tu chon (o day: sigma_pos=0.5mm, sigma_gain=1.0 V/T).
# # Day chi la DIEM KHOI DAU -- sweep xung quanh no de tim diem can bang that su.
# # """

# # import numpy as np
# # import matplotlib.pyplot as plt

# # # =============================================================================
# # # CAU HINH SWEEP
# # # =============================================================================
# # RMSE_V_NO_REG = 0.000383  # tu log khong-regularize cua ban

# # # Diem khoi dau uoc luong (KHONG phai gia tri cuoi cung)
# # LAMBDA_POS_GUESS = (RMSE_V_NO_REG / 0.0005) ** 2   # sigma_pos = 0.5 mm
# # LAMBDA_GAIN_GUESS = (RMSE_V_NO_REG / 1.0) ** 2      # sigma_gain = 1.0 V/T

# # # Sweep tu 1/100x den 100x quanh diem khoi dau, log-spaced, 9 diem
# # N_POINTS = 9
# # lambda_pos_grid = np.geomspace(LAMBDA_POS_GUESS / 100, LAMBDA_POS_GUESS * 100, N_POINTS)
# # lambda_gain_grid = np.geomspace(LAMBDA_GAIN_GUESS / 100, LAMBDA_GAIN_GUESS * 100, N_POINTS)

# # # Sensor dai dien de sweep nhanh (thay vi chay het 64 sensor cho moi lambda).
# # # Nen chon vai sensor o cac vi tri khac nhau (giua/canh mang) de dai dien.
# # SENSOR_SAMPLE_IDX = [0, 15, 31, 47, 63]  # 5 sensor rai deu tren 64 sensor


# # def run_sweep_1d(param_name, lambda_grid, sensor_positions, robot_positions,
# #                   m_world, voltage_data, sensor_z_ref):
# #     """
# #     Sweep 1 chieu: giu lambda con lai = 0, chi thay doi lambda cua param_name
# #     ('pos' hoac 'gain'). Tra ve (rmse_list, deviation_list).

# #     Goi least_squares() TRUC TIEP (khong qua calibrate_single_sensor) vi
# #     lambda_pos/lambda_gain trong sensor_residuals la default-argument, bi
# #     "dong bang" luc dinh nghia ham -- doi bien global sau do KHONG co tac
# #     dung. Truyen tuong minh qua kwargs moi dam bao dung gia tri lambda.
# #     """
# #     rmse_list = []
# #     deviation_list = []  # do lech tuong doi TB khoi prior (chuan hoa theo bound)

# #     calib_indices, rp_calib, mw_calib, vd_calib = select_stage1_calibration_set(
# #         robot_positions, m_world, voltage_data, sensor_z_ref
# #     )

# #     g0 = 7.5
# #     pos_tol = 0.0015

# #     for lam in lambda_grid:
# #         lambda_pos = lam if param_name == "pos" else 0.0
# #         lambda_gain = lam if param_name == "gain" else 0.0

# #         rmses_this_lambda = []
# #         deviations_this_lambda = []

# #         for s in SENSOR_SAMPLE_IDX:
# #             sensor_pos_init = sensor_positions[s]

# #             x0 = np.array([
# #                 sensor_pos_init[0], sensor_pos_init[1], sensor_pos_init[2],
# #                 1.618,  # offset_init
# #                 g0
# #             ])
# #             lower = [
# #                 sensor_pos_init[0] - pos_tol, sensor_pos_init[1] - pos_tol,
# #                 sensor_pos_init[2] - pos_tol, 1.618 - 0.015, 4
# #             ]
# #             upper = [
# #                 sensor_pos_init[0] + pos_tol, sensor_pos_init[1] + pos_tol,
# #                 sensor_pos_init[2] + pos_tol, 1.618 + 0.015, 9
# #             ]

# #             result = least_squares(
# #                 sensor_residuals,
# #                 x0,
# #                 bounds=(lower, upper),
# #                 args=(rp_calib, mw_calib, vd_calib[:, s]),
# #                 kwargs=dict(
# #                     pos_prior=tuple(sensor_pos_init),
# #                     g0=g0,
# #                     lambda_pos=lambda_pos,
# #                     lambda_gain=lambda_gain,
# #                 ),
# #                 method="trf",
# #                 max_nfev=250
# #             )

# #             n_voltage = vd_calib.shape[0]
# #             rmse = np.sqrt(np.mean(result.fun[:n_voltage] ** 2))
# #             rmses_this_lambda.append(rmse)

# #             if param_name == "pos":
# #                 dev = np.linalg.norm(result.x[0:3] - sensor_pos_init) / pos_tol
# #             else:  # gain
# #                 dev = abs(result.x[4] - g0) / 5.0  # chuan hoa theo bien do bound gain [4,9]

# #             deviations_this_lambda.append(dev)

# #         rmse_list.append(np.mean(rmses_this_lambda))
# #         deviation_list.append(np.mean(deviations_this_lambda))

# #         print(f"  lambda_{param_name} = {lam:.3e} | "
# #               f"mean RMSE = {np.mean(rmses_this_lambda):.6f} | "
# #               f"mean |dev|/bound = {np.mean(deviations_this_lambda):.4f}")

# #     return rmse_list, deviation_list


# # def plot_lcurve(lambda_grid, rmse_list, deviation_list, param_name, guess_value):
# #     fig, ax1 = plt.subplots(figsize=(7, 5))

# #     color1 = "tab:blue"
# #     ax1.set_xlabel(f"lambda_{param_name}")
# #     ax1.set_ylabel("Mean RMSE (V)", color=color1)
# #     ax1.plot(lambda_grid, rmse_list, "o-", color=color1, label="RMSE")
# #     ax1.set_xscale("log")
# #     ax1.tick_params(axis="y", labelcolor=color1)

# #     ax2 = ax1.twinx()
# #     color2 = "tab:red"
# #     ax2.set_ylabel("Mean |deviation| / bound", color=color2)
# #     ax2.plot(lambda_grid, deviation_list, "s--", color=color2, label="Deviation")
# #     ax2.tick_params(axis="y", labelcolor=color2)

# #     ax1.axvline(guess_value, color="gray", linestyle=":", alpha=0.7,
# #                 label=f"initial guess = {guess_value:.2e}")

# #     fig.suptitle(f"L-curve: RMSE vs Deviation ({param_name})")
# #     fig.tight_layout()
# #     output_path = BASE_DIR / f"lcurve_{param_name}.png"
# #     fig.savefig(output_path, dpi=120)
# #     plt.close(fig)
# #     print(f"  -> Da luu: {output_path}")


# # def main_sweep():
# #     sensor_positions = load_sensor_positions(SENSOR_POSITIONS_PATH)
# #     robot_positions, m_world = load_robot_pose(ROBOT_POSE_PATH)
# #     voltage_data = load_voltage_data(VOLTAGE_DATA_PATH)

# #     # QUAN TRONG: sensor_z_ref phai la 1 gia tri SCALAR (trung binh z cua
# #     # tat ca sensor), giong het cach goc dung trong run_calibration():
# #     #   sensor_z_initial_ref = sensor_positions[:, 2].mean()
# #     # KHONG duoc truyen mang 64 phan tu vao day.
# #     sensor_z_ref = sensor_positions[:, 2].mean()

# #     print("=" * 60)
# #     print(f"Diem khoi dau uoc luong: lambda_pos = {LAMBDA_POS_GUESS:.3e}, "
# #           f"lambda_gain = {LAMBDA_GAIN_GUESS:.3e}")
# #     print("=" * 60)

# #     print("\n--- SWEEP lambda_pos (lambda_gain = 0) ---")
# #     rmse_pos, dev_pos = run_sweep_1d(
# #         "pos", lambda_pos_grid, sensor_positions, robot_positions,
# #         m_world, voltage_data, sensor_z_ref
# #     )
# #     plot_lcurve(lambda_pos_grid, rmse_pos, dev_pos, "pos", LAMBDA_POS_GUESS)

# #     print("\n--- SWEEP lambda_gain (lambda_pos = 0) ---")
# #     rmse_gain, dev_gain = run_sweep_1d(
# #         "gain", lambda_gain_grid, sensor_positions, robot_positions,
# #         m_world, voltage_data, sensor_z_ref
# #     )
# #     plot_lcurve(lambda_gain_grid, rmse_gain, dev_gain, "gain", LAMBDA_GAIN_GUESS)

# #     print("\nDa luu 2 anh L-curve vao thu muc BASE_DIR (xem duong dan o tren)")
# #     print("Chon lambda tai 'diem khuyu' (elbow): noi RMSE bat dau tang nhanh")
# #     print("nhung deviation da giam ve gan 0 -- do la vung can bang tot nhat.")


# # if __name__ == "__main__":
# #     main_sweep()