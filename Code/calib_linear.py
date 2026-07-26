import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import least_squares

# =============================================================================
# FILE PATHS
# =============================================================================
BASE_DIR = Path(r"/Users/tuananhnguyen/Downloads/Hallsensor_final/Data set 18.6") #MAC

SENSOR_POSITIONS_PATH = BASE_DIR / "Hall_sensor_positions.csv"   

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

# ----  NEW: random-sampling split (replaces the old 3-region stratification) ----
# 400 points are drawn uniformly at random from the WHOLE input dataset.
# 240 of them go to Stage 1 (physical parameter fit), the remaining 160 go
# to Stage 2 (alpha(h) fit). No height-based stratification is done anymore
# -- both stages just see a random cross-section of the working volume.
N_TOTAL_CALIB_SAMPLES = 400
N_STAGE1_SAMPLES = 240
N_STAGE2_SAMPLES = N_TOTAL_CALIB_SAMPLES - N_STAGE1_SAMPLES   

# ----  Stage 1 regularization weights (physical priors)  ----
# These penalize deviation from the design/nominal sensor pose so the
# optimizer only moves a parameter away from its nominal value when the
# voltage data actually demands it. Offset is intentionally NOT regularized.
# NOTE: theta/phi (orientation) removed entirely -- sensor direction is
# fixed to straight-up [0, 0, 1], so there is no orientation prior/lambda.
LAMBDA_POS = 2000    # position prior weight (x, y, z)   [1/m^2 scale]
LAMBDA_GAIN = 2e-3   # gain prior weight                  [1/(V/T)^2 scale]

# ----  NEW: Stage 2 alpha(h) = c0 + c1*h regularization (ridge priors)  ----
# alpha(h) replaces the old "one constant alpha per region" correction with
# a single GLOBAL linear function of height h, shared across all 64 sensors
# (pooled fit, same pooling philosophy as the old per-region closed form).
# The ridge priors below pull (c0, c1) toward (1, 0) -- i.e. "no correction,
# no height dependence" -- so c1 can only pick up a nonzero slope when the
# Stage-2 voltage residuals actually demand it, instead of silently
# absorbing leftover gain/offset error from Stage 1.
#
# IMPORTANT: these two lambdas are NOT calibrated for your actual data yet.
# Same as LAMBDA_POS/LAMBDA_GAIN above, you should sweep them (see the
# L-curve sweep pattern later in the old script) and pick the elbow: small
# enough that RMSE isn't hurt, large enough that c1 stays physically
# plausible (i.e. doesn't swing wildly if you re-run with a different
# random 400-point draw).
ALPHA_C0_PRIOR = 1.0     # prior: no multiplicative correction
ALPHA_C1_PRIOR = 0.0     # prior: no height-dependence
LAMBDA_ALPHA_C0 = 1e-2   # ridge weight on c0 -> 1          [dimensionless]
LAMBDA_ALPHA_C1 = 1e2    # ridge weight on c1 -> 0          [1/m^2 scale]

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
# RESIDUAL FUNCTION  (unchanged from the region-based script)
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
# SINGLE SENSOR CALIBRATION  (unchanged from the region-based script)
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
    pos_tol = 0.0013  # 1.5mm tolerance for sensor position
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
    # Stage 2 alpha).
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
# FULL CALIBRATION  (unchanged)
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
# NEW: RANDOM 400-POINT SAMPLING -> STAGE 1 (240) / STAGE 2 (160) SPLIT
# =============================================================================
# Replaces select_region_samples() + select_stage1_calibration_set() from
# the region-based script. No height stratification anymore -- just draw
# N_TOTAL_CALIB_SAMPLES uniformly at random from the whole dataset, then
# split into a Stage 1 chunk and a Stage 2 chunk.
# =============================================================================

def select_stage1_stage2_split(
        robot_positions,
        m_world,
        voltage_data,
        n_total=N_TOTAL_CALIB_SAMPLES,
        n_stage1=N_STAGE1_SAMPLES):
    """
    Draw n_total points at random (no replacement) from the full dataset,
    then split them into a Stage 1 subset (n_stage1 points, used to fit
    the physical parameters x,y,z,a,g) and a Stage 2 subset (the remaining
    n_total - n_stage1 points, used to fit alpha(h) with the physical
    parameters frozen).

    NOTE: no random seed is set here, per the calibration spec (matches
    the "no seed" behaviour of the old select_region_samples).
    """
    n_samples = robot_positions.shape[0]
    if n_total > n_samples:
        raise ValueError(
            f"Requested {n_total} calibration points but dataset only has "
            f"{n_samples} samples."
        )

    all_idx = np.random.choice(n_samples, size=n_total, replace=False)
    stage1_idx = all_idx[:n_stage1]
    stage2_idx = all_idx[n_stage1:]

    print(f"\n[Sampling] Drew {n_total} random points out of {n_samples} "
          f"total input samples.")
    print(f"  Stage 1 (physical param fit): {len(stage1_idx)} points")
    print(f"  Stage 2 (alpha(h) fit):       {len(stage2_idx)} points")

    rp1, mw1, vd1 = (robot_positions[stage1_idx],
                      m_world[stage1_idx],
                      voltage_data[stage1_idx])
    rp2, mw2, vd2 = (robot_positions[stage2_idx],
                      m_world[stage2_idx],
                      voltage_data[stage2_idx])

    return (stage1_idx, rp1, mw1, vd1), (stage2_idx, rp2, mw2, vd2)


# =============================================================================
# NEW: STAGE 2 - GLOBAL LINEAR ALPHA(H) = C0 + C1*H  (closed-form ridge)
# =============================================================================
# height region, we fit a single alpha(h) = c0 + c1*h shared across ALL
# sensors (pooled least squares, same pooling philosophy as the old
# region-closed-form). Since alpha(h) is LINEAR in (c0, c1), the model
#
#     V[i,s] - a[s] = g[s] * B_proj[i,s] * (c0 + c1 * h[i,s])
#                    = c0 * (g*B)[i,s]  +  c1 * (g*B*h)[i,s]
#
# is still ordinary linear regression -> solved via ridge-regularized
# normal equations, no scipy least_squares/bounds needed. The ridge priors
# (c0 -> 1, c1 -> 0) keep c1 from silently absorbing residual gain/offset
# error left over from Stage 1.
# =============================================================================

def calibrate_alpha_linear(
        physical_results,
        rp_calib2,
        mw_calib2,
        vd_calib2,
        c0_prior=ALPHA_C0_PRIOR,
        c1_prior=ALPHA_C1_PRIOR,
        lambda_c0=LAMBDA_ALPHA_C0,
        lambda_c1=LAMBDA_ALPHA_C1):

    n_samples = rp_calib2.shape[0]
    n_sensors = physical_results.shape[0]

    sensor_pos = physical_results[:, 0:3]       # (n_sensors, 3): x, y, z
    a = physical_results[:, 3]                  # (n_sensors,)  offset
    g = physical_results[:, 4]                  # (n_sensors,)  gain
    sensor_dir = physical_results[:, 5:8]       # (n_sensors, 3): nx, ny, nz

    # h[i, s] = z_capsule_i - z_sensor_calibrated_s  (uses Stage-1 result)
    z_calibrated = sensor_pos[:, 2]
    h = rp_calib2[:, 2][:, None] - z_calibrated[None, :]   # (n_samples, n_sensors)

    # Raw dipole projection B_proj[i, s] using the FROZEN calibrated pose
    # (position + direction) of each sensor -- gain/offset are NOT
    # reapplied here, they're multiplied in separately below.
    B_proj = np.zeros((n_samples, n_sensors))
    for s in range(n_sensors):
        r_vec = sensor_pos[s] - rp_calib2               # (n_samples, 3)
        B = dipole_field(r_vec, mw_calib2)               # (n_samples, 3)
        B_proj[:, s] = B @ sensor_dir[s]

    gB = g[None, :] * B_proj                            # (n_samples, n_sensors)
    v_minus_a = vd_calib2 - a[None, :]                   # V_measured - a

    # ---- Pool ALL (sample, sensor) pairs into one regression ----
    x_c0 = gB.ravel()                # column for c0
    x_c1 = (gB * h).ravel()          # column for c1
    y = v_minus_a.ravel()

    X = np.column_stack([x_c0, x_c1])          # (n_samples*n_sensors, 2)

    # ---- Ridge normal equations ----
    # minimize ||y - X @ theta||^2
    #          + lambda_c0*(c0 - c0_prior)^2 + lambda_c1*(c1 - c1_prior)^2
    XtX = X.T @ X
    Xty = X.T @ y
    Lambda = np.diag([lambda_c0, lambda_c1])
    prior = np.array([c0_prior, c1_prior])

    theta = np.linalg.solve(XtX + Lambda, Xty + Lambda @ prior)
    c0, c1 = theta

    resid = y - X @ theta
    rmse = np.sqrt(np.mean(resid**2))

    print(f"\n[Stage 2] alpha(h) = {c0:.6f} + ({c1:.6f}) * h")
    print(f"  fit from {len(y)} sensor-sample pairs "
          f"({n_samples} points x {n_sensors} sensors), RMSE = {rmse:.6f} V")
    print(f"  h range in Stage-2 set: [{h.min():.4f}, {h.max():.4f}] m")

    return {"c0": c0, "c1": c1, "rmse": rmse}


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
# SAVE STAGE 1 / STAGE 2 RESULTS (required output files)
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


def save_alpha_results(alpha_params, output_file):
    """
    NEW: Save Stage 2 alpha(h) = c0 + c1*h coefficients to
    Calibration_Alpha.csv (2 rows: c0, c1) instead of the old 3
    per-region constants.
    """
    df = pd.DataFrame({
        "coefficient": ["c0", "c1"],
        "value": [alpha_params["c0"], alpha_params["c1"]],
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
# MAIN
# =============================================================================

def main():
    """Main execution function -- 2-stage calibration framework
    (random 400-point sampling; global linear alpha(h) in Stage 2)"""

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
    # SAMPLING: 400 random points from the whole dataset -> split into
    # Stage 1 (240) / Stage 2 (160). No more per-region stratification.
    # ==================================================================
    print("\n===================================")
    print("SAMPLING: 400 random points -> 240 (Stage 1) / 160 (Stage 2)")
    print("===================================")
    (stage1_idx, rp1, mw1, vd1), (stage2_idx, rp2, mw2, vd2) = \
        select_stage1_stage2_split(robot_positions, m_world, voltage_data)

    # ==================================================================
    # STAGE 1: Per-sensor physical parameter calibration
    # (unchanged residual function, dipole model, initial values, bounds)
    # ==================================================================
    print("\n===================================")
    print("STAGE 1: PHYSICAL PARAMETER FIT (240 points)")
    print("===================================")
    results, rmses = run_calibration(
        sensor_positions, offsets, rp1, mw1, vd1
    )

    print("\n========================")
    print(f"Mean RMSE = {np.mean(rmses):.6f}")
    print(f"Max RMSE  = {np.max(rmses):.6f}")
    print(f"Min RMSE  = {np.min(rmses):.6f}")
    print("========================")

    # Save Stage 1 physical parameters (frozen going into Stage 2)
    save_physical_results(results, PHYSICAL_OUTPUT_PATH)

    # Plot
    plot_rmse(rmses)

    # ==================================================================
    # STAGE 2: Global linear alpha(h) = c0 + c1*h (physical params frozen)
    # ==================================================================
    print("\n===================================")
    print("STAGE 2: ALPHA(H) CORRECTION (closed-form ridge, 160 points)")
    print("===================================")
    alpha_params = calibrate_alpha_linear(results, rp2, mw2, vd2)
    save_alpha_results(alpha_params, ALPHA_OUTPUT_PATH)

    print("\n===================================")
    print("ALL STAGES FINISHED")
    print("===================================")


if __name__ == "__main__":
    main()