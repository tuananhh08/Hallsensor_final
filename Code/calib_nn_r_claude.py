
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
 
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
 
from scipy.optimize import least_squares
import optuna
 
# =============================================================================
# FILE PATHS
# =============================================================================
BASE_DIR = Path(r"D:\Downloads\Hallsensor_final\Data_8_2026")  # WINDOWS
# BASE_DIR = Path(r"/Users/tuananhnguyen/Downloads/Hallsensor_final/Data_8_2026")  # MAC

SENSOR_POSITIONS_PATH = BASE_DIR / "Hall_sensor_positions.csv"
ROBOT_POSE_PATH = BASE_DIR / "Grid_points_coordinates.csv"
VOLTAGE_DATA_PATH = BASE_DIR / "Grid_data.csv"
OFFSET_INIT_PATH = BASE_DIR / "Offset_Sens.csv"
 
PHYSICAL_OUTPUT_PATH = BASE_DIR / "Calibration_Physical_NN_r.csv"
ALPHA_NN_OUTPUT_PATH = BASE_DIR / "Calibration_AlphaNN_r.pt"
ALPHA_NN_META_PATH = BASE_DIR / "Calibration_AlphaNN_r_meta.csv"
 
# =============================================================================
# CONSTANTS
# =============================================================================
 
MU0_OVER_4PI = 1e-7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
 
# ---- Stage 1 / Stage 2 split sizes ----
N_TOTAL_CALIB_SAMPLES = 2000
N_STAGE1_SAMPLES = 300
N_STAGE2_SAMPLES = N_TOTAL_CALIB_SAMPLES - N_STAGE1_SAMPLES  # 1700
 
# FIX 3: Stage-2 pool is now split THREE ways: train / val / test.
# val is used by Optuna + early stopping (model selection).
# test is NEVER touched until the final, one-time honest evaluation.
STAGE2_VAL_FRACTION = 0.2
STAGE2_TEST_FRACTION = 0.15  
 
# ---- Stage 1 regularization weights (physical priors) -- unchanged ----
LAMBDA_POS = 2000
LAMBDA_GAIN = 9e-3
LAMBDA_OFFSET = 750
 
# ---- Stage 2 NN: Optuna search budget ----
N_OPTUNA_TRIALS = 30
MAX_EPOCHS = 200
EARLY_STOP_PATIENCE = 10
 
 
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
 
 
# =============================================================================
# DIPOLE MODEL
# =============================================================================
 
def dipole_field(r_vec, m_vec):
    """Calculate magnetic field from dipole model. r_vec, m_vec: (N, 3)."""
    r = np.linalg.norm(r_vec, axis=1, keepdims=True)
    r3 = np.maximum(r ** 3, 1e-12)
    r5 = np.maximum(r ** 5, 1e-12)
    mdotr = np.sum(m_vec * r_vec, axis=1, keepdims=True)
    B = MU0_OVER_4PI * (3.0 * r_vec * mdotr / r5 - m_vec / r3)
    return B
 
 
# =============================================================================
# LOADERS
# =============================================================================
 
def load_sensor_positions(file_path):
    df = pd.read_csv(file_path)
    sensor_positions = df.values
    print(f"Loaded sensor positions: {sensor_positions.shape}")
    return sensor_positions
 
 
def load_robot_pose(file_path):
    df = pd.read_csv(file_path)
    required_cols = ["x", "y", "z", "mx", "my", "mz"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    positions = df[["x", "y", "z"]].values
    m_world = df[["mx", "my", "mz"]].values
    norm = np.linalg.norm(m_world, axis=1, keepdims=True)
    m_world = m_world / norm
    print(f"Loaded robot positions: {positions.shape}")
    print(f"Loaded magnetic orientations: {m_world.shape}")
    return positions, m_world
 
 
def load_voltage_data(file_path):
    df = pd.read_csv(file_path)
    voltage = df.values
    print(f"Loaded voltage data: {voltage.shape}")
    return voltage
 
 
def load_offset_initial_values(file_path, n_sensors):
    df = pd.read_csv(file_path)
    required_columns = {"sensor_index", "offset_a_V"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing column(s) in {file_path.name}: {sorted(missing_columns)}")
    if df["sensor_index"].duplicated().any():
        raise ValueError("Offset_Sens.csv contains duplicate sensor_index values.")
    df = df.sort_values("sensor_index").reset_index(drop=True)
    expected_indices = np.arange(n_sensors)
    actual_indices = df["sensor_index"].to_numpy()
    if not np.array_equal(actual_indices, expected_indices):
        raise ValueError(
            f"Offset_Sens.csv must contain exactly sensor_index values 0 to {n_sensors - 1}."
        )
    offset_initial_values = df["offset_a_V"].to_numpy(dtype=float)
    if not np.isfinite(offset_initial_values).all():
        raise ValueError("Column offset_a_V contains missing or non-finite values.")
    print(f"Loaded per-sensor offset initial values: {offset_initial_values.shape}")
    return offset_initial_values
 
 
# =============================================================================
# STAGE 1: PER-SENSOR PHYSICAL PARAMETER FIT 
# =============================================================================
 
def sensor_residuals(params, robot_positions, m_world, voltage_sensor,
                      pos_prior=None, offset_prior=None, g0=7.5,
                      lambda_pos=LAMBDA_POS, lambda_gain=LAMBDA_GAIN,
                      lambda_offset=LAMBDA_OFFSET):
    x, y, z, a, g = params
    sensor_dir = np.array([0.0, 0.0, 1.0])
    sensor_pos = np.array([x, y, z])
    r_vec = sensor_pos - robot_positions
    B = dipole_field(r_vec, m_world)
    B_proj = B @ sensor_dir
    voltage_pred = a + g * B_proj
    r_voltage = voltage_sensor - voltage_pred
 
    if not np.all(np.isfinite(r_voltage)):
        print(f"WARNING: Non-finite residuals detected at params: {params}")
 
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
 
 
def calibrate_single_sensor(sensor_index, sensor_pos_init, robot_positions,
                             m_world, voltage_sensor, offset_init):
    g0 = 7.5
    x0 = np.array([sensor_pos_init[0], sensor_pos_init[1], sensor_pos_init[2],
                    offset_init, g0])
    pos_tol = 0.001
    lower = [sensor_pos_init[0] - pos_tol, sensor_pos_init[1] - pos_tol,
             sensor_pos_init[2] - pos_tol, offset_init - 0.0011, 7]
    upper = [sensor_pos_init[0] + pos_tol, sensor_pos_init[1] + pos_tol,
             sensor_pos_init[2] + pos_tol, offset_init + 0.0011, 8]
 
    result = least_squares(
        sensor_residuals, x0, bounds=(lower, upper),
        args=(robot_positions, m_world, voltage_sensor),
        kwargs=dict(
            pos_prior=(sensor_pos_init[0], sensor_pos_init[1], sensor_pos_init[2]),
            offset_prior=offset_init, g0=g0,
        ),
        method="trf", max_nfev=250,
    )
 
    params_opt = result.x
    nx, ny, nz = 0.0, 0.0, 1.0
    params_extended = np.array([
        params_opt[0], params_opt[1], params_opt[2], params_opt[3], params_opt[4],
        nx, ny, nz,
    ])
 
    n_voltage = voltage_sensor.shape[0]
    rmse = np.sqrt(np.mean(result.fun[:n_voltage] ** 2))
    print(f"Sensor {sensor_index + 1:02d} | RMSE = {rmse:.6f} V")
    return params_extended, rmse
 
 
def run_calibration(sensor_positions, robot_positions, m_world, voltage_data,
                     offset_initial_values):
    n_sensors = sensor_positions.shape[0]
    if len(offset_initial_values) != n_sensors:
        raise ValueError(f"Expected {n_sensors} initial offsets, got {len(offset_initial_values)}.")
    results, rmses = [], []
    for i in range(n_sensors):
        params, rmse = calibrate_single_sensor(
            sensor_index=i, sensor_pos_init=sensor_positions[i],
            robot_positions=robot_positions, m_world=m_world,
            voltage_sensor=voltage_data[:, i], offset_init=offset_initial_values[i],
        )
        results.append(params)
        rmses.append(rmse)
    return np.array(results), np.array(rmses)
 
 
# =============================================================================
# SAMPLING: Stage1 / Stage2(train) / Stage2(val) / Stage2(test) split
# =============================================================================
 
def select_splits(robot_positions, m_world, voltage_data,
                   n_total=N_TOTAL_CALIB_SAMPLES, n_stage1=N_STAGE1_SAMPLES,
                   val_fraction=STAGE2_VAL_FRACTION,
                   test_fraction=STAGE2_TEST_FRACTION, seed=SEED):
    """Deterministic (seeded) split so the NN hyperparameter search and
    results are reproducible."""
    n_samples = robot_positions.shape[0]
    if n_total > n_samples:
        raise ValueError(f"Requested {n_total} calibration points but dataset only has {n_samples}.")
 
    rng = np.random.default_rng(seed)
    all_idx = rng.choice(n_samples, size=n_total, replace=False)
    stage1_idx = all_idx[:n_stage1]
    stage2_idx = all_idx[n_stage1:]
 
    n_test = int(len(stage2_idx) * test_fraction)
    stage2_test_idx = stage2_idx[:n_test]
    remaining_idx = stage2_idx[n_test:]
 
    n_val = int(len(remaining_idx) * val_fraction)
    stage2_val_idx = remaining_idx[:n_val]
    stage2_train_idx = remaining_idx[n_val:]
 
    print(f"\n[Sampling] seed={seed} | total={n_total} -> "
          f"Stage1={len(stage1_idx)}, Stage2-train={len(stage2_train_idx)}, "
          f"Stage2-val={len(stage2_val_idx)}, Stage2-test={len(stage2_test_idx)}")
 
    def gather(idx):
        return robot_positions[idx], m_world[idx], voltage_data[idx]
 
    return (
        (stage1_idx, *gather(stage1_idx)),
        (stage2_train_idx, *gather(stage2_train_idx)),
        (stage2_val_idx, *gather(stage2_val_idx)),
        (stage2_test_idx, *gather(stage2_test_idx)),
    )
 
 
# =============================================================================
# STAGE 2: NEURAL alpha(h) = 1 + delta_alpha(h)
# =============================================================================
# FIX 2: feature is h (signed height offset along the sensor's fixed
# sensitive axis: h = capsule_z - sensor_z), not r (3D distance). This
# matches the linear alpha(h) = c0 + c1*h baseline so the comparison is fair,
# and is the physically motivated feature for a single-axis sensor (see
# earlier discussion: r conflates vertical + lateral offset).
#
# FIX 1: output_scale is a genuine nn.Parameter, initialized to
# output_scale_init but free to grow/shrink during training via gradient
# descent, instead of being permanently fixed at whatever Optuna happened
# to draw at init time.
#
# Architecture is FIXED (not tuned by Optuna): 5 hidden layers
# (32, 64, 64, 64, 32), SiLU activation. Only training hyperparameters are
# searched by Optuna.
# =============================================================================
 
class DeltaAlphaNet(nn.Module):
    HIDDEN_DIMS = (32, 64, 64, 64, 16)
 
    def __init__(self, input_dim: int = 1, output_scale_init: float = 0.05):
        super().__init__()
        layers = []
        in_dim = input_dim
        for hidden_dim in self.HIDDEN_DIMS:
            layers += [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
            in_dim = hidden_dim
        layers += [nn.Linear(in_dim, 1)]
        self.net = nn.Sequential(*layers)
 
        # Zero-init the last layer so delta_alpha(h) == 0 everywhere at the
        # start of training (alpha(h) == 1, i.e. "no correction" -- matches
        # the ridge-prior philosophy of the closed-form linear version).
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
 
        # FIX 1: output_scale is now a learnable parameter, not a fixed
        # Python float. Gradient descent can grow it if the network needs
        # more correction amplitude than the initial guess allowed, and
        # weight_decay (already applied to all parameters by the optimizer)
        # keeps it from blowing up unchecked.
        self.output_scale = nn.Parameter(torch.tensor(float(output_scale_init)))
 
    def forward(self, h_norm: torch.Tensor) -> torch.Tensor:
        delta = self.net(h_norm) * self.output_scale
        return delta.squeeze(-1)  # (N,)
 
 
def build_stage2_features(physical_results, rp, mw):
    """Compute h (signed height offset along the sensor's fixed sensitive
    axis) and gB (= g * B_proj, dipole prediction times gain) for every
    (sample, sensor) pair, using the FROZEN Stage-1 physical parameters.
    Returns flat (n_samples*n_sensors,) arrays."""
    sensor_pos = physical_results[:, 0:3]
    a = physical_results[:, 3]
    g = physical_results[:, 4]
    sensor_dir = physical_results[:, 5:8]
 
    n_samples = rp.shape[0]
    n_sensors = physical_results.shape[0]
 
    h_mat = np.zeros((n_samples, n_sensors))
    gB_mat = np.zeros((n_samples, n_sensors))
    for s in range(n_sensors):
        r_vec = sensor_pos[s] - rp  # (n_samples, 3)
        h_mat[:, s] = rp[:, 2] - sensor_pos[s, 2]  # FIX 2: h, not |r_vec|
        B = dipole_field(r_vec, mw)
        gB_mat[:, s] = g[s] * (B @ sensor_dir[s])
 
    a_mat = np.broadcast_to(a[None, :], (n_samples, n_sensors))
    return h_mat.ravel(), gB_mat.ravel(), a_mat.ravel()
 
 
def make_tensors(h_flat, gB_flat, a_flat, v_flat, h_mean, h_std):
    h_norm = (h_flat - h_mean) / h_std
    return (
        torch.tensor(h_norm, dtype=torch.float32).unsqueeze(-1),
        torch.tensor(gB_flat, dtype=torch.float32),
        torch.tensor(a_flat, dtype=torch.float32),
        torch.tensor(v_flat, dtype=torch.float32),
    )
 

def train_alpha_nn(model, train_loader, val_h, val_gB, val_a, val_v,
                    lr, weight_decay, delta_l2, max_epochs=MAX_EPOCHS,
                    patience=EARLY_STOP_PATIENCE, verbose=False,
                    early_stop=True):
    """
    early_stop=True  -> normal model-selection mode: monitor (val_h,...),
                        keep the best checkpoint, stop after `patience`
                        epochs without improvement. Returns (model, best_val,
                        best_epoch).
    early_stop=False -> fixed-epoch "deploy" mode: run exactly max_epochs,
                         no checkpointing against val (val is only used to
                         LOG progress here, never to pick a checkpoint or to
                         decide when to stop). Use this for the final refit
                         on train+val+test so that split can never leak into
                         model selection.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    huber = nn.HuberLoss(delta=1e-3)
 
    best_val = float("inf")
    best_state = None
    best_epoch = -1
    epochs_no_improve = 0
 
    for epoch in range(max_epochs):
        model.train()
        for h_b, gB_b, a_b, v_b in train_loader:
            h_b, gB_b, a_b, v_b = (t.to(DEVICE) for t in (h_b, gB_b, a_b, v_b))
            optimizer.zero_grad()
            delta_alpha = model(h_b)
            alpha = 1.0 + delta_alpha
            v_pred = a_b + gB_b * alpha
            loss_data = huber(v_pred, v_b)
            loss_reg = delta_l2 * (delta_alpha ** 2).mean()
            loss = loss_data + loss_reg
            loss.backward()
            optimizer.step()
 
        model.eval()
        with torch.no_grad():
            delta_alpha_val = model(val_h.to(DEVICE))
            alpha_val = 1.0 + delta_alpha_val
            v_pred_val = val_a.to(DEVICE) + val_gB.to(DEVICE) * alpha_val
            val_rmse = torch.sqrt(torch.mean((v_pred_val - val_v.to(DEVICE)) ** 2)).item()
 
        if verbose and epoch % 5 == 0:
            tag = "val" if early_stop else "val(monitor only)"
            print(f"  epoch {epoch:4d} | {tag} RMSE = {val_rmse:.6f} V | "
                  f"output_scale = {model.output_scale.item():.5f}")
 
        if not early_stop:
            # Fixed-epoch deploy mode: never checkpoint/stop early on val.
            continue
 
        if val_rmse < best_val - 1e-9:
            best_val = val_rmse
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break
 
    if early_stop:
        if best_state is not None:
            model.load_state_dict(best_state)
        return model, best_val, best_epoch
    else:
        # Deploy mode: whatever the model looks like after max_epochs is
        # what gets returned (no checkpoint selection against any split).
        return model, val_rmse, max_epochs - 1
 
 
def optuna_objective(trial, train_tensors, val_tensors):
    # Architecture (hidden_dim / n_layers) is fixed -- see DeltaAlphaNet.HIDDEN_DIMS.
    # Only training hyperparameters are searched.
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    delta_l2 = trial.suggest_float("delta_l2", 1e-6, 1e-1, log=True)
    output_scale_init = trial.suggest_float("output_scale_init", 1e-3, 0.3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64])
 
    set_seed(SEED + trial.number)
 
    h_tr, gB_tr, a_tr, v_tr = train_tensors
    dataset = TensorDataset(h_tr, gB_tr, a_tr, v_tr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
 
    model = DeltaAlphaNet(output_scale_init=output_scale_init).to(DEVICE)
 
    _, val_rmse, best_epoch = train_alpha_nn(
        model, loader, *val_tensors,
        lr=lr, weight_decay=weight_decay, delta_l2=delta_l2,
        early_stop=True,
    )
 
    trial.set_user_attr("output_scale_init", output_scale_init)
    trial.set_user_attr("best_epoch", best_epoch)
    return val_rmse
 
 
def calibrate_alpha_nn(physical_results,
                        rp_train, mw_train, vd_train,
                        rp_val, mw_val, vd_val,
                        rp_test, mw_test, vd_test,
                        n_trials=N_OPTUNA_TRIALS):
    h_train, gB_train, a_train = build_stage2_features(physical_results, rp_train, mw_train)
    v_train = vd_train.ravel()
    h_val, gB_val, a_val = build_stage2_features(physical_results, rp_val, mw_val)
    v_val = vd_val.ravel()
    h_test, gB_test, a_test = build_stage2_features(physical_results, rp_test, mw_test)
    v_test = vd_test.ravel()
 
    # Normalization stats from TRAIN ONLY (never let val/test leak into
    # this either).
    h_mean, h_std = h_train.mean(), h_train.std()
 
    train_tensors = make_tensors(h_train, gB_train, a_train, v_train, h_mean, h_std)
    val_tensors = make_tensors(h_val, gB_val, a_val, v_val, h_mean, h_std)
    test_tensors = make_tensors(h_test, gB_test, a_test, v_test, h_mean, h_std)
 
    print(f"\n[Stage 2 / Optuna] {n_trials} trials, "
          f"{len(v_train)} train pairs, {len(v_val)} val pairs, "
          f"{len(v_test)} test pairs (held out, untouched until the end)")
    print(f"[Stage 2] h range train=[{h_train.min():.4f}, {h_train.max():.4f}] m")
    print(f"[Stage 2] Fixed architecture: {DeltaAlphaNet.HIDDEN_DIMS} (SiLU)")
 
    study = optuna.create_study(direction="minimize",
                                 sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(
        lambda trial: optuna_objective(trial, train_tensors, val_tensors),
        n_trials=n_trials, show_progress_bar=False,
    )
 
    print(f"\n[Stage 2 / Optuna] Best val RMSE = {study.best_value:.6f} V")
    print(f"[Stage 2 / Optuna] Best params = {study.best_params}")
    print(f"[Stage 2 / Optuna] Best trial's early-stop epoch = "
          f"{study.best_trial.user_attrs['best_epoch']}")
 
    best = study.best_params
    best_epoch = study.best_trial.user_attrs["best_epoch"]
 
    # ---- Model-selection run: retrain fresh on TRAIN ONLY with the winning
    # hyperparameters, early-stopping against VAL (never touches test). This
    # is the model whose performance on TEST we report as the honest,
    # leakage-free final number. ----
    set_seed(SEED)
    h_tr, gB_tr, a_tr, v_tr = train_tensors
    loader_train = DataLoader(TensorDataset(h_tr, gB_tr, a_tr, v_tr),
                               batch_size=best["batch_size"], shuffle=True)
    selection_model = DeltaAlphaNet(output_scale_init=best["output_scale_init"]).to(DEVICE)
    selection_model, selection_val_rmse, _ = train_alpha_nn(
        selection_model, loader_train, *val_tensors,
        lr=best["lr"], weight_decay=best["weight_decay"], delta_l2=best["delta_l2"],
        early_stop=True, verbose=True,
    )
 
    selection_model.eval()
    with torch.no_grad():
        h_te, gB_te, a_te, v_te = test_tensors
        delta_alpha_test = selection_model(h_te.to(DEVICE))
        alpha_test = 1.0 + delta_alpha_test
        v_pred_test = a_te.to(DEVICE) + gB_te.to(DEVICE) * alpha_test
        test_rmse = torch.sqrt(torch.mean((v_pred_test - v_te.to(DEVICE)) ** 2)).item()
 
    print(f"\n[Stage 2] HONEST held-out TEST RMSE (never used for Optuna or "
          f"early stopping) = {test_rmse:.6f} V")
 
    # ---- OPTIONAL deploy refit: train on train+val+test combined for the
    # artifact you actually ship, running a FIXED epoch count (from the
    # model-selection run above) with NO further early stopping against
    # test -- so test_rmse reported above is never influenced by this step.
    # ----
    h_full = np.concatenate([h_train, h_val, h_test])
    gB_full = np.concatenate([gB_train, gB_val, gB_test])
    a_full = np.concatenate([a_train, a_val, a_test])
    v_full = np.concatenate([v_train, v_val, v_test])
    full_tensors = make_tensors(h_full, gB_full, a_full, v_full, h_mean, h_std)
    h_f, gB_f, a_f, v_f = full_tensors
    loader_full = DataLoader(TensorDataset(h_f, gB_f, a_f, v_f),
                              batch_size=best["batch_size"], shuffle=True)
 
    set_seed(SEED)
    deploy_model = DeltaAlphaNet(output_scale_init=best["output_scale_init"]).to(DEVICE)
    fixed_epochs = best_epoch + 1 if best_epoch >= 0 else MAX_EPOCHS
    deploy_model, _, _ = train_alpha_nn(
        deploy_model, loader_full, *val_tensors,  # val used ONLY for progress logging here
        lr=best["lr"], weight_decay=best["weight_decay"], delta_l2=best["delta_l2"],
        max_epochs=fixed_epochs, early_stop=False, verbose=True,
    )
 
    meta = {
        "h_mean": float(h_mean), "h_std": float(h_std),
        "hidden_dims": "-".join(map(str, DeltaAlphaNet.HIDDEN_DIMS)),
        "activation": "SiLU",
        "output_scale_init": best["output_scale_init"],
        "output_scale_final": deploy_model.output_scale.item(),
        "selection_val_rmse": selection_val_rmse,
        "held_out_test_rmse": test_rmse,
        "deploy_fixed_epochs": fixed_epochs,
    }
    return deploy_model, meta, study, selection_model, test_tensors
 
 
# =============================================================================
# SAVE
# =============================================================================
 
def save_physical_results(results, output_file):
    df = pd.DataFrame({
        "sensor_index": np.arange(len(results)),
        "x": results[:, 0], "y": results[:, 1], "z": results[:, 2],
        "offset": results[:, 3], "gain": results[:, 4],
    })
    df.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")
 
 
def save_alpha_nn(model, meta, model_path, meta_path):
    torch.save(model.state_dict(), model_path)
    pd.DataFrame([meta]).to_csv(meta_path, index=False)
    print(f"Saved: {model_path}")
    print(f"Saved: {meta_path}")
 
 
def plot_rmse(rmses):
    plt.figure(figsize=(10, 5))
    plt.bar(np.arange(len(rmses)), rmses)
    plt.xlabel("Sensor Index")
    plt.ylabel("RMSE")
    plt.title("Stage 1 Calibration RMSE")
    plt.grid(True)
    plt.show()
 
 
def plot_alpha_curve(model, meta, h_range=(-0.02, 0.15)):
    model.eval()
    h_plot = np.linspace(*h_range, 300)
    h_norm = (h_plot - meta["h_mean"]) / meta["h_std"]
    with torch.no_grad():
        h_t = torch.tensor(h_norm, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
        delta_alpha = model(h_t).cpu().numpy()
    alpha = 1.0 + delta_alpha
    plt.figure(figsize=(8, 5))
    plt.plot(h_plot * 1000, alpha)
    plt.xlabel("h (mm)")
    plt.ylabel("alpha(h) = 1 + delta_alpha(h)")
    plt.title("Learned NN correction curve (deploy model)")
    plt.grid(True)
    plt.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    plt.show()
 
 
# =============================================================================
# MAIN
# =============================================================================
 
def main():
    set_seed(SEED)
 
    sensor_positions = load_sensor_positions(SENSOR_POSITIONS_PATH)
    robot_positions, m_world = load_robot_pose(ROBOT_POSE_PATH)
    voltage_data = load_voltage_data(VOLTAGE_DATA_PATH)
    offset_initial_values = load_offset_initial_values(
        OFFSET_INIT_PATH, n_sensors=sensor_positions.shape[0]
    )
 
    n_samples = min(len(robot_positions), len(voltage_data))
    robot_positions = robot_positions[:n_samples]
    m_world = m_world[:n_samples]
    voltage_data = voltage_data[:n_samples]
 
    print("\n===================================")
    print("SAMPLING: Stage1 / Stage2-train / Stage2-val / Stage2-test split (seeded)")
    print("===================================")
    (s1_idx, rp1, mw1, vd1), (s2t_idx, rp2t, mw2t, vd2t), \
        (s2v_idx, rp2v, mw2v, vd2v), (s2te_idx, rp2te, mw2te, vd2te) = \
        select_splits(robot_positions, m_world, voltage_data)
 
    print("\n===================================")
    print("STAGE 1: PHYSICAL PARAMETER FIT")
    print("===================================")
    results, rmses = run_calibration(
        sensor_positions, rp1, mw1, vd1, offset_initial_values=offset_initial_values,
    )
    print(f"\nStage 1 Mean RMSE = {np.mean(rmses):.6f} | "
          f"Max = {np.max(rmses):.6f} | Min = {np.min(rmses):.6f}")
    save_physical_results(results, PHYSICAL_OUTPUT_PATH)
    plot_rmse(rmses)
 
    print("\n===================================")
    print("STAGE 2: NEURAL alpha(h) CORRECTION (Optuna-tuned)")
    print("===================================")
    deploy_model, meta, study, selection_model, test_tensors = calibrate_alpha_nn(
        results, rp2t, mw2t, vd2t, rp2v, mw2v, vd2v, rp2te, mw2te, vd2te,
        n_trials=N_OPTUNA_TRIALS,
    )
    save_alpha_nn(deploy_model, meta, ALPHA_NN_OUTPUT_PATH, ALPHA_NN_META_PATH)
    plot_alpha_curve(deploy_model, meta)
 
    print("\n===================================")
    print("ALL STAGES FINISHED")
    print(f"Honest held-out TEST RMSE (Stage 2, model-selection run): "
          f"{meta['held_out_test_rmse']:.6f} V")
    print(f"(Deploy model was refit on train+val+test for "
          f"{meta['deploy_fixed_epochs']} fixed epochs -- not evaluated "
          f"against test again, to avoid leakage into this metric.)")
    print("===================================")
 
 
if __name__ == "__main__":
    main()