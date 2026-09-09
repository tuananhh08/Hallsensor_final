"""Chan doan hinh dang alpha(h): uoc luong alpha thuc nghiem tu du lieu raw + Stage 1,
roi ve theo h tren nhieu truc (linear / semi-log / log-log) de chon dang ham
(tuyen tinh, mu, log, rational, da thuc) truoc khi sua Stage 2."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# FILE PATHS  (sua lai cho khop moi truong cua ban)
# =============================================================================
# BASE_DIR = Path(r"/Users/tuananhnguyen/Downloads/Hallsensor_final/Data_8_2026") #MAC
BASE_DIR = Path(r"D:\Downloads\Hallsensor_final\Data_8_2026") #WINDOWS

PHYSICAL_PATH = BASE_DIR / "Calibration_Physical_new.csv"
VOLTAGE_PATH = BASE_DIR / "Helix_data_2.csv"
COORDS_PATH = BASE_DIR / "Helix_points_coordinates_2.csv"

OUTPUT_DIR = BASE_DIR / "outputs" / "alpha_diagnostics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MU0_OVER_4PI = 1e-7

# Bo qua cac diem qua gan sensor (h nho) khi ve log-log / rational, vi
# 1/h va log(h) phat ky khi h -> 0. Chinh lai neu don vi/khoang cach khac.
H_MIN_FOR_LOG = 1e-4  # met

# Alpha thuc nghiem qua nhieu (|g*B_proj| qua nho) se bi loai de tranh
# chia cho so gan 0 lam nhieu diem ao.
MIN_GB_MAGNITUDE = 1e-6  # V, nguong |g*B_proj| toi thieu de giu diem

N_BINS = 10  # so bin theo h (giam tu 25 -> 10 giup moi bin co du diem)

# So sensor ve rieng (overlay) khi kiem tra xem dao dong o duong median
# gop co phai do TRON LAN nhieu sensor khac alpha nhau hay khong. Chon
# ngau nhien de tranh thien vi ve mot vung cua mang sensor.
N_SENSORS_OVERLAY = 8


# =============================================================================
# DIPOLE MODEL (giu nguyen cong thuc goc)
# =============================================================================
def dipole_field(r_vec, m_vec):
    r = np.linalg.norm(r_vec, axis=1, keepdims=True)
    r3 = np.maximum(r ** 3, 1e-12)
    r5 = np.maximum(r ** 5, 1e-12)
    mdotr = np.sum(m_vec * r_vec, axis=1, keepdims=True)
    B = MU0_OVER_4PI * (3.0 * r_vec * mdotr / r5 - m_vec / r3)
    return B


# =============================================================================
# LOAD DATA
# =============================================================================
def load_physical_calib(path):
    """sensor_index, x, y, z, offset, gain, theta, phi"""
    df = pd.read_csv(path)
    df = df.sort_values("sensor_index").reset_index(drop=True)
    return df


def load_voltage_data(path):
    df = pd.read_csv(path)
    return df.values, list(df.columns)


def load_robot_pose(path):
    df = pd.read_csv(path)
    positions = df[["x", "y", "z"]].values
    m_world = df[["mx", "my", "mz"]].values
    norm = np.linalg.norm(m_world, axis=1, keepdims=True)
    m_world = m_world / norm
    return positions, m_world


# =============================================================================
# BUILD EMPIRICAL ALPHA(H) FROM RAW DATA + STAGE-1 CALIB
# =============================================================================
def build_empirical_alpha_raw(physical_df, robot_positions, m_world, voltage_data):
    """
    Tinh h, alpha thuc nghiem, va |g*B_proj| cho MOI cap (sample, sensor),
    KHONG loc gi ca -- de sweep_gb_threshold() co the thu nhieu nguong
    khac nhau tren cung 1 bo du lieu goc.

    Tra ve h_grid, alpha_grid, gB_grid, sensor_idx_grid (n_samples, n_sensors).
    """
    n_sensors = physical_df.shape[0]
    n_samples = robot_positions.shape[0]

    x = physical_df["x"].to_numpy()
    y = physical_df["y"].to_numpy()
    z = physical_df["z"].to_numpy()
    a = physical_df["offset"].to_numpy()
    g = physical_df["gain"].to_numpy()
    sensor_dir = np.array([0.0, 0.0, 1.0])  # huong sensor co dinh thang dung

    h_grid = np.zeros((n_samples, n_sensors))
    alpha_grid = np.full((n_samples, n_sensors), np.nan)
    gB_grid = np.zeros((n_samples, n_sensors))

    for s in range(n_sensors):
        sensor_pos = np.array([x[s], y[s], z[s]])
        r_vec = sensor_pos - robot_positions
        B = dipole_field(r_vec, m_world)
        B_proj = B @ sensor_dir

        h = robot_positions[:, 2] - z[s]
        gB = g[s] * B_proj

        h_grid[:, s] = h
        gB_grid[:, s] = gB
        alpha_grid[:, s] = (voltage_data[:, s] - a[s]) / gB

    sensor_idx_grid = np.tile(np.arange(n_sensors), (n_samples, 1))
    return h_grid, alpha_grid, gB_grid, sensor_idx_grid


def filter_by_gb_threshold(h_grid, alpha_grid, gB_grid, sensor_idx_grid,
                            threshold, verbose=True):
    """Loc cac cap (sample, sensor) co |g*B_proj| < threshold, tra ve cac
    mang phang h_flat, alpha_flat, sensor_idx_flat."""
    valid_mask = np.abs(gB_grid) >= threshold

    h_flat = h_grid[valid_mask]
    alpha_flat = alpha_grid[valid_mask]
    sensor_idx_flat = sensor_idx_grid[valid_mask]

    if verbose:
        n_dropped = valid_mask.size - valid_mask.sum()
        print(f"[Diagnostic] Tong cong {valid_mask.size} cap (sample, sensor); "
              f"loai {n_dropped} diem co |g*B| < {threshold:.2e} V "
              f"(gan {100*n_dropped/valid_mask.size:.1f}%).")
        print(f"[Diagnostic] alpha thuc nghiem con lai: "
              f"mean={np.nanmean(alpha_flat):.4f}, "
              f"median={np.nanmedian(alpha_flat):.4f}, "
              f"std={np.nanstd(alpha_flat):.4f}")
        print(f"[Diagnostic] h range: [{h_flat.min():.4f}, {h_flat.max():.4f}] m")

    return h_flat, alpha_flat, sensor_idx_flat


def build_empirical_alpha(physical_df, robot_positions, m_world, voltage_data,
                           threshold=MIN_GB_MAGNITUDE):
    """Tien ich: raw + filter trong 1 buoc (giu tuong thich voi code cu)."""
    h_grid, alpha_grid, gB_grid, sensor_idx_grid = build_empirical_alpha_raw(
        physical_df, robot_positions, m_world, voltage_data
    )
    return filter_by_gb_threshold(h_grid, alpha_grid, gB_grid, sensor_idx_grid,
                                   threshold)


# =============================================================================
# NEW: QUET NGUONG |g*B| (MIN_GB_MAGNITUDE) -- chon nguong theo DU LIEU
# thay vi doan mo, bang cach xem std(alpha) on dinh lai o dau.
# =============================================================================
def sweep_gb_threshold(gB_grid, alpha_grid, n_thresholds=20):
    """
    Thu mot day nguong |g*B| tang dan (log-spaced, tu percentile thap den
    cao cua |g*B|), voi moi nguong tinh: % diem con lai, median(alpha),
    std(alpha), IQR(alpha). In bang + ve std/IQR vs nguong (truc log-x) --
    nguong nen chon la noi std/IQR NGUNG GIAM MANH (elbow), khong phai
    nguong lam std nho nhat co the (qua chat se bo mat qua nhieu du lieu
    that va lam meo xu huong that cua alpha(h)).
    """
    abs_gB = np.abs(gB_grid).ravel()
    finite_alpha = np.isfinite(alpha_grid.ravel())

    # Luoi nguong: tu percentile 0.5% den 60% cua |g*B| (qua 60% se bo qua
    # nhieu du lieu that).
    lo = np.percentile(abs_gB[finite_alpha], 0.5)
    hi = np.percentile(abs_gB[finite_alpha], 60)
    thresholds = np.geomspace(max(lo, 1e-12), hi, n_thresholds)

    rows = []
    for thr in thresholds:
        mask = (abs_gB >= thr) & finite_alpha
        vals = alpha_grid.ravel()[mask]
        if len(vals) < 10:
            continue
        pct_kept = 100 * mask.sum() / mask.size
        median = np.median(vals)
        std = np.std(vals)
        iqr = np.percentile(vals, 75) - np.percentile(vals, 25)
        rows.append((thr, pct_kept, median, std, iqr))

    print("\n[Sweep nguong |g*B|]  nguong        %con lai   median   std      IQR")
    for thr, pct_kept, median, std, iqr in rows:
        print(f"    {thr:10.3e}   {pct_kept:7.2f}%   {median:7.4f}  "
              f"{std:7.4f}  {iqr:7.4f}")

    return rows


def plot_gb_threshold_sweep(rows, output_dir):
    thr = np.array([r[0] for r in rows])
    pct_kept = np.array([r[1] for r in rows])
    std = np.array([r[3] for r in rows])
    iqr = np.array([r[4] for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(thr, std, "o-", color="tab:red", label="std(alpha)")
    axes[0].plot(thr, iqr, "s-", color="tab:purple", label="IQR(alpha)")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Nguong |g*B| (V)")
    axes[0].set_ylabel("Do phan tan cua alpha")
    axes[0].set_title("(a) Do phan tan alpha vs. nguong\n"
                       "-> chon nguong o cho DUONG NGUNG GIAM MANH (elbow)")
    axes[0].legend()
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].plot(thr, pct_kept, "o-", color="tab:blue")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Nguong |g*B| (V)")
    axes[1].set_ylabel("% diem con lai")
    axes[1].set_title("(b) % du lieu con lai vs. nguong\n"
                       "-> nguong cang cao cang mat nhieu du lieu that")
    axes[1].grid(True, which="both", alpha=0.3)

    fig.suptitle("Quet nguong |g*B| de loc diem chia-gan-0 (khong doan mo)")
    fig.tight_layout()
    fig.savefig(output_dir / "gb_threshold_sweep.png", dpi=130)
    plt.close(fig)
    print(f"\n[Diagnostic] Da luu: {output_dir / 'gb_threshold_sweep.png'}")
    print("Cach doc: chon nguong MIN_GB_MAGNITUDE tai diem std/IQR vua giam "
          "manh xong va bat dau di ngang (elbow) -- nguong cao hon nua chi "
          "danh doi mat them du lieu that ma khong giam duoc bao nhieu nhieu.")


# =============================================================================
# BINNING (de ve duong trung binh + khoang tin cay theo h, giam nhieu
# khi hien thi hang chuc nghin diem tan mat)
# =============================================================================
def bin_alpha_by_h(h_flat, alpha_flat, n_bins=N_BINS):
    """Chia h thanh n_bins khoang deu, tinh median + 25/75 percentile alpha
    trong tung khoang -- dung de ve duong xu huong ro rang tren nen scatter."""
    edges = np.linspace(h_flat.min(), h_flat.max(), n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    med = np.full(n_bins, np.nan)
    p25 = np.full(n_bins, np.nan)
    p75 = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)

    bin_idx = np.digitize(h_flat, edges[1:-1])
    for b in range(n_bins):
        vals = alpha_flat[bin_idx == b]
        counts[b] = len(vals)
        if len(vals) > 0:
            med[b] = np.median(vals)
            p25[b] = np.percentile(vals, 25)
            p75[b] = np.percentile(vals, 75)

    return centers, med, p25, p75, counts


# =============================================================================
# PLOTS: linear / semi-log (y) / log-log -- de nhan dang hinh dang ham
# =============================================================================
def plot_alpha_diagnostics(h_flat, alpha_flat, sensor_idx_flat, output_dir):
    centers, med, p25, p75, counts = bin_alpha_by_h(h_flat, alpha_flat)

    # ---- Fig 1: scatter (mau theo sensor) + duong median, truc thuong ----
    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(h_flat, alpha_flat, c=sensor_idx_flat, cmap="tab20",
                     s=3, alpha=0.15, linewidths=0)
    ax.plot(centers, med, "k-", linewidth=2.5, label="Median theo bin")
    ax.fill_between(centers, p25, p75, color="black", alpha=0.15,
                     label="Khoang 25-75 percentile")
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1,
               label="alpha = 1 (khong hieu chinh)")
    ax.set_xlabel("h = z_capsule - z_sensor (m)")
    ax.set_ylabel("alpha thuc nghiem = (V_meas - a) / (g*B_proj)")
    ax.set_title("Alpha(h) thuc nghiem -- truc thuong (linear-linear)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.colorbar(sc, ax=ax, label="sensor index")
    fig.tight_layout()
    fig.savefig(output_dir / "alpha_vs_h_linear.png", dpi=130)
    plt.close(fig)

    # ---- Fig 2: median-only, 3 kieu truc de nhan dang ham ----
    # Chi dung alpha median (>0) cho cac truc log; loc h > H_MIN_FOR_LOG.
    valid_log = (centers > H_MIN_FOR_LOG) & np.isfinite(med) & (med > 0)
    h_log = centers[valid_log]
    med_log = med[valid_log]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) linear-linear: neu thang hang -> alpha tuyen tinh theo h
    axes[0].plot(centers, med, "o-", color="tab:blue")
    axes[0].axhline(1.0, color="red", linestyle="--", linewidth=1)
    axes[0].set_xlabel("h (m)")
    axes[0].set_ylabel("alpha (median)")
    axes[0].set_title("(a) Linear-linear\n-> thang hang: alpha = c0 + c1*h")
    axes[0].grid(True, alpha=0.3)

    # (b) semi-log (y=log): neu thang hang -> alpha ~ exp(c1*h)
    if len(h_log) > 1:
        axes[1].plot(h_log, med_log, "o-", color="tab:orange")
        axes[1].set_yscale("log")
    axes[1].set_xlabel("h (m)")
    axes[1].set_ylabel("alpha (median, truc log)")
    axes[1].set_title("(b) Semi-log (truc y)\n-> thang hang: alpha = c0*exp(c1*h)")
    axes[1].grid(True, which="both", alpha=0.3)

    # (c) log-log: neu thang hang -> alpha ~ h^n (hoac rational n<0)
    if len(h_log) > 1:
        axes[2].plot(h_log, med_log, "o-", color="tab:green")
        axes[2].set_xscale("log")
        axes[2].set_yscale("log")
    axes[2].set_xlabel("h (m, truc log)")
    axes[2].set_ylabel("alpha (median, truc log)")
    axes[2].set_title("(c) Log-log\n-> thang hang: alpha = c0 * h^n "
                       "(n<0 giong dang rational c0+c1/h)")
    axes[2].grid(True, which="both", alpha=0.3)

    fig.suptitle("So sanh 3 truc de chon dang ham alpha(h) "
                  "(duong nao THANG HANG nhat la dang phu hop nhat)")
    fig.tight_layout()
    fig.savefig(output_dir / "alpha_vs_h_shape_comparison.png", dpi=130)
    plt.close(fig)

    # ---- Fig 3: so luong diem moi bin (kiem tra bin nao du/thieu du lieu) ----
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.bar(centers, counts, width=(centers[1] - centers[0]) * 0.9)
    ax.set_xlabel("h (m)")
    ax.set_ylabel("So diem trong bin")
    ax.set_title("So luong diem moi khoang h (kiem tra do tin cay tung vung)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "alpha_bin_counts.png", dpi=130)
    plt.close(fig)

    print(f"\n[Diagnostic] Da luu 3 anh vao: {output_dir}")
    print("  - alpha_vs_h_linear.png            : scatter + median, truc thuong")
    print("  - alpha_vs_h_shape_comparison.png  : 3 truc (linear/semi-log/log-log)")
    print("  - alpha_bin_counts.png             : so diem moi bin (do tin cay)")
    print("\nCach doc ket qua o alpha_vs_h_shape_comparison.png:")
    print("  (a) thang hang nhat  -> alpha(h) = c0 + c1*h            (tuyen tinh, dang dang dung)")
    print("  (b) thang hang nhat  -> alpha(h) = c0 * exp(c1*h)       (ham mu)")
    print("  (c) thang hang nhat  -> alpha(h) = c0 * h^n             (luy thua / rational neu n<0)")
    print("  Neu khong truc nao thang hang ro ret -> can them bac (h^2) hoac them chieu (r_xy).")


# =============================================================================
# NEW: PER-SENSOR OVERLAY -- kiem tra dao dong o duong median GOP co phai
# do TRON LAN nhieu sensor khac alpha(h) nhau hay khong.
# =============================================================================
def bin_alpha_by_h_single_sensor(h_sensor, alpha_sensor, edges):
    """Giong bin_alpha_by_h nhung dung CHUNG 1 bo edges (h) cho moi sensor,
    de cac duong sensor so sanh duoc tren cung 1 truc x."""
    n_bins = len(edges) - 1
    centers = 0.5 * (edges[:-1] + edges[1:])
    med = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)

    bin_idx = np.digitize(h_sensor, edges[1:-1])
    for b in range(n_bins):
        vals = alpha_sensor[bin_idx == b]
        counts[b] = len(vals)
        if len(vals) > 0:
            med[b] = np.median(vals)

    return centers, med, counts


def plot_alpha_per_sensor(h_flat, alpha_flat, sensor_idx_flat, output_dir,
                           n_sensors_overlay=N_SENSORS_OVERLAY, n_bins=N_BINS,
                           seed=0):
    """
    Ve alpha(h) rieng cho tung sensor (khong gop median toan bo 64 sensor
    lai). Neu moi duong sensor rieng le MUOT hon han duong median gop --
    dao dong o cac hinh truoc la do TRON LAN nhieu sensor co alpha(h)
    khac nhau vao chung 1 bin h, khong phai ban chat alpha(h) dao dong that.
    Neu tung duong sensor van dao dong tuong tu -- van de nam o noi khac
    (vi du 1 lop do cu the), khong phai do tron sensor.
    """
    all_sensors = np.unique(sensor_idx_flat)
    rng = np.random.default_rng(seed)
    n_pick = min(n_sensors_overlay, len(all_sensors))
    chosen = rng.choice(all_sensors, size=n_pick, replace=False)
    chosen.sort()

    # Dung CHUNG 1 bo edges (tinh tren toan bo du lieu) cho moi sensor
    # de cac duong so sanh duoc tren cung 1 truc x.
    edges = np.linspace(h_flat.min(), h_flat.max(), n_bins + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    cmap = plt.get_cmap("tab10")
    for i, s in enumerate(chosen):
        mask_s = sensor_idx_flat == s
        h_s = h_flat[mask_s]
        alpha_s = alpha_flat[mask_s]
        centers, med, counts = bin_alpha_by_h_single_sensor(h_s, alpha_s, edges)

        color = cmap(i % 10)
        axes[0].plot(centers, med, "o-", color=color, alpha=0.85,
                     label=f"sensor {s}", linewidth=1.5, markersize=4)
        axes[1].plot(centers, med, "o-", color=color, alpha=0.85,
                     label=f"sensor {s}", linewidth=1.5, markersize=4)

    # Duong median GOP (tat ca sensor) de doi chieu truc tiep
    centers_all, med_all, _, _, _ = bin_alpha_by_h(h_flat, alpha_flat, n_bins=n_bins)
    axes[0].plot(centers_all, med_all, "k--", linewidth=2.5,
                 label="Median GOP (tat ca sensor)")
    axes[1].plot(centers_all, med_all, "k--", linewidth=2.5,
                 label="Median GOP (tat ca sensor)")

    axes[0].set_xlabel("h (m)")
    axes[0].set_ylabel("alpha (median, tung sensor)")
    axes[0].set_title("(a) Truc thuong")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8, ncol=2)

    axes[1].set_xlabel("h (m)")
    axes[1].set_yscale("log")
    axes[1].set_title("(b) Truc log (truc y)")
    axes[1].grid(True, which="both", alpha=0.3)

    fig.suptitle(
        "Alpha(h) theo TUNG sensor rieng le vs. median gop\n"
        "-> tung duong MUOT hon duong gop (net dut den) = dao dong truoc do "
        "la do tron lan nhieu sensor; tung duong VAN dao dong = van de o noi khac"
    )
    fig.tight_layout()
    fig.savefig(output_dir / "alpha_vs_h_per_sensor.png", dpi=130)
    plt.close(fig)

    print(f"\n[Diagnostic] Da luu them: alpha_vs_h_per_sensor.png "
          f"({n_pick} sensor duoc chon ngau nhien: {list(chosen)})")
    print("Cach doc: neu duong tung sensor muot hon nhieu so voi duong gop "
          "(net dut den) -> dao dong o cac hinh truoc la do TRON LAN sensor, "
          "khong phai ban chat alpha(h). Neu tung sensor van dao dong tuong tu "
          "-> nghi ngo lop do cu the (vi du 2 bin it diem bat thuong) thay vi "
          "loi gop median.")


# =============================================================================
# MAIN
# =============================================================================
def main():
    physical_df = load_physical_calib(PHYSICAL_PATH)
    voltage_data, _ = load_voltage_data(VOLTAGE_PATH)
    robot_positions, m_world = load_robot_pose(COORDS_PATH)

    n_samples = min(voltage_data.shape[0], robot_positions.shape[0])
    voltage_data = voltage_data[:n_samples]
    robot_positions = robot_positions[:n_samples]
    m_world = m_world[:n_samples]

    n_sensors = physical_df.shape[0]
    assert n_sensors == voltage_data.shape[1], (
        f"So sensor trong file calib ({n_sensors}) khac so cot dien ap "
        f"({voltage_data.shape[1]})"
    )

    # ---- NEW: quet nguong |g*B| truoc, de chon MIN_GB_MAGNITUDE theo du
    # lieu thay vi doan mo ----
    h_grid, alpha_grid, gB_grid, sensor_idx_grid = build_empirical_alpha_raw(
        physical_df, robot_positions, m_world, voltage_data
    )
    sweep_rows = sweep_gb_threshold(gB_grid, alpha_grid)
    plot_gb_threshold_sweep(sweep_rows, OUTPUT_DIR)

    # Dung nguong MIN_GB_MAGNITUDE hien tai o dau file. Sau khi xem
    # gb_threshold_sweep.png, chinh lai hang so nay cho khop "elbow" roi
    # chay lai.
    h_flat, alpha_flat, sensor_idx_flat = filter_by_gb_threshold(
        h_grid, alpha_grid, gB_grid, sensor_idx_grid, MIN_GB_MAGNITUDE
    )

    plot_alpha_diagnostics(h_flat, alpha_flat, sensor_idx_flat, OUTPUT_DIR)
    plot_alpha_per_sensor(h_flat, alpha_flat, sensor_idx_flat, OUTPUT_DIR)


if __name__ == "__main__":
    main()