"""
visualize.py — 3D Coordinate Visualizer
Hiển thị vị trí tương quan giữa các điểm trong không gian 3D từ các file CSV.

Màu sắc:
  - sensor_position_calib.csv : xanh lá  (#2ecc71)
  - grid_points.csv           : xanh dương (#3498db)
  - helix_points.csv          : đỏ        (#e74c3c)
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ─────────────────────────────────────────────
# CẤU HÌNH
# ─────────────────────────────────────────────
# BASE_DIR = r"D:\Downloads\Hallsensor_final\Data set 18.6" #WINDOWS
BASE_DIR = Path(r"/Users/tuananhnguyen/Downloads/Hallsensor_final/Data set 18.6") #MAC


FILES = {
    "grid": {
        "filename": "Hall_sensor_positions.csv",
        "color":    "#14b055",
        "marker":   "o",
        "size":     30,
        "alpha":    0.9,
        "label":    "grid points",
        "zorder":   5,
    },
    "helix": {
        "filename": "Helix_points_coordinates.csv",
        "color":    "#EC7438",
        "marker":   "s",
        "size":     30,
        "alpha":    0.9,
        "label":    "Helix Points",
        "zorder":   2,
    },
    "grid_points": {
        "filename": "grid_points_coordinates.csv",
        "color":    "#3498db",
        "marker":   "d",
        "size":     35,
        "alpha":    0.7,
        "label":    "Grid Points",
        "zorder":   4,
     }
}

# ─────────────────────────────────────────────
# HÀM ĐỌC CSV
# ─────────────────────────────────────────────
def load_csv(filepath: str) -> pd.DataFrame | None:
    """
    Đọc file CSV và trả về DataFrame với các cột x, y, z.
    Tự động nhận diện tên cột (x/X/col0, y/Y/col1, z/Z/col2).
    """
    if not os.path.isfile(filepath):
        print(f"  [WARN] Không tìm thấy file: {filepath}")
        return None

    df = pd.read_csv(filepath)

    # Chuẩn hoá tên cột → chữ thường
    df.columns = [c.strip().lower() for c in df.columns]

    # Nếu không có header 'x','y','z' → gán theo thứ tự cột
    col_map = {}
    for axis in ("x", "y", "z"):
        if axis in df.columns:
            col_map[axis] = axis
        else:
            # Tìm cột chứa tên axis (e.g. 'pos_x', 'x_m', ...)
            candidates = [c for c in df.columns if c.endswith(axis) or c.startswith(axis)]
            if candidates:
                col_map[axis] = candidates[0]

    # Fallback: dùng cột theo thứ tự nếu ít nhất 3 cột
    if len(col_map) < 3 and df.shape[1] >= 3:
        print(f"  [INFO] Không tìm thấy header x/y/z trong '{os.path.basename(filepath)}' "
              f"→ dùng 3 cột đầu tiên.")
        df.columns = list(df.columns[:3]) + list(df.columns[3:])
        df = df.rename(columns={df.columns[0]: "x",
                                 df.columns[1]: "y",
                                 df.columns[2]: "z"})
        col_map = {"x": "x", "y": "y", "z": "z"}

    if len(col_map) < 3:
        print(f"  [ERROR] Không thể xác định cột x/y/z trong '{os.path.basename(filepath)}'")
        return None

    result = df[[col_map["x"], col_map["y"], col_map["z"]]].copy()
    result.columns = ["x", "y", "z"]
    result = result.dropna().astype(float)
    print(f"  [OK]   {os.path.basename(filepath):35s} — {len(result):5d} điểm")
    return result


# ─────────────────────────────────────────────
# VẼ
# ─────────────────────────────────────────────
def plot_3d(datasets: dict):
    fig = plt.figure(figsize=(12, 9))
    fig.patch.set_facecolor("#1a1a2e")

    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#16213e")

    # Màu chữ & lưới
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("#334466")
    ax.tick_params(colors="#aabbcc", labelsize=8)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.label.set_color("#aabbcc")
        axis.label.set_fontsize(10)

    all_coords = []

    for key, cfg in FILES.items():
        df = datasets.get(key)
        if df is None or df.empty:
            continue

        ax.scatter(
            df["x"], df["y"], df["z"],
            c=cfg["color"],
            marker=cfg["marker"],
            s=cfg["size"],
            alpha=cfg["alpha"],
            label=cfg["label"],
            zorder=cfg["zorder"],
            edgecolors="none" if cfg["marker"] != "o" else "none",
            depthshade=True,
        )
        all_coords.append(df[["x", "y", "z"]].values)

    # Căn chỉnh trục đều nhau (equal aspect ratio)
    if all_coords:
        combined = np.vstack(all_coords)
        mins = combined.min(axis=0)
        maxs = combined.max(axis=0)
        mid  = (mins + maxs) / 2
        half = (maxs - mins).max() / 2 * 1.15  # padding 15 %

        ax.set_xlim(mid[0] - half, mid[0] + half)
        ax.set_ylim(mid[1] - half, mid[1] + half)
        ax.set_zlim(mid[2] - half, mid[2] + half)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("3D Coordinate Visualizer", color="white", fontsize=14, pad=14)

    legend = ax.legend(
        loc="upper left",
        fontsize=9,
        facecolor="#0f3460",
        edgecolor="#334466",
        labelcolor="white",
        markerscale=1.4,
    )

    # Thêm thông tin số điểm vào góc dưới
    info_lines = []
    for key, cfg in FILES.items():
        df = datasets.get(key)
        n = len(df) if df is not None else 0
        info_lines.append(f"{cfg['label']}: {n} pts")
    fig.text(0.01, 0.01, "\n".join(info_lines),
             color="#778899", fontsize=8, va="bottom",
             fontfamily="monospace")

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  3D Coordinate Visualizer")
    print(f"  Thư mục dữ liệu: {BASE_DIR}")
    print("=" * 55)

    datasets = {}
    for key, cfg in FILES.items():
        path = os.path.join(BASE_DIR, cfg["filename"])
        datasets[key] = load_csv(path)

    loaded = sum(1 for v in datasets.values() if v is not None and not v.empty)
    if loaded == 0:
        print("\n[ERROR] Không tải được bất kỳ file nào. Kiểm tra lại BASE_DIR và tên file.")
        sys.exit(1)

    print(f"\n→ Đã tải {loaded}/{len(FILES)} file. Đang vẽ...\n")
    plot_3d(datasets)


if __name__ == "__main__":
    main()