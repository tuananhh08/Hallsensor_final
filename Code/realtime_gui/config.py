"""
ROI and visualization constants.

ROI follows the physical layout: center of sensor map + fixed width/depth/height.
If `Data set/Coordinate/Grid_points_coordinates.csv` exists (relative to repo root),
sensor_center is the mean of (x,y,z) min/max from that file; otherwise a default
center is used.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

# --- Physical ROI (meters) ---
ROI_WIDTH_M = 0.12   # 12 cm
ROI_DEPTH_M = 0.12   # 12 cm
ROI_HEIGHT_M = 0.075  # 7.5 cm
Z_OFFSET_ABOVE_CENTER_M = 0.01  # z_min = sensor_center_z + this

# Padding around ROI box for nicer framing (meters)
ROI_PADDING_M = 0.005  # 5 mm

# --- Visualization ---
TRAIL_LENGTH = 200
DEFAULT_RATE_HZ = 10
CAPSULE_LENGTH_MM = 20
CAPSULE_SPHERE_RADIUS_MM = 1.5

# --- Default data paths (relative to this file's parent: Code/) ---
_CODE_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _CODE_DIR.parent

DEFAULT_HELIX_CSV = _REPO_ROOT / "Data set" / "Coordinate" / "Helix_points_coordinates.csv"
DEFAULT_GRID_CSV = _REPO_ROOT / "Data set" / "Coordinate" / "Grid_points_coordinates.csv"
DEFAULT_TESTRESULT_CSV = _CODE_DIR / "ckpt" / "testresult.csv"


def _default_sensor_center() -> Tuple[float, float, float]:
    return (0.241, 0.684, -0.0535)


def compute_sensor_center_from_grid_csv(grid_csv: Path | None = None) -> Tuple[float, float, float]:
    """Mean of per-axis min/max from grid CSV → center of nominal workspace."""
    path = grid_csv or DEFAULT_GRID_CSV
    if not path.is_file():
        return _default_sensor_center()
    df = pd.read_csv(path)
    for col in ("x", "y", "z"):
        if col not in df.columns:
            return _default_sensor_center()
    cx = float((df["x"].min() + df["x"].max()) / 2.0)
    cy = float((df["y"].min() + df["y"].max()) / 2.0)
    cz = float((df["z"].min() + df["z"].max()) / 2.0)
    return (cx, cy, cz)


def compute_roi_bounds(
    sensor_center: Tuple[float, float, float] | None = None,
) -> Tuple[float, float, float, float, float, float]:
    """
    Returns (x_min, x_max, y_min, y_max, z_min, z_max) in meters.

    x/y from center ± half width/depth; z from plan: z_min = cz + offset, z_max = cz + height.
    """
    if sensor_center is None:
        sensor_center = compute_sensor_center_from_grid_csv()
    cx, cy, cz = sensor_center
    hw = ROI_WIDTH_M / 2.0
    hd = ROI_DEPTH_M / 2.0
    x_min = cx - hw
    x_max = cx + hw
    y_min = cy - hd
    y_max = cy + hd
    z_min = cz + Z_OFFSET_ABOVE_CENTER_M
    z_max = cz + ROI_HEIGHT_M
    return (x_min, x_max, y_min, y_max, z_min, z_max)


def roi_bounds_padded() -> Tuple[float, float, float, float, float, float]:
    """ROI bounds expanded by ROI_PADDING_M on each axis."""
    x0, x1, y0, y1, z0, z1 = compute_roi_bounds()
    p = ROI_PADDING_M
    return (x0 - p, x1 + p, y0 - p, y1 + p, z0 - p, z1 + p)


def roi_center() -> Tuple[float, float, float]:
    x0, x1, y0, y1, z0, z1 = compute_roi_bounds()
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0, (z0 + z1) / 2.0)


def heading_direction(pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """
    Unit direction from pitch/yaw (degrees), same convention as visualization:
    dx = cos(yaw_rad) * cos(pitch_rad)
    dy = sin(yaw_rad) * cos(pitch_rad)
    dz = sin(pitch_rad)
    """
    pr = np.deg2rad(pitch_deg)
    yr = np.deg2rad(yaw_deg)
    dx = np.cos(yr) * np.cos(pr)
    dy = np.sin(yr) * np.cos(pr)
    dz = np.sin(pr)
    v = np.array([dx, dy, dz], dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return v / n
