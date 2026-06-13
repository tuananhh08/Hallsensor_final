from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np
import pyqtgraph.opengl as gl
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QColor, QVector3D
from pyqtgraph.opengl import GLViewWidget

from realtime_gui import config

# Cylinder resolution — số mặt xung quanh thân trụ (càng cao càng tròn)
_N_SIDES = 24


# ---------------------------------------------------------------------------
# Mesh helpers
# ---------------------------------------------------------------------------

def _local_frame(d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Trả về 2 vector u, v vuông góc với d (và với nhau)."""
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(np.dot(ref, d)) > 0.95:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u = np.cross(d, ref);  u /= np.linalg.norm(u)
    v = np.cross(d, u);    v /= np.linalg.norm(v)
    return u, v


def _make_cylinder_mesh(
    p_back: np.ndarray,
    p_front: np.ndarray,
    radius: float,
    n: int = _N_SIDES,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hình trụ kín (thân + 2 mặt đáy).

    Vertex layout:
        0   .. n-1   : ring tại p_back
        n   .. 2n-1  : ring tại p_front
        2n           : tâm đáy sau  (p_back)
        2n+1         : tâm đáy trước (p_front)
    """
    d = p_front - p_back
    length = float(np.linalg.norm(d))
    d = d / max(length, 1e-12)

    u, v = _local_frame(d)

    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    offsets = radius * (np.cos(t)[:, None] * u + np.sin(t)[:, None] * v)  # (n,3)

    ring_back  = p_back[None, :]  + offsets   # (n,3)
    ring_front = p_front[None, :] + offsets   # (n,3)

    verts = np.vstack([
        ring_back,                              # 0..n-1
        ring_front,                             # n..2n-1
        p_back[None, :],                        # 2n
        p_front[None, :],                       # 2n+1
    ]).astype(np.float32)

    faces: List[List[int]] = []
    cb = 2 * n       # tâm đáy sau
    cf = 2 * n + 1   # tâm đáy trước

    for i in range(n):
        j = (i + 1) % n
        # Thân trụ — 2 tam giác mỗi quad
        faces.append([i,     j,     n + i])
        faces.append([j,     n + j, n + i])
        # Đáy sau (pháp tuyến hướng ra ngoài → winding ngược)
        faces.append([cb, j, i])
        # Đáy trước
        faces.append([cf, n + i, n + j])

    return verts, np.array(faces, dtype=np.uint32)


def _make_cone_mesh(
    tip: np.ndarray,
    base_center: np.ndarray,
    radius: float,
    n: int = _N_SIDES,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hình nón kín (mặt bên + đáy) — dùng làm đầu mũi tên.

    Vertex layout:
        0 .. n-1  : ring đáy
        n         : đỉnh nhọn (tip)
        n+1       : tâm đáy (base_center)
    """
    d = base_center - tip
    length = float(np.linalg.norm(d))
    d = d / max(length, 1e-12)

    u, v = _local_frame(d)

    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    ring = base_center[None, :] + radius * (
        np.cos(t)[:, None] * u + np.sin(t)[:, None] * v
    )

    verts = np.vstack([
        ring,                   # 0..n-1
        tip[None, :],           # n
        base_center[None, :],   # n+1
    ]).astype(np.float32)

    tip_idx  = n
    base_idx = n + 1
    faces: List[List[int]] = []
    for i in range(n):
        j = (i + 1) % n
        faces.append([tip_idx,  i,        j       ])   # mặt bên
        faces.append([base_idx, j,        i       ])   # đáy
    return verts, np.array(faces, dtype=np.uint32)


# ---------------------------------------------------------------------------
# ROI helpers
# ---------------------------------------------------------------------------

def _roi_wireframe_edges(
    x0: float, x1: float, y0: float, y1: float, z0: float, z1: float
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """12 cạnh của hộp ROI dưới dạng cặp điểm 3D."""
    c = [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
    ]
    idx = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for a, b in idx:
        out.append((np.array(c[a], dtype=np.float64), np.array(c[b], dtype=np.float64)))
    return out


# ---------------------------------------------------------------------------
# Main viewer widget
# ---------------------------------------------------------------------------

class Viewer3D(GLViewWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackgroundColor(QColor(30, 30, 35))

        x0, x1, y0, y1, z0, z1 = config.roi_bounds_padded()
        self._roi_bounds = (x0, x1, y0, y1, z0, z1)
        cx, cy, cz = config.roi_center()
        self._origin = np.array([cx, cy, cz], dtype=np.float64)

        # --- ROI wireframe ---
        for pa, pb in _roi_wireframe_edges(x0, x1, y0, y1, z0, z1):
            self.addItem(gl.GLLinePlotItem(
                pos=np.vstack([pa, pb]),
                color=(0.75, 0.75, 0.8, 0.9),
                width=3, antialias=True,
            ))

        # --- Sensor scatter ---
        sensor_points = config.sensor_map_points()
        self._sensor_points = gl.GLScatterPlotItem(
            pos=sensor_points,
            color=(0.2, 0.9, 0.4, 0.95),
            size=9.5, pxMode=True,
        )
        self.addItem(self._sensor_points)

        # --- Floor grid ---
        grid = gl.GLGridItem()
        grid.setSize(x=abs(x1 - x0) * 1.1, y=abs(y1 - y0) * 1.1)
        grid.setSpacing(x=0.01, y=0.01)
        grid.translate((x0 + x1) / 2.0, (y0 + y1) / 2.0, z0)
        self.addItem(grid)

        # --- Kích thước viên nang & mũi tên ---
        self._capsule_len    = config.CAPSULE_LENGTH_MM    / 1000.0
        self._capsule_radius = config.CAPSULE_CYLINDER_RADIUS_MM / 1000.0
        self._arrow_len      = config.ARROW_LENGTH_MM      / 1000.0
        # Nón mũi tên: 
        self._cone_radius = self._capsule_radius * 1.3
        self._cone_len    = self._arrow_len * 1.05

        # --- dummy mesh dùng lúc khởi tạo ---
        _dv = np.zeros((4, 3), dtype=np.float32)
        _df = np.zeros((2, 3), dtype=np.uint32)

        # ---- Viên nang: hình trụ kín (GLMeshItem) ----
        self._capsule_mesh = gl.GLMeshItem(
            vertexes=_dv, faces=_df,
            color=(1.0, 0.55, 0.1, 0.92),   # cam đậm
            smooth=True,
            drawEdges=False,
            glOptions="translucent",
        )
        self.addItem(self._capsule_mesh)

        # ---- Mũi tên: đường trục + nón ----
        self._arrow_shaft = gl.GLLinePlotItem(
            pos=np.zeros((2, 3), dtype=np.float64),
            color=(1.0, 0.9, 0.2, 1.0),
            width=3, antialias=True,
        )
        self.addItem(self._arrow_shaft)

        self._arrow_cone = gl.GLMeshItem(
            vertexes=_dv, faces=_df,
            color=(1.0, 0.92, 0.1, 1.0),    # vàng sáng
            smooth=True,
            drawEdges=False,
            glOptions="opaque",
        )
        self.addItem(self._arrow_cone)

        # --- Trail ---
        self._trail: Deque[np.ndarray] = deque(maxlen=config.TRAIL_LENGTH)
        self._trail_line = gl.GLLinePlotItem(
            pos=np.zeros((1, 3), dtype=np.float64),
            color=(0.2, 0.85, 1.0, 0.75),
            width=4, antialias=True,
        )
        self.addItem(self._trail_line)

        # --- Camera nhìn vào tâm ROI ---
        self.setCameraPosition(
            pos=QVector3D(float(cx), float(cy), float(cz)),
            distance=0.28, elevation=28, azimuth=50,
        )

        self._last_pose: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def reset_trail(self) -> None:
        self._trail.clear()
        self._trail_line.setData(pos=np.zeros((1, 3), dtype=np.float64))

    # ------------------------------------------------------------------
    @pyqtSlot(object)
    def update_pose(self, pose: object) -> None:
        p = np.asarray(pose, dtype=np.float64).reshape(5,)
        self._last_pose = p

        x, y, z, pitch_deg, yaw_deg = (
            float(p[0]), float(p[1]), float(p[2]),
            float(p[3]), float(p[4]),
        )

        d = config.heading_direction(pitch_deg, yaw_deg)   # unit vector

        center  = np.array([x, y, z], dtype=np.float64)
        p_back  = center - d * (self._capsule_len / 2.0)
        p_front = center + d * (self._capsule_len / 2.0)

        # ---- Cập nhật hình trụ kín ----
        cyl_v, cyl_f = _make_cylinder_mesh(p_back, p_front, self._capsule_radius)
        self._capsule_mesh.setMeshData(vertexes=cyl_v, faces=cyl_f)

        # ---- Cập nhật mũi tên ----
        shaft_start = p_front
        cone_base   = shaft_start + d * (self._arrow_len - self._cone_len)
        arrow_tip   = shaft_start + d * self._arrow_len

        self._arrow_shaft.setData(pos=np.vstack([shaft_start, cone_base]))

        cone_v, cone_f = _make_cone_mesh(arrow_tip, cone_base, self._cone_radius)
        self._arrow_cone.setMeshData(vertexes=cone_v, faces=cone_f)

        # ---- Trail ----
        self._trail.append(np.array([x, y, z], dtype=np.float64))
        if len(self._trail) >= 2:
            self._trail_line.setData(pos=np.array(self._trail, dtype=np.float64))
        else:
            self._trail_line.setData(pos=np.array([[x, y, z]], dtype=np.float64))

    # ------------------------------------------------------------------
    def last_pose(self) -> Optional[np.ndarray]:
        return None if self._last_pose is None else self._last_pose.copy()