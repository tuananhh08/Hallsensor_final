"""OpenGL 3D view: ROI box, sensor map, vector-based capsule, trail."""
from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np
import pyqtgraph.opengl as gl
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QColor, QVector3D
from pyqtgraph.opengl import GLViewWidget

from realtime_gui import config


def _roi_wireframe_edges(
    x0: float, x1: float, y0: float, y1: float, z0: float, z1: float
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """12 edges as pairs of 3D points."""
    c = [
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1),
    ]
    idx = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for a, b in idx:
        pa = np.array(c[a], dtype=np.float64)
        pb = np.array(c[b], dtype=np.float64)
        out.append((pa, pb))
    return out


class Viewer3D(GLViewWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackgroundColor(QColor(30, 30, 35))

        x0, x1, y0, y1, z0, z1 = config.roi_bounds_padded()
        self._roi_bounds = (x0, x1, y0, y1, z0, z1)
        cx, cy, cz = config.roi_center()
        self._origin = np.array([cx, cy, cz], dtype=np.float64)

        # ROI wireframe
        edges = _roi_wireframe_edges(x0, x1, y0, y1, z0, z1)
        self._roi_lines: List[gl.GLLinePlotItem] = []
        for pa, pb in edges:
            item = gl.GLLinePlotItem(
                pos=np.vstack([pa, pb]),
                color=(0.75, 0.75, 0.8, 0.9),
                width=2,
                antialias=True,
            )
            self.addItem(item)
            self._roi_lines.append(item)

        sensor_points = config.sensor_map_points()
        self._sensor_points = gl.GLScatterPlotItem(
            pos=sensor_points,
            color=(0.2, 0.9, 0.4, 0.95),
            size=8,
            pxMode=True,
        )
        self.addItem(self._sensor_points)

        # Floor grid at z = z0
        grid = gl.GLGridItem()
        grid.setSize(x=abs(x1 - x0) * 1.1, y=abs(y1 - y0) * 1.1)
        grid.setSpacing(x=0.01, y=0.01)
        grid.translate((x0 + x1) / 2.0, (y0 + y1) / 2.0, z0)
        self.addItem(grid)

        self._capsule_len = config.CAPSULE_LENGTH_MM / 1000.0
        self._capsule_radius = config.CAPSULE_CYLINDER_RADIUS_MM / 1000.0
        self._capsule_body = gl.GLLinePlotItem(
            pos=np.zeros((2, 3), dtype=np.float64),
            color=(1.0, 0.55, 0.1, 0.95),
            width=6,
            antialias=True,
        )
        self.addItem(self._capsule_body)

        self._capsule_ring_front = gl.GLLinePlotItem(
            pos=np.zeros((2, 3), dtype=np.float64),
            color=(1.0, 0.75, 0.25, 0.8),
            width=2,
            antialias=True,
            mode="line_strip",
        )
        self._capsule_ring_back = gl.GLLinePlotItem(
            pos=np.zeros((2, 3), dtype=np.float64),
            color=(0.95, 0.45, 0.1, 0.6),
            width=2,
            antialias=True,
            mode="line_strip",
        )
        self.addItem(self._capsule_ring_front)
        self.addItem(self._capsule_ring_back)

        self._arrow = gl.GLLinePlotItem(
            pos=np.zeros((2, 3), dtype=np.float64),
            color=(1.0, 0.9, 0.2, 1.0),
            width=4,
            antialias=True,
        )
        self.addItem(self._arrow)

        self._trail: Deque[np.ndarray] = deque(maxlen=config.TRAIL_LENGTH)
        self._trail_line = gl.GLLinePlotItem(
            pos=np.zeros((1, 3), dtype=np.float64),
            color=(0.2, 0.85, 1.0, 0.75),
            width=2,
            antialias=True,
        )
        self.addItem(self._trail_line)

        # Camera — look at ROI center
        self.setCameraPosition(
            pos=QVector3D(float(cx), float(cy), float(cz)),
            distance=0.28,
            elevation=28,
            azimuth=50,
        )

        self._last_pose: Optional[np.ndarray] = None

    def reset_trail(self) -> None:
        self._trail.clear()
        self._trail_line.setData(pos=np.zeros((1, 3), dtype=np.float64))

    @pyqtSlot(object)
    def update_pose(self, pose: object) -> None:
        p = np.asarray(pose, dtype=np.float64).reshape(5,)
        self._last_pose = p
        x, y, z, pitch_deg, yaw_deg = float(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])

        d = config.heading_direction(pitch_deg, yaw_deg)
        center = np.array([x, y, z], dtype=np.float64)
        p_back = center - d * (self._capsule_len / 2.0)
        p_front = center + d * (self._capsule_len / 2.0)
        self._capsule_body.setData(pos=np.vstack([p_back, p_front]))

        ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(np.dot(ref, d)) > 0.95:
            ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        u = np.cross(d, ref)
        u /= np.linalg.norm(u)
        v = np.cross(d, u)
        v /= np.linalg.norm(v)
        t = np.linspace(0.0, 2.0 * np.pi, 25)
        ring_offset = (np.cos(t)[:, None] * u[None, :] + np.sin(t)[:, None] * v[None, :]) * self._capsule_radius
        self._capsule_ring_front.setData(pos=p_front[None, :] + ring_offset)
        self._capsule_ring_back.setData(pos=p_back[None, :] + ring_offset)

        arrow_start = p_front
        arrow_tip = arrow_start + d * (config.ARROW_LENGTH_MM / 1000.0)
        self._arrow.setData(pos=np.vstack([arrow_start, arrow_tip]))

        self._trail.append(np.array([x, y, z], dtype=np.float64))
        if len(self._trail) >= 2:
            self._trail_line.setData(pos=np.array(self._trail, dtype=np.float64))
        else:
            self._trail_line.setData(pos=np.array([[x, y, z]], dtype=np.float64))

    def last_pose(self) -> Optional[np.ndarray]:
        return None if self._last_pose is None else self._last_pose.copy()
