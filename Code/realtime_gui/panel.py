"""Side panel: numeric pose, FPS, source selection, rate, serial params, controls."""
from __future__ import annotations

from enum import IntEnum
from typing import Optional

import numpy as np
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class SourceKind(IntEnum):
    HELIX_CSV = 0
    TESTRESULT_CSV = 1
    SERIAL_FT232 = 2


class InfoPanel(QWidget):
    start_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    reset_trail_clicked = pyqtSignal()
    source_kind_changed = pyqtSignal(int)  # SourceKind value
    rate_hz_changed = pyqtSignal(float)
    serial_params_changed = pyqtSignal(str, int)  # port, baud

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(280)

        self._lbl_x = QLabel("—")
        self._lbl_y = QLabel("—")
        self._lbl_z = QLabel("—")
        self._lbl_pitch = QLabel("—")
        self._lbl_yaw = QLabel("—")
        self._lbl_fps = QLabel("—")
        self._lbl_status = QLabel("Idle")
        self._lbl_status.setWordWrap(True)

        pose_box = QGroupBox("Pose (5-DoF)")
        fl = QFormLayout(pose_box)
        fl.addRow("X (mm):", self._lbl_x)
        fl.addRow("Y (mm):", self._lbl_y)
        fl.addRow("Z (mm):", self._lbl_z)
        fl.addRow("Pitch (°):", self._lbl_pitch)
        fl.addRow("Yaw (°):", self._lbl_yaw)
        fl.addRow("FPS (~1s):", self._lbl_fps)

        src_box = QGroupBox("Data source")
        src_layout = QVBoxLayout(src_box)
        self._combo = QComboBox()
        self._combo.addItems(
            [
                "CSV replay (Helix)",
                "CSV replay (testresult)",
                "Serial (FT232)",
            ]
        )
        self._combo.currentIndexChanged.connect(self._on_combo)

        rate_row = QHBoxLayout()
        rate_row.addWidget(QLabel("Rate (Hz):"))
        self._rate = QDoubleSpinBox()
        self._rate.setRange(0.5, 120.0)
        self._rate.setDecimals(1)
        self._rate.setValue(30.0)
        self._rate.valueChanged.connect(lambda v: self.rate_hz_changed.emit(float(v)))
        rate_row.addWidget(self._rate)

        self._port = QLineEdit("COM3")
        self._baud = QSpinBox()
        self._baud.setRange(9600, 3_000_000)
        self._baud.setSingleStep(9600)
        self._baud.setValue(921600)

        ser_form = QFormLayout()
        ser_form.addRow("Port:", self._port)
        ser_form.addRow("Baud:", self._baud)

        src_layout.addWidget(self._combo)
        src_layout.addLayout(rate_row)
        src_layout.addLayout(ser_form)

        self._btn_start = QPushButton("Start")
        self._btn_stop = QPushButton("Stop")
        self._btn_reset = QPushButton("Reset trail")
        self._btn_start.clicked.connect(self.start_clicked.emit)
        self._btn_stop.clicked.connect(self.stop_clicked.emit)
        self._btn_reset.clicked.connect(self.reset_trail_clicked.emit)

        self._port.editingFinished.connect(self._emit_serial_params)
        self._baud.valueChanged.connect(self._emit_serial_params)

        btns = QHBoxLayout()
        btns.addWidget(self._btn_start)
        btns.addWidget(self._btn_stop)
        btns.addWidget(self._btn_reset)

        st_box = QGroupBox("Status")
        st_layout = QVBoxLayout(st_box)
        st_layout.addWidget(self._lbl_status)

        root = QVBoxLayout(self)
        root.addWidget(pose_box)
        root.addWidget(src_box)
        root.addLayout(btns)
        root.addWidget(st_box)
        root.addStretch(1)

        self._apply_source_ui_state()

    def _on_combo(self, index: int) -> None:
        self.source_kind_changed.emit(int(index))
        self._apply_source_ui_state()

    def _apply_source_ui_state(self) -> None:
        serial = self._combo.currentIndex() == SourceKind.SERIAL_FT232
        self._rate.setEnabled(not serial)
        self._port.setEnabled(serial)
        self._baud.setEnabled(serial)

    def _emit_serial_params(self) -> None:
        self.serial_params_changed.emit(self._port.text().strip(), int(self._baud.value()))

    def set_status(self, text: str) -> None:
        self._lbl_status.setText(text)

    def rate_hz(self) -> float:
        return float(self._rate.value())

    def source_kind(self) -> SourceKind:
        return SourceKind(self._combo.currentIndex())

    def serial_port(self) -> str:
        return self._port.text().strip()

    def serial_baud(self) -> int:
        return int(self._baud.value())

    def update_pose_display(self, pose: Optional[np.ndarray]) -> None:
        if pose is None:
            return
        p = np.asarray(pose, dtype=np.float64).reshape(5,)
        x, y, z, pitch, yaw = p
        self._lbl_x.setText(f"{x * 1000.0:.3f}")
        self._lbl_y.setText(f"{y * 1000.0:.3f}")
        self._lbl_z.setText(f"{z * 1000.0:.3f}")
        self._lbl_pitch.setText(f"{pitch:.3f}")
        self._lbl_yaw.setText(f"{yaw:.3f}")

    def update_fps(self, fps: float) -> None:
        self._lbl_fps.setText(f"{fps:.1f}")
