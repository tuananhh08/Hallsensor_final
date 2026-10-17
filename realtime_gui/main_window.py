"""Main window: 3D viewer + control panel + data worker."""
from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Deque, Optional

import numpy as np
from PyQt5.QtCore import Qt, QTimer, pyqtSlot
from PyQt5.QtWidgets import QHBoxLayout, QMainWindow, QMessageBox, QSplitter, QWidget

from realtime_gui import config
from realtime_gui.data_sources import CSVReplaySource, SerialSource
from realtime_gui.panel import InfoPanel, SourceKind
from realtime_gui.viewer3d import Viewer3D
from realtime_gui.worker import DataWorker


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-time 5-DoF Capsule Localization")
        self.resize(1200, 720)

        self._viewer = Viewer3D()
        self._panel = InfoPanel()
        self._worker = DataWorker(self)

        split = QSplitter(Qt.Horizontal)
        split.addWidget(self._viewer)
        split.addWidget(self._panel)
        split.setStretchFactor(0, 4)
        split.setStretchFactor(1, 0)
        split.setSizes([900, 280])

        central = QWidget()
        lay = QHBoxLayout(central)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addWidget(split)
        self.setCentralWidget(central)

        self._worker.pose_relayed.connect(self._on_pose)
        self._worker.status_changed.connect(self._panel.set_status)
        self._worker.error_occurred.connect(self._on_error)

        self._panel.start_clicked.connect(self._start)
        self._panel.stop_clicked.connect(self._stop)
        self._panel.reset_trail_clicked.connect(self._reset_trail_and_rewind)
        self._panel.source_kind_changed.connect(self._on_source_kind_changed)
        self._panel.rate_hz_changed.connect(self._on_rate_changed)
        self._panel.serial_params_changed.connect(self._on_serial_params_changed)

        self._fps_times: Deque[float] = deque()
        self._fps_timer = QTimer(self)
        self._fps_timer.setInterval(200)
        self._fps_timer.timeout.connect(self._refresh_fps_label)
        self._fps_timer.start()

        self._running = False
        self._current_csv: Optional[CSVReplaySource] = None
        self._current_serial: Optional[SerialSource] = None

        self._build_initial_source()

    def _build_initial_source(self) -> None:
        try:
            src = CSVReplaySource(
                config.DEFAULT_HELIX_CSV,
                rate_hz=self._panel.rate_hz(),
                loop=False,
                parent=self,
            )
        except Exception as e:
            QMessageBox.warning(self, "CSV", f"Could not load default Helix CSV:\n{e}")
            src = None
        self._worker.detach()
        if src is not None:
            self._worker.attach_source(src)
            self._current_csv = src
        self._panel.set_status("Ready. Press Start.")

    def _start(self) -> None:
        if self._worker.active_source() is None:
            self._switch_source(self._panel.source_kind(), restart=False)
        if self._worker.active_source() is None:
            return
        self._running = True
        self._worker.start()
        self._panel.set_status("Running…")

    def _stop(self) -> None:
        self._running = False
        self._worker.stop()
        self._panel.set_status("Stopped.")

    @pyqtSlot(str)
    def _on_error(self, msg: str) -> None:
        self._panel.set_status(msg)
        QMessageBox.warning(self, "Error", msg)

    @pyqtSlot(object)
    def _on_pose(self, pose: object) -> None:
        self._viewer.update_pose(pose)
        self._panel.update_pose_display(np.asarray(pose))
        now = time.perf_counter()
        self._fps_times.append(now)
        cutoff = now - 1.0
        while self._fps_times and self._fps_times[0] < cutoff:
            self._fps_times.popleft()

    def _refresh_fps_label(self) -> None:
        if len(self._fps_times) < 2:
            return
        dt = self._fps_times[-1] - self._fps_times[0]
        if dt <= 1e-6:
            return
        fps = (len(self._fps_times) - 1) / dt
        self._panel.update_fps(fps)

    def _on_source_kind_changed(self, kind: int) -> None:
        was_running = self._running
        if was_running:
            self._stop()
        self._switch_source(SourceKind(kind), restart=False)
        if was_running:
            self._start()

    def _on_rate_changed(self, hz: float) -> None:
        src = self._worker.active_source()
        if isinstance(src, CSVReplaySource):
            src.set_rate_hz(hz)

    def _reset_trail_and_rewind(self) -> None:
        self._viewer.reset_trail()
        src = self._worker.active_source()
        if not isinstance(src, CSVReplaySource):
            return
        first_pose = src.rewind_to_start()
        if first_pose is None:
            return
        self._viewer.update_pose(first_pose)
        self._panel.update_pose_display(np.asarray(first_pose))
        self._panel.set_status("Trail reset. Replay rewound to start point.")

    def _on_serial_params_changed(self, port: str, baud: int) -> None:
        src = self._worker.active_source()
        if isinstance(src, SerialSource):
            was = self._running
            if was:
                self._stop()
            src.set_port(port)
            src.set_baudrate(baud)
            if was:
                self._start()

    def _switch_source(self, kind: SourceKind, restart: bool = True) -> None:
        self._worker.detach()
        self._current_csv = None
        self._current_serial = None

        try:
            if kind == SourceKind.HELIX_CSV:
                src = CSVReplaySource(
                    config.DEFAULT_HELIX_CSV,
                    rate_hz=self._panel.rate_hz(),
                    loop=False,
                    parent=self,
                )
                self._current_csv = src
            elif kind == SourceKind.TESTRESULT_CSV:
                path = Path(config.DEFAULT_TESTRESULT_CSV)
                if not path.is_file():
                    raise FileNotFoundError(f"Missing {path}. Run Code/test.py first.")
                src = CSVReplaySource(
                    path,
                    rate_hz=self._panel.rate_hz(),
                    loop=False,
                    parent=self,
                )
                self._current_csv = src
            else:
                src = SerialSource(
                    port=self._panel.serial_port(),
                    baudrate=self._panel.serial_baud(),
                    parent=self,
                )
                self._current_serial = src
        except Exception as e:
            QMessageBox.warning(self, "Source", str(e))
            self._build_initial_source()
            return

        self._worker.attach_source(src)
        self._panel.set_status(f"Switched source: {kind.name}")
        if restart and self._running:
            self._worker.start()

    def closeEvent(self, event) -> None:  # noqa: N802
        self._stop()
        self._worker.detach()
        super().closeEvent(event)