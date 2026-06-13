"""
Relays pose updates from the active `BaseSource` to the UI.

`SerialSource` already performs blocking I/O in a `threading.Thread`; CSV replay
uses `QTimer` on the GUI thread. This `QObject` keeps a single connection point
for `MainWindow` (start/stop/switch) without blocking the render loop.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

from realtime_gui.data_sources import BaseSource


class DataWorker(QObject):
    """Attach one `BaseSource` at a time; forwards `pose_ready` as `pose_relayed`."""

    pose_relayed = pyqtSignal(object)  # np.ndarray (5,)
    status_changed = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._source: Optional[BaseSource] = None

    def attach_source(self, source: Optional[BaseSource]) -> None:
        self.detach()
        if source is None:
            return
        self._source = source
        source.pose_ready.connect(self._on_pose)
        source.status_changed.connect(self.status_changed.emit)
        source.error_occurred.connect(self.error_occurred.emit)

    def detach(self) -> None:
        if self._source is None:
            return
        try:
            self._source.stop()
        except Exception:
            pass
        try:
            self._source.pose_ready.disconnect(self._on_pose)
        except TypeError:
            pass
        try:
            self._source.status_changed.disconnect(self.status_changed.emit)
        except TypeError:
            pass
        try:
            self._source.error_occurred.disconnect(self.error_occurred.emit)
        except TypeError:
            pass
        self._source = None

    def active_source(self) -> Optional[BaseSource]:
        return self._source

    def start(self) -> None:
        if self._source is not None:
            self._source.start()

    def stop(self) -> None:
        if self._source is not None:
            self._source.stop()

    def _on_pose(self, pose: object) -> None:
        self.pose_relayed.emit(pose)
