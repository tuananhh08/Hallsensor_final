"""
Data sources: CSV replay (Helix / testresult) and UART packet reader (FT232).

Packet format (STM32 → PC, little-endian):
  [0]     sync byte 0xAA
  [1:21]  five float32: x, y, z, pitch_deg, yaw_deg
  [21]    XOR checksum over bytes [1:21] (20 bytes)
"""
from __future__ import annotations

import struct
import threading
from abc import abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PyQt5.QtCore import QObject, QTimer, pyqtSignal

SYNC_BYTE = 0xAA
PACKET_SIZE = 22  # 1 + 20 + 1


class BaseSource(QObject):
    """Emits pose_ready with shape (5,) float: x, y, z (m), pitch_deg, yaw_deg."""

    pose_ready = pyqtSignal(object)  # np.ndarray (5,)
    status_changed = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._running = False

    @abstractmethod
    def start(self) -> None:
        ...

    @abstractmethod
    def stop(self) -> None:
        ...

    def is_running(self) -> bool:
        return self._running


class CSVReplaySource(BaseSource):
    """Replay rows from CSV at fixed rate using QTimer (GUI thread)."""

    def __init__(
        self,
        csv_path: Path | str,
        rate_hz: float = 30.0,
        loop: bool = True,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self._path = Path(csv_path)
        self._rate_hz = float(rate_hz)
        self._loop = loop
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._poses: np.ndarray = np.zeros((0, 5), dtype=np.float64)
        self._index = 0
        self._load_csv()

    def _load_csv(self) -> None:
        if not self._path.is_file():
            raise FileNotFoundError(f"CSV not found: {self._path}")
        df = pd.read_csv(self._path)
        cols_lower = {c.lower(): c for c in df.columns}

        # Helix / Grid style: x,y,z,pitch,yaw
        if all(k in cols_lower for k in ("x", "y", "z", "pitch", "yaw")):
            x = df[cols_lower["x"]].astype(float)
            y = df[cols_lower["y"]].astype(float)
            z = df[cols_lower["z"]].astype(float)
            pitch = df[cols_lower["pitch"]].astype(float)
            yaw = df[cols_lower["yaw"]].astype(float)
            self._poses = np.column_stack([x, y, z, pitch, yaw]).astype(np.float64)
            self.status_changed.emit(f"Loaded Helix-style CSV: {len(self._poses)} rows")
            return

        # testresult.csv from Code/test.py
        pred_cols = ["Pred_X", "Pred_Y", "Pred_Z", "Pred_alpha_deg", "Pred_beta_deg"]
        if all(c in df.columns for c in pred_cols):
            self._poses = df[pred_cols].to_numpy(dtype=np.float64)
            self.status_changed.emit(f"Loaded testresult CSV: {len(self._poses)} rows")
            return

        raise ValueError(
            f"Unrecognized CSV columns in {self._path}. "
            "Need x,y,z,pitch,yaw or Pred_X,Pred_Y,Pred_Z,Pred_alpha_deg,Pred_beta_deg"
        )

    def set_rate_hz(self, rate_hz: float) -> None:
        self._rate_hz = max(0.1, float(rate_hz))
        if self._running:
            self._timer.setInterval(int(round(1000.0 / self._rate_hz)))

    def set_csv_path(self, csv_path: Path | str) -> None:
        was = self._running
        self.stop()
        self._path = Path(csv_path)
        self._index = 0
        self._load_csv()
        if was:
            self.start()

    def start(self) -> None:
        if self._running or len(self._poses) == 0:
            return
        self._running = True
        self._timer.start(int(round(1000.0 / self._rate_hz)))
        self.status_changed.emit(f"CSV replay started @ {self._rate_hz:.1f} Hz")

    def stop(self) -> None:
        self._timer.stop()
        self._running = False
        self.status_changed.emit("CSV replay stopped")

    def _tick(self) -> None:
        if len(self._poses) == 0:
            return
        row = self._poses[self._index]
        self.pose_ready.emit(np.asarray(row, dtype=np.float64))
        self._index += 1
        if self._index >= len(self._poses):
            if self._loop:
                self._index = 0
            else:
                self.stop()


def _xor_checksum(data: bytes) -> int:
    c = 0
    for b in data:
        c ^= b
    return c & 0xFF


class SerialSource(BaseSource):
    """
    Read fixed 22-byte packets from UART (FT232). Runs blocking I/O in a daemon thread;
    emits pose_ready on the Qt thread via signals (thread-safe).
    """

    def __init__(
        self,
        port: str = "COM3",
        baudrate: int = 921600,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self._port = port
        self._baudrate = baudrate
        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()

    def set_port(self, port: str) -> None:
        self._port = port

    def set_baudrate(self, baudrate: int) -> None:
        self._baudrate = int(baudrate)

    def start(self) -> None:
        if self._running:
            return
        self._stop_evt.clear()
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, name="SerialSource", daemon=True)
        self._thread.start()
        self.status_changed.emit(f"Serial opening {self._port} @ {self._baudrate} ...")

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._running = False
        self.status_changed.emit("Serial stopped")

    def _run_loop(self) -> None:
        try:
            import serial  # type: ignore
        except ImportError as e:
            self.error_occurred.emit("pyserial not installed: pip install pyserial")
            self._running = False
            return

        try:
            ser = serial.Serial(self._port, self._baudrate, timeout=0.1)
        except Exception as e:
            self.error_occurred.emit(f"Serial open failed: {e}")
            self._running = False
            return

        self.status_changed.emit(f"Serial connected {self._port}")
        buf = bytearray()
        try:
            while not self._stop_evt.is_set():
                chunk = ser.read(256)
                if not chunk:
                    continue
                buf.extend(chunk)
                while len(buf) >= PACKET_SIZE:
                    # find sync
                    try:
                        i = buf.index(SYNC_BYTE)
                    except ValueError:
                        buf.clear()
                        break
                    if i > 0:
                        del buf[:i]
                    if len(buf) < PACKET_SIZE:
                        break
                    packet = bytes(buf[:PACKET_SIZE])
                    payload = packet[1:21]
                    chk = packet[21]
                    if _xor_checksum(payload) != chk:
                        del buf[0:1]
                        continue
                    floats = struct.unpack("<5f", payload)
                    pose = np.array(floats, dtype=np.float64)
                    self.pose_ready.emit(pose)
                    del buf[:PACKET_SIZE]
        finally:
            try:
                ser.close()
            except Exception:
                pass
        self._running = False
