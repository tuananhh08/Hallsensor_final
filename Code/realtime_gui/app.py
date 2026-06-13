"""Entry: `cd Code` then `python -m realtime_gui.app`."""
from __future__ import annotations

import sys

from PyQt5.QtWidgets import QApplication

from realtime_gui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
