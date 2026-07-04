"""Convenience launcher: ``python run.py`` starts the desktop application.

This is also the PyInstaller entry script.
"""

from __future__ import annotations

import sys

from app.main import main

if __name__ == "__main__":
    sys.exit(main())
