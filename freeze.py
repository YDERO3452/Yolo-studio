"""Paths for PyInstaller builds and normal `python main.py` runs.

Copyright (C) 2024–2026 YDERO3452
SPDX-License-Identifier: GPL-3.0-only
"""

import sys
from pathlib import Path


def _is_frozen() -> bool:
    """Return True if running as a PyInstaller-bundled executable."""
    return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")


def get_app_root() -> Path:
    """Return the application root directory.

    In development: the directory containing this file (project root).
    In frozen mode:  the directory containing the executable.
    """
    if _is_frozen():
        return Path(sys.executable).parent
    return Path(__file__).parent


def get_resource_path(relative_path: str) -> Path:
    """Resolve a path to a bundled resource file.

    In development: relative to the project root.
    In frozen mode: relative to sys._MEIPASS (PyInstaller temp dir).

    Args:
        relative_path: Relative path from the project root, e.g.
                       "resources/icons/add.svg" or "configs/default.yaml".
    """
    if _is_frozen():
        return Path(sys._MEIPASS) / relative_path
    return get_app_root() / relative_path


def get_writable_dir() -> Path:
    """Return a user-writable directory for config, cache, and data.

    Frozen mode:  %APPDATA%/YoloStudio on Windows,
                  ~/.local/share/YoloStudio on Linux,
                  ~/Library/Application Support/YoloStudio on macOS.
    Development:  project root (convenient for debugging).
    """
    if _is_frozen():
        if sys.platform == "win32":
            base = Path.home() / "AppData" / "Roaming"
        elif sys.platform == "darwin":
            base = Path.home() / "Library" / "Application Support"
        else:
            # Linux / freedesktop
            base = Path.home() / ".local" / "share"
        return base / "YoloStudio"
    return get_app_root()
