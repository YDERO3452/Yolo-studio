"""Capture YOLO Studio workspaces for visual regression review."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_OPENGL", "software")
os.environ.setdefault("QT_SCALE_FACTOR", "1")
os.environ.setdefault("QT_FONT_DPI", "96")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from gui.main_window import MainWindow  # noqa: E402
from gui.project_panel import ProjectPanel  # noqa: E402

WORKSPACE_NAMES = (
    "annotate",
    "train",
    "inference",
    "dataset",
    "export",
    "quality",
    "results",
)


def capture_workspaces(output_dir: Path, width: int, height: int) -> list[Path]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication([])
    original_env_check = ProjectPanel._start_env_check
    original_cwd = Path.cwd()
    ProjectPanel._start_env_check = lambda self: None

    try:
        with tempfile.TemporaryDirectory(prefix="yolo-studio-ui-") as temp_dir:
            try:
                os.chdir(temp_dir)
                window = MainWindow()
                window.resize(width, height)

                # Exercise every page without reading models or project state from the repository.
                window.current_project = {"name": "UI Preview", "root": temp_dir}
                window.image_list = ["sample-001.jpg", "sample-002.jpg", "sample-003.jpg"]
                window._update_project_gate()
                window.show()
                QTest.qWait(180)

                captured: list[Path] = []
                for index, name in enumerate(WORKSPACE_NAMES):
                    window._switch_workspace(index)
                    app.processEvents()
                    QTest.qWait(80)
                    path = output_dir / f"{index:02d}-{name}-{width}x{height}.png"
                    if not window.grab().save(str(path), "PNG"):
                        raise RuntimeError(f"Failed to save screenshot: {path}")
                    captured.append(path)

                window.close()
                app.processEvents()
                return captured
            finally:
                os.chdir(original_cwd)
    finally:
        os.chdir(original_cwd)
        ProjectPanel._start_env_check = original_env_check


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=920)
    args = parser.parse_args()

    for path in capture_workspaces(args.output_dir, args.width, args.height):
        print(path)


if __name__ == "__main__":
    main()
