"""Yolo Studio entry point.

Copyright (C) 2024–2026 YDERO3452
SPDX-License-Identifier: GPL-3.0-only
"""

import faulthandler
import sys
import traceback
from pathlib import Path

# 添加项目路径到 sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Ensure logs directory exists before opening crash log
(project_root / "logs").mkdir(exist_ok=True)

# Enable faulthandler to print C-level traceback on segfault
try:
    _fault_log = open(project_root / "logs" / "crash.log", "w", encoding="utf-8")
    faulthandler.enable(file=_fault_log, all_threads=True)
except OSError:
    _fault_log = None  # harmless: crash log unavailable (disk full / permissions)

from loguru import logger  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from gui.main_window import MainWindow  # noqa: E402


def global_exception_hook(exc_type, exc_value, exc_tb):
    """Catch unhandled Python exceptions and log them."""
    msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    logger.critical(f"Unhandled exception:\n{msg}")
    if _fault_log is not None:
        _fault_log.write(f"PYTHON CRASH:\n{msg}\n")
        _fault_log.flush()
    sys.__excepthook__(exc_type, exc_value, exc_tb)


def setup_logging() -> None:
    """设置日志系统."""
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)

    logger.remove()  # 移除默认处理器
    if sys.stderr:
        logger.add(
            sys.stderr,
            format="<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level="INFO"
        )
    logger.add(
        log_dir / "yolo_studio.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="500 MB",
        retention="10 days"
    )

    logger.info("Logging system initialized")


def main():
    """应用主函数."""
    # 设置日志
    setup_logging()

    # Install global exception hook
    sys.excepthook = global_exception_hook

    logger.info("Starting Yolo Studio...")

    # 创建应用
    app = QApplication(sys.argv)

    # 设置应用信息
    app.setApplicationName("Yolo Studio")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("Yolo Studio")

    # 创建主窗口
    window = MainWindow()
    window.show()

    logger.info("Yolo Studio started successfully")

    # 运行应用
    ret = app.exec()

    # Cleanup
    if _fault_log is not None:
        _fault_log.close()
    sys.exit(ret)


if __name__ == "__main__":
    main()
