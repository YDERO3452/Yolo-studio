# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for YoloStudio onefile build.

生成:  pyinstaller YoloStudio.spec
"""

import os
import sys
from pathlib import Path

# ── 数据文件 ───────────────────────────────────────────────
# (源路径, 目标目录) — 目标目录是 sys._MEIPASS 下的相对路径

_datas = [
    ("resources/icons", "resources/icons"),   # SVG 工具栏图标 (17 files)
    ("configs", "configs"),                   # default.yaml
]

# 配置目录 (llm_config.json 等)
if Path("config").is_dir():
    _datas.append(("config", "config"))

# 训练字体 (CI 脚本下载，约 23 MB；可选)
if Path("Arial.Unicode.ttf").exists():
    _datas.append(("Arial.Unicode.ttf", "."))

# ── 隐藏导入 ──────────────────────────────────────────────
# 这些模块在代码中动态 import (非顶层)，PyInstaller 分析不到

_hidden = [
    # Qt 插件
    "PyQt6.QtSvg",              # main_window._tool_icon() 中延迟导入

    # 动态对话框 (通过菜单触发，非顶层 import)
    "gui.env_check_dialog",     # 环境检测 → "环境" 菜单
    "gui.video_capture_dialog", # 视频截帧 → "视频截帧" 菜单
    "gui.format_conversion_dialog",  # 格式转换 → "格式转换" 菜单

    # 工作流优化面板中动态导入的 core 模块
    "core.batch_processor",
    "core.format_converter",

    # cv2 在部分模块中动态引用，显式声明以防原生库缺失
    "cv2",
]

# ── Analysis ──────────────────────────────────────────────
a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=[],
    datas=_datas,
    hiddenimports=_hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="YoloStudio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
