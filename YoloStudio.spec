# -*- mode: python ; coding: utf-8 -*-
# ruff: noqa: F821 — Analysis, PYZ, EXE are PyInstaller builtins injected at build time
"""PyInstaller spec for YoloStudio onefile build.

生成:  pyinstaller YoloStudio.spec
"""

from pathlib import Path

# ── PyInstaller 工具 ──────────────────────────────────────
from PyInstaller.utils.hooks import (
    collect_dynamic_libs,
    collect_submodules,
)

# ── 数据文件 ───────────────────────────────────────────────
# (源路径, 目标目录) — 目标目录是 sys._MEIPASS 下的相对路径

_datas = [
    ("resources/icons", "resources/icons"),   # SVG 工具栏图标 (17 files)
    ("configs", "configs"),                   # default.yaml
]

# ── 二进制文件 ────────────────────────────────────────────
_binaries = []

# cv2 原生库和 Qt 插件 (opencv-python wheel 内嵌)
try:
    import cv2
    cv2_dir = Path(cv2.__file__).parent
    # 收集 cv2 动态库 (abi3.so, qt plugins .so)
    _binaries += collect_dynamic_libs('cv2')
    # 收集 cv2 Qt 插件目录下的所有 .so
    qt_plugins = cv2_dir / 'qt' / 'plugins'
    if qt_plugins.is_dir():
        for so_file in qt_plugins.rglob('*.so'):
            dest = str(so_file.parent.relative_to(cv2_dir.parent))
            _binaries.append((str(so_file), dest))
    # 收集 cv2 Qt 字体
    qt_fonts = cv2_dir / 'qt' / 'fonts'
    if qt_fonts.is_dir():
        for font_file in qt_fonts.rglob('*.ttf'):
            dest = str(font_file.parent.relative_to(cv2_dir.parent))
            _datas.append((str(font_file), dest))
except Exception:
    pass  # harmless: cv2 not available at build time (should not happen in CI)

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

    # cv2 及子模块 — 显式声明以防原生库/配置文件缺失
    "cv2",
    "numpy",                    # cv2 依赖
]

# 补充 cv2 子模块 (config, mat_wrapper, misc, gapi 等)
try:
    _hidden += collect_submodules('cv2', filter=lambda n: n != 'cv2.load_config_py2')
except Exception:
    pass  # harmless: cv2 not available

# ── Analysis ──────────────────────────────────────────────
# PyInstaller builtins (Analysis, PYZ, EXE) are injected at build time
a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=_binaries,
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
