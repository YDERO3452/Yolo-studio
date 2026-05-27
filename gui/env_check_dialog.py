"""Environment check dialog.

Provides a visual wizard that detects GPU, driver, CUDA, and PyTorch
compatibility, and guides the user through fixing any issues.
"""

import os
import shlex
import subprocess
import sys
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTextEdit, QProgressBar, QGroupBox,
    QMessageBox, QScrollArea, QWidget, QFrame, QFileDialog,
)
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QFont
from loguru import logger

from core.env_setup import (
    detect_environment, diagnose_environment, format_diagnosis_html,
    _get_pytorch_install_cmd,
    get_python_wheel_tags, get_pytorch_install_commands,
    get_pytorch_install_plan,
    EnvInfo, DiagnosisItem, GPUVendor,
)
from gui.theme import build_stylesheet


class _EnvDetectWorker(QThread):
    """Background thread for environment detection."""
    finished = pyqtSignal(object, list)  # EnvInfo, list[DiagnosisItem]

    def run(self):
        try:
            env = detect_environment()
            diag = diagnose_environment(env)
            self.finished.emit(env, diag)
        except Exception as e:
            logger.error(f"Environment detection failed: {e}")
            self.finished.emit(None, [])


class _PipInstallWorker(QThread):
    """Background thread for pip install operations."""

    finished = pyqtSignal(int, str, str)  # returncode, stdout, stderr

    def __init__(self, cmd: list, parent=None):
        super().__init__(parent)
        self.cmd = cmd

    def run(self):
        try:
            result = subprocess.run(
                self.cmd, capture_output=True, text=True, timeout=600
            )
            self.finished.emit(result.returncode, result.stdout, result.stderr)
        except subprocess.TimeoutExpired:
            self.finished.emit(-1, "", "安装超时（10分钟限制），请手动安装。")
        except Exception as e:
            self.finished.emit(-1, "", str(e))


class EnvironmentCheckDialog(QDialog):
    """Dialog for checking and configuring the runtime environment."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("环境检测与配置")
        self.setMinimumSize(680, 560)
        self.resize(750, 620)

        self._env: EnvInfo | None = None
        self._diag: list[DiagnosisItem] = []

        self._init_ui()
        self._apply_dark_style()
        self._start_detection()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Title
        title = QLabel("环境检测与配置")
        title.setObjectName("BrandTitle")
        layout.addWidget(title)

        desc = QLabel(
            "自动检测您的 GPU、NVIDIA 驱动、CUDA 和 PyTorch 环境，\n"
            "并按当前机器动态生成匹配的 PyTorch 安装命令。"
        )
        desc.setObjectName("MutedText")
        layout.addWidget(desc)

        # Progress bar (shown during detection)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # indeterminate
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(6)
        layout.addWidget(self.progress_bar)

        # Status label during detection
        self.status_label = QLabel("正在检测环境...")
        self.status_label.setObjectName("StatusPill")
        self.status_label.setProperty("variant", "accent")
        layout.addWidget(self.status_label)

        # Results area (scroll)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        self.result_widget = QWidget()
        self.result_layout = QVBoxLayout(self.result_widget)
        self.result_layout.setSpacing(8)

        # -- System info group --
        sys_group = QGroupBox("系统信息")
        sys_form = QVBoxLayout()
        self.sys_info_label = QLabel("检测中...")
        self.sys_info_label.setFont(QFont("Consolas", 10))
        self.sys_info_label.setWordWrap(True)
        sys_form.addWidget(self.sys_info_label)
        sys_group.setLayout(sys_form)
        self.result_layout.addWidget(sys_group)

        # -- Diagnosis results group --
        diag_group = QGroupBox("诊断结果")
        diag_layout = QVBoxLayout()
        self.diag_text = QTextEdit()
        self.diag_text.setReadOnly(True)
        self.diag_text.setFont(QFont("Consolas", 10))
        self.diag_text.setMinimumHeight(200)
        diag_layout.addWidget(self.diag_text)
        diag_group.setLayout(diag_layout)
        self.result_layout.addWidget(diag_group)

        # -- Install guide group --
        self.install_group = QGroupBox("推荐安装命令")
        install_layout = QVBoxLayout()
        self.install_text = QTextEdit()
        self.install_text.setReadOnly(True)
        self.install_text.setFont(QFont("Consolas", 10))
        self.install_text.setMaximumHeight(260)
        install_layout.addWidget(self.install_text)

        # Wheel download buttons (clickable, open browser)
        wheel_btn_row = QHBoxLayout()
        self.open_torch_wheel_btn = QPushButton("打开 torch whl 下载页")
        self.open_torch_wheel_btn.clicked.connect(lambda: self._open_url(self._torch_wheel_url))
        wheel_btn_row.addWidget(self.open_torch_wheel_btn)
        self.open_torchvision_wheel_btn = QPushButton("打开 torchvision whl 下载页")
        self.open_torchvision_wheel_btn.clicked.connect(lambda: self._open_url(self._torchvision_wheel_url))
        wheel_btn_row.addWidget(self.open_torchvision_wheel_btn)
        install_layout.addLayout(wheel_btn_row)

        self._torch_wheel_url = ""
        self._torchvision_wheel_url = ""

        self.copy_btn = QPushButton("复制安装命令")
        self.copy_btn.setObjectName("PrimaryButton")
        self.copy_btn.clicked.connect(self._copy_install_cmd)
        install_layout.addWidget(self.copy_btn)

        # Quick install buttons
        quick_layout = QHBoxLayout()
        self.download_wheels_btn = QPushButton("下载匹配 wheel")
        self.download_wheels_btn.setToolTip("自动下载匹配当前机器的 torch + torchvision wheel 到本地")
        self.download_wheels_btn.clicked.connect(self._download_pytorch_wheels)
        quick_layout.addWidget(self.download_wheels_btn)

        self.install_pytorch_btn = QPushButton("在线安装 PyTorch")
        self.install_pytorch_btn.setToolTip("使用官方 download.pytorch.org 源在线安装")
        self.install_pytorch_btn.clicked.connect(self._install_pytorch)
        quick_layout.addWidget(self.install_pytorch_btn)

        self.install_local_wheel_btn = QPushButton("安装本地 wheel")
        self.install_local_wheel_btn.setToolTip("选择下载好的 torch / torchvision / 依赖 .whl 文件安装")
        self.install_local_wheel_btn.clicked.connect(self._install_local_wheels)
        quick_layout.addWidget(self.install_local_wheel_btn)

        self.install_ultralytics_btn = QPushButton("安装 Ultralytics")
        self.install_ultralytics_btn.clicked.connect(self._install_ultralytics)
        quick_layout.addWidget(self.install_ultralytics_btn)

        self.update_driver_btn = QPushButton("打开 NVIDIA 驱动下载页")
        self.update_driver_btn.clicked.connect(self._open_driver_download)
        quick_layout.addWidget(self.update_driver_btn)

        install_layout.addLayout(quick_layout)
        self.install_group.setLayout(install_layout)
        self.result_layout.addWidget(self.install_group)

        self.result_layout.addStretch()
        scroll.setWidget(self.result_widget)
        layout.addWidget(scroll, stretch=1)

        # Bottom buttons
        btn_layout = QHBoxLayout()
        self.recheck_btn = QPushButton("重新检测")
        self.recheck_btn.setObjectName("PrimaryButton")
        self.recheck_btn.clicked.connect(self._start_detection)
        btn_layout.addWidget(self.recheck_btn)

        btn_layout.addStretch()

        self.close_btn = QPushButton("关闭")
        self.close_btn.clicked.connect(self.close)
        btn_layout.addWidget(self.close_btn)

        layout.addLayout(btn_layout)

    def _apply_dark_style(self):
        self.setStyleSheet(build_stylesheet())

    def _start_detection(self):
        """Start environment detection in a background thread."""
        self.progress_bar.setVisible(True)
        self.status_label.setVisible(True)
        self.status_label.setText("正在检测环境...")
        self.install_group.setEnabled(False)
        self.recheck_btn.setEnabled(False)

        self._worker = _EnvDetectWorker()
        self._worker.finished.connect(self._on_detection_done)
        self._worker.start()

    def _on_detection_done(self, env, diag):
        """Handle detection completion."""
        self.progress_bar.setVisible(False)
        self.status_label.setVisible(False)
        self.recheck_btn.setEnabled(True)
        self.install_group.setEnabled(True)

        if env is None:
            self.diag_text.setHtml(
                '<p style="color:#FF453A;">环境检测失败，请检查日志。</p>'
            )
            return

        self._env = env
        self._diag = diag

        # System info
        sys_lines = [
            f"操作系统: {env.os_name} {env.os_version} ({env.os_arch})",
            f"Python: {env.python_version}",
        ]

        if env.gpus:
            for gpu in env.gpus:
                vram = f" ({gpu.vram_mb}MB)" if gpu.vram_mb else ""
                vendor_tag = ""
                if gpu.vendor == GPUVendor.NVIDIA:
                    vendor_tag = " [NVIDIA]"
                elif gpu.vendor == GPUVendor.AMD:
                    vendor_tag = " [AMD]"
                elif gpu.vendor == GPUVendor.Intel:
                    vendor_tag = " [Intel]"
                sys_lines.append(f"GPU: {gpu.name}{vram}{vendor_tag}")
                if gpu.compute_capability:
                    sys_lines.append(f"  计算能力: {gpu.compute_capability}")
                if gpu.driver_version:
                    sys_lines.append(f"  驱动版本: {gpu.driver_version}")
        else:
            sys_lines.append("GPU: 未检测到任何显卡")

        if env.nvidia_driver_version:
            sys_lines.append(f"NVIDIA 驱动: {env.nvidia_driver_version}")
        if env.driver_max_cuda:
            sys_lines.append(f"驱动支持 CUDA 最高: {env.driver_max_cuda}")
        if env.cuda_toolkit_version:
            sys_lines.append(f"CUDA Toolkit: {env.cuda_toolkit_version}")
        else:
            sys_lines.append("CUDA Toolkit: 未检测到（PyTorch pip 版通常不需要单独安装）")
        if env.pytorch_version:
            sys_lines.append(f"PyTorch: {env.pytorch_version} (CUDA {env.pytorch_cuda_version or 'CPU'})")
        else:
            sys_lines.append("PyTorch: 未安装")
        if env.ultralytics_version:
            sys_lines.append(f"Ultralytics: {env.ultralytics_version}")
        else:
            sys_lines.append("Ultralytics: 未安装")

        self.sys_info_label.setText("\n".join(sys_lines))

        # Diagnosis
        self.diag_text.setHtml(format_diagnosis_html(diag))

        # ---- Install commands ----
        torch_plan = get_pytorch_install_plan(env)
        wheel_tags = get_python_wheel_tags()
        torch_cmds = get_pytorch_install_commands(env)
        already_ok = bool(torch_plan.get("already_ok"))

        install_lines = []

        # --- Always show matching summary in a compact way ---
        if env.has_nvidia_gpu:
            nvidia_gpus = [g for g in env.gpus if g.vendor == GPUVendor.NVIDIA]
            gpu_names = ", ".join(g.name for g in nvidia_gpus)
            install_lines.append(f"GPU: {gpu_names}")
            if env.nvidia_driver_version:
                install_lines.append(f"驱动: {env.nvidia_driver_version}")
            if env.driver_max_cuda:
                install_lines.append(f"驱动支持 CUDA: {env.driver_max_cuda}")
            install_lines.append(f"匹配 PyTorch: {torch_plan.get('wheel_tag', 'cpu')}")
        elif env.has_amd_gpu:
            install_lines.append("GPU: AMD（不支持 CUDA 加速）")
        elif env.has_intel_gpu:
            install_lines.append("GPU: Intel（兼容性有限）")
        else:
            install_lines.append("未检测到独立 GPU，将使用 CPU 模式")
        install_lines.append(f"Python: {wheel_tags['python']} | 平台: {wheel_tags['platform']}")

        # Status indicator
        if already_ok:
            install_lines.append("")
            install_lines.append(f"✅ {torch_plan.get('reason')}")
        else:
            install_lines.append("")
            install_lines.append(f"❌ {torch_plan.get('reason')}")

        # --- Always show wheel download links + install commands ---
        install_lines.append("")
        install_lines.append("====== 下载 wheel 文件 ======")
        install_lines.append(f"torch whl:  {torch_cmds['torch_page']}")
        install_lines.append(f"torchvision whl:  {torch_cmds['torchvision_page']}")
        install_lines.append(f"(选择匹配 {torch_cmds['torch_pattern']} 的文件)")
        install_lines.append(f"(选择匹配 {torch_cmds['torchvision_pattern']} 的文件)")
        install_lines.append("")
        install_lines.append("====== pip 安装命令 ======")
        install_lines.append(f"在线安装:")
        install_lines.append(torch_cmds["online_install"])
        install_lines.append("")
        install_lines.append(f"离线安装 (下载后):")
        install_lines.append(torch_cmds["install_local_dir"])
        install_lines.append("")
        install_lines.append("pip install ultralytics")

        # AMD Linux specific
        if env.has_amd_gpu and not env.has_nvidia_gpu and not env.is_windows:
            install_lines.insert(1, "# AMD ROCm (仅Linux): pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2")

        self.install_text.setPlainText("\n".join(install_lines))

        # Store wheel URLs for the download buttons
        self._torch_wheel_url = torch_cmds.get("torch_page", "")
        self._torchvision_wheel_url = torch_cmds.get("torchvision_page", "")

        # Enable/disable quick install buttons based on current state
        if env.pytorch_installed and env.pytorch_cuda_available:
            self.install_pytorch_btn.setEnabled(False)
            self.install_pytorch_btn.setText("PyTorch CUDA - 已可用")
            self.install_local_wheel_btn.setEnabled(False)
            self.download_wheels_btn.setEnabled(False)
        elif env.has_amd_gpu and not env.has_nvidia_gpu:
            self.install_pytorch_btn.setEnabled(True)
            self.install_pytorch_btn.setText("安装 PyTorch (CPU)")
            self.install_local_wheel_btn.setEnabled(True)
            self.download_wheels_btn.setEnabled(True)
        else:
            self.install_pytorch_btn.setEnabled(True)
            self.install_local_wheel_btn.setEnabled(True)
            self.download_wheels_btn.setEnabled(True)
            if torch_plan.get("backend") == "cuda":
                self.install_pytorch_btn.setText(f"安装 PyTorch ({torch_plan.get('wheel_tag')})")
            else:
                self.install_pytorch_btn.setText("安装 PyTorch")

        if env.ultralytics_installed:
            self.install_ultralytics_btn.setEnabled(False)
            self.install_ultralytics_btn.setText("Ultralytics - 已安装")
        else:
            self.install_ultralytics_btn.setEnabled(True)

        # Driver button: show/hide based on GPU vendor
        if not env.has_nvidia_gpu:
            self.update_driver_btn.setVisible(False)
        elif env.nvidia_driver_installed and env.driver_max_cuda:
            self.update_driver_btn.setEnabled(True)
            self.update_driver_btn.setText("打开 NVIDIA 驱动下载页")
            try:
                if float(env.driver_max_cuda) < 11.8:
                    self.update_driver_btn.setText("更新 NVIDIA 驱动")
            except ValueError:
                pass
        else:
            self.update_driver_btn.setEnabled(True)

    @staticmethod
    def _open_url(url: str) -> None:
        """Open a URL in the system browser."""
        if not url:
            return
        try:
            import webbrowser
            webbrowser.open(url)
        except Exception:
            pass  # harmless: browser may not be available or URL fails to open

    def _copy_install_cmd(self):
        """Copy the install commands to clipboard (only executable lines)."""
        text = self.install_text.toPlainText()
        if not text:
            return
        # Extract only executable lines (skip comments, status, separators, hints)
        exec_lines = []
        for line in text.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                continue
            if stripped.startswith("✅") or stripped.startswith("❌") or stripped.startswith("⚠️"):
                continue
            if stripped.startswith("======"):
                continue
            if stripped.startswith("(选择匹配"):
                continue
            if stripped.startswith("在线安装") or stripped.startswith("离线安装"):
                continue
            exec_lines.append(stripped)
        copy_text = "\n".join(exec_lines) if exec_lines else text
        from PyQt6.QtWidgets import QApplication
        QApplication.clipboard().setText(copy_text)
        self.copy_btn.setText("已复制!")
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(2000, lambda: self.copy_btn.setText("复制安装命令"))

    def _install_pytorch(self):
        """Attempt to install PyTorch GPU version via pip."""
        if not self._env:
            return

        plan = get_pytorch_install_plan(self._env)
        if plan.get("already_ok"):
            QMessageBox.information(self, "PyTorch", "当前 PyTorch CUDA 已可用，无需重装。")
            return

        cmd = _get_pytorch_install_cmd(self._env)

        # Show an extra warning when we're replacing a CPU-only torch
        is_replacing = (
            self._env.pytorch_installed
            and not self._env.pytorch_cuda_available
        )
        if is_replacing:
            from PyQt6.QtWidgets import QMessageBox
            reply = QMessageBox.warning(
                self, "替换 CPU 版 PyTorch",
                f"检测到当前安装的是 CPU 版 PyTorch，将强制替换为推荐版本。\n\n"
                f"执行命令:\n{cmd}\n\n"
                f"如果下载慢，请取消并使用“下载匹配 wheel 到本地目录 + 本地安装”。是否继续在线安装？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            self._run_pip_install_direct(f"PyTorch ({plan.get('wheel_tag')})", cmd)
        else:
            self._run_pip_install(f"PyTorch ({plan.get('wheel_tag')})", cmd)

    def _download_pytorch_wheels(self):
        """Download matching PyTorch wheels using pip download (async)."""
        if not self._env:
            return
        torch_cmds = get_pytorch_install_commands(self._env)
        download_dir = torch_cmds["download_dir"]
        cmd = torch_cmds["download_wheels"]

        reply = QMessageBox.question(
            self, "下载匹配 wheel",
            f"将下载匹配当前机器的 torch + torchvision wheel 到:\n"
            f"{download_dir}\n\n"
            f"执行命令:\n{cmd}\n\n"
            f"确认下载？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._pip_install_name = "下载 PyTorch wheel"
        self._pip_extra_info = f"PyTorch wheel 已下载到:\n{download_dir}\n\n可使用「安装本地 wheel」按钮从该目录安装，\n或手动执行:\n{torch_cmds['install_local_dir']}"
        self._install_btn_set_enabled(False)
        self._pip_worker = _PipInstallWorker(shlex.split(cmd), self)
        self._pip_worker.finished.connect(self._on_pip_install_done)
        self._pip_worker.start()

    def _install_local_wheels(self):
        if not self._env:
            return
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "选择 PyTorch wheel 文件",
            "",
            "Python Wheel (*.whl);;所有文件 (*)",
        )
        if not paths:
            return
        lower_names = [os.path.basename(path).lower() for path in paths]
        has_torch = any(name.startswith("torch-") for name in lower_names)
        has_torchvision = any(name.startswith("torchvision-") for name in lower_names)
        if not (has_torch and has_torchvision):
            reply = QMessageBox.warning(
                self,
                "wheel 文件可能不完整",
                "PyTorch 本地安装通常至少需要 torch 和 torchvision 两个 wheel。\n"
                "你当前选择的文件里没有同时检测到这两个包，继续安装可能失败。\n\n"
                "是否继续？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        cmd = f'"{sys.executable}" -m pip install ' + " ".join(f'"{path}"' for path in paths)
        self._run_pip_install("本地 PyTorch wheel", cmd)

    def _run_pip_install_direct(self, name: str, cmd: str):
        """Run a pip install command (user already confirmed) in background."""
        self._pip_install_name = name
        self._pip_extra_info = ""
        self._install_btn_set_enabled(False)
        self._pip_worker = _PipInstallWorker(shlex.split(cmd), self)
        self._pip_worker.finished.connect(self._on_pip_install_done)
        self._pip_worker.start()

    def _install_ultralytics(self):
        """Install Ultralytics via pip."""
        self._run_pip_install("Ultralytics", "pip install ultralytics")

    def _run_pip_install(self, name: str, cmd: str):
        """Run a pip install command (async) and show result."""
        reply = QMessageBox.question(
            self, f"安装 {name}",
            f"即将执行:\n\n{cmd}\n\n确认安装？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._pip_install_name = name
        self._pip_extra_info = ""
        self._install_btn_set_enabled(False)
        self._pip_worker = _PipInstallWorker(shlex.split(cmd), self)
        self._pip_worker.finished.connect(self._on_pip_install_done)
        self._pip_worker.start()

    def _on_pip_install_done(self, returncode: int, stdout: str, stderr: str):
        """Handle pip install completion."""
        self._install_btn_set_enabled(True)
        if returncode == 0:
            msg = f"{self._pip_install_name} 安装完成！\n\n点击「重新检测」验证安装结果。"
            if self._pip_extra_info:
                msg = self._pip_extra_info
            QMessageBox.information(self, "操作成功", msg)
        else:
            QMessageBox.critical(
                self, "操作失败",
                f"{self._pip_install_name} 失败:\n\n{stderr[:2000]}"
            )

    def _install_btn_set_enabled(self, enabled: bool):
        """Enable or disable all quick install buttons, saving/restoring state."""
        btn_names = ("install_pytorch_btn", "download_wheels_btn",
                      "install_local_wheel_btn", "install_ultralytics_btn")
        if not enabled:
            self._saved_btn_states = {}
            for name in btn_names:
                btn = getattr(self, name, None)
                if btn is not None:
                    self._saved_btn_states[name] = btn.isEnabled()
                    btn.setEnabled(False)
        else:
            saved = getattr(self, "_saved_btn_states", {})
            for name in btn_names:
                btn = getattr(self, name, None)
                if btn is not None and name in saved:
                    btn.setEnabled(saved[name])

    def _open_driver_download(self):
        """Open NVIDIA driver download page in browser."""
        url = "https://www.nvidia.com/Download/index.aspx"
        try:
            import webbrowser
            webbrowser.open(url)
        except Exception:
            QMessageBox.information(self, "下载驱动", f"请在浏览器中打开:\n{url}")
