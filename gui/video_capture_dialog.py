"""Video frame extraction dialog.

Provides:
- Video playback with seek bar
- Auto-extraction (interval / scene-change mode)
- Manual capture (spacebar / button)
- Online video URL download (yt-dlp)
- Real-time preview of captured frames
"""

import os
import threading
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QSlider, QLineEdit, QFileDialog, QProgressBar, QListWidget,
    QListWidgetItem, QTabWidget, QWidget, QFormLayout, QMessageBox,
    QSplitter,
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QTimer, QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtGui import QImage, QPixmap, QFont, QKeySequence, QShortcut
from loguru import logger

from core.video_extractor import VideoFrameExtractor
from gui.theme import Theme, build_stylesheet


class _VideoDownloadWorker(QThread):
    """Background thread for downloading online videos."""
    progress = pyqtSignal(str)
    finished = pyqtSignal(str)  # path to downloaded file
    error = pyqtSignal(str)

    def __init__(self, url: str, output_dir: str, proxy: str = "",
                 cookie_file: str = "", cookie_browser: str = ""):
        super().__init__()
        self.url = url
        self.output_dir = output_dir
        self.proxy = proxy
        self.cookie_file = cookie_file
        self.cookie_browser = cookie_browser

    def run(self):
        try:
            path = VideoFrameExtractor.download_video(
                self.url, self.output_dir,
                progress_callback=lambda msg: self.progress.emit(msg),
                proxy=self.proxy,
                cookie_file=self.cookie_file,
                cookie_browser=self.cookie_browser,
            )
            if path:
                self.finished.emit(path)
            else:
                self.error.emit("下载失败，请检查URL、网络连接或Cookie配置")
        except Exception as e:
            self.error.emit(str(e))


class _AutoExtractWorker(QThread):
    """Background thread for auto frame extraction."""
    progress = pyqtSignal(int, int)  # current, total
    finished = pyqtSignal(list)  # list of saved paths

    def __init__(self, extractor: VideoFrameExtractor, output_dir: str,
                 mode: str, interval_frames: int, interval_seconds: float,
                 scene_threshold: float, dedup: bool, dedup_threshold: int,
                 max_frames: int):
        super().__init__()
        self.extractor = extractor
        self.output_dir = output_dir
        self.mode = mode
        self.interval_frames = interval_frames
        self.interval_seconds = interval_seconds
        self.scene_threshold = scene_threshold
        self.dedup = dedup
        self.dedup_threshold = dedup_threshold
        self.max_frames = max_frames

    def run(self):
        try:
            # Re-open the video in this thread (cv2 VideoCapture is not thread-safe)
            self.extractor.open(self.extractor.video_path)

            if self.mode == "interval" and self.interval_seconds > 0:
                paths = self.extractor.extract_by_seconds(
                    self.output_dir,
                    every_seconds=self.interval_seconds,
                    dedup=self.dedup,
                    dedup_threshold=self.dedup_threshold,
                    max_frames=self.max_frames,
                    progress_callback=lambda c, t: self.progress.emit(c, t),
                )
            else:
                paths = self.extractor.extract_auto(
                    self.output_dir,
                    mode=self.mode,
                    interval_frames=self.interval_frames,
                    scene_threshold=self.scene_threshold,
                    dedup=self.dedup,
                    dedup_threshold=self.dedup_threshold,
                    max_frames=self.max_frames,
                    progress_callback=lambda c, t: self.progress.emit(c, t),
                )
            self.finished.emit(paths)
        except Exception as e:
            logger.error(f"Auto extraction failed: {e}")
            self.finished.emit([])


class VideoCaptureDialog(QDialog):
    """Dialog for extracting frames from video sources."""

    frames_captured = pyqtSignal(list)  # list of image paths to load for annotation

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("视频截帧")
        self.setMinimumSize(1000, 700)
        self.resize(1200, 800)

        self.extractor = VideoFrameExtractor()
        self._capture_count = 0
        self._captured_paths: list[str] = []
        self._download_worker = None
        self._extract_worker = None
        self._output_dir = ""
        self._timer = QTimer(self)
        self._timer.setInterval(33)  # ~30fps preview
        self._timer.timeout.connect(self._update_preview)
        self._is_playing = False
        self._current_frame_idx = 0

        self._init_ui()
        self._apply_dark_style()

    # ------------------------------------------------------------------
    # UI Setup
    # ------------------------------------------------------------------

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # ---- Top: Source tabs ----
        source_tabs = QTabWidget()

        # Tab 1: Local video
        local_tab = QWidget()
        local_layout = QHBoxLayout(local_tab)
        self.video_path_edit = QLineEdit()
        self.video_path_edit.setPlaceholderText("选择视频文件...")
        local_layout.addWidget(self.video_path_edit, stretch=1)
        browse_btn = QPushButton("浏览...")
        browse_btn.clicked.connect(self._browse_video)
        local_layout.addWidget(browse_btn)
        load_btn = QPushButton("加载视频")
        load_btn.setObjectName("PrimaryButton")
        load_btn.clicked.connect(self._load_local_video)
        local_layout.addWidget(load_btn)
        source_tabs.addTab(local_tab, "本地视频")

        # Tab 2: Online video
        online_tab = QWidget()
        online_layout = QVBoxLayout(online_tab)
        online_layout.setContentsMargins(4, 4, 4, 4)

        # Platform hint
        platforms = VideoFrameExtractor.get_supported_platforms()
        self.platform_hint = QLabel(
            f"支持平台: {' / '.join(platforms)}  (快手需安装 you-get)"
        )
        self.platform_hint.setObjectName("MutedText")
        self.platform_hint.setWordWrap(True)
        online_layout.addWidget(self.platform_hint)

        # URL input
        url_layout = QHBoxLayout()
        self.url_edit = QLineEdit()
        self.url_edit.setPlaceholderText("输入视频链接 (抖音/B站/快手/YouTube等)...")
        self.url_edit.textChanged.connect(self._on_url_changed)
        url_layout.addWidget(self.url_edit, stretch=1)
        self.download_btn = QPushButton("下载视频")
        self.download_btn.setObjectName("PrimaryButton")
        self.download_btn.clicked.connect(self._download_online_video)
        url_layout.addWidget(self.download_btn)
        online_layout.addLayout(url_layout)

        # Detected platform label
        self.detected_platform_label = QLabel("")
        self.detected_platform_label.setObjectName("StatusPill")
        self.detected_platform_label.setProperty("variant", "accent")
        online_layout.addWidget(self.detected_platform_label)

        # Proxy and Cookie row
        proxy_cookie_layout = QHBoxLayout()

        proxy_cookie_layout.addWidget(QLabel("代理:"))
        self.proxy_edit = QLineEdit()
        self.proxy_edit.setPlaceholderText("http://127.0.0.1:7890")
        self.proxy_edit.setMaximumWidth(180)
        proxy_cookie_layout.addWidget(self.proxy_edit)

        proxy_cookie_layout.addWidget(QLabel("Cookie来源:"))
        self.cookie_browser_combo = QComboBox()
        self.cookie_browser_combo.addItems(["自动获取", "Chrome", "Edge", "Firefox", "Safari"])
        self.cookie_browser_combo.setMaximumWidth(100)
        self.cookie_browser_combo.setToolTip(
            "从浏览器自动提取Cookie (需在浏览器中登录过抖音等平台)\n"
            "选择'自动获取'将依次尝试 Chrome/Edge/Firefox"
        )
        proxy_cookie_layout.addWidget(self.cookie_browser_combo)

        self.cookie_edit = QLineEdit()
        self.cookie_edit.setPlaceholderText("或手动指定 cookies.txt 路径")
        self.cookie_edit.setMaximumWidth(180)
        proxy_cookie_layout.addWidget(self.cookie_edit, stretch=1)

        cookie_browse_btn = QPushButton("...")
        cookie_browse_btn.setFixedWidth(30)
        cookie_browse_btn.clicked.connect(self._browse_cookie_file)
        proxy_cookie_layout.addWidget(cookie_browse_btn)

        online_layout.addLayout(proxy_cookie_layout)

        # Download tool status
        status_layout = QHBoxLayout()
        self.ytdlp_status = QLabel()
        self.youget_status = QLabel()
        self._update_download_status()
        status_layout.addWidget(self.ytdlp_status)
        status_layout.addWidget(self.youget_status)
        status_layout.addStretch()
        online_layout.addLayout(status_layout)

        source_tabs.addTab(online_tab, "在线视频")

        layout.addWidget(source_tabs)

        # ---- Video info bar ----
        self.info_label = QLabel("未加载视频")
        self.info_label.setObjectName("MutedText")
        layout.addWidget(self.info_label)

        # ---- Middle: Preview + Controls ----
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Video preview
        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)
        preview_layout.setContentsMargins(0, 0, 0, 0)

        self.preview_label = QLabel("视频预览区域")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(640, 360)
        self.preview_label.setObjectName("PreviewSurface")
        preview_layout.addWidget(self.preview_label)

        # Seek bar
        seek_layout = QHBoxLayout()
        self.time_label = QLabel("00:00")
        self.time_label.setFixedWidth(50)
        seek_layout.addWidget(self.time_label)
        self.seek_slider = QSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setRange(0, 1000)
        self.seek_slider.setValue(0)
        self.seek_slider.sliderPressed.connect(self._on_seek_pressed)
        self.seek_slider.sliderReleased.connect(self._on_seek_released)
        self.seek_slider.valueChanged.connect(self._on_seek_changed)
        self._seek_dragging = False
        seek_layout.addWidget(self.seek_slider)
        self.duration_label = QLabel("00:00")
        self.duration_label.setFixedWidth(50)
        seek_layout.addWidget(self.duration_label)
        preview_layout.addLayout(seek_layout)

        # Playback controls
        ctrl_layout = QHBoxLayout()
        self.play_btn = QPushButton("播放")
        self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(self._toggle_play)
        ctrl_layout.addWidget(self.play_btn)

        self.capture_btn = QPushButton("截取当前帧")
        self.capture_btn.setEnabled(False)
        self.capture_btn.setObjectName("PrimaryButton")
        self.capture_btn.clicked.connect(self._capture_current_frame)
        ctrl_layout.addWidget(self.capture_btn)

        ctrl_layout.addStretch()

        self.frame_counter_label = QLabel("已截取: 0 帧")
        self.frame_counter_label.setObjectName("StatusPill")
        self.frame_counter_label.setProperty("variant", "accent")
        ctrl_layout.addWidget(self.frame_counter_label)

        preview_layout.addLayout(ctrl_layout)
        splitter.addWidget(preview_widget)

        # Right: Extraction settings + captured list
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Auto extraction settings
        auto_group = QGroupBox("自动截帧设置")
        auto_form = QFormLayout()

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["按时间间隔", "按帧间隔", "场景变化检测"])
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        auto_form.addRow("模式:", self.mode_combo)

        self.interval_sec_spin = QDoubleSpinBox()
        self.interval_sec_spin.setRange(0.1, 300.0)
        self.interval_sec_spin.setValue(1.0)
        self.interval_sec_spin.setSingleStep(0.5)
        self.interval_sec_spin.setSuffix(" 秒")
        auto_form.addRow("时间间隔:", self.interval_sec_spin)

        self.interval_frame_spin = QSpinBox()
        self.interval_frame_spin.setRange(1, 10000)
        self.interval_frame_spin.setValue(30)
        self.interval_frame_spin.setSuffix(" 帧")
        auto_form.addRow("帧间隔:", self.interval_frame_spin)

        self.scene_thresh_spin = QDoubleSpinBox()
        self.scene_thresh_spin.setRange(1.0, 200.0)
        self.scene_thresh_spin.setValue(30.0)
        self.scene_thresh_spin.setSingleStep(5.0)
        auto_form.addRow("场景变化阈值:", self.scene_thresh_spin)

        self.dedup_check = QCheckBox("去重（跳过相似帧）")
        self.dedup_check.setChecked(True)
        auto_form.addRow(self.dedup_check)

        self.dedup_thresh_spin = QSpinBox()
        self.dedup_thresh_spin.setRange(1, 32)
        self.dedup_thresh_spin.setValue(8)
        auto_form.addRow("去重阈值:", self.dedup_thresh_spin)

        self.max_frames_spin = QSpinBox()
        self.max_frames_spin.setRange(0, 100000)
        self.max_frames_spin.setValue(0)
        self.max_frames_spin.setSpecialValueText("无限制")
        auto_form.addRow("最大帧数:", self.max_frames_spin)

        self.auto_extract_btn = QPushButton("开始自动截帧")
        self.auto_extract_btn.setEnabled(False)
        self.auto_extract_btn.setObjectName("PrimaryButton")
        self.auto_extract_btn.clicked.connect(self._start_auto_extract)
        auto_form.addRow(self.auto_extract_btn)

        self.extract_progress = QProgressBar()
        self.extract_progress.setValue(0)
        auto_form.addRow(self.extract_progress)

        auto_group.setLayout(auto_form)
        right_layout.addWidget(auto_group)

        # Captured frames list
        list_group = QGroupBox("已截取帧")
        list_layout = QVBoxLayout()
        self.captured_list = QListWidget()
        self.captured_list.setMaximumHeight(200)
        list_layout.addWidget(self.captured_list)

        list_btn_layout = QHBoxLayout()
        open_dir_btn = QPushButton("打开文件夹")
        open_dir_btn.clicked.connect(self._open_output_dir)
        list_btn_layout.addWidget(open_dir_btn)
        clear_btn = QPushButton("清空列表")
        clear_btn.clicked.connect(lambda: (self.captured_list.clear(), self._captured_paths.clear(),
                                           self._update_frame_counter()))
        list_btn_layout.addWidget(clear_btn)
        list_layout.addLayout(list_btn_layout)

        list_group.setLayout(list_layout)
        right_layout.addWidget(list_group)

        splitter.addWidget(right_widget)
        splitter.setSizes([700, 300])

        layout.addWidget(splitter)

        # ---- Bottom: Output + Confirm ----
        bottom_layout = QHBoxLayout()

        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("输出目录:"))
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("截帧保存位置（默认: 数据集目录下）")
        output_layout.addWidget(self.output_edit, stretch=1)
        output_browse = QPushButton("浏览...")
        output_browse.clicked.connect(self._browse_output)
        output_layout.addWidget(output_browse)
        bottom_layout.addLayout(output_layout)

        self.confirm_btn = QPushButton("加载到标注")
        self.confirm_btn.setObjectName("PrimaryButton")
        self.confirm_btn.clicked.connect(self._confirm_and_load)
        bottom_layout.addWidget(self.confirm_btn)

        self.close_btn = QPushButton("关闭")
        self.close_btn.clicked.connect(self.close)
        bottom_layout.addWidget(self.close_btn)

        layout.addLayout(bottom_layout)

        # Keyboard shortcuts
        capture_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        capture_shortcut.activated.connect(self._capture_current_frame)

        # Initialize mode visibility
        self._on_mode_changed(0)

    def _apply_dark_style(self):
        self.setStyleSheet(build_stylesheet())

    # ------------------------------------------------------------------
    # Local video
    # ------------------------------------------------------------------

    def _browse_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "",
            "视频文件 (*.mp4 *.avi *.mkv *.mov *.webm *.flv *.wmv);;所有文件 (*)"
        )
        if path:
            self.video_path_edit.setText(path)

    def _load_local_video(self):
        path = self.video_path_edit.text().strip()
        if not path or not os.path.isfile(path):
            QMessageBox.warning(self, "错误", "请选择有效的视频文件")
            return

        self._open_video(path)

    # ------------------------------------------------------------------
    # Online video
    # ------------------------------------------------------------------

    def _update_download_status(self):
        """Update download tool availability status."""
        ytdlp_ok = VideoFrameExtractor.is_ytdlp_available()
        youget_ok = VideoFrameExtractor.is_youget_available()

        if ytdlp_ok:
            self.ytdlp_status.setText("yt-dlp: 已安装")
            self.ytdlp_status.setStyleSheet(f"color: {Theme.ACCENT}; font-size: 11px;")
        else:
            self.ytdlp_status.setText("yt-dlp: 未安装 (pip install yt-dlp)")
            self.ytdlp_status.setStyleSheet(f"color: {Theme.DANGER}; font-size: 11px;")

        if youget_ok:
            self.youget_status.setText("you-get: 已安装")
            self.youget_status.setStyleSheet(f"color: {Theme.ACCENT}; font-size: 11px;")
        else:
            self.youget_status.setText("you-get: 未安装 (pip install you-get, 快手等备用)")
            self.youget_status.setStyleSheet(f"color: {Theme.TEXT_MUTED}; font-size: 11px;")

    def _on_url_changed(self, text: str):
        """Auto-detect platform when URL changes."""
        text = text.strip()
        if not text:
            self.detected_platform_label.setText("")
            return

        platform = VideoFrameExtractor.detect_platform(text)
        if platform == "其他":
            self.detected_platform_label.setText("未识别平台，将尝试通用下载")
            self.detected_platform_label.setProperty("variant", "warning")
        else:
            needs_cookie = platform in VideoFrameExtractor.COOKIE_PLATFORMS
            hint = f"检测到: {platform}"
            if needs_cookie:
                hint += " (如需登录内容请配置Cookie)"
            self.detected_platform_label.setText(hint)
            self.detected_platform_label.setProperty("variant", "accent")
        self.detected_platform_label.style().unpolish(self.detected_platform_label)
        self.detected_platform_label.style().polish(self.detected_platform_label)

    def _browse_cookie_file(self):
        """Browse for cookie file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择Cookie文件", "",
            "Cookie文件 (*.txt *.json);;所有文件 (*)"
        )
        if path:
            self.cookie_edit.setText(path)

    def _download_online_video(self):
        # Prevent double-click triggering duplicate downloads
        if self._download_worker is not None and self._download_worker.isRunning():
            return

        url = self.url_edit.text().strip()
        if not url:
            QMessageBox.warning(self, "错误", "请输入视频链接")
            return

        if not VideoFrameExtractor.is_ytdlp_available() and not VideoFrameExtractor.is_youget_available():
            QMessageBox.critical(
                self, "缺少依赖",
                "在线视频下载需要 yt-dlp 或 you-get。\n\n"
                "推荐安装:\n"
                "  pip install yt-dlp\n"
                "  pip install you-get  (快手等备用)"
            )
            return

        output_dir = self._get_output_dir()
        proxy = self.proxy_edit.text().strip()
        cookie_file = self.cookie_edit.text().strip()

        # Resolve cookie browser setting
        browser_choice = self.cookie_browser_combo.currentText()
        cookie_browser = ""
        if browser_choice == "自动获取":
            # Let download_video auto-detect and try browsers
            cookie_browser = ""
        else:
            cookie_browser = browser_choice.lower()

        self.download_btn.setEnabled(False)
        platform = VideoFrameExtractor.detect_platform(url)
        self.info_label.setText(f"下载中 ({platform})...")

        self._download_worker = _VideoDownloadWorker(
            url, output_dir, proxy, cookie_file, cookie_browser
        )
        self._download_worker.progress.connect(
            lambda msg: self.info_label.setText(f"下载: {msg}")
        )
        self._download_worker.finished.connect(self._on_video_downloaded)
        self._download_worker.error.connect(self._on_download_error)
        self._download_worker.start()

    def _on_video_downloaded(self, path: str):
        self.download_btn.setEnabled(True)
        self.info_label.setText(f"下载完成: {path}")
        self._open_video(path)

    def _on_download_error(self, msg: str):
        self.download_btn.setEnabled(True)
        self.info_label.setText(f"下载失败")
        QMessageBox.critical(self, "下载失败", msg)

    # ------------------------------------------------------------------
    # Common video open
    # ------------------------------------------------------------------

    def _open_video(self, path: str):
        if self.extractor.is_opened():
            self._stop_playback()
            self.extractor.close()

        if not self.extractor.open(path):
            QMessageBox.critical(self, "错误", f"无法打开视频: {path}")
            return

        info = self.extractor.get_info()
        self.info_label.setText(
            f"{os.path.basename(path)} | "
            f"{info['width']}x{info['height']} | "
            f"{info['fps']:.1f}fps | "
            f"{info['duration_str']} | "
            f"{info['total_frames']} 帧"
        )
        self.duration_label.setText(info['duration_str'])

        self.play_btn.setEnabled(True)
        self.capture_btn.setEnabled(True)
        self.auto_extract_btn.setEnabled(True)

        # Show first frame
        frame = self.extractor.read_frame_at(0)
        if frame is not None:
            self._show_frame(frame)
            self._current_frame_idx = 0

    # ------------------------------------------------------------------
    # Preview / Playback
    # ------------------------------------------------------------------

    def _toggle_play(self):
        if not self.extractor.is_opened():
            return
        if self._is_playing:
            self._stop_playback()
        else:
            self._start_playback()

    def _start_playback(self):
        self._is_playing = True
        self.play_btn.setText("暂停")
        self._timer.start()

    def _stop_playback(self):
        self._is_playing = False
        self.play_btn.setText("播放")
        self._timer.stop()

    def _update_preview(self):
        if not self.extractor.is_opened():
            self._stop_playback()
            return

        frame = self.extractor.read_frame()
        if frame is None:
            self._stop_playback()
            return

        self._current_frame_idx = int(self.extractor.cap.get(cv2.CAP_PROP_POS_FRAMES))
        self._show_frame(frame)
        self._update_seek_position()

    def _show_frame(self, frame: np.ndarray):
        # Limit display size to reduce memory pressure
        h, w = frame.shape[:2]
        max_dim = 1280
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        # Use ascontiguousarray + data to avoid an extra copy (tobytes duplicates)
        rgb = np.ascontiguousarray(rgb)
        try:
            self._frame_bytes = rgb.data
        except MemoryError:
            import gc
            gc.collect()
            self._frame_bytes = rgb.tobytes()
        qimg = QImage(self._frame_bytes, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        # Scale to fit preview label
        label_size = self.preview_label.size()
        scaled = pixmap.scaled(
            label_size.width() - 4, label_size.height() - 4,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview_label.setPixmap(scaled)

    def _update_seek_position(self):
        if not self.extractor.is_opened() or self.extractor.total_frames <= 0:
            return
        if not self._seek_dragging:
            pos = int(self._current_frame_idx / self.extractor.total_frames * 1000)
            self.seek_slider.blockSignals(True)
            self.seek_slider.setValue(pos)
            self.seek_slider.blockSignals(False)

        # Update time label
        current_sec = self._current_frame_idx / self.extractor.fps if self.extractor.fps > 0 else 0
        self.time_label.setText(VideoFrameExtractor._format_duration(current_sec))

    def _on_seek_pressed(self):
        self._seek_dragging = True
        if self._is_playing:
            self._timer.stop()

    def _on_seek_released(self):
        self._seek_dragging = False
        if self.extractor.is_opened():
            target = int(self.seek_slider.value() / 1000 * self.extractor.total_frames)
            self.extractor.seek_frame(target)
            frame = self.extractor.read_frame()
            if frame is not None:
                self._current_frame_idx = target
                self._show_frame(frame)
        if self._is_playing:
            self._timer.start()

    def _on_seek_changed(self, value: int):
        if self._seek_dragging and self.extractor.is_opened():
            current_sec = (value / 1000 * self.extractor.total_frames) / self.extractor.fps
            self.time_label.setText(VideoFrameExtractor._format_duration(current_sec))

    # ------------------------------------------------------------------
    # Manual capture
    # ------------------------------------------------------------------

    def _capture_current_frame(self):
        if not self.extractor.is_opened():
            return

        output_dir = self._get_output_dir()
        # Use actual current position from VideoCapture (more reliable than cached index)
        path = self.extractor.save_current_frame(output_dir)
        if path:
            self._captured_paths.append(path)
            self._add_to_captured_list(path)
            self._capture_count += 1
            self._update_frame_counter()
            logger.info(f"Captured frame: {path}")

    def _add_to_captured_list(self, path: str):
        name = os.path.basename(path)
        item = QListWidgetItem(name)
        item.setData(Qt.ItemDataRole.UserRole, path)
        self.captured_list.addItem(item)
        self.captured_list.scrollToBottom()

    def _update_frame_counter(self):
        self.frame_counter_label.setText(f"已截取: {len(self._captured_paths)} 帧")

    # ------------------------------------------------------------------
    # Auto extraction
    # ------------------------------------------------------------------

    def _on_mode_changed(self, index: int):
        """Show/hide relevant settings based on mode."""
        is_interval = index in (0, 1)  # time or frame interval
        is_time = index == 0
        is_frame = index == 1
        is_scene = index == 2

        self.interval_sec_spin.setVisible(is_time)
        self.interval_frame_spin.setVisible(is_frame)
        self.scene_thresh_spin.setVisible(is_scene)

        # Update label visibility
        form = self.interval_sec_spin.parent().layout()
        if form:
            for i in range(form.count()):
                item = form.itemAt(i)
                if item and item.widget():
                    label = form.labelForField(item.widget())
                    if label:
                        if item.widget() is self.interval_sec_spin:
                            label.setVisible(is_time)
                        elif item.widget() is self.interval_frame_spin:
                            label.setVisible(is_frame)
                        elif item.widget() is self.scene_thresh_spin:
                            label.setVisible(is_scene)

    def _start_auto_extract(self):
        if not self.extractor.is_opened():
            return

        output_dir = self._get_output_dir()
        mode_map = {0: "interval", 1: "interval", 2: "scene"}
        mode = mode_map.get(self.mode_combo.currentIndex(), "interval")

        self._stop_playback()

        # Save current position
        current_path = self.extractor.video_path

        self.auto_extract_btn.setEnabled(False)
        self.auto_extract_btn.setText("截帧中...")
        self.extract_progress.setValue(0)

        self._extract_worker = _AutoExtractWorker(
            extractor=VideoFrameExtractor(),  # fresh instance for the thread
            output_dir=output_dir,
            mode=mode,
            interval_frames=self.interval_frame_spin.value(),
            interval_seconds=self.interval_sec_spin.value(),
            scene_threshold=self.scene_thresh_spin.value(),
            dedup=self.dedup_check.isChecked(),
            dedup_threshold=self.dedup_thresh_spin.value(),
            max_frames=self.max_frames_spin.value(),
        )
        # Set the video path on the new extractor
        self._extract_worker.extractor.video_path = current_path

        self._extract_worker.progress.connect(self._on_extract_progress)
        self._extract_worker.finished.connect(self._on_extract_finished)
        self._extract_worker.start()

    def _on_extract_progress(self, current: int, total: int):
        if total > 0:
            pct = int(current / total * 100)
            self.extract_progress.setValue(pct)
        self.info_label.setText(f"截帧中: {current}/{total} 帧")

    def _on_extract_finished(self, paths: list[str]):
        self.auto_extract_btn.setEnabled(True)
        self.auto_extract_btn.setText("开始自动截帧")
        self.extract_progress.setValue(100)

        for path in paths:
            if path not in self._captured_paths:
                self._captured_paths.append(path)
                self._add_to_captured_list(path)

        self._update_frame_counter()
        self.info_label.setText(f"截帧完成! 共 {len(paths)} 帧")

        # Re-open video for preview and show first frame
        if self.extractor.video_path:
            self.extractor.open(self.extractor.video_path)
            frame = self.extractor.read_frame_at(0)
            if frame is not None:
                self._current_frame_idx = 0
                self._show_frame(frame)
                self._update_seek_position()

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------

    def _get_output_dir(self) -> str:
        path = self.output_edit.text().strip()
        if not path:
            # Default: data/raw/video_name/
            video_name = "frames"
            if self.extractor.video_path:
                video_name = Path(self.extractor.video_path).stem
            path = os.path.join(os.getcwd(), "data", "raw", video_name)
            self.output_edit.setText(path)
        os.makedirs(path, exist_ok=True)
        self._output_dir = path
        return path

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if path:
            self.output_edit.setText(path)

    def _open_output_dir(self):
        output_dir = self._output_dir or self._get_output_dir()
        if os.path.isdir(output_dir):
            QDesktopServices.openUrl(QUrl.fromLocalFile(output_dir))

    # ------------------------------------------------------------------
    # Confirm and load
    # ------------------------------------------------------------------

    def _confirm_and_load(self):
        if not self._captured_paths:
            QMessageBox.warning(self, "提示", "还没有截取任何帧")
            return

        # Emit signal with captured paths
        self.frames_captured.emit(self._captured_paths)
        self.info_label.setText(f"已加载 {len(self._captured_paths)} 帧到标注面板")
        self.close()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        self._stop_playback()
        self.extractor.close()
        super().closeEvent(event)
