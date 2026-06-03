"""Inference panel for YOLO model prediction."""

import os
import csv
import time
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QComboBox, QDoubleSpinBox, QPushButton, QLineEdit,
    QTextEdit, QFileDialog, QCheckBox, QSplitter, QMessageBox,
    QProgressBar, QTabWidget, QSlider, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QMutex, QSize
from PyQt6.QtGui import QImage, QPixmap, QFont, QIcon

import cv2
import numpy as np

from gui.ui_components import StatusPill


class InferenceWorker(QThread):
    """Worker thread for video/webcam inference.

    Supports:
    - Pause / resume via ``pause()`` / ``resume()``
    - Seek to a specific frame via ``seek_frame(frame_idx)``
    - Playback speed control via ``set_speed(rate)``
    - Frame-level progress reporting for video files

    Performance notes:
    - ``half`` and ``imgsz`` are read from the inferencer's config so
      FP16 / resolution tuning take effect automatically.
    - FPS is computed from **inference-only** time (not including frame
      display) for a more accurate measurement.
    """
    frame_ready = pyqtSignal(np.ndarray, dict)
    progress_ready = pyqtSignal(int, int)  # current_frame, total_frames
    finished = pyqtSignal()

    def __init__(self, inferencer, source, is_webcam=False, **kwargs):
        super().__init__()
        self.inferencer = inferencer
        self.source = source
        self.is_webcam = is_webcam
        self.kwargs = kwargs
        self._running = True
        self._running_mutex = QMutex()
        self._fps_samples = []

        # Playback controls
        self._paused = False
        self._pause_mutex = QMutex()
        self._speed = 1.0          # 1.0 = normal speed
        self._seek_frame = -1      # -1 = no seek pending
        self._total_frames = 0

    # ---- Playback control API (called from main thread) ----

    def _is_running(self) -> bool:
        """Thread-safe check if worker should keep running."""
        self._running_mutex.lock()
        try:
            return self._running
        finally:
            self._running_mutex.unlock()

    def pause(self):
        self._pause_mutex.lock()
        try:
            self._paused = True
        finally:
            self._pause_mutex.unlock()

    def resume(self):
        self._pause_mutex.lock()
        self._paused = False
        self._pause_mutex.unlock()

    def toggle_pause(self):
        """Toggle pause state. Returns True if now paused."""
        self._pause_mutex.lock()
        try:
            if self._paused:
                self._paused = False
                return False
            else:
                self._paused = True
                return True
        finally:
            self._pause_mutex.unlock()

    def is_paused(self) -> bool:
        self._pause_mutex.lock()
        try:
            return self._paused
        finally:
            self._pause_mutex.unlock()

    def set_speed(self, rate: float):
        """Set playback speed multiplier (0.25 - 4.0)."""
        self._speed = max(0.25, min(4.0, rate))

    def seek_frame(self, frame_idx: int):
        """Request seeking to a specific frame (0-based)."""
        self._seek_frame = max(0, frame_idx)
        # If paused, wake up so seek takes effect
        if self.is_paused():
            self.resume()

    # ---- Main loop ----

    def run(self):
        try:
            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                return

            self._total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps_source = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_interval = 1.0 / fps_source  # seconds per frame at 1x

            frame_idx = 0

            while cap.isOpened() and self._is_running():
                # --- Handle seek ---
                if self._seek_frame >= 0:
                    target = min(self._seek_frame, self._total_frames - 1)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                    frame_idx = target
                    self._seek_frame = -1

                # --- Handle pause ---
                if self.is_paused():
                    # Busy-wait with small sleep (responsive to resume)
                    while self.is_paused() and self._is_running():
                        time.sleep(0.05)
                    if not self._is_running():
                        break

                # --- Read frame ---
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

                # --- Inference (timed) ---
                infer_start = time.perf_counter()
                results = self.inferencer.predict_frame(frame, **self.kwargs)
                infer_elapsed = max(time.perf_counter() - infer_start, 1e-6)

                # --- Post-processing ---
                result0 = results["results"][0]
                annotated = result0.plot().copy()  # copy to make it writable
                detections = self.inferencer.get_detections(results["results"])

                # --- FPS overlay (smoothed) ---
                fps = 1.0 / infer_elapsed
                self._fps_samples.append(fps)
                if len(self._fps_samples) > 30:
                    self._fps_samples.pop(0)
                avg_fps = sum(self._fps_samples) / len(self._fps_samples)

                # Draw FPS + device tag on frame
                device_tag = self.inferencer._device if self.inferencer._device else "auto"
                half_tag = "FP16" if self.inferencer._half else "FP32"
                label = f"FPS: {avg_fps:.1f}  |  {device_tag} {half_tag}"

                cv2.putText(
                    annotated, label, (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA,
                )
                cv2.putText(
                    annotated, label, (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 210, 130), 2, cv2.LINE_AA,
                )

                self.frame_ready.emit(annotated, {
                    "detections": detections,
                    "fps": avg_fps,
                    "elapsed": infer_elapsed,
                    "frame_idx": frame_idx,
                    "total_frames": self._total_frames,
                })

                if self._total_frames > 0:
                    self.progress_ready.emit(frame_idx, self._total_frames)

                # --- Speed control: throttle to match video playback rate ---
                if self._speed > 0 and not self.is_webcam:
                    # Compute how long we should wait before next frame
                    target_interval = frame_interval / self._speed
                    wait_time = target_interval - infer_elapsed
                    if wait_time > 0:
                        time.sleep(wait_time)

            cap.release()
        except Exception as e:
            from loguru import logger
            logger.error(f"Inference worker error: {e}")
        finally:
            self.finished.emit()

    def stop(self):
        self._running_mutex.lock()
        self._running = False
        self._running_mutex.unlock()
        self.resume()  # Wake up if paused


class BatchInferenceWorker(QThread):
    """Worker thread for batch image inference."""
    progress = pyqtSignal(int, int, str)  # current, total, filename
    result_ready = pyqtSignal(str, list, float)  # image_path, detections, elapsed
    finished = pyqtSignal(int)  # total processed

    def __init__(self, inferencer, image_paths, output_dir, **kwargs):
        super().__init__()
        self.inferencer = inferencer
        self.image_paths = image_paths
        self.output_dir = output_dir
        self.kwargs = kwargs
        self._running = True
        self._running_mutex = QMutex()

    def _is_running(self) -> bool:
        self._running_mutex.lock()
        try:
            return self._running
        finally:
            self._running_mutex.unlock()

    def run(self):
        processed = 0
        total = len(self.image_paths)

        for i, img_path in enumerate(self.image_paths):
            if not self._is_running():
                break

            try:
                result = self.inferencer.predict_image(img_path, **self.kwargs)
                results = result["results"]
                elapsed = result["elapsed"]
                detections = self.inferencer.get_detections(results)

                # Save annotated image
                if self.output_dir:
                    annotated = results[0].plot().copy()
                    out_name = os.path.splitext(os.path.basename(img_path))[0] + "_det.jpg"
                    out_path = os.path.join(self.output_dir, out_name)
                    cv2.imwrite(out_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

                self.progress.emit(i + 1, total, os.path.basename(img_path))
                self.result_ready.emit(img_path, detections, elapsed)
                processed += 1

            except Exception as e:
                from loguru import logger
                logger.error(f"Batch inference error on {img_path}: {e}")

        self.finished.emit(processed)

    def stop(self):
        self._running_mutex.lock()
        self._running = False
        self._running_mutex.unlock()


class InferencePanel(QWidget):
    """Panel for running YOLO inference."""

    def __init__(self, config_manager=None, parent=None):
        super().__init__(parent)
        self.config = config_manager
        self.inferencer = None
        self.worker = None
        self.batch_worker = None
        self.current_image = None
        self.batch_results = []
        self._is_video = False  # Track whether current source is video
        self._video_fps = 30.0    # Cached FPS of current video
        self.init_ui()

    def _load_icon(self, name: str) -> QIcon:
        """Load icon from resources/icons folder."""
        from freeze import get_resource_path
        svg_path = get_resource_path(f"resources/icons/{name}.svg")
        if svg_path.exists():
            return QIcon(str(svg_path))
        return QIcon()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # =====================================================================
        # Compact top toolbar: model + settings + source controls in one row
        # =====================================================================
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        # --- Model path + load ---
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("模型文件 (.pt)")
        self.model_path_edit.setMinimumWidth(200)
        toolbar.addWidget(self.model_path_edit)

        model_browse = QPushButton("浏览")
        model_browse.setFixedWidth(50)
        model_browse.clicked.connect(self.browse_model)
        toolbar.addWidget(model_browse)

        self.load_model_btn = QPushButton("加载")
        self.load_model_btn.setFixedWidth(50)
        self.load_model_btn.clicked.connect(self.load_model)
        toolbar.addWidget(self.load_model_btn)

        toolbar.addWidget(QLabel("|"))

        # --- Recent models dropdown ---
        self.recent_model_combo = QComboBox()
        self.recent_model_combo.setPlaceholderText("最近训练...")
        self.recent_model_combo.setMinimumWidth(160)
        self._refresh_recent_models()
        self.recent_model_combo.currentTextChanged.connect(self._on_recent_model_selected)
        toolbar.addWidget(self.recent_model_combo)

        refresh_btn = QPushButton("刷新")
        refresh_btn.setFixedWidth(50)
        refresh_btn.setToolTip("刷新最近模型列表")
        refresh_btn.clicked.connect(self._refresh_recent_models)
        toolbar.addWidget(refresh_btn)

        toolbar.addWidget(QLabel("|"))

        # --- Inference params (compact) ---
        toolbar.addWidget(QLabel("置信度"))
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 1.0)
        self.conf_spin.setValue(0.25)
        self.conf_spin.setDecimals(2)
        self.conf_spin.setFixedWidth(60)
        toolbar.addWidget(self.conf_spin)

        toolbar.addWidget(QLabel("IoU"))
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.01, 1.0)
        self.iou_spin.setValue(0.7)
        self.iou_spin.setDecimals(2)
        self.iou_spin.setFixedWidth(60)
        toolbar.addWidget(self.iou_spin)

        toolbar.addWidget(QLabel("分辨率"))
        self.imgsz_combo = QComboBox()
        self.imgsz_combo.addItems(["320", "416", "640", "1280"])
        self.imgsz_combo.setCurrentIndex(2)
        self.imgsz_combo.setFixedWidth(60)
        self.imgsz_combo.setToolTip("推理分辨率")
        toolbar.addWidget(self.imgsz_combo)

        self.half_check = QCheckBox("FP16")
        self.half_check.setChecked(True)
        self.half_check.setToolTip("GPU 半精度加速")
        toolbar.addWidget(self.half_check)

        toolbar.addStretch()

        # --- Source buttons (plain text, no emoji) ---
        self.image_btn = QPushButton("图片")
        self.image_btn.clicked.connect(self.predict_image)
        toolbar.addWidget(self.image_btn)

        self.video_btn = QPushButton("视频")
        self.video_btn.clicked.connect(self.predict_video)
        toolbar.addWidget(self.video_btn)

        self.webcam_btn = QPushButton("摄像头")
        self.webcam_btn.clicked.connect(self.predict_webcam)
        toolbar.addWidget(self.webcam_btn)

        self.stop_btn = QPushButton("停止")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_inference)
        toolbar.addWidget(self.stop_btn)

        self.runtime_status_label = StatusPill("就绪")
        toolbar.addWidget(self.runtime_status_label)

        layout.addLayout(toolbar)

        # --- Device info bar (thin, below toolbar) ---
        dev_bar = QHBoxLayout()
        self.device_info_label = QLabel("设备: 检测中...")
        self.device_info_label.setStyleSheet("color: #8b949e; font-size: 11px;")
        dev_bar.addWidget(self.device_info_label)
        dev_bar.addStretch()
        layout.addLayout(dev_bar)
        self._detect_device_info()

        # =====================================================================
        # Main content: preview (large) + results sidebar
        # =====================================================================
        self.inference_tabs = QTabWidget()

        # Single inference tab
        single_tab = QWidget()
        single_layout = QVBoxLayout(single_tab)
        single_layout.setContentsMargins(0, 0, 0, 0)
        single_layout.setSpacing(6)

        # --- Preview + Results splitter ---
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(6)

        # Preview container (so we can add video controls below it)
        preview_container = QWidget()
        preview_layout = QVBoxLayout(preview_container)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(4)

        # Image display — THIS is the big preview area
        self.image_label = QLabel("加载图片或视频开始推理")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.image_label.setObjectName("PreviewSurface")
        preview_layout.addWidget(self.image_label, stretch=1)

        # Video controls (inline, below preview, no GroupBox border)
        self.video_control_widget = QWidget()
        self.video_control_widget.setVisible(False)
        vc_layout = QVBoxLayout(self.video_control_widget)
        vc_layout.setContentsMargins(4, 4, 4, 4)
        vc_layout.setSpacing(4)

        # Progress slider row
        slider_row = QHBoxLayout()
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.setValue(0)
        self.frame_slider.setToolTip("拖动跳转到指定帧")
        self.frame_slider.sliderPressed.connect(self._on_slider_pressed)
        self.frame_slider.sliderReleased.connect(self._on_slider_released)
        self.frame_slider.valueChanged.connect(self._on_slider_value_changed)
        slider_row.addWidget(self.frame_slider)

        self.frame_label = QLabel("0 / 0")
        self.frame_label.setFixedWidth(100)
        self.frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        slider_row.addWidget(self.frame_label)
        vc_layout.addLayout(slider_row)

        # Buttons row with icons
        btn_row = QHBoxLayout()
        self.skip_back_btn = QPushButton()
        self.skip_back_btn.setIcon(self._load_icon("backward"))
        self.skip_back_btn.setIconSize(QSize(18, 18))
        self.skip_back_btn.setFixedSize(32, 28)
        self.skip_back_btn.setToolTip("后退 30 帧")
        self.skip_back_btn.clicked.connect(self._skip_backward)
        btn_row.addWidget(self.skip_back_btn)

        self.play_pause_btn = QPushButton()
        self.play_pause_btn.setIcon(self._load_icon("play"))
        self.play_pause_btn.setIconSize(QSize(18, 18))
        self.play_pause_btn.setFixedSize(36, 28)
        self.play_pause_btn.setToolTip("暂停 / 继续")
        self.play_pause_btn.clicked.connect(self._toggle_pause)
        btn_row.addWidget(self.play_pause_btn)

        self.skip_fwd_btn = QPushButton()
        self.skip_fwd_btn.setIcon(self._load_icon("forward"))
        self.skip_fwd_btn.setIconSize(QSize(18, 18))
        self.skip_fwd_btn.setFixedSize(32, 28)
        self.skip_fwd_btn.setToolTip("前进 30 帧")
        self.skip_fwd_btn.clicked.connect(self._skip_forward)
        btn_row.addWidget(self.skip_fwd_btn)

        btn_row.addWidget(QLabel("速度:"))
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.25x", "0.5x", "1x", "1.5x", "2x", "4x"])
        self.speed_combo.setCurrentIndex(2)
        self.speed_combo.setFixedWidth(70)
        self.speed_combo.currentIndexChanged.connect(self._on_speed_changed)
        btn_row.addWidget(self.speed_combo)

        btn_row.addStretch()

        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setStyleSheet("color: #8b949e; font-size: 11px;")
        btn_row.addWidget(self.time_label)
        vc_layout.addLayout(btn_row)

        preview_layout.addWidget(self.video_control_widget)
        splitter.addWidget(preview_container)

        # Results sidebar
        results_widget = QWidget()
        results_widget.setMinimumWidth(200)
        results_widget.setMaximumWidth(350)
        results_layout = QVBoxLayout(results_widget)
        results_layout.setContentsMargins(6, 6, 6, 6)

        results_header = QLabel("检测结果")
        results_header.setStyleSheet("font-weight: bold; font-size: 12px;")
        results_layout.addWidget(results_header)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setFont(QFont("monospace", 9))
        results_layout.addWidget(self.results_text)

        splitter.addWidget(results_widget)
        splitter.setSizes([900, 260])
        splitter.setStretchFactor(0, 1)  # preview expands
        splitter.setStretchFactor(1, 0)  # sidebar fixed-ish

        single_layout.addWidget(splitter, stretch=1)
        self.inference_tabs.addTab(single_tab, "单张推理")

        # Batch inference tab
        batch_tab = QWidget()
        batch_layout = QVBoxLayout(batch_tab)

        # Batch source
        batch_source_group = QGroupBox("批量推理")
        batch_source_layout = QVBoxLayout()

        row1 = QHBoxLayout()
        self.batch_folder_edit = QLineEdit()
        self.batch_folder_edit.setPlaceholderText("选择图片文件夹...")
        batch_browse = QPushButton("浏览...")
        batch_browse.clicked.connect(self.browse_batch_folder)
        row1.addWidget(self.batch_folder_edit)
        row1.addWidget(batch_browse)
        batch_source_layout.addLayout(row1)

        row2 = QHBoxLayout()
        self.batch_output_edit = QLineEdit()
        self.batch_output_edit.setPlaceholderText("输出目录 (可选)...")
        output_browse = QPushButton("浏览...")
        output_browse.clicked.connect(self.browse_batch_output)
        row2.addWidget(self.batch_output_edit)
        row2.addWidget(output_browse)
        batch_source_layout.addLayout(row2)

        row3 = QHBoxLayout()
        self.batch_start_btn = QPushButton("开始批量推理")
        self.batch_start_btn.setObjectName("PrimaryButton")
        self.batch_start_btn.clicked.connect(self.start_batch_inference)

        self.batch_stop_btn = QPushButton("停止")
        self.batch_stop_btn.setObjectName("DangerButton")
        self.batch_stop_btn.setEnabled(False)
        self.batch_stop_btn.clicked.connect(self.stop_batch_inference)

        self.batch_export_btn = QPushButton("导出结果 (CSV)")
        self.batch_export_btn.setEnabled(False)
        self.batch_export_btn.clicked.connect(self.export_batch_results)

        row3.addWidget(self.batch_start_btn)
        row3.addWidget(self.batch_stop_btn)
        row3.addWidget(self.batch_export_btn)
        batch_source_layout.addLayout(row3)

        batch_source_group.setLayout(batch_source_layout)
        batch_layout.addWidget(batch_source_group)

        # Batch progress
        self.batch_progress = QProgressBar()
        self.batch_progress.setRange(0, 100)
        self.batch_progress.setValue(0)
        batch_layout.addWidget(self.batch_progress)

        self.batch_status_label = QLabel("就绪")
        batch_layout.addWidget(self.batch_status_label)

        # Batch results
        self.batch_results_text = QTextEdit()
        self.batch_results_text.setReadOnly(True)
        self.batch_results_text.setFont(QFont("monospace", 9))
        batch_layout.addWidget(self.batch_results_text)

        self.inference_tabs.addTab(batch_tab, "批量推理")

        layout.addWidget(self.inference_tabs)

        # Internal state for slider seeking
        self._slider_pressed = False

    # ---- Video control handlers ----

    def _on_slider_pressed(self):
        """User started dragging the slider — pause to avoid conflicts."""
        self._slider_pressed = True
        self._was_playing_before_slider = self.worker and not self.worker.is_paused()
        if self._was_playing_before_slider:
            self.worker.pause()
            self.play_pause_btn.setText("继续")

    def _on_slider_released(self):
        """User released the slider — seek to the position."""
        if self.worker and self._is_video:
            target_frame = self.frame_slider.value()
            self.worker.seek_frame(target_frame)
            if self._was_playing_before_slider:
                self.worker.resume()
                self.play_pause_btn.setText("暂停")
        self._slider_pressed = False

    def _on_slider_value_changed(self, value):
        """Update frame/time label as slider moves."""
        total = self.frame_slider.maximum()
        self.frame_label.setText(f"{value} / {total}")
        fps = self._video_fps or 30.0
        cur_s = value / fps
        tot_s = total / fps
        self.time_label.setText(
            f"{int(cur_s//60):02d}:{int(cur_s%60):02d} / "
            f"{int(tot_s//60):02d}:{int(tot_s%60):02d}"
        )

    def _toggle_pause(self):
        """Toggle play/pause."""
        if not self.worker:
            return
        now_paused = self.worker.toggle_pause()
        self.play_pause_btn.setText("继续" if now_paused else "暂停")

    def _skip_backward(self):
        """Skip 30 frames backward."""
        if not self.worker or not self._is_video:
            return
        current = self.frame_slider.value()
        target = max(0, current - 30)
        self.worker.seek_frame(target)

    def _skip_forward(self):
        """Skip 30 frames forward."""
        if not self.worker or not self._is_video:
            return
        current = self.frame_slider.value()
        total = self.frame_slider.maximum()
        target = min(total, current + 30)
        self.worker.seek_frame(target)

    def _on_speed_changed(self, index):
        """Handle speed combo change."""
        speed_map = [0.25, 0.5, 1.0, 1.5, 2.0, 4.0]
        if 0 <= index < len(speed_map) and self.worker:
            self.worker.set_speed(speed_map[index])

    # ---- Model & config methods ----

    def browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择模型", "", "模型文件 (*.pt *.onnx *.engine);;所有文件 (*)"
        )
        if path:
            self.model_path_edit.setText(path)

    def set_model_path(self, path: str):
        """Set the model path from external source (e.g. training panel)."""
        self.model_path_edit.setText(path)

    def load_model_from_path(self, path: str):
        """Set model path and automatically load the model."""
        self.model_path_edit.setText(path)
        self.load_model()

    def _refresh_recent_models(self):
        """Scan runs/ directory for recently trained best.pt files."""
        self.recent_model_combo.blockSignals(True)
        self.recent_model_combo.clear()
        self.recent_model_combo.addItem("", "")  # 默认空选项

        # Search common training output directories
        search_dirs = []
        for root in ["runs", os.path.join(os.getcwd(), "runs")]:
            if os.path.isdir(root):
                for sub in os.listdir(root):
                    sub_path = os.path.join(root, sub)
                    if os.path.isdir(sub_path):
                        search_dirs.append(sub_path)
                        # Also check nested exp dirs
                        for nested in os.listdir(sub_path):
                            nested_path = os.path.join(sub_path, nested)
                            if os.path.isdir(nested_path):
                                search_dirs.append(nested_path)

        candidates = []
        for d in search_dirs:
            best_pt = os.path.join(d, "weights", "best.pt")
            last_pt = os.path.join(d, "weights", "last.pt")
            for pt in [best_pt, last_pt]:
                if os.path.isfile(pt):
                    # Get modification time for sorting
                    mtime = os.path.getmtime(pt)
                    # Create a display name from the directory structure
                    rel = os.path.relpath(pt, os.getcwd())
                    candidates.append((mtime, rel, pt))

        # Sort by most recent first
        candidates.sort(key=lambda x: x[0], reverse=True)

        for mtime, rel, full_path in candidates[:20]:  # Keep top 20
            import time as _time
            timestr = _time.strftime("%Y-%m-%d %H:%M", _time.localtime(mtime))
            display = f"{rel}  ({timestr})"
            self.recent_model_combo.addItem(display, full_path)

        self.recent_model_combo.blockSignals(False)

    def _on_recent_model_selected(self, text: str):
        """When user selects a recent model from the dropdown, set the path."""
        path = self.recent_model_combo.currentData()
        if path:
            self.model_path_edit.setText(path)

    def _detect_device_info(self):
        """Detect and display GPU/CPU device info."""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
                self.device_info_label.setText(f"GPU: {gpu_name} ({vram:.1f}GB)")
                self.device_info_label.setStyleSheet("color: #3fb950; font-size: 11px;")
            else:
                self.device_info_label.setText("CUDA 不可用，将使用 CPU")
                self.device_info_label.setStyleSheet("color: #f85149; font-size: 11px;")
                self.half_check.setChecked(False)
                self.half_check.setEnabled(False)
        except ImportError:
            self.device_info_label.setText("PyTorch 未安装")
            self.device_info_label.setStyleSheet("color: #f85149; font-size: 11px;")
            self.half_check.setChecked(False)
            self.half_check.setEnabled(False)

    def _get_imgsz_value(self) -> int:
        """Get the numeric imgsz from the combo box selection."""
        text = self.imgsz_combo.currentText()
        # Extract leading number: "320 (最快)" → 320
        return int(text.split()[0])

    def load_model(self):
        from core.inference import YOLOInference
        model_path = self.model_path_edit.text()
        if not model_path or not os.path.exists(model_path):
            QMessageBox.warning(self, "错误", "请选择有效的模型文件")
            return

        try:
            self.inferencer = YOLOInference(self.config.config if self.config else None)

            # Apply UI performance settings before loading
            imgsz = self._get_imgsz_value()
            half = self.half_check.isChecked()

            # Auto-detect device
            try:
                import torch
                device = "0" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

            # Override config with UI selections
            if self.config:
                self.config.config.inference.imgsz = imgsz
                self.config.config.inference.half = half

            self.inferencer.load_model(model_path, device=device)

            # Update device info after load (shows actual device used)
            dev_info = self.inferencer.get_device_info()
            gpu_name = dev_info.get("gpu_name", "")
            if "N/A" not in gpu_name and gpu_name:
                self.device_info_label.setText(
                    f"GPU: {gpu_name}  |  {dev_info['device']}  FP{'16' if dev_info['half'] else '32'}"
                )
                self.device_info_label.setStyleSheet("color: #3fb950; font-size: 11px;")
            else:
                self.device_info_label.setText("CPU 模式  FP32")
                self.device_info_label.setStyleSheet("color: #d29922; font-size: 11px;")

            QMessageBox.information(
                self, "成功",
                f"模型加载成功: {model_path}\n"
                f"设备: {dev_info['device']}  FP{'16' if dev_info['half'] else '32'}  分辨率: {dev_info['imgsz']}"
            )
        except Exception as e:
            QMessageBox.critical(self, "加载失败", str(e))

    def get_predict_args(self) -> dict:
        return {
            "conf": self.conf_spin.value(),
            "iou": self.iou_spin.value(),
            "max_det": 300,
            "imgsz": self._get_imgsz_value(),
            "half": self.half_check.isChecked(),
        }

    def predict_image(self):
        if not self.inferencer:
            QMessageBox.warning(self, "错误", "请先加载模型")
            return

        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "图片 (*.jpg *.jpeg *.png *.bmp);;所有文件 (*)"
        )
        if not path:
            return

        try:
            result = self.inferencer.predict_image(path, **self.get_predict_args())
            results = result["results"]

            # Display annotated image
            annotated = results[0].plot().copy()
            self.display_frame(annotated)

            # Show detections
            detections = self.inferencer.get_detections(results)
            self.show_detections(detections, result["elapsed"])

            # Hide video controls for image mode
            self.video_control_widget.setVisible(False)

        except Exception as e:
            QMessageBox.critical(self, "推理失败", str(e))

    def predict_video(self):
        if not self.inferencer:
            QMessageBox.warning(self, "错误", "请先加载模型")
            return

        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", "", "视频 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*)"
        )
        if not path:
            return

        self._is_video = True
        self.stop_btn.setEnabled(True)
        self.image_btn.setEnabled(False)
        self.video_btn.setEnabled(False)
        self.webcam_btn.setEnabled(False)
        self.runtime_status_label.setText("推理中...")
        self.runtime_status_label.set_variant("warning")

        # Show video controls
        self.video_control_widget.setVisible(True)
        self.play_pause_btn.setText("暂停")
        self.speed_combo.setCurrentIndex(2)  # 1x

        # Get video info for slider
        cap = cv2.VideoCapture(path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fps = self._video_fps
        cap.release()

        self.frame_slider.setRange(0, max(total_frames - 1, 0))
        self.frame_slider.setValue(0)
        total_s = total_frames / fps if fps > 0 else 0
        self.time_label.setText(f"00:00 / {int(total_s//60):02d}:{int(total_s%60):02d}")
        self.frame_label.setText(f"0 / {total_frames}")

        self.stop_inference()
        self._cleanup_worker()
        self.worker = InferenceWorker(
            self.inferencer, path, **self.get_predict_args()
        )
        self.worker.frame_ready.connect(self.on_frame_ready)
        self.worker.progress_ready.connect(self.on_video_progress)
        self.worker.finished.connect(self.on_inference_finished)
        self.worker.start()

    def predict_webcam(self):
        if not self.inferencer:
            QMessageBox.warning(self, "错误", "请先加载模型")
            return

        self._is_video = False
        self.stop_btn.setEnabled(True)
        self.image_btn.setEnabled(False)
        self.video_btn.setEnabled(False)
        self.webcam_btn.setEnabled(False)
        self.runtime_status_label.setText("推理中...")
        self.runtime_status_label.set_variant("warning")

        # Hide video controls for webcam (no seeking)
        self.video_control_widget.setVisible(False)

        self.stop_inference()
        self._cleanup_worker()
        self.worker = InferenceWorker(
            self.inferencer, 0, is_webcam=True, **self.get_predict_args()
        )
        self.worker.frame_ready.connect(self.on_frame_ready)
        self.worker.finished.connect(self.on_inference_finished)
        self.worker.start()

    def stop_inference(self):
        if self.worker:
            self.worker.stop()

    def _cleanup_worker(self):
        if self.worker is not None:
            if self.worker.isRunning():
                self.worker.stop()
                if not self.worker.wait(3000):
                    self.worker.terminate()
                    self.worker.wait(1000)
            self.worker.deleteLater()
            self.worker = None

    def _cleanup_batch_worker(self):
        if self.batch_worker is not None:
            if self.batch_worker.isRunning():
                self.batch_worker.stop()
                if not self.batch_worker.wait(3000):
                    self.batch_worker.terminate()
                    self.batch_worker.wait(1000)
            self.batch_worker.deleteLater()
            self.batch_worker = None

    def on_frame_ready(self, frame: np.ndarray, info: dict):
        self.display_frame(frame)
        detections = info.get("detections", [])
        elapsed = info.get("elapsed", 0)
        # FPS is drawn inside the video frame — don't duplicate outside
        self.runtime_status_label.setText(f"检测中 | 目标: {len(detections)}")
        self.runtime_status_label.set_variant("accent")
        self.show_detections(detections, elapsed=elapsed)

    def on_video_progress(self, current_frame: int, total_frames: int):
        """Update slider position (without triggering seek)."""
        if not self._slider_pressed:
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(min(current_frame, self.frame_slider.maximum()))
            self.frame_slider.blockSignals(False)
            self.frame_label.setText(f"{current_frame} / {total_frames}")

            # Update time label using cached FPS
            fps = self._video_fps or 30.0
            cur_s = current_frame / fps if fps > 0 else 0
            tot_s = total_frames / fps if fps > 0 else 0
            self.time_label.setText(
                f"{int(cur_s//60):02d}:{int(cur_s%60):02d} / "
                f"{int(tot_s//60):02d}:{int(tot_s%60):02d}"
            )

    def on_inference_finished(self):
        self.stop_btn.setEnabled(False)
        self.image_btn.setEnabled(True)
        self.video_btn.setEnabled(True)
        self.webcam_btn.setEnabled(True)
        self.runtime_status_label.setText("就绪")
        self.runtime_status_label.set_variant("")
        self.play_pause_btn.setText("暂停")
        self._cleanup_worker()

    def display_frame(self, frame: np.ndarray):
        """Display a frame (numpy BGR array) in the image label.

        Strategy to avoid MemoryError on low-memory systems:
          1. Resize the frame *before* colour conversion & bytes extraction
             so that the intermediate buffers are much smaller.
          2. Use ``np.ascontiguousarray().data`` instead of ``.tobytes()``
             to avoid an extra copy when the array is already contiguous.
          3. If we still hit MemoryError, force a GC and retry once.
        """
        import gc

        # Limit display size — 1280 px on the longest side is plenty for
        # a preview and keeps the QImage buffer under ~5 MB (vs 4K = ~24 MB).
        max_dim = 1280
        h, w = frame.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        # Colour conversion
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            rgb = frame

        h, w, ch = rgb.shape
        bytes_per_line = ch * w

        # Ensure the array is contiguous so QImage can reference it directly
        rgb = np.ascontiguousarray(rgb)

        try:
            self._frame_bytes = rgb.data  # memoryview — no extra copy
        except MemoryError:
            gc.collect()
            self._frame_bytes = rgb.tobytes()  # fallback (copies)

        q_img = QImage(self._frame_bytes, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        scaled = pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)

    def show_detections(self, detections: list, elapsed: float = 0):
        self.results_text.clear()
        if elapsed > 0:
            self.results_text.append(f"推理耗时: {elapsed*1000:.1f}ms\n")
        self.results_text.append(f"检测到 {len(detections)} 个目标:\n")

        for i, det in enumerate(detections, 1):
            det_type = det.get("type", "bbox")
            conf = det.get('confidence', 0)
            class_name = det.get('class_name', 'unknown')

            if det_type == "obb":
                corners = det.get("corners", [])
                corners_str = ", ".join(f"({c[0]:.0f},{c[1]:.0f})" for c in corners) if corners else "N/A"
                self.results_text.append(
                    f"{i}. [{det_type.upper()}] {class_name} - 置信度: {conf:.2%}\n"
                    f"   角点: {corners_str}\n"
                )
            elif det_type == "keypoint":
                bbox = det.get('bbox', {})
                if isinstance(bbox, dict):
                    pos_str = f"({bbox.get('x1', 0):.0f}, {bbox.get('y1', 0):.0f}) - ({bbox.get('x2', 0):.0f}, {bbox.get('y2', 0):.0f})"
                else:
                    pos_str = "N/A"
                kps = det.get("keypoints", [])
                kp_str = f"{len(kps)}个关键点" if kps else ""
                self.results_text.append(
                    f"{i}. [{det_type.upper()}] {class_name} - 置信度: {conf:.2%}\n"
                    f"   位置: {pos_str}  {kp_str}\n"
                )
            else:
                # Standard bbox
                bbox = det.get('bbox', {})
                if isinstance(bbox, dict):
                    self.results_text.append(
                        f"{i}. {class_name} - 置信度: {conf:.2%}\n"
                        f"   位置: ({bbox.get('x1', 0):.0f}, {bbox.get('y1', 0):.0f}) - "
                        f"({bbox.get('x2', 0):.0f}, {bbox.get('y2', 0):.0f})\n"
                    )
                elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                    self.results_text.append(
                        f"{i}. {class_name} - 置信度: {conf:.2%}\n"
                        f"   位置: ({bbox[0]:.0f}, {bbox[1]:.0f}) - ({bbox[2]:.0f}, {bbox[3]:.0f})\n"
                    )

    # Batch inference methods
    def browse_batch_folder(self):
        path = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if path:
            self.batch_folder_edit.setText(path)

    def browse_batch_output(self):
        path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if path:
            self.batch_output_edit.setText(path)

    def start_batch_inference(self):
        if not self.inferencer:
            QMessageBox.warning(self, "错误", "请先加载模型")
            return

        folder = self.batch_folder_edit.text()
        if not folder or not os.path.exists(folder):
            QMessageBox.warning(self, "错误", "请选择有效的图片文件夹")
            return

        # Collect images
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        image_paths = sorted([
            os.path.join(folder, f) for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in extensions
        ])

        if not image_paths:
            QMessageBox.warning(self, "错误", "文件夹中没有图片")
            return

        output_dir = self.batch_output_edit.text()
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        self.batch_results = []
        self.batch_progress.setRange(0, len(image_paths))
        self.batch_progress.setValue(0)
        self.batch_results_text.clear()
        self.batch_results_text.append(f"开始批量推理 {len(image_paths)} 张图片...\n")

        self.batch_start_btn.setEnabled(False)
        self.batch_stop_btn.setEnabled(True)
        self.batch_export_btn.setEnabled(False)

        self.stop_batch_inference()
        self._cleanup_batch_worker()
        self.batch_worker = BatchInferenceWorker(
            self.inferencer, image_paths, output_dir, **self.get_predict_args()
        )
        self.batch_worker.progress.connect(self.on_batch_progress)
        self.batch_worker.result_ready.connect(self.on_batch_result)
        self.batch_worker.finished.connect(self.on_batch_finished)
        self.batch_worker.start()

    def stop_batch_inference(self):
        if self.batch_worker:
            self.batch_worker.stop()

    def on_batch_progress(self, current: int, total: int, filename: str):
        self.batch_progress.setValue(current)
        self.batch_status_label.setText(f"处理中: {current}/{total} - {filename}")

    def on_batch_result(self, image_path: str, detections: list, elapsed: float):
        self.batch_results.append({
            "path": image_path,
            "detections": detections,
            "elapsed": elapsed,
        })
        count = len(detections)
        self.batch_results_text.append(
            f"{os.path.basename(image_path)}: {count} 个目标, {elapsed*1000:.1f}ms"
        )

    def on_batch_finished(self, total: int):
        self.batch_start_btn.setEnabled(True)
        self.batch_stop_btn.setEnabled(False)
        self.batch_export_btn.setEnabled(True)
        self.batch_status_label.setText(f"完成! 共处理 {total} 张图片")
        self.batch_results_text.append(f"\n批量推理完成! 共处理 {total} 张图片")
        self._cleanup_batch_worker()

    def export_batch_results(self):
        if not self.batch_results:
            QMessageBox.information(self, "提示", "没有可导出的结果")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "导出结果", "检测结果.csv", "CSV 文件 (*.csv);;所有文件 (*)"
        )
        if not path:
            return

        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["图片", "类别", "类型", "置信度", "x1", "y1", "x2", "y2", "额外信息", "耗时(ms)"])

                for result in self.batch_results:
                    img_name = os.path.basename(result["path"])
                    elapsed_ms = result["elapsed"] * 1000
                    for det in result["detections"]:
                        det_type = det.get("type", "bbox")
                        bbox = det.get("bbox", {})
                        if isinstance(bbox, dict):
                            x1, y1 = f"{bbox.get('x1', 0):.0f}", f"{bbox.get('y1', 0):.0f}"
                            x2, y2 = f"{bbox.get('x2', 0):.0f}", f"{bbox.get('y2', 0):.0f}"
                        elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                            x1, y1 = f"{bbox[0]:.0f}", f"{bbox[1]:.0f}"
                            x2, y2 = f"{bbox[2]:.0f}", f"{bbox[3]:.0f}"
                        else:
                            x1, y1, x2, y2 = "", "", "", ""
                        # Extra info for OBB/Keypoint
                        extra = ""
                        if det_type == "obb":
                            corners = det.get("corners", [])
                            extra = "; ".join(f"({c[0]:.0f},{c[1]:.0f})" for c in corners)
                        elif det_type == "keypoint":
                            kps = det.get("keypoints", [])
                            extra = f"{len(kps)} keypoints"
                        writer.writerow([
                            img_name,
                            det.get("class_name", ""),
                            det_type,
                            f"{det.get('confidence', 0):.4f}",
                            x1, y1, x2, y2,
                            extra,
                            f"{elapsed_ms:.1f}",
                        ])

            QMessageBox.information(self, "导出成功", f"结果已导出到:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))
