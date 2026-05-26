"""Training configuration and monitoring panel."""

import os
import sys
import io
import threading
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QComboBox, QSpinBox, QDoubleSpinBox, QPushButton, QLineEdit,
    QTextEdit, QProgressBar, QFileDialog, QCheckBox, QScrollArea,
    QFormLayout, QTabWidget, QMessageBox, QSplitter, QDialog,
    QDialogButtonBox, QInputDialog
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt6.QtGui import QFont
from loguru import logger

from core.config import ConfigManager
from core.trainer import YOLOTrainer
from core.gpu import detect_cuda, format_gpu_summary
from gui.theme import Theme
from gui.ui_components import StatusPill

# Matplotlib integration
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class TrainingChart(FigureCanvas):
    """Matplotlib canvas for training metrics visualization."""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(8, 4), dpi=100, facecolor=Theme.SURFACE_2)
        super().__init__(self.fig)
        self.setParent(parent)
        self.axes_loss = self.fig.add_subplot(121)
        self.axes_metrics = self.fig.add_subplot(122)
        self._style_axes(self.axes_loss, "Loss")
        self._style_axes(self.axes_metrics, "Metrics")
        self.fig.tight_layout(pad=2.0)

    def _style_axes(self, ax, title):
        ax.set_facecolor(Theme.SURFACE)
        ax.set_title(title, color=Theme.TEXT, fontsize=10)
        ax.tick_params(colors=Theme.TEXT_MUTED, labelsize=8)
        ax.spines['bottom'].set_color(Theme.BORDER_STRONG)
        ax.spines['top'].set_color(Theme.BORDER_STRONG)
        ax.spines['left'].set_color(Theme.BORDER_STRONG)
        ax.spines['right'].set_color(Theme.BORDER_STRONG)
        ax.xaxis.label.set_color(Theme.TEXT_MUTED)
        ax.yaxis.label.set_color(Theme.TEXT_MUTED)

    def update_chart(self, logs: dict):
        """Update chart with training logs."""
        self.axes_loss.clear()
        self.axes_metrics.clear()
        self._style_axes(self.axes_loss, "Loss")
        self._style_axes(self.axes_metrics, "Metrics")

        epochs = logs.get("epochs", [])
        if not epochs:
            self.draw()
            return

        # Plot losses
        for key, color in [
            ("train/box_loss", "#ff6b6b"),
            ("train/cls_loss", "#ffd93d"),
            ("train/dfl_loss", "#6bcb77"),
            ("val/box_loss", "#ff6b6b"),
            ("val/cls_loss", "#ffd93d"),
            ("val/dfl_loss", "#6bcb77"),
        ]:
            values = logs.get(key, [])
            if values:
                style = "--" if key.startswith("val/") else "-"
                alpha = 0.6 if key.startswith("val/") else 1.0
                label = key.replace("train/", "tr/").replace("val/", "v/")
                self.axes_loss.plot(epochs[:len(values)], values, style, color=color, alpha=alpha, label=label, linewidth=1.5)

        self.axes_loss.set_xlabel("Epoch")
        self.axes_loss.set_ylabel("Loss")
        self.axes_loss.legend(fontsize=7, facecolor=Theme.SURFACE, edgecolor=Theme.BORDER_STRONG, labelcolor=Theme.TEXT)

        # Plot metrics
        for key, color in [
            ("metrics/precision(B)", "#4ecdc4"),
            ("metrics/recall(B)", "#45b7d1"),
            ("metrics/mAP50(B)", "#96ceb4"),
            ("metrics/mAP50-95(B)", "#ffeaa7"),
        ]:
            values = logs.get(key, [])
            if values:
                self.axes_metrics.plot(epochs[:len(values)], values, color=color, label=key.split("(")[0].split("/")[1], linewidth=1.5)

        self.axes_metrics.set_xlabel("Epoch")
        self.axes_metrics.set_ylabel("Score")
        self.axes_metrics.legend(fontsize=7, facecolor=Theme.SURFACE, edgecolor=Theme.BORDER_STRONG, labelcolor=Theme.TEXT)
        self.axes_metrics.set_ylim(0, 1)

        self.fig.tight_layout(pad=2.0)
        self.draw()

    def clear_chart(self):
        """Clear all charts."""
        self.axes_loss.clear()
        self.axes_metrics.clear()
        self._style_axes(self.axes_loss, "Loss")
        self._style_axes(self.axes_metrics, "Metrics")
        self.draw()


class _StreamTee:
    """Tee stdout/stderr: writes go to both the original stream and a thread-safe buffer.

    Fully compatible with tqdm / Ultralytics output — implements all attributes
    that tqdm checks (encoding, writable, isatty, fileno, etc.).
    """

    def __init__(self, original):
        self._original = original
        self._buffer = io.StringIO()
        self._lock = threading.Lock()
        # Mirror attributes that tqdm / Ultralytics may inspect
        for attr in ("encoding", "errors", "newlines", "line_buffering"):
            try:
                setattr(self, attr, getattr(original, attr, None))
            except Exception:
                pass

    # ---- Stream interface ----
    def write(self, s):
        if not isinstance(s, str):
            try:
                s = str(s)
            except Exception:
                return 0
        with self._lock:
            self._original.write(s)
            self._buffer.write(s)
        return len(s)

    def flush(self):
        self._original.flush()

    def writable(self):
        return True

    def readable(self):
        return False

    def seekable(self):
        return False

    def isatty(self):
        return False

    def fileno(self):
        return self._original.fileno()

    @property
    def closed(self):
        return False

    # ---- Buffer access ----
    def read_and_clear(self) -> str:
        """Atomically read buffered text and clear the buffer."""
        with self._lock:
            text = self._buffer.getvalue()
            self._buffer.seek(0)
            self._buffer.truncate(0)
        return text


class TrainingWorker(QThread):
    """Worker thread for training with real-time log & metric capture."""
    progress = pyqtSignal(str)            # captured stdout/stderr text
    finished = pyqtSignal(dict)           # final result dict
    epoch_update = pyqtSignal(int, dict)  # epoch number + metrics dict

    def __init__(self, trainer, data_yaml, **kwargs):
        super().__init__()
        self.trainer = trainer
        self.data_yaml = data_yaml
        self.kwargs = kwargs
        # Thread-safe queue for epoch metrics (avoid unsafe cross-thread emit)
        self._epoch_queue: list[tuple[int, dict]] = []
        self._epoch_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Ultralytics callbacks  (called from TRAINING thread)
    # ------------------------------------------------------------------
    def _on_fit_epoch_end(self, trainer_obj):
        """Ultralytics callback: fired after each fit epoch (includes validation).

        In Ultralytics ≥ 8.1, after validation trainer_obj.metrics is a dict
        with keys like 'metrics/precision(B)', 'val/box_loss', etc.
        We collect into a thread-safe queue; the Qt timer in TrainingPanel
        polls this queue and updates the chart on the GUI thread.
        """
        try:
            epoch = trainer_obj.epoch + 1  # 0-based to 1-based
            metrics = {}

            # --- Training losses (from tloss = smoothed loss_items) ---
            tloss = getattr(trainer_obj, "tloss", None)
            if tloss is not None:
                try:
                    if hasattr(tloss, "__len__") and len(tloss) >= 3:
                        metrics["train/box_loss"] = float(tloss[0])
                        metrics["train/cls_loss"] = float(tloss[1])
                        metrics["train/dfl_loss"] = float(tloss[2])
                except Exception:
                    pass

            # --- Validation metrics & losses ---
            # trainer_obj.metrics is a dict set by trainer.validate(), e.g.:
            #   {'metrics/precision(B)': 0.8, 'val/box_loss': 1.2, ...}
            m = getattr(trainer_obj, "metrics", None)
            if m and isinstance(m, dict):
                for key in (
                    "val/box_loss", "val/cls_loss", "val/dfl_loss",
                    "metrics/precision(B)", "metrics/recall(B)",
                    "metrics/mAP50(B)", "metrics/mAP50-95(B)",
                ):
                    if key in m and key not in metrics:
                        try:
                            metrics[key] = float(m[key])
                        except (TypeError, ValueError):
                            pass

            with self._epoch_lock:
                self._epoch_queue.append((epoch, metrics))

            logger.debug(f"Epoch {epoch} metrics collected: {list(metrics.keys())}")
        except Exception as exc:
            logger.warning(f"fit_epoch callback error: {exc}")

    # ------------------------------------------------------------------
    # Public: poll epoch data from GUI thread
    # ------------------------------------------------------------------
    def drain_epoch_queue(self) -> list[tuple[int, dict]]:
        """Drain all pending epoch updates (call from GUI thread)."""
        with self._epoch_lock:
            items = list(self._epoch_queue)
            self._epoch_queue.clear()
        return items

    def _register_callback(self):
        """Register our epoch callback on the YOLO model object.

        Ultralytics stores callbacks on the model via add_callback().
        These callbacks are carried over to the internal trainer when
        model.train() is called, so registering here is safe.
        """
        model = self.trainer.model
        if model is not None:
            try:
                # Remove any previous registration to avoid duplicates
                if hasattr(model, "callbacks"):
                    cb_list = model.callbacks.get("on_fit_epoch_end", [])
                    # Remove old references to our method
                    model.callbacks["on_fit_epoch_end"] = [
                        cb for cb in cb_list if cb is not self._on_fit_epoch_end
                    ]
                model.add_callback("on_fit_epoch_end", self._on_fit_epoch_end)
                logger.info("Registered on_fit_epoch_end callback on YOLO model")
            except Exception as e:
                logger.warning(f"Failed to register callback: {e}")

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------
    def run(self):
        # ---- Install stdout/stderr tee BEFORE anything else ----
        tee_out = _StreamTee(sys.stdout)
        tee_err = _StreamTee(sys.stderr)
        sys.stdout = tee_out
        sys.stderr = tee_err

        # ---- Background poll: flush captured text to UI every 150 ms ----
        _poll_interval = 0.15
        _stop_poll = threading.Event()

        def _poll_loop():
            while not _stop_poll.is_set():
                text_out = tee_out.read_and_clear()
                text_err = tee_err.read_and_clear()
                text = text_out + text_err
                if text.strip():
                    self.progress.emit(text)
                _stop_poll.wait(_poll_interval)

        poll_thread = threading.Thread(target=_poll_loop, daemon=True)
        poll_thread.start()

        # ---- Register Ultralytics callbacks on the YOLO model ----
        # Must register AFTER load_model() is called (which happens inside
        # trainer.train if model was None). We patch train() to register
        # callbacks right before model.train() runs.
        self._register_callback()

        # ---- Run training ----
        try:
            result = self.trainer.train(self.data_yaml, **self.kwargs)
            # Also try to get save_dir from the ultralytics results object
            if result.get("success") and result.get("save_dir") is None:
                model = self.trainer.model
                if model and hasattr(model, "trainer") and model.trainer:
                    result["save_dir"] = str(model.trainer.save_dir)
            self.finished.emit(result)
        except Exception as e:
            self.finished.emit({"success": False, "error": str(e)})
        finally:
            # Stop poll thread
            _stop_poll.set()
            poll_thread.join(timeout=2)
            # Flush remaining text
            text = tee_out.read_and_clear() + tee_err.read_and_clear()
            if text.strip():
                self.progress.emit(text)
            # Restore original streams
            sys.stdout = tee_out._original
            sys.stderr = tee_err._original


class TrainingCompleteDialog(QDialog):
    """Dialog shown after training completes, with quick actions."""

    def __init__(self, save_dir: str, best_pt: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("训练完成")
        self.setMinimumWidth(480)
        self.best_pt = best_pt
        self.action = None  # 'infer', 'export', 'annotate', or None (close)

        layout = QVBoxLayout(self)

        # Summary
        info = QLabel(f"训练结果已保存到:\n{save_dir}")
        info.setWordWrap(True)
        layout.addWidget(info)

        if best_pt and os.path.isfile(best_pt):
            model_info = QLabel(f"最佳模型: {os.path.basename(best_pt)}")
            model_info.setObjectName("MutedText")
            layout.addWidget(model_info)

            # Quick action buttons
            btn_layout = QHBoxLayout()

            self.infer_btn = QPushButton("立即推理")
            self.infer_btn.setObjectName("PrimaryButton")
            self.infer_btn.clicked.connect(self._on_infer)

            self.export_btn = QPushButton("导出模型")
            self.export_btn.clicked.connect(self._on_export)

            self.annotate_btn = QPushButton("自动标注")
            self.annotate_btn.clicked.connect(self._on_annotate)

            btn_layout.addWidget(self.infer_btn)
            btn_layout.addWidget(self.export_btn)
            btn_layout.addWidget(self.annotate_btn)
            layout.addLayout(btn_layout)
        else:
            no_model = QLabel("未找到 best.pt 模型文件")
            no_model.setObjectName("MutedText")
            layout.addWidget(no_model)

        # Close button
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.reject)
        layout.addWidget(close_btn)

    def _on_infer(self):
        self.action = "infer"
        self.accept()

    def _on_export(self):
        self.action = "export"
        self.accept()

    def _on_annotate(self):
        self.action = "annotate"
        self.accept()


class TrainingPanel(QWidget):
    """Panel for configuring and monitoring YOLO training."""

    training_started = pyqtSignal()
    training_finished = pyqtSignal(dict)
    model_ready = pyqtSignal(str, str)  # Emitted with (best.pt path, action) when training completes

    def __init__(self, config_manager: ConfigManager, parent=None):
        super().__init__(parent)
        self.config = config_manager
        self.trainer = YOLOTrainer(config_manager.config)
        self.worker = None
        self.log_file = None
        self.gpu_detection = detect_cuda()
        # Live training metrics accumulator
        self._live_logs: dict[str, list] = {"epochs": []}
        # Timer to poll epoch metrics from worker (GUI-thread safe)
        self._epoch_timer = QTimer(self)
        self._epoch_timer.setInterval(500)  # poll every 500 ms
        self._epoch_timer.timeout.connect(self._poll_epoch_updates)
        # Early stopping tracking
        self._best_map = 0.0
        self._best_map_epoch = 0
        self._no_improve_count = 0
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        def create_tab():
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            content = QWidget()
            tab_layout = QVBoxLayout(content)
            tab_layout.setContentsMargins(10, 10, 10, 10)
            tab_layout.setSpacing(10)
            scroll.setWidget(content)
            return scroll, tab_layout

        settings_tabs = QTabWidget()
        settings_tabs.setMinimumWidth(390)

        # Model selection -- two-level: series to model
        model_group = QGroupBox("模型配置")
        model_layout = QFormLayout()
        model_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.model_series_combo = QComboBox()
        self._model_series = [
            # YOLO26 (2025.9, newest)
            ("YOLO26 检测",   ["yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt"]),
            ("YOLO26 分割",   ["yolo26n-seg.pt", "yolo26s-seg.pt", "yolo26m-seg.pt", "yolo26l-seg.pt", "yolo26x-seg.pt"]),
            ("YOLO26 分类",   ["yolo26n-cls.pt", "yolo26s-cls.pt", "yolo26m-cls.pt", "yolo26l-cls.pt", "yolo26x-cls.pt"]),
            ("YOLO26 姿态",   ["yolo26n-pose.pt", "yolo26s-pose.pt", "yolo26m-pose.pt", "yolo26l-pose.pt", "yolo26x-pose.pt"]),
            ("YOLO26 旋转框", ["yolo26n-obb.pt", "yolo26s-obb.pt", "yolo26m-obb.pt", "yolo26l-obb.pt", "yolo26x-obb.pt"]),
            # YOLO12 (2025.2)
            ("YOLO12 检测",   ["yolo12n.pt", "yolo12s.pt", "yolo12m.pt", "yolo12l.pt", "yolo12x.pt"]),
            ("YOLO12 分割",   ["yolo12n-seg.pt", "yolo12s-seg.pt", "yolo12m-seg.pt", "yolo12l-seg.pt", "yolo12x-seg.pt"]),
            ("YOLO12 分类",   ["yolo12n-cls.pt", "yolo12s-cls.pt", "yolo12m-cls.pt", "yolo12l-cls.pt", "yolo12x-cls.pt"]),
            ("YOLO12 姿态",   ["yolo12n-pose.pt", "yolo12s-pose.pt", "yolo12m-pose.pt", "yolo12l-pose.pt", "yolo12x-pose.pt"]),
            ("YOLO12 旋转框", ["yolo12n-obb.pt", "yolo12s-obb.pt", "yolo12m-obb.pt", "yolo12l-obb.pt", "yolo12x-obb.pt"]),
            # YOLO11
            ("YOLO11 检测",   ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"]),
            ("YOLO11 分割",   ["yolo11n-seg.pt", "yolo11s-seg.pt", "yolo11m-seg.pt", "yolo11l-seg.pt", "yolo11x-seg.pt"]),
            ("YOLO11 分类",   ["yolo11n-cls.pt", "yolo11s-cls.pt", "yolo11m-cls.pt", "yolo11l-cls.pt", "yolo11x-cls.pt"]),
            ("YOLO11 姿态",   ["yolo11n-pose.pt", "yolo11s-pose.pt", "yolo11m-pose.pt", "yolo11l-pose.pt", "yolo11x-pose.pt"]),
            ("YOLO11 旋转框", ["yolo11n-obb.pt", "yolo11s-obb.pt", "yolo11m-obb.pt", "yolo11l-obb.pt", "yolo11x-obb.pt"]),
            # YOLOv10
            ("YOLOv10 检测",  ["yolov10n.pt", "yolov10s.pt", "yolov10m.pt", "yolov10l.pt", "yolov10x.pt"]),
            # YOLOv8
            ("YOLOv8 检测",   ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]),
            ("YOLOv8 分割",   ["yolov8n-seg.pt", "yolov8s-seg.pt", "yolov8m-seg.pt", "yolov8l-seg.pt", "yolov8x-seg.pt"]),
            ("YOLOv8 分类",   ["yolov8n-cls.pt", "yolov8s-cls.pt", "yolov8m-cls.pt", "yolov8l-cls.pt", "yolov8x-cls.pt"]),
            ("YOLOv8 姿态",   ["yolov8n-pose.pt", "yolov8s-pose.pt", "yolov8m-pose.pt", "yolov8l-pose.pt", "yolov8x-pose.pt"]),
            ("YOLOv8 旋转框", ["yolov8n-obb.pt", "yolov8s-obb.pt", "yolov8m-obb.pt", "yolov8l-obb.pt", "yolov8x-obb.pt"]),
            # YOLOv9
            ("YOLOv9 检测",   ["yolov9c.pt", "yolov9e.pt"]),
            # YOLOv5
            ("YOLOv5 检测",   ["yolov5nu.pt", "yolov5su.pt", "yolov5mu.pt", "yolov5lu.pt", "yolov5xu.pt"]),
            # RT-DETR
            ("RT-DETR 检测",  ["rtdetr-l.pt", "rtdetr-x.pt"]),
        ]
        for name, _ in self._model_series:
            self.model_series_combo.addItem(name)
        self.model_series_combo.currentIndexChanged.connect(self._on_series_changed)
        model_layout.addRow("模型系列:", self.model_series_combo)

        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self._on_series_changed(0)  # populate initial models
        model_layout.addRow("预训练模型:", self.model_combo)

        self.custom_model_btn = QPushButton("浏览...")
        self.custom_model_btn.clicked.connect(self.browse_model)
        model_layout.addRow("自定义模型:", self.custom_model_btn)

        model_group.setLayout(model_layout)

        # Dataset
        data_group = QGroupBox("数据集")
        data_layout = QFormLayout()
        data_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.data_yaml_edit = QLineEdit()
        self.data_yaml_edit.setPlaceholderText("选择 data.yaml 文件...")
        data_browse = QPushButton("浏览...")
        data_browse.clicked.connect(self.browse_data_yaml)
        data_row = QHBoxLayout()
        data_row.addWidget(self.data_yaml_edit)
        data_row.addWidget(data_browse)
        data_layout.addRow("data.yaml:", data_row)

        data_group.setLayout(data_layout)

        # Device info (GPU + CPU combined)
        device_group = QGroupBox("计算设备")
        device_layout = QFormLayout()
        device_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.device_combo = QComboBox()
        self._populate_device_combo()
        self.device_combo.setEditable(True)
        device_layout.addRow("训练设备:", self.device_combo)

        self.device_status_label = StatusPill()
        self.device_combo.currentTextChanged.connect(self._update_device_status)
        self._update_device_status(self.device_combo.currentText())
        device_layout.addRow("设备状态:", self.device_status_label)

        self.gpu_info_label = QLabel()
        self.gpu_info_label.setWordWrap(True)
        self.gpu_info_label.setFont(QFont("monospace", 9))
        self.gpu_info_label.setText(self._format_device_summary())
        device_layout.addRow("硬件摘要:", self.gpu_info_label)

        self.refresh_gpu_btn = QPushButton("刷新设备信息")
        self.refresh_gpu_btn.clicked.connect(self.refresh_gpu_info)
        device_layout.addRow("", self.refresh_gpu_btn)

        device_group.setLayout(device_layout)

        # Save directory
        save_group = QGroupBox("保存")
        save_layout = QFormLayout()
        save_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.project_edit = QLineEdit("runs/train")
        save_browse = QPushButton("浏览...")
        save_browse.clicked.connect(self.browse_project)
        save_row = QHBoxLayout()
        save_row.addWidget(self.project_edit)
        save_row.addWidget(save_browse)
        save_layout.addRow("项目目录:", save_row)

        self.name_edit = QLineEdit("exp")
        save_layout.addRow("实验名称:", self.name_edit)

        self.exist_ok_check = QCheckBox("覆盖已有实验")
        save_layout.addRow(self.exist_ok_check)

        save_group.setLayout(save_layout)

        # Font (avoids ultralytics downloading Arial.Unicode.ttf from internet)
        font_group = QGroupBox("字体 (避免训练卡死)")
        font_layout = QFormLayout()
        font_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.font_path_edit = QLineEdit()
        self.font_path_edit.setPlaceholderText("选择中文字体文件 (.ttf)，留空则跳过...")
        self.font_path_edit.textChanged.connect(self._update_font_status)
        font_browse = QPushButton("浏览...")
        font_browse.clicked.connect(self.browse_font)
        font_row = QHBoxLayout()
        font_row.addWidget(self.font_path_edit)
        font_row.addWidget(font_browse)
        font_layout.addRow("字体文件:", font_row)

        self.font_status_label = QLabel("")
        self.font_status_label.setStyleSheet(f"color: {Theme.TEXT_MUTED}; font-size: 11px;")
        font_layout.addRow(self.font_status_label)

        font_group.setLayout(font_layout)

        # Auto-detect font in project root
        detected = self._auto_detect_font()
        if detected:
            self.font_path_edit.setText(detected)
            self._update_font_status()

        # Smart recommendation
        smart_group = QGroupBox("智能推荐")
        smart_layout = QVBoxLayout()
        self.recommend_btn = QPushButton("分析数据集并推荐参数")
        self.recommend_btn.setObjectName("PrimaryButton")
        self.recommend_btn.clicked.connect(self._auto_recommend_params)
        smart_layout.addWidget(self.recommend_btn)
        smart_group.setLayout(smart_layout)

        # Training templates
        template_group = QGroupBox("训练模板")
        template_layout = QVBoxLayout()

        template_btn_row = QHBoxLayout()
        self.template_combo = QComboBox()
        self.template_combo.setMinimumWidth(160)
        self._load_template_list()
        template_btn_row.addWidget(self.template_combo, 1)

        self.save_template_btn = QPushButton("保存")
        self.save_template_btn.setFixedWidth(52)
        self.save_template_btn.clicked.connect(self._save_template)
        template_btn_row.addWidget(self.save_template_btn)

        self.load_template_btn = QPushButton("加载")
        self.load_template_btn.setFixedWidth(52)
        self.load_template_btn.clicked.connect(self._load_template)
        template_btn_row.addWidget(self.load_template_btn)

        self.delete_template_btn = QPushButton("删除")
        self.delete_template_btn.setFixedWidth(52)
        self.delete_template_btn.clicked.connect(self._delete_template)
        template_btn_row.addWidget(self.delete_template_btn)

        template_layout.addLayout(template_btn_row)
        template_group.setLayout(template_layout)

        # Training parameters
        params_group = QGroupBox("训练参数")
        params_layout = QFormLayout()
        params_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 10000)
        self.epochs_spin.setValue(self.config.get("training", "epochs", 100))
        params_layout.addRow("训练轮数 (epochs):", self.epochs_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 256)
        self.batch_spin.setValue(self.config.get("training", "batch", 16))
        params_layout.addRow("批次大小 (batch):", self.batch_spin)

        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(32, 4096)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(self.config.get("training", "imgsz", 640))
        params_layout.addRow("图像尺寸 (imgsz):", self.imgsz_spin)

        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(0, 32)
        self.workers_spin.setValue(self.config.get("training", "workers", 8))
        self.workers_spin.setToolTip(
            "DataLoader 工作线程数。\n"
        )
        params_layout.addRow("工作线程:", self.workers_spin)

        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(0, 1000)
        self.patience_spin.setValue(self.config.get("training", "patience", 100))
        params_layout.addRow("早停轮数 (patience):", self.patience_spin)

        self.amp_check = QCheckBox("启用混合精度训练 (推荐开启)")
        self.amp_check.setChecked(self.config.get("training", "amp", True))
        self.amp_check.setToolTip("AMP 可大幅降低显存占用，低显存显卡建议开启")
        params_layout.addRow(self.amp_check)

        self.cache_check = QCheckBox("缓存图片到内存 (加速训练)")
        self.cache_check.setChecked(self.config.get("training", "cache", False))
        self.cache_check.setToolTip("第一轮后图片直接从内存读取，加速明显但会占用更多内存")
        params_layout.addRow(self.cache_check)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 99999)
        self.seed_spin.setValue(self.config.get("training", "seed", 0))
        self.seed_spin.setToolTip("随机种子，0=随机")
        params_layout.addRow("随机种子 (seed):", self.seed_spin)

        params_group.setLayout(params_layout)

        # Optimizer
        optim_group = QGroupBox("优化器")
        optim_layout = QFormLayout()
        optim_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["auto", "SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp"])
        optim_layout.addRow("优化器:", self.optimizer_combo)

        self.lr0_spin = QDoubleSpinBox()
        self.lr0_spin.setRange(0.0001, 1.0)
        self.lr0_spin.setDecimals(5)
        self.lr0_spin.setValue(self.config.get("training", "lr0", 0.01))
        optim_layout.addRow("初始学习率:", self.lr0_spin)

        self.momentum_spin = QDoubleSpinBox()
        self.momentum_spin.setRange(0.0, 1.0)
        self.momentum_spin.setDecimals(3)
        self.momentum_spin.setValue(self.config.get("training", "momentum", 0.937))
        optim_layout.addRow("动量:", self.momentum_spin)

        self.weight_decay_spin = QDoubleSpinBox()
        self.weight_decay_spin.setRange(0.0, 0.1)
        self.weight_decay_spin.setDecimals(5)
        self.weight_decay_spin.setValue(self.config.get("training", "weight_decay", 0.0005))
        optim_layout.addRow("权重衰减:", self.weight_decay_spin)

        self.lrf_spin = QDoubleSpinBox()
        self.lrf_spin.setRange(0.001, 1.0)
        self.lrf_spin.setDecimals(3)
        self.lrf_spin.setValue(self.config.get("training", "lrf", 0.01))
        self.lrf_spin.setToolTip("最终学习率 = lr0 × lrf")
        optim_layout.addRow("最终学习率因子 (lrf):", self.lrf_spin)

        self.warmup_epochs_spin = QDoubleSpinBox()
        self.warmup_epochs_spin.setRange(0.0, 50.0)
        self.warmup_epochs_spin.setDecimals(1)
        self.warmup_epochs_spin.setValue(self.config.get("training", "warmup_epochs", 3.0))
        optim_layout.addRow("预热轮数:", self.warmup_epochs_spin)

        self.warmup_momentum_spin = QDoubleSpinBox()
        self.warmup_momentum_spin.setRange(0.0, 1.0)
        self.warmup_momentum_spin.setDecimals(3)
        self.warmup_momentum_spin.setValue(self.config.get("training", "warmup_momentum", 0.8))
        optim_layout.addRow("预热动量:", self.warmup_momentum_spin)

        self.cos_lr_check = QCheckBox("余弦学习率调度 (cos_lr)")
        self.cos_lr_check.setChecked(self.config.get("training", "cos_lr", False))
        optim_layout.addRow(self.cos_lr_check)

        self.close_mosaic_spin = QSpinBox()
        self.close_mosaic_spin.setRange(0, 100)
        self.close_mosaic_spin.setValue(self.config.get("training", "close_mosaic", 10))
        self.close_mosaic_spin.setToolTip("最后 N 轮关闭马赛克增强，有助于最终收敛")
        optim_layout.addRow("关闭马赛克轮数:", self.close_mosaic_spin)

        optim_group.setLayout(optim_layout)

        # Data augmentation
        aug_group = QGroupBox("数据增强")
        aug_layout = QFormLayout()
        aug_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.fliplr_spin = QDoubleSpinBox()
        self.fliplr_spin.setRange(0.0, 1.0)
        self.fliplr_spin.setDecimals(2)
        self.fliplr_spin.setValue(0.5)
        aug_layout.addRow("水平翻转:", self.fliplr_spin)

        self.mosaic_spin = QDoubleSpinBox()
        self.mosaic_spin.setRange(0.0, 1.0)
        self.mosaic_spin.setDecimals(2)
        self.mosaic_spin.setValue(1.0)
        aug_layout.addRow("马赛克:", self.mosaic_spin)

        self.hsv_h_spin = QDoubleSpinBox()
        self.hsv_h_spin.setRange(0.0, 1.0)
        self.hsv_h_spin.setDecimals(3)
        self.hsv_h_spin.setValue(0.015)
        aug_layout.addRow("HSV-色调:", self.hsv_h_spin)

        self.hsv_s_spin = QDoubleSpinBox()
        self.hsv_s_spin.setRange(0.0, 1.0)
        self.hsv_s_spin.setDecimals(2)
        self.hsv_s_spin.setValue(0.7)
        aug_layout.addRow("HSV-饱和度:", self.hsv_s_spin)

        self.degrees_spin = QDoubleSpinBox()
        self.degrees_spin.setRange(0.0, 180.0)
        self.degrees_spin.setDecimals(1)
        self.degrees_spin.setValue(0.0)
        aug_layout.addRow("旋转角度:", self.degrees_spin)

        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(0.0, 2.0)
        self.scale_spin.setDecimals(2)
        self.scale_spin.setValue(0.5)
        aug_layout.addRow("缩放:", self.scale_spin)

        self.hsv_v_spin = QDoubleSpinBox()
        self.hsv_v_spin.setRange(0.0, 1.0)
        self.hsv_v_spin.setDecimals(2)
        self.hsv_v_spin.setValue(0.4)
        aug_layout.addRow("HSV-明度:", self.hsv_v_spin)

        self.translate_spin = QDoubleSpinBox()
        self.translate_spin.setRange(0.0, 1.0)
        self.translate_spin.setDecimals(2)
        self.translate_spin.setValue(0.1)
        aug_layout.addRow("平移:", self.translate_spin)

        self.shear_spin = QDoubleSpinBox()
        self.shear_spin.setRange(0.0, 30.0)
        self.shear_spin.setDecimals(1)
        self.shear_spin.setValue(0.0)
        aug_layout.addRow("剪切:", self.shear_spin)

        self.flipud_spin = QDoubleSpinBox()
        self.flipud_spin.setRange(0.0, 1.0)
        self.flipud_spin.setDecimals(2)
        self.flipud_spin.setValue(0.0)
        aug_layout.addRow("垂直翻转:", self.flipud_spin)

        self.mixup_spin = QDoubleSpinBox()
        self.mixup_spin.setRange(0.0, 1.0)
        self.mixup_spin.setDecimals(2)
        self.mixup_spin.setValue(0.0)
        self.mixup_spin.setToolTip("MixUp 数据增强概率")
        aug_layout.addRow("MixUp:", self.mixup_spin)

        self.erasing_spin = QDoubleSpinBox()
        self.erasing_spin.setRange(0.0, 1.0)
        self.erasing_spin.setDecimals(2)
        self.erasing_spin.setValue(0.4)
        self.erasing_spin.setToolTip("随机擦除增强概率")
        aug_layout.addRow("随机擦除:", self.erasing_spin)

        aug_group.setLayout(aug_layout)

        setup_tab, setup_layout = create_tab()
        setup_layout.addWidget(model_group)
        setup_layout.addWidget(data_group)
        setup_layout.addWidget(device_group)
        setup_layout.addWidget(save_group)
        setup_layout.addWidget(font_group)
        setup_layout.addWidget(smart_group)
        setup_layout.addWidget(template_group)
        setup_layout.addStretch()
        settings_tabs.addTab(setup_tab, "运行设置")

        params_tab, params_tab_layout = create_tab()
        params_tab_layout.addWidget(params_group)
        params_tab_layout.addStretch()
        settings_tabs.addTab(params_tab, "基础参数")

        optim_tab, optim_tab_layout = create_tab()
        optim_tab_layout.addWidget(optim_group)
        optim_tab_layout.addStretch()
        settings_tabs.addTab(optim_tab, "优化策略")

        aug_tab, aug_tab_layout = create_tab()
        aug_tab_layout.addWidget(aug_group)
        aug_tab_layout.addStretch()
        settings_tabs.addTab(aug_tab, "数据增强")

        main_splitter.addWidget(settings_tabs)

        # Right: chart + log
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(10, 0, 0, 0)
        right_layout.setSpacing(10)

        monitor_group = QGroupBox("训练监控")
        monitor_layout = QVBoxLayout()

        self.chart_tabs = QTabWidget()

        self.training_chart = TrainingChart()
        self.chart_tabs.addTab(self.training_chart, "训练曲线")

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("monospace", 9))
        self.chart_tabs.addTab(self.log_text, "训练日志")

        monitor_layout.addWidget(self.chart_tabs)
        monitor_group.setLayout(monitor_layout)
        right_layout.addWidget(monitor_group, 1)

        control_group = QGroupBox("运行控制")
        control_layout = QVBoxLayout()
        btn_layout = QHBoxLayout()

        self.start_btn = QPushButton("开始训练")
        self.start_btn.setObjectName("PrimaryButton")
        self.start_btn.clicked.connect(self.start_training)

        self.stop_btn = QPushButton("停止训练")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setObjectName("DangerButton")
        self.stop_btn.clicked.connect(self.stop_training)

        self.refresh_chart_btn = QPushButton("刷新曲线")
        self.refresh_chart_btn.clicked.connect(self.refresh_chart)

        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        btn_layout.addWidget(self.refresh_chart_btn)
        control_layout.addLayout(btn_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        control_layout.addWidget(self.progress_bar)

        self.early_stop_label = QLabel("")
        self.early_stop_label.setStyleSheet(f"color: {Theme.TEXT_MUTED}; font-size: 12px; padding: 4px 0;")
        self.early_stop_label.setVisible(False)
        control_layout.addWidget(self.early_stop_label)

        control_group.setLayout(control_layout)
        right_layout.addWidget(control_group)

        main_splitter.addWidget(right_widget)
        main_splitter.setSizes([420, 680])

        layout.addWidget(main_splitter)

    def _on_series_changed(self, index: int):
        """When model series changes, populate the model combo with matching models."""
        self.model_combo.clear()
        if 0 <= index < len(self._model_series):
            for model_name in self._model_series[index][1]:
                self.model_combo.addItem(model_name)
            self.model_combo.setCurrentIndex(0)

    def _get_selected_model(self) -> str:
        """Get the actual model name from combo (or custom path from editable text)."""
        text = self.model_combo.currentText().strip()
        return text

    def browse_model(self):
        """Browse for a custom model file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "", "PyTorch 模型 (*.pt);;ONNX 模型 (*.onnx);;所有文件 (*)"
        )
        if path:
            self.model_combo.setCurrentText(path)

    def refresh_gpu_info(self):
        """Re-detect GPU and update display."""
        self.gpu_detection = detect_cuda()
        self.gpu_info_label.setText(self._format_device_summary())

        # Update device combo
        self._populate_device_combo()

    def _format_device_summary(self) -> str:
        """Format a combined GPU + CPU device summary for display."""
        lines = []

        # CPU info
        import platform
        cpu_name = platform.processor() or "Unknown CPU"
        lines.append(f"CPU: {cpu_name}")
        lines.append(f"   系统: {platform.system()} {platform.release()}")

        lines.append("")

        # GPU info
        d = self.gpu_detection
        if d.torch_version:
            lines.append(f"PyTorch: {d.torch_version}")

        if d.cuda_available:
            lines.append(f"CUDA: {d.cuda_version}")
            if d.driver_version:
                lines.append(f"   驱动: {d.driver_version}")
            lines.append(f"   GPU 数量: {d.gpu_count}")
            for gpu in d.gpus:
                vr = f"{gpu.vram_total_mb}MB"
                if gpu.vram_free_mb:
                    vr += f" (可用: {gpu.vram_free_mb}MB)"
                temp = f" | {gpu.temperature}°C" if gpu.temperature else ""
                util = f" | 利用率 {gpu.utilization}%" if gpu.utilization else ""
                lines.append(f"   [{gpu.index}] {gpu.name}")
                lines.append(f"      显存: {vr}{temp}{util}")
        else:
            lines.append("CUDA: 不可用")
            if d.error:
                lines.append(f"   原因: {d.error}")
            lines.append("   将使用 CPU 训练 (速度较慢)")

        return "\n".join(lines)

    def _populate_device_combo(self):
        """Populate the device combo with clearly labeled GPU/CPU options."""
        self.device_combo.clear()
        d = self.gpu_detection

        if d.cuda_available and d.gpus:
            # Add GPU options with names
            for gpu in d.gpus:
                label = f"GPU {gpu.index}: {gpu.name}"
                if gpu.vram_free_mb:
                    label += f" (可用 {gpu.vram_free_mb}MB)"
                self.device_combo.addItem(label, str(gpu.index))

            # Multi-GPU option
            if d.gpu_count > 1:
                multi = ",".join(str(i) for i in range(d.gpu_count))
                self.device_combo.addItem(f"多GPU: {multi}", multi)

        # CPU option (always available)
        import platform
        cpu_name = platform.processor() or "CPU"
        self.device_combo.addItem(f"CPU: {cpu_name}", "cpu")

        # Set recommended device
        recommended = d.recommended_device
        for i in range(self.device_combo.count()):
            if self.device_combo.itemData(i) == recommended:
                self.device_combo.setCurrentIndex(i)
                break

    def _update_device_status(self, text: str):
        """Update the device status label based on selected device."""
        device_data = self.device_combo.currentData() or text

        if device_data == "cpu" or device_data.lower() == "cpu":
            self.device_status_label.setText("CPU 模式 — 训练速度较慢")
            self.device_status_label.set_variant("warning")
        elif device_data and device_data.replace(",", "").isdigit():
            gpu_count = len(device_data.split(","))
            if gpu_count > 1:
                self.device_status_label.setText(f"多GPU模式 — {gpu_count} 张GPU并行训练")
                self.device_status_label.set_variant("accent")
            else:
                # Find GPU name
                gpu_idx = int(device_data)
                gpu_name = "GPU"
                if gpu_idx < len(self.gpu_detection.gpus):
                    gpu = self.gpu_detection.gpus[gpu_idx]
                    free_mb = gpu.vram_free_mb or 0
                    gpu_name = f"{gpu.name} (可用 {free_mb}MB)"
                self.device_status_label.setText(f"GPU 加速 — {gpu_name}")
                self.device_status_label.set_variant("accent")
        else:
            self.device_status_label.setText(f"设备: {text}")
            self.device_status_label.set_variant("")

    def browse_data_yaml(self):
        """Browse for data.yaml file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 data.yaml", "", "YAML 文件 (*.yaml *.yml);;所有文件 (*)"
        )
        if path:
            self.data_yaml_edit.setText(path)

    def browse_project(self):
        """Browse for project directory."""
        path = QFileDialog.getExistingDirectory(self, "选择项目目录")
        if path:
            self.project_edit.setText(path)

    def browse_font(self):
        """Browse for a font file (.ttf) to use for training plots."""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择中文字体文件", "", "TrueType 字体 (*.ttf);;OpenType 字体 (*.otf);;所有文件 (*)"
        )
        if path:
            self.font_path_edit.setText(path)
            self._update_font_status()

    def _auto_detect_font(self) -> str:
        """Auto-detect Arial.Unicode.ttf in project root or common locations."""
        candidates = [
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Arial.Unicode.ttf"),
            os.path.join(os.getcwd(), "Arial.Unicode.ttf"),
            os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "Ultralytics", "Arial.Unicode.ttf"),
            os.path.join(os.path.expanduser("~"), ".config", "Ultralytics", "Arial.Unicode.ttf"),
        ]
        for path in candidates:
            if os.path.isfile(path):
                logger.info(f"Auto-detected font: {path}")
                return path
        return ""

    def _update_font_status(self):
        """Update the font status label."""
        font_path = self.font_path_edit.text().strip()
        if font_path and os.path.isfile(font_path):
            self.font_status_label.setText(f"已选择: {os.path.basename(font_path)}")
            self.font_status_label.setStyleSheet("color: #3fb950; font-size: 11px;")
        elif font_path:
            self.font_status_label.setText("字体文件不存在")
            self.font_status_label.setStyleSheet("color: #f85149; font-size: 11px;")
        else:
            self.font_status_label.setText("未指定字体 — 将跳过字体注入")
            self.font_status_label.setStyleSheet("color: #8b949e; font-size: 11px;")

    @staticmethod
    def _setup_font_for_training(font_path: str):
        """Copy the user's font to ultralytics' expected location so it skips the download.

        Ultralytics calls check_font() which looks in:
        - Windows: %APPDATA%/Ultralytics/
        - Linux/Mac: ~/.config/Ultralytics/
        """
        if not font_path or not os.path.isfile(font_path):
            return False

        import shutil
        target_name = "Arial.Unicode.ttf"

        if sys.platform == "win32":
            target_dir = os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")), "Ultralytics")
        else:
            target_dir = os.path.join(os.path.expanduser("~"), ".config", "Ultralytics")

        os.makedirs(target_dir, exist_ok=True)
        target_path = os.path.join(target_dir, target_name)

        try:
            shutil.copy2(font_path, target_path)
            logger.info(f"Font injected: {font_path} -> {target_path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to inject font: {e}")
            return False

    def _auto_recommend_params(self):
        """Analyze data.yaml and GPU, then auto-adjust training parameters.

        Heuristics based on:
        - Dataset size (few images → more epochs, less augmentation)
        - GPU VRAM (low VRAM → smaller batch/imgsz, no AMP on old cards)
        - Number of classes
        - Windows vs Linux (worker handling)
        """
        data_yaml = self.data_yaml_edit.text().strip()
        if not data_yaml or not os.path.isfile(data_yaml):
            QMessageBox.warning(self, "提示", "请先选择有效的 data.yaml 文件")
            return

        # Parse data.yaml
        try:
            import yaml
        except ImportError:
            QMessageBox.critical(self, "错误", "需要安装 PyYAML: pip install pyyaml")
            return

        with open(data_yaml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        yaml_dir = os.path.dirname(os.path.abspath(data_yaml))

        def _count_images(dir_path: str) -> int:
            if not dir_path or not os.path.isdir(dir_path):
                return 0
            exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
            return sum(
                1 for f in os.listdir(dir_path)
                if os.path.isfile(os.path.join(dir_path, f))
                and os.path.splitext(f)[1].lower() in exts
            )

        # Resolve paths relative to yaml dir
        train_path = data.get("train", "")
        if train_path and not os.path.isabs(train_path):
            train_path = os.path.join(yaml_dir, train_path)
        val_path = data.get("val", "")
        if val_path and not os.path.isabs(val_path):
            val_path = os.path.join(yaml_dir, val_path)

        train_count = _count_images(train_path)
        val_count = _count_images(val_path)
        total = train_count + val_count
        nc = data.get("nc", 1)

        # GPU info
        vram_gb = 0
        is_nvidia = False
        cc = ""
        if self.gpu_detection.cuda_available and self.gpu_detection.gpus:
            gpu = self.gpu_detection.gpus[0]
            vram_gb = (gpu.vram_total_mb or 0) / 1024
            is_nvidia = gpu.name and "nvidia" in gpu.name.lower()
            # Read compute capability if stored
            cc = getattr(gpu, "compute_capability", "")

        model_name = self._get_selected_model()

        # Model size factor: larger models need more VRAM per image
        m = model_name.lower()
        if "x" in m or "e" in m:
            model_factor = 0.35
        elif "l" in m:
            model_factor = 0.5
        elif "m" in m:
            model_factor = 0.65
        elif "s" in m:
            model_factor = 0.8
        else:  # nano
            model_factor = 1.0
        # Segmentation/pose/OBB tasks use ~30% more VRAM
        if any(t in m for t in ("-seg", "-pose", "-obb")):
            model_factor *= 0.7

        # ── Heuristics ──────────────────────────────────────────
        changes = []

        # epochs: fewer images → more epochs
        if total < 30:
            epochs = 300
        elif total < 100:
            epochs = 200
        elif total < 500:
            epochs = 150
        elif total < 2000:
            epochs = 100
        elif total < 10000:
            epochs = 60
        else:
            epochs = 30
        self.epochs_spin.setValue(epochs)
        changes.append(f"epochs → {epochs}")

        # batch: based on GPU VRAM, scaled by model size factor
        usable_vram = max(1, vram_gb - 1.0)  # reserve 1GB for framework overhead
        raw_batch = int(usable_vram * 5 * model_factor)  # 5 images/GB for nano@640
        if raw_batch >= 128:
            batch = 128
        elif raw_batch >= 96:
            batch = 96
        elif raw_batch >= 64:
            batch = 64
        elif raw_batch >= 48:
            batch = 48
        elif raw_batch >= 32:
            batch = 32
        elif raw_batch >= 24:
            batch = 24
        elif raw_batch >= 16:
            batch = 16
        elif raw_batch >= 8:
            batch = 8
        else:
            batch = 4
        self.batch_spin.setValue(batch)
        changes.append(f"batch → {batch} ({model_factor:.0%} factor)")

        # imgsz: 640 for 4GB+, otherwise scale down
        if vram_gb >= 4:
            imgsz = 640
        elif vram_gb >= 2:
            imgsz = 480
        else:
            imgsz = 320
        self.imgsz_spin.setValue(imgsz)
        changes.append(f"imgsz → {imgsz}")

        # workers: Windows conservative, Linux liberal
        if sys.platform == "win32":
            workers = min(4, max(0, batch // 8))
        else:
            workers = min(8, max(2, batch // 4))
        self.workers_spin.setValue(workers)
        changes.append(f"workers → {workers}")

        # patience: scale with epochs
        patience = min(epochs, max(20, epochs // 2))
        self.patience_spin.setValue(patience)
        changes.append(f"patience → {patience}")

        # amp: disable on old Pascal / unknown NVIDIA (CC < 7.0)
        is_legacy = False
        if cc:
            try:
                is_legacy = float(cc) < 7.0
            except ValueError:
                pass
        if is_legacy:
            self.amp_check.setChecked(False)
            changes.append("amp → 关闭 (GPU 架构不支持)")
        else:
            self.amp_check.setChecked(True)

        # cache: enable for datasets > 200 images with enough RAM
        if total > 200 and vram_gb >= 4:
            self.cache_check.setChecked(True)
            changes.append("cache → 开启")

        # lr0: linear scaling with batch size (base 0.01 @ batch=16)
        lr0 = round(0.01 * (batch / 16), 5)
        lr0 = max(0.001, min(0.05, lr0))
        self.lr0_spin.setValue(lr0)
        changes.append(f"lr0 → {lr0}")

        # warmup_epochs
        if total < 100:
            warmup = 5.0
        elif total < 500:
            warmup = 3.0
        else:
            warmup = 2.0
        self.warmup_epochs_spin.setValue(warmup)
        changes.append(f"warmup_epochs → {warmup}")

        # close_mosaic
        if total > 200:
            close_mosaic = 10
        elif total > 30:
            close_mosaic = 5
        else:
            close_mosaic = 0
        self.close_mosaic_spin.setValue(close_mosaic)
        changes.append(f"close_mosaic → {close_mosaic}")

        # mosaic
        if total < 20:
            self.mosaic_spin.setValue(0.0)
            changes.append("mosaic → 关闭 (数据量太少)")
        else:
            self.mosaic_spin.setValue(1.0)

        # mixup: only for larger datasets
        if total >= 500:
            self.mixup_spin.setValue(0.1)
            changes.append("mixup → 0.1")
        elif total >= 200:
            self.mixup_spin.setValue(0.05)
            changes.append("mixup → 0.05")
        else:
            self.mixup_spin.setValue(0.0)

        # cos_lr: enable for small datasets
        if total < 500:
            self.cos_lr_check.setChecked(True)
            changes.append("cos_lr → 开启 (小数据集)")
        else:
            self.cos_lr_check.setChecked(False)

        # seed: keep current or randomize
        if self.seed_spin.value() == 0:
            import random
            self.seed_spin.setValue(random.randint(1, 99999))
            changes.append(f"seed → {self.seed_spin.value()}")

        # ── Show summary ────────────────────────────────────────
        gpu_info = f"{vram_gb:.0f}GB" + (f", CC={cc}" if cc else "") if vram_gb > 0 else "CPU"
        summary = (
            f"数据集: {total} 张图片 ({train_count} train / {val_count} val)\n"
            f"类别数: {nc}\n"
            f"模型: {model_name}\n"
            f"GPU: {gpu_info}\n\n"
            f"推荐参数调整 ({len(changes)} 项):\n"
            + "\n".join(f"  ✓ {c}" for c in changes)
        )

        QMessageBox.information(self, "智能推荐完成", summary)
        self.log_text.append(f"\n{'─'*50}\n[智能推荐] 分析 {total} 张图\n" + "\n".join(changes) + "\n")

    def get_training_args(self) -> dict:
        """Get training arguments from UI.

        Keys match ultralytics YOLO.train() parameter names exactly.
        "model' and 'data_yaml' are handled separately by start_training().
        """
        return {
            "model": self._get_selected_model(),
            "data_yaml": self.data_yaml_edit.text(),
            "epochs": self.epochs_spin.value(),
            "batch": self.batch_spin.value(),
            "imgsz": self.imgsz_spin.value(),
            "device": self.device_combo.currentData() or self.device_combo.currentText(),
            "workers": self.workers_spin.value(),
            "patience": self.patience_spin.value(),
            "amp": self.amp_check.isChecked(),
            "cache": self.cache_check.isChecked(),
            "seed": self.seed_spin.value(),
            "optimizer": self.optimizer_combo.currentText(),
            "lr0": self.lr0_spin.value(),
            "lrf": self.lrf_spin.value(),
            "momentum": self.momentum_spin.value(),
            "weight_decay": self.weight_decay_spin.value(),
            "warmup_epochs": self.warmup_epochs_spin.value(),
            "warmup_momentum": self.warmup_momentum_spin.value(),
            "cos_lr": self.cos_lr_check.isChecked(),
            "close_mosaic": self.close_mosaic_spin.value(),
            "fliplr": self.fliplr_spin.value(),
            "mosaic": self.mosaic_spin.value(),
            "hsv_h": self.hsv_h_spin.value(),
            "hsv_s": self.hsv_s_spin.value(),
            "hsv_v": self.hsv_v_spin.value(),
            "degrees": self.degrees_spin.value(),
            "scale": self.scale_spin.value(),
            "translate": self.translate_spin.value(),
            "shear": self.shear_spin.value(),
            "flipud": self.flipud_spin.value(),
            "mixup": self.mixup_spin.value(),
            "erasing": self.erasing_spin.value(),
            "project": self.project_edit.text(),
            "name": self.name_edit.text(),
            "exist_ok": self.exist_ok_check.isChecked(),
        }

    def start_training(self):
        """Start the training process."""
        args = self.get_training_args()

        if not args["model"]:
            QMessageBox.warning(self, "错误", "请先选择一个预训练模型或输入自定义模型路径")
            return

        if not args["data_yaml"] or not os.path.exists(args["data_yaml"]):
            QMessageBox.warning(self, "错误", "请先选择有效的 data.yaml 文件")
            return

        # Inject font to avoid ultralytics downloading Arial.Unicode.ttf from internet
        font_path = self.font_path_edit.text().strip()
        if not font_path:
            font_path = self._auto_detect_font()
        if font_path:
            self._setup_font_for_training(font_path)
            self.log_text.append(f"字体已注入: {font_path}")

        # Load model
        try:
            self.trainer.load_model(args["model"])
        except Exception as e:
            QMessageBox.critical(self, "模型加载失败", str(e))
            return

        # Prepare kwargs — remove keys that are handled separately
        kwargs = {k: v for k, v in args.items() if k not in ("model", "data_yaml")}

        # Update UI state
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.log_text.clear()
        self.log_text.append(f"训练开始...\n模型: {args['model']}\n数据: {args['data_yaml']}\n")
        self.training_chart.clear_chart()

        # Reset early stopping tracking
        self._best_map = 0.0
        self._best_map_epoch = 0
        self._no_improve_count = 0
        self.early_stop_label.setVisible(False)

        self.training_started.emit()

        # Reset live metrics
        self._live_logs = {"epochs": []}

        # Start worker thread
        self.worker = TrainingWorker(self.trainer, args["data_yaml"], **kwargs)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

        # Start polling epoch metrics from worker
        self._epoch_timer.start()

    def stop_training(self):
        """Stop the training process."""
        # Signal Ultralytics trainer to stop
        if self.trainer.model and hasattr(self.trainer.model, 'trainer'):
            try:
                self.trainer.model.trainer.stop_training = True
            except Exception:
                pass
        self.trainer.stop_training()
        self.log_text.append("\n正在停止训练...\n")

    def on_progress(self, text: str):
        """Handle training progress output — append to log panel."""
        # Filter out ANSI escape codes for cleaner display
        import re
        # Strip ANSI sequences
        clean = re.sub(r'\x1b\[[0-9;]*[mGKHJF]', '', text)
        if clean.strip():
            self.log_text.append(clean.rstrip())
            self.log_text.verticalScrollBar().setValue(
                self.log_text.verticalScrollBar().maximum()
            )

    def _poll_epoch_updates(self):
        """Timer callback: drain epoch queue from worker and update UI."""
        if self.worker is None:
            return
        items = self.worker.drain_epoch_queue()
        for epoch, metrics in items:
            self.on_epoch_update(epoch, metrics)

    def on_epoch_update(self, epoch: int, metrics: dict):
        """Handle per-epoch metrics: update progress bar + live chart."""
        # Update progress bar
        total = self.epochs_spin.value()
        if total > 0:
            pct = int(epoch / total * 100)
            self.progress_bar.setValue(min(pct, 99))  # 100% only on finish

        # Append to live logs
        self._live_logs["epochs"].append(epoch)
        for key, val in metrics.items():
            self._live_logs.setdefault(key, []).append(val)

        # Update chart every epoch so users can see progress immediately
        try:
            self.training_chart.update_chart(self._live_logs)
            # Auto-switch to chart tab on first epoch
            if epoch == 1:
                self.chart_tabs.setCurrentWidget(self.training_chart)
        except Exception:
            pass  # chart update failures are non-critical

        # Log a compact metric summary to the log panel
        if metrics:
            parts = []
            for key in ("train/box_loss", "train/cls_loss", "train/dfl_loss",
                        "val/box_loss", "val/cls_loss", "val/dfl_loss",
                        "metrics/mAP50(B)", "metrics/mAP50-95(B)"):
                if key in metrics:
                    short = key.replace("train/", "tr/").replace("val/", "v/").replace("metrics/", "")
                    parts.append(f"{short}={metrics[key]:.4f}")
            if parts:
                self.log_text.append(f"  [Epoch {epoch}] {' | '.join(parts)}")

        # Track best mAP for early stopping display
        current_map = metrics.get("metrics/mAP50-95(B)", 0)
        if current_map > 0:
            if current_map > self._best_map + 0.001:  # 0.1% improvement threshold
                self._best_map = current_map
                self._best_map_epoch = epoch
                self._no_improve_count = 0
            else:
                self._no_improve_count += 1

            patience = self.patience_spin.value()
            if patience > 0:
                remaining = patience - self._no_improve_count
                color = "#3fb950" if self._no_improve_count == 0 else (
                    "#f85149" if remaining < patience * 0.3 else "#d29922"
                )
                self.early_stop_label.setText(
                    f"Best mAP: {self._best_map:.4f} @ epoch {self._best_map_epoch}  |  "
                    f"未提升: {self._no_improve_count}/{patience} 轮"
                )
                self.early_stop_label.setStyleSheet(
                    f"color: {color}; font-size: 12px; padding: 4px 0; font-weight: bold;"
                )
                self.early_stop_label.setVisible(True)

    def on_finished(self, result: dict):
        """Handle training completion."""
        # Stop epoch polling timer
        self._epoch_timer.stop()

        # Drain any remaining epoch updates
        if self.worker:
            for epoch, metrics in self.worker.drain_epoch_queue():
                self.on_epoch_update(epoch, metrics)

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        if result.get("success"):
            self.progress_bar.setValue(100)

            # Show best mAP summary in the early stop label
            if self._best_map > 0:
                self.early_stop_label.setText(
                    f"训练完成 — Best mAP50-95: {self._best_map:.4f} @ epoch {self._best_map_epoch}"
                )
                self.early_stop_label.setStyleSheet("color: #3fb950; font-size: 12px; padding: 4px 0; font-weight: bold;")
                self.early_stop_label.setVisible(True)

            save_dir = result.get("save_dir") or "unknown"
            self.log_text.append(f"\n训练完成! 结果保存在: {save_dir}")

            # Try loading the full results.csv for the final chart
            loaded = False
            if save_dir and save_dir != "unknown" and os.path.isdir(save_dir):
                loaded = self._load_training_logs(save_dir)

            # Fallback: use accumulated live metrics
            if not loaded and self._live_logs.get("epochs"):
                self.training_chart.update_chart(self._live_logs)
                self.chart_tabs.setCurrentWidget(self.training_chart)

            # Find best.pt path
            best_pt = ""
            if save_dir and save_dir != "unknown":
                _best = os.path.join(save_dir, "weights", "best.pt")
                if os.path.isfile(_best):
                    best_pt = _best

            # Show completion dialog with quick actions
            dlg = TrainingCompleteDialog(save_dir, best_pt, self)
            dlg.exec()

            if dlg.action and best_pt:
                self.model_ready.emit(best_pt, dlg.action)
        else:
            self.log_text.append(f"\n训练失败: {result.get('error', '未知错误')}")
            # Still show whatever chart data we collected
            if self._live_logs.get("epochs"):
                self.training_chart.update_chart(self._live_logs)
                self.chart_tabs.setCurrentWidget(self.training_chart)
            QMessageBox.critical(self, "训练失败", result.get("error", "未知错误"))

        self.training_finished.emit(result)

    def _load_training_logs(self, project_dir: str) -> bool:
        """Load training logs from results.csv and update chart.

        Returns True if logs were successfully loaded, False otherwise.
        """
        try:
            logs = self.trainer.get_training_logs(project_dir)
            if logs and logs.get("epochs"):
                self.training_chart.update_chart(logs)
                self.chart_tabs.setCurrentWidget(self.training_chart)
                return True
        except Exception as e:
            self.log_text.append(f"\n加载训练日志失败: {e}")
        return False

    def refresh_chart(self):
        """Refresh the training chart from current project directory.

        Tries multiple possible locations for the training results.
        """
        # If we have live data, use it directly
        if self._live_logs.get("epochs"):
            self.training_chart.update_chart(self._live_logs)
            self.chart_tabs.setCurrentWidget(self.training_chart)
            return

        # Otherwise try to find results.csv on disk
        project_dir = self.project_edit.text()
        name = self.name_edit.text()
        candidates = [
            os.path.join(project_dir, name),
            os.path.join("runs", "detect", name),
            os.path.join("runs", "train", name),
        ]
        for full_path in candidates:
            if os.path.exists(full_path):
                if self._load_training_logs(full_path):
                    return

        QMessageBox.information(self, "提示", f"未找到训练结果。\n已搜索: {', '.join(candidates)}")

    # ------------------------------------------------------------------
    # Training templates
    # ------------------------------------------------------------------

    def _template_key(self) -> str:
        return "training_templates"

    def _load_template_list(self):
        """Populate template combo with saved template names."""
        current = self.template_combo.currentText() if hasattr(self, "template_combo") else ""
        self.template_combo.clear()
        from PyQt6.QtCore import QSettings
        settings = QSettings("YOLOStudio", "TrainingTemplates")
        templates = settings.value(self._template_key(), [])
        if not isinstance(templates, list):
            templates = []
        for name in sorted({str(name) for name in templates if str(name).strip()}):
            if isinstance(name, str):
                self.template_combo.addItem(name)
        if current:
            self.template_combo.setCurrentText(current)

    def _save_template(self):
        """Save current training configuration as a named template."""
        name, ok = QInputDialog.getText(self, "保存模板", "模板名称:")
        if not ok or not name.strip():
            return
        name = name.strip()

        from PyQt6.QtCore import QSettings
        settings = QSettings("YOLOStudio", "TrainingTemplates")
        template_names = settings.value(self._template_key(), []) or []
        if not isinstance(template_names, list):
            template_names = []

        template_data = {
            "model_series": self.model_series_combo.currentIndex(),
            "model": self._get_selected_model(),
            "data_yaml": self.data_yaml_edit.text(),
            "device": self.device_combo.currentData() or self.device_combo.currentText(),
            "project": self.project_edit.text(),
            "name": self.name_edit.text(),
            "exist_ok": self.exist_ok_check.isChecked(),
            "epochs": self.epochs_spin.value(),
            "batch": self.batch_spin.value(),
            "imgsz": self.imgsz_spin.value(),
            "workers": self.workers_spin.value(),
            "patience": self.patience_spin.value(),
            "amp": self.amp_check.isChecked(),
            "cache": self.cache_check.isChecked(),
            "seed": self.seed_spin.value(),
            "optimizer": self.optimizer_combo.currentIndex(),
            "lr0": self.lr0_spin.value(),
            "lrf": self.lrf_spin.value(),
            "momentum": self.momentum_spin.value(),
            "weight_decay": self.weight_decay_spin.value(),
            "warmup_epochs": self.warmup_epochs_spin.value(),
            "warmup_momentum": self.warmup_momentum_spin.value(),
            "cos_lr": self.cos_lr_check.isChecked(),
            "close_mosaic": self.close_mosaic_spin.value(),
            "fliplr": self.fliplr_spin.value(),
            "mosaic": self.mosaic_spin.value(),
            "hsv_h": self.hsv_h_spin.value(),
            "hsv_s": self.hsv_s_spin.value(),
            "hsv_v": self.hsv_v_spin.value(),
            "degrees": self.degrees_spin.value(),
            "scale": self.scale_spin.value(),
            "translate": self.translate_spin.value(),
            "shear": self.shear_spin.value(),
            "flipud": self.flipud_spin.value(),
            "mixup": self.mixup_spin.value(),
            "erasing": self.erasing_spin.value(),
            "font_path": self.font_path_edit.text(),
        }

        template_data["_name"] = name
        cleaned_names = [str(item) for item in template_names if str(item).strip() and str(item) != name]
        cleaned_names.append(name)
        settings.setValue(self._template_key(), cleaned_names)
        settings.setValue(f"template_{name}", template_data)

        settings.sync()
        self._load_template_list()
        self.template_combo.setCurrentText(name)
        self.log_text.append(f"模板已保存: {name}")

    def _load_template(self):
        """Load selected template into the UI."""
        name = self.template_combo.currentText()
        if not name:
            return

        from PyQt6.QtCore import QSettings
        settings = QSettings("YOLOStudio", "TrainingTemplates")
        template = settings.value(f"template_{name}")
        if not template or not isinstance(template, dict):
            QMessageBox.warning(self, "提示", f"模板 '{name}' 数据无效")
            return

        # Restore values
        if "model_series" in template:
            idx = int(template["model_series"])
            if 0 <= idx < self.model_series_combo.count():
                self.model_series_combo.setCurrentIndex(idx)
        if "model" in template:
            self.model_combo.setCurrentText(str(template["model"]))
        if "data_yaml" in template:
            self.data_yaml_edit.setText(str(template["data_yaml"]))
        if "device" in template:
            dev = str(template["device"])
            for i in range(self.device_combo.count()):
                if self.device_combo.itemData(i) == dev or self.device_combo.itemText(i) == dev:
                    self.device_combo.setCurrentIndex(i)
                    break
            else:
                self.device_combo.setCurrentText(dev)
        if "project" in template:
            self.project_edit.setText(str(template["project"]))
        if "name" in template:
            self.name_edit.setText(str(template["name"]))
        if "exist_ok" in template:
            self.exist_ok_check.setChecked(bool(template["exist_ok"]))
        if "epochs" in template:
            self.epochs_spin.setValue(int(template["epochs"]))
        if "batch" in template:
            self.batch_spin.setValue(int(template["batch"]))
        if "imgsz" in template:
            self.imgsz_spin.setValue(int(template["imgsz"]))
        if "workers" in template:
            self.workers_spin.setValue(int(template["workers"]))
        if "patience" in template:
            self.patience_spin.setValue(int(template["patience"]))
        if "amp" in template:
            self.amp_check.setChecked(bool(template["amp"]))
        if "cache" in template:
            self.cache_check.setChecked(bool(template["cache"]))
        if "seed" in template:
            self.seed_spin.setValue(int(template["seed"]))
        if "optimizer" in template:
            oidx = int(template["optimizer"])
            if 0 <= oidx < self.optimizer_combo.count():
                self.optimizer_combo.setCurrentIndex(oidx)
        if "lr0" in template:
            self.lr0_spin.setValue(float(template["lr0"]))
        if "lrf" in template:
            self.lrf_spin.setValue(float(template["lrf"]))
        if "momentum" in template:
            self.momentum_spin.setValue(float(template["momentum"]))
        if "weight_decay" in template:
            self.weight_decay_spin.setValue(float(template["weight_decay"]))
        if "warmup_epochs" in template:
            self.warmup_epochs_spin.setValue(float(template["warmup_epochs"]))
        if "warmup_momentum" in template:
            self.warmup_momentum_spin.setValue(float(template["warmup_momentum"]))
        if "cos_lr" in template:
            self.cos_lr_check.setChecked(bool(template["cos_lr"]))
        if "close_mosaic" in template:
            self.close_mosaic_spin.setValue(int(template["close_mosaic"]))
        if "fliplr" in template:
            self.fliplr_spin.setValue(float(template["fliplr"]))
        if "mosaic" in template:
            self.mosaic_spin.setValue(float(template["mosaic"]))
        if "hsv_h" in template:
            self.hsv_h_spin.setValue(float(template["hsv_h"]))
        if "hsv_s" in template:
            self.hsv_s_spin.setValue(float(template["hsv_s"]))
        if "hsv_v" in template:
            self.hsv_v_spin.setValue(float(template["hsv_v"]))
        if "degrees" in template:
            self.degrees_spin.setValue(float(template["degrees"]))
        if "scale" in template:
            self.scale_spin.setValue(float(template["scale"]))
        if "translate" in template:
            self.translate_spin.setValue(float(template["translate"]))
        if "shear" in template:
            self.shear_spin.setValue(float(template["shear"]))
        if "flipud" in template:
            self.flipud_spin.setValue(float(template["flipud"]))
        if "mixup" in template:
            self.mixup_spin.setValue(float(template["mixup"]))
        if "erasing" in template:
            self.erasing_spin.setValue(float(template["erasing"]))
        if "font_path" in template:
            self.font_path_edit.setText(str(template["font_path"]))
            self._update_font_status()

        self.log_text.append(f"模板已加载: {name}")

    def _delete_template(self):
        """Delete the selected template."""
        name = self.template_combo.currentText()
        if not name:
            return

        reply = QMessageBox.question(
            self, "删除模板", f"确定删除模板 '{name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        from PyQt6.QtCore import QSettings
        settings = QSettings("YOLOStudio", "TrainingTemplates")
        saved = settings.value(self._template_key(), []) or []
        if isinstance(saved, list):
            saved = [str(t) for t in saved if str(t) != name]
            settings.setValue(self._template_key(), saved)
        settings.remove(f"template_{name}")
        settings.sync()
        self._load_template_list()
        self.log_text.append(f"模板已删除: {name}")
