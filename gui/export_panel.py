"""Model export panel."""

import os

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.exporter import ModelExporter


class ExportWorker(QThread):
    """Worker thread for model export."""
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)

    def __init__(self, exporter, format, **kwargs):
        super().__init__()
        self.exporter = exporter
        self.format = format
        self.kwargs = kwargs

    def run(self):
        try:
            result = self.exporter.export(self.format, **self.kwargs)
            self.finished.emit(result)
        except Exception as e:
            self.finished.emit({"success": False, "error": str(e)})


class ExportPanel(QWidget):
    """Panel for exporting YOLO models."""

    export_finished = pyqtSignal(dict)

    def __init__(self, config_manager=None, parent=None):
        super().__init__(parent)
        self.config = config_manager
        self.exporter = None
        self.worker = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(6)

        settings_widget = QWidget()
        settings_widget.setMinimumWidth(380)
        settings_layout = QVBoxLayout(settings_widget)
        settings_layout.setContentsMargins(0, 0, 8, 0)
        settings_layout.setSpacing(8)

        settings_title = QLabel("导出设置")
        settings_title.setObjectName("PanelTitle")
        settings_layout.addWidget(settings_title)

        model_label = QLabel("模型")
        model_label.setObjectName("MutedText")
        settings_layout.addWidget(model_label)

        path_row = QHBoxLayout()
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("选择要导出的模型 (.pt)")
        browse_btn = QPushButton("浏览...")
        browse_btn.setObjectName("QuietButton")
        browse_btn.clicked.connect(self.browse_model)
        self.load_btn = QPushButton("加载")
        self.load_btn.setObjectName("SecondaryButton")
        self.load_btn.clicked.connect(self.load_model)

        path_row.addWidget(self.model_path_edit, stretch=1)
        path_row.addWidget(browse_btn)
        path_row.addWidget(self.load_btn)
        settings_layout.addLayout(path_row)

        recent_row = QHBoxLayout()
        recent_row.addWidget(QLabel("最近训练:"))
        self.recent_model_combo = QComboBox()
        self.recent_model_combo.setPlaceholderText("选择最近训练的模型...")
        self.recent_model_combo.setMinimumWidth(240)
        self._refresh_recent_models()
        self.recent_model_combo.currentTextChanged.connect(self._on_recent_model_selected)
        recent_row.addWidget(self.recent_model_combo, stretch=1)

        refresh_btn = QPushButton("刷新")
        refresh_btn.setObjectName("QuietButton")
        refresh_btn.clicked.connect(self._refresh_recent_models)
        recent_row.addWidget(refresh_btn)
        settings_layout.addLayout(recent_row)

        format_label = QLabel("格式与参数")
        format_label.setObjectName("MutedText")
        settings_layout.addWidget(format_label)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("导出格式:"))
        self.format_combo = QComboBox()
        formats = ModelExporter.get_supported_formats()
        for key, info in formats.items():
            self.format_combo.addItem(f"{info['description']} ({key})", key)
        row1.addWidget(self.format_combo, stretch=1)
        settings_layout.addLayout(row1)

        row2 = QHBoxLayout()
        self.half_check = QCheckBox("FP16 半精度")
        self.half_check.setToolTip("FP16 需要 NVIDIA GPU，CPU 模式不可用")
        self.dynamic_check = QCheckBox("动态输入")
        self.simplify_check = QCheckBox("简化模型")
        self.simplify_check.setChecked(True)
        row2.addWidget(self.half_check)
        row2.addWidget(self.dynamic_check)
        row2.addWidget(self.simplify_check)
        row2.addStretch()
        settings_layout.addLayout(row2)

        # Auto-disable FP16 if CUDA is not available
        self._check_cuda_for_half()

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("图像尺寸:"))
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(32, 4096)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(640)
        row3.addWidget(self.imgsz_spin)

        row3.addSpacing(8)
        row3.addWidget(QLabel("ONNX Opset:"))
        self.opset_spin = QSpinBox()
        self.opset_spin.setRange(0, 23)
        self.opset_spin.setSpecialValueText("自动")
        self.opset_spin.setValue(17)
        row3.addWidget(self.opset_spin)
        row3.addStretch()
        settings_layout.addLayout(row3)

        self.export_btn = QPushButton("导出模型")
        self.export_btn.setObjectName("PrimaryButton")
        self.export_btn.clicked.connect(self.export_model)
        settings_layout.addWidget(self.export_btn)
        settings_layout.addStretch()
        splitter.addWidget(settings_widget)

        activity_widget = QWidget()
        activity_layout = QVBoxLayout(activity_widget)
        activity_layout.setContentsMargins(8, 0, 0, 0)
        activity_layout.setSpacing(8)

        activity_title = QLabel("导出活动")
        activity_title.setObjectName("PanelTitle")
        activity_layout.addWidget(activity_title)

        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setFont(QFont("monospace", 9))
        self.status_text.setPlaceholderText(
            "尚无导出记录\n加载模型并开始导出后，进度、错误与输出路径会显示在这里。"
        )
        activity_layout.addWidget(self.status_text, stretch=1)

        info_title = QLabel("格式说明")
        info_title.setObjectName("MutedText")
        activity_layout.addWidget(info_title)

        self.format_info_label = QLabel()
        self.format_info_label.setObjectName("MutedText")
        self.format_info_label.setWordWrap(True)
        self.format_info_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        self.format_info_label.setText(self._get_format_info("onnx"))
        activity_layout.addWidget(self.format_info_label)
        splitter.addWidget(activity_widget)

        splitter.setSizes([460, 700])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, stretch=1)

        self.format_combo.currentIndexChanged.connect(
            lambda: self.format_info_label.setText(
                self._get_format_info(self.format_combo.currentData())
            )
        )

    def browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择模型", "", "模型文件 (*.pt);;所有文件 (*)"
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
                    import time
                    mtime = os.path.getmtime(pt)
                    rel = os.path.relpath(pt, os.getcwd())
                    candidates.append((mtime, rel, pt))

        # Sort by most recent first
        candidates.sort(key=lambda x: x[0], reverse=True)

        for mtime, rel, full_path in candidates[:20]:
            import time
            timestr = time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime))
            display = f"{rel}  ({timestr})"
            self.recent_model_combo.addItem(display, full_path)

        self.recent_model_combo.blockSignals(False)

    def _on_recent_model_selected(self, text: str):
        """When user selects a recent model from the dropdown, set the path."""
        path = self.recent_model_combo.currentData()
        if path:
            self.model_path_edit.setText(path)

    def load_model(self):
        model_path = self.model_path_edit.text()
        if not model_path or not os.path.exists(model_path):
            QMessageBox.warning(self, "错误", "请选择有效的模型文件")
            return

        try:
            self.exporter = ModelExporter(model_path)
            self.exporter.load_model()
            self.status_text.append(f"模型加载成功: {model_path}")
            QMessageBox.information(self, "成功", "模型加载成功!")
        except Exception as e:
            QMessageBox.critical(self, "加载失败", str(e))

    def _check_cuda_for_half(self):
        """Disable FP16 checkbox if CUDA is not available (CPU-only torch or no GPU)."""
        try:
            from core.gpu import detect_cuda
            detection = detect_cuda()
            if not detection.cuda_available:
                self.half_check.setChecked(False)
                self.half_check.setEnabled(False)
                if detection.torch_version:
                    self.half_check.setToolTip("CUDA 不可用（安装了 CPU 版 PyTorch），FP16 不可用")
                else:
                    self.half_check.setToolTip("PyTorch 未安装，FP16 不可用")
        except Exception:
            pass  # harmless: GPU detection failed, leave checkbox as-is

    def export_model(self, *, workflow_mode: bool = False):
        if not self.exporter:
            if not workflow_mode:
                QMessageBox.warning(self, "错误", "请先加载模型")
            return False

        format_key = self.format_combo.currentData()

        # Check format requirements
        reqs = ModelExporter.check_format_requirements(format_key)
        if not reqs["available"]:
            if not workflow_mode:
                QMessageBox.warning(
                    self, "缺少依赖",
                    f"导出 {format_key} 需要安装: {', '.join(reqs['missing'])}"
                )
            return False

        kwargs = {
            "imgsz": self.imgsz_spin.value(),
            "half": self.half_check.isChecked(),
            "dynamic": self.dynamic_check.isChecked(),
            "simplify": self.simplify_check.isChecked(),
        }

        if format_key == "onnx":
            kwargs["opset"] = self.opset_spin.value()

        self.status_text.clear()
        self.status_text.append(f"正在导出为 {format_key} 格式...")
        self.export_btn.setEnabled(False)
        self._workflow_mode = workflow_mode

        self._cleanup_worker()
        self.worker = ExportWorker(self.exporter, format_key, **kwargs)
        self.worker.finished.connect(self.on_export_finished)
        self.worker.start()
        return True

    def _cleanup_worker(self):
        if self.worker is not None:
            if self.worker.isRunning():
                self.worker.quit()
                if not self.worker.wait(3000):
                    self.worker.terminate()
                    self.worker.wait(1000)
            self.worker.deleteLater()
            self.worker = None

    def on_export_finished(self, result: dict):
        self.export_btn.setEnabled(True)
        self._cleanup_worker()
        quiet = getattr(self, "_workflow_mode", False)
        self._workflow_mode = False

        if result.get("success"):
            path = result.get("path", "")
            self.status_text.append(f"\n导出成功!\n保存路径: {path}")
            if not quiet:
                QMessageBox.information(self, "导出成功", f"模型已导出到:\n{path}")
        else:
            error = result.get("error", "未知错误")
            self.status_text.append(f"\n导出失败: {error}")
            if not quiet:
                QMessageBox.critical(self, "导出失败", error)
        self.export_finished.emit(result)

    def _get_format_info(self, format_key: str) -> str:
        infos = {
            "onnx": "ONNX (Open Neural Network Exchange)\n通用格式，支持多种推理引擎。\n推荐用于跨平台部署。",
            "torchscript": "TorchScript\nPyTorch 原生序列化格式。\n适合 PyTorch 生态部署。",
            "engine": "TensorRT\nNVIDIA GPU 高性能推理引擎。\n需要安装 TensorRT 库。",
            "coreml": "CoreML\nApple 设备推理格式。\n支持 iOS/macOS 设备。",
            "openvino": "OpenVINO\nIntel 硬件优化推理引擎。\n支持 CPU/GPU/VPU。",
            "tflite": "TensorFlow Lite\n移动端和嵌入式设备推理格式。\n支持 Android/iOS。",
        }
        return infos.get(format_key, "选择导出格式查看详情。")
