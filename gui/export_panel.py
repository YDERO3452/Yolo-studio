"""Model export panel."""

import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QLineEdit, QTextEdit, QFileDialog,
    QComboBox, QCheckBox, QSpinBox, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QFont

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

    def __init__(self, config_manager=None, parent=None):
        super().__init__(parent)
        self.config = config_manager
        self.exporter = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Model selection
        model_group = QGroupBox("模型")
        model_layout = QVBoxLayout()

        # Row 1: path + browse + load
        path_row = QHBoxLayout()
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("选择要导出的模型 (.pt)")
        browse_btn = QPushButton("浏览...")
        browse_btn.clicked.connect(self.browse_model)
        self.load_btn = QPushButton("加载")
        self.load_btn.clicked.connect(self.load_model)

        path_row.addWidget(self.model_path_edit)
        path_row.addWidget(browse_btn)
        path_row.addWidget(self.load_btn)
        model_layout.addLayout(path_row)

        # Row 2: recent trained models
        recent_row = QHBoxLayout()
        recent_row.addWidget(QLabel("最近训练:"))
        self.recent_model_combo = QComboBox()
        self.recent_model_combo.setPlaceholderText("选择最近训练的模型...")
        self.recent_model_combo.setMinimumWidth(300)
        self._refresh_recent_models()
        self.recent_model_combo.currentTextChanged.connect(self._on_recent_model_selected)
        recent_row.addWidget(self.recent_model_combo)

        refresh_btn = QPushButton("刷新")
        refresh_btn.setFixedWidth(60)
        refresh_btn.clicked.connect(self._refresh_recent_models)
        recent_row.addWidget(refresh_btn)
        model_layout.addLayout(recent_row)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Export format
        format_group = QGroupBox("导出格式")
        format_layout = QVBoxLayout()

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("导出格式:"))
        self.format_combo = QComboBox()
        formats = ModelExporter.get_supported_formats()
        for key, info in formats.items():
            self.format_combo.addItem(f"{info['description']} ({key})", key)
        row1.addWidget(self.format_combo)
        format_layout.addLayout(row1)

        # Export options
        row2 = QHBoxLayout()
        self.half_check = QCheckBox("FP16 半精度")
        self.dynamic_check = QCheckBox("动态输入")
        self.simplify_check = QCheckBox("简化模型")
        self.simplify_check.setChecked(True)
        row2.addWidget(self.half_check)
        row2.addWidget(self.dynamic_check)
        row2.addWidget(self.simplify_check)
        format_layout.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("图像尺寸:"))
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(32, 4096)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(640)
        row3.addWidget(self.imgsz_spin)

        row3.addWidget(QLabel("ONNX Opset:"))
        self.opset_spin = QSpinBox()
        self.opset_spin.setRange(7, 23)
        self.opset_spin.setSpecialValueText("自动")
        self.opset_spin.setValue(17)
        row3.addWidget(self.opset_spin)
        format_layout.addLayout(row3)

        format_group.setLayout(format_layout)
        layout.addWidget(format_group)

        # Export button
        self.export_btn = QPushButton("导出模型")
        self.export_btn.setObjectName("PrimaryButton")
        self.export_btn.clicked.connect(self.export_model)
        layout.addWidget(self.export_btn)

        # Status
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setFont(QFont("Consolas", 9))
        layout.addWidget(self.status_text)

        # Format info
        info_group = QGroupBox("格式说明")
        info_layout = QVBoxLayout()
        self.format_info_label = QLabel()
        self.format_info_label.setWordWrap(True)
        self.format_info_label.setText(self._get_format_info("onnx"))
        info_layout.addWidget(self.format_info_label)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

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

    def export_model(self):
        if not self.exporter:
            QMessageBox.warning(self, "错误", "请先加载模型")
            return

        format_key = self.format_combo.currentData()

        # Check format requirements
        reqs = ModelExporter.check_format_requirements(format_key)
        if not reqs["available"]:
            QMessageBox.warning(
                self, "缺少依赖",
                f"导出 {format_key} 需要安装: {', '.join(reqs['missing'])}"
            )
            return

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

        self.worker = ExportWorker(self.exporter, format_key, **kwargs)
        self.worker.finished.connect(self.on_export_finished)
        self.worker.start()

    def on_export_finished(self, result: dict):
        self.export_btn.setEnabled(True)

        if result.get("success"):
            path = result.get("path", "")
            self.status_text.append(f"\n导出成功!\n保存路径: {path}")
            QMessageBox.information(self, "导出成功", f"模型已导出到:\n{path}")
        else:
            error = result.get("error", "未知错误")
            self.status_text.append(f"\n导出失败: {error}")
            QMessageBox.critical(self, "导出失败", error)

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
