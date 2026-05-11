"""Auto-labeling panel for automatic annotation using YOLO models."""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton, QComboBox,
    QSpinBox, QDoubleSpinBox, QLabel, QProgressBar, QMessageBox, QFileDialog,
    QCheckBox, QListWidget, QListWidgetItem
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QObject
from PyQt6.QtGui import QIcon
from typing import Optional, List
from loguru import logger

from core.model_manager import ModelManager


class InferenceWorker(QObject):
    """Worker thread for running inference."""

    progress = pyqtSignal(int)  # Progress percentage
    finished = pyqtSignal(dict)  # Results
    error = pyqtSignal(str)  # Error message

    def __init__(self, model_manager: ModelManager, image_paths: List[str],
                 conf: float, iou: float, max_det: int):
        super().__init__()
        self.model_manager = model_manager
        self.image_paths = image_paths
        self.conf = conf
        self.iou = iou
        self.max_det = max_det

    def run(self):
        """Run inference on all images."""
        try:
            results = {}
            total = len(self.image_paths)

            for i, image_path in enumerate(self.image_paths):
                try:
                    detections = self.model_manager.predict(
                        image_path, self.conf, self.iou, self.max_det
                    )
                    results[image_path] = detections or []
                except Exception as e:
                    logger.error(f"Failed to process {image_path}: {e}")
                    results[image_path] = []

                # Update progress
                progress = int((i + 1) / total * 100)
                self.progress.emit(progress)

            self.finished.emit(results)

        except Exception as e:
            logger.error(f"Inference error: {e}")
            self.error.emit(str(e))


class AutoLabelingPanel(QWidget):
    """Panel for automatic annotation using YOLO models."""

    # Signals
    labeling_started = pyqtSignal()
    labeling_finished = pyqtSignal(dict)  # Results
    labeling_error = pyqtSignal(str)

    def __init__(self, model_manager: ModelManager, parent=None):
        super().__init__(parent)
        self.model_manager = model_manager
        self.inference_thread: Optional[QThread] = None
        self.inference_worker: Optional[InferenceWorker] = None
        self._current_image_provider = None
        self.init_ui()

    def set_current_image_provider(self, provider):
        """Set a callable that returns the current image path.

        This avoids relying on a specific parent widget shape when the panel is
        embedded inside dialogs or workbench pages.
        """
        self._current_image_provider = provider

    def init_ui(self):
        """Initialize UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # Model selection
        model_group = QGroupBox("模型选择")
        model_layout = QVBoxLayout()

        model_label = QLabel("选择模型:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(self.model_manager.list_available_models())
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)

        self.load_model_btn = QPushButton("加载模型")
        self.load_model_btn.clicked.connect(self.load_model)
        model_layout.addWidget(self.load_model_btn)

        self.model_status_label = QLabel("状态: 未加载")
        model_layout.addWidget(self.model_status_label)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Inference parameters
        param_group = QGroupBox("推理参数")
        param_layout = QVBoxLayout()

        # Confidence threshold
        conf_layout = QHBoxLayout()
        conf_label = QLabel("置信度阈值:")
        self.conf_spinbox = QDoubleSpinBox()
        self.conf_spinbox.setRange(0.0, 1.0)
        self.conf_spinbox.setValue(0.25)
        self.conf_spinbox.setSingleStep(0.05)
        conf_layout.addWidget(conf_label)
        conf_layout.addWidget(self.conf_spinbox)
        param_layout.addLayout(conf_layout)

        # IOU threshold
        iou_layout = QHBoxLayout()
        iou_label = QLabel("IOU 阈值:")
        self.iou_spinbox = QDoubleSpinBox()
        self.iou_spinbox.setRange(0.0, 1.0)
        self.iou_spinbox.setValue(0.7)
        self.iou_spinbox.setSingleStep(0.05)
        iou_layout.addWidget(iou_label)
        iou_layout.addWidget(self.iou_spinbox)
        param_layout.addLayout(iou_layout)

        # Max detections
        max_det_layout = QHBoxLayout()
        max_det_label = QLabel("最大检测数:")
        self.max_det_spinbox = QSpinBox()
        self.max_det_spinbox.setRange(1, 1000)
        self.max_det_spinbox.setValue(300)
        max_det_layout.addWidget(max_det_label)
        max_det_layout.addWidget(self.max_det_spinbox)
        param_layout.addLayout(max_det_layout)

        param_group.setLayout(param_layout)
        layout.addWidget(param_group)

        # Action buttons
        action_group = QGroupBox("操作")
        action_layout = QVBoxLayout()

        self.label_current_btn = QPushButton("标注当前图片")
        self.label_current_btn.clicked.connect(self.label_current_image)
        action_layout.addWidget(self.label_current_btn)

        self.label_folder_btn = QPushButton("标注文件夹")
        self.label_folder_btn.clicked.connect(self.label_folder)
        action_layout.addWidget(self.label_folder_btn)

        action_group.setLayout(action_layout)
        layout.addWidget(action_group)

        # Progress
        progress_group = QGroupBox("进度")
        progress_layout = QVBoxLayout()

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        self.progress_label = QLabel("就绪")
        progress_layout.addWidget(self.progress_label)

        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)

        # Results
        results_group = QGroupBox("结果")
        results_layout = QVBoxLayout()

        self.results_list = QListWidget()
        results_layout.addWidget(self.results_list)

        results_group.setLayout(results_layout)
        layout.addWidget(results_group)

        layout.addStretch()

    def load_model(self):
        """Load selected model."""
        model_name = self.model_combo.currentText()
        if not model_name:
            QMessageBox.warning(self, "错误", "请选择一个模型")
            return

        self.progress_label.setText(f"加载模型: {model_name}...")
        self.load_model_btn.setEnabled(False)

        if self.model_manager.load_model(model_name):
            self.model_status_label.setText(f"状态: 已加载 ({model_name})")
            self.progress_label.setText("就绪")
            logger.info(f"Model loaded: {model_name}")
        else:
            self.model_status_label.setText("状态: 加载失败")
            self.progress_label.setText("加载失败")
            QMessageBox.critical(self, "错误", f"无法加载模型: {model_name}")

        self.load_model_btn.setEnabled(True)

    def label_current_image(self):
        """Label the current image."""
        if not self.model_manager.is_model_loaded():
            QMessageBox.warning(self, "错误", "请先加载模型")
            return

        image_path = None
        if self._current_image_provider:
            image_path = self._current_image_provider()
        else:
            parent = self.parent()
            if hasattr(parent, 'current_image_path'):
                image_path = parent.current_image_path

        if not image_path:
            QMessageBox.warning(self, "错误", "没有打开图片")
            return

        self.run_inference([image_path])

    def label_folder(self):
        """Label all images in a folder."""
        if not self.model_manager.is_model_loaded():
            QMessageBox.warning(self, "错误", "请先加载模型")
            return

        folder = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if not folder:
            return

        # Get all images in folder
        import os
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        image_paths = [
            os.path.join(folder, f) for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in extensions
        ]

        if not image_paths:
            QMessageBox.warning(self, "错误", "文件夹中没有图片")
            return

        self.run_inference(image_paths)

    def run_inference(self, image_paths: List[str]):
        """Run inference on images."""
        if self.inference_thread is not None and self.inference_thread.isRunning():
            QMessageBox.warning(self, "错误", "推理正在进行中")
            return

        self.labeling_started.emit()
        self.progress_bar.setValue(0)
        self.progress_label.setText(f"处理中: 0/{len(image_paths)}")
        self.results_list.clear()

        # Create worker
        self.inference_worker = InferenceWorker(
            self.model_manager,
            image_paths,
            self.conf_spinbox.value(),
            self.iou_spinbox.value(),
            self.max_det_spinbox.value(),
        )

        # Create thread
        self.inference_thread = QThread()
        self.inference_worker.moveToThread(self.inference_thread)

        # Connect signals
        self.inference_thread.started.connect(self.inference_worker.run)
        self.inference_worker.progress.connect(self.on_progress)
        self.inference_worker.finished.connect(self.on_finished)
        self.inference_worker.error.connect(self.on_error)
        self.inference_worker.finished.connect(self.inference_thread.quit)
        self.inference_worker.error.connect(self.inference_thread.quit)

        # Start thread
        self.inference_thread.start()

    def on_progress(self, progress: int):
        """Handle progress update."""
        self.progress_bar.setValue(progress)
        self.progress_label.setText(f"处理中: {progress}%")

    def on_finished(self, results: dict):
        """Handle inference finished."""
        self.progress_bar.setValue(100)
        self.progress_label.setText("完成")

        # Display results
        self.results_list.clear()
        for image_path, detections in results.items():
            item_text = f"{image_path}: {len(detections)} 个检测"
            self.results_list.addItem(item_text)

        self.labeling_finished.emit(results)
        logger.info(f"Inference finished: {len(results)} images processed")

    def on_error(self, error_msg: str):
        """Handle inference error."""
        self.progress_label.setText("错误")
        QMessageBox.critical(self, "错误", f"推理失败: {error_msg}")
        self.labeling_error.emit(error_msg)
        logger.error(f"Inference error: {error_msg}")

    def is_model_loaded(self) -> bool:
        """Check if a model is loaded."""
        return self.model_manager.is_model_loaded()
