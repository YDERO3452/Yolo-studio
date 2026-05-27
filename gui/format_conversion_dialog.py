"""Format conversion dialog for batch annotation format conversion."""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QComboBox,
    QPushButton, QFileDialog, QProgressBar, QMessageBox
)
from PyQt6.QtCore import pyqtSignal, QThread, QObject
from pathlib import Path
from typing import Optional
from loguru import logger

from core.format_converter import FormatConverter
from core.class_manager import ClassManager
from gui.theme import build_stylesheet


class ConversionWorker(QObject):
    """Worker thread for format conversion."""

    progress = pyqtSignal(int)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    status = pyqtSignal(str)

    def __init__(self, converter: FormatConverter, input_dir: str, output_dir: str,
                 input_format: str, output_format: str):
        super().__init__()
        self.converter = converter
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.input_format = input_format
        self.output_format = output_format

    def run(self):
        """Run conversion."""
        try:
            self.status.emit(f"正在从 {self.input_format} 转换为 {self.output_format}...")
            self.converter.convert_folder(
                self.input_dir,
                self.output_dir,
                self.input_format,
                self.output_format
            )
            self.progress.emit(100)
            self.status.emit("转换完成")
            self.finished.emit()
        except Exception as e:
            logger.error(f"Conversion error: {e}")
            self.error.emit(str(e))


class FormatConversionDialog(QDialog):
    """Dialog for converting annotation formats."""

    conversion_finished = pyqtSignal(str)  # Output directory

    def __init__(self, class_manager: ClassManager, parent=None):
        super().__init__(parent)
        self.class_manager = class_manager
        self.converter = FormatConverter(class_manager.get_all_classes())
        self.conversion_thread: Optional[QThread] = None
        self.conversion_worker: Optional[ConversionWorker] = None

        self.setWindowTitle("格式转换")
        self.setMinimumWidth(500)
        self.setStyleSheet(build_stylesheet())
        self.init_ui()

    def init_ui(self):
        """Initialize UI."""
        layout = QVBoxLayout(self)

        title = QLabel("格式转换")
        title.setObjectName("BrandTitle")
        layout.addWidget(title)

        desc = QLabel("批量转换 YOLO、VOC、COCO、DOTA 标注格式。")
        desc.setObjectName("MutedText")
        layout.addWidget(desc)

        # Input format
        input_group = QGroupBox("输入格式")
        input_layout = QVBoxLayout()

        input_format_layout = QHBoxLayout()
        input_format_layout.addWidget(QLabel("格式:"))
        self.input_format_combo = QComboBox()
        self.input_format_combo.addItems(["YOLO", "VOC", "COCO", "DOTA"])
        input_format_layout.addWidget(self.input_format_combo)
        input_layout.addLayout(input_format_layout)

        input_dir_layout = QHBoxLayout()
        input_dir_layout.addWidget(QLabel("目录:"))
        self.input_dir_label = QLabel("未选择")
        input_dir_layout.addWidget(self.input_dir_label)
        self.input_dir_btn = QPushButton("浏览")
        self.input_dir_btn.clicked.connect(self.select_input_dir)
        input_dir_layout.addWidget(self.input_dir_btn)
        input_layout.addLayout(input_dir_layout)

        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        # Output format
        output_group = QGroupBox("输出格式")
        output_layout = QVBoxLayout()

        output_format_layout = QHBoxLayout()
        output_format_layout.addWidget(QLabel("格式:"))
        self.output_format_combo = QComboBox()
        self.output_format_combo.addItems(["YOLO", "VOC", "COCO", "DOTA"])
        self.output_format_combo.setCurrentIndex(1)  # Default to VOC
        output_format_layout.addWidget(self.output_format_combo)
        output_layout.addLayout(output_format_layout)

        output_dir_layout = QHBoxLayout()
        output_dir_layout.addWidget(QLabel("目录:"))
        self.output_dir_label = QLabel("未选择")
        output_dir_layout.addWidget(self.output_dir_label)
        self.output_dir_btn = QPushButton("浏览")
        self.output_dir_btn.clicked.connect(self.select_output_dir)
        output_dir_layout.addWidget(self.output_dir_btn)
        output_layout.addLayout(output_dir_layout)

        output_group.setLayout(output_layout)
        layout.addWidget(output_group)

        # Progress
        progress_group = QGroupBox("进度")
        progress_layout = QVBoxLayout()

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("就绪")
        self.status_label.setObjectName("StatusPill")
        progress_layout.addWidget(self.status_label)

        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.convert_btn = QPushButton("开始转换")
        self.convert_btn.setObjectName("PrimaryButton")
        self.convert_btn.clicked.connect(self.start_conversion)
        button_layout.addWidget(self.convert_btn)

        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)

        layout.addLayout(button_layout)

    def select_input_dir(self):
        """Select input directory."""
        dir_path = QFileDialog.getExistingDirectory(self, "选择输入目录")
        if dir_path:
            self.input_dir_label.setText(dir_path)

    def select_output_dir(self):
        """Select output directory."""
        dir_path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if dir_path:
            self.output_dir_label.setText(dir_path)

    def start_conversion(self):
        """Start format conversion."""
        input_dir = self.input_dir_label.text()
        output_dir = self.output_dir_label.text()
        input_format = self.input_format_combo.currentText()
        output_format = self.output_format_combo.currentText()

        if input_dir == "未选择" or output_dir == "未选择":
            QMessageBox.warning(self, "错误", "请选择输入和输出目录")
            return

        if input_format == output_format:
            QMessageBox.warning(self, "错误", "输入和输出格式不能相同")
            return

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Start conversion
        self.progress_bar.setValue(0)
        self.status_label.setText("转换中...")
        self.convert_btn.setEnabled(False)

        self._cleanup_conversion()

        self.conversion_worker = ConversionWorker(
            self.converter,
            input_dir,
            output_dir,
            input_format.lower(),
            output_format.lower()
        )

        self.conversion_thread = QThread()
        self.conversion_worker.moveToThread(self.conversion_thread)

        # Connect signals
        self.conversion_thread.started.connect(self.conversion_worker.run)
        self.conversion_worker.progress.connect(self.progress_bar.setValue)
        self.conversion_worker.status.connect(self.status_label.setText)
        self.conversion_worker.finished.connect(self.on_conversion_finished)
        self.conversion_worker.error.connect(self.on_conversion_error)
        self.conversion_worker.finished.connect(self.conversion_thread.quit)
        self.conversion_worker.error.connect(self.conversion_thread.quit)

        self.conversion_thread.start()

    def _cleanup_conversion(self):
        """Clean up worker and thread from previous conversion."""
        if hasattr(self, "conversion_thread") and self.conversion_thread is not None:
            if self.conversion_thread.isRunning():
                self.conversion_thread.quit()
                self.conversion_thread.wait(3000)
            self.conversion_thread.deleteLater()
            self.conversion_thread = None
        if hasattr(self, "conversion_worker") and self.conversion_worker is not None:
            self.conversion_worker.deleteLater()
            self.conversion_worker = None

    def closeEvent(self, event):
        """Stop running conversion before closing."""
        self._cleanup_conversion()
        super().closeEvent(event)

    def on_conversion_finished(self):
        """Handle conversion finished."""
        self.convert_btn.setEnabled(True)
        output_dir = self.output_dir_label.text()
        QMessageBox.information(self, "成功", f"转换完成！\n输出目录: {output_dir}")
        self.conversion_finished.emit(output_dir)
        logger.info(f"Conversion finished: {output_dir}")
        self._cleanup_conversion()

    def on_conversion_error(self, error_msg: str):
        """Handle conversion error."""
        self.convert_btn.setEnabled(True)
        QMessageBox.critical(self, "错误", f"转换失败: {error_msg}")
        logger.error(f"Conversion error: {error_msg}")
