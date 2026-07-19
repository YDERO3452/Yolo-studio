"""Workflow optimization panel for batch processing and quality checks."""

from pathlib import Path

from loguru import logger
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.class_manager import ClassManager
from core.model_manager import ModelManager
from core.workflow_optimizer import AnnotationValidator, DataQualityChecker, PresetManager


class _BatchLabelWorker(QThread):
    """Worker thread for batch auto-labeling using ModelManager + BatchProcessor."""
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(int, int)
    error_msg = pyqtSignal(str)

    def __init__(self, model_name, input_dir, output_dir, class_names,
                 conf=0.25, iou=0.7, device="", parent=None):
        super().__init__(parent)
        self._model_name = model_name
        self._input_dir = input_dir
        self._output_dir = output_dir
        self._class_names = class_names
        self._conf = conf
        self._iou = iou
        self._device = device

    def run(self):
        from core.batch_processor import BatchProcessingConfig, BatchProcessor

        model_mgr = ModelManager()
        processor = BatchProcessor(model_mgr, self._class_names)

        config = BatchProcessingConfig(
            model_name=self._model_name,
            input_dir=self._input_dir,
            output_dir=self._output_dir,
            conf_threshold=self._conf,
            iou_threshold=self._iou,
            device=self._device,
        )

        def on_progress(cur, total):
            self.progress.emit(cur, total)

        try:
            results = processor.process_directory(config, progress_callback=on_progress)
            ok = sum(1 for r in results if r.success)
            ng = sum(1 for r in results if not r.success)
            self.finished.emit(ok, ng)
        except Exception as exc:
            self.error_msg.emit(str(exc))


class WorkflowOptimizationPanel(QWidget):
    """Panel for workflow optimization."""

    batch_processing_started = pyqtSignal()
    batch_processing_finished = pyqtSignal(dict)
    validation_completed = pyqtSignal(dict)
    quality_check_completed = pyqtSignal(dict)

    def __init__(self, class_manager: ClassManager, parent=None):
        super().__init__(parent)
        self.class_manager = class_manager
        self.validator = AnnotationValidator()
        self.quality_checker = DataQualityChecker()
        self.preset_manager = PresetManager()

        self.init_ui()

    def init_ui(self):
        """Initialize UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Create tabs
        tabs = QTabWidget()

        # Batch Processing Tab
        batch_tab = self.create_batch_tab()
        tabs.addTab(batch_tab, "批量处理")

        # Validation Tab
        validation_tab = self.create_validation_tab()
        tabs.addTab(validation_tab, "标注检查")

        # Quality Check Tab
        quality_tab = self.create_quality_tab()
        tabs.addTab(quality_tab, "质量检查")

        # Presets Tab
        presets_tab = self.create_presets_tab()
        tabs.addTab(presets_tab, "预设管理")

        layout.addWidget(tabs)

    @staticmethod
    def _create_tab_layout(widget: QWidget) -> QVBoxLayout:
        """Create the shared, compact layout used by each tab."""
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        return layout

    @staticmethod
    def _add_section_header(layout: QVBoxLayout, title: str, description: str):
        title_label = QLabel(title)
        title_label.setObjectName("SectionTitle")
        layout.addWidget(title_label)

        description_label = QLabel(description)
        description_label.setObjectName("MutedText")
        description_label.setWordWrap(True)
        layout.addWidget(description_label)

    def create_batch_tab(self) -> QWidget:
        """Create batch processing tab."""
        widget = QWidget()
        layout = self._create_tab_layout(widget)

        self._add_section_header(
            layout,
            "批量导出",
            "在常用标注格式之间转换整个目录。",
        )

        export_btn_layout = QHBoxLayout()
        export_btn_layout.setSpacing(8)
        self.export_input_btn = QPushButton("选择输入目录")
        self.export_input_btn.clicked.connect(self.select_export_input)
        export_btn_layout.addWidget(self.export_input_btn)

        self.export_output_btn = QPushButton("选择输出目录")
        self.export_output_btn.clicked.connect(self.select_export_output)
        export_btn_layout.addWidget(self.export_output_btn)
        export_btn_layout.addStretch(1)
        layout.addLayout(export_btn_layout)

        format_layout = QHBoxLayout()
        format_layout.setSpacing(8)
        format_layout.addWidget(QLabel("输入格式"))
        self.export_input_format_combo = QComboBox()
        self.export_input_format_combo.addItems(["YOLO", "VOC", "COCO", "DOTA"])
        self.export_input_format_combo.setMinimumWidth(96)
        format_layout.addWidget(self.export_input_format_combo)

        format_layout.addSpacing(8)
        format_layout.addWidget(QLabel("输出格式"))
        self.export_format_combo = QComboBox()
        self.export_format_combo.addItems(["YOLO", "VOC", "COCO", "DOTA"])
        self.export_format_combo.setMinimumWidth(96)
        format_layout.addWidget(self.export_format_combo)
        format_layout.addStretch(1)

        self.export_btn = QPushButton("开始导出")
        self.export_btn.setObjectName("PrimaryButton")
        self.export_btn.clicked.connect(self.start_batch_export)
        format_layout.addWidget(self.export_btn)
        layout.addLayout(format_layout)

        layout.addSpacing(8)
        self._add_section_header(
            layout,
            "批量自动标注",
            "使用所选模型为图片目录生成 YOLO 标注文件。",
        )

        label_btn_layout = QHBoxLayout()
        label_btn_layout.setSpacing(8)
        self.label_input_btn = QPushButton("选择图片目录")
        self.label_input_btn.clicked.connect(self.select_label_input)
        label_btn_layout.addWidget(self.label_input_btn)

        self.label_output_btn = QPushButton("选择输出目录")
        self.label_output_btn.clicked.connect(self.select_label_output)
        label_btn_layout.addWidget(self.label_output_btn)
        label_btn_layout.addStretch(1)
        layout.addLayout(label_btn_layout)

        model_row = QHBoxLayout()
        model_row.setSpacing(8)
        model_row.addWidget(QLabel("模型"))
        self.label_model_combo = QComboBox()
        self.label_model_combo.setEditable(True)
        self.label_model_combo.setMinimumWidth(150)
        self._populate_label_models()
        model_row.addWidget(self.label_model_combo, 1)
        model_row.addWidget(QLabel("置信度"))
        self.label_conf_spin = QDoubleSpinBox()
        self.label_conf_spin.setRange(0.01, 1.0)
        self.label_conf_spin.setValue(0.25)
        self.label_conf_spin.setSingleStep(0.05)
        model_row.addWidget(self.label_conf_spin)
        model_row.addWidget(QLabel("IoU"))
        self.label_iou_spin = QDoubleSpinBox()
        self.label_iou_spin.setRange(0.1, 1.0)
        self.label_iou_spin.setValue(0.7)
        self.label_iou_spin.setSingleStep(0.05)
        model_row.addWidget(self.label_iou_spin)
        model_row.addSpacing(8)

        self.label_btn = QPushButton("开始标注")
        self.label_btn.setObjectName("PrimaryButton")
        self.label_btn.clicked.connect(self.start_batch_label)
        model_row.addWidget(self.label_btn)
        layout.addLayout(model_row)

        layout.addSpacing(8)
        progress_header = QHBoxLayout()
        progress_title = QLabel("任务进度")
        progress_title.setObjectName("SectionTitle")
        progress_header.addWidget(progress_title)
        progress_header.addStretch(1)

        self.batch_status_label = QLabel("就绪")
        self.batch_status_label.setObjectName("MutedText")
        progress_header.addWidget(self.batch_status_label)
        layout.addLayout(progress_header)

        self.batch_progress_bar = QProgressBar()
        layout.addWidget(self.batch_progress_bar)

        layout.addStretch(1)
        return widget

    def create_validation_tab(self) -> QWidget:
        """Create validation tab."""
        widget = QWidget()
        layout = self._create_tab_layout(widget)

        self._add_section_header(
            layout,
            "标注验证",
            "检查图片缺少标注、无效标注和常见格式问题。",
        )

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)
        self.validate_input_btn = QPushButton("选择目录")
        self.validate_input_btn.clicked.connect(self.select_validate_input)
        btn_layout.addWidget(self.validate_input_btn)
        btn_layout.addStretch(1)

        self.validate_btn = QPushButton("开始验证")
        self.validate_btn.setObjectName("PrimaryButton")
        self.validate_btn.clicked.connect(self.start_validation)
        btn_layout.addWidget(self.validate_btn)
        layout.addLayout(btn_layout)

        layout.addSpacing(8)
        self._add_section_header(
            layout,
            "验证结果",
            "问题文件和统计摘要会显示在这里。",
        )

        self.validation_results = QTextEdit()
        self.validation_results.setReadOnly(True)
        self.validation_results.setPlaceholderText("选择数据目录并开始验证")
        layout.addWidget(self.validation_results, 1)

        return widget

    def create_quality_tab(self) -> QWidget:
        """Create quality check tab."""
        widget = QWidget()
        layout = self._create_tab_layout(widget)

        self._add_section_header(
            layout,
            "数据质量检查",
            "汇总覆盖率、类别分布和标注尺寸等质量指标。",
        )

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)
        self.quality_input_btn = QPushButton("选择目录")
        self.quality_input_btn.clicked.connect(self.select_quality_input)
        btn_layout.addWidget(self.quality_input_btn)
        btn_layout.addStretch(1)

        self.quality_check_btn = QPushButton("开始检查")
        self.quality_check_btn.setObjectName("PrimaryButton")
        self.quality_check_btn.clicked.connect(self.start_quality_check)
        btn_layout.addWidget(self.quality_check_btn)
        layout.addLayout(btn_layout)

        layout.addSpacing(8)
        self._add_section_header(
            layout,
            "检查结果",
            "完成检查后会显示数据规模、缺失标注和类别明细。",
        )

        self.quality_results = QTextEdit()
        self.quality_results.setReadOnly(True)
        self.quality_results.setPlaceholderText("选择数据目录并开始质量检查")
        layout.addWidget(self.quality_results, 1)

        return widget

    def create_presets_tab(self) -> QWidget:
        """Create presets tab."""
        widget = QWidget()
        layout = self._create_tab_layout(widget)

        self._add_section_header(
            layout,
            "预设列表",
            "保存当前类别配置，便于在后续项目中重复使用。",
        )

        self.presets_empty_label = QLabel("暂无预设\n新建预设后会显示在这里")
        self.presets_empty_label.setObjectName("MutedText")
        self.presets_empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.presets_empty_label.setWordWrap(True)

        self.presets_list = QListWidget()
        layout.addWidget(self.presets_empty_label, 1)
        layout.addWidget(self.presets_list, 1)
        self.refresh_presets_list()

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)

        self.delete_preset_btn = QPushButton("删除预设")
        self.delete_preset_btn.setObjectName("DangerButton")
        self.delete_preset_btn.clicked.connect(self.delete_preset)
        btn_layout.addWidget(self.delete_preset_btn)
        btn_layout.addStretch(1)

        self.new_preset_btn = QPushButton("新建预设")
        self.new_preset_btn.setObjectName("PrimaryButton")
        self.new_preset_btn.clicked.connect(self.create_new_preset)
        btn_layout.addWidget(self.new_preset_btn)
        layout.addLayout(btn_layout)

        return widget

    def select_export_input(self):
        """Select export input directory."""
        dir_path = QFileDialog.getExistingDirectory(self, "选择输入目录")
        if dir_path:
            self.export_input_dir = dir_path
            self.export_input_btn.setText(f"输入: {Path(dir_path).name}")

    def select_export_output(self):
        """Select export output directory."""
        dir_path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if dir_path:
            self.export_output_dir = dir_path
            self.export_output_btn.setText(f"输出: {Path(dir_path).name}")

    def start_batch_export(self):
        """Start batch export."""
        if not hasattr(self, 'export_input_dir') or not hasattr(self, 'export_output_dir'):
            QMessageBox.warning(self, "错误", "请选择输入和输出目录")
            return

        self.batch_status_label.setText("导出中...")
        self.batch_progress_bar.setValue(0)
        self.export_btn.setEnabled(False)

        try:
            from core.format_converter import FormatConverter
            converter = FormatConverter(self.class_manager.get_all_classes())
            input_fmt = self.export_input_format_combo.currentText().lower()
            output_fmt = self.export_format_combo.currentText().lower()

            if input_fmt == output_fmt:
                QMessageBox.warning(self, "提示", "输入格式和输出格式相同，无需转换")
                self.export_btn.setEnabled(True)
                return

            total_files = [0]

            def on_progress(cur, total):
                total_files[0] = total
                if total > 0:
                    self.batch_progress_bar.setValue(int(cur / total * 100))

            conversion_results = converter.convert_folder(
                self.export_input_dir,
                self.export_output_dir,
                input_fmt,
                output_fmt,
                image_dir=None,
                progress_callback=on_progress,
            )

            success_count = sum(1 for r in conversion_results if r.success)
            fail_count = sum(1 for r in conversion_results if not r.success)

            self.batch_progress_bar.setValue(100)
            self.batch_status_label.setText(
                f"导出完成: 成功 {success_count}, 失败 {fail_count}"
            )
            self.export_btn.setEnabled(True)
            self.batch_processing_finished.emit({"success": True, "results": conversion_results})

            if fail_count == 0:
                QMessageBox.information(self, "完成", f"导出成功，共处理 {success_count} 个文件")
            else:
                errors = [r.error_message for r in conversion_results if r.error_message]
                QMessageBox.warning(
                    self, "部分失败",
                    f"成功 {success_count} 个，失败 {fail_count} 个\n\n"
                    f"错误信息:\n" + "\n".join(errors[:5])
                )
        except Exception as e:
            self.batch_progress_bar.setValue(0)
            self.batch_status_label.setText(f"导出失败: {e}")
            self.export_btn.setEnabled(True)

    def select_label_input(self):
        """Select label input directory."""
        dir_path = QFileDialog.getExistingDirectory(self, "选择图片目录")
        if dir_path:
            self.label_input_dir = dir_path
            self.label_input_btn.setText(f"输入: {Path(dir_path).name}")

    def select_label_output(self):
        """Select label output directory."""
        dir_path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if dir_path:
            self.label_output_dir = dir_path
            self.label_output_btn.setText(f"输出: {Path(dir_path).name}")

    def start_batch_label(self):
        """Start batch auto-labeling with YOLO model inference."""
        if not hasattr(self, 'label_input_dir') or not hasattr(self, 'label_output_dir'):
            QMessageBox.warning(self, "错误", "请选择输入和输出目录")
            return

        model_name = self.label_model_combo.currentText().strip()
        if not model_name:
            QMessageBox.warning(self, "错误", "请选择或输入模型名称")
            return

        self.batch_status_label.setText("标注中...")
        self.batch_progress_bar.setValue(0)
        self.label_btn.setEnabled(False)

        self._label_worker = _BatchLabelWorker(
            model_name=model_name,
            input_dir=self.label_input_dir,
            output_dir=self.label_output_dir,
            class_names=self.class_manager.get_all_classes(),
            conf=self.label_conf_spin.value(),
            iou=self.label_iou_spin.value(),
        )
        self._label_worker.progress.connect(self._on_label_progress)
        self._label_worker.finished.connect(self._on_label_finished)
        self._label_worker.error_msg.connect(self._on_label_error)
        self._label_worker.start()

    def _on_label_progress(self, cur, total):
        if total > 0:
            self.batch_progress_bar.setValue(int(cur / total * 100))
            self.batch_status_label.setText(f"标注中... {cur}/{total}")

    def _on_label_finished(self, ok, ng):
        self.batch_progress_bar.setValue(100)
        self.batch_status_label.setText(f"完成: 成功 {ok}, 失败 {ng}")
        self.label_btn.setEnabled(True)
        QMessageBox.information(
            self, "批量标注完成",
            f"成功: {ok} 张\n失败: {ng} 张\n\n标注文件已保存到输出目录。"
        )

    def _on_label_error(self, msg):
        self.batch_progress_bar.setValue(0)
        self.batch_status_label.setText(f"错误: {msg}")
        self.label_btn.setEnabled(True)
        QMessageBox.critical(self, "标注失败", msg)

    def _populate_label_models(self):
        """Populate model combo with available models."""
        try:
            mgr = ModelManager()
            models = mgr.list_available_models()
            local = [Path(m).name for m in mgr.list_local_models()]
            for m in local:
                if m not in models:
                    models.append(m)
            if models:
                self.label_model_combo.addItems(models)
                self.label_model_combo.setCurrentText("yolov8n.pt")
            else:
                self.label_model_combo.addItem("yolov8n.pt")
        except Exception:
            logger.warning("Failed to enumerate YOLO models, falling back to default")
            self.label_model_combo.addItem("yolov8n.pt")

    def select_validate_input(self):
        """Select validation input directory."""
        dir_path = QFileDialog.getExistingDirectory(self, "选择目录")
        if dir_path:
            self.validate_input_dir = dir_path
            self.validate_input_btn.setText(f"目录: {Path(dir_path).name}")

    def start_validation(self):
        """Start validation."""
        if not hasattr(self, 'validate_input_dir'):
            QMessageBox.warning(self, "错误", "请选择目录")
            return

        self.validate_btn.setEnabled(False)

        from gui.annotation_io import labels_dir_for_image_dir

        image_dir = self.validate_input_dir
        labels_dir = labels_dir_for_image_dir(image_dir)
        root = Path(image_dir)
        if (root / "images").is_dir():
            image_dir = str(root / "images")
            labels_dir = str(root / "labels") if (root / "labels").is_dir() else labels_dir_for_image_dir(image_dir)

        # Run validation
        result = self.validator.validate_folder(
            image_dir,
            labels_dir,
            self.class_manager.get_all_classes(),
        )

        # Display results
        results_text = "验证结果:\n"
        results_text += f"总图片数: {result['total_images']}\n"
        results_text += f"已标注: {result['annotated_images']}\n"
        results_text += f"缺少标注: {len(result['missing_annotations'])}\n"
        results_text += f"无效标注: {len(result['invalid_annotations'])}\n"
        results_text += f"警告: {len(result['warnings'])}\n\n"

        if result['missing_annotations']:
            results_text += "缺少标注的文件:\n"
            for f in result['missing_annotations'][:10]:
                results_text += f"  - {f}\n"

        self.validation_results.setText(results_text)
        self.validate_btn.setEnabled(True)
        self.validation_completed.emit(result)

    def select_quality_input(self):
        """Select quality check input directory."""
        dir_path = QFileDialog.getExistingDirectory(self, "选择目录")
        if dir_path:
            self.quality_input_dir = dir_path
            self.quality_input_btn.setText(f"目录: {Path(dir_path).name}")

    def start_quality_check(self):
        """Start quality check."""
        if not hasattr(self, 'quality_input_dir'):
            QMessageBox.warning(self, "错误", "请选择目录")
            return

        self.quality_check_btn.setEnabled(False)

        from gui.annotation_io import labels_dir_for_image_dir

        image_dir = self.quality_input_dir
        labels_dir = labels_dir_for_image_dir(image_dir)
        root = Path(image_dir)
        if (root / "images").is_dir():
            image_dir = str(root / "images")
            labels_dir = str(root / "labels") if (root / "labels").is_dir() else labels_dir_for_image_dir(image_dir)

        # Run quality check
        metrics = self.quality_checker.check_quality(
            image_dir,
            labels_dir,
            self.class_manager.get_all_classes(),
        )

        # Display results
        results_text = "质量检查结果:\n"
        results_text += f"总图片数: {metrics.total_images}\n"
        results_text += f"总标注数: {metrics.total_annotations}\n"
        results_text += f"平均每张图片标注数: {metrics.avg_annotations_per_image:.2f}\n"
        results_text += f"缺少标注: {len(metrics.missing_annotations)}\n\n"

        results_text += "类别分布:\n"
        for class_name, count in sorted(metrics.class_distribution.items(), key=lambda x: x[1], reverse=True):
            results_text += f"  {class_name}: {count}\n"

        if metrics.annotation_size_stats:
            results_text += "\n标注大小统计:\n"
            results_text += f"  最小: {metrics.annotation_size_stats['min']}\n"
            results_text += f"  最大: {metrics.annotation_size_stats['max']}\n"
            results_text += f"  平均: {metrics.annotation_size_stats['mean']:.2f}\n"

        self.quality_results.setText(results_text)
        self.quality_check_btn.setEnabled(True)
        self.quality_check_completed.emit({"metrics": metrics})

    def refresh_presets_list(self):
        """Refresh presets list."""
        self.presets_list.clear()
        presets = self.preset_manager.list_presets()
        for preset in presets:
            self.presets_list.addItem(preset)
        self.presets_empty_label.setVisible(not presets)
        self.presets_list.setVisible(bool(presets))

    def create_new_preset(self):
        """Create new preset from current class configuration."""
        from PyQt6.QtWidgets import QInputDialog

        name, ok = QInputDialog.getText(self, "新建预设", "请输入预设名称:")
        if not ok or not name.strip():
            return

        name = name.strip()

        # Check if preset already exists
        existing = self.preset_manager.list_presets()
        if name in existing:
            replace = QMessageBox.question(
                self, "确认", f"预设 '{name}' 已存在，是否覆盖？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if replace != QMessageBox.StandardButton.Yes:
                return

        # Collect current configuration as preset data
        classes = self.class_manager.get_all_classes()
        preset_data = {
            "name": name,
            "classes": classes,
            "class_count": len(classes),
        }

        if self.preset_manager.save_preset(name, preset_data):
            self.refresh_presets_list()
            QMessageBox.information(self, "成功", f"预设已保存: {name}\n包含 {len(classes)} 个类别")
        else:
            QMessageBox.warning(self, "错误", "保存预设失败")

    def delete_preset(self):
        """Delete selected preset."""
        current_item = self.presets_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "错误", "请选择一个预设")
            return

        preset_name = current_item.text()
        if self.preset_manager.delete_preset(preset_name):
            self.refresh_presets_list()
            QMessageBox.information(self, "成功", f"预设已删除: {preset_name}")
        else:
            QMessageBox.warning(self, "错误", "删除预设失败")
