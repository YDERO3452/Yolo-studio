"""Advanced features panel for statistics and reporting."""

from pathlib import Path

from loguru import logger
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.advanced_features import (
    AnnotationStatisticsCollector,
    DataAugmentationHelper,
    ModelFineTuningHelper,
    ReportGenerator,
)
from core.class_manager import ClassManager


class _StatsCollectWorker(QThread):
    """Background thread for statistics collection."""

    finished = pyqtSignal(object)

    def __init__(self, collector, image_dir, annotation_dir, class_names):
        super().__init__()
        self.collector = collector
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.class_names = class_names

    def run(self):
        try:
            stats = self.collector.collect_statistics(
                self.image_dir, self.annotation_dir, self.class_names
            )
            self.finished.emit(stats)
        except Exception as e:
            logger.error(f"Statistics collection error: {e}")
            self.finished.emit(None)


class _AugmentWorker(QThread):
    """Background thread for dataset augmentation."""

    finished = pyqtSignal(int, str)  # count, error

    def __init__(self, images_dir, labels_dir, out_images, out_labels, augmentations_per_image=2):
        super().__init__()
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.out_images = out_images
        self.out_labels = out_labels
        self.augmentations_per_image = augmentations_per_image

    def run(self):
        try:
            from core.augmentor import DataAugmentor

            count = DataAugmentor().augment_dataset(
                self.images_dir,
                self.labels_dir,
                self.out_images,
                self.out_labels,
                augmentations_per_image=self.augmentations_per_image,
            )
            self.finished.emit(int(count or 0), "")
        except Exception as exc:
            logger.error(f"Augmentation error: {exc}")
            self.finished.emit(0, str(exc))


class AdvancedFeaturesPanel(QWidget):
    """Panel for advanced features."""

    statistics_collected = pyqtSignal(dict)
    report_generated = pyqtSignal(str)
    apply_training_config = pyqtSignal(dict)  # epochs/batch/lr0/optimizer → TrainingPanel

    def __init__(self, class_manager: ClassManager, parent=None):
        super().__init__(parent)
        self.class_manager = class_manager
        self.collector = AnnotationStatisticsCollector()
        self.report_generator = ReportGenerator()
        self.augmentation_helper = DataAugmentationHelper()
        self.finetuning_helper = ModelFineTuningHelper()
        self.current_statistics = None
        self._last_training_config: dict | None = None
        self._augment_worker = None

        self.init_ui()

    def init_ui(self):
        """Initialize UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Create tabs
        tabs = QTabWidget()

        # Statistics tab
        stats_tab = self.create_statistics_tab()
        tabs.addTab(stats_tab, "统计分析")

        # Reports tab
        reports_tab = self.create_reports_tab()
        tabs.addTab(reports_tab, "报告生成")

        # Augmentation tab
        augmentation_tab = self.create_augmentation_tab()
        tabs.addTab(augmentation_tab, "数据增强")

        # Fine-tuning tab
        finetuning_tab = self.create_finetuning_tab()
        tabs.addTab(finetuning_tab, "模型微调")

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

    def create_statistics_tab(self) -> QWidget:
        """Create statistics tab."""
        widget = QWidget()
        layout = self._create_tab_layout(widget)

        self._add_section_header(
            layout,
            "数据范围",
            "从同一目录读取图片和标注，统计类别分布、覆盖率与标注尺寸。",
        )

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)
        self.stats_input_btn = QPushButton("选择目录")
        self.stats_input_btn.clicked.connect(self.select_statistics_dir)
        btn_layout.addWidget(self.stats_input_btn)
        btn_layout.addStretch(1)

        self.collect_stats_btn = QPushButton("收集统计")
        self.collect_stats_btn.setObjectName("PrimaryButton")
        self.collect_stats_btn.clicked.connect(self.collect_statistics)
        btn_layout.addWidget(self.collect_stats_btn)
        layout.addLayout(btn_layout)

        self.stats_progress = QProgressBar()
        layout.addWidget(self.stats_progress)

        layout.addSpacing(8)
        self._add_section_header(
            layout,
            "统计结果",
            "完成收集后，这里会显示数据集概览和各类别明细。",
        )

        self.stats_results = QTextEdit()
        self.stats_results.setReadOnly(True)
        self.stats_results.setPlaceholderText("选择数据目录并收集统计结果")
        layout.addWidget(self.stats_results, 1)

        return widget

    def create_reports_tab(self) -> QWidget:
        """Create reports tab."""
        widget = QWidget()
        layout = self._create_tab_layout(widget)

        self._add_section_header(
            layout,
            "报告设置",
            "基于统计结果生成可保存的 HTML、JSON 或文本报告。",
        )

        format_btn_layout = QHBoxLayout()
        format_btn_layout.setSpacing(8)
        format_btn_layout.addWidget(QLabel("报告格式"))
        self.report_format_combo = QComboBox()
        self.report_format_combo.addItems(["HTML", "JSON", "Text"])
        self.report_format_combo.setMinimumWidth(120)
        format_btn_layout.addWidget(self.report_format_combo)
        format_btn_layout.addStretch(1)

        self.generate_report_btn = QPushButton("生成报告")
        self.generate_report_btn.setObjectName("PrimaryButton")
        self.generate_report_btn.clicked.connect(self.generate_report)
        format_btn_layout.addWidget(self.generate_report_btn)
        layout.addLayout(format_btn_layout)

        layout.addSpacing(8)
        self._add_section_header(
            layout,
            "报告预览",
            "报告生成后会保存到所选位置。",
        )

        self.report_preview = QTextEdit()
        self.report_preview.setReadOnly(True)
        self.report_preview.setPlaceholderText(
            "请先在“统计分析”中收集统计结果，再生成报告"
        )
        layout.addWidget(self.report_preview, 1)

        return widget

    def create_augmentation_tab(self) -> QWidget:
        """Create augmentation tab."""
        widget = QWidget()
        layout = self._create_tab_layout(widget)

        self._add_section_header(
            layout,
            "数据增强",
            "先看建议，也可以直接对统计目录跑增强。",
        )

        action_layout = QHBoxLayout()
        action_layout.addStretch(1)
        self.show_config_btn = QPushButton("显示配置")
        self.show_config_btn.setObjectName("PrimaryButton")
        self.show_config_btn.clicked.connect(self.show_augmentation_config)
        action_layout.addWidget(self.show_config_btn)
        self.run_augment_btn = QPushButton("执行增强")
        self.run_augment_btn.clicked.connect(self.run_dataset_augmentation)
        action_layout.addWidget(self.run_augment_btn)
        layout.addLayout(action_layout)

        self.augmentation_suggestions = QTextEdit()
        self.augmentation_suggestions.setReadOnly(True)
        self.augmentation_suggestions.setPlaceholderText(
            "请先在“统计分析”中收集数据，再查看增强建议或执行增强"
        )
        layout.addWidget(self.augmentation_suggestions, 1)

        return widget

    def create_finetuning_tab(self) -> QWidget:
        """Create fine-tuning tab."""
        widget = QWidget()
        layout = self._create_tab_layout(widget)

        self._add_section_header(
            layout,
            "训练参数",
            "按数据量估一下 epochs / batch 等，可写回训练页。",
        )

        action_layout = QHBoxLayout()
        action_layout.addStretch(1)
        self.show_training_config_btn = QPushButton("显示推荐配置")
        self.show_training_config_btn.setObjectName("PrimaryButton")
        self.show_training_config_btn.clicked.connect(self.show_training_config)
        action_layout.addWidget(self.show_training_config_btn)
        self.apply_training_btn = QPushButton("应用到训练页")
        self.apply_training_btn.clicked.connect(self.apply_training_config_to_panel)
        action_layout.addWidget(self.apply_training_btn)
        layout.addLayout(action_layout)

        self.training_tips = QTextEdit()
        self.training_tips.setReadOnly(True)
        self.training_tips.setPlaceholderText(
            "请先在“统计分析”中收集数据，再查看推荐训练配置"
        )
        layout.addWidget(self.training_tips, 1)

        return widget

    def select_statistics_dir(self):
        """Select directory for statistics."""
        dir_path = QFileDialog.getExistingDirectory(self, "选择目录")
        if dir_path:
            self.stats_dir = dir_path
            self.stats_input_btn.setText(f"目录: {Path(dir_path).name}")

    def collect_statistics(self):
        """Collect statistics in background thread."""
        if not hasattr(self, 'stats_dir'):
            QMessageBox.warning(self, "错误", "请选择目录")
            return

        self.collect_stats_btn.setEnabled(False)
        self.stats_progress.setRange(0, 0)  # indeterminate

        self._stats_worker = _StatsCollectWorker(
            self.collector,
            self.stats_dir,
            self.stats_dir,
            self.class_manager.get_all_classes(),
        )
        self._stats_worker.finished.connect(self._on_stats_collected)
        self._stats_worker.start()

    def _on_stats_collected(self, stats):
        """Handle statistics collection result."""
        self.stats_progress.setRange(0, 100)
        self.stats_progress.setValue(100)
        self.collect_stats_btn.setEnabled(True)

        if stats is None:
            QMessageBox.critical(self, "错误", "统计收集失败")
            return

        self.current_statistics = stats

        results_text = f"""
统计结果:
{'='*50}
总图片数: {stats.total_images}
总标注数: {stats.total_annotations}
平均每张图片标注数: {stats.avg_annotations_per_image:.2f}
图片覆盖率: {stats.image_coverage*100:.1f}%

类别分布:
{'-'*50}
"""

        for class_name, count in sorted(
            stats.class_distribution.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            total = sum(stats.class_distribution.values())
            percentage = (count / total * 100) if total > 0 else 0
            results_text += f"{class_name:20s} {count:6d} ({percentage:5.1f}%)\n"

        results_text += f"""
标注大小统计:
{'-'*50}
"""

        for key, value in stats.annotation_size_stats.items():
            results_text += f"{key:20s} {value:8.2f}\n"

        self.stats_results.setText(results_text)
        self.statistics_collected.emit({"statistics": stats})

    def generate_report(self):
        """Generate report."""
        if not self.current_statistics:
            QMessageBox.warning(self, "错误", "请先收集统计数据")
            return

        report_format = self.report_format_combo.currentText()
        file_filter = f"{report_format} 文件 (*.{report_format.lower()})"

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            f"保存 {report_format} 报告",
            "",
            file_filter
        )

        if not file_path:
            return

        success = False
        if report_format == "HTML":
            success = self.report_generator.generate_html_report(
                self.current_statistics,
                file_path
            )
        elif report_format == "JSON":
            success = self.report_generator.generate_json_report(
                self.current_statistics,
                file_path
            )
        elif report_format == "Text":
            success = self.report_generator.generate_text_report(
                self.current_statistics,
                file_path
            )

        if success:
            QMessageBox.information(self, "成功", f"报告已生成: {file_path}")
            self.report_generated.emit(file_path)
        else:
            QMessageBox.critical(self, "错误", "报告生成失败")

    def show_augmentation_config(self):
        """Show augmentation configuration."""
        if not self.current_statistics:
            QMessageBox.warning(self, "错误", "请先收集统计数据")
            return

        suggestions = self.augmentation_helper.suggest_augmentation(
            self.current_statistics
        )
        config = self.augmentation_helper.get_augmentation_config()

        suggestions_text = "增强建议:\n"
        suggestions_text += "="*50 + "\n"

        if suggestions["strategies"]:
            for strategy, reason in zip(suggestions["strategies"], suggestions["reasons"]):
                suggestions_text += f"- {strategy}: {reason}\n"
        else:
            suggestions_text += "数据集平衡，无需特殊增强\n"

        suggestions_text += "\n推荐配置:\n"
        suggestions_text += "-"*50 + "\n"
        import json
        suggestions_text += json.dumps(config, indent=2, ensure_ascii=False)

        self.augmentation_suggestions.setText(suggestions_text)

    def run_dataset_augmentation(self):
        """Run DataAugmentor on the statistics directory."""
        image_dir = getattr(self, "stats_dir", None)
        if not image_dir:
            QMessageBox.warning(self, "错误", "请先在统计分析中选择数据目录")
            return
        if self._augment_worker and self._augment_worker.isRunning():
            QMessageBox.warning(self, "提示", "增强任务正在进行")
            return

        from gui.annotation_io import labels_dir_for_image_dir

        labels_dir = labels_dir_for_image_dir(image_dir)
        out_root = QFileDialog.getExistingDirectory(
            self, "选择增强输出目录（将创建 images/ 与 labels/）",
            str(Path(image_dir).parent),
        )
        if not out_root:
            return
        out_images = str(Path(out_root) / "images")
        out_labels = str(Path(out_root) / "labels")

        self.run_augment_btn.setEnabled(False)
        self.augmentation_suggestions.append("\n正在执行数据增强…")

        self._augment_worker = _AugmentWorker(
            image_dir, labels_dir, out_images, out_labels, augmentations_per_image=2,
        )
        self._augment_worker.finished.connect(self._on_augment_finished)
        self._augment_worker.start()

    def _on_augment_finished(self, count: int, error: str) -> None:
        self.run_augment_btn.setEnabled(True)
        if error:
            self.augmentation_suggestions.append(f"\n增强失败: {error}")
            QMessageBox.critical(self, "增强失败", error)
            return
        self.augmentation_suggestions.append(f"\n增强完成: 新生成 {count} 张图片")
        QMessageBox.information(self, "完成", f"已生成 {count} 张增强图片")

    def show_training_config(self):
        """Show training configuration."""
        if not self.current_statistics:
            QMessageBox.warning(self, "错误", "请先收集统计数据")
            return

        config = self.finetuning_helper.suggest_training_config(
            self.current_statistics
        )
        tips = self.finetuning_helper.get_training_tips()
        self._last_training_config = dict(config)

        config_text = "推荐训练配置:\n"
        config_text += "="*50 + "\n"
        import json
        config_text += json.dumps(config, indent=2, ensure_ascii=False)

        config_text += "\n\n训练建议:\n"
        config_text += "-"*50 + "\n"
        for i, tip in enumerate(tips, 1):
            config_text += f"{i}. {tip}\n"

        self.training_tips.setText(config_text)

    def apply_training_config_to_panel(self):
        """Push the last suggested training config into TrainingPanel."""
        if not self._last_training_config:
            if not self.current_statistics:
                QMessageBox.warning(self, "错误", "请先收集统计数据并显示推荐配置")
                return
            self.show_training_config()
        if not self._last_training_config:
            return
        self.apply_training_config.emit(dict(self._last_training_config))
        QMessageBox.information(self, "好了", "参数已写到训练页")
