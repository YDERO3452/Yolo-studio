"""Advanced features panel for statistics and reporting."""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QPushButton,
    QFileDialog, QMessageBox, QTabWidget, QTextEdit, QComboBox,
    QSpinBox, QDoubleSpinBox, QCheckBox, QTableWidget, QTableWidgetItem,
    QProgressBar
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QObject
from pathlib import Path
from typing import Optional
from loguru import logger

from core.advanced_features import (
    AnnotationStatisticsCollector, ReportGenerator,
    DataAugmentationHelper, ModelFineTuningHelper
)
from core.class_manager import ClassManager


class StatisticsWorker(QObject):
    """Worker thread for statistics collection."""

    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, collector, image_dir, annotation_dir, class_names):
        super().__init__()
        self.collector = collector
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.class_names = class_names

    def run(self):
        """Run statistics collection."""
        try:
            stats = self.collector.collect_statistics(
                self.image_dir,
                self.annotation_dir,
                self.class_names
            )
            self.progress.emit(100)
            self.finished.emit({"statistics": stats})
        except Exception as e:
            logger.error(f"Statistics collection error: {e}")
            self.error.emit(str(e))


class AdvancedFeaturesPanel(QWidget):
    """Panel for advanced features."""

    statistics_collected = pyqtSignal(dict)
    report_generated = pyqtSignal(str)

    def __init__(self, class_manager: ClassManager, parent=None):
        super().__init__(parent)
        self.class_manager = class_manager
        self.collector = AnnotationStatisticsCollector()
        self.report_generator = ReportGenerator()
        self.augmentation_helper = DataAugmentationHelper()
        self.finetuning_helper = ModelFineTuningHelper()
        self.current_statistics = None

        self.init_ui()

    def init_ui(self):
        """Initialize UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

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

    def create_statistics_tab(self) -> QWidget:
        """Create statistics tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Directory selection
        dir_group = QGroupBox("目录选择")
        dir_layout = QVBoxLayout()

        btn_layout = QHBoxLayout()
        self.stats_input_btn = QPushButton("选择目录")
        self.stats_input_btn.clicked.connect(self.select_statistics_dir)
        btn_layout.addWidget(self.stats_input_btn)

        self.collect_stats_btn = QPushButton("收集统计")
        self.collect_stats_btn.clicked.connect(self.collect_statistics)
        btn_layout.addWidget(self.collect_stats_btn)

        dir_layout.addLayout(btn_layout)
        dir_group.setLayout(dir_layout)
        layout.addWidget(dir_group)

        # Progress
        self.stats_progress = QProgressBar()
        layout.addWidget(self.stats_progress)

        # Results
        results_group = QGroupBox("统计结果")
        results_layout = QVBoxLayout()

        self.stats_results = QTextEdit()
        self.stats_results.setReadOnly(True)
        results_layout.addWidget(self.stats_results)

        results_group.setLayout(results_layout)
        layout.addWidget(results_group)

        layout.addStretch()
        return widget

    def create_reports_tab(self) -> QWidget:
        """Create reports tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Report format
        format_group = QGroupBox("报告格式")
        format_layout = QVBoxLayout()

        format_btn_layout = QHBoxLayout()
        format_btn_layout.addWidget(QLabel("格式:"))
        self.report_format_combo = QComboBox()
        self.report_format_combo.addItems(["HTML", "JSON", "Text"])
        format_btn_layout.addWidget(self.report_format_combo)
        format_layout.addLayout(format_btn_layout)

        format_group.setLayout(format_layout)
        layout.addWidget(format_group)

        # Generate report
        report_group = QGroupBox("生成报告")
        report_layout = QVBoxLayout()

        self.generate_report_btn = QPushButton("生成报告")
        self.generate_report_btn.clicked.connect(self.generate_report)
        report_layout.addWidget(self.generate_report_btn)

        report_group.setLayout(report_layout)
        layout.addWidget(report_group)

        # Report preview
        preview_group = QGroupBox("报告预览")
        preview_layout = QVBoxLayout()

        self.report_preview = QTextEdit()
        self.report_preview.setReadOnly(True)
        preview_layout.addWidget(self.report_preview)

        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)

        layout.addStretch()
        return widget

    def create_augmentation_tab(self) -> QWidget:
        """Create augmentation tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Augmentation suggestions
        suggestions_group = QGroupBox("增强建议")
        suggestions_layout = QVBoxLayout()

        self.augmentation_suggestions = QTextEdit()
        self.augmentation_suggestions.setReadOnly(True)
        suggestions_layout.addWidget(self.augmentation_suggestions)

        suggestions_group.setLayout(suggestions_layout)
        layout.addWidget(suggestions_group)

        # Augmentation config
        config_group = QGroupBox("增强配置")
        config_layout = QVBoxLayout()

        self.show_config_btn = QPushButton("显示配置")
        self.show_config_btn.clicked.connect(self.show_augmentation_config)
        config_layout.addWidget(self.show_config_btn)

        config_group.setLayout(config_layout)
        layout.addWidget(config_group)

        layout.addStretch()
        return widget

    def create_finetuning_tab(self) -> QWidget:
        """Create fine-tuning tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Training config
        config_group = QGroupBox("训练配置")
        config_layout = QVBoxLayout()

        self.show_training_config_btn = QPushButton("显示推荐配置")
        self.show_training_config_btn.clicked.connect(self.show_training_config)
        config_layout.addWidget(self.show_training_config_btn)

        config_group.setLayout(config_layout)
        layout.addWidget(config_group)

        # Training tips
        tips_group = QGroupBox("训练建议")
        tips_layout = QVBoxLayout()

        self.training_tips = QTextEdit()
        self.training_tips.setReadOnly(True)
        tips_layout.addWidget(self.training_tips)

        tips_group.setLayout(tips_layout)
        layout.addWidget(tips_group)

        layout.addStretch()
        return widget

    def select_statistics_dir(self):
        """Select directory for statistics."""
        dir_path = QFileDialog.getExistingDirectory(self, "选择目录")
        if dir_path:
            self.stats_dir = dir_path
            self.stats_input_btn.setText(f"目录: {Path(dir_path).name}")

    def collect_statistics(self):
        """Collect statistics."""
        if not hasattr(self, 'stats_dir'):
            QMessageBox.warning(self, "错误", "请选择目录")
            return

        self.collect_stats_btn.setEnabled(False)
        self.stats_progress.setValue(0)

        # Collect statistics
        self.current_statistics = self.collector.collect_statistics(
            self.stats_dir,
            self.stats_dir,
            self.class_manager.get_all_classes()
        )

        # Display results
        results_text = f"""
统计结果:
{'='*50}
总图片数: {self.current_statistics.total_images}
总标注数: {self.current_statistics.total_annotations}
平均每张图片标注数: {self.current_statistics.avg_annotations_per_image:.2f}
图片覆盖率: {self.current_statistics.image_coverage*100:.1f}%

类别分布:
{'-'*50}
"""

        for class_name, count in sorted(
            self.current_statistics.class_distribution.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            total = sum(self.current_statistics.class_distribution.values())
            percentage = (count / total * 100) if total > 0 else 0
            results_text += f"{class_name:20s} {count:6d} ({percentage:5.1f}%)\n"

        results_text += f"""
标注大小统计:
{'-'*50}
"""

        for key, value in self.current_statistics.annotation_size_stats.items():
            results_text += f"{key:20s} {value:8.2f}\n"

        self.stats_results.setText(results_text)
        self.stats_progress.setValue(100)
        self.collect_stats_btn.setEnabled(True)
        self.statistics_collected.emit({"statistics": self.current_statistics})

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

    def show_training_config(self):
        """Show training configuration."""
        if not self.current_statistics:
            QMessageBox.warning(self, "错误", "请先收集统计数据")
            return

        config = self.finetuning_helper.suggest_training_config(
            self.current_statistics
        )
        tips = self.finetuning_helper.get_training_tips()

        config_text = "推荐训练配置:\n"
        config_text += "="*50 + "\n"
        import json
        config_text += json.dumps(config, indent=2, ensure_ascii=False)

        config_text += "\n\n训练建议:\n"
        config_text += "-"*50 + "\n"
        for i, tip in enumerate(tips, 1):
            config_text += f"{i}. {tip}\n"

        self.training_tips.setText(config_text)
