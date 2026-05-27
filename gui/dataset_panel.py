"""Dataset management panel."""

import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QLineEdit, QTextEdit, QFileDialog,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
    QDoubleSpinBox, QSplitter, QFormLayout, QTabWidget
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from core.dataset import DatasetManager
from core.annotation import AnnotationManager
from gui.annotation_io import labels_dir_for_image_dir


class DatasetPanel(QWidget):
    """Panel for managing YOLO datasets."""

    dataset_loaded = pyqtSignal(str)  # emits data.yaml path

    def __init__(self, config_manager=None, parent=None):
        super().__init__(parent)
        self.config = config_manager
        self.dataset_manager = None
        self.annotation_manager = AnnotationManager()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        workflow_tabs = QTabWidget()
        workflow_tabs.setMinimumWidth(390)

        # Create new dataset
        create_group = QGroupBox("创建数据集")
        create_layout = QFormLayout()
        create_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.dataset_name_edit = QLineEdit()
        self.dataset_name_edit.setPlaceholderText("我的数据集")
        create_layout.addRow("数据集名称:", self.dataset_name_edit)

        self.save_path_edit = QLineEdit("data")
        browse_btn = QPushButton("浏览...")
        browse_btn.clicked.connect(self.browse_save_path)
        save_row = QHBoxLayout()
        save_row.addWidget(self.save_path_edit)
        save_row.addWidget(browse_btn)
        create_layout.addRow("保存路径:", save_row)

        self.classes_edit = QLineEdit()
        self.classes_edit.setPlaceholderText("猫,狗,人")
        create_layout.addRow("类别:", self.classes_edit)

        self.create_btn = QPushButton("创建数据集")
        self.create_btn.setObjectName("PrimaryButton")
        self.create_btn.clicked.connect(self.create_dataset)
        create_layout.addRow("", self.create_btn)

        create_group.setLayout(create_layout)

        # Load existing dataset
        load_group = QGroupBox("加载数据集")
        load_layout = QFormLayout()
        load_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.data_yaml_edit = QLineEdit()
        self.data_yaml_edit.setPlaceholderText("选择 data.yaml 文件...")
        load_browse = QPushButton("浏览...")
        load_browse.clicked.connect(self.browse_data_yaml)
        self.load_btn = QPushButton("加载")
        self.load_btn.setObjectName("PrimaryButton")
        self.load_btn.clicked.connect(self.load_dataset)

        yaml_row = QHBoxLayout()
        yaml_row.addWidget(self.data_yaml_edit)
        yaml_row.addWidget(load_browse)
        load_layout.addRow("data.yaml:", yaml_row)
        load_layout.addRow("", self.load_btn)

        load_group.setLayout(load_layout)

        # Split dataset
        split_group = QGroupBox("拆分数据集")
        split_layout = QFormLayout()
        split_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.src_images_edit = QLineEdit()
        src_img_btn = QPushButton("浏览...")
        src_img_btn.clicked.connect(lambda: self.browse_dir(self.src_images_edit, "选择图片目录"))
        src_img_row = QHBoxLayout()
        src_img_row.addWidget(self.src_images_edit)
        src_img_row.addWidget(src_img_btn)
        split_layout.addRow("源图片目录:", src_img_row)

        self.src_labels_edit = QLineEdit()
        self.src_labels_edit.setPlaceholderText("留空则自动从图片目录推导 (images/ → labels/)")
        src_lbl_btn = QPushButton("浏览...")
        src_lbl_btn.clicked.connect(lambda: self.browse_dir(self.src_labels_edit, "选择标注目录"))
        src_lbl_row = QHBoxLayout()
        src_lbl_row.addWidget(self.src_labels_edit)
        src_lbl_row.addWidget(src_lbl_btn)
        split_layout.addRow("源标注目录:", src_lbl_row)

        self.train_ratio_spin = QDoubleSpinBox()
        self.train_ratio_spin.setRange(0.1, 0.95)
        self.train_ratio_spin.setValue(0.8)
        split_layout.addRow("训练比:", self.train_ratio_spin)

        self.split_btn = QPushButton("拆分")
        self.split_btn.setObjectName("PrimaryButton")
        self.split_btn.clicked.connect(self.split_dataset)
        split_layout.addRow("", self.split_btn)

        split_group.setLayout(split_layout)

        create_page = QWidget()
        create_page_layout = QVBoxLayout(create_page)
        create_page_layout.setContentsMargins(10, 10, 10, 10)
        create_page_layout.addWidget(create_group)
        create_page_layout.addStretch()
        workflow_tabs.addTab(create_page, "创建")

        load_page = QWidget()
        load_page_layout = QVBoxLayout(load_page)
        load_page_layout.setContentsMargins(10, 10, 10, 10)
        load_page_layout.addWidget(load_group)
        load_page_layout.addStretch()
        workflow_tabs.addTab(load_page, "加载")

        split_page = QWidget()
        split_page_layout = QVBoxLayout(split_page)
        split_page_layout.setContentsMargins(10, 10, 10, 10)
        split_page_layout.addWidget(split_group)
        split_page_layout.addStretch()
        workflow_tabs.addTab(split_page, "拆分")

        main_splitter.addWidget(workflow_tabs)

        # Dataset info
        inspect_widget = QWidget()
        inspect_layout = QVBoxLayout(inspect_widget)
        inspect_layout.setContentsMargins(10, 0, 0, 0)
        inspect_layout.setSpacing(10)
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Info table
        info_group = QGroupBox("数据集信息")
        info_layout = QVBoxLayout()

        self.info_table = QTableWidget()
        self.info_table.setColumnCount(3)
        self.info_table.setHorizontalHeaderLabels(["集合", "图片数", "标注数"])
        self.info_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        info_layout.addWidget(self.info_table)

        # Class distribution
        self.class_table = QTableWidget()
        self.class_table.setColumnCount(2)
        self.class_table.setHorizontalHeaderLabels(["类别", "ID"])
        self.class_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        class_label = QLabel("类别分布:")
        class_label.setObjectName("MutedText")
        info_layout.addWidget(class_label)
        info_layout.addWidget(self.class_table)

        info_group.setLayout(info_layout)
        splitter.addWidget(info_group)

        # Validation results
        valid_group = QGroupBox("数据集验证")
        valid_layout = QVBoxLayout()

        self.validate_btn = QPushButton("验证数据集")
        self.validate_btn.setObjectName("PrimaryButton")
        self.validate_btn.clicked.connect(self.validate_dataset)
        valid_layout.addWidget(self.validate_btn)

        self.validation_text = QTextEdit()
        self.validation_text.setReadOnly(True)
        self.validation_text.setFont(QFont("monospace", 9))
        self.validation_text.setMaximumHeight(150)
        valid_layout.addWidget(self.validation_text)

        valid_group.setLayout(valid_layout)
        splitter.addWidget(valid_group)

        splitter.setSizes([420, 180])
        inspect_layout.addWidget(splitter)
        main_splitter.addWidget(inspect_widget)
        main_splitter.setSizes([420, 680])

        layout.addWidget(main_splitter)

    def browse_save_path(self):
        path = QFileDialog.getExistingDirectory(self, "选择保存路径")
        if path:
            self.save_path_edit.setText(path)

    def browse_data_yaml(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 data.yaml", "", "YAML 文件 (*.yaml *.yml);;所有文件 (*)"
        )
        if path:
            self.data_yaml_edit.setText(path)

    def browse_dir(self, edit: QLineEdit, title: str):
        path = QFileDialog.getExistingDirectory(self, title)
        if path:
            edit.setText(path)

    def create_dataset(self):
        name = self.dataset_name_edit.text().strip()
        classes_text = self.classes_edit.text().strip()

        if not name:
            QMessageBox.warning(self, "错误", "请输入数据集名称")
            return
        if not classes_text:
            QMessageBox.warning(self, "错误", "请输入至少一个类别")
            return

        classes = [c.strip() for c in classes_text.split(",") if c.strip()]
        save_path = self.save_path_edit.text()

        try:
            manager = DatasetManager(save_path)
            dataset_path = manager.create_yolo_dataset(name, classes)
            data_yaml = os.path.join(dataset_path, "data.yaml")

            QMessageBox.information(self, "成功", f"数据集创建成功!\n路径: {dataset_path}")
            self.data_yaml_edit.setText(data_yaml)
            self.load_dataset()
        except Exception as e:
            QMessageBox.critical(self, "创建失败", str(e))

    def load_dataset(self):
        yaml_path = self.data_yaml_edit.text().strip()
        if not yaml_path or not os.path.exists(yaml_path):
            QMessageBox.warning(self, "错误", "请选择有效的 data.yaml 文件")
            return

        try:
            import yaml
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            dataset_path = data.get("path", os.path.dirname(yaml_path))
            self.dataset_manager = DatasetManager(dataset_path)
            self.annotation_manager.set_classes(list(data.get("names", {}).values()))

            # Display info
            info = self.dataset_manager.get_dataset_info(dataset_path)
            self.display_dataset_info(info, data)

            self.dataset_loaded.emit(yaml_path)
            QMessageBox.information(self, "成功", "数据集加载成功!")
        except Exception as e:
            QMessageBox.critical(self, "加载失败", str(e))

    def split_dataset(self):
        images_dir = self.src_images_edit.text().strip()
        labels_dir = self.src_labels_edit.text().strip()

        if not images_dir or not os.path.exists(images_dir):
            QMessageBox.warning(self, "错误", "请选择有效的图片目录")
            return

        # Auto-derive labels directory using YOLO convention if not specified
        if not labels_dir:
            labels_dir = labels_dir_for_image_dir(images_dir)

        classes_text = self.classes_edit.text().strip()
        if not classes_text:
            QMessageBox.warning(self, "错误", "请输入类别")
            return

        classes = [c.strip() for c in classes_text.split(",") if c.strip()]
        output_dir = os.path.join(self.save_path_edit.text(), "split_dataset")

        try:
            manager = DatasetManager(self.save_path_edit.text())
            ratio = self.train_ratio_spin.value()
            result_path = manager.split_dataset(
                images_dir, labels_dir, output_dir, classes,
                train_ratio=ratio, val_ratio=1 - ratio - 0.05, test_ratio=0.05
            )
            self.data_yaml_edit.setText(os.path.join(result_path, "data.yaml"))
            QMessageBox.information(self, "成功", f"数据集拆分完成!\n路径: {result_path}")
        except Exception as e:
            QMessageBox.critical(self, "拆分失败", str(e))

    def display_dataset_info(self, info: dict, yaml_data: dict):
        """Display dataset info in tables."""
        # Split info
        splits = info.get("splits", {})
        self.info_table.setRowCount(len(splits))
        for i, (split, counts) in enumerate(splits.items()):
            self.info_table.setItem(i, 0, QTableWidgetItem(split))
            self.info_table.setItem(i, 1, QTableWidgetItem(str(counts.get("images", 0))))
            self.info_table.setItem(i, 2, QTableWidgetItem(str(counts.get("labels", 0))))

        # Class info
        names = yaml_data.get("names", {})
        self.class_table.setRowCount(len(names))
        for i, (idx, name) in enumerate(names.items()):
            self.class_table.setItem(i, 0, QTableWidgetItem(str(name)))
            self.class_table.setItem(i, 1, QTableWidgetItem(str(idx)))

    def validate_dataset(self):
        yaml_path = self.data_yaml_edit.text().strip()
        if not yaml_path or not os.path.exists(yaml_path):
            QMessageBox.warning(self, "错误", "请先加载数据集")
            return

        import yaml
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        dataset_path = data.get("path", os.path.dirname(yaml_path))
        manager = DatasetManager(dataset_path)

        issues = manager.validate_dataset(dataset_path)

        self.validation_text.clear()
        if issues:
            self.validation_text.append("发现以下问题:\n")
            for issue in issues:
                self.validation_text.append(f"  - {issue}")
        else:
            self.validation_text.append("数据集验证通过，未发现问题!")
