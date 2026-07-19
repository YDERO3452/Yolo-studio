"""Dataset management panel."""

import os

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.annotation import AnnotationManager
from core.dataset import DatasetManager
from gui.annotation_io import labels_dir_for_image_dir
from gui.theme import Theme


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
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.setHandleWidth(6)
        main_splitter.setChildrenCollapsible(False)

        def mark_quiet(*buttons):
            for button in buttons:
                button.setObjectName("QuietButton")

        workflow_tabs = QTabWidget()
        workflow_tabs.setMinimumWidth(350)
        workflow_tabs.setDocumentMode(True)

        # Create new dataset
        create_group = QGroupBox("创建数据集")
        create_layout = QFormLayout()
        create_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.dataset_name_edit = QLineEdit()
        self.dataset_name_edit.setPlaceholderText("我的数据集")
        create_layout.addRow("数据集名称:", self.dataset_name_edit)

        self.save_path_edit = QLineEdit("data")
        browse_btn = QPushButton("浏览...")
        mark_quiet(browse_btn)
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
        mark_quiet(load_browse)
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
        mark_quiet(src_img_btn)
        src_img_btn.clicked.connect(lambda: self.browse_dir(self.src_images_edit, "选择图片目录"))
        src_img_row = QHBoxLayout()
        src_img_row.addWidget(self.src_images_edit)
        src_img_row.addWidget(src_img_btn)
        split_layout.addRow("源图片目录:", src_img_row)

        self.src_labels_edit = QLineEdit()
        self.src_labels_edit.setPlaceholderText("留空则自动从图片目录推导 (images/ → labels/)")
        src_lbl_btn = QPushButton("浏览...")
        mark_quiet(src_lbl_btn)
        src_lbl_btn.clicked.connect(lambda: self.browse_dir(self.src_labels_edit, "选择标注目录"))
        src_lbl_row = QHBoxLayout()
        src_lbl_row.addWidget(self.src_labels_edit)
        src_lbl_row.addWidget(src_lbl_btn)
        split_layout.addRow("源标注目录:", src_lbl_row)

        self.split_classes_edit = QLineEdit(self.classes_edit.text())
        self.split_classes_edit.setPlaceholderText("猫, 狗, 人")
        self.classes_edit.textChanged.connect(self.split_classes_edit.setText)
        self.split_classes_edit.textChanged.connect(self.classes_edit.setText)
        split_layout.addRow("类别:", self.split_classes_edit)

        self.split_output_edit = QLineEdit(self.save_path_edit.text())
        split_output_btn = QPushButton("浏览...")
        mark_quiet(split_output_btn)
        split_output_btn.clicked.connect(
            lambda: self.browse_dir(self.split_output_edit, "选择输出根目录")
        )
        split_output_row = QHBoxLayout()
        split_output_row.addWidget(self.split_output_edit)
        split_output_row.addWidget(split_output_btn)
        self.save_path_edit.textChanged.connect(self.split_output_edit.setText)
        self.split_output_edit.textChanged.connect(self.save_path_edit.setText)
        split_layout.addRow("输出根目录:", split_output_row)

        self.train_ratio_spin = QDoubleSpinBox()
        self.train_ratio_spin.setRange(0.1, 0.95)
        self.train_ratio_spin.setValue(0.8)
        split_layout.addRow("训练比:", self.train_ratio_spin)

        self.split_btn = QPushButton("拆分")
        self.split_btn.setObjectName("PrimaryButton")
        self.split_btn.clicked.connect(self.split_dataset)
        split_layout.addRow("", self.split_btn)

        split_group.setLayout(split_layout)

        flat_section_style = f"""
            QGroupBox {{
                background: transparent;
                border: none;
                border-top: 1px solid {Theme.BORDER};
                border-radius: 0;
                margin-top: 14px;
                padding: 9px 0 0 0;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 0;
                padding: 0 8px 0 0;
                color: {Theme.TEXT_MUTED};
            }}
        """
        for section in (create_group, load_group, split_group):
            section.setFlat(True)
            section.setStyleSheet(flat_section_style)

        for form_layout in (create_layout, load_layout, split_layout):
            form_layout.setContentsMargins(0, 7, 0, 0)
            form_layout.setHorizontalSpacing(8)
            form_layout.setVerticalSpacing(7)
            form_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)

        create_page = QWidget()
        create_page_layout = QVBoxLayout(create_page)
        create_page_layout.setContentsMargins(6, 6, 6, 6)
        create_page_layout.setSpacing(4)
        create_page_layout.addWidget(create_group)
        create_page_layout.addStretch()
        workflow_tabs.addTab(create_page, "创建")

        load_page = QWidget()
        load_page_layout = QVBoxLayout(load_page)
        load_page_layout.setContentsMargins(6, 6, 6, 6)
        load_page_layout.setSpacing(4)
        load_page_layout.addWidget(load_group)
        load_page_layout.addStretch()
        workflow_tabs.addTab(load_page, "加载")

        split_page = QWidget()
        split_page_layout = QVBoxLayout(split_page)
        split_page_layout.setContentsMargins(6, 6, 6, 6)
        split_page_layout.setSpacing(4)
        split_page_layout.addWidget(split_group)
        split_page_layout.addStretch()
        workflow_tabs.addTab(split_page, "拆分")

        main_splitter.addWidget(workflow_tabs)

        # Dataset info
        inspect_widget = QWidget()
        inspect_layout = QVBoxLayout(inspect_widget)
        inspect_layout.setContentsMargins(8, 2, 2, 2)
        inspect_layout.setSpacing(6)
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setHandleWidth(6)
        splitter.setChildrenCollapsible(False)

        # Info table
        info_group = QGroupBox("数据集信息")
        info_layout = QVBoxLayout()

        self.dataset_empty_label = QLabel("尚未加载数据集")
        self.dataset_empty_label.setObjectName("MutedText")
        self.dataset_empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_layout.addWidget(self.dataset_empty_label)

        self.info_table = QTableWidget()
        self.info_table.setColumnCount(3)
        self.info_table.setHorizontalHeaderLabels(["集合", "图片数", "标注数"])
        self.info_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.info_table.verticalHeader().setVisible(False)
        self.info_table.setShowGrid(False)
        self.info_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.info_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        info_layout.addWidget(self.info_table)

        # Class distribution
        self.class_table = QTableWidget()
        self.class_table.setColumnCount(2)
        self.class_table.setHorizontalHeaderLabels(["类别", "ID"])
        self.class_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.class_table.verticalHeader().setVisible(False)
        self.class_table.setShowGrid(False)
        self.class_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.class_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
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
        self.validate_btn.setObjectName("SecondaryButton")
        self.validate_btn.clicked.connect(self.validate_dataset)
        valid_layout.addWidget(self.validate_btn)

        self.validation_text = QTextEdit()
        self.validation_text.setReadOnly(True)
        self.validation_text.setFont(QFont("monospace", 9))
        self.validation_text.setMinimumHeight(96)
        self.validation_text.setPlainText("暂无验证结果")
        valid_layout.addWidget(self.validation_text)

        valid_group.setLayout(valid_layout)
        splitter.addWidget(valid_group)

        for section in (info_group, valid_group):
            section.setFlat(True)
            section.setStyleSheet(flat_section_style)

        for section_layout in (info_layout, valid_layout):
            section_layout.setContentsMargins(0, 7, 0, 0)
            section_layout.setSpacing(6)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([460, 170])
        inspect_layout.addWidget(splitter)
        main_splitter.addWidget(inspect_widget)
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setSizes([380, 720])

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
            raw_names = data.get("names", {})
            if isinstance(raw_names, dict):
                class_names = list(raw_names.values())
            elif isinstance(raw_names, list):
                class_names = raw_names
            else:
                class_names = []
            self.annotation_manager.set_classes(class_names)

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

        classes_text = self.split_classes_edit.text().strip()
        if not classes_text:
            QMessageBox.warning(self, "错误", "请输入类别")
            return

        classes = [c.strip() for c in classes_text.split(",") if c.strip()]
        output_root = self.split_output_edit.text().strip()
        if not output_root:
            QMessageBox.warning(self, "错误", "请选择输出根目录")
            return
        output_dir = os.path.join(output_root, "split_dataset")

        try:
            manager = DatasetManager(output_root)
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
        if isinstance(names, dict):
            name_items = list(names.items())
        elif isinstance(names, list):
            name_items = list(enumerate(names))
        else:
            name_items = []
        self.class_table.setRowCount(len(name_items))
        for i, (idx, name) in enumerate(name_items):
            self.class_table.setItem(i, 0, QTableWidgetItem(str(name)))
            self.class_table.setItem(i, 1, QTableWidgetItem(str(idx)))

        has_summary = bool(splits or name_items)
        self.dataset_empty_label.setText(
            "尚未加载数据集" if has_summary else "暂无可显示的数据集统计"
        )
        self.dataset_empty_label.setVisible(not has_summary)

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
