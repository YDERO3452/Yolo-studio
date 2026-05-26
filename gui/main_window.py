"""Main application window for YOLO Studio.

The window is structured as a professional workbench:
top status bar, left workspace navigation, central work area, and a right
inspector for annotation-specific context.
"""

from __future__ import annotations

import os
import traceback
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PyQt6.QtCore import QObject, QEvent, QPoint, QSize, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QFont, QIcon, QImageReader, QKeySequence, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QTabWidget,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
from loguru import logger

from core.annotation import ShapeType
from core.class_manager import ClassManager
from core.config import ConfigManager
from core.dataset import DatasetManager
from core.model_manager import ModelManager
from core.project_manager import ProjectManager
from gui.advanced_features_panel import AdvancedFeaturesPanel
from gui.annotation_io import (
    label_path_for_image,
    labels_dir_for_image_dir,
    load_yolo_shapes,
    save_yolo_shapes,
    shape_type_value,
)
from gui.canvas import AnnotationCanvas, CanvasMode
from gui.class_panel import ClassListPanel
from gui.dataset_panel import DatasetPanel
from gui.export_panel import ExportPanel
from gui.inference_panel import InferencePanel
from gui.project_panel import ProjectPanel
from gui.theme import Theme, build_stylesheet
from gui.training_panel import TrainingPanel
from gui.training_results_panel import TrainingResultsPanel
from gui.ui_components import Card, SectionTitle, StatusPill
from gui.workflow_optimization_panel import WorkflowOptimizationPanel
from gui.llm_handler import LLMBatchInferenceWorker, LLMInferenceWorker, load_llm_config, save_llm_config
from gui.yolo_label_worker import YOLOAutoLabelWorker
from gui.file_list_widget import FileListWidget
from gui.annotation_list_widget import AnnotationListWidget

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

class MainWindow(QMainWindow):
    """Main window with a workbench-oriented UI."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Studio")
        self.setMinimumSize(1280, 800)
        self.resize(1600, 920)

        self.config_manager = ConfigManager()
        self.class_manager = ClassManager()
        self._ensure_default_classes()
        self.model_manager = ModelManager()

        self.current_image_path: Optional[str] = None
        self.current_image_dir: Optional[str] = None
        self.current_project: Optional[dict] = None
        self.image_list: List[str] = []
        self.current_image_index = -1
        self._yolo_label_thread: Optional[QThread] = None
        self._yolo_label_worker: Optional[YOLOAutoLabelWorker] = None
        self._updating_annot_list = False
        self._dirty = False
        self._last_dataset_yaml: Optional[str] = None
        self._offered_training_for_yaml: Optional[str] = None
        self._yolo_tools_dialog: Optional[QDialog] = None
        self.prompt_for_class_after_draw = True
        self._llm_worker = None
        self._llm_batch_worker = None
        self._llm_progress_dialog = None
        self._llm_batch_class_id = 0
        self._llm_batch_class_name = ""

        self._build_ui()
        self._init_menus()
        self._init_statusbar()
        self._connect_signals()
        self._apply_theme()
        self._show_launch_page()
        self._update_status()

        logger.info("MainWindow initialized")

    def _ensure_default_classes(self) -> None:
        if self.class_manager.get_all_classes():
            return
        defaults = self.config_manager.get("annotation", "default_classes", ["目标"]) or ["目标"]
        for class_name in defaults:
            name = str(class_name).strip()
            if name:
                self.class_manager.get_or_create_class(name)
        self.class_manager.save()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.app_stack = QStackedWidget()
        self.setCentralWidget(self.app_stack)

        self.project_panel = ProjectPanel(self.class_manager, parent=self)

        self.launch_page = QWidget()
        launch_layout = QVBoxLayout(self.launch_page)
        launch_layout.setContentsMargins(0, 0, 0, 0)
        launch_layout.setSpacing(0)
        self.launch_project_host = QWidget()
        launch_host_layout = QVBoxLayout(self.launch_project_host)
        launch_host_layout.setContentsMargins(0, 0, 0, 0)
        launch_host_layout.setSpacing(0)
        launch_layout.addWidget(self.launch_project_host)
        self.app_stack.addWidget(self.launch_page)

        self.workbench_page = QWidget()
        root = QHBoxLayout(self.workbench_page)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.workspace_stack = QStackedWidget()
        self.workspace_stack.addWidget(self._create_annotation_workspace())

        self.training_panel = TrainingPanel(config_manager=self.config_manager, parent=self)
        self.inference_panel = InferencePanel(config_manager=self.config_manager, parent=self)
        self.dataset_panel = DatasetPanel(config_manager=self.config_manager, parent=self)
        self.export_panel = ExportPanel(config_manager=self.config_manager, parent=self)
        self.quality_panel = self._create_quality_workspace()
        self.results_panel = TrainingResultsPanel(parent=self)
        self.project_workspace_host = QWidget()
        project_host_layout = QVBoxLayout(self.project_workspace_host)
        project_host_layout.setContentsMargins(0, 0, 0, 0)
        project_host_layout.setSpacing(0)

        self.workspace_stack.addWidget(
            self._wrap_workspace(
                "训练",
                "配置 YOLO 训练，监控实时指标，并使用最佳权重继续工作。",
                self.training_panel,
            )
        )
        self.workspace_stack.addWidget(
            self._wrap_workspace(
                "推理",
                "对图片、文件夹、视频或摄像头画面运行模型检测。",
                self.inference_panel,
            )
        )
        self.workspace_stack.addWidget(
            self._wrap_workspace(
                "数据",
                "创建 YOLO 数据集，加载 data.yaml，拆分数据并验证结构。",
                self.dataset_panel,
            )
        )
        self.workspace_stack.addWidget(
            self._wrap_workspace(
                "导出",
                "将训练权重转换为可部署的运行时格式。",
                self.export_panel,
            )
        )
        self.workspace_stack.addWidget(
            self._wrap_workspace(
                "质检",
                "检查标注覆盖、格式转换状态和项目预设。",
                self.quality_panel,
            )
        )
        self.workspace_stack.addWidget(
            self._wrap_workspace(
                "项目",
                "按项目管理图片、标签、类别、data.yaml 和训练输出。",
                self.project_workspace_host,
            )
        )
        self.workspace_stack.addWidget(
            self._wrap_workspace(
                "训练结果",
                "浏览训练产物，直接送到推理或导出流程。",
                self.results_panel,
            )
        )
        self.nav_rail = self._create_nav_rail()
        root.addWidget(self.nav_rail)
        root.addWidget(self.workspace_stack, stretch=1)
        self.app_stack.addWidget(self.workbench_page)
        self._move_project_panel_to(self.launch_project_host)
        self.app_stack.setCurrentWidget(self.launch_page)
        self.workspace_stack.setCurrentIndex(6)
        if hasattr(self, "annotation_tools_container"):
            self.annotation_tools_container.setVisible(False)
        self._update_project_gate()

        self.training_panel.model_ready.connect(self._on_trained_model_ready)

    def _move_project_panel_to(self, host: QWidget) -> None:
        old_parent = self.project_panel.parentWidget()
        if old_parent is host:
            return
        if old_parent is not None and old_parent.layout() is not None:
            old_parent.layout().removeWidget(self.project_panel)
        layout = host.layout()
        if layout is None:
            layout = QVBoxLayout(host)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)
        layout.addWidget(self.project_panel)

    def _show_launch_page(self) -> None:
        if not hasattr(self, "app_stack"):
            return
        self._move_project_panel_to(self.launch_project_host)
        self.app_stack.setCurrentWidget(self.launch_page)
        self.menuBar().setVisible(False)
        self.statusBar().setVisible(False)

    def _show_workbench_page(self) -> None:
        if not hasattr(self, "app_stack"):
            return
        self._move_project_panel_to(self.project_workspace_host)
        self.app_stack.setCurrentWidget(self.workbench_page)
        self.menuBar().setVisible(True)
        self.statusBar().setVisible(True)

    def _create_nav_rail(self) -> QWidget:
        rail = QWidget()
        rail.setObjectName("NavRail")
        rail.setFixedWidth(48)
        layout = QVBoxLayout(rail)
        layout.setContentsMargins(6, 8, 6, 8)
        layout.setSpacing(4)

        self.workspace_tab_group = QButtonGroup(self)
        self.workspace_tab_group.setExclusive(True)
        self.workspace_tab_buttons: dict[int, QPushButton] = {}
        for icon_name, index, tip in [
            ("ws_project", 6, "项目"),
            ("ws_annotate", 0, "标注"),
            ("ws_dataset", 3, "数据集"),
            ("ws_train", 1, "训练"),
            ("ws_results", 7, "训练结果"),
            ("ws_infer", 2, "推理"),
            ("ws_export", 4, "导出"),
            ("ws_qa", 5, "质检"),
        ]:
            btn = QPushButton()
            btn.setObjectName("NavButton")
            btn.setCheckable(True)
            btn.setToolTip(tip)
            btn.setFixedSize(34, 34)
            btn.setIcon(self._tool_icon(icon_name))
            btn.setIconSize(QSize(22, 22))
            btn.clicked.connect(lambda checked=False, i=index: self._switch_workspace(i))
            self.workspace_tab_group.addButton(btn, index)
            self.workspace_tab_buttons[index] = btn
            layout.addWidget(btn)
        self.workspace_tab_buttons[6].setChecked(True)

        # Separator between workspace nav and annotation tools
        sep = QWidget()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background: {Theme.BORDER};")
        layout.addSpacing(6)
        layout.addWidget(sep)
        layout.addSpacing(6)

        # Annotation drawing tools (visible only in annotation workspace)
        self.annotation_tools_container = QWidget()
        tools_layout = QVBoxLayout(self.annotation_tools_container)
        tools_layout.setContentsMargins(0, 0, 0, 0)
        tools_layout.setSpacing(7)

        self.mode_actions: dict[CanvasMode, QPushButton] = {}
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        for mode, icon_name, tip in [
            (CanvasMode.EDIT, "select", "选择并编辑标注 (V)"),
            (CanvasMode.CREATE_BBOX, "rect", "绘制矩形框 (R)"),
            (CanvasMode.CREATE_POLYGON, "poly", "绘制多边形 (P)"),
            (CanvasMode.CREATE_OBB, "obb", "绘制旋转框 (O)"),
            (CanvasMode.CREATE_KEYPOINT, "keypoint", "创建关键点 (K)"),
        ]:
            btn = QPushButton()
            btn.setObjectName("ToolButton")
            btn.setCheckable(True)
            btn.setToolTip(tip)
            btn.setFixedSize(34, 34)
            btn.setIcon(self._tool_icon(icon_name))
            btn.setIconSize(QSize(22, 22))
            btn.clicked.connect(lambda checked=False, m=mode: self._set_canvas_mode(m))
            self.mode_group.addButton(btn)
            self.mode_actions[mode] = btn
            tools_layout.addWidget(btn)

        tools_layout.addSpacing(10)
        fit_btn = QPushButton()
        fit_btn.setObjectName("ToolButton")
        fit_btn.setToolTip("适配图片到视口 (F)")
        fit_btn.setFixedSize(34, 34)
        fit_btn.setIcon(self._tool_icon("fit"))
        fit_btn.setIconSize(QSize(22, 22))
        fit_btn.clicked.connect(self.canvas.fit_to_window)
        tools_layout.addWidget(fit_btn)

        layout.addWidget(self.annotation_tools_container)
        self.mode_actions[CanvasMode.CREATE_BBOX].setChecked(True)

        layout.addStretch()
        return rail

    def _create_annotation_workspace(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)

        self.canvas = AnnotationCanvas()
        self.canvas.set_classes(self.class_manager.get_all_classes())

        center_layout.addWidget(self._create_annotation_control_bar_v2())

        self.scroll_area = QScrollArea()
        self.scroll_area.setObjectName("CanvasScrollArea")
        self.scroll_area.setWidget(self.canvas)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.scroll_area.viewport().installEventFilter(self)
        center_layout.addWidget(self.scroll_area, stretch=1)

        layout.addWidget(center, stretch=1)

        canvas_separator = QWidget()
        canvas_separator.setObjectName("CanvasInspectorSeparator")
        canvas_separator.setFixedWidth(8)
        layout.addWidget(canvas_separator)

        layout.addWidget(self._create_inspector_v2())
        return page

    @staticmethod
    def _tool_icon(name: str) -> QIcon:
        svg_path = Path(__file__).parent.parent / "resources" / "icons" / f"{name}.svg"
        if svg_path.exists():
            svg_data = svg_path.read_text(encoding="utf-8")
            svg_data = svg_data.replace("currentColor", Theme.TEXT)
            from PyQt6.QtSvg import QSvgRenderer
            from PyQt6.QtCore import QByteArray
            renderer = QSvgRenderer(QByteArray(svg_data.encode()))
            pixmap = QPixmap(24, 24)
            pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pixmap)
            renderer.render(painter)
            painter.end()
            return QIcon(pixmap)
        pixmap = QPixmap(24, 24)
        pixmap.fill(Qt.GlobalColor.transparent)
        return QIcon(pixmap)

    def _create_annotation_control_bar_v2(self) -> QWidget:
        header = QWidget()
        header.setObjectName("AnnotationControlBar")
        header.setFixedHeight(96)
        layout = QVBoxLayout(header)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)

        model_row = QHBoxLayout()
        model_row.setContentsMargins(0, 0, 0, 0)
        model_row.setSpacing(7)
        layout.addLayout(model_row)

        model_row.addWidget(QLabel("模型"))
        self.yolo_model_combo = QComboBox()
        self.yolo_model_combo.setEditable(True)
        self.yolo_model_combo.setMinimumWidth(180)
        self.yolo_model_combo.setMaximumWidth(520)
        self.yolo_model_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._populate_yolo_models()
        model_row.addWidget(self.yolo_model_combo, stretch=1)

        browse_btn = QPushButton("浏览...")
        browse_btn.setFixedWidth(58)
        browse_btn.clicked.connect(self._browse_yolo_model)
        model_row.addWidget(browse_btn)

        self.yolo_load_btn = QPushButton("加载")
        self.yolo_load_btn.setFixedWidth(52)
        self.yolo_load_btn.clicked.connect(self._load_yolo_model)
        model_row.addWidget(self.yolo_load_btn)

        self.yolo_status_label = QLabel("未加载")
        self.yolo_status_label.setObjectName("InlineStatus")
        self.yolo_status_label.setMinimumWidth(64)
        self.yolo_status_label.setMaximumWidth(140)
        self.yolo_status_label.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        model_row.addWidget(self.yolo_status_label)

        model_row.addStretch(1)
        open_btn = QPushButton("打开文件夹")
        open_btn.setMinimumWidth(92)
        open_btn.clicked.connect(self._open_image_dir)
        save_btn = QPushButton("保存")
        save_btn.setMinimumWidth(56)
        save_btn.setObjectName("PrimaryButton")
        save_btn.clicked.connect(self._save_annotations)
        model_row.addWidget(open_btn)
        model_row.addWidget(save_btn)

        action_row = QHBoxLayout()
        action_row.setContentsMargins(0, 0, 0, 0)
        action_row.setSpacing(7)
        layout.addLayout(action_row)

        action_row.addWidget(QLabel("置信"))
        self.yolo_conf_spin = QDoubleSpinBox()
        self.yolo_conf_spin.setRange(0.01, 1.0)
        self.yolo_conf_spin.setSingleStep(0.05)
        self.yolo_conf_spin.setValue(float(self.config_manager.get("inference", "conf", 0.25)))
        action_row.addWidget(self.yolo_conf_spin)

        action_row.addWidget(QLabel("IOU"))
        self.yolo_iou_spin = QDoubleSpinBox()
        self.yolo_iou_spin.setRange(0.01, 1.0)
        self.yolo_iou_spin.setSingleStep(0.05)
        self.yolo_iou_spin.setValue(float(self.config_manager.get("inference", "iou", 0.7)))
        action_row.addWidget(self.yolo_iou_spin)

        self.yolo_max_det_spin = QSpinBox()
        self.yolo_max_det_spin.setRange(1, 3000)
        self.yolo_max_det_spin.setValue(int(self.config_manager.get("inference", "max_det", 300)))
        self.yolo_max_det_spin.setVisible(False)

        self.yolo_replace_check = QCheckBox("覆盖")
        self.yolo_replace_check.setToolTip("开启后自动标注会替换旧标注")
        action_row.addWidget(self.yolo_replace_check)

        self.yolo_model_class_check = QCheckBox("模型类别(中文)")
        self.yolo_model_class_check.setChecked(True)
        self.yolo_model_class_check.setToolTip("开启时自动翻译模型类别名称为中文")
        self.yolo_model_class_check.setVisible(False)

        self.yolo_current_btn = QPushButton("标注当前")
        self.yolo_current_btn.setObjectName("PrimaryButton")
        self.yolo_current_btn.setMinimumWidth(80)
        self.yolo_current_btn.clicked.connect(self._run_yolo_auto_label_current)
        action_row.addWidget(self.yolo_current_btn)

        self.yolo_all_btn = QPushButton("标注全部")
        self.yolo_all_btn.setMinimumWidth(80)
        self.yolo_all_btn.clicked.connect(self._run_yolo_auto_label_all)
        action_row.addWidget(self.yolo_all_btn)

        self.yolo_progress_bar = QProgressBar()
        self.yolo_progress_bar.setRange(0, 100)
        self.yolo_progress_bar.setValue(0)
        self.yolo_progress_bar.setFixedWidth(96)
        self.yolo_progress_bar.setFixedHeight(24)
        action_row.addWidget(self.yolo_progress_bar)

        action_row.addStretch(1)

        # LLM / negative sample buttons
        self.llm_btn = QPushButton("LLM")
        self.llm_btn.setObjectName("SecondaryButton")
        self.llm_btn.setMinimumWidth(52)
        self.llm_btn.setToolTip("LLM 自动标注")
        llm_menu = QMenu(self.llm_btn)
        llm_menu.addAction("单张推理", self._run_llm_auto_label)
        llm_menu.addAction("批量推理", self._run_llm_auto_label_batch)
        llm_menu.addSeparator()
        llm_menu.addAction("设置", self._show_auto_label_dialog_llm)
        self.llm_btn.setMenu(llm_menu)
        action_row.addWidget(self.llm_btn)

        self.negative_btn = QPushButton("无框")
        self.negative_btn.setObjectName("SecondaryButton")
        self.negative_btn.setCheckable(True)
        self.negative_btn.setMinimumWidth(52)
        self.negative_btn.setToolTip("标记当前图片为无目标(负样本)")
        self.negative_btn.toggled.connect(self._toggle_negative_sample)
        action_row.addWidget(self.negative_btn)

        return header

    def _create_inspector_v2(self) -> QWidget:
        inspector = QWidget()
        inspector.setObjectName("Inspector")
        inspector.setFixedWidth(256)
        inspector.setAutoFillBackground(True)
        inspector.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        layout = QVBoxLayout(inspector)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(4)
        title = QLabel("标注面板")
        title.setObjectName("PanelTitle")
        header_row.addWidget(title)
        header_row.addStretch()
        self.dirty_badge = StatusPill("已保存", "success")
        header_row.addWidget(self.dirty_badge)
        layout.addLayout(header_row)

        summary_row = QHBoxLayout()
        summary_row.setContentsMargins(0, 0, 0, 0)
        summary_row.setSpacing(4)
        self.image_pill = StatusPill("无图片")
        self.model_pill = StatusPill("无模型")
        self.dirty_pill = StatusPill("已保存", "success")
        self.image_pill.setVisible(False)
        self.model_pill.setVisible(False)
        self.dirty_pill.setVisible(False)
        summary_row.addWidget(self.image_pill)
        summary_row.addWidget(self.model_pill)
        summary_row.addStretch()
        # Keep pills for status updates, but hide this row in V2 for a tighter inspector.

        tabs = QTabWidget()
        tabs.setObjectName("InspectorTabs")
        tabs.setDocumentMode(True)
        tabs.tabBar().setExpanding(True)
        tabs.tabBar().setUsesScrollButtons(False)

        labels_tab = QWidget()
        labels_layout = QVBoxLayout(labels_tab)
        labels_layout.setContentsMargins(0, 4, 0, 0)
        labels_layout.setSpacing(6)
        self.class_panel = ClassListPanel(class_manager=self.class_manager, parent=self)
        self.class_panel.setObjectName("EmbeddedClassPanel")
        if hasattr(self.class_panel, "count_label"):
            self.class_panel.count_label.setVisible(False)
        self.class_panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        if hasattr(self.class_panel, "class_list_widget"):
            self.class_panel.class_list_widget.setMinimumHeight(150)
            self.class_panel.class_list_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        if hasattr(self.class_panel, "add_btn"):
            self.class_panel.add_btn.setFixedHeight(24)
        if hasattr(self.class_panel, "remove_btn"):
            self.class_panel.remove_btn.setFixedHeight(24)
        if hasattr(self.class_panel, "color_btn"):
            self.class_panel.color_btn.setFixedHeight(24)
        labels_layout.addWidget(self.class_panel, 1)
        tabs.addTab(labels_tab, "标签")

        objects_tab = QWidget()
        objects_layout = QVBoxLayout(objects_tab)
        objects_layout.setContentsMargins(0, 4, 0, 0)
        objects_layout.setSpacing(6)
        self.annot_list = AnnotationListWidget()
        self.annot_list.setMinimumHeight(80)
        objects_layout.addWidget(self.annot_list, stretch=1)

        object_actions = QHBoxLayout()
        object_actions.setSpacing(4)
        del_btn = QPushButton("删除")
        del_btn.setFixedHeight(24)
        del_btn.clicked.connect(self._delete_selected_shape)
        edit_btn = QPushButton("编辑")
        edit_btn.setFixedHeight(24)
        edit_btn.clicked.connect(self._delete_or_edit_selected_label)
        clr_btn = QPushButton("清空")
        clr_btn.setFixedHeight(24)
        clr_btn.clicked.connect(self._clear_shapes)
        object_actions.addWidget(del_btn)
        object_actions.addWidget(edit_btn)
        object_actions.addWidget(clr_btn)
        objects_layout.addLayout(object_actions)
        tabs.addTab(objects_tab, "对象")

        queue_tab = QWidget()
        queue_layout = QVBoxLayout(queue_tab)
        queue_layout.setContentsMargins(0, 4, 0, 0)
        queue_layout.setSpacing(6)
        self.file_search = QLineEdit()
        self.file_search.setPlaceholderText("搜索...")
        self.file_search.setFixedHeight(24)
        self.file_search.setClearButtonEnabled(True)
        self.file_search.textChanged.connect(self._filter_file_list)
        queue_layout.addWidget(self.file_search)

        self.file_list = FileListWidget()
        self.file_list.file_selected.connect(self._on_file_list_selected)
        self.file_list.setMinimumHeight(120)
        queue_layout.addWidget(self.file_list, stretch=1)

        nav_row = QHBoxLayout()
        nav_row.setSpacing(4)
        prev_btn = QPushButton("◀")
        prev_btn.setFixedSize(24, 24)
        prev_btn.setToolTip("上一张 (A/←)")
        prev_btn.clicked.connect(self._prev_image)
        next_btn = QPushButton("▶")
        next_btn.setFixedSize(24, 24)
        next_btn.setToolTip("下一张 (D/→)")
        next_btn.clicked.connect(self._next_image)
        self.queue_counter_label = QLabel("0 / 0")
        self.queue_counter_label.setObjectName("CounterText")
        self.queue_counter_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nav_row.addWidget(prev_btn)
        nav_row.addWidget(self.queue_counter_label, stretch=1)
        nav_row.addWidget(next_btn)
        queue_layout.addLayout(nav_row)

        autosave_row = QHBoxLayout()
        autosave_row.setSpacing(4)
        self.autosave_check = QCheckBox("自动保存")
        self.autosave_check.setChecked(bool(self.config_manager.get("annotation", "auto_save", True)))
        self.autosave_check.toggled.connect(lambda checked: self.config_manager.update("annotation", auto_save=checked))
        autosave_row.addWidget(self.autosave_check)
        autosave_row.addStretch()
        queue_layout.addLayout(autosave_row)

        main_actions = QHBoxLayout()
        main_actions.setSpacing(4)
        save_btn = QPushButton("保存标注")
        save_btn.setObjectName("PrimaryButton")
        save_btn.setFixedHeight(24)
        save_btn.clicked.connect(self._save_annotations)
        import_btn = QPushButton("导入")
        import_btn.setFixedHeight(24)
        import_btn.clicked.connect(self._open_image_dir)
        main_actions.addWidget(save_btn, stretch=1)
        main_actions.addWidget(import_btn, stretch=1)
        queue_layout.addLayout(main_actions)

        dataset_btn = QPushButton("生成训练数据集")
        dataset_btn.setObjectName("DatasetButton")
        dataset_btn.setFixedHeight(28)
        dataset_btn.clicked.connect(self._generate_training_dataset_from_queue)
        queue_layout.addWidget(dataset_btn)

        export_row = QHBoxLayout()
        export_row.setSpacing(4)
        export_row.addWidget(QLabel("导出:"))
        yolo_btn = QPushButton("YOLO")
        yolo_btn.setFixedHeight(24)
        yolo_btn.clicked.connect(self._generate_training_dataset_from_queue)
        export_btn = QPushButton("导出")
        export_btn.setFixedHeight(24)
        export_btn.clicked.connect(lambda: self._switch_workspace(4))
        export_row.addWidget(yolo_btn)
        export_row.addWidget(export_btn)
        queue_layout.addLayout(export_row)
        tabs.addTab(queue_tab, "队列")

        layout.addWidget(tabs, stretch=1)
        return inspector

    def _build_tool_dialog(self, title: str, content: QWidget, size: QSize) -> QDialog:
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.setModal(False)
        dialog.resize(size)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        layout.addWidget(content)
        return dialog

    def _create_quality_workspace(self) -> QWidget:
        tabs = QTabWidget()
        tabs.addTab(AdvancedFeaturesPanel(self.class_manager, parent=self), "统计")
        tabs.addTab(WorkflowOptimizationPanel(self.class_manager, parent=self), "流程")
        return tabs

    def _wrap_workspace(self, title: str, subtitle: str, content: QWidget) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        header = QFrame()
        header.setObjectName("Card")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(14, 12, 14, 12)
        header_layout.setSpacing(8)

        title_box = QVBoxLayout()
        title_box.setContentsMargins(0, 0, 0, 0)
        page_title = QLabel(title)
        page_title.setObjectName("BrandTitle")
        page_subtitle = QLabel(subtitle)
        page_subtitle.setObjectName("MutedText")
        title_box.addWidget(page_title)
        title_box.addWidget(page_subtitle)
        header_layout.addLayout(title_box)
        header_layout.addStretch()
        layout.addWidget(header)

        content_card = QFrame()
        content_card.setObjectName("Card")
        content_layout = QVBoxLayout(content_card)
        content_layout.setContentsMargins(10, 10, 10, 10)
        content_layout.addWidget(content)
        layout.addWidget(content_card, stretch=1)
        return page

    def _populate_yolo_models(self) -> None:
        self.yolo_model_combo.clear()
        seen: set[str] = set()
        seen_names: set[str] = set()

        for path in self._find_recent_yolo_weights():
            display = os.path.relpath(path, os.getcwd())
            self.yolo_model_combo.addItem(display, path)
            seen.add(os.path.normcase(os.path.abspath(path)))
            seen_names.add(os.path.basename(path).lower())

        for path in self._find_workspace_model_files():
            abs_path = os.path.normcase(os.path.abspath(path))
            if abs_path in seen:
                continue
            self.yolo_model_combo.addItem(os.path.basename(path), path)
            seen.add(abs_path)
            seen_names.add(os.path.basename(path).lower())

        for model_name in self.model_manager.list_available_models():
            if model_name.lower() not in seen_names:
                self.yolo_model_combo.addItem(model_name, model_name)
                seen.add(model_name)
                seen_names.add(model_name.lower())

    def _find_workspace_model_files(self) -> list[str]:
        model_files: list[str] = []
        roots = [Path.cwd(), Path.cwd() / "models"]
        if self.current_project and self.current_project.get("root"):
            roots.insert(0, Path(self.current_project["root"]) / "models")
        for root in roots:
            if not root.exists():
                continue
            for pattern in ("*.pt", "*.onnx", "*.engine"):
                model_files.extend(str(path) for path in root.glob(pattern) if path.is_file())
        return sorted(set(model_files), key=lambda path: os.path.getmtime(path), reverse=True)

    def _find_recent_yolo_weights(self) -> list[str]:
        weights: list[str] = []
        roots = []
        if self.current_project and self.current_project.get("root"):
            roots.append(Path(self.current_project["root"]) / "runs")
        roots.append(Path.cwd() / "runs")
        for runs_dir in roots:
            if not runs_dir.exists():
                continue
            for path in runs_dir.rglob("weights/best.pt"):
                if path.is_file():
                    weights.append(str(path))
            for path in runs_dir.rglob("weights/last.pt"):
                if path.is_file():
                    weights.append(str(path))
        return sorted(set(weights), key=lambda path: os.path.getmtime(path), reverse=True)[:20]

    # ------------------------------------------------------------------
    # Menus and status
    # ------------------------------------------------------------------

    def _init_menus(self) -> None:
        menubar = self.menuBar()
        self.workspace_actions: dict = {}
        self.project_required_actions: list[QAction] = []

        file_menu = menubar.addMenu("文件")
        self.project_required_actions.append(self._add_action(file_menu, "打开目录", self._open_image_dir, "Ctrl+O"))
        self.project_required_actions.append(self._add_action(file_menu, "打开图片", self._open_single_image, "Ctrl+I"))
        file_menu.addSeparator()
        self.project_required_actions.append(self._add_action(file_menu, "保存", self._save_annotations, "Ctrl+S"))
        file_menu.addSeparator()
        self._add_action(file_menu, "退出", self.close, "Ctrl+Q")

        edit_menu = menubar.addMenu("编辑")
        self.project_required_actions.append(self._add_action(edit_menu, "撤销", self.canvas.undo, "Ctrl+Z"))
        self.project_required_actions.append(self._add_action(edit_menu, "重做", self.canvas.redo, "Ctrl+Y"))
        edit_menu.addSeparator()
        self.project_required_actions.append(self._add_action(edit_menu, "删除", self._delete_selected_shape, "Delete"))
        self.project_required_actions.append(self._add_action(edit_menu, "清空全部", self._clear_shapes))

        view_menu = menubar.addMenu("视图")
        self.project_required_actions.append(self._add_action(view_menu, "适配窗口", self.canvas.fit_to_window, "Ctrl+F"))

        help_menu = menubar.addMenu("帮助")
        self._add_action(help_menu, "关于", self._show_about)

        self.project_required_actions.append(self._add_action(menubar, "自动标注", self._focus_auto_labeling_panel))
        self.project_required_actions.append(self._add_action(menubar, "视频截帧", self._show_video_capture))
        self.project_required_actions.append(self._add_action(menubar, "格式转换", self._show_format_conversion))
        self._add_action(menubar, "环境", self._show_env_check)
        self._update_project_gate()

    def _add_action(self, menu, text: str, callback, shortcut: str | None = None) -> QAction:
        action = QAction(text, self)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        action.triggered.connect(callback)
        menu.addAction(action)
        return action

    def _init_statusbar(self) -> None:
        self.statusBar().showMessage("就绪")
        self.status_image_label = QLabel("无图片")
        self.status_compute_label = QLabel(self._format_compute_status())
        self.status_zoom_label = QLabel("100%")
        self.status_pos_label = QLabel("")
        self.status_norm_pos_label = QLabel("")
        self.statusBar().addPermanentWidget(self.status_image_label)
        self.statusBar().addPermanentWidget(self.status_compute_label)
        self.statusBar().addPermanentWidget(self.status_zoom_label)
        self.statusBar().addPermanentWidget(self.status_pos_label)
        self.statusBar().addPermanentWidget(self.status_norm_pos_label)

    def _format_compute_status(self) -> str:
        detection = getattr(self.training_panel, "gpu_detection", None)
        if detection is None:
            return "计算设备: 检测中..."

        if detection.cuda_available and detection.gpus:
            gpu = detection.gpus[0]
            free = f", 空闲 {gpu.vram_free_mb}MB" if gpu.vram_free_mb else ""
            util = f", {gpu.utilization}%" if gpu.utilization else ""
            cuda = detection.cuda_version or "available"
            suffix = f" +{len(detection.gpus) - 1}" if len(detection.gpus) > 1 else ""
            return f"CUDA {cuda} | GPU {gpu.index}: {gpu.name}{suffix}{free}{util}"

        if detection.torch_version:
            return f"CPU 模式 | PyTorch {detection.torch_version}"
        return "CPU 模式 | CUDA 不可用"

    def _connect_signals(self) -> None:
        self.canvas.shape_created.connect(self._on_shape_created)
        self.canvas.shape_selected.connect(self._on_shape_selected)
        self.canvas.shape_deleted.connect(self._delete_shape_by_index)
        self.canvas.shapes_changed.connect(self._on_shapes_changed)
        self.canvas.mouse_position.connect(self._on_mouse_position)
        self.canvas.edit_label_requested.connect(self._on_edit_label)
        self.canvas.zoom_changed.connect(self._on_zoom_changed)
        self.canvas.class_switch_requested.connect(self.class_panel.set_current_class_id)
        self.class_panel.class_id_selected.connect(self._on_class_selected)
        self.class_panel.class_added.connect(lambda _name: self._refresh_class_quick_buttons())
        self.class_panel.class_removed.connect(lambda _name: self._refresh_class_quick_buttons())
        self.class_panel.class_renamed.connect(lambda _old, _new: self._refresh_class_quick_buttons())
        self.class_panel.class_color_changed.connect(lambda _name, _color: self._refresh_class_quick_buttons())
        self.annot_list.annotation_selected.connect(self._on_annot_list_selected)
        self.annot_list.annotation_delete_requested.connect(self._delete_shape_by_index)
        self.annot_list.annotation_edit_requested.connect(self._on_edit_label)
        self.project_panel.project_opened.connect(self._on_project_opened)
        self.project_panel.data_yaml_ready.connect(self._on_project_data_yaml_ready)
        self.results_panel.load_inference_requested.connect(self._load_result_for_inference)
        self.results_panel.load_export_requested.connect(self._load_result_for_export)
        self._refresh_class_quick_buttons()

    def _apply_theme(self) -> None:
        app = QApplication.instance()
        if app is not None:
            font = QFont()
            font.setPointSize(9)
            app.setFont(font)
        self.setStyleSheet(build_stylesheet())

    def eventFilter(self, obj, event):
        if (
            hasattr(self, "scroll_area")
            and obj is self.scroll_area.viewport()
            and event.type() == QEvent.Type.Resize
            and self.canvas.original_image is not None
        ):
            self.canvas.fit_to_window(self.scroll_area.viewport().size())
        return super().eventFilter(obj, event)

    # ------------------------------------------------------------------
    # Workspace switching
    # ------------------------------------------------------------------

    def _switch_workspace(self, index: int) -> None:
        if index != 6 and not self._is_project_ready():
            index = 6
            if hasattr(self, "project_panel"):
                self.statusBar().showMessage("请先在项目页新建/导入项目，并导入图片或视频截帧", 3500)
        if index == 6 and not self.current_project:
            self.workspace_stack.setCurrentIndex(6)
            for tab_index, button in getattr(self, "workspace_tab_buttons", {}).items():
                button.setChecked(tab_index == 6)
            if hasattr(self, "annotation_tools_container"):
                self.annotation_tools_container.setVisible(False)
            self._show_launch_page()
            return
        self._show_workbench_page()
        self.workspace_stack.setCurrentIndex(index)
        for action_index, action in getattr(self, "workspace_actions", {}).items():
            action.setChecked(action_index == index)
        for tab_index, button in getattr(self, "workspace_tab_buttons", {}).items():
            button.setChecked(tab_index == index)
        if hasattr(self, "annotation_tools_container"):
            self.annotation_tools_container.setVisible(index == 0)

    def _update_project_gate(self) -> None:
        ready = self._is_project_ready()
        for index, button in getattr(self, "workspace_tab_buttons", {}).items():
            button.setEnabled(index == 6 or ready)
        for action in getattr(self, "project_required_actions", []):
            action.setEnabled(ready)

    def _is_project_ready(self) -> bool:
        return bool(self.current_project and self.image_list)

    # ------------------------------------------------------------------
    # Project workflow
    # ------------------------------------------------------------------

    def _on_project_opened(self, project: dict) -> None:
        if not project or not project.get("root"):
            self.current_project = None
            self.image_list = []
            self.current_image_index = -1
            self.results_panel.set_project(None)
            self._update_project_gate()
            self._switch_workspace(6)
            self.statusBar().showMessage("项目已关闭", 2500)
            return

        self.current_project = project
        project_root = Path(project["root"])

        self.class_manager = ClassManager(str(project_root))
        self._ensure_default_classes()

        if hasattr(self.class_panel, "update_class_manager"):
            self.class_panel.update_class_manager(self.class_manager)
        else:
            self.class_panel.class_manager = self.class_manager
            self.class_panel.refresh_list()
        self.project_panel.set_class_manager(self.class_manager)
        self.canvas.set_classes(self.class_manager.get_all_classes())
        self._refresh_class_quick_buttons()
        self._update_quality_class_manager()

        images = ProjectManager.list_images(project)
        self.image_list = images
        self.current_image_dir = str(project_root / "images")
        self.current_image_index = 0 if images else -1
        self.file_search.clear()
        self.file_list.load_image_list(self.image_list)
        self._update_project_gate()

        yaml_path = project_root / "data.yaml"
        if yaml_path.exists():
            self._last_dataset_yaml = str(yaml_path)
            self.dataset_panel.data_yaml_edit.setText(str(yaml_path))
            self.training_panel.data_yaml_edit.setText(str(yaml_path))
        self.training_panel.project_edit.setText(str(project_root / "runs"))
        if not self.training_panel.name_edit.text().strip():
            self.training_panel.name_edit.setText("exp")
        self.results_panel.set_project(project)

        if images:
            self._switch_workspace(0)
            self._load_current_image()
        else:
            self._switch_workspace(6)
            self.current_image_path = None
            self.annot_list.refresh([])
            self.canvas.clear_shapes()
            self.canvas.original_image = None
            self.canvas.display_pixmap = None
            self.canvas.image_width = 0
            self.canvas.image_height = 0
            self.canvas.update()
            self._update_status()

        self.statusBar().showMessage(
            f"项目已打开: {project.get('name', project_root.name)} ({len(images)} 张图片)",
            3500,
        )

    def _on_project_data_yaml_ready(self, yaml_path: str) -> None:
        self._last_dataset_yaml = yaml_path
        self.dataset_panel.data_yaml_edit.setText(yaml_path)
        self.training_panel.data_yaml_edit.setText(yaml_path)
        if self.current_project:
            self.image_list = ProjectManager.list_images(self.current_project)
            self.current_image_dir = str(Path(self.current_project["root"]) / "images")
            if self.image_list:
                self.current_image_index = min(max(self.current_image_index, 0), len(self.image_list) - 1)
            else:
                self.current_image_index = -1
            self.file_list.load_image_list(self.image_list)
            if self.current_image_index >= 0:
                self._load_current_image()
        self._switch_workspace(1)
        self.statusBar().showMessage(f"data.yaml 已生成: {yaml_path}", 3500)

    def _load_result_for_inference(self, model_path: str) -> None:
        self._switch_workspace(2)
        self.inference_panel.load_model_from_path(model_path)

    def _load_result_for_export(self, model_path: str) -> None:
        self._switch_workspace(4)
        self.export_panel.load_model_from_path(model_path)

    def _update_quality_class_manager(self) -> None:
        tabs = getattr(self, "quality_panel", None)
        if not isinstance(tabs, QTabWidget):
            return
        for index in range(tabs.count()):
            widget = tabs.widget(index)
            if hasattr(widget, "class_manager"):
                widget.class_manager = self.class_manager

    # ------------------------------------------------------------------
    # File operations
    # ------------------------------------------------------------------

    def _open_image_dir(self) -> None:
        if not self.current_project:
            QMessageBox.warning(self, "需要项目", "请先新建或导入项目，再导入图片。")
            self._switch_workspace(6)
            return
        dir_path = QFileDialog.getExistingDirectory(self, "打开图片目录")
        if not dir_path:
            return

        project_root = Path(self.current_project["root"]).resolve()
        selected_dir = Path(dir_path).resolve()
        try:
            selected_dir.relative_to(project_root)
            inside_project = True
        except ValueError:
            inside_project = False
        if not inside_project:
            try:
                imported, skipped = ProjectManager().import_folder(self.current_project, selected_dir)
                self.current_project = ProjectManager().open_project(project_root)
                self._on_project_opened(self.current_project)
                QMessageBox.information(self, "导入完成", f"图片导入当前项目: {imported} 张，跳过 {skipped} 张")
            except Exception as exc:
                QMessageBox.critical(self, "导入失败", str(exc))
            return

        # ------------------------------------------------------------------
        # Auto-detect YOLO-standard directory structure.
        # Supports:
        #   1. dataset_root/images/           → images inside, labels in sibling labels/
        #   2. dataset_root/images/train/     → also val/, test/ subdirs
        #   3. plain folder of images         → legacy: labels beside images
        # ------------------------------------------------------------------
        images_dir = dir_path
        sub_dir = os.path.join(dir_path, "images")
        if os.path.isdir(sub_dir):
            # User selected the dataset root which has an images/ sub-dir.
            # Collect images from all images/ subdirectories (train/val/test…)
            image_list = []
            for entry in sorted(os.listdir(sub_dir)):
                entry_path = os.path.join(sub_dir, entry)
                if os.path.isdir(entry_path):
                    # e.g. images/train/, images/val/
                    image_list.extend([
                        os.path.join(entry_path, name)
                        for name in sorted(os.listdir(entry_path))
                        if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS
                    ])
                elif os.path.splitext(entry)[1].lower() in IMAGE_EXTENSIONS:
                    # images/ directly contains image files
                    image_list.append(entry_path)
            images_dir = sub_dir
        else:
            # No images/ sub-dir — the selected dir may itself be
            # images/train/ or just a plain folder of images.
            image_list = [
                os.path.join(dir_path, name)
                for name in os.listdir(dir_path)
                if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS
            ]

        self.image_list = sorted(image_list)
        if not self.image_list:
            QMessageBox.warning(self, "提示", "未找到图片")
            return

        self.current_image_dir = images_dir
        self.current_image_index = 0
        self.file_list.load_image_list(self.image_list)
        self._switch_workspace(0)
        self._load_current_image()

        # Log with directory info
        labels_dir = labels_dir_for_image_dir(images_dir)
        structure_info = f"images={images_dir}, labels={labels_dir}"
        logger.info(f"Opened image directory: {dir_path} ({len(self.image_list)} images, {structure_info})")

    def _open_single_image(self) -> None:
        if not self.current_project:
            QMessageBox.warning(self, "需要项目", "请先新建或导入项目，再导入图片。")
            self._switch_workspace(6)
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "打开图片",
            "",
            "图片 (*.jpg *.jpeg *.png *.bmp *.tiff *.webp);;所有文件 (*)",
        )
        if not path:
            return
        try:
            imported, skipped = ProjectManager().import_images(self.current_project, [path])
            self.current_project = ProjectManager().open_project(self.current_project["root"])
            self._on_project_opened(self.current_project)
            self.statusBar().showMessage(f"已导入图片: {imported}，跳过: {skipped}", 2500)
            return
        except Exception as exc:
            QMessageBox.critical(self, "导入失败", str(exc))
            return

    def _load_current_image(self) -> None:
        if not (0 <= self.current_image_index < len(self.image_list)):
            return
        path = self.image_list[self.current_image_index]
        if not self.canvas.load_image(path):
            QMessageBox.warning(self, "错误", f"无法加载图片:\n{path}")
            return
        self.current_image_path = path
        self.canvas.set_classes(self.class_manager.get_all_classes())
        self._load_annotations_for_image(path)
        self.file_list.highlight_current(self.current_image_index)
        self._set_dirty(False)
        self._update_status()
        # Update negative sample button state
        label_path = label_path_for_image(path)
        if os.path.isfile(label_path) and os.path.getsize(label_path) == 0:
            self.negative_btn.setChecked(True)
        else:
            self.negative_btn.setChecked(False)

    def _prev_image(self) -> None:
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self._load_current_image()

    def _next_image(self) -> None:
        if self.current_image_index < len(self.image_list) - 1:
            self.current_image_index += 1
            self._load_current_image()

    def _on_file_list_selected(self, index: int) -> None:
        if 0 <= index < len(self.image_list):
            self.current_image_index = index
            self._load_current_image()

    def _filter_file_list(self, text: str) -> None:
        needle = text.lower()
        for index in range(self.file_list.count()):
            item = self.file_list.item(index)
            item.setHidden(needle not in item.text().lower())

    # ------------------------------------------------------------------
    # Annotation IO
    # ------------------------------------------------------------------

    def _load_annotations_for_image(self, image_path: str) -> None:
        try:
            shapes = load_yolo_shapes(
                image_path,
                self.canvas.image_width,
                self.canvas.image_height,
                self.class_manager,
            )
            self.canvas.set_shapes(shapes)
            self.annot_list.refresh(shapes)
        except Exception as exc:
            logger.error(f"Failed to load labels for {image_path}: {exc}\n{traceback.format_exc()}")
            self.canvas.clear_shapes()
            self.annot_list.refresh([])
            QMessageBox.warning(self, "标签加载错误", f"无法加载标签:\n{exc}")

    def _autosave_annotations(self) -> None:
        if not self.current_image_path:
            return
        if not self.config_manager.get("annotation", "auto_save", True):
            return
        try:
            label_path = save_yolo_shapes(
                self.current_image_path,
                self.canvas.get_shapes(),
                self.canvas.image_width,
                self.canvas.image_height,
            )
            self.class_manager.save()
            self.file_list.load_image_list(self.image_list)
            self.file_list.highlight_current(self.current_image_index)
            self._set_dirty(False)
            self.statusBar().showMessage(f"已自动保存: {os.path.basename(label_path)}", 1800)
            self._maybe_offer_training_after_annotation()
        except Exception as exc:
            logger.error(f"Autosave error: {exc}")
            self.statusBar().showMessage(f"自动保存失败: {exc}", 3500)

    def _save_annotations(self) -> None:
        if not self.current_image_path:
            QMessageBox.warning(self, "提示", "尚未加载图片")
            return
        try:
            label_path = save_yolo_shapes(
                self.current_image_path,
                self.canvas.get_shapes(),
                self.canvas.image_width,
                self.canvas.image_height,
            )
            self.class_manager.save()
            self.file_list.load_image_list(self.image_list)
            self.file_list.highlight_current(self.current_image_index)
            self._set_dirty(False)
            self.statusBar().showMessage(f"已保存: {label_path}", 3000)
            logger.info(f"Saved labels: {label_path}")
            self._maybe_offer_training_after_annotation()
        except Exception as exc:
            logger.error(f"Save error: {exc}")
            QMessageBox.critical(self, "错误", f"保存失败:\n{exc}")

    # ------------------------------------------------------------------
    # Canvas and annotation events
    # ------------------------------------------------------------------

    def _set_canvas_mode(self, mode: CanvasMode) -> None:
        self.canvas.set_mode(mode)
        for item_mode, button in self.mode_actions.items():
            button.setChecked(item_mode == mode)
        mode_names = {
            CanvasMode.EDIT: "选择",
            CanvasMode.CREATE_BBOX: "矩形",
            CanvasMode.CREATE_POLYGON: "多边形",
            CanvasMode.CREATE_OBB: "OBB",
            CanvasMode.CREATE_KEYPOINT: "关键点",
        }
        self.statusBar().showMessage(f"模式: {mode_names.get(mode, mode.value)}", 1800)

    def _on_shape_created(self, shape: dict) -> None:
        if not self._assign_class_to_new_shape(shape):
            shapes = self.canvas.get_shapes()
            if shape in shapes:
                shapes.remove(shape)
                self.canvas.set_shapes(shapes)
            self.statusBar().showMessage("已取消标注: 需要先创建类别", 2200)
            return
        self.canvas.set_classes(self.class_manager.get_all_classes())
        self.annot_list.refresh(self.canvas.get_shapes())
        self._set_dirty(True)
        self._autosave_annotations()

    def _assign_class_to_new_shape(self, shape: dict) -> bool:
        if not self.class_manager.get_all_classes():
            class_id = self._create_class_from_prompt("新建类别", "类别名称:")
            if class_id is None:
                return False
            class_name = self.class_manager.get_class_name(class_id) or f"类别_{class_id}"
            shape["class_id"] = class_id
            shape["class_name"] = class_name
            self.class_panel.set_current_class_id(class_id)
            return True

        current_class = self.class_panel.get_selected_class()
        current_id = self.class_manager.get_class_id(current_class) if current_class else None
        if current_id is None:
            current_id = shape.get("class_id", self.class_panel.get_current_class_id())
            current_class = self.class_manager.get_class_name(current_id)
            if current_class is None:
                return False

        class_id = current_id
        class_name = current_class
        if self.prompt_for_class_after_draw and self.class_manager.get_all_classes():
            selected_id = self._ask_class_for_shape(class_id)
            if selected_id is None:
                return False
            class_id = selected_id
            class_name = self.class_manager.get_class_name(class_id) or f"类别_{class_id}"

        shape["class_id"] = class_id
        shape["class_name"] = class_name
        self.class_panel.set_current_class_id(class_id)
        return True

    def _ask_class_for_shape(self, current_id: int = 0) -> Optional[int]:
        classes = self.class_manager.get_all_classes()
        if not classes:
            return None
        current_id = max(0, min(current_id, len(classes) - 1))
        label, ok = QInputDialog.getItem(
            self,
            "选择类别",
            "类别:",
            classes + ["新建类别..."],
            current_id,
            False,
        )
        if not ok:
            return current_id
        if label == "新建类别...":
            return self._create_class_from_prompt("新建类别", "类别名称:")
        class_id = self.class_manager.get_or_create_class(label)
        self.class_manager.save()
        self.class_panel.refresh_list()
        self._refresh_class_quick_buttons()
        return class_id

    def _create_class_from_prompt(self, title: str = "新建类别", label: str = "类别名称:") -> Optional[int]:
        name, ok = QInputDialog.getText(self, title, label)
        if not ok or not name.strip():
            return None
        class_id = self.class_manager.get_or_create_class(name.strip())
        self.class_manager.save()
        self.class_panel.refresh_list()
        self._refresh_class_quick_buttons()
        self.canvas.set_classes(self.class_manager.get_all_classes())
        return class_id

    def _on_shape_selected(self, index: int) -> None:
        if self._updating_annot_list:
            return
        self._updating_annot_list = True
        self.annot_list.highlight_shape(index)
        self._updating_annot_list = False

    def _on_shapes_changed(self) -> None:
        self.annot_list.refresh(self.canvas.get_shapes())
        self._set_dirty(True)
        self._autosave_annotations()

    def _on_annot_list_selected(self, index: int) -> None:
        if self._updating_annot_list:
            return
        self._updating_annot_list = True
        self.canvas.selected_shape = index
        self.canvas.update()
        self._updating_annot_list = False

    def _on_mouse_position(self, x: int, y: int) -> None:
        self.status_pos_label.setText(f"X: {x}  Y: {y}")
        if self.canvas.image_width and self.canvas.image_height:
            self.status_norm_pos_label.setText(f"({x / self.canvas.image_width:.4f}, {y / self.canvas.image_height:.4f})")
        else:
            self.status_norm_pos_label.setText("")

    def _on_zoom_changed(self, level: float) -> None:
        self.status_zoom_label.setText(f"{level:.1f}x")

    def _on_edit_label(self, shape_index: int) -> None:
        shapes = self.canvas.get_shapes()
        if not (0 <= shape_index < len(shapes)):
            return
        current_name = shapes[shape_index].get("class_name", "目标")
        new_name, ok = QInputDialog.getText(self, "编辑标签", "类别:", text=current_name)
        if ok and new_name.strip():
            class_name = new_name.strip()
            class_id = self.class_manager.get_or_create_class(class_name)
            shapes[shape_index]["class_id"] = class_id
            shapes[shape_index]["class_name"] = class_name
            self.canvas.set_shapes(shapes)
            self.canvas.set_classes(self.class_manager.get_all_classes())
            self.annot_list.refresh(shapes)
            self.class_panel.refresh_list()
            self._refresh_class_quick_buttons()
            self._set_dirty(True)
            self._autosave_annotations()

    def _delete_or_edit_selected_label(self) -> None:
        index = self.canvas.selected_shape
        if index < 0 and self.annot_list.currentItem() is not None:
            data = self.annot_list.currentItem().data(0, Qt.ItemDataRole.UserRole)
            index = int(data) if data is not None else -1
        if index >= 0:
            self._on_edit_label(index)

    def _on_class_selected(self, class_id: int) -> None:
        self.canvas.set_current_class_id(class_id)
        for index, button in enumerate(getattr(self, "class_quick_buttons", [])):
            button.setChecked(index == class_id)

    def _delete_selected_shape(self) -> None:
        self._delete_shape_by_index(self.canvas.selected_shape)

    def _delete_shape_by_index(self, index: int) -> None:
        shapes = self.canvas.get_shapes()
        if 0 <= index < len(shapes):
            shapes.pop(index)
            self.canvas.selected_shape = -1
            self.canvas.set_shapes(shapes)
            self.annot_list.refresh(shapes)
            self._set_dirty(True)
            self._autosave_annotations()

    def _clear_shapes(self) -> None:
        if not self.canvas.get_shapes():
            return
        self.canvas.push_undo()
        self.canvas.clear_shapes()
        self.annot_list.refresh([])
        self._set_dirty(True)
        self._autosave_annotations()

    def _refresh_class_quick_buttons(self) -> None:
        if not hasattr(self, "class_quick_layout"):
            return
        while self.class_quick_layout.count():
            item = self.class_quick_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self.class_quick_buttons = []
        self.class_quick_group = QButtonGroup(self)
        self.class_quick_group.setExclusive(True)

        current_id = self.class_panel.get_current_class_id() if hasattr(self, "class_panel") else 0
        for index, class_name in enumerate(self.class_manager.get_all_classes()[:8]):
            full_text = f"{index + 1} {class_name}"
            btn = QPushButton()
            btn.setObjectName("ClassChip")
            btn.setCheckable(True)
            btn.setMaximumWidth(104)
            btn.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
            btn.setText(btn.fontMetrics().elidedText(full_text, Qt.TextElideMode.ElideRight, 96))
            btn.setToolTip(f"切换到类别: {class_name} (快捷键 {index + 1})")
            btn.clicked.connect(lambda checked=False, i=index: self.class_panel.set_current_class_id(i))
            self.class_quick_group.addButton(btn, index)
            self.class_quick_layout.addWidget(btn)
            self.class_quick_buttons.append(btn)
            if index == current_id:
                btn.setChecked(True)

    def _all_queue_images_labeled(self) -> bool:
        if not self.image_list:
            return False
        return all(os.path.isfile(label_path_for_image(path)) for path in self.image_list)

    def _build_training_yaml_for_current_queue(self) -> Optional[str]:
        if not self.image_list:
            return None
        images_dir = self.current_image_dir or os.path.dirname(self.image_list[0])
        labels_dir = labels_dir_for_image_dir(images_dir)
        try:
            yaml_path = DatasetManager.build_data_yaml(
                images_dir=images_dir,
                labels_dir=labels_dir,
                classes=self.class_manager.get_all_classes(),
                train_ratio=0.8,
                val_ratio=0.2,
                test_ratio=0.0,
            )
            self._last_dataset_yaml = yaml_path
            self.dataset_panel.data_yaml_edit.setText(yaml_path)
            self.training_panel.data_yaml_edit.setText(yaml_path)
            return yaml_path
        except Exception as exc:
            logger.error(f"Failed to build data.yaml: {exc}")
            QMessageBox.warning(self, "生成 data.yaml 失败", str(exc))
            return None

    def _maybe_offer_training_after_annotation(self) -> None:
        if not self._all_queue_images_labeled():
            return
        yaml_path = self._build_training_yaml_for_current_queue()
        if not yaml_path or yaml_path == self._offered_training_for_yaml:
            return
        self._offered_training_for_yaml = yaml_path
        reply = QMessageBox.question(
            self,
            "标注完成",
            f"当前队列已全部有标签，并已生成 data.yaml:\n{yaml_path}\n\n是否前往训练？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.training_panel.data_yaml_edit.setText(yaml_path)
            self._switch_workspace(1)

    def _generate_training_dataset_from_queue(self) -> None:
        if not self.image_list:
            QMessageBox.warning(self, "提示", "请先导入图片队列")
            return
        yaml_path = self._build_training_yaml_for_current_queue()
        if not yaml_path:
            return
        self.statusBar().showMessage(f"已生成训练数据集: {yaml_path}", 4000)
        QMessageBox.information(self, "生成训练数据集", f"已生成 data.yaml:\n{yaml_path}")

    # ------------------------------------------------------------------
    # Open-vocabulary detection
    # ------------------------------------------------------------------

    def _browse_yolo_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 YOLO 模型",
            "",
            "模型文件 (*.pt *.onnx *.engine);;所有文件 (*)",
        )
        if path:
            self.yolo_model_combo.setCurrentText(path)

    def _selected_yolo_model(self) -> str:
        text = self.yolo_model_combo.currentText().strip()
        index = self.yolo_model_combo.currentIndex()
        data = self.yolo_model_combo.currentData() if index >= 0 and self.yolo_model_combo.itemText(index) == text else None
        return str(data or text)

    def _load_yolo_model(self) -> None:
        model_name = self._selected_yolo_model()
        if not model_name:
            QMessageBox.warning(self, "提示", "请选择 YOLO 模型")
            return

        self.yolo_load_btn.setEnabled(False)
        self.yolo_status_label.setText(f"正在加载: {os.path.basename(model_name)}")
        QApplication.processEvents()
        success = self.model_manager.load_model(model_name)
        if success:
            self.yolo_status_label.setText(f"已加载: {os.path.basename(model_name)}")
            self.model_pill.setText(os.path.basename(model_name))
        else:
            self.yolo_status_label.setText("加载失败")
            QMessageBox.critical(self, "错误", f"无法加载 YOLO 模型:\n{model_name}")
        self.yolo_load_btn.setEnabled(True)

    def _ensure_yolo_model_loaded(self) -> bool:
        if self.model_manager.is_model_loaded():
            return True
        self._load_yolo_model()
        return self.model_manager.is_model_loaded()

    def _run_yolo_auto_label_current(self) -> None:
        if not self.current_image_path:
            QMessageBox.warning(self, "提示", "请先打开图片")
            return
        if not self._ensure_yolo_model_loaded():
            return
        self._start_yolo_auto_label([self.current_image_path])

    def _run_yolo_auto_label_all(self) -> None:
        if not self.image_list:
            QMessageBox.warning(self, "提示", "请先加载图片队列")
            return
        if not self._ensure_yolo_model_loaded():
            return
        self._start_yolo_auto_label(list(self.image_list))

    def _start_yolo_auto_label(self, image_paths: list[str]) -> None:
        if self._yolo_label_thread and self._yolo_label_thread.isRunning():
            QMessageBox.warning(self, "提示", "YOLO 自动标注正在进行")
            return

        self.yolo_progress_bar.setValue(0)
        self.yolo_status_label.setText(f"处理中: 0/{len(image_paths)}")
        self.yolo_current_btn.setEnabled(False)
        self.yolo_all_btn.setEnabled(False)

        self._yolo_label_thread = QThread()
        self._yolo_label_worker = YOLOAutoLabelWorker(
            self.model_manager,
            image_paths,
            conf=self.yolo_conf_spin.value(),
            iou=self.yolo_iou_spin.value(),
            max_det=self.yolo_max_det_spin.value(),
        )
        self._yolo_label_worker.moveToThread(self._yolo_label_thread)
        self._yolo_label_thread.started.connect(self._yolo_label_worker.run)
        self._yolo_label_worker.progress.connect(self._on_yolo_auto_label_progress)
        self._yolo_label_worker.finished.connect(self._on_yolo_auto_label_finished)
        self._yolo_label_worker.error.connect(self._on_yolo_auto_label_error)
        self._yolo_label_worker.finished.connect(self._yolo_label_thread.quit)
        self._yolo_label_worker.error.connect(self._yolo_label_thread.quit)
        self._yolo_label_thread.start()

    def _on_yolo_auto_label_progress(self, current: int, total: int, image_path: str) -> None:
        value = int(current / total * 100) if total else 0
        self.yolo_progress_bar.setValue(value)
        self.yolo_status_label.setText(f"处理中: {current}/{total} - {os.path.basename(image_path)}")

    def _on_yolo_auto_label_finished(self, results: dict) -> None:
        total_boxes = 0
        current_abs = os.path.normcase(os.path.abspath(self.current_image_path)) if self.current_image_path else ""
        current_shapes = None
        target_class_id = self._class_id_for_yolo_auto_label()
        if target_class_id is None:
            self.yolo_status_label.setText("已取消: 需要先创建类别")
            self.yolo_current_btn.setEnabled(True)
            self.yolo_all_btn.setEnabled(True)
            return
        for image_path, detections in results.items():
            image_abs = os.path.normcase(os.path.abspath(image_path))
            shapes = self._detections_to_shapes(detections, target_class_id)
            total_boxes += len(shapes)
            if not self.yolo_replace_check.isChecked():
                if current_abs and image_abs == current_abs:
                    shapes = self.canvas.get_shapes() + shapes
                else:
                    width, height = self._read_image_size(image_path)
                    if width > 0 and height > 0:
                        shapes = load_yolo_shapes(image_path, width, height, self.class_manager) + shapes
            try:
                width, height = self._image_size_for_save(image_path)
            except ValueError:
                logger.warning(f"跳过无法读取尺寸的图片: {image_path}")
                continue
            save_yolo_shapes(image_path, shapes, width, height)
            if current_abs and image_abs == current_abs:
                current_shapes = shapes

        self.class_manager.save()
        self.canvas.set_classes(self.class_manager.get_all_classes())
        self.class_panel.refresh_list()
        self._refresh_class_quick_buttons()
        if current_shapes is not None:
            self.canvas.set_shapes(current_shapes)
            self.annot_list.refresh(current_shapes)
            self._set_dirty(False)
            self.file_list.load_image_list(self.image_list)
        self.file_list.highlight_current(self.current_image_index)
        self.yolo_progress_bar.setValue(100)
        self.yolo_status_label.setText(f"完成: {len(results)} 张图片 / {total_boxes} 个框")
        self.yolo_current_btn.setEnabled(True)
        self.yolo_all_btn.setEnabled(True)
        self._maybe_offer_training_after_annotation()

    def _on_yolo_auto_label_error(self, error_msg: str) -> None:
        self.yolo_status_label.setText("标注失败")
        self.yolo_current_btn.setEnabled(True)
        self.yolo_all_btn.setEnabled(True)
        QMessageBox.critical(self, "YOLO 自动标注失败", error_msg)

    def _class_id_for_yolo_auto_label(self) -> Optional[int]:
        if self.yolo_model_class_check.isChecked():
            return -1
        classes = self.class_manager.get_all_classes()
        if classes:
            class_id = self.class_panel.get_current_class_id()
            if 0 <= class_id < len(classes):
                return class_id
            return 0
        return self._create_class_from_prompt("新建自动标注类别", "类别名称:")

    def _detections_to_shapes(self, detections: list, target_class_id: Optional[int] = None) -> list[dict]:
        """Convert YOLO detections to canvas shape dicts.

        Supports bbox, OBB (rotated box), and keypoint (pose) detections.
        """
        shapes = []
        for detection in detections:
            det_type = detection.get("type", "bbox")

            # Resolve class_id and class_name
            if self.yolo_model_class_check.isChecked():
                model_class_id = int(detection.get("class_id", 0))
                raw_class_name = detection.get("class_name") or self._model_class_name(model_class_id)
                class_name = self.class_manager.map_class_name(str(raw_class_name)) if raw_class_name else None
                if class_name:
                    class_id = self.class_manager.get_or_create_class(str(class_name))
                else:
                    if target_class_id is None or target_class_id < 0:
                        continue
                    class_id = target_class_id
                    class_name = self.class_manager.get_class_name(class_id)
                    if class_name is None:
                        continue
            else:
                if target_class_id is None or target_class_id < 0:
                    continue
                class_id = target_class_id
                class_name = self.class_manager.get_class_name(class_id)
                if class_name is None:
                    continue

            confidence = float(detection.get("confidence", 0.0))

            # --- OBB shape ---
            if det_type == "obb":
                corners = detection.get("corners")
                if corners and len(corners) == 4:
                    shapes.append({
                        "type": ShapeType.OBB,
                        "class_id": class_id,
                        "class_name": class_name,
                        "data": {"corners": [(float(c[0]), float(c[1])) for c in corners]},
                        "confidence": confidence,
                    })
                continue

            # --- Keypoint / Pose shape ---
            if det_type == "keypoint":
                bbox = detection.get("bbox", [])
                if isinstance(bbox, dict):
                    x1, y1, x2, y2 = bbox.get("x1"), bbox.get("y1"), bbox.get("x2"), bbox.get("y2")
                elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                    x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                else:
                    continue
                shape_data = {
                    "x1": float(x1), "y1": float(y1),
                    "x2": float(x2), "y2": float(y2),
                    "keypoints": [],
                }
                # Add keypoints (pixel coords)
                for kp in detection.get("keypoints", []):
                    if isinstance(kp, (list, tuple)) and len(kp) >= 3:
                        shape_data["keypoints"].append((float(kp[0]), float(kp[1]), int(kp[2])))
                    elif isinstance(kp, (list, tuple)) and len(kp) >= 2:
                        shape_data["keypoints"].append((float(kp[0]), float(kp[1]), 2))
                shapes.append({
                    "type": ShapeType.KEYPOINT,
                    "class_id": class_id,
                    "class_name": class_name,
                    "data": shape_data,
                    "confidence": confidence,
                })
                continue

            # --- Standard BBox shape ---
            bbox = detection.get("bbox", [])
            if isinstance(bbox, dict):
                x1, y1, x2, y2 = bbox.get("x1"), bbox.get("y1"), bbox.get("x2"), bbox.get("y2")
            elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            else:
                continue
            shapes.append({
                "type": ShapeType.BBOX,
                "class_id": class_id,
                "class_name": class_name,
                "data": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)},
                "confidence": confidence,
            })
        return shapes

    def _model_class_name(self, class_id: int) -> Optional[str]:
        model = self.model_manager.get_current_model()
        names = getattr(model, "names", None)
        if isinstance(names, dict):
            return str(names.get(class_id)) if class_id in names else None
        if isinstance(names, list) and 0 <= class_id < len(names):
            return str(names[class_id])
        return self.class_manager.get_class_name(class_id)

    def _show_class_name_map_dialog(self) -> None:
        """Show dialog to edit model class name → project class name mapping."""
        from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem, QDialogButtonBox, QHeaderView

        dlg = QDialog(self)
        dlg.setWindowTitle("类别名映射表 (模型英文名 → 项目中文名)")
        dlg.setMinimumSize(560, 480)
        layout = QVBoxLayout(dlg)

        # Info label
        info = QLabel("编辑模型类别名到项目类别名的映射。自动标注时，模型返回的类别名会先查此表翻译。\n例如: person → 人, car → 汽车")
        info.setWordWrap(True)
        info.setObjectName("MutedText")
        layout.addWidget(info)

        # If a model is loaded, offer to import its class names
        model = self.model_manager.get_current_model()
        if model is not None:
            model_names = getattr(model, "names", None)
            if isinstance(model_names, dict) and model_names:
                import_row = QHBoxLayout()
                import_row.addStretch()
                import_btn = QPushButton("从当前模型导入类别名")
                import_btn.setToolTip(f"当前模型有 {len(model_names)} 个类别，点击将缺失的类别名添加到映射表")
                import_btn.clicked.connect(lambda: self._import_model_names_to_map(model_names, table))
                import_row.addWidget(import_btn)
                layout.addLayout(import_row)

        # Table
        name_map = self.class_manager.get_name_map()
        table = QTableWidget(len(name_map), 2)
        table.setHorizontalHeaderLabels(["模型类别名 (英文)", "项目类别名 (中文)"])
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        table.setAlternatingRowColors(True)

        for row, (model_name, project_name) in enumerate(sorted(name_map.items())):
            table.setItem(row, 0, QTableWidgetItem(model_name))
            table.setItem(row, 1, QTableWidgetItem(project_name))

        layout.addWidget(table)

        # Add / Remove row buttons
        edit_row = QHBoxLayout()
        add_btn = QPushButton("添加映射")
        add_btn.clicked.connect(lambda: table.insertRow(table.rowCount()))
        edit_row.addWidget(add_btn)
        remove_btn = QPushButton("删除选中")
        remove_btn.clicked.connect(lambda: table.removeRow(table.currentRow()))
        edit_row.addWidget(remove_btn)
        edit_row.addStretch()
        layout.addLayout(edit_row)

        # OK / Cancel
        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btn_box.accepted.connect(dlg.accept)
        btn_box.rejected.connect(dlg.reject)
        layout.addWidget(btn_box)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            # Collect mapping from table
            new_map = {}
            for row in range(table.rowCount()):
                key_item = table.item(row, 0)
                val_item = table.item(row, 1)
                if key_item and val_item:
                    key = key_item.text().strip()
                    val = val_item.text().strip()
                    if key and val:
                        new_map[key] = val
            self.class_manager.name_map = new_map
            self.class_manager._save_name_map()
            logger.info(f"Updated name map with {len(new_map)} entries")

    def _import_model_names_to_map(self, model_names: dict, table: "QTableWidget") -> None:
        """Import class names from the current model into the mapping table."""
        from core.class_manager import COCO_EN_ZH_MAP
        from PyQt6.QtWidgets import QTableWidgetItem

        existing = set()
        for row in range(table.rowCount()):
            item = table.item(row, 0)
            if item:
                existing.add(item.text().strip())

        added = 0
        for cls_id, cls_name in model_names.items():
            cls_name = str(cls_name)
            if cls_name not in existing:
                row = table.rowCount()
                table.insertRow(row)
                # Translate using COCO map if available
                zh_name = COCO_EN_ZH_MAP.get(cls_name, cls_name)
                table.setItem(row, 0, QTableWidgetItem(cls_name))
                table.setItem(row, 1, QTableWidgetItem(zh_name))
                existing.add(cls_name)
                added += 1

        if added > 0:
            QMessageBox.information(self, "导入完成", f"已添加 {added} 个新映射")
        else:
            QMessageBox.information(self, "导入完成", "所有模型类别名已在映射表中")

    def _read_image_size(self, image_path: str) -> tuple[int, int]:
        reader = QImageReader(image_path)
        size = reader.size()
        if size.isValid():
            return size.width(), size.height()
        return 0, 0

    def _image_size_for_save(self, image_path: str) -> tuple[int, int]:
        if image_path == self.current_image_path and self.canvas.image_width and self.canvas.image_height:
            return self.canvas.image_width, self.canvas.image_height
        width, height = self._read_image_size(image_path)
        if width <= 0 or height <= 0:
            raise ValueError(f"无法读取图片尺寸: {image_path}")
        return width, height

    # ------------------------------------------------------------------
    # Tool entry points
    # ------------------------------------------------------------------

    def _focus_auto_labeling_panel(self) -> None:
        self._switch_workspace(0)
        self.yolo_status_label.setText("请先加载模型")
        self.yolo_model_combo.setFocus()

    def _show_yolo_tools_dialog(self) -> None:
        self._switch_workspace(0)
        self.yolo_model_combo.setFocus()
        self.statusBar().showMessage("YOLO 自动标注位于顶部参数栏", 2500)

    # ------------------------------------------------------------------
    # LLM auto-labeling
    # ------------------------------------------------------------------

        if not self.current_image_path or self.canvas.original_image is None:
            QMessageBox.warning(self, "提示", "请先打开一张图片")
            return None, None
            return None, None

            return

    def _run_llm_auto_label(self):
        if not self._is_llm_detect_project():
            return
        if not self.current_image_path:
            QMessageBox.warning(self, "提示", "请先打开图片")
            return

        class_info = self._current_llm_class()
        if class_info is None:
            return
        _class_id, target_class = class_info

        llm_config = load_llm_config()
        if not llm_config.get("base_url"):
            QMessageBox.warning(self, "提示", "请先在 LLM 配置中设置 Base URL")
            self._show_auto_label_dialog_llm()
            return

        self.statusBar().showMessage(f"LLM 推理中 (检测: {target_class})...", 0)
        self._llm_worker = LLMInferenceWorker(
            self.current_image_path, target_class, llm_config
        )
        self._llm_worker.finished.connect(self._on_llm_result)
        self._llm_worker.error.connect(self._on_llm_error)
        self._llm_worker.start()

    def _on_llm_result(self, detections):
        self.statusBar().showMessage(f"LLM 完成: {len(detections)} 个检测", 3000)
        if not detections:
            QMessageBox.information(self, "完成", "未检测到目标")
            return

        img_w = self.canvas.image_width
        img_h = self.canvas.image_height
        if img_w <= 0 or img_h <= 0:
            return

        class_info = self._current_llm_class()
        if class_info is None:
            return
        class_id, class_name = class_info

        existing_count = len(self.canvas.get_shapes())
        self.canvas.push_undo()
        shapes = list(self.canvas.get_shapes())
        shapes.extend(self._llm_detections_to_shapes(detections, img_w, img_h, class_id, class_name))
        if len(shapes) == existing_count:
            QMessageBox.information(self, "完成", "未生成有效标注")
            return
        self.canvas.set_shapes(shapes)
        self.class_manager.save()
        self.class_panel.refresh_list()
        self.canvas.set_classes(self.class_manager.get_all_classes())
        self._refresh_class_quick_buttons()
        self.annot_list.refresh(shapes)
        self._set_dirty(True)
        self._autosave_annotations()
        self.file_list.load_image_list(self.image_list)
        self.file_list.highlight_current(self.current_image_index)

    def _run_llm_auto_label_batch(self):
        if not self._is_llm_detect_project():
            return
        if not self.image_list:
            QMessageBox.warning(self, "提示", "项目中没有图片")
            return
        if self._llm_batch_worker is not None and self._llm_batch_worker.isRunning():
            QMessageBox.warning(self, "提示", "LLM 批量推理正在进行")
            return

        class_info = self._current_llm_class()
        if class_info is None:
            return
        class_id, target_class = class_info

        llm_config = load_llm_config()
        if not llm_config.get("base_url"):
            QMessageBox.warning(self, "提示", "请先在 LLM 配置中设置 Base URL")
            self._show_auto_label_dialog_llm()
            return

        self._llm_batch_class_id = class_id
        self._llm_batch_class_name = target_class
        self._llm_progress_dialog = QProgressDialog("正在使用 LLM 进行批量检测...", "取消", 0, len(self.image_list), self)
        self._llm_progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self._llm_progress_dialog.setMinimumDuration(0)
        self._llm_progress_dialog.setValue(0)

        self._llm_batch_worker = LLMBatchInferenceWorker(list(self.image_list), target_class, llm_config)
        self._llm_progress_dialog.canceled.connect(self._llm_batch_worker.stop)
        self._llm_batch_worker.progress.connect(self._on_llm_batch_progress)
        self._llm_batch_worker.finished.connect(self._on_llm_batch_finished)
        self._llm_batch_worker.error.connect(self._on_llm_batch_error)
        self._llm_batch_worker.start()

    def _on_llm_batch_progress(self, current: int, total: int, image_path: str) -> None:
        if self._llm_progress_dialog:
            self._llm_progress_dialog.setMaximum(total)
            self._llm_progress_dialog.setValue(max(0, current - 1))
            self._llm_progress_dialog.setLabelText(
                f"正在处理: {os.path.basename(image_path)} ({current}/{total})"
            )

    def _on_llm_batch_finished(self, results: dict) -> None:
        if self._llm_progress_dialog:
            self._llm_progress_dialog.setValue(self._llm_progress_dialog.maximum())
            self._llm_progress_dialog.close()
            self._llm_progress_dialog = None

        total_added = 0
        processed = 0
        for image_path, detections in results.items():
            width, height = self._image_size_for_path(image_path)
            if width <= 0 or height <= 0:
                continue
            processed += 1
            new_shapes = self._llm_detections_to_shapes(
                detections,
                width,
                height,
                self._llm_batch_class_id,
                self._llm_batch_class_name,
            )
            if not new_shapes:
                continue
            existing = load_yolo_shapes(image_path, width, height, self.class_manager)
            save_yolo_shapes(image_path, existing + new_shapes, width, height)
            total_added += len(new_shapes)

        self.file_list.load_image_list(self.image_list)
        self.file_list.highlight_current(self.current_image_index)
        if self.current_image_path in results:
            self._load_annotations_for_image(self.current_image_path)
        self.statusBar().showMessage(f"LLM 批量完成: 处理 {processed} 张，添加 {total_added} 个标注", 5000)
        QMessageBox.information(self, "完成", f"批量推理完成\n处理了 {processed} 张图片\n共添加 {total_added} 个标注")

    def _on_llm_batch_error(self, error_msg: str) -> None:
        if self._llm_progress_dialog:
            self._llm_progress_dialog.close()
            self._llm_progress_dialog = None
        self._on_llm_error(error_msg)

    def _is_llm_detect_project(self) -> bool:
        if not self.current_project:
            QMessageBox.warning(self, "提示", "请先选择一个项目")
            return False
        if self.current_project.get("task", "detect") != "detect":
            QMessageBox.information(self, "提示", "功能还在完善，敬请期待")
            return False
        return True

    def _current_llm_class(self) -> Optional[tuple[int, str]]:
        classes = self.class_manager.get_all_classes()
        if not classes:
            QMessageBox.warning(self, "提示", "请先创建类别")
            return None
        class_id = self.class_panel.get_current_class_id() if hasattr(self, "class_panel") else self.canvas.current_class_id
        if not 0 <= class_id < len(classes):
            class_id = 0
        return class_id, classes[class_id]

    @staticmethod
    def _llm_detections_to_shapes(
        detections,
        image_width: int,
        image_height: int,
        class_id: int,
        class_name: str,
    ) -> list[dict]:
        shapes = []
        for _label, x1, y1, x2, y2 in detections:
            if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.0:
                abs_x1 = int(x1 * image_width)
                abs_y1 = int(y1 * image_height)
                abs_x2 = int(x2 * image_width)
                abs_y2 = int(y2 * image_height)
            else:
                abs_x1, abs_y1, abs_x2, abs_y2 = int(x1), int(y1), int(x2), int(y2)
            abs_x1 = max(0, min(abs_x1, image_width))
            abs_y1 = max(0, min(abs_y1, image_height))
            abs_x2 = max(0, min(abs_x2, image_width))
            abs_y2 = max(0, min(abs_y2, image_height))
            abs_x1, abs_x2 = sorted((abs_x1, abs_x2))
            abs_y1, abs_y2 = sorted((abs_y1, abs_y2))
            if abs_x2 - abs_x1 < 2 or abs_y2 - abs_y1 < 2:
                continue
            shapes.append({
                "type": ShapeType.BBOX,
                "class_id": class_id,
                "class_name": class_name,
                "data": {"x1": abs_x1, "y1": abs_y1, "x2": abs_x2, "y2": abs_y2},
            })
        return shapes

    def _on_llm_error(self, error_msg: str):
        self.statusBar().showMessage(f"LLM 错误: {error_msg}", 5000)
        QMessageBox.critical(self, "LLM 错误", error_msg)

    # ------------------------------------------------------------------
    # Auto-label settings dialogs
    # ------------------------------------------------------------------

    def _show_auto_label_dialog_llm(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("LLM 自动标注配置")
        dlg.setMinimumWidth(600)
        dlg.setModal(True)
        layout = QVBoxLayout(dlg)

        llm_config = load_llm_config()

        # Preset selector
        preset_row = QHBoxLayout()
        preset_label = QLabel("API 预设:")
        preset_combo = QComboBox()
        preset_combo.addItems(["自定义", "阿里云通义千问", "DeepSeek", "Ollama (本地)"])
        preset_map = {
            "阿里云通义千问": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-vl-max"),
            "DeepSeek": ("https://api.deepseek.com/v1", "deepseek-chat"),
            "Ollama (本地)": ("http://localhost:11434/v1", "llava"),
        }
        
        # Detect current preset
        current_base = llm_config.get("base_url", "")
        current_model = llm_config.get("model_name", "")
        selected_preset = "自定义"
        for name, (base, model) in preset_map.items():
            if current_base == base and current_model == model:
                selected_preset = name
                break
        preset_combo.setCurrentText(selected_preset)
        
        preset_row.addWidget(preset_label)
        preset_row.addWidget(preset_combo)
        preset_row.addStretch()
        layout.addLayout(preset_row)

        form = QFormLayout()
        api_key_edit = QLineEdit(llm_config.get("api_key", ""))
        api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow("API Key:", api_key_edit)

        base_url_edit = QLineEdit(llm_config.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
        form.addRow("Base URL:", base_url_edit)

        model_name_edit = QLineEdit(llm_config.get("model_name", "qwen-vl-max"))
        form.addRow("模型名称:", model_name_edit)

        def on_preset_changed(index):
            text = preset_combo.currentText()
            if text in preset_map:
                base, model = preset_map[text]
                base_url_edit.setText(base)
                model_name_edit.setText(model)
        
        preset_combo.currentIndexChanged.connect(on_preset_changed)

        sys_prompt_edit = QTextEdit()
        sys_prompt_edit.setPlainText(llm_config.get("system_prompt", ""))
        sys_prompt_edit.setMaximumHeight(80)
        form.addRow("系统提示词:", sys_prompt_edit)

        user_prompt_edit = QTextEdit()
        user_prompt_edit.setPlainText(llm_config.get("user_prompt", ""))
        user_prompt_edit.setMaximumHeight(80)
        form.addRow("用户提示词:", user_prompt_edit)
        layout.addLayout(form)

        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.accepted.connect(dlg.accept)
        btn_box.rejected.connect(dlg.reject)
        layout.addWidget(btn_box)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            new_config = {
                "api_key": api_key_edit.text(),
                "base_url": base_url_edit.text(),
                "model_name": model_name_edit.text(),
                "system_prompt": sys_prompt_edit.toPlainText(),
                "user_prompt": user_prompt_edit.toPlainText(),
            }
            save_llm_config(new_config)

    # ------------------------------------------------------------------
    # Negative sample toggle
    # ------------------------------------------------------------------

    def _toggle_negative_sample(self, checked: bool):
        if not self.current_image_path:
            self.negative_btn.setChecked(False)
            return
        label_path = label_path_for_image(self.current_image_path)
        if checked:
            label_dir = os.path.dirname(label_path)
            if label_dir:
                os.makedirs(label_dir, exist_ok=True)
            Path(label_path).write_text("", encoding="utf-8")
            self.statusBar().showMessage("已标记为负样本(无目标)", 2000)
        else:
            if os.path.isfile(label_path):
                content = Path(label_path).read_text(encoding="utf-8").strip()
                if not content:
                    os.remove(label_path)
                    self.statusBar().showMessage("已取消负样本标记", 2000)
        self.file_list.load_image_list(self.image_list)
        self.file_list.highlight_current(self.current_image_index)

    def _show_env_check(self) -> None:
        try:
            from gui.env_check_dialog import EnvironmentCheckDialog

            dialog = EnvironmentCheckDialog(self)
            dialog.exec()
        except Exception as exc:
            QMessageBox.critical(self, "错误", f"无法打开环境检测:\n{exc}")

    def _show_video_capture(self) -> None:
        if not self.current_project:
            QMessageBox.warning(self, "需要项目", "请先新建或导入项目，再进行视频截帧。")
            self._switch_workspace(6)
            return
        try:
            from gui.video_capture_dialog import VideoCaptureDialog

            dialog = VideoCaptureDialog(self)
            output_dir = Path(self.current_project["root"]) / "images" / "video_frames"
            output_dir.mkdir(parents=True, exist_ok=True)
            dialog.output_edit.setText(str(output_dir))
            dialog.frames_captured.connect(self._load_captured_frames)
            dialog.exec()
        except Exception as exc:
            QMessageBox.critical(self, "错误", f"无法打开视频截帧:\n{exc}")

    def _load_captured_frames(self, paths: list[str]) -> None:
        valid_paths = [
            os.path.abspath(path)
            for path in paths
            if path and os.path.isfile(path) and os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS
        ]
        if not valid_paths:
            QMessageBox.warning(self, "提示", "没有可加载的截帧图片")
            return

        if not self.current_project:
            QMessageBox.warning(self, "需要项目", "请先新建或导入项目")
            self._switch_workspace(6)
            return

        images_root = (Path(self.current_project["root"]) / "images").resolve()
        external_paths: list[str] = []
        for path in valid_paths:
            frame_path = Path(path).resolve()
            try:
                frame_path.relative_to(images_root)
            except ValueError:
                external_paths.append(str(frame_path))

        if external_paths:
            ProjectManager().import_images(self.current_project, external_paths)
        self.current_project = ProjectManager().open_project(self.current_project["root"])
        self._on_project_opened(self.current_project)
        self.statusBar().showMessage(f"已导入 {len(valid_paths)} 帧到当前项目", 3000)

    def _show_format_conversion(self) -> None:
        try:
            from gui.format_conversion_dialog import FormatConversionDialog

            dialog = FormatConversionDialog(class_manager=self.class_manager, parent=self)
            dialog.exec()
        except Exception as exc:
            QMessageBox.critical(self, "错误", f"无法打开格式转换:\n{exc}")

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "关于 YOLO Studio",
            "YOLO Studio v1.0.0\n\n"
            "用于数据集标注、模型训练、推理和导出的桌面工作台。\n\n"
            "核心能力:\n"
            "- YOLO 数据集标注\n"
            "- 训练、验证、推理与模型导出\n",
        )

    def _on_trained_model_ready(self, best_pt: str, action: str) -> None:
        if hasattr(self, "results_panel"):
            self.results_panel.refresh_runs()
        if action == "infer":
            self._switch_workspace(2)
            self.inference_panel.load_model_from_path(best_pt)
        elif action == "export":
            self._switch_workspace(4)
            self.export_panel.load_model_from_path(best_pt)
        elif action == "annotate":
            self._switch_workspace(0)
            self.model_manager.load_model(best_pt)
            self.model_pill.setText(os.path.basename(best_pt))

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def _set_dirty(self, dirty: bool) -> None:
        self._dirty = dirty
        self.dirty_pill.setText("未保存" if dirty else "已保存")
        self.dirty_pill.set_variant("warning" if dirty else "success")
        if hasattr(self, "dirty_badge"):
            self.dirty_badge.setText("未保存" if dirty else "已保存")
            self.dirty_badge.set_variant("warning" if dirty else "success")

    def _update_status(self) -> None:
        if self.current_image_path:
            filename = os.path.basename(self.current_image_path)
            index = self.current_image_index + 1
            total = len(self.image_list)
            self.status_image_label.setText(f"{filename} [{index}/{total}]")
            self.image_pill.setText(f"{index}/{total} 张图片")
            if hasattr(self, "queue_counter_label"):
                self.queue_counter_label.setText(f"{index} / {total}")
        else:
            self.status_image_label.setText("无图片")
            self.image_pill.setText("无图片")
            if hasattr(self, "queue_counter_label"):
                self.queue_counter_label.setText("0 / 0")

    # ------------------------------------------------------------------
    # Window events
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        if hasattr(self, "project_panel"):
            self.project_panel.shutdown()
        if self._llm_batch_worker and self._llm_batch_worker.isRunning():
            self._llm_batch_worker.stop()
            self._llm_batch_worker.quit()
            self._llm_batch_worker.wait(3000)
        if self._llm_worker and self._llm_worker.isRunning():
            self._llm_worker.quit()
            self._llm_worker.wait(3000)
        if self._yolo_label_thread and self._yolo_label_thread.isRunning():
            self._yolo_label_thread.quit()
            self._yolo_label_thread.wait(3000)
        self.model_manager.clear_cache()
        logger.info("MainWindow closed")
        event.accept()

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key in (Qt.Key.Key_Left, Qt.Key.Key_A):
            self._prev_image()
            return
        if key in (Qt.Key.Key_Right, Qt.Key.Key_D):
            self._next_image()
            return
        if key in (Qt.Key.Key_V, Qt.Key.Key_1):
            self._set_canvas_mode(CanvasMode.EDIT)
            return
        if key in (Qt.Key.Key_R, Qt.Key.Key_2):
            self._set_canvas_mode(CanvasMode.CREATE_BBOX)
            return
        if key in (Qt.Key.Key_P, Qt.Key.Key_3):
            self._set_canvas_mode(CanvasMode.CREATE_POLYGON)
            return
        if key in (Qt.Key.Key_O, Qt.Key.Key_4):
            self._set_canvas_mode(CanvasMode.CREATE_OBB)
            return
        if key in (Qt.Key.Key_K, Qt.Key.Key_5):
            self._set_canvas_mode(CanvasMode.CREATE_KEYPOINT)
            return
        if key == Qt.Key.Key_Delete:
            self._delete_selected_shape()
            return
        super().keyPressEvent(event)
