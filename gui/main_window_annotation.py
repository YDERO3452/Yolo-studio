"""Annotation workbench UI and auto-label/LLM handlers for MainWindow."""

from __future__ import annotations

import os
import traceback
from pathlib import Path
from typing import Optional

from loguru import logger
from PyQt6.QtCore import Qt, QThread
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QProgressBar,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.annotation import ShapeType
from core.dataset import DatasetManager
from core.image_utils import read_image_size
from gui.annotation_io import (
    label_path_for_image,
    labels_dir_for_image_dir,
    load_yolo_shapes,
    save_yolo_shapes,
)
from gui.annotation_list_widget import AnnotationListWidget
from gui.canvas import AnnotationCanvas, CanvasMode
from gui.class_panel import ClassListPanel
from gui.file_list_widget import FileListWidget
from gui.llm_handler import LLMBatchInferenceWorker, LLMInferenceWorker, load_llm_config, save_llm_config
from gui.theme import Theme
from gui.ui_components import StatusPill
from gui.yolo_label_worker import YOLOAutoLabelWorker

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


class AnnotationWorkbenchMixin:
    def _create_tool_rail(self) -> QWidget:
        """Left tool rail for annotation modes only (single-char labels)."""
        rail = QWidget()
        rail.setObjectName("ToolRail")
        rail.setFixedWidth(36)
        layout = QVBoxLayout(rail)
        layout.setContentsMargins(3, 6, 3, 6)
        layout.setSpacing(2)

        self.annotation_tools_container = QWidget()
        tools_layout = QVBoxLayout(self.annotation_tools_container)
        tools_layout.setContentsMargins(0, 0, 0, 0)
        tools_layout.setSpacing(2)

        self.mode_actions: dict[CanvasMode, QPushButton] = {}
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        for mode, label, tip in [
            (CanvasMode.EDIT, "选", "选择并编辑标注 (V)"),
            (CanvasMode.CREATE_BBOX, "框", "绘制矩形框 (R)"),
            (CanvasMode.CREATE_POLYGON, "多", "绘制多边形 (P)"),
            (CanvasMode.CREATE_OBB, "旋", "绘制旋转框 (O)"),
            (CanvasMode.CREATE_KEYPOINT, "点", "创建关键点 (K)"),
        ]:
            btn = QPushButton(label)
            btn.setObjectName("ToolButton")
            btn.setCheckable(True)
            btn.setToolTip(tip)
            btn.setFixedSize(30, 28)
            btn.clicked.connect(lambda checked=False, m=mode: self._set_canvas_mode(m))
            self.mode_group.addButton(btn)
            self.mode_actions[mode] = btn
            tools_layout.addWidget(btn)

        tools_layout.addSpacing(2)
        fit_btn = QPushButton("适")
        fit_btn.setObjectName("ToolButton")
        fit_btn.setToolTip("适配图片到视口 (F)")
        fit_btn.setFixedSize(30, 28)
        fit_btn.clicked.connect(self.canvas.fit_to_window)
        tools_layout.addWidget(fit_btn)

        # Kept for existing refresh logic; chips stay hidden.
        self.class_quick_container = QWidget()
        self.class_quick_container.setVisible(False)
        self.class_quick_layout = QVBoxLayout(self.class_quick_container)
        self.class_quick_layout.setContentsMargins(0, 0, 0, 0)
        self.class_quick_layout.setSpacing(2)
        tools_layout.addWidget(self.class_quick_container)

        layout.addWidget(self.annotation_tools_container)
        self.mode_actions[CanvasMode.CREATE_BBOX].setChecked(True)
        layout.addStretch()
        return rail

    def _create_annotation_workspace(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.canvas = AnnotationCanvas()
        self.canvas.set_classes(self.class_manager.get_all_classes())

        # Brush rail lives inside the node sheet, not as a window chrome.
        layout.addWidget(self._create_tool_rail())

        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)

        self.scroll_area = QScrollArea()
        self.scroll_area.setObjectName("CanvasScrollArea")
        self.scroll_area.setWidget(self.canvas)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.scroll_area.viewport().installEventFilter(self)
        center_layout.addWidget(self.scroll_area, stretch=1)

        center.setMinimumWidth(640)
        self.annotation_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.annotation_splitter.setChildrenCollapsible(False)
        self.annotation_splitter.setHandleWidth(1)
        self.annotation_splitter.addWidget(center)
        self.annotation_splitter.addWidget(self._create_inspector_v2())
        self.annotation_splitter.setStretchFactor(0, 1)
        self.annotation_splitter.setStretchFactor(1, 0)
        self.annotation_splitter.setSizes([1100, 280])
        layout.addWidget(self.annotation_splitter, stretch=1)
        return page

    def _set_model_status(self, text: str, ok: bool | None = None) -> None:
        """Update model status label; ok=True green, False red, None muted."""
        self.yolo_status_label.setText(text)
        if ok is True:
            color = Theme.SUCCESS
        elif ok is False:
            color = Theme.DANGER
        else:
            color = Theme.TEXT_MUTED
        self.yolo_status_label.setStyleSheet(f"color: {color}; font-size: 12px;")

    def _create_annotation_control_bar_v2(self) -> QWidget:
        """Auto-label controls for the inspector (not a window top toolbar)."""
        panel = QWidget()
        panel.setObjectName("AutoLabelPanel")
        col = QVBoxLayout(panel)
        col.setContentsMargins(4, 6, 4, 6)
        col.setSpacing(8)

        file_row = QHBoxLayout()
        open_btn = QPushButton("打开")
        open_btn.setFixedHeight(28)
        open_btn.clicked.connect(self._open_image_dir)
        save_btn = QPushButton("保存")
        save_btn.setFixedHeight(28)
        save_btn.clicked.connect(self._save_annotations)
        file_row.addWidget(open_btn)
        file_row.addWidget(save_btn)
        col.addLayout(file_row)

        self.yolo_model_combo = QComboBox()
        self.yolo_model_combo.setEditable(True)
        self.yolo_model_combo.setFixedHeight(28)
        self.yolo_model_combo.setToolTip("自动标注模型")
        self._populate_yolo_models()
        col.addWidget(self.yolo_model_combo)

        model_row = QHBoxLayout()
        browse_btn = QPushButton("浏览")
        browse_btn.setFixedHeight(28)
        browse_btn.clicked.connect(self._browse_yolo_model)
        self.yolo_load_btn = QPushButton("加载模型")
        self.yolo_load_btn.setFixedHeight(28)
        self.yolo_load_btn.clicked.connect(self._load_yolo_model)
        model_row.addWidget(browse_btn)
        model_row.addWidget(self.yolo_load_btn)
        col.addLayout(model_row)

        self.yolo_status_label = QLabel("未加载")
        self.yolo_status_label.setObjectName("MutedText")
        self._set_model_status("未加载", None)
        col.addWidget(self.yolo_status_label)

        run_row = QHBoxLayout()
        self.yolo_current_btn = QPushButton("标注当前")
        self.yolo_current_btn.setObjectName("PrimaryButton")
        self.yolo_current_btn.setFixedHeight(28)
        self.yolo_current_btn.clicked.connect(self._run_yolo_auto_label_current)
        self.yolo_all_btn = QPushButton("标注全部")
        self.yolo_all_btn.setFixedHeight(28)
        self.yolo_all_btn.clicked.connect(self._run_yolo_auto_label_all)
        run_row.addWidget(self.yolo_current_btn)
        run_row.addWidget(self.yolo_all_btn)
        col.addLayout(run_row)

        self.yolo_progress_bar = QProgressBar()
        self.yolo_progress_bar.setRange(0, 100)
        self.yolo_progress_bar.setValue(0)
        self.yolo_progress_bar.setTextVisible(True)
        self.yolo_progress_bar.setVisible(False)
        col.addWidget(self.yolo_progress_bar)

        extra_row = QHBoxLayout()
        self.llm_btn = QPushButton("LLM")
        self.llm_btn.setFixedHeight(28)
        llm_menu = QMenu(self.llm_btn)
        llm_menu.addAction("单张推理", self._run_llm_auto_label)
        llm_menu.addAction("批量推理", self._run_llm_auto_label_batch)
        llm_menu.addSeparator()
        llm_menu.addAction("自由检测 (自动发现类别)", self._run_llm_free_detect)
        llm_menu.addSeparator()
        llm_menu.addAction("设置", self._show_auto_label_dialog_llm)
        self.llm_btn.setMenu(llm_menu)
        extra_row.addWidget(self.llm_btn)

        self.negative_btn = QPushButton("无框")
        self.negative_btn.setCheckable(True)
        self.negative_btn.setFixedHeight(28)
        self.negative_btn.toggled.connect(self._toggle_negative_sample)
        extra_row.addWidget(self.negative_btn)
        col.addLayout(extra_row)

        params_host = QWidget()
        params_layout = QFormLayout(params_host)
        params_layout.setContentsMargins(0, 4, 0, 0)
        params_layout.setSpacing(6)

        self.yolo_conf_spin = QDoubleSpinBox()
        self.yolo_conf_spin.setRange(0.01, 1.0)
        self.yolo_conf_spin.setDecimals(2)
        self.yolo_conf_spin.setSingleStep(0.05)
        self.yolo_conf_spin.setValue(float(self.config_manager.get("inference", "conf", 0.25)))
        self.yolo_conf_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        params_layout.addRow("置信度", self.yolo_conf_spin)

        self.yolo_iou_spin = QDoubleSpinBox()
        self.yolo_iou_spin.setRange(0.01, 1.0)
        self.yolo_iou_spin.setDecimals(2)
        self.yolo_iou_spin.setSingleStep(0.05)
        self.yolo_iou_spin.setValue(float(self.config_manager.get("inference", "iou", 0.7)))
        self.yolo_iou_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        params_layout.addRow("IOU", self.yolo_iou_spin)

        self.yolo_max_det_spin = QSpinBox()
        self.yolo_max_det_spin.setRange(1, 3000)
        self.yolo_max_det_spin.setValue(int(self.config_manager.get("inference", "max_det", 300)))
        self.yolo_max_det_spin.setVisible(False)

        self.yolo_replace_check = QCheckBox("覆盖旧标注")
        params_layout.addRow("", self.yolo_replace_check)

        self.yolo_model_class_check = QCheckBox("模型类别(中文)")
        self.yolo_model_class_check.setChecked(True)
        self.yolo_model_class_check.setVisible(False)

        col.addWidget(params_host)

        map_btn = QPushButton("类别名映射")
        map_btn.setFixedHeight(28)
        map_btn.clicked.connect(self._show_class_name_map_dialog)
        col.addWidget(map_btn)
        col.addStretch(1)
        return panel

    def _create_inspector_v2(self) -> QWidget:
        inspector = QWidget()
        inspector.setObjectName("Inspector")
        inspector.setMinimumWidth(252)
        inspector.setMaximumWidth(320)
        inspector.setAutoFillBackground(True)
        inspector.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        layout = QVBoxLayout(inspector)
        layout.setContentsMargins(10, 10, 10, 8)
        layout.setSpacing(8)

        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(6)
        title = QLabel("标签")
        title.setObjectName("PanelTitle")
        header_row.addWidget(title)
        header_row.addStretch()
        self.dirty_badge = QLabel("已保存")
        self.dirty_badge.setObjectName("MutedText")
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
            self.class_panel.add_btn.setFixedHeight(28)
        if hasattr(self.class_panel, "remove_btn"):
            self.class_panel.remove_btn.setFixedHeight(28)
        if hasattr(self.class_panel, "color_btn"):
            self.class_panel.color_btn.setFixedHeight(28)
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
        del_btn.setFixedHeight(28)
        del_btn.clicked.connect(self._delete_selected_shape)
        edit_btn = QPushButton("编辑")
        edit_btn.setFixedHeight(28)
        edit_btn.clicked.connect(self._delete_or_edit_selected_label)
        clr_btn = QPushButton("清空")
        clr_btn.setFixedHeight(28)
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
        self.file_search.setFixedHeight(28)
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
        prev_btn.setFixedSize(28, 28)
        prev_btn.setToolTip("上一张 (A/←)")
        prev_btn.clicked.connect(self._prev_image)
        next_btn = QPushButton("▶")
        next_btn.setFixedSize(28, 28)
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
        save_btn.setFixedHeight(30)
        save_btn.clicked.connect(self._save_annotations)
        import_btn = QPushButton("导入")
        import_btn.setFixedHeight(30)
        import_btn.clicked.connect(self._open_image_dir)
        main_actions.addWidget(save_btn, stretch=1)
        main_actions.addWidget(import_btn, stretch=1)
        queue_layout.addLayout(main_actions)

        dataset_btn = QPushButton("生成训练数据集")
        dataset_btn.setObjectName("DatasetButton")
        dataset_btn.setFixedHeight(30)
        dataset_btn.clicked.connect(self._generate_training_dataset_from_queue)
        queue_layout.addWidget(dataset_btn)

        export_row = QHBoxLayout()
        export_row.setSpacing(4)
        export_row.addWidget(QLabel("导出:"))
        yolo_btn = QPushButton("YOLO")
        yolo_btn.setFixedHeight(28)
        yolo_btn.clicked.connect(self._generate_training_dataset_from_queue)
        export_btn = QPushButton("导出")
        export_btn.setFixedHeight(28)
        export_btn.clicked.connect(lambda: self._switch_workspace(4))
        export_row.addWidget(yolo_btn)
        export_row.addWidget(export_btn)
        queue_layout.addLayout(export_row)
        tabs.addTab(queue_tab, "队列")

        auto_tab = self._create_annotation_control_bar_v2()
        self._auto_label_tab_index = tabs.addTab(auto_tab, "自动")
        self.inspector_tabs = tabs

        layout.addWidget(tabs, stretch=1)
        return inspector

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

    def _open_image_dir(self) -> None:
        # Auto-create project if none exists (LabelImg-style: select dir = start working)
        if not self.current_project:
            dir_path = QFileDialog.getExistingDirectory(self, "打开图片目录")
            if not dir_path:
                return
            project_root = Path(dir_path)
            self.statusBar().showMessage("正在创建项目...", 2000)
            try:
                project = ProjectManager().open_project(str(project_root))
            except Exception as exc:
                QMessageBox.critical(self, "创建项目失败", str(exc))
                return
            # If selected dir has images/ subdir with content, use as-is (YOLO dataset root).
            # Otherwise import images from the selected dir into images/.
            images_dir = project_root / "images"
            if not images_dir.exists() or not any(images_dir.iterdir()):
                try:
                    imported, skipped = ProjectManager().import_folder(project, dir_path)
                    if imported > 0:
                        self.statusBar().showMessage(
                            f"已导入 {imported} 张图片" +
                            (f"，跳过 {skipped} 张" if skipped else ""), 3500
                        )
                except Exception as exc:
                    logger.warning(f"Auto-import images failed: {exc}")
            self.current_project = project
            self.project_panel.refresh_projects()
            self.project_panel._select_project(project)
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
            QMessageBox.information(self, "提示", "请先打开图片目录")
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
        self._set_model_status(f"加载中…", None)
        QApplication.processEvents()
        success = self.model_manager.load_model(model_name)
        if success:
            self._set_model_status("已加载", True)
            self.model_pill.setText(os.path.basename(model_name))
        else:
            self._set_model_status("加载失败", False)
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
        self.yolo_progress_bar.setVisible(True)
        self._set_model_status(f"0/{len(image_paths)}", None)
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
        self._set_model_status(f"{current}/{total}", None)

    def _on_yolo_auto_label_finished(self, results: dict) -> None:
        total_boxes = 0
        current_abs = os.path.normcase(os.path.abspath(self.current_image_path)) if self.current_image_path else ""
        current_shapes = None
        target_class_id = self._class_id_for_yolo_auto_label()
        if target_class_id is None:
            self._set_model_status("需先创建类别", False)
            self.yolo_progress_bar.setVisible(False)
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
                    width, height = read_image_size(image_path)
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
        self.yolo_progress_bar.setVisible(False)
        self._set_model_status(f"完成 {len(results)}张/{total_boxes}框", True)
        self.yolo_current_btn.setEnabled(True)
        self.yolo_all_btn.setEnabled(True)
        self._maybe_offer_training_after_annotation()
        self._cleanup_yolo_label()

    def _on_yolo_auto_label_error(self, error_msg: str) -> None:
        self.yolo_progress_bar.setVisible(False)
        self._set_model_status("标注失败", False)
        self.yolo_current_btn.setEnabled(True)
        self.yolo_all_btn.setEnabled(True)
        QMessageBox.critical(self, "YOLO 自动标注失败", error_msg)
        self._cleanup_yolo_label()

    def _cleanup_yolo_label(self) -> None:
        if self._yolo_label_thread is not None:
            if self._yolo_label_thread.isRunning():
                self._yolo_label_thread.quit()
                self._yolo_label_thread.wait(3000)
            self._yolo_label_thread.deleteLater()
            self._yolo_label_thread = None
        if self._yolo_label_worker is not None:
            self._yolo_label_worker.deleteLater()
            self._yolo_label_worker = None

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
        """Open class-name mapping as a workflow overlay page."""
        self._open_stage(11)  # STAGE_NAMEMAP

    def _import_model_names_to_map(self, model_names: dict, table) -> None:
        """Import class names from the current model into the mapping table."""
        from PyQt6.QtWidgets import QTableWidgetItem

        from core.class_manager import COCO_EN_ZH_MAP

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

    def _image_size_for_save(self, image_path: str) -> tuple[int, int]:
        if image_path == self.current_image_path and self.canvas.image_width and self.canvas.image_height:
            return self.canvas.image_width, self.canvas.image_height
        width, height = read_image_size(image_path)
        if width <= 0 or height <= 0:
            raise ValueError(f"无法读取图片尺寸: {image_path}")
        return width, height

    # ------------------------------------------------------------------
    # Tool entry points
    # ------------------------------------------------------------------

    def _focus_auto_labeling_panel(self) -> None:
        self._switch_workspace(0)
        if hasattr(self, "inspector_tabs") and hasattr(self, "_auto_label_tab_index"):
            self.inspector_tabs.setCurrentIndex(self._auto_label_tab_index)
        self._set_model_status("请先加载模型", False)
        if hasattr(self, "yolo_model_combo"):
            self.yolo_model_combo.setFocus()

    # ------------------------------------------------------------------
    # LLM auto-labeling
    # ------------------------------------------------------------------

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
            width, height = self._image_size_for_save(image_path)
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

    def _llm_detections_to_shapes(
        detections,
        image_width: int,
        image_height: int,
        class_id: int,
        class_name: str,
    ) -> list[dict]:
        shapes = []
        for _label, x1, y1, x2, y2 in detections:
            m = max(abs(x1), abs(y1), abs(x2), abs(y2))
            if m <= 1.0:
                # 0-1 normalized coordinates
                scale_x, scale_y = image_width, image_height
            elif m <= 1000:
                # Qwen 0-1000 normalized coordinates
                scale_x, scale_y = image_width / 1000.0, image_height / 1000.0
            else:
                # Absolute pixel coordinates
                scale_x, scale_y = 1.0, 1.0
            abs_x1 = int(x1 * scale_x)
            abs_y1 = int(y1 * scale_y)
            abs_x2 = int(x2 * scale_x)
            abs_y2 = int(y2 * scale_y)
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
    # LLM free detection (auto-discover classes)
    # ------------------------------------------------------------------

    def _run_llm_free_detect(self):
        """LLM 自由检测 — 让模型自己发现所有物体并自动创建类别."""
        if not self.current_image_path:
            QMessageBox.warning(self, "提示", "请先打开图片")
            return

        llm_config = load_llm_config()
        if not llm_config.get("base_url"):
            QMessageBox.warning(self, "提示", "请先在 LLM 配置中设置 Base URL")
            self._show_auto_label_dialog_llm()
            return

        self.statusBar().showMessage("LLM 自由检测中...", 0)
        self._llm_worker = LLMInferenceWorker(
            self.current_image_path, "__free_detect__", llm_config,
        )
        self._llm_worker.finished.connect(self._on_llm_free_result)
        self._llm_worker.error.connect(self._on_llm_error)
        self._llm_worker.start()

    def _on_llm_free_result(self, detections):
        self.statusBar().showMessage(f"LLM 自由检测完成: {len(detections)} 个目标", 3000)
        if not detections:
            QMessageBox.information(self, "完成", "未检测到目标")
            return

        img_w = self.canvas.image_width
        img_h = self.canvas.image_height
        if img_w <= 0 or img_h <= 0:
            return

        shapes = list(self.canvas.get_shapes())
        existing_count = len(shapes)
        self.canvas.push_undo()
        shapes.extend(self._llm_free_detections_to_shapes(detections, img_w, img_h))
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

    def _llm_free_detections_to_shapes(
        self, detections, image_width: int, image_height: int,
    ) -> list[dict]:
        """Convert free-detection results to shapes, auto-creating new classes."""
        shapes = []
        classes = self.class_manager.get_all_classes()

        for label, x1, y1, x2, y2 in detections:
            label = label.strip()
            if not label:
                continue

            # Convert coordinates
            m = max(abs(x1), abs(y1), abs(x2), abs(y2))
            if m <= 1.0:
                # 0-1 normalized coordinates
                scale_x, scale_y = image_width, image_height
            elif m <= 1000:
                # Qwen 0-1000 normalized coordinates
                scale_x, scale_y = image_width / 1000.0, image_height / 1000.0
            else:
                # Absolute pixel coordinates
                scale_x, scale_y = 1.0, 1.0
            abs_x1 = int(x1 * scale_x)
            abs_y1 = int(y1 * scale_y)
            abs_x2 = int(x2 * scale_x)
            abs_y2 = int(y2 * scale_y)

            abs_x1 = max(0, min(abs_x1, image_width))
            abs_y1 = max(0, min(abs_y1, image_height))
            abs_x2 = max(0, min(abs_x2, image_width))
            abs_y2 = max(0, min(abs_y2, image_height))
            abs_x1, abs_x2 = sorted((abs_x1, abs_x2))
            abs_y1, abs_y2 = sorted((abs_y1, abs_y2))

            if abs_x2 - abs_x1 < 2 or abs_y2 - abs_y1 < 2:
                continue

            # Find or create class
            class_id = None
            for i, c in enumerate(classes):
                if c == label:
                    class_id = i
                    break

            if class_id is None:
                # Auto-create new class
                class_id = self.class_manager.add_class(label)
                classes = self.class_manager.get_all_classes()
                logger.info(f"自由检测自动创建类别: {label} (id={class_id})")

            shapes.append({
                "type": ShapeType.BBOX,
                "class_id": class_id,
                "class_name": label,
                "data": {"x1": abs_x1, "y1": abs_y1, "x2": abs_x2, "y2": abs_y2},
            })

        return shapes

    # ------------------------------------------------------------------
    # Auto-label settings dialogs
    # ------------------------------------------------------------------

    def _show_auto_label_dialog_llm(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("LLM 自动标注配置")
        dlg.setMinimumWidth(620)
        dlg.setModal(True)
        layout = QVBoxLayout(dlg)

        llm_config = load_llm_config()

        # ── Presets: base_url + known vision models ──
        SILICONFLOW_MODELS = [
            "Qwen/Qwen3-VL-8B-Instruct",
            "Qwen/Qwen3-VL-8B-Thinking",
            "Qwen/Qwen3-VL-30B-A3B-Instruct",
            "Qwen/Qwen3-VL-30B-A3B-Thinking",
            "Qwen/Qwen3-VL-32B-Instruct",
            "Qwen/Qwen3-VL-32B-Thinking",
            "zai-org/GLM-4.5V",
            "zai-org/GLM-4.6V",
        ]
        ALIYUN_MODELS = [
            "qwen-vl-max", "qwen-vl-plus",
        ]

        preset_map = {
            "硅基流动": ("https://api.siliconflow.cn/v1", SILICONFLOW_MODELS),
            "阿里云通义千问": ("https://dashscope.aliyuncs.com/compatible-mode/v1", ALIYUN_MODELS),
        }

        # Detect current preset
        current_base = llm_config.get("base_url", "")
        current_model = llm_config.get("model_name", "")
        selected_preset = "自定义"
        for name, (base, models) in preset_map.items():
            if current_base == base:
                selected_preset = name
                break

        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("API 预设:"))
        preset_combo = QComboBox()
        preset_combo.addItems(["自定义"] + list(preset_map.keys()))
        preset_combo.setCurrentText(selected_preset)
        preset_row.addWidget(preset_combo)
        preset_row.addStretch()
        layout.addLayout(preset_row)

        # ── Form fields ──
        form = QFormLayout()

        api_key_edit = QLineEdit(llm_config.get("api_key", ""))
        api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        api_key_edit.setPlaceholderText("sk-...")
        form.addRow("API Key:", api_key_edit)

        base_url_edit = QLineEdit(current_base)
        base_url_edit.setPlaceholderText("https://api.siliconflow.cn/v1")
        form.addRow("Base URL:", base_url_edit)

        # Model: editable combo (dropdown + manual input)
        model_label = QLabel("视觉模型:")
        model_combo = QComboBox()
        model_combo.setEditable(True)
        model_combo.setMinimumWidth(300)
        if selected_preset in preset_map:
            model_combo.addItems(preset_map[selected_preset][1])
        model_combo.setCurrentText(current_model)
        form.addRow(model_label, model_combo)

        def on_preset_changed(_index):
            text = preset_combo.currentText()
            model_combo.clear()
            if text in preset_map:
                base, models = preset_map[text]
                base_url_edit.setText(base)
                model_combo.addItems(models)
                if models:
                    model_combo.setCurrentIndex(0)
            else:
                model_combo.setEditText("")

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
                "model_name": model_combo.currentText(),
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

    def _load_captured_frames(self, paths: list[str]) -> None:
        valid_paths = [
            os.path.abspath(path)
            for path in paths
            if path and os.path.isfile(path) and os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS
        ]
        if not valid_paths:
            QMessageBox.warning(self, "提示", "没有可加载的截帧图片")
            return

        # 没有项目时自动创建一个
        if not self.current_project:
            from core.project_manager import ProjectManager as _PM
            pm = _PM()
            project = pm.create_project(str(Path("data").resolve()), "视频截帧项目")
            pm.import_images(project, valid_paths)
            self.current_project = pm.open_project(project["root"])
            self._on_project_opened(self.current_project)
            self.statusBar().showMessage(f"已创建项目并导入 {len(valid_paths)} 帧", 3000)
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

    def _set_dirty(self, dirty: bool) -> None:
        self._dirty = dirty
        self.dirty_pill.setText("未保存" if dirty else "已保存")
        self.dirty_pill.set_variant("warning" if dirty else "success")
        if hasattr(self, "dirty_badge"):
            self.dirty_badge.setText("未保存" if dirty else "已保存")
            color = Theme.WARNING if dirty else Theme.SUCCESS
            self.dirty_badge.setStyleSheet(f"color: {color}; font-size: 12px;")

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

