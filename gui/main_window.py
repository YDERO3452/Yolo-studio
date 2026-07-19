"""Main application window for YOLO Studio.

Orchestrates the workflow canvas (home) and stage modules opened from nodes.
Annotation and workflow execution live in mixins to keep this file as a shell.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from loguru import logger
from PyQt6.QtCore import QEvent, Qt, QThread
from PyQt6.QtGui import QAction, QFont, QKeySequence
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.class_manager import ClassManager
from core.config import ConfigManager
from core.model_manager import ModelManager
from core.project_manager import ProjectManager
from gui.advanced_features_panel import AdvancedFeaturesPanel
from gui.canvas import CanvasMode
from gui.dataset_panel import DatasetPanel
from gui.export_panel import ExportPanel
from gui.inference_panel import InferencePanel
from gui.main_window_annotation import AnnotationWorkbenchMixin
from gui.main_window_workflow import WorkflowOpsMixin
from gui.node_sheet import NodeSheet
from gui.project_panel import ProjectPanel
from gui.theme import Theme, apply_light_palette, build_stylesheet
from gui.training_panel import TrainingPanel
from gui.training_results_panel import TrainingResultsPanel
from gui.workflow_canvas_panel import ALL_STAGES, WorkflowCanvasPanel
from gui.workflow_optimization_panel import WorkflowOptimizationPanel
from gui.yolo_label_worker import YOLOAutoLabelWorker

# stage_stack indices for tool overlays (after main stages 0-6)
STAGE_PROJECT = 7
STAGE_VIDEO = 8
STAGE_ENV = 9
STAGE_FORMAT = 10
STAGE_NAMEMAP = 11


class MainWindow(WorkflowOpsMixin, AnnotationWorkbenchMixin, QMainWindow):
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
        self.prompt_for_class_after_draw = True
        self._llm_worker = None
        self._llm_batch_worker = None
        self._llm_progress_dialog = None
        self._llm_batch_class_id = 0
        self._llm_batch_class_name = ""
        self._video_dialog = None
        self._env_dialog = None
        self._format_dialog = None
        self._namemap_table: Optional[QTableWidget] = None

        self._build_ui()
        self._init_menus()
        self._init_statusbar()
        self._connect_signals()
        self._apply_theme()
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
        self.project_panel = ProjectPanel(self.class_manager, parent=self)

        self.workbench_page = QWidget()
        self.setCentralWidget(self.workbench_page)
        root = QVBoxLayout(self.workbench_page)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # --- Stage / tool panels (hosted inside NodeSheet) ---
        self.training_panel = TrainingPanel(config_manager=self.config_manager, parent=self)
        self.inference_panel = InferencePanel(config_manager=self.config_manager, parent=self)
        self.dataset_panel = DatasetPanel(config_manager=self.config_manager, parent=self)
        self.export_panel = ExportPanel(config_manager=self.config_manager, parent=self)
        self.quality_panel = self._create_quality_workspace()
        self.results_panel = TrainingResultsPanel(parent=self)

        self.stage_stack = QStackedWidget()
        self.stage_stack.addWidget(self._create_annotation_workspace())  # 0
        self.stage_stack.addWidget(self.training_panel)                 # 1
        self.stage_stack.addWidget(self.inference_panel)                # 2
        self.stage_stack.addWidget(self.dataset_panel)                  # 3
        self.stage_stack.addWidget(self.export_panel)                   # 4
        self.stage_stack.addWidget(self.quality_panel)                  # 5
        self.stage_stack.addWidget(self.results_panel)                  # 6
        self.stage_stack.addWidget(self.project_panel)                  # 7
        self.stage_stack.addWidget(self._create_video_overlay_page())    # 8
        self.stage_stack.addWidget(self._create_env_overlay_page())      # 9
        self.stage_stack.addWidget(self._create_format_overlay_page())   # 10
        self.stage_stack.addWidget(self._create_namemap_workspace())     # 11

        self._stage_titles = {
            0: "标注",
            1: "训练",
            2: "推理",
            3: "数据",
            4: "导出",
            5: "质检",
            6: "训练结果",
            STAGE_PROJECT: "管理项目",
            STAGE_VIDEO: "视频截帧",
            STAGE_ENV: "环境",
            STAGE_FORMAT: "格式转换",
            STAGE_NAMEMAP: "类别名映射",
        }
        self._stage_keys = {
            0: "annotate",
            1: "train",
            2: "infer",
            3: "dataset",
            4: "export",
            5: "quality",
            6: "results",
            STAGE_PROJECT: "project",
            STAGE_VIDEO: "video",
            STAGE_ENV: "env",
            STAGE_FORMAT: "format",
            STAGE_NAMEMAP: "namemap",
        }
        self._stage_meta: dict[int, tuple[str, str, str]] = {}
        for spec in ALL_STAGES:
            if spec.workspace_index >= 0 and spec.kind == "main":
                self._stage_meta[spec.workspace_index] = (spec.title, spec.subtitle, spec.accent)
        self._stage_meta[STAGE_PROJECT] = ("管理项目", "创建 / 打开项目", "#2F6FED")
        self._stage_meta[STAGE_VIDEO] = ("视频截帧", "抽帧进数据集", "#2F6FED")
        self._stage_meta[STAGE_ENV] = ("环境", "CUDA / PyTorch", "#5C6B7A")
        self._stage_meta[STAGE_FORMAT] = ("格式转换", "YOLO / VOC / COCO", "#2F6FED")
        self._stage_meta[STAGE_NAMEMAP] = ("类别名映射", "模型名 ↔ 项目类", "#1B7F5A")

        # Dimmed canvas + centered node sheet (grid peeks around edges).
        self.stage_overlay = QWidget()
        self.stage_overlay.setObjectName("StageOverlayDim")
        overlay_layout = QGridLayout(self.stage_overlay)
        overlay_layout.setContentsMargins(24, 24, 24, 24)
        overlay_layout.setSpacing(0)

        self.node_sheet = NodeSheet()
        self.node_sheet.closed.connect(self._return_to_workflow)
        self.node_sheet.set_body(self.stage_stack)
        overlay_layout.addWidget(self.node_sheet, 0, 0)
        self.stage_overlay.hide()

        self.stage_host = self.stage_overlay
        self.workspace_stack = self.stage_stack

        self.workflow_panel = WorkflowCanvasPanel(parent=self)
        self.workflow_panel.open_workspace.connect(self._open_stage)
        self.workflow_panel.open_action.connect(self._on_workflow_action)
        self.workflow_panel.menu_requested.connect(self._show_app_menu)
        self._setup_workflow_executor()

        self.canvas_area = QWidget()
        canvas_grid = QGridLayout(self.canvas_area)
        canvas_grid.setContentsMargins(0, 0, 0, 0)
        canvas_grid.setSpacing(0)
        canvas_grid.addWidget(self.workflow_panel, 0, 0)
        canvas_grid.addWidget(self.stage_overlay, 0, 0)

        self.main_stack = self.stage_stack
        root.addWidget(self.canvas_area, stretch=1)

        self._update_project_gate()
        self.training_panel.model_ready.connect(self._on_trained_model_ready)

    @staticmethod
    def _embed_dialog(dialog: QDialog) -> QWidget:
        """Host a QDialog as an in-overlay page (no modal window)."""
        page = QWidget()
        page.setObjectName("WorkspacePage")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        dialog.setWindowFlags(Qt.WindowType.Widget)
        layout.addWidget(dialog)
        return page

    def _create_video_overlay_page(self) -> QWidget:
        from gui.video_capture_dialog import VideoCaptureDialog

        self._video_dialog = VideoCaptureDialog(self)
        self._video_dialog.frames_captured.connect(self._load_captured_frames)
        return self._embed_dialog(self._video_dialog)

    def _create_env_overlay_page(self) -> QWidget:
        from gui.env_check_dialog import EnvironmentCheckDialog

        self._env_dialog = EnvironmentCheckDialog(self)
        return self._embed_dialog(self._env_dialog)

    def _create_format_overlay_page(self) -> QWidget:
        from gui.format_conversion_dialog import FormatConversionDialog

        self._format_dialog = FormatConversionDialog(
            class_manager=self.class_manager, parent=self
        )
        return self._embed_dialog(self._format_dialog)

    def _create_namemap_workspace(self) -> QWidget:
        page = QWidget()
        page.setObjectName("WorkspacePage")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        info = QLabel(
            "编辑模型类别名到项目类别名的映射。自动标注时，模型返回的类别名会先查此表翻译。\n"
            "例如: person → 人, car → 汽车"
        )
        info.setWordWrap(True)
        info.setObjectName("MutedText")
        layout.addWidget(info)

        import_row = QHBoxLayout()
        import_row.addStretch()
        self._namemap_import_btn = QPushButton("从当前模型导入类别名")
        self._namemap_import_btn.clicked.connect(self._namemap_import_from_model)
        import_row.addWidget(self._namemap_import_btn)
        layout.addLayout(import_row)

        self._namemap_table = QTableWidget(0, 2)
        self._namemap_table.setHorizontalHeaderLabels(["模型类别名 (英文)", "项目类别名 (中文)"])
        self._namemap_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._namemap_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._namemap_table.setAlternatingRowColors(True)
        layout.addWidget(self._namemap_table, stretch=1)

        edit_row = QHBoxLayout()
        add_btn = QPushButton("添加映射")
        add_btn.clicked.connect(lambda: self._namemap_table.insertRow(self._namemap_table.rowCount()))
        edit_row.addWidget(add_btn)
        remove_btn = QPushButton("删除选中")
        remove_btn.clicked.connect(
            lambda: self._namemap_table.currentRow() >= 0
            and self._namemap_table.removeRow(self._namemap_table.currentRow())
        )
        edit_row.addWidget(remove_btn)
        edit_row.addStretch()
        layout.addLayout(edit_row)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self._return_to_workflow)
        btn_row.addWidget(cancel_btn)
        save_btn = QPushButton("保存")
        save_btn.setObjectName("PrimaryButton")
        save_btn.clicked.connect(self._namemap_save)
        btn_row.addWidget(save_btn)
        layout.addLayout(btn_row)
        return page

    def _namemap_reload_table(self) -> None:
        if self._namemap_table is None:
            return
        name_map = self.class_manager.get_name_map()
        self._namemap_table.setRowCount(0)
        for model_name, project_name in sorted(name_map.items()):
            row = self._namemap_table.rowCount()
            self._namemap_table.insertRow(row)
            self._namemap_table.setItem(row, 0, QTableWidgetItem(model_name))
            self._namemap_table.setItem(row, 1, QTableWidgetItem(project_name))

    def _namemap_import_from_model(self) -> None:
        model = self.model_manager.get_current_model()
        if model is None:
            QMessageBox.information(self, "提示", "请先加载模型")
            return
        model_names = getattr(model, "names", None)
        if not isinstance(model_names, dict) or not model_names:
            QMessageBox.information(self, "提示", "当前模型没有可用的类别名")
            return
        self._import_model_names_to_map(model_names, self._namemap_table)

    def _namemap_save(self) -> None:
        if self._namemap_table is None:
            return
        new_map = {}
        for row in range(self._namemap_table.rowCount()):
            key_item = self._namemap_table.item(row, 0)
            val_item = self._namemap_table.item(row, 1)
            if key_item and val_item:
                key = key_item.text().strip()
                val = val_item.text().strip()
                if key and val:
                    new_map[key] = val
        self.class_manager.name_map = new_map
        self.class_manager._save_name_map()
        logger.info(f"Updated name map with {len(new_map)} entries")
        self.statusBar().showMessage(f"已保存 {len(new_map)} 条类别映射", 3000)
        self._return_to_workflow()

    def _create_quality_workspace(self) -> QWidget:
        tabs = QTabWidget()
        stats_panel = AdvancedFeaturesPanel(self.class_manager, parent=self)
        stats_panel.apply_training_config.connect(self._apply_suggested_training_config)
        tabs.addTab(stats_panel, "统计")
        tabs.addTab(WorkflowOptimizationPanel(self.class_manager, parent=self), "流程")
        self._quality_stats_panel = stats_panel
        return tabs

    def _apply_suggested_training_config(self, config: dict) -> None:
        panel = self.training_panel
        if "epochs" in config:
            panel.epochs_spin.setValue(int(config["epochs"]))
        if "batch" in config:
            panel.batch_spin.setValue(int(config["batch"]))
        if "lr0" in config and hasattr(panel, "lr0_spin"):
            panel.lr0_spin.setValue(float(config["lr0"]))
        if "optimizer" in config and hasattr(panel, "optimizer_combo"):
            opt = str(config["optimizer"])
            idx = panel.optimizer_combo.findText(opt)
            if idx >= 0:
                panel.optimizer_combo.setCurrentIndex(idx)
            else:
                panel.optimizer_combo.setCurrentText(opt)
        if "imgsz" in config and hasattr(panel, "imgsz_spin"):
            panel.imgsz_spin.setValue(int(config["imgsz"]))
        self.statusBar().showMessage("已应用推荐训练参数", 3000)

    def _init_menus(self) -> None:
        """Build actions into a HUD menu; native MenuBar stays hidden."""
        self.project_required_actions: list[QAction] = []
        self._app_menu = QMenu(self)

        file_menu = self._app_menu.addMenu("文件")
        self._add_action(file_menu, "打开目录", self._open_image_dir, "Ctrl+O")
        self._add_action(file_menu, "打开图片", self._open_single_image, "Ctrl+I")
        file_menu.addSeparator()
        self.project_required_actions.append(self._add_action(file_menu, "保存", self._save_annotations, "Ctrl+S"))
        file_menu.addSeparator()
        self._add_action(file_menu, "退出", self.close, "Ctrl+Q")

        edit_menu = self._app_menu.addMenu("编辑")
        self.project_required_actions.append(self._add_action(edit_menu, "撤销", self.canvas.undo, "Ctrl+Z"))
        self.project_required_actions.append(self._add_action(edit_menu, "重做", self.canvas.redo, "Ctrl+Y"))
        edit_menu.addSeparator()
        self.project_required_actions.append(self._add_action(edit_menu, "删除", self._delete_selected_shape, "Delete"))
        self.project_required_actions.append(self._add_action(edit_menu, "清空全部", self._clear_shapes))

        view_menu = self._app_menu.addMenu("视图")
        self.project_required_actions.append(self._add_action(view_menu, "适配窗口", self.canvas.fit_to_window, "Ctrl+F"))

        tools_menu = self._app_menu.addMenu("工具")
        self._add_action(tools_menu, "管理项目", lambda: self._open_stage(STAGE_PROJECT))
        self._add_action(tools_menu, "视频截帧", lambda: self._open_stage(STAGE_VIDEO))
        self._add_action(tools_menu, "环境", lambda: self._open_stage(STAGE_ENV))
        tools_menu.addSeparator()
        self._add_action(tools_menu, "格式转换", lambda: self._open_stage(STAGE_FORMAT))
        tools_menu.addSeparator()
        self.project_required_actions.append(
            self._add_action(tools_menu, "类别名映射", lambda: self._open_stage(STAGE_NAMEMAP))
        )
        self.project_required_actions.append(
            self._add_action(tools_menu, "自动标注", self._focus_auto_labeling_panel)
        )

        self._app_menu.addSeparator()
        self._add_action(self._app_menu, "关于", self._show_about)

        # Native menubar unused — workflow HUD owns discovery.
        self.menuBar().setVisible(False)
        self._update_project_gate()

    def _show_app_menu(self) -> None:
        btn = getattr(self.workflow_panel, "menu_btn", None)
        if btn is None:
            self._app_menu.exec(self.mapToGlobal(self.rect().topLeft()))
            return
        self._app_menu.exec(btn.mapToGlobal(btn.rect().bottomLeft()))

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
        self.dataset_panel.dataset_loaded.connect(self._on_dataset_panel_loaded)
        self.results_panel.load_inference_requested.connect(self._load_result_for_inference)
        self.results_panel.load_export_requested.connect(self._load_result_for_export)
        self._refresh_class_quick_buttons()

    def _apply_theme(self) -> None:
        app = QApplication.instance()
        if app is not None:
            font = QFont()
            font.setPointSize(9)
            app.setFont(font)
            apply_light_palette(app)
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

    def _open_stage(self, index: int) -> None:
        """Open a module/tool as an expanded node sheet on the canvas."""
        if index < 0 or index >= self.stage_stack.count():
            return
        if 0 < index <= 6 and not self._is_project_ready():
            self.statusBar().showMessage("尚未打开图片目录 — 可先在节点卡内配置，再返回运行工作流", 4000)

        if index == STAGE_VIDEO:
            self._prepare_video_overlay()
        elif index == STAGE_FORMAT:
            self._refresh_format_overlay()
        elif index == STAGE_NAMEMAP:
            self._namemap_reload_table()

        self.stage_stack.setCurrentIndex(index)
        title, subtitle, accent = self._stage_meta.get(
            index,
            (self._stage_titles.get(index, "模块"), "", Theme.ACCENT),
        )
        self.node_sheet.set_meta(title, subtitle, accent)
        self.stage_overlay.show()
        self.stage_overlay.raise_()

        key = self._stage_keys.get(index)
        if key and hasattr(self, "workflow_panel"):
            for k, node in self.workflow_panel.scene.nodes.items():
                node.setSelected(k == key)

    def _return_to_workflow(self) -> None:
        """Close the node sheet and reveal the grid canvas."""
        self._release_project_media()
        if hasattr(self, "stage_overlay"):
            self.stage_overlay.hide()
        self.statusBar().showMessage("已返回工作流画布", 2000)

    def _release_project_media(self) -> None:
        """Release VideoCapture so project files can be deleted on Windows."""
        dialog = getattr(self, "_video_dialog", None)
        if dialog is None:
            return
        try:
            if hasattr(dialog, "_stop_playback"):
                dialog._stop_playback()
            extractor = getattr(dialog, "extractor", None)
            if extractor is not None:
                extractor.close()
        except Exception:
            pass

    def _switch_workspace(self, index: int) -> None:
        """Compatibility alias — stages open as canvas overlays."""
        self._open_stage(index)

    def _prepare_video_overlay(self) -> None:
        dialog = self._video_dialog
        if dialog is None:
            return
        if self.current_project:
            output_dir = Path(self.current_project["root"]) / "images" / "video_frames"
        else:
            output_dir = Path("data") / "video_frames"
        output_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(dialog, "output_edit"):
            dialog.output_edit.setText(str(output_dir))

    def _refresh_format_overlay(self) -> None:
        dialog = self._format_dialog
        if dialog is None:
            return
        dialog.class_manager = self.class_manager
        if hasattr(dialog, "converter"):
            from core.format_converter import FormatConverter

            dialog.converter = FormatConverter(self.class_manager.get_all_classes())

    def _update_project_gate(self) -> None:
        ready = self._is_project_ready()
        for action in getattr(self, "project_required_actions", []):
            action.setEnabled(ready)

    def _is_project_ready(self) -> bool:
        return bool(self.current_project and self.image_list)

    # ------------------------------------------------------------------
    # Project workflow
    # ------------------------------------------------------------------

    def _on_project_opened(self, project: dict) -> None:
        # Drop video handles from the previous project before switching roots.
        self._release_project_media()
        if not project or not project.get("root"):
            self.current_project = None
            self.image_list = []
            self.current_image_index = -1
            self.results_panel.set_project(None)
            self._update_project_gate()
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
        self._apply_project_task(project.get("task", "detect"))

        if images:
            self._switch_workspace(0)
            self._load_current_image()
        else:
            self.statusBar().showMessage(
                f"已加载项目: {project.get('name', '')} — 请导入图片开始标注", 5000
            )
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
            self._refresh_image_list_after_dataset_rebuild()
        self._switch_workspace(1)
        self.statusBar().showMessage(f"data.yaml 已生成: {yaml_path}", 3500)

    def _on_dataset_panel_loaded(self, yaml_path: str) -> None:
        """Keep training path in sync when user loads a dataset from the data panel."""
        if not yaml_path:
            return
        self._last_dataset_yaml = yaml_path
        self.training_panel.data_yaml_edit.setText(yaml_path)
        if self.current_project:
            self._refresh_image_list_after_dataset_rebuild()

    def _refresh_image_list_after_dataset_rebuild(self) -> None:
        """Reload image paths after train/val split moves files on disk."""
        if self.current_project and self.current_project.get("root"):
            self.image_list = ProjectManager.list_images(self.current_project)
            self.current_image_dir = str(Path(self.current_project["root"]) / "images")
        else:
            self.image_list = [p for p in self.image_list if os.path.isfile(p)]
        if self.image_list:
            self.current_image_index = min(
                max(self.current_image_index, 0),
                len(self.image_list) - 1,
            )
            self.current_image_path = self.image_list[self.current_image_index]
        else:
            self.current_image_index = -1
            self.current_image_path = None
        if hasattr(self, "file_list"):
            self.file_list.load_image_list(self.image_list)
            if self.current_image_index >= 0:
                self.file_list.highlight_current(self.current_image_index)
        if self.current_image_index >= 0:
            self._load_current_image()
        elif hasattr(self, "canvas"):
            self.canvas.clear_shapes()
            self.canvas.original_image = None
            self.canvas.update()

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
    # Dialogs / helpers
    # ------------------------------------------------------------------

    def _on_workflow_action(self, action: str) -> None:
        """Open a sub-node / utility action as a canvas overlay."""
        if action == "dialog:project":
            self._open_stage(STAGE_PROJECT)
            return
        if action == "dialog:video":
            self._open_stage(STAGE_VIDEO)
            return
        if action == "dialog:env":
            self._open_stage(STAGE_ENV)
            return
        if action == "dialog:format":
            self._open_stage(STAGE_FORMAT)
            return
        if action == "dialog:namemap":
            self._open_stage(STAGE_NAMEMAP)
            return
        if action == "focus:annotate":
            self._focus_auto_labeling_panel()
            return
        if action.startswith("stage:"):
            parts = action.split(":")
            if len(parts) >= 3:
                try:
                    stage_index = int(parts[1])
                    tab_index = int(parts[2])
                except ValueError:
                    return
                self._open_stage(stage_index)
                tabs = getattr(self, "quality_panel", None)
                if isinstance(tabs, QTabWidget) and 0 <= tab_index < tabs.count():
                    tabs.setCurrentIndex(tab_index)

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "关于 YOLO Studio",
            "YOLO Studio v1.0.0\n\n"
            "YOLO 标注、训练、推理工具。\n"
            "首页是节点画布，双击进模块；⋯ 菜单里有环境检测等。\n\n"
            "GPL-3.0 · YDERO3452",
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

    def closeEvent(self, event) -> None:
        self._release_project_media()
        if hasattr(self, "project_panel"):
            self.project_panel.shutdown()
            if hasattr(self.project_panel, "_release_media_handles"):
                self.project_panel._release_media_handles()
        # Stop LLM workers
        if self._llm_batch_worker and self._llm_batch_worker.isRunning():
            self._llm_batch_worker.stop()
            self._llm_batch_worker.quit()
            if not self._llm_batch_worker.wait(3000):
                self._llm_batch_worker.terminate()
                self._llm_batch_worker.wait(1000)
        if self._llm_worker and self._llm_worker.isRunning():
            self._llm_worker.quit()
            if not self._llm_worker.wait(3000):
                self._llm_worker.terminate()
                self._llm_worker.wait(1000)
        if self._yolo_label_thread and self._yolo_label_thread.isRunning():
            worker = getattr(self, "_yolo_label_worker", None)
            if worker is not None and hasattr(worker, "stop"):
                worker.stop()
            self._yolo_label_thread.quit()
            if not self._yolo_label_thread.wait(3000):
                self._yolo_label_thread.terminate()
                self._yolo_label_thread.wait(1000)
        # Stop panel workers that may still be running
        self.inference_panel._cleanup_worker()
        self.inference_panel._cleanup_batch_worker()
        self.export_panel._cleanup_worker()
        # Signal Ultralytics trainer to stop, then clean up training worker
        self.training_panel.stop_training()
        self.training_panel._cleanup_worker()
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
        if key == Qt.Key.Key_V:
            self._set_canvas_mode(CanvasMode.EDIT)
            return
        if key == Qt.Key.Key_R:
            self._set_canvas_mode(CanvasMode.CREATE_BBOX)
            return
        if key == Qt.Key.Key_P:
            self._set_canvas_mode(CanvasMode.CREATE_POLYGON)
            return
        if key == Qt.Key.Key_O:
            self._set_canvas_mode(CanvasMode.CREATE_OBB)
            return
        if key == Qt.Key.Key_K:
            self._set_canvas_mode(CanvasMode.CREATE_KEYPOINT)
            return
        if key == Qt.Key.Key_Delete:
            self._delete_selected_shape()
            return
        super().keyPressEvent(event)
