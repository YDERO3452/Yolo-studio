"""Project management and import panel."""

from __future__ import annotations

import os
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.class_manager import ClassManager
from core.env_setup import detect_environment, get_pytorch_install_commands, get_pytorch_install_plan
from core.project_manager import ProjectManager
from gui.ui_components import StatusPill


class _ProjectEnvDetectWorker(QThread):
    finished = pyqtSignal(object, object, object)

    def run(self):
        try:
            env = detect_environment()
            plan = get_pytorch_install_plan(env)
            commands = get_pytorch_install_commands(env)
            self.finished.emit(env, plan, commands)
        except Exception as exc:
            self.finished.emit(None, {"reason": str(exc)}, {})


class ProjectPanel(QWidget):
    """Project-oriented workflow inspired by EzYOLO's import page."""

    project_opened = pyqtSignal(dict)
    data_yaml_ready = pyqtSignal(str)

    def __init__(self, class_manager: ClassManager, parent=None):
        super().__init__(parent)
        self.class_manager = class_manager
        self.manager = ProjectManager()
        self.current_project: dict | None = None
        self._video_capture_dialog = None
        self._env_worker = None
        self._build_ui()
        self.refresh_projects()
        self._set_project_controls_enabled(False)
        self._start_env_check()

    def set_class_manager(self, class_manager: ClassManager) -> None:
        self.class_manager = class_manager

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)
        left.setMinimumWidth(360)
        left.setMaximumWidth(480)

        project_group = QGroupBox("1. 新建或导入项目")
        project_form = QFormLayout(project_group)
        project_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.project_combo = QComboBox()
        self.project_combo.currentIndexChanged.connect(self._on_project_selected)
        project_form.addRow("当前项目:", self.project_combo)

        project_btns = QVBoxLayout()
        row1 = QHBoxLayout()
        self.new_btn = QPushButton("新建")
        self.new_btn.setObjectName("PrimaryButton")
        self.new_btn.clicked.connect(self.create_project)
        self.open_folder_btn = QPushButton("打开目录")
        self.open_folder_btn.clicked.connect(self.open_project_folder)
        row1.addWidget(self.new_btn)
        row1.addWidget(self.open_folder_btn)

        row2 = QHBoxLayout()
        self.import_project_btn = QPushButton("导入项目")
        self.import_project_btn.clicked.connect(self.import_project_folder)
        self.delete_btn = QPushButton("删除")
        self.delete_btn.setObjectName("DangerButton")
        self.delete_btn.clicked.connect(self.delete_project)
        row2.addWidget(self.import_project_btn)
        row2.addWidget(self.delete_btn)

        project_btns.addLayout(row1)
        project_btns.addLayout(row2)
        project_form.addRow("", project_btns)

        self.project_status = StatusPill("未选择")
        project_form.addRow("状态:", self.project_status)
        left_layout.addWidget(project_group)

        self.import_group = QGroupBox("2. 准备图片 / 视频截帧")
        import_layout = QVBoxLayout(self.import_group)
        folder_btn = QPushButton("导入图片文件夹")
        folder_btn.clicked.connect(self.import_folder)
        images_btn = QPushButton("导入图片文件")
        images_btn.clicked.connect(self.import_images)
        video_row = QHBoxLayout()
        video_btn = QPushButton("导入视频抽帧")
        video_btn.clicked.connect(self.import_video)
        self.frame_interval_spin = QSpinBox()
        self.frame_interval_spin.setRange(1, 10000)
        self.frame_interval_spin.setValue(30)
        self.frame_interval_spin.setSuffix(" 帧")
        video_row.addWidget(video_btn)
        video_row.addWidget(self.frame_interval_spin)
        import_layout.addWidget(folder_btn)
        import_layout.addWidget(images_btn)
        import_layout.addLayout(video_row)
        left_layout.addWidget(self.import_group)

        self.ann_group = QGroupBox("3. 导入已有标注")
        ann_layout = QVBoxLayout(self.ann_group)
        self.overwrite_check = QCheckBox("覆盖同名标签")
        ann_layout.addWidget(self.overwrite_check)
        yolo_btn = QPushButton("导入 YOLO 标签目录")
        yolo_btn.clicked.connect(self.import_yolo_labels)
        voc_btn = QPushButton("导入 VOC XML 目录")
        voc_btn.clicked.connect(self.import_voc_labels)
        coco_btn = QPushButton("导入 COCO JSON")
        coco_btn.clicked.connect(self.import_coco_labels)
        ann_layout.addWidget(yolo_btn)
        ann_layout.addWidget(voc_btn)
        ann_layout.addWidget(coco_btn)
        left_layout.addWidget(self.ann_group)

        self.yaml_group = QGroupBox("4. 生成训练数据")
        yaml_layout = QFormLayout(self.yaml_group)
        yaml_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.train_ratio_spin = QSpinBox()
        self.train_ratio_spin.setRange(50, 95)
        self.train_ratio_spin.setValue(80)
        self.train_ratio_spin.setSuffix("%")
        yaml_layout.addRow("训练集比例:", self.train_ratio_spin)
        self.build_yaml_btn = QPushButton("生成 data.yaml")
        self.build_yaml_btn.setObjectName("PrimaryButton")
        self.build_yaml_btn.clicked.connect(self.build_data_yaml)
        yaml_layout.addRow("", self.build_yaml_btn)
        left_layout.addWidget(self.yaml_group)

        left_layout.addStretch()
        layout.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        self.summary_label = QLabel("未选择项目")
        self.summary_label.setObjectName("PanelTitle")
        right_layout.addWidget(self.summary_label)

        self.project_info = QTextEdit()
        self.project_info.setReadOnly(True)
        self.project_info.setMinimumHeight(180)
        right_layout.addWidget(self.project_info)

        env_group = QGroupBox("环境检测 / CUDA 安装")
        env_layout = QVBoxLayout(env_group)
        self.env_summary_label = QLabel("正在检测环境...")
        self.env_summary_label.setWordWrap(True)
        env_layout.addWidget(self.env_summary_label)

        self.env_install_text = QTextEdit()
        self.env_install_text.setReadOnly(True)
        self.env_install_text.setMaximumHeight(160)
        env_layout.addWidget(self.env_install_text)

        env_btn_row = QHBoxLayout()
        self.env_recheck_btn = QPushButton("重新检测")
        self.env_recheck_btn.clicked.connect(self._start_env_check)
        self.env_copy_btn = QPushButton("复制命令")
        self.env_copy_btn.clicked.connect(self._copy_env_command)
        self.env_dialog_btn = QPushButton("打开环境配置")
        self.env_dialog_btn.clicked.connect(self._open_env_dialog)
        env_btn_row.addWidget(self.env_recheck_btn)
        env_btn_row.addWidget(self.env_copy_btn)
        env_btn_row.addWidget(self.env_dialog_btn)
        env_btn_row.addStretch()
        env_layout.addLayout(env_btn_row)
        right_layout.addWidget(env_group)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        right_layout.addWidget(self.progress)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        right_layout.addWidget(self.log_text, 1)

        layout.addWidget(right, 1)

    def refresh_projects(self) -> None:
        current_root = self.current_project.get("root") if self.current_project else None
        self.project_combo.blockSignals(True)
        self.project_combo.clear()
        self.project_combo.addItem("请选择项目...", None)
        for project in self.manager.list_projects():
            self.project_combo.addItem(project.get("name", Path(project["root"]).name), project)
        self.project_combo.blockSignals(False)
        if current_root:
            for i in range(self.project_combo.count()):
                project = self.project_combo.itemData(i)
                if project and project.get("root") == current_root:
                    self.project_combo.setCurrentIndex(i)
                    return
        self._update_summary()
        self._set_project_controls_enabled(bool(self.current_project))

    def create_project(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("新建项目")
        layout = QFormLayout(dialog)
        name_edit = QLineEdit()
        name_edit.setPlaceholderText("项目名称")
        task_combo = QComboBox()
        task_combo.addItems(["detect", "segment", "pose", "obb", "classify"])
        classes_edit = QLineEdit(",".join(self.class_manager.get_all_classes()) or "目标")
        layout.addRow("名称:", name_edit)
        layout.addRow("任务:", task_combo)
        layout.addRow("类别:", classes_edit)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        name = name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "提示", "请输入项目名称")
            return
        classes = [item.strip() for item in classes_edit.text().split(",") if item.strip()]
        project = self.manager.create_project(name, task_combo.currentText(), classes or ["目标"])
        self.current_project = project
        self.refresh_projects()
        self._select_project(project)
        self._log(f"已创建项目: {project['name']}")

    def open_project_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择项目目录")
        if not path:
            return
        project = self.manager.open_project(path)
        self.current_project = project
        self.refresh_projects()
        self._select_project(project)
        self._log(f"已打开项目目录: {path}")

    def import_project_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择已有项目或YOLO数据集目录")
        if not path:
            return
        try:
            project, imported, skipped, label_imported, label_skipped = self.manager.import_dataset_as_project(
                path,
                progress=self._progress,
            )
            self.current_project = project
            self.refresh_projects()
            self._select_project(project)
            self._log(
                "已导入项目: "
                f"{project['name']} | 图片 {imported}, 跳过 {skipped}, "
                f"标签 {label_imported}, 标签跳过 {label_skipped}"
            )
        except Exception as exc:
            QMessageBox.critical(self, "导入失败", str(exc))

    def delete_project(self) -> None:
        if not self.current_project:
            return
        delete_files = QMessageBox.question(
            self,
            "删除项目",
            "是否同时删除项目文件夹？\n选择“否”只会从项目列表移除。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
        )
        if delete_files == QMessageBox.StandardButton.Cancel:
            return
        self.manager.delete_project(self.current_project, delete_files == QMessageBox.StandardButton.Yes)
        self.current_project = None
        self.refresh_projects()
        self._set_project_controls_enabled(False)
        self.project_opened.emit({})
        self._log("项目已删除")

    def _on_project_selected(self, index: int) -> None:
        project = self.project_combo.itemData(index)
        if project:
            self._select_project(project)

    def _select_project(self, project: dict) -> None:
        self.current_project = self.manager.open_project(project["root"])
        self.project_status.setText("已加载")
        self.project_status.set_variant("success")
        self._set_project_controls_enabled(True)
        self.project_opened.emit(self.current_project)
        self._update_summary()

    def import_folder(self) -> None:
        project = self._require_project()
        if not project:
            return
        path = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if not path:
            return
        imported, skipped = self.manager.import_folder(project, path, progress=self._progress)
        self._finish_import(f"图片文件夹导入完成: 成功 {imported}, 跳过 {skipped}")

    def import_images(self) -> None:
        project = self._require_project()
        if not project:
            return
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "选择图片",
            "",
            "图片 (*.jpg *.jpeg *.png *.bmp *.tiff *.webp);;所有文件 (*)",
        )
        if not paths:
            return
        imported, skipped = self.manager.import_images(project, paths, progress=self._progress)
        self._finish_import(f"图片导入完成: 成功 {imported}, 跳过 {skipped}")

    def import_video(self) -> None:
        project = self._require_project()
        if not project:
            return
        # Close old dialog to prevent leak
        if self._video_capture_dialog is not None:
            try:
                self._video_capture_dialog.close()
            except Exception:
                # harmless: dialog already destroyed by Qt
                pass
            self._video_capture_dialog = None
        try:
            from gui.video_capture_dialog import VideoCaptureDialog

            dialog = VideoCaptureDialog(self)
            output_dir = Path(project["root"]) / "images" / "video_frames"
            output_dir.mkdir(parents=True, exist_ok=True)
            dialog.output_edit.setText(str(output_dir))
            if hasattr(dialog, "interval_frame_spin"):
                dialog.interval_frame_spin.setValue(self.frame_interval_spin.value())
            dialog.frames_captured.connect(self._on_video_frames_captured)
            self._video_capture_dialog = dialog
            dialog.show()
            dialog.raise_()
            dialog.activateWindow()
        except Exception as exc:
            QMessageBox.critical(self, "视频截帧失败", str(exc))

    def _on_video_frames_captured(self, paths: list[str]) -> None:
        project = self._require_project()
        if not project:
            return
        images_root = (Path(project["root"]) / "images").resolve()
        external_paths: list[str] = []
        in_project = 0
        for path in paths:
            frame_path = Path(path).resolve()
            try:
                frame_path.relative_to(images_root)
                in_project += 1
            except ValueError:
                external_paths.append(str(frame_path))

        imported = 0
        skipped = 0
        if external_paths:
            imported, skipped = self.manager.import_images(project, external_paths, progress=self._progress)
        self._finish_import(
            f"视频截帧导入完成: 项目内 {in_project}, 新导入 {imported}, 跳过 {skipped}"
        )

    def import_yolo_labels(self) -> None:
        project = self._require_project()
        if not project:
            return
        path = QFileDialog.getExistingDirectory(self, "选择 YOLO 标签目录")
        if not path:
            return
        imported, skipped = self.manager.import_yolo_labels(project, path, self.overwrite_check.isChecked())
        self._finish_import(f"YOLO 标签导入完成: 成功 {imported}, 跳过 {skipped}")

    def import_voc_labels(self) -> None:
        project = self._require_project()
        if not project:
            return
        path = QFileDialog.getExistingDirectory(self, "选择 VOC XML 目录")
        if not path:
            return
        imported, skipped = self.manager.import_voc_labels(project, path, self.overwrite_check.isChecked())
        self._finish_import(f"VOC 标签导入完成: 成功 {imported}, 跳过 {skipped}")

    def import_coco_labels(self) -> None:
        project = self._require_project()
        if not project:
            return
        path, _ = QFileDialog.getOpenFileName(self, "选择 COCO JSON", "", "JSON (*.json);;所有文件 (*)")
        if not path:
            return
        imported, skipped = self.manager.import_coco_labels(project, path, self.overwrite_check.isChecked())
        self._finish_import(f"COCO 标签导入完成: 成功 {imported}, 跳过 {skipped}")

    def build_data_yaml(self) -> None:
        project = self._require_project()
        if not project:
            return
        train_ratio = self.train_ratio_spin.value() / 100
        val_ratio = max(0.0, min(0.45, 1.0 - train_ratio - 0.05))
        try:
            yaml_path = self.manager.build_data_yaml(project, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=0.05)
            self.data_yaml_ready.emit(yaml_path)
            self._finish_import(f"已生成 data.yaml: {yaml_path}")
        except Exception as exc:
            QMessageBox.critical(self, "生成失败", str(exc))

    def _finish_import(self, message: str) -> None:
        self.progress.setValue(100)
        self._log(message)
        if self.current_project:
            self.current_project = self.manager.open_project(self.current_project["root"])
            self.project_opened.emit(self.current_project)
        self._update_summary()

    def _progress(self, current: int, total: int, name: str) -> None:
        value = int(current / total * 100) if total else 0
        self.progress.setValue(value)
        self.progress.setFormat(f"{current}/{total} {name}")

    def _require_project(self) -> dict | None:
        if not self.current_project:
            QMessageBox.warning(self, "提示", "请先创建或打开项目")
            return None
        return self.current_project

    def _update_summary(self) -> None:
        if not self.current_project:
            self.summary_label.setText("先创建或导入项目")
            self.project_info.setPlainText(
                "流程:\n"
                "1. 新建空项目，或导入已有 YOLO 数据集。\n"
                "2. 在项目页导入图片，或打开视频截帧窗口截取帧。\n"
                "3. 项目里有图片后，左侧标注、训练、推理等菜单才会开放。\n\n"
                "自动标注、SAM、LLM、格式转换都绑定当前项目，不再允许散文件模式。"
            )
            self.project_status.setText("未选择")
            self.project_status.set_variant("")
            return
        root = Path(self.current_project["root"])
        images = self.manager.list_images(self.current_project)
        labels = list((root / "labels").rglob("*.txt")) if (root / "labels").exists() else []
        yaml_path = root / "data.yaml"
        ready_text = "可进入标注/训练菜单" if images else "请先导入图片或视频截帧"
        self.summary_label.setText(self.current_project.get("name", root.name))
        lines = [
            f"根目录: {root}",
            f"任务类型: {self.current_project.get('task', 'detect')}",
            f"图片数: {len(images)}",
            f"标签数: {len(labels)}",
            f"data.yaml: {'已生成' if yaml_path.exists() else '未生成'}",
            f"下一步: {ready_text}",
        ]
        self.project_info.setPlainText("\n".join(lines))

    def _log(self, message: str) -> None:
        self.log_text.append(message)

    def _set_project_controls_enabled(self, enabled: bool) -> None:
        for group_name in ("import_group", "ann_group", "yaml_group"):
            group = getattr(self, group_name, None)
            if group is not None:
                group.setEnabled(enabled)
        if hasattr(self, "delete_btn"):
            self.delete_btn.setEnabled(enabled)

    def _start_env_check(self) -> None:
        worker = getattr(self, "_env_worker", None)
        if worker is not None and worker.isRunning():
            return
        # Disconnect old worker signal to prevent leaks
        if worker is not None:
            try:
                worker.finished.disconnect(self._on_env_check_done)
            except (TypeError, RuntimeError):
                pass
        self.env_recheck_btn.setEnabled(False)
        self.env_summary_label.setText("正在检测 GPU、CUDA、PyTorch 和 Ultralytics...")
        self.env_install_text.setPlainText("")
        self._env_worker = _ProjectEnvDetectWorker(self)
        self._env_worker.finished.connect(self._on_env_check_done)
        self._env_worker.start()

    def _on_env_check_done(self, env, plan, commands) -> None:
        self.env_recheck_btn.setEnabled(True)
        if env is None:
            reason = str((plan or {}).get("reason", "环境检测失败"))
            self.env_summary_label.setText(f"环境检测失败: {reason}")
            self.env_install_text.setPlainText("")
            return

        gpu_names = ", ".join(g.name for g in env.gpus) if env.gpus else "未检测到 GPU"
        torch_state = "可用" if env.pytorch_cuda_available else ("CPU/不可用" if env.pytorch_installed else "未安装")
        ultra_state = "已安装" if env.ultralytics_installed else "未安装"
        summary = [
            f"GPU: {gpu_names}",
            f"NVIDIA 驱动: {env.nvidia_driver_version or '未检测到'}",
            f"驱动支持 CUDA: {env.driver_max_cuda or '未知'}",
            f"PyTorch: {env.pytorch_version or '未安装'} ({torch_state})",
            f"Ultralytics: {ultra_state}",
            f"建议: {(plan or {}).get('reason', '')}",
        ]
        self.env_summary_label.setText("\n".join(summary))

        install_lines = []
        online_cmd = (commands or {}).get("online_install", "")
        download_cmd = (commands or {}).get("download_wheels", "")
        local_cmd = (commands or {}).get("install_local_dir", "")
        if online_cmd:
            install_lines.extend(["在线安装 PyTorch:", online_cmd, ""])
        if download_cmd:
            install_lines.extend(["离线下载 wheel:", download_cmd, ""])
        if local_cmd:
            install_lines.extend(["本地安装 wheel:", local_cmd, ""])
        install_lines.append("安装 Ultralytics:")
        install_lines.append(f'"{os.sys.executable}" -m pip install ultralytics')
        self.env_install_text.setPlainText("\n".join(install_lines))

    def _copy_env_command(self) -> None:
        text = self.env_install_text.toPlainText().strip()
        if text:
            QApplication.clipboard().setText(text)
            QMessageBox.information(self, "已复制", "环境安装命令已复制到剪贴板")

    def _open_env_dialog(self) -> None:
        try:
            from gui.env_check_dialog import EnvironmentCheckDialog

            dialog = EnvironmentCheckDialog(self)
            dialog.exec()
        except Exception as exc:
            QMessageBox.critical(self, "错误", f"无法打开环境配置:\n{exc}")

    def shutdown(self) -> None:
        worker = getattr(self, "_env_worker", None)
        if worker is None:
            return
        try:
            worker.finished.disconnect(self._on_env_check_done)
        except (TypeError, RuntimeError):
            pass
        if worker.isRunning():
            worker.requestInterruption()
            worker.quit()
            if not worker.wait(3000):
                worker.terminate()
                worker.wait(1000)
        self._env_worker = None

    def closeEvent(self, event) -> None:
        self.shutdown()
        super().closeEvent(event)
