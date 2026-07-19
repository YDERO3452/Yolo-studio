"""Workflow run handlers mixed into MainWindow."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from gui.workflow_executor import WorkflowExecutor

_TASK_MODEL = {
    "detect": "yolo11n.pt",
    "segment": "yolo11n-seg.pt",
    "pose": "yolo11n-pose.pt",
    "obb": "yolo11n-obb.pt",
    "classify": "yolo11n-cls.pt",
}


class WorkflowOpsMixin:
    def _setup_workflow_executor(self) -> None:
        self.workflow_executor = WorkflowExecutor(self)
        panel = self.workflow_panel
        panel.run_requested.connect(self._start_workflow_run)
        panel.stop_requested.connect(self._stop_workflow_run)

        ex = self.workflow_executor
        ex.log.connect(panel.append_log)
        ex.node_started.connect(lambda k: panel.set_node_status(k, "running"))
        ex.node_finished.connect(
            lambda k, ok, msg: panel.set_node_status(k, "done" if ok else "error", msg)
        )
        ex.pipeline_started.connect(lambda: panel.set_running_ui(True))
        ex.pipeline_finished.connect(self._on_workflow_pipeline_finished)

        ex.set_handler("dataset", self._wf_run_dataset)
        ex.set_handler("annotate", self._wf_run_annotate)
        ex.set_handler("train", self._wf_run_train)
        ex.set_handler("results", self._wf_run_results)
        ex.set_handler("infer", self._wf_run_infer)
        ex.set_handler("export", self._wf_run_export)
        ex.set_handler("quality", self._wf_run_quality)

    def _start_workflow_run(self) -> None:
        if self.workflow_executor.is_running:
            return
        if hasattr(self, "_return_to_workflow"):
            self._return_to_workflow()
        keys, edges = self.workflow_panel.collect_graph()
        self.workflow_panel.reset_all_status()
        for k in keys:
            self.workflow_panel.set_node_status(k, "pending")
        self.workflow_executor.start(keys, edges)

    def _stop_workflow_run(self) -> None:
        self.workflow_executor.stop()
        self._abort_workflow_jobs()

    def _abort_workflow_jobs(self) -> None:
        try:
            self.training_panel.stop_training()
        except Exception:
            pass
        try:
            self.inference_panel.stop_batch_inference()
        except Exception:
            pass
        try:
            self.inference_panel.stop_inference()
        except Exception:
            pass
        try:
            self.export_panel.stop_export()
        except Exception:
            pass
        worker = getattr(self, "_yolo_label_worker", None)
        if worker is not None and hasattr(worker, "stop"):
            try:
                worker.stop()
            except Exception:
                pass

    def _on_workflow_pipeline_finished(self, ok: bool, summary: str) -> None:
        self.workflow_panel.set_running_ui(False)
        for key, node in getattr(self.workflow_panel, "nodes", {}).items():
            if getattr(node, "status", "") == "pending":
                self.workflow_panel.set_node_status(key, "skipped", "已跳过")
        self.workflow_panel.append_log(("✓ " if ok else "✗ ") + summary)
        self.statusBar().showMessage(summary, 5000)

    def _wf_resolve_data_yaml(self) -> Optional[str]:
        for edit in (
            getattr(self.dataset_panel, "data_yaml_edit", None),
            getattr(self.training_panel, "data_yaml_edit", None),
        ):
            if edit is None:
                continue
            path = edit.text().strip()
            if path and os.path.isfile(path):
                return path
        if self.current_project and self.current_project.get("root"):
            candidate = Path(self.current_project["root"]) / "data.yaml"
            if candidate.is_file():
                return str(candidate)
        return None

    def _wf_run_dataset(self, key: str, ex: WorkflowExecutor) -> None:
        yaml_path = self._wf_resolve_data_yaml()
        if not yaml_path and self.current_project and self.current_project.get("root"):
            try:
                from core.project_manager import ProjectManager

                yaml_path = ProjectManager().build_data_yaml(self.current_project)
                if yaml_path:
                    self.dataset_panel.data_yaml_edit.setText(yaml_path)
                    self.training_panel.data_yaml_edit.setText(yaml_path)
                    self.workflow_panel.append_log(f"已自动生成 data.yaml: {yaml_path}")
            except Exception as exc:
                ex.finish_node(key, False, f"生成 data.yaml 失败: {exc}")
                return
        if not yaml_path:
            ex.finish_node(key, False, "未找到 data.yaml，请先打开项目或在「数据」页创建")
            return
        from core.dataset import DatasetManager

        dataset_dir = str(Path(yaml_path).resolve().parent)
        issues = DatasetManager(dataset_dir).validate_dataset(dataset_dir)
        self.training_panel.data_yaml_edit.setText(yaml_path)
        self.dataset_panel.data_yaml_edit.setText(yaml_path)
        if issues:
            ex.finish_node(key, False, f"校验失败: {issues[0]}")
            return
        ex.finish_node(key, True, Path(yaml_path).name)

    def _wf_run_annotate(self, key: str, ex: WorkflowExecutor) -> None:
        if not self.image_list:
            if self.current_project and self.current_project.get("root"):
                from core.project_manager import ProjectManager

                images = ProjectManager.list_images(self.current_project)
                if images:
                    self.image_list = images
                    self.current_image_dir = str(Path(self.current_project["root"]) / "images")
            if not self.image_list:
                ex.finish_node(key, False, "无图片 — 请先打开项目或图片目录")
                return

        labeled = 0
        unlabeled: list[str] = []
        for image_path in self.image_list:
            try:
                from gui.annotation_io import label_path_for_image

                lp = label_path_for_image(image_path)
                if lp and os.path.isfile(lp) and os.path.getsize(lp) > 0:
                    labeled += 1
                else:
                    unlabeled.append(image_path)
            except Exception:
                unlabeled.append(image_path)

        total = len(self.image_list)
        if not unlabeled:
            ex.finish_node(key, True, f"已全部标注 {total}")
            return

        if self.model_manager.is_model_loaded():
            self.workflow_panel.append_log(
                f"标注节点: 自动标注未标注图片 {len(unlabeled)}/{total}"
            )

            def _on_done(results: dict) -> None:
                if ex.stop_requested:
                    ex.finish_node(
                        key, False,
                        f"已停止（已处理 {len(results)} 张）",
                    )
                    return
                boxes = sum(len(v or []) for v in results.values())
                ex.finish_node(
                    key, True,
                    f"自动标注 {len(results)} 张 / {boxes} 检测（合计队列 {total}）",
                )

            def _on_error(msg: str) -> None:
                if ex.stop_requested:
                    ex.finish_node(key, False, "已停止")
                    return
                ex.finish_node(key, False, f"自动标注失败: {msg}")

            started = self._start_yolo_auto_label(
                unlabeled,
                quiet=True,
                on_finished=_on_done,
                on_error=_on_error,
            )
            if not started:
                ex.finish_node(
                    key,
                    True,
                    f"队列 {total}，已标注 {labeled}（自动标注忙，作人工门禁通过）",
                )
            return

        ex.finish_node(
            key,
            True,
            f"队列 {total}，已标注 {labeled}，未标 {len(unlabeled)}（未加载模型，跳过自动标注）",
        )

    def _wf_run_train(self, key: str, ex: WorkflowExecutor) -> None:
        yaml_path = self._wf_resolve_data_yaml()
        if yaml_path:
            self.training_panel.data_yaml_edit.setText(yaml_path)

        def _on_finished(result: dict) -> None:
            try:
                self.training_panel.training_finished.disconnect(_on_finished)
            except TypeError:
                pass
            if ex.stop_requested:
                ex.finish_node(key, False, "已停止")
                return
            ok = bool(result.get("success"))
            msg = result.get("save_dir") or result.get("error") or ("完成" if ok else "失败")
            if ok and isinstance(msg, str) and len(msg) > 40:
                msg = os.path.basename(msg.rstrip("\\/"))
            ex.finish_node(key, ok, str(msg))

        self.training_panel.training_finished.connect(_on_finished)
        started = self.training_panel.start_training(workflow_mode=True)
        if not started:
            try:
                self.training_panel.training_finished.disconnect(_on_finished)
            except TypeError:
                pass
            ex.finish_node(key, False, "无法启动训练（检查模型与 data.yaml）")

    def _wf_run_results(self, key: str, ex: WorkflowExecutor) -> None:
        self.results_panel.refresh_runs()
        best = None
        for run in self.results_panel._find_runs()[:8]:
            candidate = run / "weights" / "best.pt"
            if candidate.is_file():
                best = str(candidate)
                break
        if not best:
            # Also scan relative runs/
            for root in (Path("runs"), Path("runs/detect"), Path("runs/train")):
                if not root.exists():
                    continue
                for path in sorted(root.rglob("best.pt"), key=lambda p: p.stat().st_mtime, reverse=True):
                    best = str(path)
                    break
                if best:
                    break
        if not best:
            ex.finish_node(key, False, "未找到 best.pt")
            return
        # Push into infer / export panels for downstream nodes
        if hasattr(self.inference_panel, "model_path_edit"):
            self.inference_panel.model_path_edit.setText(best)
        if hasattr(self.export_panel, "set_model_path"):
            self.export_panel.set_model_path(best)
        ex.finish_node(key, True, os.path.basename(best))

    def _wf_run_infer(self, key: str, ex: WorkflowExecutor) -> None:
        # Ensure model loaded
        model_path = ""
        if hasattr(self.inference_panel, "model_path_edit"):
            model_path = self.inference_panel.model_path_edit.text().strip()
        if model_path and os.path.isfile(model_path):
            try:
                self.inference_panel.load_model_from_path(model_path)
            except Exception:
                if hasattr(self.inference_panel, "load_model"):
                    self.inference_panel.load_model()
        if not self.inference_panel.inferencer:
            # Try load_model which reads the edit box
            try:
                self.inference_panel.load_model()
            except Exception:
                pass
        if not self.inference_panel.inferencer:
            ex.finish_node(key, False, "请先在推理页加载模型")
            return

        # Default batch folder to current image dir if empty
        folder = self.inference_panel.batch_folder_edit.text().strip()
        if not folder and self.current_image_dir and os.path.isdir(self.current_image_dir):
            self.inference_panel.batch_folder_edit.setText(self.current_image_dir)
            folder = self.current_image_dir
        if not folder and self.image_list:
            folder = str(Path(self.image_list[0]).parent)
            self.inference_panel.batch_folder_edit.setText(folder)

        def _on_batch_done(total: int) -> None:
            try:
                self.inference_panel.batch_inference_finished.disconnect(_on_batch_done)
            except TypeError:
                pass
            if ex.stop_requested:
                ex.finish_node(key, False, f"已停止（已处理 {total}）")
                return
            ex.finish_node(key, True, f"处理 {total} 张")

        self.inference_panel.batch_inference_finished.connect(_on_batch_done)
        started = self.inference_panel.start_batch_inference(workflow_mode=True)
        if not started:
            try:
                self.inference_panel.batch_inference_finished.disconnect(_on_batch_done)
            except TypeError:
                pass
            ex.finish_node(key, False, "无法启动批量推理（检查模型与图片目录）")

    def _wf_run_export(self, key: str, ex: WorkflowExecutor) -> None:
        model_path = ""
        if hasattr(self.export_panel, "model_path_edit"):
            model_path = self.export_panel.model_path_edit.text().strip()
        if model_path and os.path.isfile(model_path):
            self.export_panel.load_model_from_path(model_path)
        elif not self.export_panel.exporter:
            self.export_panel.load_model()

        def _on_export_done(result: dict) -> None:
            try:
                self.export_panel.export_finished.disconnect(_on_export_done)
            except TypeError:
                pass
            if ex.stop_requested:
                ex.finish_node(key, False, "已停止")
                return
            ok = bool(result.get("success"))
            msg = result.get("path") or result.get("error") or ""
            if ok and isinstance(msg, str) and len(msg) > 48:
                msg = os.path.basename(msg)
            ex.finish_node(key, ok, str(msg) if msg else ("完成" if ok else "失败"))

        self.export_panel.export_finished.connect(_on_export_done)
        started = self.export_panel.export_model(workflow_mode=True)
        if not started:
            try:
                self.export_panel.export_finished.disconnect(_on_export_done)
            except TypeError:
                pass
            ex.finish_node(key, False, "无法启动导出（检查模型与格式依赖）")

    def _wf_run_quality(self, key: str, ex: WorkflowExecutor) -> None:
        image_dir = self.current_image_dir
        if not image_dir and self.image_list:
            image_dir = str(Path(self.image_list[0]).parent)
        if not image_dir or not os.path.isdir(image_dir):
            ex.finish_node(key, False, "无图片目录可质检")
            return
        from gui.annotation_io import labels_dir_for_image_dir
        from core.workflow_optimizer import DataQualityChecker

        labels_dir = labels_dir_for_image_dir(image_dir)
        classes = self.class_manager.get_all_classes()
        metrics = DataQualityChecker().check_quality(image_dir, labels_dir, classes)
        missing = len(getattr(metrics, "missing_annotations", []) or [])
        detail = f"{metrics.total_annotations}框/{metrics.total_images}图"
        if missing:
            detail += f"，缺标{missing}"
        ex.finish_node(key, True, detail)

    def _apply_project_task(self, task: str) -> None:
        from gui.canvas import CanvasMode

        task = (task or "detect").lower()
        mode_map = {
            "detect": CanvasMode.CREATE_BBOX,
            "segment": CanvasMode.CREATE_POLYGON,
            "pose": CanvasMode.CREATE_KEYPOINT,
            "obb": CanvasMode.CREATE_OBB,
            "classify": CanvasMode.EDIT,
        }
        mode = mode_map.get(task, CanvasMode.CREATE_BBOX)
        if hasattr(self, "_set_canvas_mode"):
            self._set_canvas_mode(mode)
        elif hasattr(self, "canvas"):
            self.canvas.set_mode(mode)

        if hasattr(self, "canvas") and hasattr(self.canvas, "num_keypoints"):
            self.canvas.num_keypoints = 17 if task == "pose" else 0

        model = _TASK_MODEL.get(task)
        if model and hasattr(self, "training_panel"):
            combo = getattr(self.training_panel, "model_combo", None)
            if combo is not None:
                current = combo.currentText().strip().lower()
                if not current or current in {
                    "yolo11n.pt", "yolov8n.pt", "yolo11n-seg.pt",
                    "yolo11n-pose.pt", "yolo11n-obb.pt", "yolo11n-cls.pt",
                }:
                    combo.setCurrentText(model)

        if task == "classify" and hasattr(self, "statusBar"):
            self.statusBar().showMessage(
                "分类任务一般用文件夹当类别，画布已切到选择模式", 5000
            )

    # ------------------------------------------------------------------
    # Workspace switching
    # ------------------------------------------------------------------
