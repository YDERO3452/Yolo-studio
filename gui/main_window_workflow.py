"""Workflow canvas execution handlers for MainWindow."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from gui.workflow_executor import WorkflowExecutor


class WorkflowOpsMixin:
    def _setup_workflow_executor(self) -> None:
        """Wire node canvas run/stop to real panel actions."""
        self.workflow_executor = WorkflowExecutor(self)
        panel = self.workflow_panel
        panel.run_requested.connect(self._start_workflow_run)
        panel.stop_requested.connect(self.workflow_executor.stop)

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
        # Close any open overlay so the run starts from the canvas.
        if hasattr(self, "_return_to_workflow"):
            self._return_to_workflow()
        keys, edges = self.workflow_panel.collect_graph()
        # Only mark runnable (main) nodes; sub/utility nodes stay out of the run.
        self.workflow_panel.reset_all_status()
        for k in keys:
            self.workflow_panel.set_node_status(k, "pending")
        self.workflow_executor.start(keys, edges)

    def _on_workflow_pipeline_finished(self, ok: bool, summary: str) -> None:
        self.workflow_panel.set_running_ui(False)
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
        if not yaml_path:
            ex.finish_node(key, False, "未找到 data.yaml，请先在「数据」页加载或创建")
            return
        from core.dataset import DatasetManager

        dataset_dir = str(Path(yaml_path).resolve().parent)
        issues = DatasetManager(dataset_dir).validate_dataset(dataset_dir)
        # Sync yaml into training panel for later nodes
        self.training_panel.data_yaml_edit.setText(yaml_path)
        self.dataset_panel.data_yaml_edit.setText(yaml_path)
        if issues:
            ex.finish_node(key, False, f"校验失败: {issues[0]}")
            return
        ex.finish_node(key, True, Path(yaml_path).name)

    def _wf_run_annotate(self, key: str, ex: WorkflowExecutor) -> None:
        if not self.image_list:
            ex.finish_node(key, False, "无图片 — 请先打开图片目录")
            return
        labeled = 0
        for image_path in self.image_list:
            try:
                from gui.annotation_io import label_path_for_image

                lp = label_path_for_image(image_path)
                if lp and os.path.isfile(lp) and os.path.getsize(lp) > 0:
                    labeled += 1
            except Exception:
                continue
        total = len(self.image_list)
        # Annotation is a human gate: pass if there is at least a queue.
        ex.finish_node(key, True, f"队列 {total}，已标注 {labeled}")

    def _wf_run_train(self, key: str, ex: WorkflowExecutor) -> None:
        yaml_path = self._wf_resolve_data_yaml()
        if yaml_path:
            self.training_panel.data_yaml_edit.setText(yaml_path)

        def _on_finished(result: dict) -> None:
            try:
                self.training_panel.training_finished.disconnect(_on_finished)
            except TypeError:
                pass
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

    # ------------------------------------------------------------------
    # Workspace switching
    # ------------------------------------------------------------------

