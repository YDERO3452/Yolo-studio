"""Training result browser panel."""

from __future__ import annotations

import csv
import os
import shutil
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class TrainingResultsPanel(QWidget):
    """Browse YOLO run folders and reuse/export trained weights."""

    load_inference_requested = pyqtSignal(str)
    load_export_requested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_project: dict | None = None
        self.current_run_dir: str | None = None
        self._build_ui()
        self.refresh_runs()

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)
        left.setMinimumWidth(320)
        left.setMaximumWidth(440)

        self.project_label = QLabel("训练结果")
        self.project_label.setObjectName("PanelTitle")
        left_layout.addWidget(self.project_label)

        self.run_list = QListWidget()
        self.run_list.itemSelectionChanged.connect(self._on_run_selected)
        left_layout.addWidget(self.run_list, 1)

        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self.refresh_runs)
        left_layout.addWidget(refresh_btn)
        layout.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        self.detail_label = QLabel("未选择训练结果")
        self.detail_label.setObjectName("PanelTitle")
        right_layout.addWidget(self.detail_label)

        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)
        right_layout.addWidget(self.detail_text, 1)

        action_row = QHBoxLayout()
        self.infer_btn = QPushButton("用于推理")
        self.infer_btn.setObjectName("PrimaryButton")
        self.infer_btn.clicked.connect(self.load_for_inference)
        self.export_btn = QPushButton("用于导出")
        self.export_btn.clicked.connect(self.load_for_export)
        self.copy_model_btn = QPushButton("导出 best.pt")
        self.copy_model_btn.clicked.connect(self.copy_best_model)
        self.copy_folder_btn = QPushButton("导出结果文件夹")
        self.copy_folder_btn.clicked.connect(self.copy_result_folder)
        for btn in (self.infer_btn, self.export_btn, self.copy_model_btn, self.copy_folder_btn):
            action_row.addWidget(btn)
        action_row.addStretch()
        right_layout.addLayout(action_row)
        layout.addWidget(right, 1)

    def set_project(self, project: dict | None) -> None:
        self.current_project = project if project and project.get("root") else None
        if self.current_project:
            self.project_label.setText(f"训练结果 - {self.current_project.get('name', Path(self.current_project['root']).name)}")
        else:
            self.project_label.setText("训练结果")
        self.refresh_runs()

    def refresh_runs(self) -> None:
        self.run_list.clear()
        runs = self._find_runs()
        for run_dir in runs:
            item = QListWidgetItem(self._display_name(run_dir))
            item.setToolTip(str(run_dir))
            item.setData(Qt.ItemDataRole.UserRole, str(run_dir))
            self.run_list.addItem(item)
        if self.run_list.count():
            self.run_list.setCurrentRow(0)
        else:
            self.current_run_dir = None
            self.detail_label.setText("未找到训练结果")
            self.detail_text.setPlainText("训练完成后会在这里显示 results.csv、weights/best.pt 和输出图表。")

    def _find_runs(self) -> list[Path]:
        roots: list[Path] = []
        if self.current_project:
            roots.append(Path(self.current_project["root"]) / "runs")
        roots.extend([Path("runs"), Path("runs/train"), Path("runs/detect")])
        found: set[Path] = set()
        for root in roots:
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if not path.is_dir():
                    continue
                if (path / "results.csv").exists() or (path / "weights" / "best.pt").exists():
                    found.add(path.resolve())
        return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)

    def _on_run_selected(self) -> None:
        items = self.run_list.selectedItems()
        if not items:
            return
        self.current_run_dir = items[0].data(Qt.ItemDataRole.UserRole)
        self._show_run(self.current_run_dir)

    def _show_run(self, run_dir: str) -> None:
        path = Path(run_dir)
        self.detail_label.setText(path.name)
        lines = [f"目录: {path}"]
        best = path / "weights" / "best.pt"
        last = path / "weights" / "last.pt"
        lines.append(f"best.pt: {'存在' if best.exists() else '缺失'}")
        lines.append(f"last.pt: {'存在' if last.exists() else '缺失'}")
        metrics = self._read_last_metrics(path / "results.csv")
        if metrics:
            lines.append("")
            lines.append("最后一轮指标:")
            for key, value in metrics.items():
                lines.append(f"  {key}: {value}")
        artifacts = sorted([p.name for p in path.glob("*.png")] + [p.name for p in path.glob("*.jpg")])
        if artifacts:
            lines.append("")
            lines.append("图表/预览:")
            lines.extend(f"  {name}" for name in artifacts[:24])
        self.detail_text.setPlainText("\n".join(lines))

    def load_for_inference(self) -> None:
        best = self._best_model()
        if best:
            self.load_inference_requested.emit(str(best))

    def load_for_export(self) -> None:
        best = self._best_model()
        if best:
            self.load_export_requested.emit(str(best))

    def copy_best_model(self) -> None:
        best = self._best_model()
        if not best:
            return
        save_path, _ = QFileDialog.getSaveFileName(self, "导出 best.pt", "best.pt", "PyTorch 模型 (*.pt)")
        if save_path:
            shutil.copy2(best, save_path)
            QMessageBox.information(self, "完成", f"已导出模型:\n{save_path}")

    def copy_result_folder(self) -> None:
        if not self.current_run_dir:
            return
        target_root = QFileDialog.getExistingDirectory(self, "选择导出目录")
        if not target_root:
            return
        src = Path(self.current_run_dir)
        dest = Path(target_root) / src.name
        if dest.exists():
            reply = QMessageBox.question(
                self,
                "覆盖目录",
                f"{dest} 已存在，是否合并覆盖？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        shutil.copytree(src, dest, dirs_exist_ok=True)
        QMessageBox.information(self, "完成", f"已导出结果文件夹:\n{dest}")

    def _best_model(self) -> Path | None:
        if not self.current_run_dir:
            return None
        best = Path(self.current_run_dir) / "weights" / "best.pt"
        if not best.exists():
            QMessageBox.warning(self, "提示", "当前训练结果没有 weights/best.pt")
            return None
        return best

    @staticmethod
    def _display_name(run_dir: Path) -> str:
        parent = run_dir.parent.name
        if parent in {"train", "detect", "runs"}:
            return run_dir.name
        return f"{parent}/{run_dir.name}"

    @staticmethod
    def _read_last_metrics(results_csv: Path) -> dict[str, str]:
        if not results_csv.exists():
            return {}
        try:
            with results_csv.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            if not rows:
                return {}
            last = rows[-1]
            keep = [
                "epoch",
                "train/box_loss",
                "train/cls_loss",
                "val/box_loss",
                "metrics/precision(B)",
                "metrics/recall(B)",
                "metrics/mAP50(B)",
                "metrics/mAP50-95(B)",
            ]
            return {
                key.strip(): str(last.get(key, "")).strip()
                for key in keep
                if key in last and str(last.get(key, "")).strip()
            }
        except Exception:
            return {}
