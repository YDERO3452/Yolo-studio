"""SAM memory object management dialog."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QDialog, QHBoxLayout, QLabel, QListWidget, QPushButton, QVBoxLayout


class SAMMemoryObjectsDialog(QDialog):
    add_requested = pyqtSignal()
    delete_requested = pyqtSignal(int)
    save_requested = pyqtSignal()
    single_requested = pyqtSignal()
    batch_requested = pyqtSignal()
    clear_requested = pyqtSignal()
    closed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SAM 记忆对象")
        self.setMinimumWidth(460)
        self.setModal(False)

        layout = QVBoxLayout(self)
        self.info_label = QLabel("在画布上添加点或框提示后，点击添加对象。对象 ID 相同会更新已有记忆。")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        self.count_label = QLabel("当前对象: 0")
        layout.addWidget(self.count_label)

        self.obj_list = QListWidget()
        layout.addWidget(self.obj_list, 1)

        edit_row = QHBoxLayout()
        self.add_btn = QPushButton("添加对象")
        self.delete_btn = QPushButton("删除对象")
        self.clear_btn = QPushButton("清空记忆")
        edit_row.addWidget(self.add_btn)
        edit_row.addWidget(self.delete_btn)
        edit_row.addWidget(self.clear_btn)
        layout.addLayout(edit_row)

        run_row = QHBoxLayout()
        self.save_btn = QPushButton("保存并更新")
        self.save_btn.setObjectName("PrimaryButton")
        self.single_btn = QPushButton("单张推理")
        self.batch_btn = QPushButton("批量推理")
        run_row.addWidget(self.save_btn)
        run_row.addWidget(self.single_btn)
        run_row.addWidget(self.batch_btn)
        layout.addLayout(run_row)

        self.add_btn.clicked.connect(self.add_requested.emit)
        self.delete_btn.clicked.connect(self._delete_current)
        self.clear_btn.clicked.connect(self.clear_requested.emit)
        self.save_btn.clicked.connect(self.save_requested.emit)
        self.single_btn.clicked.connect(self.single_requested.emit)
        self.batch_btn.clicked.connect(self.batch_requested.emit)

    def update_objects(self, objects: list[dict]) -> None:
        self.obj_list.clear()
        for obj in objects:
            class_name = obj.get("class_name") or f"class_{obj.get('class_id', 0)}"
            point_count = len(obj.get("points") or [])
            box_count = len(obj.get("bboxes") or [])
            self.obj_list.addItem(
                f"ID={obj.get('obj_id')}  {class_name}  points={point_count}  boxes={box_count}"
            )
        self.count_label.setText(f"当前对象: {len(objects)}")

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)

    def _delete_current(self) -> None:
        row = self.obj_list.currentRow()
        if row >= 0:
            self.delete_requested.emit(row)
