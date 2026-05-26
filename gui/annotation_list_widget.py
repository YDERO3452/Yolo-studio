"""Annotation list widget for viewing/editing shapes."""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QTreeWidget, QTreeWidgetItem, QHeaderView, QMenu

from core.annotation import ShapeType
from gui.annotation_io import shape_type_value


class AnnotationListWidget(QTreeWidget):
    annotation_selected = pyqtSignal(int)
    annotation_delete_requested = pyqtSignal(int)
    annotation_edit_requested = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderLabels(["类别", "类型", "置信"])
        self.header().setStretchLastSection(False)
        self.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.header().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.setRootIsDecorated(False)
        self.setIndentation(0)
        self.setSelectionBehavior(QTreeWidget.SelectionBehavior.SelectRows)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
        self.currentItemChanged.connect(self._on_item_changed)
        self.itemDoubleClicked.connect(self._on_item_double_clicked)

    def refresh(self, shapes: list) -> None:
        self.clear()
        type_names = {
            ShapeType.BBOX.value: "矩形",
            ShapeType.POLYGON.value: "多边形",
            ShapeType.OBB.value: "OBB",
            ShapeType.KEYPOINT.value: "关键点",
        }
        for index, shape in enumerate(shapes):
            class_name = shape.get("class_name", f"类别_{shape.get('class_id', 0)}")
            stype = type_names.get(shape_type_value(shape.get("type", "")), str(shape.get("type", "")))
            conf = shape.get("confidence", 0)
            conf_text = f"{conf:.0%}" if conf else "-"
            item = QTreeWidgetItem([class_name, stype, conf_text])
            item.setData(0, Qt.ItemDataRole.UserRole, index)
            self.addTopLevelItem(item)

    def highlight_shape(self, index: int) -> None:
        if 0 <= index < self.topLevelItemCount():
            self.setCurrentItem(self.topLevelItem(index))

    def _on_item_changed(self, current, previous) -> None:
        if current:
            index = current.data(0, Qt.ItemDataRole.UserRole)
            if index is not None:
                self.annotation_selected.emit(index)

    def _on_item_double_clicked(self, item, column) -> None:
        index = item.data(0, Qt.ItemDataRole.UserRole)
        if index is not None:
            self.annotation_edit_requested.emit(index)

    def _show_context_menu(self, pos) -> None:
        item = self.itemAt(pos)
        if not item:
            return
        index = item.data(0, Qt.ItemDataRole.UserRole)
        menu = QMenu(self)
        edit_action = menu.addAction("编辑标签")
        delete_action = menu.addAction("删除")
        action = menu.exec(self.mapToGlobal(pos))
        if action == edit_action:
            self.annotation_edit_requested.emit(index)
        elif action == delete_action:
            self.annotation_delete_requested.emit(index)
