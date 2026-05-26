"""File list widget for image queue navigation."""

import os
from typing import List

from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPixmap, QIcon
from PyQt6.QtWidgets import QListWidget, QListWidgetItem

from gui.annotation_io import label_path_for_image
from gui.theme import Theme


class FileListWidget(QListWidget):
    file_selected = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_row = -1
        self.setIconSize(QSize(10, 10))
        self.itemClicked.connect(self._on_item_clicked)

    def load_image_list(self, image_list: List[str]) -> None:
        self.clear()
        for index, path in enumerate(image_list):
            item = QListWidgetItem(os.path.basename(path))
            item.setToolTip(path)
            item.setData(Qt.ItemDataRole.UserRole, index)
            item.setIcon(self._status_icon(os.path.exists(label_path_for_image(path))))
            self.addItem(item)

    def highlight_current(self, index: int) -> None:
        if 0 <= index < self.count():
            self.setCurrentRow(index)
            self.current_row = index

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        index = item.data(Qt.ItemDataRole.UserRole)
        if index is not None:
            self.file_selected.emit(index)

    @staticmethod
    def _status_icon(done: bool) -> QIcon:
        pixmap = QPixmap(10, 10)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QColor(Theme.SUCCESS if done else Theme.TEXT_DIM))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(1, 1, 8, 8)
        painter.end()
        return QIcon(pixmap)
