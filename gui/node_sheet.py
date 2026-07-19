"""Node-sheet host: expanded workflow-node chrome for open modules."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from gui.theme import Theme


class NodeSheet(QWidget):
    """Large card that looks like an expanded workflow node.

    Left accent strip + title/subtitle + close; body hosts any stage panel.
    """

    closed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("NodeSheet")
        self._accent = Theme.ACCENT

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._accent_bar = QFrame()
        self._accent_bar.setObjectName("NodeSheetAccent")
        self._accent_bar.setFixedWidth(5)
        self._accent_bar.setStyleSheet(f"background: {self._accent}; border: none;")
        root.addWidget(self._accent_bar)

        main = QWidget()
        main.setObjectName("NodeSheetBody")
        main_layout = QVBoxLayout(main)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        header = QWidget()
        header.setObjectName("NodeSheetHeader")
        header.setFixedHeight(44)
        header_row = QHBoxLayout(header)
        header_row.setContentsMargins(14, 6, 10, 6)
        header_row.setSpacing(10)

        titles = QVBoxLayout()
        titles.setContentsMargins(0, 0, 0, 0)
        titles.setSpacing(0)
        self.title_label = QLabel("")
        self.title_label.setObjectName("PageTitle")
        titles.addWidget(self.title_label)
        self.subtitle_label = QLabel("")
        self.subtitle_label.setObjectName("MutedText")
        titles.addWidget(self.subtitle_label)
        header_row.addLayout(titles, stretch=1)

        self.close_btn = QPushButton("关闭")
        self.close_btn.setFixedSize(56, 28)
        self.close_btn.setToolTip("关闭节点卡，返回画布")
        self.close_btn.clicked.connect(self.closed.emit)
        header_row.addWidget(self.close_btn, alignment=Qt.AlignmentFlag.AlignTop)

        main_layout.addWidget(header)

        self.body_host = QWidget()
        self.body_layout = QVBoxLayout(self.body_host)
        self.body_layout.setContentsMargins(0, 0, 0, 0)
        self.body_layout.setSpacing(0)
        main_layout.addWidget(self.body_host, stretch=1)

        root.addWidget(main, stretch=1)
        self._apply_chrome()

    def _apply_chrome(self) -> None:
        self.setStyleSheet(
            f"""
            QWidget#NodeSheet {{
                background: {Theme.SURFACE_2};
                border: 1px solid {Theme.BORDER_STRONG};
            }}
            QWidget#NodeSheetHeader {{
                background: {Theme.SURFACE};
                border-bottom: 1px solid {Theme.BORDER};
            }}
            QWidget#NodeSheetBody {{
                background: {Theme.SURFACE_2};
            }}
            """
        )

    def set_meta(self, title: str, subtitle: str = "", accent: str | None = None) -> None:
        self.title_label.setText(title)
        self.subtitle_label.setText(subtitle)
        self.subtitle_label.setVisible(bool(subtitle))
        if accent:
            self._accent = accent
            self._accent_bar.setStyleSheet(f"background: {accent}; border: none;")

    def set_body(self, widget: QWidget) -> None:
        while self.body_layout.count():
            item = self.body_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
        self.body_layout.addWidget(widget, stretch=1)
