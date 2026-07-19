"""Small reusable widgets for the desktop UI."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QSizePolicy


class StatusPill(QLabel):
    def __init__(self, text: str = "", variant: str = "", parent=None):
        super().__init__(text, parent)
        self.setObjectName("StatusPill")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        if variant:
            self.setProperty("variant", variant)

    def set_variant(self, variant: str) -> None:
        self.setProperty("variant", variant)
        self.style().unpolish(self)
        self.style().polish(self)
