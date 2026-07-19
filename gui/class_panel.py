"""Class list panel widget for annotation tab.

标签列表设计要点:
- 点击类别直接切换当前绘制类别
- 显示颜色色块 + 快捷键编号
- 支持数字快捷键 1-9 快速切换
- 双击类别弹出编辑
"""

from typing import Optional

from loguru import logger
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QPixmap
from PyQt6.QtWidgets import (
    QColorDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.class_manager import ClassManager
from gui.theme import Theme


class ClassListPanel(QWidget):
    """Panel for displaying and managing annotation classes.

    Features:
    - 点击类别即切换画布当前绘制类别
    - 显示快捷键编号 (1-9)
    - 右键上下文菜单 (重命名/改颜色/删除)
    """

    # Signals
    class_selected = pyqtSignal(str)  # Emitted when a class is selected
    class_added = pyqtSignal(str)     # Emitted when a class is added
    class_removed = pyqtSignal(str)   # Emitted when a class is removed
    class_renamed = pyqtSignal(str, str)  # Emitted when a class is renamed (old, new)
    class_color_changed = pyqtSignal(str, tuple)  # Emitted when color changes (class_name, (r,g,b))
    class_id_selected = pyqtSignal(int)  # Emitted with class index for canvas sync

    def __init__(self, class_manager: ClassManager, parent=None):
        super().__init__(parent)
        self.class_manager = class_manager
        self.current_selected_class: Optional[str] = None
        self._current_class_id: int = 0
        self.init_ui()
        self.refresh_list()

    def init_ui(self):
        """Initialize UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(3, 3, 3, 3)
        layout.setSpacing(3)

        # Title row
        title_layout = QHBoxLayout()
        title_layout.setSpacing(4)
        title_label = QLabel("类别列表")
        title_label.setObjectName("SectionTitle")
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        self.current_class_label = QLabel("当前: 目标")
        self.current_class_label.setObjectName("ClassCurrent")
        title_layout.addWidget(self.current_class_label)
        layout.addLayout(title_layout)

        # Class list
        self.class_list_widget = QListWidget()
        self.class_list_widget.itemClicked.connect(self.on_class_clicked)
        self.class_list_widget.itemDoubleClicked.connect(self._on_class_double_clicked)
        self.class_list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.class_list_widget.customContextMenuRequested.connect(self.show_context_menu)
        layout.addWidget(self.class_list_widget)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(3)
        self.add_btn = QPushButton("添加")
        self.add_btn.setObjectName("PrimaryButton")
        self.add_btn.setProperty("compact", True)
        self.add_btn.setFixedHeight(20)
        self.add_btn.clicked.connect(self.add_class)
        btn_layout.addWidget(self.add_btn)

        self.remove_btn = QPushButton("删除")
        self.remove_btn.setObjectName("DangerButton")
        self.remove_btn.setProperty("compact", True)
        self.remove_btn.setFixedHeight(20)
        self.remove_btn.clicked.connect(self.remove_class)
        btn_layout.addWidget(self.remove_btn)

        self.color_btn = QPushButton("颜色")
        self.color_btn.setProperty("compact", True)
        self.color_btn.setFixedHeight(20)
        self.color_btn.clicked.connect(self.change_color)
        btn_layout.addWidget(self.color_btn)
        layout.addLayout(btn_layout)

        # Count
        self.count_label = QLabel("共 0 个类别")
        self.count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.count_label.setObjectName("MutedText")
        layout.addWidget(self.count_label)

    def refresh_list(self):
        """Refresh the class list display."""
        self.class_list_widget.clear()

        for i, class_name in enumerate(self.class_manager.get_all_classes()):
            color = self.class_manager.get_color(class_name)
            item = QListWidgetItem(f"{i+1}. {class_name}")

            # Create color icon (larger for visibility)
            pixmap = QPixmap(20, 20)
            pixmap.fill(QColor(*color))
            item.setIcon(QIcon(pixmap))

            # Highlight current class
            if i == self._current_class_id:
                font = item.font()
                font.setBold(True)
                item.setFont(font)
                item.setBackground(QColor(197, 139, 66, 48))

            self.class_list_widget.addItem(item)

        # Update count
        count = len(self.class_manager)
        self.count_label.setText(f"共 {count} 个类别")

        # Update current class indicator
        self._update_current_class_label()

    def _update_current_class_label(self):
        """Update the 'current class' indicator label."""
        classes = self.class_manager.get_all_classes()
        if 0 <= self._current_class_id < len(classes):
            name = classes[self._current_class_id]
            color = self.class_manager.get_color(name)
            r, g, b = color if color else (48, 209, 88)
            self.current_class_label.setText(f"当前: {name}")
            self.current_selected_class = name
            self.current_class_label.setStyleSheet(
                f"color: rgb({r},{g},{b}); font-weight: bold; font-size: 11px; "
                f"padding: 2px 6px; border: 1px solid {Theme.BORDER}; "
                f"border-radius: 0px; background: rgba({r},{g},{b},26);"
            )
        else:
            self.current_class_label.setText("当前: —")
            self.current_selected_class = None
            self.current_class_label.setStyleSheet("")

    def set_current_class_id(self, class_id: int):
        """Set the current active class by index."""
        classes = self.class_manager.get_all_classes()
        if 0 <= class_id < len(classes):
            self._current_class_id = class_id
            self.current_selected_class = classes[class_id]
            # Highlight in list
            self.class_list_widget.setCurrentRow(class_id)
            self._update_current_class_label()
            self.class_selected.emit(classes[class_id])
            self.class_id_selected.emit(class_id)
            logger.info(f"Switched to class: {classes[class_id]} (id={class_id})")

    def get_current_class_id(self) -> int:
        """Get the current active class index."""
        return self._current_class_id

    def on_class_clicked(self, item: QListWidgetItem):
        """Handle class selection — 切换当前绘制类别."""
        row = self.class_list_widget.row(item)
        if row >= 0:
            self.set_current_class_id(row)

    def _on_class_double_clicked(self, item: QListWidgetItem):
        """Double-click on class to rename it."""
        class_name = item.text().split(". ", 1)[1] if ". " in item.text() else item.text()
        self.rename_class(class_name)

    def add_class(self):
        """Add a new class."""
        text, ok = QInputDialog.getText(
            self, "添加类别", "输入新类别名称:"
        )
        if ok and text.strip():
            class_name = text.strip()
            if self.class_manager.add_class(class_name):
                self.refresh_list()
                self.class_added.emit(class_name)
                logger.info(f"Added class: {class_name}")
            else:
                QMessageBox.warning(self, "错误", f"类别 '{class_name}' 已存在")

    def remove_class(self):
        """Remove the selected class."""
        if not self.current_selected_class:
            QMessageBox.warning(self, "错误", "请先选择一个类别")
            return

        reply = QMessageBox.question(
            self, "确认删除",
            f"确定要删除类别 '{self.current_selected_class}' 吗？"
        )
        if reply == QMessageBox.StandardButton.Yes:
            class_name = self.current_selected_class
            if self.class_manager.remove_class(class_name):
                self.current_selected_class = None
                self._current_class_id = 0
                self.refresh_list()
                self.class_removed.emit(class_name)
                logger.info(f"Removed class: {class_name}")

    def change_color(self):
        """Change the color of the selected class."""
        if not self.current_selected_class:
            QMessageBox.warning(self, "错误", "请先选择一个类别")
            return

        current_color = self.class_manager.get_color(self.current_selected_class)
        color = QColorDialog.getColor(
            QColor(*current_color),
            self,
            f"选择 '{self.current_selected_class}' 的颜色"
        )

        if color.isValid():
            rgb = (color.red(), color.green(), color.blue())
            self.class_manager.set_color(self.current_selected_class, rgb)
            self.refresh_list()
            self.class_color_changed.emit(self.current_selected_class, rgb)
            logger.info(f"Changed color for '{self.current_selected_class}': {rgb}")

    def show_context_menu(self, position):
        """Show context menu for class list."""
        item = self.class_list_widget.itemAt(position)
        if not item:
            return

        row = self.class_list_widget.row(item)
        class_name = self.class_manager.get_all_classes()[row] if row < len(self.class_manager.get_all_classes()) else item.text()
        menu = QMenu(self)

        # Select action
        select_action = menu.addAction("选择为当前类别")
        select_action.triggered.connect(lambda: self.set_current_class_id(row))

        menu.addSeparator()

        # Rename action
        rename_action = menu.addAction("重命名")
        rename_action.triggered.connect(lambda: self.rename_class(class_name))

        # Change color action
        color_action = menu.addAction("改变颜色")
        color_action.triggered.connect(lambda: self._change_color_for(class_name))

        menu.addSeparator()

        # Delete action
        delete_action = menu.addAction("删除")
        delete_action.triggered.connect(lambda: self._delete_class(class_name))

        menu.exec(self.class_list_widget.mapToGlobal(position))

    def rename_class(self, class_name: str):
        """Rename a class."""
        text, ok = QInputDialog.getText(
            self, "重命名类别",
            f"输入新名称 (当前: {class_name}):",
            text=class_name
        )
        if ok and text.strip() and text != class_name:
            new_name = text.strip()
            if self.class_manager.rename_class(class_name, new_name):
                self.refresh_list()
                self.class_renamed.emit(class_name, new_name)
                logger.info(f"Renamed class: {class_name} -> {new_name}")
            else:
                QMessageBox.warning(self, "错误", "无法重命名类别")

    def _change_color_for(self, class_name: str):
        """Change color for a specific class."""
        self.current_selected_class = class_name
        self.change_color()

    def _delete_class(self, class_name: str):
        """Delete a specific class."""
        self.current_selected_class = class_name
        self.remove_class()

    def get_selected_class(self) -> Optional[str]:
        """Get the currently selected class name."""
        return self.current_selected_class

    def set_selected_class(self, class_name: str):
        """Set the selected class by name."""
        classes = self.class_manager.get_all_classes()
        if class_name in classes:
            idx = classes.index(class_name)
            self.set_current_class_id(idx)

    def update_class_manager(self, class_manager: ClassManager):
        """Update the class manager reference."""
        self.class_manager = class_manager
        self.refresh_list()
