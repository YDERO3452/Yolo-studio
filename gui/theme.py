"""Application theme for the YOLO Studio desktop workbench."""

from __future__ import annotations


class Theme:
    BG = "#121416"
    RAIL = "#0E1012"
    SURFACE = "#181B1E"
    SURFACE_2 = "#202429"
    SURFACE_3 = "#292F35"
    BORDER = "#2D3339"
    BORDER_STRONG = "#414950"
    TEXT = "#E7E9EC"
    TEXT_MUTED = "#A8AFB7"
    TEXT_DIM = "#727A83"
    ACCENT = "#4F7FAF"
    ACCENT_HOVER = "#5B8CBB"
    ACCENT_DARK = "#365F87"
    SUCCESS = "#659A75"
    WARNING = "#C49347"
    DANGER = "#C4605C"
    SELECTION = "#203247"
    CANVAS_BG = "#151719"


def build_stylesheet() -> str:
    """Return the global Qt stylesheet."""
    return f"""
    QMainWindow, QDialog, QMessageBox {{
        background: {Theme.BG};
    }}
    QWidget {{
        color: {Theme.TEXT};
        font-family: "Microsoft YaHei UI", "Segoe UI", "Noto Sans CJK SC", "WenQuanYi Micro Hei", sans-serif;
        font-size: 13px;
    }}
    QLabel {{
        background: transparent;
        color: {Theme.TEXT};
    }}
    QDialog QWidget {{
        background: transparent;
    }}
    QMessageBox QLabel {{
        color: {Theme.TEXT};
        background: transparent;
    }}
    QToolTip {{
        color: {Theme.TEXT};
        background: {Theme.SURFACE_3};
        border: 1px solid {Theme.BORDER_STRONG};
        padding: 4px 6px;
    }}
    QWidget#TopBar {{
        background: {Theme.SURFACE};
        border-bottom: 1px solid {Theme.BORDER};
    }}
    QWidget#WorkspaceTabs {{
        background: {Theme.SURFACE};
        border-bottom: 1px solid {Theme.BORDER};
        min-height: 34px;
        max-height: 34px;
    }}
    QPushButton#WorkspaceTab {{
        background: {Theme.SURFACE_2};
        color: {Theme.TEXT_MUTED};
        border: 1px solid {Theme.BORDER};
        border-bottom: none;
        border-radius: 6px 6px 0 0;
        padding: 7px 18px;
        min-width: 70px;
        font-weight: 700;
    }}
    QPushButton#WorkspaceTab:hover {{
        color: {Theme.TEXT};
        background: {Theme.SURFACE_3};
    }}
    QPushButton#WorkspaceTab:checked {{
        color: {Theme.TEXT};
        background: #111820;
        border-color: {Theme.ACCENT};
        border-bottom: 2px solid {Theme.ACCENT};
    }}
    QWidget#ToolRail, QWidget#NavRail, QScrollArea#NavRailScroll {{
        background: {Theme.RAIL};
        border-right: 1px solid {Theme.BORDER};
    }}
    QWidget#Inspector {{
        background: {Theme.SURFACE};
        border-left: 1px solid {Theme.BORDER};
    }}
    QWidget#InspectorContent, QWidget#InspectorViewport {{
        background: {Theme.SURFACE};
    }}
    QWidget#CanvasInspectorSeparator {{
        background: {Theme.BORDER};
    }}
    QWidget#WorkspaceHeader {{
        background: transparent;
        border-bottom: 1px solid {Theme.BORDER};
    }}
    QWidget#AnnotationControlBar {{
        background: {Theme.SURFACE};
        border-bottom: 1px solid {Theme.BORDER};
    }}
    QLabel#ModeHint {{
        color: {Theme.TEXT_MUTED};
        font-size: 12px;
    }}
    QLabel#InlineStatus {{
        color: {Theme.SUCCESS};
        font-size: 12px;
        font-weight: 600;
    }}
    QWidget#Card, QFrame#Card {{
        background: {Theme.SURFACE_2};
        border: 1px solid {Theme.BORDER};
        border-radius: 6px;
    }}
    QLabel#BrandTitle, QLabel#PageTitle {{
        color: {Theme.TEXT};
        font-size: 15px;
        font-weight: 600;
    }}
    QLabel#PanelTitle {{
        color: {Theme.TEXT};
        font-size: 14px;
        font-weight: 600;
    }}
    QLabel#BrandSubtitle, QLabel#MutedText {{
        color: {Theme.TEXT_MUTED};
        font-size: 12px;
    }}
    QLabel#SectionTitle {{
        color: {Theme.TEXT};
        font-size: 13px;
        font-weight: 600;
        padding: 0 0 4px 0;
    }}
    QLabel#StatusPill {{
        color: {Theme.TEXT_MUTED};
        background: {Theme.SURFACE_2};
        border: 1px solid {Theme.BORDER};
        border-radius: 4px;
        padding: 2px 7px;
        font-size: 11px;
    }}
    QLabel#StatusPill[variant="accent"] {{
        color: #9FC0DE;
        background: rgba(79, 127, 175, 0.14);
        border-color: #3F668D;
        font-weight: 600;
    }}
    QLabel#StatusPill[variant="success"] {{
        color: #91B99D;
        background: rgba(101, 154, 117, 0.14);
        border-color: #4C7358;
        font-weight: 600;
    }}
    QLabel#StatusPill[variant="warning"] {{
        color: #D6B77E;
        background: rgba(196, 147, 71, 0.14);
        border-color: #806432;
        font-weight: 600;
    }}
    QLabel#ClassCurrent {{
        border: 1px solid {Theme.BORDER};
        border-radius: 4px;
        padding: 2px 7px;
        font-size: 11px;
        font-weight: 600;
    }}
    QLabel#PreviewSurface {{
        background: {Theme.CANVAS_BG};
        border: 1px solid {Theme.BORDER};
        border-radius: 6px;
        color: {Theme.TEXT_MUTED};
    }}
    QPushButton {{
        background: {Theme.SURFACE_2};
        color: {Theme.TEXT};
        border: 1px solid {Theme.BORDER};
        border-radius: 4px;
        padding: 0 10px;
        min-height: 28px;
    }}
    QPushButton:hover {{
        background: {Theme.SURFACE_3};
        border-color: {Theme.BORDER_STRONG};
    }}
    QPushButton:pressed {{
        background: #1A1E22;
    }}
    QPushButton:disabled {{
        color: {Theme.TEXT_DIM};
        background: #1A1E21;
        border-color: #2B3136;
    }}
    QPushButton:checked {{
        color: {Theme.TEXT};
        background: {Theme.SELECTION};
        border-color: {Theme.ACCENT};
        font-weight: 600;
    }}
    QPushButton#PrimaryButton {{
        color: #FFFFFF;
        background: {Theme.ACCENT};
        border-color: {Theme.ACCENT};
        font-weight: 600;
        min-height: 30px;
    }}
    QPushButton#PrimaryButton:hover {{
        background: {Theme.ACCENT_HOVER};
        border-color: {Theme.ACCENT_HOVER};
    }}
    QPushButton#SecondaryButton {{
        background: {Theme.SURFACE_3};
        border-color: {Theme.BORDER_STRONG};
        font-weight: 600;
    }}
    QPushButton#SecondaryButton:hover {{
        background: {Theme.SELECTION};
        border-color: {Theme.ACCENT};
    }}
    QPushButton#QuietButton {{
        background: transparent;
        border-color: transparent;
        color: {Theme.TEXT_MUTED};
    }}
    QPushButton#QuietButton:hover {{
        background: {Theme.SURFACE_3};
        border-color: {Theme.BORDER};
        color: {Theme.TEXT};
    }}
    QPushButton#DangerButton {{
        color: #FFFFFF;
        background: {Theme.DANGER};
        border-color: {Theme.DANGER};
        font-weight: 600;
    }}
    QPushButton#DangerButton:hover {{
        background: #D66A63;
        border-color: #D66A63;
    }}
    QPushButton#ToolButton {{
        background: transparent;
        border: 1px solid transparent;
        border-radius: 4px;
        padding: 0;
        min-width: 32px;
        min-height: 32px;
        color: {Theme.TEXT_MUTED};
    }}
    QPushButton#ToolButton:hover {{
        background: {Theme.SURFACE_3};
        border-color: {Theme.BORDER};
        color: {Theme.TEXT};
    }}
    QPushButton#ToolButton:checked {{
        color: {Theme.TEXT};
        background: {Theme.SELECTION};
        border-color: #355879;
    }}
    QPushButton#NavButton {{
        background: transparent;
        border: 1px solid transparent;
        border-radius: 4px;
        color: {Theme.TEXT_MUTED};
    }}
    QPushButton#NavButton:hover {{
        background: {Theme.SURFACE_3};
        border-color: {Theme.BORDER};
        color: {Theme.TEXT};
    }}
    QPushButton#NavButton:checked {{
        color: {Theme.TEXT};
        background: {Theme.SELECTION};
        border: 1px solid #355879;
        border-left: 2px solid {Theme.ACCENT};
    }}
    QPushButton#ClassChip {{
        background: {Theme.SURFACE_3};
        color: {Theme.TEXT};
        border: 1px solid {Theme.BORDER};
        border-radius: 4px;
        padding: 0 8px;
        font-size: 11px;
        min-height: 26px;
        max-height: 26px;
    }}
    QPushButton#ClassChip:hover {{
        background: {Theme.SURFACE_3};
        border-color: {Theme.BORDER_STRONG};
    }}
    QPushButton#ClassChip:checked {{
        color: {Theme.TEXT};
        background: {Theme.SELECTION};
        border-color: {Theme.ACCENT};
        font-weight: 600;
    }}
    QPushButton#DatasetButton {{
        color: {Theme.TEXT};
        background: {Theme.SURFACE_3};
        border-color: {Theme.BORDER_STRONG};
        font-weight: 600;
        padding: 0 11px;
        min-height: 30px;
    }}
    QPushButton#DatasetButton:hover {{
        background: {Theme.SELECTION};
        border-color: {Theme.ACCENT};
    }}
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
        background: {Theme.SURFACE_2};
        color: {Theme.TEXT};
        border: 1px solid {Theme.BORDER};
        border-radius: 4px;
        padding: 0 7px;
        min-height: 28px;
        selection-background-color: {Theme.SELECTION};
    }}
    QComboBox {{
        padding-right: 22px;
    }}
    QComboBox::drop-down {{
        subcontrol-origin: padding;
        subcontrol-position: top right;
        width: 20px;
        border-left: 1px solid {Theme.BORDER};
        border-top-right-radius: 4px;
        border-bottom-right-radius: 4px;
        background: {Theme.SURFACE_3};
    }}
    QComboBox QAbstractItemView {{
        background: {Theme.SURFACE_2};
        color: {Theme.TEXT};
        border: 1px solid {Theme.BORDER_STRONG};
        outline: none;
        selection-background-color: {Theme.SELECTION};
        min-height: 26px;
    }}
    QComboBox QAbstractItemView::item {{
        min-height: 26px;
        padding: 5px 8px;
    }}
    QComboBox QAbstractItemView::item:selected {{
        background: {Theme.SELECTION};
        color: {Theme.TEXT};
    }}
    QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
        border-color: {Theme.ACCENT};
    }}
    QTextEdit, QPlainTextEdit {{
        background: {Theme.SURFACE_2};
        color: {Theme.TEXT};
        border: 1px solid {Theme.BORDER};
        border-radius: 4px;
        padding: 6px;
        selection-background-color: {Theme.SELECTION};
        font-family: "Cascadia Mono", "Consolas", "DejaVu Sans Mono", "Noto Sans Mono", monospace;
    }}
    QListWidget, QTreeWidget, QTableWidget {{
        background: {Theme.SURFACE_2};
        alternate-background-color: {Theme.SURFACE};
        border: 1px solid {Theme.BORDER};
        border-radius: 4px;
        color: {Theme.TEXT};
        outline: none;
    }}
    QWidget#EmbeddedClassPanel {{
        background: transparent;
    }}
    QWidget#EmbeddedClassPanel QPushButton {{
        padding: 2px 6px;
    }}
    QWidget#EmbeddedClassPanel QPushButton[compact="true"] {{
        min-height: 24px;
        max-height: 24px;
        padding: 0 6px;
    }}
    QFrame#InspectorMeta {{
        background: transparent;
        border: none;
        border-top: 1px solid {Theme.BORDER};
    }}
    QLabel#CounterText {{
        color: {Theme.TEXT};
        font-weight: 600;
    }}
    QListWidget::item, QTreeWidget::item {{
        min-height: 22px;
        padding: 4px 6px;
        border-bottom: 1px solid #252B30;
    }}
    QListWidget::item:hover, QTreeWidget::item:hover {{
        background: {Theme.SURFACE_3};
    }}
    QListWidget::item:selected, QTreeWidget::item:selected {{
        background: {Theme.SELECTION};
        color: {Theme.TEXT};
    }}
    QHeaderView::section {{
        background: {Theme.SURFACE_3};
        color: {Theme.TEXT_MUTED};
        border: none;
        border-bottom: 1px solid {Theme.BORDER};
        padding: 6px;
        font-size: 11px;
        font-weight: 600;
    }}
    QTabWidget::pane {{
        border: none;
        background: {Theme.SURFACE_2};
    }}
    QTabWidget QWidget {{
        background: transparent;
    }}
    QTabBar::tab {{
        background: transparent;
        color: {Theme.TEXT_MUTED};
        padding: 7px 10px;
        border: none;
        border-bottom: 2px solid transparent;
    }}
    QTabBar::tab:hover {{
        color: {Theme.TEXT};
    }}
    QTabBar::tab:selected {{
        color: {Theme.TEXT};
        background: transparent;
        border-bottom-color: {Theme.ACCENT};
        font-weight: 600;
    }}
    QTabWidget#InspectorTabs::pane {{
        border: none;
        border-top: 1px solid {Theme.BORDER};
        background: transparent;
    }}
    QTabWidget#InspectorTabs QTabBar::tab {{
        background: transparent;
        color: {Theme.TEXT_MUTED};
        padding: 4px 8px;
        border: none;
        border-bottom: 2px solid transparent;
        min-height: 22px;
    }}
    QTabWidget#InspectorTabs QTabBar::tab:hover {{
        color: {Theme.TEXT};
    }}
    QTabWidget#InspectorTabs QTabBar::tab:selected {{
        color: {Theme.TEXT};
        background: transparent;
        border-bottom-color: {Theme.ACCENT};
        font-weight: 600;
    }}
    QGroupBox {{
        background: transparent;
        border: none;
        margin-top: 16px;
        padding: 10px 0 0 0;
        font-weight: 600;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 0;
        padding: 0;
        color: {Theme.TEXT};
    }}
    QToolBar {{
        background: {Theme.SURFACE};
        border: none;
        spacing: 4px;
    }}
    QStatusBar {{
        background: {Theme.SURFACE};
        color: {Theme.TEXT_MUTED};
        border-top: 1px solid {Theme.BORDER};
    }}
    QMenuBar {{
        background: {Theme.SURFACE};
        color: {Theme.TEXT_MUTED};
    }}
    QMenuBar::item {{
        padding: 5px 9px;
    }}
    QMenuBar::item:selected {{
        background: {Theme.SURFACE_3};
        color: {Theme.TEXT};
    }}
    QMenu {{
        background: {Theme.SURFACE_2};
        color: {Theme.TEXT};
        border: 1px solid {Theme.BORDER};
        padding: 5px;
    }}
    QMenu::item {{
        padding: 6px 24px 6px 18px;
        border-radius: 3px;
    }}
    QMenu::item:selected {{
        background: {Theme.SELECTION};
    }}
    QScrollArea {{
        background: transparent;
        border: none;
    }}
    QScrollArea > QWidget > QWidget {{
        background: transparent;
    }}
    QScrollArea#CanvasScrollArea {{
        background: {Theme.CANVAS_BG};
    }}
    QScrollArea#CanvasScrollArea > QWidget > QWidget {{
        background: {Theme.CANVAS_BG};
    }}
    QScrollArea#InspectorScroll {{
        background: {Theme.SURFACE};
    }}
    QSplitter::handle {{
        background: {Theme.BORDER};
    }}
    QScrollBar:vertical {{
        background: transparent;
        width: 8px;
        margin: 2px;
    }}
    QScrollBar::handle:vertical {{
        background: #3C454D;
        border-radius: 3px;
        min-height: 28px;
    }}
    QScrollBar:horizontal {{
        background: transparent;
        height: 8px;
        margin: 2px;
    }}
    QScrollBar::handle:horizontal {{
        background: #3C454D;
        border-radius: 3px;
        min-width: 28px;
    }}
    QProgressBar {{
        background: {Theme.SURFACE_2};
        color: {Theme.TEXT_MUTED};
        border: 1px solid {Theme.BORDER};
        border-radius: 4px;
        text-align: center;
        min-height: 18px;
        max-height: 22px;
    }}
    QProgressBar::chunk {{
        background: {Theme.ACCENT};
        border-radius: 3px;
    }}
    QCheckBox {{
        spacing: 7px;
    }}
    """
