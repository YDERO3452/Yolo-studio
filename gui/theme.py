"""Application theme for the YOLO Studio desktop UI.

The visual direction is intentionally restrained: graphite surfaces, warm
accent actions, sharp spacing, and no generative-AI visual motifs.
"""

from __future__ import annotations


class Theme:
    BG = "#111315"
    SURFACE = "#171A1D"
    SURFACE_2 = "#202428"
    SURFACE_3 = "#2A3036"
    BORDER = "#353B42"
    BORDER_STRONG = "#4A5660"
    TEXT = "#F1EFE8"
    TEXT_MUTED = "#B7B1A8"
    TEXT_DIM = "#7D858D"
    ACCENT = "#1E8CFF"
    ACCENT_HOVER = "#389BFF"
    ACCENT_DARK = "#115999"
    SUCCESS = "#6AA56A"
    WARNING = "#D0A647"
    DANGER = "#C65B55"
    SELECTION = "#33424B"
    CANVAS_BG = "#202222"


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
    QWidget#ToolRail {{
        background: {Theme.SURFACE_2};
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
        background: {Theme.BG};
        border-left: 1px solid {Theme.BORDER};
        border-right: 1px solid {Theme.BORDER};
    }}
    QWidget#WorkspaceHeader {{
        background: {Theme.SURFACE_2};
        border-bottom: 1px solid {Theme.BORDER};
    }}
    QWidget#AnnotationControlBar {{
        background: #1C222B;
        border-bottom: 1px solid {Theme.BORDER};
    }}
    QLabel#ModeHint {{
        color: {Theme.TEXT_MUTED};
        font-size: 12px;
    }}
    QLabel#InlineStatus {{
        color: #35D060;
        font-size: 12px;
        font-weight: 700;
    }}
    QWidget#Card, QFrame#Card {{
        background: {Theme.SURFACE_2};
        border: 1px solid {Theme.BORDER};
        border-radius: 10px;
    }}
    QLabel#BrandTitle {{
        color: {Theme.TEXT};
        font-size: 16px;
        font-weight: 700;
    }}
    QLabel#PanelTitle {{
        color: {Theme.TEXT};
        font-size: 15px;
        font-weight: 800;
    }}
    QLabel#BrandSubtitle, QLabel#MutedText {{
        color: {Theme.TEXT_MUTED};
        font-size: 11px;
    }}
    QLabel#SectionTitle {{
        color: {Theme.TEXT};
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.4px;
        padding: 0 0 4px 0;
    }}
    QLabel#StatusPill {{
        color: {Theme.TEXT};
        background: {Theme.SURFACE_3};
        border: 1px solid {Theme.BORDER};
        border-radius: 10px;
        padding: 2px 9px;
        font-size: 11px;
    }}
    QLabel#StatusPill[variant="accent"] {{
        color: #16110A;
        background: {Theme.ACCENT};
        border-color: {Theme.ACCENT};
        font-weight: 700;
    }}
    QLabel#StatusPill[variant="success"] {{
        color: #0E180F;
        background: {Theme.SUCCESS};
        border-color: {Theme.SUCCESS};
        font-weight: 700;
    }}
    QLabel#StatusPill[variant="warning"] {{
        color: #171104;
        background: {Theme.WARNING};
        border-color: {Theme.WARNING};
        font-weight: 700;
    }}
    QLabel#ClassCurrent {{
        border: 1px solid {Theme.BORDER};
        border-radius: 9px;
        padding: 2px 8px;
        font-size: 11px;
        font-weight: 700;
    }}
    QLabel#PreviewSurface {{
        background: {Theme.CANVAS_BG};
        border: 1px solid {Theme.BORDER};
        border-radius: 10px;
        color: {Theme.TEXT_MUTED};
    }}
    QPushButton {{
        background: {Theme.SURFACE_3};
        color: {Theme.TEXT};
        border: 1px solid {Theme.BORDER_STRONG};
        border-radius: 7px;
        padding: 0 11px;
        min-height: 30px;
    }}
    QPushButton:hover {{
        background: #2E363D;
        border-color: #5B6872;
    }}
    QPushButton:pressed {{
        background: #20262B;
    }}
    QPushButton:disabled {{
        color: {Theme.TEXT_DIM};
        background: #1A1E21;
        border-color: #2B3136;
    }}
    QPushButton:checked {{
        color: #16110A;
        background: {Theme.ACCENT};
        border-color: {Theme.ACCENT};
        font-weight: 700;
    }}
    QPushButton#PrimaryButton {{
        color: #FFFFFF;
        background: {Theme.ACCENT};
        border-color: {Theme.ACCENT};
        font-weight: 700;
    }}
    QPushButton#PrimaryButton:hover {{
        background: {Theme.ACCENT_HOVER};
        border-color: {Theme.ACCENT_HOVER};
    }}
    QPushButton#DangerButton {{
        color: #1B0D0B;
        background: {Theme.DANGER};
        border-color: {Theme.DANGER};
        font-weight: 700;
    }}
    QPushButton#DangerButton:hover {{
        background: #D66A63;
        border-color: #D66A63;
    }}
    QPushButton#ToolButton {{
        background: transparent;
        border: 1px solid transparent;
        border-radius: 8px;
        padding: 6px 4px;
        min-width: 34px;
        min-height: 30px;
        color: {Theme.TEXT_MUTED};
        font-weight: 700;
    }}
    QPushButton#ToolButton:hover {{
        background: {Theme.SURFACE_3};
        border-color: {Theme.BORDER};
        color: {Theme.TEXT};
    }}
    QPushButton#ToolButton:checked {{
        color: #FFFFFF;
        background: {Theme.ACCENT};
        border-color: {Theme.ACCENT};
    }}
    QPushButton#NavButton {{
        background: transparent;
        border: 1px solid transparent;
        border-radius: 8px;
        color: {Theme.TEXT_MUTED};
    }}
    QPushButton#NavButton:hover {{
        background: {Theme.SURFACE_3};
        border-color: {Theme.BORDER};
        color: {Theme.TEXT};
    }}
    QPushButton#NavButton:checked {{
        color: #FFFFFF;
        background: {Theme.ACCENT};
        border-color: {Theme.ACCENT};
    }}
    QPushButton#ClassChip {{
        background: {Theme.SURFACE_3};
        color: {Theme.TEXT};
        border: 1px solid {Theme.BORDER};
        border-radius: 8px;
        padding: 0 8px;
        font-size: 11px;
        min-height: 26px;
        max-height: 26px;
    }}
    QPushButton#ClassChip:hover {{
        background: #2E363D;
        border-color: {Theme.BORDER_STRONG};
    }}
    QPushButton#ClassChip:checked {{
        color: #FFFFFF;
        background: {Theme.ACCENT};
        border-color: {Theme.ACCENT};
        font-weight: 700;
    }}
    QPushButton#DatasetButton {{
        color: #171004;
        background: #FF9800;
        border-color: #FF9800;
        font-weight: 800;
        padding: 0 11px;
        min-height: 34px;
    }}
    QPushButton#DatasetButton:hover {{
        background: #FFA726;
        border-color: #FFA726;
    }}
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
        background: {Theme.SURFACE_2};
        color: {Theme.TEXT};
        border: 1px solid {Theme.BORDER};
        border-radius: 7px;
        padding: 0 7px;
        min-height: 30px;
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
        border-top-right-radius: 7px;
        border-bottom-right-radius: 7px;
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
        border-radius: 8px;
        padding: 6px;
        selection-background-color: {Theme.SELECTION};
        font-family: "Cascadia Mono", "Consolas", "DejaVu Sans Mono", "Noto Sans Mono", monospace;
    }}
    QListWidget, QTreeWidget, QTableWidget {{
        background: {Theme.SURFACE_2};
        alternate-background-color: {Theme.SURFACE};
        border: 1px solid {Theme.BORDER};
        border-radius: 8px;
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
        background: {Theme.SURFACE_2};
        border: 1px solid {Theme.BORDER};
        border-radius: 6px;
    }}
    QLabel#CounterText {{
        color: {Theme.TEXT};
        font-weight: 700;
    }}
    QListWidget::item, QTreeWidget::item {{
        min-height: 24px;
        padding: 5px 7px;
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
        font-weight: 700;
    }}
    QTabWidget::pane {{
        border: none;
        background: {Theme.SURFACE_2};
    }}
    QTabWidget QWidget {{
        background: transparent;
    }}
    QTabBar::tab {{
        background: {Theme.SURFACE};
        color: {Theme.TEXT_MUTED};
        padding: 8px 10px;
        border: none;
        border-bottom: 2px solid transparent;
    }}
    QTabBar::tab:hover {{
        color: {Theme.TEXT};
    }}
    QTabBar::tab:selected {{
        color: {Theme.TEXT};
        background: {Theme.SURFACE_2};
        border-bottom-color: {Theme.ACCENT};
        font-weight: 700;
    }}
    QTabWidget#InspectorTabs::pane {{
        border: 1px solid {Theme.BORDER};
        border-radius: 5px;
        background: {Theme.SURFACE_2};
        margin-top: 2px;
    }}
    QTabWidget#InspectorTabs QTabBar::tab {{
        background: {Theme.SURFACE};
        color: {Theme.TEXT_MUTED};
        padding: 2px 7px;
        border: 1px solid {Theme.BORDER};
        border-bottom: none;
        border-top-left-radius: 5px;
        border-top-right-radius: 5px;
        min-height: 20px;
    }}
    QTabWidget#InspectorTabs QTabBar::tab:hover {{
        color: {Theme.TEXT};
    }}
    QTabWidget#InspectorTabs QTabBar::tab:selected {{
        color: #FFFFFF;
        background: #1D2E40;
        border-color: {Theme.ACCENT};
        font-weight: 700;
    }}
    QGroupBox {{
        background: {Theme.SURFACE_2};
        border: 1px solid {Theme.BORDER};
        border-radius: 10px;
        margin-top: 16px;
        padding: 12px 10px 10px 10px;
        font-weight: 700;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 6px;
        color: {Theme.TEXT_MUTED};
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
        border-radius: 5px;
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
        width: 10px;
        margin: 2px;
    }}
    QScrollBar::handle:vertical {{
        background: #3C454D;
        border-radius: 5px;
        min-height: 28px;
    }}
    QScrollBar:horizontal {{
        background: transparent;
        height: 10px;
        margin: 2px;
    }}
    QScrollBar::handle:horizontal {{
        background: #3C454D;
        border-radius: 5px;
        min-width: 28px;
    }}
    QProgressBar {{
        background: {Theme.SURFACE_2};
        color: {Theme.TEXT_MUTED};
        border: 1px solid {Theme.BORDER};
        border-radius: 6px;
        text-align: center;
        min-height: 18px;
        max-height: 22px;
    }}
    QProgressBar::chunk {{
        background: {Theme.ACCENT};
        border-radius: 5px;
    }}
    QCheckBox {{
        spacing: 7px;
    }}
    """
