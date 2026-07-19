"""Application theme for YOLO Studio — LabelImg-style light workbench."""

from __future__ import annotations


class Theme:
    BG = "#E8E8E8"
    RAIL = "#DCDCDC"
    SURFACE = "#F2F2F2"
    SURFACE_2 = "#FFFFFF"
    SURFACE_3 = "#E0E0E0"
    BORDER = "#B8B8B8"
    BORDER_STRONG = "#909090"
    TEXT = "#222222"
    TEXT_MUTED = "#555555"
    TEXT_DIM = "#777777"
    ACCENT = "#2F6FED"
    ACCENT_HOVER = "#1F5AD6"
    ACCENT_DARK = "#1847B0"
    SUCCESS = "#2E7D32"
    WARNING = "#B86E00"
    DANGER = "#C62828"
    SELECTION = "#D6E4FF"
    # Light mid-gray stage — enough contrast for images, no dark-mode slab
    CANVAS_BG = "#D4D4D4"
    CANVAS_HINT = "#555555"


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
        background: {Theme.SURFACE_2};
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
        background: {Theme.SURFACE_3};
        color: {Theme.TEXT};
        border: 1px solid {Theme.BORDER};
        border-bottom: none;
        border-radius: 0px;
        padding: 4px 14px;
        min-width: 56px;
        font-size: 13px;
    }}
    QPushButton#WorkspaceTab:hover {{
        color: {Theme.TEXT};
        background: {Theme.SURFACE_2};
    }}
    QPushButton#WorkspaceTab:checked {{
        color: #FFFFFF;
        background: {Theme.ACCENT};
        border-color: {Theme.ACCENT};
        font-weight: 600;
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
    QWidget#StageOverlayDim {{
        background: {Theme.BG};
    }}
    QWidget#StageOverlay {{
        background: {Theme.SURFACE_2};
        border-left: 1px solid {Theme.BORDER};
    }}
    QWidget#StageHost {{
        background: {Theme.SURFACE_2};
    }}
    QWidget#WorkspacePage {{
        background: {Theme.SURFACE_2};
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
        border-radius: 0px;
    }}
    QLabel#BrandTitle, QLabel#PageTitle {{
        color: {Theme.TEXT};
        font-size: 14px;
        font-weight: 600;
    }}
    QLabel#PanelTitle {{
        color: {Theme.TEXT};
        font-size: 13px;
        font-weight: 600;
    }}
    QLabel#BrandSubtitle, QLabel#MutedText {{
        color: {Theme.TEXT_MUTED};
        font-size: 12px;
    }}
    QLabel#SectionTitle {{
        color: {Theme.TEXT};
        font-size: 12px;
        font-weight: 600;
        padding: 0 0 2px 0;
    }}
    QLabel#StatusPill {{
        color: {Theme.TEXT_MUTED};
        background: {Theme.SURFACE_3};
        border: 1px solid {Theme.BORDER};
        border-radius: 0px;
        padding: 1px 6px;
        font-size: 11px;
    }}
    QLabel#StatusPill[variant="accent"] {{
        color: {Theme.ACCENT_DARK};
        background: {Theme.SELECTION};
        border-color: #9BB8F0;
        font-weight: 600;
    }}
    QLabel#StatusPill[variant="success"] {{
        color: {Theme.SUCCESS};
        background: #E8F5E9;
        border-color: #A5D6A7;
        font-weight: 600;
    }}
    QLabel#StatusPill[variant="warning"] {{
        color: {Theme.WARNING};
        background: #FFF3E0;
        border-color: #FFCC80;
        font-weight: 600;
    }}
    QLabel#ClassCurrent {{
        border: 1px solid {Theme.BORDER};
        border-radius: 0px;
        padding: 2px 6px;
        font-size: 11px;
        font-weight: 600;
    }}
    QLabel#PreviewSurface {{
        background: {Theme.CANVAS_BG};
        border: 1px solid {Theme.BORDER};
        border-radius: 0px;
        color: {Theme.CANVAS_HINT};
    }}
    QPushButton {{
        background: {Theme.SURFACE_3};
        color: {Theme.TEXT};
        border: 1px solid {Theme.BORDER};
        border-radius: 0px;
        padding: 0 10px;
        min-height: 26px;
    }}
    QPushButton:hover {{
        background: #E2E2E2;
        border-color: {Theme.BORDER_STRONG};
    }}
    QPushButton:pressed {{
        background: #D8D8D8;
    }}
    QPushButton:disabled {{
        color: {Theme.TEXT_DIM};
        background: #F5F5F5;
        border-color: #D8D8D8;
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
        min-height: 28px;
    }}
    QPushButton#PrimaryButton:hover {{
        background: {Theme.ACCENT_HOVER};
        border-color: {Theme.ACCENT_HOVER};
    }}
    QPushButton#SecondaryButton {{
        background: {Theme.SURFACE_2};
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
        background: #B71C1C;
        border-color: #B71C1C;
    }}
    QPushButton#ToolButton {{
        background: {Theme.SURFACE_2};
        border: 1px solid {Theme.BORDER};
        border-radius: 0px;
        padding: 0;
        min-width: 30px;
        max-width: 30px;
        min-height: 28px;
        max-height: 28px;
        color: {Theme.TEXT};
        font-size: 13px;
    }}
    QPushButton#ToolButton:hover {{
        background: #E2E2E2;
        border-color: {Theme.BORDER_STRONG};
        color: {Theme.TEXT};
    }}
    QPushButton#ToolButton:checked {{
        color: #FFFFFF;
        background: {Theme.ACCENT};
        border-color: {Theme.ACCENT};
        font-weight: 600;
    }}
    QPushButton#ClassChip {{
        background: {Theme.SURFACE_2};
        color: {Theme.TEXT};
        border: 1px solid {Theme.BORDER};
        border-radius: 0px;
        padding: 0 8px;
        font-size: 11px;
        min-height: 24px;
        max-height: 24px;
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
        background: {Theme.SURFACE_2};
        border-color: {Theme.BORDER_STRONG};
        font-weight: 600;
        padding: 0 11px;
        min-height: 28px;
    }}
    QPushButton#DatasetButton:hover {{
        background: {Theme.SELECTION};
        border-color: {Theme.ACCENT};
    }}
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
        background: {Theme.SURFACE_2};
        color: {Theme.TEXT};
        border: 1px solid {Theme.BORDER};
        border-radius: 0px;
        padding: 2px 8px;
        min-height: 26px;
        selection-background-color: {Theme.SELECTION};
    }}
    QComboBox {{
        padding-right: 24px;
    }}
    QComboBox::drop-down {{
        subcontrol-origin: padding;
        subcontrol-position: top right;
        width: 20px;
        border-left: 1px solid {Theme.BORDER};
        border-top-right-radius: 0px;
        border-bottom-right-radius: 0px;
        background: {Theme.SURFACE_3};
    }}
    QComboBox::down-arrow {{
        width: 0px;
        height: 0px;
        border-left: 4px solid transparent;
        border-right: 4px solid transparent;
        border-top: 5px solid {Theme.TEXT_MUTED};
        margin-right: 4px;
    }}
    QComboBox QAbstractItemView {{
        background: {Theme.SURFACE_2};
        color: {Theme.TEXT};
        border: 1px solid {Theme.BORDER_STRONG};
        outline: none;
        selection-background-color: {Theme.SELECTION};
        min-height: 24px;
    }}
    QComboBox QAbstractItemView::item {{
        min-height: 24px;
        padding: 4px 8px;
    }}
    QComboBox QAbstractItemView::item:selected {{
        background: {Theme.SELECTION};
        color: {Theme.TEXT};
    }}
    /* Keep spin arrows in a clean strip so they never paint over the value. */
    QSpinBox, QDoubleSpinBox {{
        padding-right: 18px;
    }}
    QSpinBox::up-button, QDoubleSpinBox::up-button,
    QSpinBox::down-button, QDoubleSpinBox::down-button {{
        subcontrol-origin: border;
        width: 16px;
        border-left: 1px solid {Theme.BORDER};
        background: {Theme.SURFACE_3};
    }}
    QSpinBox::up-button, QDoubleSpinBox::up-button {{
        subcontrol-position: top right;
        border-bottom: 1px solid {Theme.BORDER};
    }}
    QSpinBox::down-button, QDoubleSpinBox::down-button {{
        subcontrol-position: bottom right;
    }}
    QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
        width: 0px;
        height: 0px;
        border-left: 3px solid transparent;
        border-right: 3px solid transparent;
        border-bottom: 4px solid {Theme.TEXT_MUTED};
    }}
    QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
        width: 0px;
        height: 0px;
        border-left: 3px solid transparent;
        border-right: 3px solid transparent;
        border-top: 4px solid {Theme.TEXT_MUTED};
    }}
    QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
        border-color: {Theme.ACCENT};
    }}
    QTextEdit, QPlainTextEdit {{
        background: {Theme.SURFACE_2};
        color: {Theme.TEXT};
        border: 1px solid {Theme.BORDER};
        border-radius: 0px;
        padding: 6px;
        selection-background-color: {Theme.SELECTION};
        font-family: "Cascadia Mono", "Consolas", "DejaVu Sans Mono", "Noto Sans Mono", monospace;
    }}
    QListWidget, QTreeWidget, QTableWidget {{
        background: {Theme.SURFACE_2};
        alternate-background-color: {Theme.SURFACE};
        border: 1px solid {Theme.BORDER};
        border-radius: 0px;
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
        padding: 3px 6px;
        border-bottom: 1px solid #E6E6E6;
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
        padding: 5px;
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
        padding: 6px 10px;
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
        margin-top: 14px;
        padding: 8px 0 0 0;
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
        color: {Theme.TEXT};
    }}
    QMenuBar::item {{
        padding: 5px 9px;
    }}
    QMenuBar::item:selected {{
        background: {Theme.SELECTION};
        color: {Theme.TEXT};
    }}
    QMenu {{
        background: {Theme.SURFACE_2};
        color: {Theme.TEXT};
        border: 1px solid {Theme.BORDER};
        padding: 4px;
    }}
    QMenu::item {{
        padding: 5px 22px 5px 16px;
        border-radius: 0px;
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
    QGraphicsView#WorkflowView {{
        background: {Theme.BG};
        border: none;
    }}
    QWidget#WorkflowCanvasPanel {{
        background: {Theme.BG};
    }}
    QScrollArea#InspectorScroll {{
        background: {Theme.SURFACE};
    }}
    QSplitter::handle {{
        background: {Theme.BORDER};
    }}
    QScrollBar:vertical {{
        background: {Theme.BG};
        width: 12px;
        margin: 0;
    }}
    QScrollBar::handle:vertical {{
        background: #B8B8B8;
        border-radius: 0px;
        min-height: 28px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: #9A9A9A;
    }}
    QScrollBar:horizontal {{
        background: {Theme.BG};
        height: 12px;
        margin: 0;
    }}
    QScrollBar::handle:horizontal {{
        background: #B8B8B8;
        border-radius: 0px;
        min-width: 28px;
    }}
    QProgressBar {{
        background: {Theme.SURFACE_2};
        color: {Theme.TEXT};
        border: 1px solid {Theme.BORDER};
        border-radius: 0px;
        text-align: center;
        min-height: 16px;
        max-height: 20px;
    }}
    QProgressBar::chunk {{
        background: {Theme.ACCENT};
        border-radius: 0px;
    }}
    QCheckBox {{
        spacing: 6px;
    }}
    """
