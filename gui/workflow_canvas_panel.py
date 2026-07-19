"""Node-based workflow canvas for YOLO Studio.

Main nodes form the runnable YOLO pipeline. Sub-nodes hang under parents and
open dialogs / stage tabs; they are not part of workflow execution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPainterPath,
    QPen,
    QTransform,
    QWheelEvent,
)
from PyQt6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from gui.theme import Theme


@dataclass(frozen=True)
class StageSpec:
    key: str
    title: str
    subtitle: str
    workspace_index: int
    accent: str
    kind: str = "main"  # main | sub | utility
    parent_key: Optional[str] = None
    action: Optional[str] = None  # dialog:* | stage:index:tab | focus:annotate


# Runnable YOLO pipeline. workspace_index maps to MainWindow.stage_stack.
MAIN_STAGES: tuple[StageSpec, ...] = (
    StageSpec("dataset", "数据", "导入 / 划分 / data.yaml", 3, "#2F6FED", "main"),
    StageSpec("annotate", "标注", "画框 · 队列 · 保存", 0, "#1B7F5A", "main"),
    StageSpec("train", "训练", "Ultralytics 训练与监控", 1, "#C47A00", "main"),
    StageSpec("results", "结果", "权重与训练产物", 6, "#6B5B95", "main"),
    StageSpec("infer", "推理", "图片 · 视频 · 摄像头", 2, "#B85C38", "main"),
    StageSpec("export", "导出", "ONNX / 部署格式", 4, "#3D7EA6", "main"),
    StageSpec("quality", "质检", "覆盖率与检查", 5, "#5A6A7A", "main"),
)

# Attached helpers — double-click opens action; never run in pipeline.
SUB_STAGES: tuple[StageSpec, ...] = (
    StageSpec("project", "管理项目", "创建 / 打开项目", -1, "#2F6FED", "sub", "dataset", "dialog:project"),
    StageSpec("video", "视频截帧", "抽帧进数据集", -1, "#2F6FED", "sub", "dataset", "dialog:video"),
    StageSpec("format", "格式转换", "YOLO / VOC / COCO", -1, "#2F6FED", "sub", "dataset", "dialog:format"),
    StageSpec("autolabel", "自动标注", "YOLO / LLM", -1, "#1B7F5A", "sub", "annotate", "focus:annotate"),
    StageSpec("namemap", "类别映射", "模型名 ↔ 项目类", -1, "#1B7F5A", "sub", "annotate", "dialog:namemap"),
    StageSpec("stats", "统计", "类别分布与报告", -1, "#5A6A7A", "sub", "quality", "stage:5:0"),
    StageSpec("flow", "流程", "批量与预设", -1, "#5A6A7A", "sub", "quality", "stage:5:1"),
    StageSpec("env", "环境", "CUDA / PyTorch", -1, "#666666", "sub", "system", "dialog:env"),
)

UTILITY_STAGES: tuple[StageSpec, ...] = (
    StageSpec("system", "系统", "环境与本机配置", -1, "#666666", "utility", None, None),
)

ALL_STAGES: tuple[StageSpec, ...] = MAIN_STAGES + UTILITY_STAGES + SUB_STAGES

# Default edges: linear main path + quality branch (no cycles for execution).
DEFAULT_EDGES: tuple[tuple[str, str], ...] = (
    ("dataset", "annotate"),
    ("annotate", "train"),
    ("annotate", "quality"),
    ("train", "results"),
    ("results", "infer"),
    ("infer", "export"),
)

# Back-compat alias for anything importing STAGES.
STAGES = MAIN_STAGES


class _PortItem(QGraphicsEllipseItem):
    """Input/output connector on a main pipeline node."""

    def __init__(self, node: "WorkflowNode", is_output: bool):
        super().__init__(-5, -5, 10, 10, node)
        self.node = node
        self.is_output = is_output
        self.setBrush(QBrush(QColor(Theme.SURFACE_2)))
        self.setPen(QPen(QColor(Theme.BORDER_STRONG), 1.2))
        self.setZValue(2)
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def hoverEnterEvent(self, event):
        self.setBrush(QBrush(QColor(Theme.ACCENT)))
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.setBrush(QBrush(QColor(Theme.SURFACE_2)))
        super().hoverLeaveEvent(event)


class WorkflowEdge(QGraphicsPathItem):
    """Solid pipeline edge between main nodes."""

    def __init__(self, source: WorkflowNode, target: WorkflowNode):
        super().__init__()
        self.source = source
        self.target = target
        self.setZValue(0)
        self.setPen(QPen(QColor(Theme.BORDER_STRONG), 2.0))
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        source.edges.add(self)
        target.edges.add(self)
        self.update_path()

    def update_path(self) -> None:
        p1 = self.source.output_port_scene_pos()
        p2 = self.target.input_port_scene_pos()
        path = QPainterPath(p1)
        dx = max(60.0, abs(p2.x() - p1.x()) * 0.45)
        path.cubicTo(p1.x() + dx, p1.y(), p2.x() - dx, p2.y(), p2.x(), p2.y())
        self.setPath(path)

    def detach(self) -> None:
        self.source.edges.discard(self)
        self.target.edges.discard(self)


class ParentLink(QGraphicsPathItem):
    """Dashed parent→child attachment (not a runnable pipeline edge)."""

    def __init__(self, parent_node: WorkflowNode, child_node: WorkflowNode):
        super().__init__()
        self.parent_node = parent_node
        self.child_node = child_node
        self.setZValue(0)
        pen = QPen(QColor(Theme.BORDER), 1.2, Qt.PenStyle.DashLine)
        self.setPen(pen)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        parent_node.parent_links.add(self)
        child_node.parent_links.add(self)
        self.update_path()

    def update_path(self) -> None:
        pret = self.parent_node.sceneBoundingRect()
        cret = self.child_node.sceneBoundingRect()
        p1 = QPointF(pret.center().x(), pret.bottom())
        p2 = QPointF(cret.center().x(), cret.top())
        path = QPainterPath(p1)
        mid_y = (p1.y() + p2.y()) / 2
        path.cubicTo(p1.x(), mid_y, p2.x(), mid_y, p2.x(), p2.y())
        self.setPath(path)

    def detach(self) -> None:
        self.parent_node.parent_links.discard(self)
        self.child_node.parent_links.discard(self)


class WorkflowNode(QGraphicsRectItem):
    MAIN_W, MAIN_H = 168, 86
    SUB_W, SUB_H = 132, 52
    UTIL_W, UTIL_H = 132, 64

    STATUS_COLORS = {
        "idle": Theme.BORDER_STRONG,
        "pending": Theme.TEXT_DIM,
        "running": Theme.ACCENT,
        "done": Theme.SUCCESS,
        "error": Theme.DANGER,
        "skipped": Theme.WARNING,
    }

    def __init__(self, spec: StageSpec):
        kind = spec.kind
        if kind == "sub":
            w, h = self.SUB_W, self.SUB_H
        elif kind == "utility":
            w, h = self.UTIL_W, self.UTIL_H
        else:
            w, h = self.MAIN_W, self.MAIN_H
        super().__init__(0, 0, w, h)
        self.spec = spec
        self.status = "idle"
        self.edges: set[WorkflowEdge] = set()
        self.parent_links: set[ParentLink] = set()
        self.child_nodes: list[WorkflowNode] = []
        self._child_offsets: dict[str, QPointF] = {}
        self._following = False
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)
        self.setZValue(1 if kind != "sub" else 0.8)
        self.setBrush(QBrush(QColor(Theme.SURFACE_2)))
        self.setPen(QPen(QColor(Theme.BORDER), 1.2))
        self.setCursor(Qt.CursorShape.OpenHandCursor)

        accent_w = 3 if kind == "sub" else 4
        accent_color = QColor(spec.accent)
        if kind == "sub":
            accent_color.setAlpha(160)
        self._accent = QGraphicsRectItem(0, 0, accent_w, h, self)
        self._accent.setBrush(QBrush(accent_color))
        self._accent.setPen(QPen(Qt.PenStyle.NoPen))

        title_size = 10 if kind == "sub" else 12
        title = QGraphicsSimpleTextItem(spec.title, self)
        title.setBrush(QBrush(QColor(Theme.TEXT)))
        title.setFont(QFont("Microsoft YaHei UI", title_size, QFont.Weight.DemiBold))
        title.setPos(14, 8 if kind == "sub" else 10)

        sub = QGraphicsSimpleTextItem(spec.subtitle, self)
        sub.setBrush(QBrush(QColor(Theme.TEXT_MUTED)))
        sub.setFont(QFont("Microsoft YaHei UI", 8))
        sub.setPos(14, 28 if kind == "sub" else 34)

        self._status_text = QGraphicsSimpleTextItem("", self)
        self._status_text.setBrush(QBrush(QColor(Theme.TEXT_DIM)))
        self._status_text.setFont(QFont("Microsoft YaHei UI", 8))
        self._status_text.setPos(14, 58)
        if kind == "main":
            self._status_text.setText("待命")
        elif kind == "utility":
            self._status_text.setText("工具")
            self._status_text.setPos(14, 40)
        else:
            self._status_text.setVisible(False)

        self.in_port: Optional[_PortItem] = None
        self.out_port: Optional[_PortItem] = None
        if kind == "main":
            self.in_port = _PortItem(self, is_output=False)
            self.in_port.setPos(0, h / 2)
            self.out_port = _PortItem(self, is_output=True)
            self.out_port.setPos(w, h / 2)

    @property
    def is_runnable(self) -> bool:
        return self.spec.kind == "main"

    @property
    def has_ports(self) -> bool:
        return self.in_port is not None and self.out_port is not None

    def set_status(self, status: str, detail: str = "") -> None:
        if not self.is_runnable:
            return
        self.status = status
        labels = {
            "idle": "待命",
            "pending": "排队",
            "running": "运行中…",
            "done": "完成",
            "error": "失败",
            "skipped": "跳过",
        }
        text = labels.get(status, status)
        if detail:
            text = f"{text} · {detail[:18]}"
        self._status_text.setText(text)
        color = self.STATUS_COLORS.get(status, Theme.BORDER_STRONG)
        self._status_text.setBrush(QBrush(QColor(color)))
        self.update()

    def input_port_scene_pos(self) -> QPointF:
        if self.in_port is None:
            return self.sceneBoundingRect().center()
        return self.in_port.scenePos()

    def output_port_scene_pos(self) -> QPointF:
        if self.out_port is None:
            return self.sceneBoundingRect().center()
        return self.out_port.scenePos()

    def remember_child_offsets(self) -> None:
        origin = self.pos()
        for child in self.child_nodes:
            self._child_offsets[child.spec.key] = child.pos() - origin

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            for edge in self.edges:
                edge.update_path()
            for link in self.parent_links:
                link.update_path()
            if not self._following and self.child_nodes:
                self._follow_children()
        return super().itemChange(change, value)

    def _follow_children(self) -> None:
        origin = self.pos()
        for child in self.child_nodes:
            offset = self._child_offsets.get(child.spec.key)
            if offset is None:
                continue
            child._following = True
            child.setPos(origin + offset)
            child._following = False

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        super().mouseReleaseEvent(event)

    def paint(self, painter: QPainter, option, widget=None):
        selected = bool(self.isSelected())
        if self.spec.kind == "sub":
            border = Theme.ACCENT if selected else Theme.BORDER
            width = 1.2
            fill = QColor(Theme.SURFACE)
        elif self.spec.kind == "utility":
            border = Theme.ACCENT if selected else Theme.BORDER_STRONG
            width = 1.4
            fill = QColor(Theme.SURFACE_2)
        else:
            status_color = self.STATUS_COLORS.get(self.status, Theme.BORDER_STRONG)
            border = Theme.ACCENT if selected else status_color
            width = 2.0 if self.status == "running" else 1.4
            fill = QColor(Theme.SURFACE_2)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QBrush(fill))
        painter.setPen(QPen(QColor(border), width))
        painter.drawRoundedRect(self.rect(), 2, 2)


class _TempEdge(QGraphicsPathItem):
    def __init__(self):
        super().__init__()
        self.setZValue(0.5)
        pen = QPen(QColor(Theme.ACCENT), 2.0, Qt.PenStyle.DashLine)
        self.setPen(pen)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

    def set_endpoints(self, a: QPointF, b: QPointF) -> None:
        path = QPainterPath(a)
        dx = max(40.0, abs(b.x() - a.x()) * 0.4)
        path.cubicTo(a.x() + dx, a.y(), b.x() - dx, b.y(), b.x(), b.y())
        self.setPath(path)


class WorkflowScene(QGraphicsScene):
    node_activated = pyqtSignal(int)  # workspace_index for main stages
    action_activated = pyqtSignal(str)  # action string for sub / focused tools

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSceneRect(-4000, -3000, 8000, 6000)
        self.setBackgroundBrush(QBrush(QColor(Theme.BG)))
        self.nodes: dict[str, WorkflowNode] = {}
        self._wire_source: Optional[WorkflowNode] = None
        self._temp_edge: Optional[_TempEdge] = None
        self._build_default_graph()

    def _build_default_graph(self) -> None:
        positions = {
            "dataset": QPointF(-420, -40),
            "annotate": QPointF(-180, -40),
            "train": QPointF(60, -40),
            "results": QPointF(300, -40),
            "infer": QPointF(540, -40),
            "export": QPointF(780, -40),
            "quality": QPointF(-180, 200),
            "system": QPointF(-420, 320),
        }
        for spec in MAIN_STAGES + UTILITY_STAGES:
            node = WorkflowNode(spec)
            node.setPos(positions[spec.key])
            self.addItem(node)
            self.nodes[spec.key] = node

        # Place children under parents in vertical stacks.
        children_by_parent: dict[str, list[StageSpec]] = {}
        for spec in SUB_STAGES:
            children_by_parent.setdefault(spec.parent_key or "", []).append(spec)

        for parent_key, kids in children_by_parent.items():
            parent = self.nodes.get(parent_key)
            if parent is None:
                continue
            base = parent.pos()
            for i, spec in enumerate(kids):
                child = WorkflowNode(spec)
                child.setPos(base + QPointF(18, parent.rect().height() + 28 + i * 62))
                self.addItem(child)
                self.nodes[spec.key] = child
                parent.child_nodes.append(child)
                link = ParentLink(parent, child)
                self.addItem(link)
            parent.remember_child_offsets()

        for src, dst in DEFAULT_EDGES:
            self.add_edge(self.nodes[src], self.nodes[dst])

    def add_edge(self, source: WorkflowNode, target: WorkflowNode) -> Optional[WorkflowEdge]:
        if source is target:
            return None
        if not source.is_runnable or not target.is_runnable:
            return None
        for edge in source.edges:
            if edge.source is source and edge.target is target:
                return edge
        edge = WorkflowEdge(source, target)
        self.addItem(edge)
        return edge

    def remove_edge(self, edge: WorkflowEdge) -> None:
        edge.detach()
        self.removeItem(edge)

    def _activate_node(self, node: WorkflowNode) -> None:
        if node.spec.action:
            self.action_activated.emit(node.spec.action)
            return
        if node.spec.kind == "main" and node.spec.workspace_index >= 0:
            self.node_activated.emit(node.spec.workspace_index)

    def mouseDoubleClickEvent(self, event):
        item = self.itemAt(event.scenePos(), QTransform())
        node = self._node_from_item(item)
        if node is not None:
            self._activate_node(node)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            for item in self.selectedItems():
                if isinstance(item, WorkflowNode):
                    self._activate_node(item)
                    event.accept()
                    return
        super().keyPressEvent(event)

    def mousePressEvent(self, event):
        item = self.itemAt(event.scenePos(), QTransform())
        if (
            isinstance(item, _PortItem)
            and item.is_output
            and item.node.is_runnable
            and event.button() == Qt.MouseButton.LeftButton
        ):
            self._wire_source = item.node
            self._temp_edge = _TempEdge()
            self.addItem(self._temp_edge)
            self._temp_edge.set_endpoints(item.scenePos(), event.scenePos())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._temp_edge and self._wire_source:
            self._temp_edge.set_endpoints(
                self._wire_source.output_port_scene_pos(),
                event.scenePos(),
            )
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._temp_edge and self._wire_source:
            item = self.itemAt(event.scenePos(), QTransform())
            target_node = None
            if isinstance(item, _PortItem) and not item.is_output and item.node.is_runnable:
                target_node = item.node
            else:
                node = self._node_from_item(item)
                if node is not None and node is not self._wire_source and node.is_runnable:
                    target_node = node
            if target_node is not None:
                self.add_edge(self._wire_source, target_node)
            self.removeItem(self._temp_edge)
            self._temp_edge = None
            self._wire_source = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    @staticmethod
    def _node_from_item(item) -> Optional[WorkflowNode]:
        while item is not None:
            if isinstance(item, WorkflowNode):
                return item
            item = item.parentItem()
        return None

    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:
        super().drawBackground(painter, rect)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        grid = 32
        left = int(math.floor(rect.left() / grid) * grid)
        top = int(math.floor(rect.top() / grid) * grid)
        pen_minor = QPen(QColor("#D0D0D0"))
        pen_minor.setWidth(0)
        painter.setPen(pen_minor)
        x = left
        while x < rect.right():
            painter.drawLine(int(x), int(rect.top()), int(x), int(rect.bottom()))
            x += grid
        y = top
        while y < rect.bottom():
            painter.drawLine(int(rect.left()), int(y), int(rect.right()), int(y))
            y += grid


class WorkflowView(QGraphicsView):
    def __init__(self, scene: WorkflowScene, parent=None):
        super().__init__(scene, parent)
        self.setObjectName("WorkflowView")
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.BoundingRectViewportUpdate)
        self._panning = False
        self._pan_start = QPointF()

    def wheelEvent(self, event: QWheelEvent):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton or (
            event.button() == Qt.MouseButton.LeftButton
            and event.modifiers() & Qt.KeyboardModifier.AltModifier
        ):
            self._panning = True
            self._pan_start = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning:
            delta = event.position() - self._pan_start
            self._pan_start = event.position()
            self.horizontalScrollBar().setValue(int(self.horizontalScrollBar().value() - delta.x()))
            self.verticalScrollBar().setValue(int(self.verticalScrollBar().value() - delta.y()))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._panning and event.button() in (
            Qt.MouseButton.MiddleButton,
            Qt.MouseButton.LeftButton,
        ):
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)


class WorkflowCanvasPanel(QWidget):
    """Host widget for the node workflow board."""

    open_workspace = pyqtSignal(int)
    open_action = pyqtSignal(str)
    run_requested = pyqtSignal()
    stop_requested = pyqtSignal()
    menu_requested = pyqtSignal()  # open app ⋯ menu near HUD

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("WorkflowCanvasPanel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.scene = WorkflowScene(self)
        self.scene.node_activated.connect(self.open_workspace.emit)
        self.scene.action_activated.connect(self.open_action.emit)
        self.view = WorkflowView(self.scene, self)
        layout.addWidget(self.view, stretch=1)

        # Corner HUD — no full-width toolbar.
        self._hud = QWidget(self)
        self._hud.setObjectName("WorkflowHud")
        hud_layout = QHBoxLayout(self._hud)
        hud_layout.setContentsMargins(8, 8, 8, 8)
        hud_layout.setSpacing(6)

        self.menu_btn = QPushButton("⋯")
        self.menu_btn.setFixedSize(32, 28)
        self.menu_btn.setToolTip("文件与工具")
        self.menu_btn.clicked.connect(self.menu_requested.emit)
        hud_layout.addWidget(self.menu_btn)

        self.run_btn = QPushButton("运行")
        self.run_btn.setObjectName("PrimaryButton")
        self.run_btn.setFixedSize(56, 28)
        self.run_btn.setToolTip("按主链路运行工作流")
        self.run_btn.clicked.connect(self.run_requested.emit)
        hud_layout.addWidget(self.run_btn)

        self.stop_btn = QPushButton("停止")
        self.stop_btn.setFixedSize(48, 28)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_requested.emit)
        hud_layout.addWidget(self.stop_btn)

        fit_btn = QPushButton("适应")
        fit_btn.setFixedSize(48, 28)
        fit_btn.clicked.connect(self.fit_view)
        hud_layout.addWidget(fit_btn)

        reset_btn = QPushButton("重置")
        reset_btn.setFixedSize(48, 28)
        reset_btn.setToolTip("重置节点布局")
        reset_btn.clicked.connect(self.reset_layout)
        hud_layout.addWidget(reset_btn)

        self._hud.setStyleSheet(
            f"""
            QWidget#WorkflowHud {{
                background: {Theme.SURFACE_2};
                border: 1px solid {Theme.BORDER_STRONG};
            }}
            """
        )
        self._hud.adjustSize()
        self._hud.raise_()

        self.log_label = QLabel(self)
        self.log_label.setObjectName("MutedText")
        self.log_label.setText("双击节点打开展开卡 · 运行只走主链路")
        self.log_label.setStyleSheet(
            f"background: {Theme.SURFACE_2}; border: 1px solid {Theme.BORDER}; padding: 4px 8px;"
        )
        self.log_label.adjustSize()
        self.log_label.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._place_hud()

    def showEvent(self, event):
        super().showEvent(event)
        self._place_hud()
        if not getattr(self, "_fitted_once", False):
            self._fitted_once = True
            self.fit_view()

    def _place_hud(self) -> None:
        margin = 12
        self._hud.adjustSize()
        self._hud.move(self.width() - self._hud.width() - margin, margin)
        self.log_label.adjustSize()
        self.log_label.move(margin, self.height() - self.log_label.height() - margin)
        self._hud.raise_()
        self.log_label.raise_()

    def collect_graph(self) -> tuple[list[str], list[tuple[str, str]]]:
        """Only main (runnable) nodes participate in workflow execution."""
        keys = [k for k, n in self.scene.nodes.items() if n.is_runnable]
        edges: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for node in self.scene.nodes.values():
            if not node.is_runnable:
                continue
            for edge in node.edges:
                if not isinstance(edge, WorkflowEdge):
                    continue
                if not edge.source.is_runnable or not edge.target.is_runnable:
                    continue
                pair = (edge.source.spec.key, edge.target.spec.key)
                if pair not in seen:
                    seen.add(pair)
                    edges.append(pair)
        return keys, edges

    def set_node_status(self, key: str, status: str, detail: str = "") -> None:
        node = self.scene.nodes.get(key)
        if node:
            node.set_status(status, detail)

    def reset_all_status(self) -> None:
        for node in self.scene.nodes.values():
            node.set_status("idle")

    def set_running_ui(self, running: bool) -> None:
        self.run_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)

    def append_log(self, text: str) -> None:
        self.log_label.setText(text)
        self.log_label.adjustSize()
        self._place_hud()

    def fit_view(self) -> None:
        items = [n for n in self.scene.nodes.values()]
        if not items:
            return
        rect = items[0].sceneBoundingRect()
        for n in items[1:]:
            rect = rect.united(n.sceneBoundingRect())
        self.view.fitInView(rect.adjusted(-80, -80, 80, 80), Qt.AspectRatioMode.KeepAspectRatio)

    def reset_layout(self) -> None:
        self.scene.clear()
        self.scene.nodes.clear()
        self.scene._wire_source = None
        self.scene._temp_edge = None
        self.scene._build_default_graph()
        self.fit_view()
