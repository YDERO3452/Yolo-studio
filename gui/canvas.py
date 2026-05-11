"""Annotation canvas — multi-mode drawing for bbox, polygon, OBB, and keypoint."""

import copy
from collections import deque
from enum import Enum

from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QRect, QPoint, QPointF, QSize, pyqtSignal
from PyQt6.QtGui import (
    QPainter, QPen, QColor, QPixmap, QImage, QFont, QPolygon
)

import cv2
import numpy as np


class CanvasMode(str, Enum):
    EDIT = "edit"
    CREATE_BBOX = "create_bbox"
    CREATE_POLYGON = "create_polygon"
    CREATE_OBB = "create_obb"
    CREATE_KEYPOINT = "create_keypoint"


def _shape_type_value(shape_type) -> str:
    """Return a stable string value for ShapeType enum or legacy strings."""
    return getattr(shape_type, "value", str(shape_type))


class AnnotationCanvas(QWidget):
    """Canvas for displaying images and drawing annotations."""

    shape_created = pyqtSignal(dict)   # {"type": ShapeType, "class_id": int, "data": ...}
    shape_selected = pyqtSignal(int)   # index
    shape_deleted = pyqtSignal(int)    # index
    shapes_changed = pyqtSignal()      # undo/redo/clear — annotation list needs refresh
    mouse_position = pyqtSignal(int, int)
    zoom_changed = pyqtSignal(float)
    class_switch_requested = pyqtSignal(int)  # 数字键切换类别 (0-based index)
    edit_label_requested = pyqtSignal(int)    # 双击编辑标注标签 (shape index)

    # Colors for classes
    COLORS = [
        QColor(255, 0, 0), QColor(0, 255, 0), QColor(0, 0, 255),
        QColor(255, 255, 0), QColor(255, 0, 255), QColor(0, 255, 255),
        QColor(128, 0, 0), QColor(0, 128, 0), QColor(0, 0, 128),
        QColor(128, 128, 0), QColor(128, 0, 128), QColor(0, 128, 128),
        QColor(64, 0, 0), QColor(0, 64, 0), QColor(0, 0, 64),
        QColor(64, 64, 0), QColor(64, 0, 64), QColor(0, 64, 64),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Image state
        self.original_image = None
        self.display_pixmap = None
        self.image_width = 0
        self.image_height = 0
        self._resized_buf = None      # Keep reference to prevent GC while displaying
        self._display_bytes = None    # Keep reference to prevent GC while QImage uses it

        # Mode
        self.current_mode = CanvasMode.CREATE_BBOX
        self.current_class_id = 0

        # Shapes storage — each shape is a dict from Annotation.to_canvas_shape()
        self.shapes = []
        self.selected_shape = -1
        self.hover_shape = -1

        # BBOX drawing state
        self.drawing_bbox = False
        self.bbox_start = QPoint()
        self.bbox_end = QPoint()

        # POLYGON drawing state
        self.polygon_points = []  # list of QPoint (widget coords)
        self.drawing_polygon = False

        # OBB drawing state
        self.drawing_obb = False
        self.obb_start = QPoint()
        self.obb_end = QPoint()
        self.obb_angle = 0.0  # degrees

        # KEYPOINT drawing — no intermediate state, single click

        # Move / resize state
        self.moving = False
        self.move_offset = QPoint()
        self.resize_handle = None
        self.resize_vertex = -1  # for polygon vertex resize

        # Zoom / pan
        self.zoom_level = 1.0
        self.pan_offset = QPointF(0, 0)
        self.panning = False
        self.pan_start = QPoint()
        self.min_zoom = 0.1
        self.max_zoom = 20.0
        self.fit_scale = 1.0
        self.fit_padding = 16

        # Display options
        self.show_labels = True
        self.shape_opacity = 0.18  # Fill opacity for shapes

        # Crosshair guides
        self.crosshair_show = True
        self.crosshair_width = 1.0
        self.crosshair_color = "#00FF00"
        self.crosshair_opacity = 0.5
        self._last_mouse_pos = None  # For crosshair tracking

        # Brightness / contrast adjustments (applied to display)
        self._brightness = 0
        self._contrast = 0
        self._adjusted_image = None  # Cached adjusted image

        # Undo / redo
        self.undo_stack = deque(maxlen=50)
        self.redo_stack = deque(maxlen=50)

        self.classes = ["目标"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_classes(self, classes: list[str]):
        self.classes = classes

    def get_shapes(self) -> list[dict]:
        """Return a copy of the current shapes list."""
        return self.shapes

    def set_mode(self, mode: CanvasMode):
        """Set the current canvas interaction mode."""
        self.current_mode = mode
        self.cancel_drawing()
        self.update()

    def set_current_class_id(self, class_id: int):
        """Set the current drawing class by index."""
        self.current_class_id = class_id

    def set_cross_line(self, show: bool, width: float = 1.0, color: str = "#00FF00", opacity: float = 0.5):
        """Set crosshair guide options."""
        self.crosshair_show = show
        self.crosshair_width = width
        self.crosshair_color = color
        self.crosshair_opacity = opacity
        self.update()

    def load_image(self, image_path: str) -> bool:
        # Use numpy to read file (handles non-ASCII paths like Chinese characters)
        try:
            with open(image_path, "rb") as f:
                data = np.frombuffer(f.read(), dtype=np.uint8)
            self.original_image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception:
            self.original_image = cv2.imread(image_path)

        if self.original_image is None:
            return False
        self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        self.image_height, self.image_width = self.original_image.shape[:2]
        self._adjusted_image = None  # Reset brightness/contrast cache
        self._brightness = 0
        self._contrast = 0
        self.fit_to_window()
        return True

    def load_image_array(self, image: np.ndarray):
        self.original_image = image.copy()
        self.image_height, self.image_width = self.original_image.shape[:2]
        self.fit_to_window()

    def fit_to_window(self, target_size: QSize | None = None):
        if self.original_image is None:
            return
        if target_size is not None and target_size.isValid():
            widget_w, widget_h = target_size.width(), target_size.height()
        else:
            viewport = self.parentWidget()
            widget_w = viewport.width() if viewport is not None else self.width()
            widget_h = viewport.height() if viewport is not None else self.height()
        if widget_w <= 0 or widget_h <= 0:
            return
        available_w = max(1, widget_w - self.fit_padding * 2)
        available_h = max(1, widget_h - self.fit_padding * 2)
        scale_x = available_w / self.image_width
        scale_y = available_h / self.image_height
        self.fit_scale = min(scale_x, scale_y)
        self.zoom_level = 1.0
        self.pan_offset = QPointF(0, 0)
        self._update_display()
        self.zoom_changed.emit(self.zoom_level)
        self.update()

    def set_shapes(self, shapes: list[dict]):
        self.shapes = shapes
        self.selected_shape = -1
        self.update()

    def clear_shapes(self):
        self.shapes.clear()
        self.selected_shape = -1
        self.update()

    def cancel_drawing(self):
        """Cancel any in-progress drawing."""
        self.drawing_bbox = False
        self.drawing_polygon = False
        self.drawing_obb = False
        self.polygon_points.clear()
        self.update()

    # Undo / redo

    def push_undo(self):
        self.undo_stack.append(copy.deepcopy(self.shapes))
        self.redo_stack.clear()

    def undo(self):
        if not self.undo_stack:
            return
        self.redo_stack.append(copy.deepcopy(self.shapes))
        self.shapes = self.undo_stack.pop()
        self.selected_shape = -1
        self.shapes_changed.emit()
        self.update()

    def redo(self):
        if not self.redo_stack:
            return
        self.undo_stack.append(copy.deepcopy(self.shapes))
        self.shapes = self.redo_stack.pop()
        self.selected_shape = -1
        self.shapes_changed.emit()
        self.update()

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def _update_display(self):
        if self.original_image is None:
            return
        widget_w, widget_h = self.width(), self.height()
        if widget_w <= 0 or widget_h <= 0:
            return

        # Apply brightness/contrast adjustments
        src = self._adjusted_image if self._adjusted_image is not None else self.original_image

        effective_scale = self.fit_scale * self.zoom_level
        display_w = max(1, int(self.image_width * effective_scale))
        display_h = max(1, int(self.image_height * effective_scale))
        self._resized_buf = cv2.resize(src, (display_w, display_h))
        h, w, ch = self._resized_buf.shape
        # QImage must not reference numpy buffer that can be GC'd.
        # Use ascontiguousarray + data to avoid an extra copy when possible.
        self._resized_buf = np.ascontiguousarray(self._resized_buf)
        try:
            self._display_bytes = self._resized_buf.data
        except MemoryError:
            import gc
            gc.collect()
            self._display_bytes = self._resized_buf.tobytes()
        q_image = QImage(self._display_bytes, w, h, ch * w, QImage.Format.Format_RGB888)
        self.display_pixmap = QPixmap.fromImage(q_image)

    def apply_brightness_contrast(self, brightness: int = 0, contrast: int = 0):
        """Apply brightness and contrast adjustments to the display image.

        Args:
            brightness: -100 to 100
            contrast: -100 to 100
        """
        self._brightness = brightness
        self._contrast = contrast

        if self.original_image is None:
            return

        img = self.original_image.astype(np.float32)

        # Apply brightness
        if brightness != 0:
            img = img + brightness * 2.55  # Scale -100..100 to -255..255

        # Apply contrast
        if contrast != 0:
            factor = (259 * (contrast + 255)) / (255 * (259 - contrast))
            img = factor * (img - 128) + 128

        img = np.clip(img, 0, 255).astype(np.uint8)
        self._adjusted_image = img
        self._update_display()
        self.update()

    def _get_transform(self):
        effective_scale = self.fit_scale * self.zoom_level
        display_w = int(self.image_width * effective_scale)
        display_h = int(self.image_height * effective_scale)
        offset_x = (self.width() - display_w) // 2 + int(self.pan_offset.x())
        offset_y = (self.height() - display_h) // 2 + int(self.pan_offset.y())
        return offset_x, offset_y, effective_scale

    def _widget_to_image(self, pos: QPoint) -> tuple:
        ox, oy, scale = self._get_transform()
        img_x = max(0, min(int((pos.x() - ox) / scale), self.image_width))
        img_y = max(0, min(int((pos.y() - oy) / scale), self.image_height))
        return img_x, img_y

    def _image_to_widget(self, img_x: int, img_y: int) -> QPoint:
        ox, oy, scale = self._get_transform()
        return QPoint(int(img_x * scale) + ox, int(img_y * scale) + oy)

    # ------------------------------------------------------------------
    # Resize
    # ------------------------------------------------------------------

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fit_to_window()

    # ------------------------------------------------------------------
    # Paint
    # ------------------------------------------------------------------

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(32, 34, 34))

        if not self.display_pixmap:
            painter.end()
            return

        ox, oy, scale = self._get_transform()
        painter.drawPixmap(ox, oy, self.display_pixmap)

        # Draw existing shapes
        for i, shape in enumerate(self.shapes):
            self._paint_shape(painter, shape, i, ox, oy, scale)

        # Draw in-progress drawing
        self._paint_active_drawing(painter, ox, oy, scale)

        # Draw crosshair guides
        if self.crosshair_show and self._last_mouse_pos is not None and self.display_pixmap:
            pen = QPen(
                QColor(self.crosshair_color),
                max(1, int(round(self.crosshair_width))),
                Qt.PenStyle.DashLine,
            )
            painter.setPen(pen)
            painter.setOpacity(self.crosshair_opacity)
            # Vertical line
            painter.drawLine(
                QPointF(self._last_mouse_pos.x(), 0),
                QPointF(self._last_mouse_pos.x(), self.height()),
            )
            # Horizontal line
            painter.drawLine(
                QPointF(0, self._last_mouse_pos.y()),
                QPointF(self.width(), self._last_mouse_pos.y()),
            )
            painter.setOpacity(1.0)

        # Zoom indicator
        if self.zoom_level != 1.0:
            painter.setPen(QColor(200, 200, 200, 180))
            painter.setFont(QFont("Arial", 10))
            painter.drawText(10, 20, f"{self.zoom_level:.1f}x")

        painter.end()

    def _paint_shape(self, painter, shape, index, ox, oy, scale):
        from core.annotation import ShapeType
        stype = _shape_type_value(shape["type"])
        cid = shape["class_id"]
        data = shape["data"]
        color = self.COLORS[cid % len(self.COLORS)]

        selected = (index == self.selected_shape)
        hovered = (index == self.hover_shape)

        if selected:
            pen = QPen(QColor(0, 255, 255), 3)
        elif hovered:
            pen = QPen(color, 2, Qt.PenStyle.DashLine)
        else:
            pen = QPen(color, 2)

        painter.setPen(pen)

        if stype == ShapeType.BBOX.value:
            sx1 = int(data["x1"] * scale) + ox
            sy1 = int(data["y1"] * scale) + oy
            sx2 = int(data["x2"] * scale) + ox
            sy2 = int(data["y2"] * scale) + oy
            # Fill with semi-transparent color
            fill_color = QColor(color)
            fill_color.setAlphaF(self.shape_opacity)
            painter.setBrush(fill_color)
            painter.drawRect(QRect(QPoint(sx1, sy1), QPoint(sx2, sy2)))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            if selected:
                self._draw_resize_handles_rect(painter, sx1, sy1, sx2, sy2)
            if self.show_labels and cid < len(self.classes):
                self._draw_label(painter, self.classes[cid], sx1, sy1, color)

        elif stype == ShapeType.POLYGON.value:
            pts = [QPoint(int(p[0] * scale) + ox, int(p[1] * scale) + oy) for p in data["points"]]
            if len(pts) >= 2:
                fill_color = QColor(color)
                fill_color.setAlphaF(self.shape_opacity)
                painter.setBrush(fill_color)
                painter.drawPolygon(QPolygon(pts))
                painter.setBrush(Qt.BrushStyle.NoBrush)
            if selected:
                self._draw_vertex_handles(painter, pts)
            if self.show_labels and cid < len(self.classes) and pts:
                self._draw_label(painter, self.classes[cid], pts[0].x(), pts[0].y(), color)

        elif stype == ShapeType.OBB.value:
            pts = [QPoint(int(p[0] * scale) + ox, int(p[1] * scale) + oy) for p in data["corners"]]
            if len(pts) == 4:
                fill_color = QColor(color)
                fill_color.setAlphaF(self.shape_opacity)
                painter.setBrush(fill_color)
                painter.drawPolygon(QPolygon(pts))
                painter.setBrush(Qt.BrushStyle.NoBrush)
            if selected:
                self._draw_vertex_handles(painter, pts)
            if self.show_labels and cid < len(self.classes) and pts:
                self._draw_label(painter, self.classes[cid], pts[0].x(), pts[0].y(), color)

        elif stype == ShapeType.KEYPOINT.value:
            sx1 = int(data["x1"] * scale) + ox
            sy1 = int(data["y1"] * scale) + oy
            sx2 = int(data["x2"] * scale) + ox
            sy2 = int(data["y2"] * scale) + oy
            fill_color = QColor(color)
            fill_color.setAlphaF(self.shape_opacity * 0.5)  # Lighter fill for keypoint bbox
            painter.setPen(QPen(color, 1))
            painter.setBrush(fill_color)
            painter.drawRect(QRect(QPoint(sx1, sy1), QPoint(sx2, sy2)))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            # Keypoints
            kp_colors = {0: QColor(128, 128, 128), 1: QColor(0, 255, 255), 2: QColor(0, 255, 0)}
            for kx, ky, vis in data.get("keypoints", []):
                kpx = int(kx * scale) + ox
                kpy = int(ky * scale) + oy
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(kp_colors.get(vis, QColor(255, 255, 255)))
                painter.drawEllipse(QPoint(kpx, kpy), 4, 4)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            if selected:
                self._draw_resize_handles_rect(painter, sx1, sy1, sx2, sy2)
            if self.show_labels and cid < len(self.classes):
                self._draw_label(painter, self.classes[cid], sx1, sy1, color)

    def _paint_active_drawing(self, painter, ox, oy, scale):
        from core.annotation import ShapeType

        if self.drawing_bbox:
            pen = QPen(QColor(255, 255, 0), 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawRect(QRect(self.bbox_start, self.bbox_end).normalized())

        elif self.drawing_polygon and self.polygon_points:
            pen = QPen(QColor(255, 255, 0), 2)
            painter.setPen(pen)
            # Draw lines between committed points
            for j in range(len(self.polygon_points) - 1):
                painter.drawLine(self.polygon_points[j], self.polygon_points[j + 1])
            # Draw dashed closing line to mouse cursor (last point is live cursor position)
            if len(self.polygon_points) >= 2:
                painter.setPen(QPen(QColor(255, 255, 0), 1, Qt.PenStyle.DashLine))
                painter.drawLine(self.polygon_points[-1], self.polygon_points[0])
            # Draw vertices
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(255, 255, 0))
            for pt in self.polygon_points:
                painter.drawEllipse(pt, 4, 4)
            painter.setBrush(Qt.BrushStyle.NoBrush)

        elif self.drawing_obb:
            pen = QPen(QColor(255, 255, 0), 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            # Draw the base rectangle, then show rotation preview
            rect = QRect(self.obb_start, self.obb_end).normalized()
            if rect.width() > 5 and rect.height() > 5:
                cx = rect.center().x()
                cy = rect.center().y()
                painter.save()
                painter.translate(cx, cy)
                painter.rotate(self.obb_angle)
                painter.drawRect(QRect(-rect.width() // 2, -rect.height() // 2, rect.width(), rect.height()))
                painter.restore()

    def _draw_resize_handles_rect(self, painter, sx1, sy1, sx2, sy2):
        painter.setPen(QPen(QColor(0, 255, 255), 1))
        painter.setBrush(QColor(0, 255, 255))
        hs = 6
        for hx, hy in [(sx1, sy1), (sx2, sy1), (sx1, sy2), (sx2, sy2)]:
            painter.drawRect(hx - hs // 2, hy - hs // 2, hs, hs)
        painter.setBrush(Qt.BrushStyle.NoBrush)

    def _draw_vertex_handles(self, painter, points):
        painter.setPen(QPen(QColor(0, 255, 255), 1))
        painter.setBrush(QColor(0, 255, 255))
        hs = 6
        for pt in points:
            painter.drawRect(pt.x() - hs // 2, pt.y() - hs // 2, hs, hs)
        painter.setBrush(Qt.BrushStyle.NoBrush)

    def _draw_label(self, painter, label, x, y, color):
        font = QFont("Arial", 10)
        painter.setFont(font)
        fm = painter.fontMetrics()
        tw = fm.horizontalAdvance(label)
        th = fm.height()
        painter.fillRect(QRect(x, y - th - 4, tw + 8, th + 4), color)
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(QPoint(x + 4, y - 4), label)

    # ------------------------------------------------------------------
    # Mouse events
    # ------------------------------------------------------------------

    def mousePressEvent(self, event):
        from core.annotation import ShapeType
        pos = event.pos()

        if event.button() == Qt.MouseButton.MiddleButton:
            self.panning = True
            self.pan_start = pos
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        if event.button() == Qt.MouseButton.LeftButton:
            mode = self.current_mode

            if mode == CanvasMode.EDIT:
                # Check resize handle
                handle, vidx = self._get_resize_handle(pos)
                if handle is not None:
                    self.resize_handle = handle
                    self.resize_vertex = vidx
                    return
                # Check shape selection
                clicked = self._get_shape_at(pos)
                if clicked >= 0:
                    self.selected_shape = clicked
                    self.moving = True
                    self.move_offset = pos
                    self.shape_selected.emit(clicked)
                    self.update()
                    return
                # Deselect
                self.selected_shape = -1
                self.update()

            elif mode == CanvasMode.CREATE_BBOX:
                self.drawing_bbox = True
                self.bbox_start = pos
                self.bbox_end = pos

            elif mode == CanvasMode.CREATE_POLYGON:
                if not self.drawing_polygon:
                    self.drawing_polygon = True
                    self.polygon_points = [pos, pos]  # first committed + live cursor
                else:
                    # Insert a new committed point before the live cursor
                    self.polygon_points.insert(-1, pos)
                self.update()

            elif mode == CanvasMode.CREATE_OBB:
                self.drawing_obb = True
                self.obb_start = pos
                self.obb_end = pos
                self.obb_angle = 0.0

            elif mode == CanvasMode.CREATE_KEYPOINT:
                img_x, img_y = self._widget_to_image(pos)
                self.push_undo()
                # Use a reasonable default bbox size (50px) around the click point
                half = 25
                x1 = max(0, img_x - half)
                y1 = max(0, img_y - half)
                x2 = min(self.image_width, img_x + half)
                y2 = min(self.image_height, img_y + half)
                shape = {
                    "type": ShapeType.KEYPOINT,
                    "class_id": self.current_class_id,
                    "data": {
                        "x1": x1, "y1": y1,
                        "x2": x2, "y2": y2,
                        "keypoints": [(img_x, img_y, 2)],
                    },
                }
                self.shapes.append(shape)
                self.shape_created.emit(shape)
                self.update()

        elif event.button() == Qt.MouseButton.RightButton:
            if self.drawing_polygon:
                # Close polygon
                self._finish_polygon()
                return
            # Delete shape on right click in edit mode
            clicked = self._get_shape_at(pos)
            if clicked >= 0:
                self.push_undo()
                self.shape_deleted.emit(clicked)

    def mouseMoveEvent(self, event):
        from core.annotation import ShapeType
        pos = event.pos()

        # Track mouse position for crosshair
        self._last_mouse_pos = pos

        img_x, img_y = self._widget_to_image(pos)
        self.mouse_position.emit(img_x, img_y)

        if self.panning:
            dx = pos.x() - self.pan_start.x()
            dy = pos.y() - self.pan_start.y()
            self.pan_start = pos
            self.pan_offset = QPointF(self.pan_offset.x() + dx, self.pan_offset.y() + dy)
            self.update()
            return

        if self.drawing_bbox:
            self.bbox_end = pos
            self.update()

        elif self.drawing_polygon:
            # Keep last point as live cursor preview
            if self.polygon_points:
                self.polygon_points[-1] = pos
            self.update()

        elif self.drawing_obb:
            self.obb_end = pos
            self.update()

        elif self.moving and self.selected_shape >= 0:
            dx = pos.x() - self.move_offset.x()
            dy = pos.y() - self.move_offset.y()
            self.move_offset = pos
            self._move_shape(self.selected_shape, dx, dy)
            self.update()

        elif self.resize_handle is not None and self.selected_shape >= 0:
            self._resize_shape(self.selected_shape, pos)
            self.update()

        else:
            self.hover_shape = self._get_shape_at(pos)
            self.update()

    def mouseReleaseEvent(self, event):
        from core.annotation import ShapeType

        if event.button() == Qt.MouseButton.MiddleButton:
            self.panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            return

        if event.button() == Qt.MouseButton.LeftButton:
            if self.drawing_bbox:
                self.drawing_bbox = False
                x1, y1 = self._widget_to_image(self.bbox_start)
                x2, y2 = self._widget_to_image(self.bbox_end)
                if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
                    nx1, nx2 = min(x1, x2), max(x1, x2)
                    ny1, ny2 = min(y1, y2), max(y1, y2)
                    self.push_undo()
                    shape = {
                        "type": ShapeType.BBOX,
                        "class_id": self.current_class_id,
                        "data": {"x1": nx1, "y1": ny1, "x2": nx2, "y2": ny2},
                    }
                    self.shapes.append(shape)
                    self.shape_created.emit(shape)
                    self.update()

            elif self.drawing_obb:
                self.drawing_obb = False
                x1, y1 = self._widget_to_image(self.obb_start)
                x2, y2 = self._widget_to_image(self.obb_end)
                if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
                    # Compute rotated corners
                    import math
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    w = abs(x2 - x1)
                    h = abs(y2 - y1)
                    angle_rad = math.radians(self.obb_angle)
                    cos_a = math.cos(angle_rad)
                    sin_a = math.sin(angle_rad)
                    corners = []
                    for dx, dy in [(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)]:
                        rx = cx + dx * cos_a - dy * sin_a
                        ry = cy + dx * sin_a + dy * cos_a
                        corners.append((int(rx), int(ry)))
                    self.push_undo()
                    shape = {
                        "type": ShapeType.OBB,
                        "class_id": self.current_class_id,
                        "data": {"corners": corners},
                    }
                    self.shapes.append(shape)
                    self.shape_created.emit(shape)
                    self.update()

            if self.moving:
                self.moving = False
            self.resize_handle = None
            self.resize_vertex = -1

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.drawing_polygon:
                self._finish_polygon()
                return
            # 双击编辑已有标注的标签
            if self.current_mode == CanvasMode.EDIT:
                clicked = self._get_shape_at(event.pos())
                if clicked >= 0:
                    self.selected_shape = clicked
                    self.edit_label_requested.emit(clicked)
                    self.update()
                    return

    def leaveEvent(self, event):
        """Clear crosshair when mouse leaves canvas."""
        self._last_mouse_pos = None
        self.update()
        super().leaveEvent(event)

    def wheelEvent(self, event):
        if self.original_image is None:
            return
        mouse_pos = event.position()
        old_img_x, old_img_y = self._widget_to_image(QPoint(int(mouse_pos.x()), int(mouse_pos.y())))

        # If drawing OBB, use wheel for rotation
        if self.drawing_obb:
            delta = event.angleDelta().y()
            self.obb_angle += 5.0 if delta > 0 else -5.0
            self.update()
            return

        delta = event.angleDelta().y()
        zoom_factor = 1.1 if delta > 0 else 1.0 / 1.1
        new_zoom = max(self.min_zoom, min(self.max_zoom, self.zoom_level * zoom_factor))
        if new_zoom == self.zoom_level:
            return
        self.zoom_level = new_zoom

        effective_scale = self.fit_scale * self.zoom_level
        new_screen_x = old_img_x * effective_scale + self.pan_offset.x() + (self.width() - int(self.image_width * effective_scale)) // 2
        new_screen_y = old_img_y * effective_scale + self.pan_offset.y() + (self.height() - int(self.image_height * effective_scale)) // 2
        self.pan_offset = QPointF(
            self.pan_offset.x() + mouse_pos.x() - new_screen_x,
            self.pan_offset.y() + mouse_pos.y() - new_screen_y,
        )
        self._update_display()
        self.zoom_changed.emit(self.zoom_level)
        self.update()

    def keyPressEvent(self, event):
        key = event.key()
        modifiers = event.modifiers()

        if key == Qt.Key.Key_Z and modifiers == Qt.KeyboardModifier.ControlModifier:
            self.undo()
            return
        if key == Qt.Key.Key_Y and modifiers == Qt.KeyboardModifier.ControlModifier:
            self.redo()
            return

        if key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            if self.selected_shape >= 0:
                self.push_undo()
                self.shape_deleted.emit(self.selected_shape)
            return

        if key == Qt.Key.Key_Escape:
            self.cancel_drawing()
            self.selected_shape = -1
            self.update()
            return

        if key == Qt.Key.Key_Return or key == Qt.Key.Key_Enter:
            if self.drawing_polygon and len(self.polygon_points) >= 3:
                self._finish_polygon()
            return

        if key == Qt.Key.Key_F:
            self.fit_to_window()
            return

        if key == Qt.Key.Key_L:
            self.show_labels = not self.show_labels
            self.update()
            return

        if key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self.zoom_level = min(self.max_zoom, self.zoom_level * 1.2)
            self._update_display()
            self.zoom_changed.emit(self.zoom_level)
            self.update()
            return
        if key == Qt.Key.Key_Minus:
            self.zoom_level = max(self.min_zoom, self.zoom_level / 1.2)
            self._update_display()
            self.zoom_changed.emit(self.zoom_level)
            self.update()
            return

        if Qt.Key.Key_1 <= key <= Qt.Key.Key_9:
            idx = key - Qt.Key.Key_1
            if idx < len(self.classes):
                self.current_class_id = idx
                self.class_switch_requested.emit(idx)
            return

        if key == Qt.Key.Key_0 and len(self.classes) >= 10:
            self.current_class_id = 9
            self.class_switch_requested.emit(9)
            return

    # ------------------------------------------------------------------
    # Polygon finish
    # ------------------------------------------------------------------

    def _finish_polygon(self):
        from core.annotation import ShapeType
        self.drawing_polygon = False
        # Remove the trailing live-cursor point before converting
        if self.polygon_points and len(self.polygon_points) >= 2:
            self.polygon_points.pop()  # remove live cursor
        if len(self.polygon_points) >= 3:
            # Convert widget coords to image coords
            img_points = [self._widget_to_image(pt) for pt in self.polygon_points]
            if len(img_points) >= 3:
                self.push_undo()
                shape = {
                    "type": ShapeType.POLYGON,
                    "class_id": self.current_class_id,
                    "data": {"points": img_points},
                }
                self.shapes.append(shape)
                self.shape_created.emit(shape)
                self.update()
        self.polygon_points.clear()
        self.update()

    # ------------------------------------------------------------------
    # Hit testing
    # ------------------------------------------------------------------

    def _get_shape_at(self, pos: QPoint) -> int:
        from core.annotation import ShapeType
        ox, oy, scale = self._get_transform()

        for i in reversed(range(len(self.shapes))):
            shape = self.shapes[i]
            stype = _shape_type_value(shape["type"])
            data = shape["data"]

            if stype == ShapeType.BBOX.value:
                sx1 = int(data["x1"] * scale) + ox
                sy1 = int(data["y1"] * scale) + oy
                sx2 = int(data["x2"] * scale) + ox
                sy2 = int(data["y2"] * scale) + oy
                if QRect(QPoint(sx1, sy1), QPoint(sx2, sy2)).normalized().contains(pos):
                    return i

            elif stype == ShapeType.POLYGON.value:
                pts = [QPoint(int(p[0] * scale) + ox, int(p[1] * scale) + oy) for p in data["points"]]
                if len(pts) >= 3:
                    poly = QPolygon(pts)
                    if poly.containsPoint(pos, Qt.FillRule.OddEvenFill):
                        return i

            elif stype == ShapeType.OBB.value:
                pts = [QPoint(int(p[0] * scale) + ox, int(p[1] * scale) + oy) for p in data["corners"]]
                if len(pts) == 4:
                    poly = QPolygon(pts)
                    if poly.containsPoint(pos, Qt.FillRule.OddEvenFill):
                        return i

            elif stype == ShapeType.KEYPOINT.value:
                sx1 = int(data["x1"] * scale) + ox
                sy1 = int(data["y1"] * scale) + oy
                sx2 = int(data["x2"] * scale) + ox
                sy2 = int(data["y2"] * scale) + oy
                if QRect(QPoint(sx1, sy1), QPoint(sx2, sy2)).normalized().contains(pos):
                    return i
                # Also check keypoints
                for kx, ky, _ in data.get("keypoints", []):
                    kpx = int(kx * scale) + ox
                    kpy = int(ky * scale) + oy
                    if abs(pos.x() - kpx) < 10 and abs(pos.y() - kpy) < 10:
                        return i
        return -1

    def _get_resize_handle(self, pos: QPoint):
        """Returns (handle_name, vertex_index) or (None, -1)."""
        from core.annotation import ShapeType
        if self.selected_shape < 0:
            return None, -1

        shape = self.shapes[self.selected_shape]
        stype = _shape_type_value(shape["type"])
        data = shape["data"]
        ox, oy, scale = self._get_transform()
        handle_size = 10

        if stype == ShapeType.BBOX.value:
            sx1 = int(data["x1"] * scale) + ox
            sy1 = int(data["y1"] * scale) + oy
            sx2 = int(data["x2"] * scale) + ox
            sy2 = int(data["y2"] * scale) + oy
            handles = {"tl": (sx1, sy1), "tr": (sx2, sy1), "bl": (sx1, sy2), "br": (sx2, sy2)}
            for name, (hx, hy) in handles.items():
                if abs(pos.x() - hx) < handle_size and abs(pos.y() - hy) < handle_size:
                    return name, -1

        elif stype == ShapeType.POLYGON.value:
            for idx, p in enumerate(data["points"]):
                px = int(p[0] * scale) + ox
                py = int(p[1] * scale) + oy
                if abs(pos.x() - px) < handle_size and abs(pos.y() - py) < handle_size:
                    return "vertex", idx

        elif stype == ShapeType.OBB.value:
            for idx, p in enumerate(data["corners"]):
                px = int(p[0] * scale) + ox
                py = int(p[1] * scale) + oy
                if abs(pos.x() - px) < handle_size and abs(pos.y() - py) < handle_size:
                    return "vertex", idx

        elif stype == ShapeType.KEYPOINT.value:
            sx1 = int(data["x1"] * scale) + ox
            sy1 = int(data["y1"] * scale) + oy
            sx2 = int(data["x2"] * scale) + ox
            sy2 = int(data["y2"] * scale) + oy
            handles = {"tl": (sx1, sy1), "tr": (sx2, sy1), "bl": (sx1, sy2), "br": (sx2, sy2)}
            for name, (hx, hy) in handles.items():
                if abs(pos.x() - hx) < handle_size and abs(pos.y() - hy) < handle_size:
                    return name, -1

        return None, -1

    # ------------------------------------------------------------------
    # Move / resize
    # ------------------------------------------------------------------

    def _move_shape(self, index, dx, dy):
        from core.annotation import ShapeType
        shape = self.shapes[index]
        stype = _shape_type_value(shape["type"])
        data = shape["data"]
        scale = self.fit_scale * self.zoom_level
        dix = dx / scale
        diy = dy / scale

        if stype in (ShapeType.BBOX.value, ShapeType.KEYPOINT.value):
            for key in ("x1", "y1", "x2", "y2"):
                data[key] = int(data[key] + (dix if "x" in key else diy))
            if stype == ShapeType.KEYPOINT.value:
                data["keypoints"] = [
                    (int(kx + dix), int(ky + diy), v)
                    for kx, ky, v in data.get("keypoints", [])
                ]

        elif stype == ShapeType.POLYGON.value:
            data["points"] = [(int(px + dix), int(py + diy)) for px, py in data["points"]]

        elif stype == ShapeType.OBB.value:
            data["corners"] = [(int(cx + dix), int(cy + diy)) for cx, cy in data["corners"]]

    def _resize_shape(self, index, pos):
        from core.annotation import ShapeType
        shape = self.shapes[index]
        stype = _shape_type_value(shape["type"])
        data = shape["data"]
        img_x, img_y = self._widget_to_image(pos)

        if stype in (ShapeType.BBOX.value, ShapeType.KEYPOINT.value):
            if self.resize_handle == "tl":
                data["x1"] = min(img_x, data["x2"] - 5)
                data["y1"] = min(img_y, data["y2"] - 5)
            elif self.resize_handle == "tr":
                data["x2"] = max(img_x, data["x1"] + 5)
                data["y1"] = min(img_y, data["y2"] - 5)
            elif self.resize_handle == "bl":
                data["x1"] = min(img_x, data["x2"] - 5)
                data["y2"] = max(img_y, data["y1"] + 5)
            elif self.resize_handle == "br":
                data["x2"] = max(img_x, data["x1"] + 5)
                data["y2"] = max(img_y, data["y1"] + 5)

        elif stype == ShapeType.POLYGON.value and self.resize_vertex >= 0:
            if self.resize_vertex < len(data["points"]):
                data["points"][self.resize_vertex] = (img_x, img_y)

        elif stype == ShapeType.OBB.value and self.resize_vertex >= 0:
            if self.resize_vertex < len(data["corners"]):
                data["corners"][self.resize_vertex] = (img_x, img_y)
