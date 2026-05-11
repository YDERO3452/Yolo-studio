"""Annotation module — supports bbox, polygon, OBB, and keypoint annotations."""

import copy
import os
from enum import Enum
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from loguru import logger


# ---------------------------------------------------------------------------
# Shape type enum
# ---------------------------------------------------------------------------

class ShapeType(str, Enum):
    BBOX = "bbox"
    POLYGON = "polygon"
    OBB = "obb"
    KEYPOINT = "keypoint"


# ---------------------------------------------------------------------------
# Annotation base class
# ---------------------------------------------------------------------------

class Annotation:
    """Base class for all annotation types."""

    def __init__(self, class_id: int, shape_type: ShapeType):
        self.class_id = class_id
        self.shape_type = shape_type

    def to_yolo(self) -> str:
        raise NotImplementedError

    def to_canvas_shape(self, img_width: int, img_height: int) -> dict:
        """Convert to a canvas-ready dict for rendering."""
        raise NotImplementedError

    def copy(self):
        return copy.deepcopy(self)

    @staticmethod
    def from_yolo_line(line: str) -> "Annotation":
        """Auto-detect annotation type from a YOLO-format text line."""
        parts = line.strip().split()
        if len(parts) < 2:
            raise ValueError(f"Invalid YOLO line: {line}")

        class_id = int(parts[0])
        values = [float(x) for x in parts[1:]]
        n = len(values)

        if n == 4:
            # bbox: cls x_c y_c w h
            return BBoxAnnotation(class_id, *values)

        if n == 8:
            # OBB: cls x1 y1 x2 y2 x3 y3 x4 y4 (4 corner points)
            corners = [(values[i], values[i + 1]) for i in range(0, 8, 2)]
            return OBBoxAnnotation(class_id, corners)

        # Keypoint: cls x_c y_c w h kx1 ky1 v1 kx2 ky2 v2 ...
        # Extra values after bbox (4) must be divisible by 3
        if n >= 7 and (n - 4) % 3 == 0:
            xc, yc, w, h = values[0], values[1], values[2], values[3]
            kps = []
            for i in range(4, n, 3):
                kps.append((values[i], values[i + 1], int(values[i + 2])))
            return KeypointAnnotation(class_id, xc, yc, w, h, kps)

        # Polygon: cls x1 y1 x2 y2 ... xN yN (N >= 3 points, even count)
        if n >= 6 and n % 2 == 0:
            points = [(values[i], values[i + 1]) for i in range(0, n, 2)]
            return PolygonAnnotation(class_id, points)

        raise ValueError(f"Cannot detect annotation type from line: {line}")


# ---------------------------------------------------------------------------
# BBox annotation (axis-aligned rectangle)
# ---------------------------------------------------------------------------

class BBoxAnnotation(Annotation):
    """Axis-aligned bounding box in YOLO format (normalized)."""

    def __init__(self, class_id: int, x_center: float, y_center: float,
                 width: float, height: float):
        super().__init__(class_id, ShapeType.BBOX)
        self.x_center = x_center
        self.y_center = y_center
        self.width = width
        self.height = height

    def to_yolo(self) -> str:
        return f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f}"

    def to_xyxy(self, img_width: int, img_height: int) -> tuple:
        x1 = (self.x_center - self.width / 2) * img_width
        y1 = (self.y_center - self.height / 2) * img_height
        x2 = (self.x_center + self.width / 2) * img_width
        y2 = (self.y_center + self.height / 2) * img_height
        return round(x1), round(y1), round(x2), round(y2)

    def to_canvas_shape(self, img_width: int, img_height: int) -> dict:
        x1, y1, x2, y2 = self.to_xyxy(img_width, img_height)
        return {
            "type": ShapeType.BBOX,
            "class_id": self.class_id,
            "data": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        }

    @classmethod
    def from_xyxy(cls, class_id: int, x1: int, y1: int, x2: int, y2: int,
                  img_width: int, img_height: int) -> "BBoxAnnotation":
        if img_width <= 0 or img_height <= 0:
            raise ValueError(f"Invalid image dimensions: {img_width}x{img_height}")
        x_center = ((x1 + x2) / 2) / img_width
        y_center = ((y1 + y2) / 2) / img_height
        width = abs(x2 - x1) / img_width
        height = abs(y2 - y1) / img_height
        return cls(class_id, x_center, y_center, width, height)

    @classmethod
    def from_yolo(cls, line: str) -> "BBoxAnnotation":
        parts = line.strip().split()
        if len(parts) < 5:
            raise ValueError(f"Invalid bbox YOLO format: {line}")
        return cls(int(parts[0]), float(parts[1]), float(parts[2]),
                   float(parts[3]), float(parts[4]))


# ---------------------------------------------------------------------------
# Polygon annotation
# ---------------------------------------------------------------------------

class PolygonAnnotation(Annotation):
    """Polygon annotation with N vertices (normalized coordinates)."""

    def __init__(self, class_id: int, points: list[tuple[float, float]]):
        super().__init__(class_id, ShapeType.POLYGON)
        self.points = list(points)  # [(x_norm, y_norm), ...]

    def to_yolo(self) -> str:
        coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in self.points)
        return f"{self.class_id} {coords}"

    def to_canvas_shape(self, img_width: int, img_height: int) -> dict:
        pixel_points = [
            (round(x * img_width), round(y * img_height))
            for x, y in self.points
        ]
        return {
            "type": ShapeType.POLYGON,
            "class_id": self.class_id,
            "data": {"points": pixel_points},
        }

    @classmethod
    def from_pixel_points(cls, class_id: int, points: list[tuple[int, int]],
                          img_width: int, img_height: int) -> "PolygonAnnotation":
        if img_width <= 0 or img_height <= 0:
            raise ValueError(f"Invalid image dimensions: {img_width}x{img_height}")
        norm_points = [(x / img_width, y / img_height) for x, y in points]
        return cls(class_id, norm_points)


# ---------------------------------------------------------------------------
# Oriented bounding box (OBB) — 4 corner points
# ---------------------------------------------------------------------------

class OBBoxAnnotation(Annotation):
    """Oriented bounding box defined by 4 corner points (normalized)."""

    def __init__(self, class_id: int, corners: list[tuple[float, float]]):
        super().__init__(class_id, ShapeType.OBB)
        self.corners = list(corners)  # [(x, y), ...] 4 corners

    def to_yolo(self) -> str:
        coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in self.corners)
        return f"{self.class_id} {coords}"

    def to_canvas_shape(self, img_width: int, img_height: int) -> dict:
        pixel_corners = [
            (round(x * img_width), round(y * img_height))
            for x, y in self.corners
        ]
        return {
            "type": ShapeType.OBB,
            "class_id": self.class_id,
            "data": {"corners": pixel_corners},
        }

    @classmethod
    def from_pixel_corners(cls, class_id: int, corners: list[tuple[int, int]],
                           img_width: int, img_height: int) -> "OBBoxAnnotation":
        if img_width <= 0 or img_height <= 0:
            raise ValueError(f"Invalid image dimensions: {img_width}x{img_height}")
        norm_corners = [(x / img_width, y / img_height) for x, y in corners]
        return cls(class_id, norm_corners)


# ---------------------------------------------------------------------------
# Keypoint annotation
# ---------------------------------------------------------------------------

class KeypointAnnotation(Annotation):
    """Keypoint annotation with bounding box + keypoints."""

    VISIBILITY_HIDDEN = 0
    VISIBILITY_OCCLUDED = 1
    VISIBILITY_VISIBLE = 2

    def __init__(self, class_id: int, x_center: float, y_center: float,
                 width: float, height: float,
                 keypoints: list[tuple[float, float, int]] = None):
        super().__init__(class_id, ShapeType.KEYPOINT)
        self.x_center = x_center
        self.y_center = y_center
        self.width = width
        self.height = height
        self.keypoints = keypoints or []  # [(x_norm, y_norm, visibility), ...]

    def to_yolo(self) -> str:
        bbox_part = f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f}"
        kp_parts = " ".join(f"{x:.6f} {y:.6f} {v}" for x, y, v in self.keypoints)
        return f"{bbox_part} {kp_parts}" if self.keypoints else bbox_part

    def to_canvas_shape(self, img_width: int, img_height: int) -> dict:
        x1, y1, x2, y2 = self._bbox_xyxy(img_width, img_height)
        pixel_kps = [
            (round(x * img_width), round(y * img_height), v)
            for x, y, v in self.keypoints
        ]
        return {
            "type": ShapeType.KEYPOINT,
            "class_id": self.class_id,
            "data": {
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "keypoints": pixel_kps,
            },
        }

    def _bbox_xyxy(self, img_width: int, img_height: int) -> tuple:
        x1 = (self.x_center - self.width / 2) * img_width
        y1 = (self.y_center - self.height / 2) * img_height
        x2 = (self.x_center + self.width / 2) * img_width
        y2 = (self.y_center + self.height / 2) * img_height
        return round(x1), round(y1), round(x2), round(y2)

    @classmethod
    def from_pixel_data(cls, class_id: int, x1: int, y1: int, x2: int, y2: int,
                        img_width: int, img_height: int,
                        keypoints: list[tuple[int, int, int]] = None) -> "KeypointAnnotation":
        if img_width <= 0 or img_height <= 0:
            raise ValueError(f"Invalid image dimensions: {img_width}x{img_height}")
        xc = ((x1 + x2) / 2) / img_width
        yc = ((y1 + y2) / 2) / img_height
        w = abs(x2 - x1) / img_width
        h = abs(y2 - y1) / img_height
        norm_kps = []
        if keypoints:
            norm_kps = [(kx / img_width, ky / img_height, v) for kx, ky, v in keypoints]
        return cls(class_id, xc, yc, w, h, norm_kps)


# ---------------------------------------------------------------------------
# Annotation Manager
# ---------------------------------------------------------------------------

class AnnotationManager:
    """Manages annotations for a dataset — supports all shape types."""

    def __init__(self, classes: Optional[list[str]] = None):
        self.classes = classes or ["目标"]
        self.current_annotations: list[Annotation] = []
        self.current_image_path: Optional[str] = None
        self.current_label_path: Optional[str] = None
        self.is_modified = False

    # Backwards-compatible alias
    @property
    def current_boxes(self):
        return self.current_annotations

    def set_classes(self, classes: list[str]):
        self.classes = classes

    def load_annotation(self, image_path: str) -> list[Annotation]:
        """Load annotations for an image.

        Also loads classes.txt if found alongside the labels.
        """
        self.current_image_path = image_path
        label_path = self._get_label_path(image_path)
        self.current_label_path = label_path

        # Try to load classes.txt
        self._load_classes_from_label_path(label_path)

        self.current_annotations = []
        if os.path.exists(label_path):
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            ann = Annotation.from_yolo_line(line)
                            self.current_annotations.append(ann)
                        except ValueError as e:
                            logger.warning(f"Skipping invalid label in {label_path}: {e}")

        self.is_modified = False
        return self.current_annotations

    def _load_classes_from_label_path(self, label_path: str):
        """Try to load classes.txt from the dataset root.

        Looks in:
        1. <labels>/../classes.txt  (standard YOLO layout)
        2. <labels>/classes.txt     (fallback)
        """
        label_dir = os.path.dirname(label_path)
        candidates = []
        if os.path.basename(label_dir) == "labels":
            candidates.append(os.path.normpath(os.path.join(label_dir, "..", "classes.txt")))
        candidates.append(os.path.join(label_dir, "classes.txt"))

        for classes_path in candidates:
            if os.path.exists(classes_path):
                try:
                    with open(classes_path, "r", encoding="utf-8") as f:
                        loaded = [line.strip() for line in f if line.strip()]
                    if loaded and loaded != self.classes:
                        self.classes = loaded
                        logger.info(f"Loaded {len(loaded)} classes from {classes_path}")
                    return
                except Exception as e:
                    logger.debug(f"Failed to load classes from {classes_path}: {e}")

    def save_annotation(self, image_path: Optional[str] = None):
        """Save current annotations to file.

        Also writes classes.txt next to the label file so that training
        tools can find the class mapping.
        """
        if image_path:
            label_path = self._get_label_path(image_path)
        else:
            label_path = self.current_label_path

        if not label_path:
            raise ValueError("No label path specified")

        label_dir = os.path.dirname(label_path)
        os.makedirs(label_dir, exist_ok=True)

        # Save label .txt
        with open(label_path, "w", encoding="utf-8") as f:
            for ann in self.current_annotations:
                f.write(ann.to_yolo() + "\n")

        # Save classes.txt alongside labels
        classes_path = os.path.join(label_dir, "..", "classes.txt")
        # If labels are in a "labels" subdir, put classes.txt in dataset root
        if os.path.basename(label_dir) == "labels":
            classes_path = os.path.join(label_dir, "..", "classes.txt")
        else:
            classes_path = os.path.join(label_dir, "classes.txt")
        classes_path = os.path.normpath(classes_path)

        with open(classes_path, "w", encoding="utf-8") as f:
            for cls_name in self.classes:
                f.write(cls_name + "\n")

        self.is_modified = False
        logger.debug(f"Saved {len(self.current_annotations)} annotations to {label_path}")
        logger.debug(f"Saved {len(self.classes)} classes to {classes_path}")

    # -- Add methods for each type --

    def add_bbox(self, class_id: int, x1: int, y1: int, x2: int, y2: int,
                 img_width: int, img_height: int) -> BBoxAnnotation:
        ann = BBoxAnnotation.from_xyxy(class_id, x1, y1, x2, y2, img_width, img_height)
        self.current_annotations.append(ann)
        self.is_modified = True
        return ann

    def add_polygon(self, class_id: int, points: list[tuple[int, int]],
                    img_width: int, img_height: int) -> PolygonAnnotation:
        ann = PolygonAnnotation.from_pixel_points(class_id, points, img_width, img_height)
        self.current_annotations.append(ann)
        self.is_modified = True
        return ann

    def add_obb(self, class_id: int, corners: list[tuple[int, int]],
                img_width: int, img_height: int) -> OBBoxAnnotation:
        ann = OBBoxAnnotation.from_pixel_corners(class_id, corners, img_width, img_height)
        self.current_annotations.append(ann)
        self.is_modified = True
        return ann

    def add_keypoint(self, class_id: int, x1: int, y1: int, x2: int, y2: int,
                     img_width: int, img_height: int,
                     keypoints: list[tuple[int, int, int]] = None) -> KeypointAnnotation:
        ann = KeypointAnnotation.from_pixel_data(
            class_id, x1, y1, x2, y2, img_width, img_height, keypoints
        )
        self.current_annotations.append(ann)
        self.is_modified = True
        return ann

    # Generic add (for canvas integration)
    def add_annotation(self, ann: Annotation):
        self.current_annotations.append(ann)
        self.is_modified = True

    def remove_annotation(self, index: int):
        if 0 <= index < len(self.current_annotations):
            self.current_annotations.pop(index)
            self.is_modified = True

    # Backwards-compatible alias
    def remove_box(self, index: int):
        self.remove_annotation(index)

    def update_box_class(self, index: int, class_id: int):
        if 0 <= index < len(self.current_annotations):
            self.current_annotations[index].class_id = class_id
            self.is_modified = True

    def clear_annotations(self):
        self.current_annotations.clear()
        self.is_modified = True

    # Backwards-compatible alias
    def clear_boxes(self):
        self.clear_annotations()

    def get_annotation_count(self) -> int:
        return len(self.current_annotations)

    def draw_annotations(self, image: np.ndarray, annotations: Optional[list] = None,
                         selected_index: int = -1) -> np.ndarray:
        """Draw annotations on an image using OpenCV."""
        annotated = image.copy()
        annotations = annotations or self.current_annotations
        h, w = image.shape[:2]

        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 0, 0), (0, 128, 0), (0, 0, 128),
            (128, 128, 0), (128, 0, 128), (0, 128, 128),
        ]

        for i, ann in enumerate(annotations):
            color = colors[ann.class_id % len(colors)]
            class_name = self.classes[ann.class_id] if ann.class_id < len(self.classes) else str(ann.class_id)
            selected = (i == selected_index)

            if isinstance(ann, BBoxAnnotation):
                x1, y1, x2, y2 = ann.to_xyxy(w, h)
                if selected:
                    cv2.rectangle(annotated, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (0, 255, 255), 3)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                self._draw_label(annotated, class_name, x1, y1, color)

            elif isinstance(ann, PolygonAnnotation):
                pts = np.array([(round(px * w), round(py * h)) for px, py in ann.points], dtype=np.int32)
                if selected:
                    cv2.polylines(annotated, [pts], True, (0, 255, 255), 3)
                cv2.polylines(annotated, [pts], True, color, 2)
                if len(pts) > 0:
                    self._draw_label(annotated, class_name, pts[0][0], pts[0][1], color)

            elif isinstance(ann, OBBoxAnnotation):
                pts = np.array([(round(px * w), round(py * h)) for px, py in ann.corners], dtype=np.int32)
                if selected:
                    cv2.polylines(annotated, [pts], True, (0, 255, 255), 3)
                cv2.polylines(annotated, [pts], True, color, 2)
                if len(pts) > 0:
                    self._draw_label(annotated, class_name, pts[0][0], pts[0][1], color)

            elif isinstance(ann, KeypointAnnotation):
                x1, y1, x2, y2 = ann._bbox_xyxy(w, h)
                if selected:
                    cv2.rectangle(annotated, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (0, 255, 255), 3)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)
                self._draw_label(annotated, class_name, x1, y1, color)
                # Draw keypoints
                kp_colors = {0: (128, 128, 128), 1: (0, 255, 255), 2: (0, 255, 0)}
                for kx, ky, vis in ann.keypoints:
                    kpx, kpy = round(kx * w), round(ky * h)
                    kp_color = kp_colors.get(vis, (255, 255, 255))
                    cv2.circle(annotated, (kpx, kpy), 4, kp_color, -1)

        return annotated

    @staticmethod
    def _draw_label(image, label, x, y, color):
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(image, (x, y - th - 10), (x + tw, y), color, -1)
        cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    def _get_label_path(self, image_path: str) -> str:
        img_path = Path(image_path)
        parent = img_path.parent
        if parent.name == "images":
            label_dir = parent.parent / "labels"
        else:
            label_dir = parent / "labels"
        return str(label_dir / (img_path.stem + ".txt"))

    def get_annotation_stats(self, dataset_dir: str) -> dict:
        dataset_path = Path(dataset_dir)
        stats = {
            "total_images": 0,
            "annotated_images": 0,
            "total_annotations": 0,
            "class_distribution": {},
            "type_distribution": {},
        }

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

        for img_file in dataset_path.rglob("*"):
            if img_file.suffix.lower() in image_extensions:
                stats["total_images"] += 1
                label_file = self._get_label_path(str(img_file))
                if os.path.exists(label_file):
                    stats["annotated_images"] += 1
                    with open(label_file, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                try:
                                    ann = Annotation.from_yolo_line(line)
                                    stats["total_annotations"] += 1
                                    cls_name = self.classes[ann.class_id] if ann.class_id < len(self.classes) else str(ann.class_id)
                                    stats["class_distribution"][cls_name] = stats["class_distribution"].get(cls_name, 0) + 1
                                    type_name = ann.shape_type.value
                                    stats["type_distribution"][type_name] = stats["type_distribution"].get(type_name, 0) + 1
                                except ValueError:
                                    pass

        return stats

    @staticmethod
    def convert_voc_to_yolo(xml_path: str, img_width: int, img_height: int, class_map: dict) -> list[BBoxAnnotation]:
        """Convert Pascal VOC XML to YOLO format."""
        import xml.etree.ElementTree as ET

        tree = ET.parse(xml_path)
        root = tree.getroot()

        boxes = []
        for obj in root.findall("object"):
            class_name = obj.find("name").text
            if class_name not in class_map:
                continue

            class_id = class_map[class_name]
            bbox = obj.find("bndbox")
            x1 = int(bbox.find("xmin").text)
            y1 = int(bbox.find("ymin").text)
            x2 = int(bbox.find("xmax").text)
            y2 = int(bbox.find("ymax").text)

            boxes.append(BBoxAnnotation.from_xyxy(class_id, x1, y1, x2, y2, img_width, img_height))

        return boxes

    @staticmethod
    def convert_coco_to_yolo(coco_annotation: dict, img_width: int, img_height: int) -> BBoxAnnotation:
        """Convert COCO format annotation to YOLO format."""
        x, y, w, h = coco_annotation["bbox"]
        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height
        width = w / img_width
        height = h / img_height
        return BBoxAnnotation(coco_annotation["category_id"], x_center, y_center, width, height)
