"""YOLO label file IO for the annotation canvas.

Supports the official Ultralytics YOLO directory structure:
    dataset_root/
    ├── images/
    │   ├── train/       # training images
    │   ├── val/         # validation images
    │   └── ...
    ├── labels/
    │   ├── train/       # label .txt files (same stem as images)
    │   ├── val/
    │   └── ...
    └── data.yaml

Key rule:  YOLO automatically replaces "images" with "labels" in the path
           to find the corresponding label file.

Fallback:  If the label file does not exist under the labels/ tree but
           *does* exist beside the image (legacy "same-dir" layout), that
           path is returned instead, preserving backward compatibility.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from core.annotation import Annotation, ShapeType


# ------------------------------------------------------------------
# Path helpers
# ------------------------------------------------------------------

def label_path_for_image(image_path: str, labels_root: Optional[str] = None) -> str:
    """Return the YOLO label .txt path for a given image.

    Resolution order:
    1. If *labels_root* is given, use ``labels_root / <stem>.txt``.
    2. Walk up from *image_path*: if a parent directory is named
       ``images``, replace it with ``labels`` (YOLO convention).
    3. Legacy fallback: label sits beside the image (same directory,
       same stem, .txt suffix).

    Args:
        image_path: Absolute or relative path to an image file.
        labels_root: Optional explicit labels directory root.

    Returns:
        The resolved label file path (may or may not exist on disk).
    """
    img = Path(image_path)

    # 1. Explicit labels root
    if labels_root:
        return str(Path(labels_root) / (img.stem + ".txt"))

    # 2. YOLO standard: replace the nearest "images" dir with "labels"
    #    e.g. .../dataset/images/train/001.jpg → .../dataset/labels/train/001.txt
    #    or   .../dataset/images/001.jpg       → .../dataset/labels/001.txt
    parts = list(img.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            label_parts = parts[:i] + ["labels"] + parts[i + 1:]
            label_path = Path(*label_parts).with_suffix(".txt")
            return str(label_path)

    # 3. Legacy: label beside image
    return str(img.with_suffix(".txt"))


def labels_dir_for_image_dir(image_dir: str) -> str:
    """Return the corresponding labels directory for an images directory.

    Examples:
        .../images/train  → .../labels/train
        .../images        → .../labels
        .../my_photos     → .../my_photos  (no "images" segment → same dir)

    This follows the Ultralytics YOLO convention where the "images"
    directory name is swapped for "labels".
    """
    p = Path(image_dir)
    parts = list(p.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            label_parts = parts[:i] + ["labels"] + parts[i + 1:]
            return str(Path(*label_parts))
    # No "images" segment — fall back to same directory (legacy)
    return image_dir


def resolve_labels_root(image_path: str, image_dir: Optional[str] = None) -> str:
    """Return the label file path using YOLO-standard labels/ resolution.

    This always tries the YOLO-standard path first (replacing "images" →
    "labels" in the parent chain). If that label file doesn't exist but
    a legacy same-directory label does, the legacy path is returned.

    Args:
        image_path: Path to the image file.
        image_dir:  The image directory (used for context, optional).

    Returns:
        Best label file path for this image.
    """
    # Try YOLO standard path first
    yolo_path = label_path_for_image(image_path)
    legacy_path = str(Path(image_path).with_suffix(".txt"))

    if yolo_path != legacy_path:
        # There is a distinct YOLO labels/ path — prefer it if it exists
        if os.path.exists(yolo_path):
            return yolo_path
        # YOLO path doesn't exist yet, but legacy does — still use YOLO
        # path for new saves (we'll migrate). But for reading, fall back.
        if os.path.exists(legacy_path):
            return legacy_path
        # Neither exists — return YOLO path for future writes
        return yolo_path

    # No "images" segment in path — legacy same-dir style
    return legacy_path


# ------------------------------------------------------------------
# Shape type helper
# ------------------------------------------------------------------

def shape_type_value(shape_type: Any) -> str:
    if isinstance(shape_type, ShapeType):
        return shape_type.value
    return str(shape_type)


# ------------------------------------------------------------------
# Load / Save
# ------------------------------------------------------------------

def load_yolo_shapes(
    image_path: str,
    image_width: int,
    image_height: int,
    class_manager,
    labels_root: Optional[str] = None,
) -> list[dict]:
    """Load a YOLO txt file and return canvas-ready shape dicts.

    Args:
        image_path:   Path to the image file.
        image_width:  Image width in pixels.
        image_height: Image height in pixels.
        class_manager: ClassManager instance for name lookups.
        labels_root:  Optional explicit labels directory root.
    """
    label_path = label_path_for_image(image_path, labels_root=labels_root)

    # Fallback: if YOLO path doesn't exist, try legacy same-dir path
    if not os.path.exists(label_path):
        legacy_path = str(Path(image_path).with_suffix(".txt"))
        if os.path.exists(legacy_path):
            label_path = legacy_path
        else:
            return []

    shapes: list[dict] = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ann = Annotation.from_yolo_line(line)
            shape = ann.to_canvas_shape(image_width, image_height)
            class_name = class_manager.get_class_name(shape["class_id"])
            shape["class_name"] = class_name or f"class_{shape['class_id']}"
            shapes.append(shape)
    return shapes


def save_yolo_shapes(
    image_path: str,
    shapes: list[dict],
    image_width: int,
    image_height: int,
    labels_root: Optional[str] = None,
) -> str:
    """Save canvas shape dicts to YOLO txt format.

    Labels are always written to the YOLO-standard path (replacing
    "images" → "labels" in the directory tree). If the labels
    directory doesn't exist it is created automatically.

    For images not under an ``images/`` directory (legacy layout), the
    label file is placed beside the image as before.

    Args:
        image_path:   Path to the image file.
        shapes:       List of canvas shape dicts.
        image_width:  Image width in pixels.
        image_height: Image height in pixels.
        labels_root:  Optional explicit labels directory root.

    Returns:
        Path to the written label file.
    """
    label_path = label_path_for_image(image_path, labels_root=labels_root)

    # Ensure the labels directory exists
    label_dir = os.path.dirname(label_path)
    if label_dir:
        os.makedirs(label_dir, exist_ok=True)

    lines = [_shape_to_yolo_line(shape, image_width, image_height) for shape in shapes]
    with open(label_path, "w", encoding="utf-8") as f:
        for line in lines:
            if line:
                f.write(line + "\n")
    return label_path


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _shape_to_yolo_line(shape: dict, image_width: int, image_height: int) -> str:
    stype = shape_type_value(shape.get("type", "bbox"))
    class_id = int(shape.get("class_id", 0))
    data = shape.get("data", {})

    if stype == ShapeType.BBOX.value:
        x1, y1, x2, y2 = _bbox_data_to_xyxy(data)
        xc = ((x1 + x2) / 2) / image_width
        yc = ((y1 + y2) / 2) / image_height
        w = abs(x2 - x1) / image_width
        h = abs(y2 - y1) / image_height
        return f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"

    if stype == ShapeType.POLYGON.value:
        points = data.get("points", []) if isinstance(data, dict) else data
        coords = []
        for x, y in points:
            coords.extend([x / image_width, y / image_height])
        return _format_line(class_id, coords)

    if stype == ShapeType.OBB.value:
        corners = data.get("corners", []) if isinstance(data, dict) else data
        coords = []
        for x, y in corners:
            coords.extend([x / image_width, y / image_height])
        return _format_line(class_id, coords)

    if stype == ShapeType.KEYPOINT.value:
        x1, y1, x2, y2 = _bbox_data_to_xyxy(data)
        xc = ((x1 + x2) / 2) / image_width
        yc = ((y1 + y2) / 2) / image_height
        w = abs(x2 - x1) / image_width
        h = abs(y2 - y1) / image_height
        coords = [xc, yc, w, h]
        for kx, ky, vis in data.get("keypoints", []):
            coords.extend([kx / image_width, ky / image_height, int(vis)])
        return _format_line(class_id, coords)

    return ""


def _bbox_data_to_xyxy(data: Any) -> tuple[float, float, float, float]:
    if isinstance(data, dict):
        return (
            float(data.get("x1", 0)),
            float(data.get("y1", 0)),
            float(data.get("x2", 0)),
            float(data.get("y2", 0)),
        )
    if isinstance(data, (list, tuple)) and len(data) >= 4:
        return float(data[0]), float(data[1]), float(data[2]), float(data[3])
    return 0.0, 0.0, 0.0, 0.0


def _format_line(class_id: int, values: list[float]) -> str:
    return f"{class_id} " + " ".join(
        str(int(v)) if isinstance(v, int) else f"{float(v):.6f}"
        for v in values
    )
