"""Annotation utility functions shared across the codebase."""

from collections import Counter
from pathlib import Path
from typing import Counter as CounterType
from typing import List

from loguru import logger


def collect_annotation_stats(image_dir: str, annotation_dir: str,
                             class_names: List[str]):
    """Iterate over images, read YOLO-format label files, and return raw
    annotation statistics.

    This is the shared core used by both ``AnnotationStatisticsCollector``
    and ``DataQualityChecker``.  Each caller wraps the returned raw values
    into its own dataclass.

    Args:
        image_dir: Directory containing image files.
        annotation_dir: Directory containing ``.txt`` label files.
        class_names: Ordered list of class names (index = class id).

    Returns:
        Tuple of ``(total_images, total_annotations, annotated_images,
        class_counts, annotation_sizes, missing_annotations)``.

        * **total_images** – number of image files found.
        * **total_annotations** – sum of all bounding-box lines across
          all label files.
        * **annotated_images** – number of images that actually have a
          corresponding label file.
        * **class_counts** – ``Counter`` mapping class name to count.
        * **annotation_sizes** – list where each element is the number of
          annotations in one label file (only for annotated images).
        * **missing_annotations** – list of image *file names* that have
          no matching label file.
    """
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    image_files = [
        f for f in Path(image_dir).rglob("*")
        if f.is_file() and f.suffix.lower() in image_extensions
    ]

    total_images = len(image_files)
    total_annotations = 0
    annotated_images = 0
    class_counts: CounterType[str] = Counter()
    annotation_sizes: List[int] = []
    missing_annotations: List[str] = []

    for image_file in image_files:
        ann_file = Path(annotation_dir) / image_file.stem
        ann_file = ann_file.with_suffix(".txt")

        if not ann_file.exists():
            missing_annotations.append(image_file.name)
            continue

        annotated_images += 1

        try:
            with open(ann_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                total_annotations += len(lines)
                annotation_sizes.append(len(lines))

                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 1:
                        try:
                            class_id = int(parts[0])
                            if class_id < len(class_names):
                                class_counts[class_names[class_id]] += 1
                        except ValueError:
                            pass

        except Exception as e:
            logger.error(f"Error reading {ann_file}: {e}")

    return (total_images, total_annotations, annotated_images,
            class_counts, annotation_sizes, missing_annotations)
