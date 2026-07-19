"""Annotation utility functions shared across the codebase."""

from collections import Counter
from pathlib import Path
from typing import Counter as CounterType
from typing import List, Optional

from loguru import logger


def find_label_file(
    image_file: Path,
    annotation_dir: str | Path,
    image_dir: Optional[str | Path] = None,
) -> Optional[Path]:
    """Resolve the YOLO label path for an image.

    Tries, in order:
    1. Official Ultralytics layout (``.../images/...`` → ``.../labels/...``)
    2. Relative mirror under *annotation_dir* (``train/a.jpg`` → ``labels/train/a.txt``)
    3. Flat ``annotation_dir/<stem>.txt``
    4. Label beside the image (legacy)
    """
    image_file = Path(image_file)
    annotation_dir = Path(annotation_dir)
    candidates: list[Path] = []

    parts = list(image_file.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            candidates.append(Path(*parts[:i], "labels", *parts[i + 1:]).with_suffix(".txt"))
            break

    if image_dir is not None:
        try:
            rel = image_file.relative_to(Path(image_dir))
            candidates.append(annotation_dir / rel.with_suffix(".txt"))
        except ValueError:
            pass

    candidates.append(annotation_dir / f"{image_file.stem}.txt")
    candidates.append(image_file.with_suffix(".txt"))

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            return candidate
    return None


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
    image_root = Path(image_dir)
    image_files = [
        f for f in image_root.rglob("*")
        if f.is_file() and f.suffix.lower() in image_extensions
    ]

    total_images = len(image_files)
    total_annotations = 0
    annotated_images = 0
    class_counts: CounterType[str] = Counter()
    annotation_sizes: List[int] = []
    missing_annotations: List[str] = []

    for image_file in image_files:
        ann_file = find_label_file(image_file, annotation_dir, image_root)

        if ann_file is None:
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
                            # harmless: non-integer class ID, skip line
                            pass

        except Exception as e:
            logger.error(f"Error reading {ann_file}: {e}")

    return (total_images, total_annotations, annotated_images,
            class_counts, annotation_sizes, missing_annotations)
