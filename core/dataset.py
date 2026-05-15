"""Dataset management module — YOLO standard directory structure.

Follows the official Ultralytics YOLO format:
    dataset_root/
    ├── images/
    │   ├── train/       # training images
    │   ├── val/         # validation images
    │   └── test/        # optional test images
    ├── labels/
    │   ├── train/       # label .txt files (same stem as images)
    │   ├── val/
    │   └── test/
    └── data.yaml

data.yaml format:
    path: /absolute/path/to/dataset_root
    train: images/train
    val: images/val
    test: images/test       # optional
    nc: <num_classes>
    names:
        0: person
        1: bicycle
        ...

YOLO automatically replaces "images" with "labels" in the path to
find label files, so only image paths need to be specified.
"""

import os
import shutil
import random
from pathlib import Path
from typing import Optional

import yaml
from loguru import logger


class DatasetManager:
    """Manages YOLO datasets — creation, splitting, validation."""

    def __init__(self, dataset_root: str):
        self.root = Path(dataset_root)
        self.images_dir = self.root / "images"
        self.labels_dir = self.root / "labels"

    # ------------------------------------------------------------------
    # Build data.yaml — the bridge between annotation and training
    # ------------------------------------------------------------------

    @staticmethod
    def build_data_yaml(
        images_dir: str,
        labels_dir: Optional[str] = None,
        classes: Optional[list[str]] = None,
        output_yaml: Optional[str] = None,
        train_ratio: float = 0.8,
        val_ratio: float = 0.15,
        test_ratio: float = 0.05,
        kpt_shape: Optional[list[int]] = None,
        flip_idx: Optional[list[int]] = None,
    ) -> str:
        """Scan images (and labels), split train/val, and write data.yaml.

        This follows the **official Ultralytics YOLO dataset format**.
        When images are already under an ``images/`` directory tree, no
        files are copied — the data.yaml simply references the existing
        directory structure.  When images are in a flat folder (legacy),
        they are copied into the standard ``images/ + labels/`` tree.

        Args:
            images_dir:  Folder that contains images.
            labels_dir:  Folder that contains YOLO .txt labels.
                         If None, auto-detected (replace "images" → "labels").
            classes:     List of class names. If None, auto-detects from labels.
            output_yaml: Where to write data.yaml. Defaults to dataset_root/data.yaml.
            train_ratio: Fraction of data for training.
            val_ratio:   Fraction of data for validation.
            test_ratio:  Fraction of data for testing.
        Returns:
            Absolute path to the generated data.yaml.
        """
        images_path = Path(images_dir)

        # Auto-detect labels directory using YOLO convention
        if labels_dir:
            labels_path = Path(labels_dir)
        else:
            labels_path = DatasetManager._auto_labels_path(images_path)

        # Auto-detect annotation type and keypoint config from labels
        detected_kpt_shape, detected_flip_idx = DatasetManager._detect_kpt_config(images_path, labels_path)
        if kpt_shape is None:
            kpt_shape = detected_kpt_shape
        if flip_idx is None:
            flip_idx = detected_flip_idx

        # Store detected config for use by builder methods
        DatasetManager._detected_kpt_shape = kpt_shape
        DatasetManager._detected_flip_idx = flip_idx

        # Determine if images are already in YOLO-standard layout
        # (i.e. under an "images" directory with train/val subdirs)
        yolo_standard = DatasetManager._is_yolo_standard_layout(images_path)

        if yolo_standard:
            # Images are already in images/train, images/val, etc.
            # No need to copy — just write data.yaml referencing existing paths
            return DatasetManager._build_yaml_for_existing_layout(
                images_path=images_path,
                labels_path=labels_path,
                classes=classes,
                output_yaml=output_yaml,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
            )
        else:
            # Flat folder or non-standard layout — need to split & copy
            return DatasetManager._build_yaml_from_flat_folder(
                images_path=images_path,
                labels_path=labels_path,
                classes=classes,
                output_yaml=output_yaml,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
            )

    @staticmethod
    def _auto_labels_path(images_path: Path) -> Path:
        """Derive labels directory from images directory using YOLO convention.

        Replaces the nearest "images" segment with "labels".
        If no "images" segment exists, looks for a sibling "labels" directory.
        """
        parts = list(images_path.parts)
        for i in range(len(parts) - 1, -1, -1):
            if parts[i] == "images":
                label_parts = parts[:i] + ["labels"] + parts[i + 1:]
                return Path(*label_parts)
        # No "images" segment — try sibling "labels" dir
        sibling = images_path.parent / "labels"
        if sibling.exists():
            return sibling
        # Fallback: same directory (legacy)
        return images_path

    @staticmethod
    def _is_yolo_standard_layout(images_path: Path) -> bool:
        """Check if images are in a YOLO-standard directory tree.

        Returns True when the path contains an "images" segment, which
        means YOLO's automatic images→labels path substitution will work.
        This includes:
          - images/ with train/val subdirs (already split)
          - images/ with image files directly (not yet split)
          - images/train/ (inside a split subdir)
        """
        # images/ directory itself (with subdirs or direct files)
        if images_path.name == "images":
            return True

        # Inside images/train or similar
        if images_path.parent.name == "images":
            return True

        # Check any parent for "images" segment
        for part in images_path.parts:
            if part == "images":
                return True

        return False

    # Shared state for passing detected keypoint config to builder methods
    _detected_kpt_shape: Optional[list[int]] = None
    _detected_flip_idx: Optional[list[int]] = None

    @staticmethod
    def _detect_kpt_config(
        images_path: Path,
        labels_path: Path,
    ) -> tuple[Optional[list[int]], Optional[list[int]]]:
        """Auto-detect keypoint configuration from label files.

        Scans a sample of label files to determine:
        - Whether the dataset contains keypoint annotations
        - The number of keypoints per object (kpt_shape = [num_kpts, 3])

        Returns:
            (kpt_shape, flip_idx) where kpt_shape is [num_keypoints, dims] or None,
            and flip_idx is a list of symmetric keypoint indices or None.
        """
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        sample_labels: list[str] = []

        # Collect up to 20 label files to sample
        if labels_path.is_dir():
            for img_file in images_path.rglob("*") if images_path.is_dir() else []:
                if not img_file.is_file() or img_file.suffix.lower() not in image_extensions:
                    continue
                rel = img_file.relative_to(images_path)
                label_file = labels_path / rel.with_suffix(".txt")
                if label_file.exists():
                    with open(label_file, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                    if content:
                        sample_labels.append(content)
                if len(sample_labels) >= 20:
                    break

        if not sample_labels:
            return None, None

        # Parse lines to find keypoint annotations
        max_keypoints = 0
        for content in sample_labels:
            for line in content.splitlines():
                parts = line.strip().split()
                if len(parts) < 6:
                    continue
                n_values = len(parts) - 1  # exclude class_id
                # Keypoint format: cls xc yc w h kx1 ky1 v1 kx2 ky2 v2 ...
                # (n_values - 4) must be divisible by 3 and >= 3
                if n_values > 4 and (n_values - 4) % 3 == 0:
                    num_kpts = (n_values - 4) // 3
                    max_keypoints = max(max_keypoints, num_kpts)

        if max_keypoints == 0:
            return None, None

        # kpt_shape = [num_keypoints, 3]  (3 = x, y, visibility per keypoint)
        kpt_shape = [max_keypoints, 3]

        # Default flip_idx: identity mapping (no symmetric flipping)
        # For COCO pose (17 keypoints), the standard flip_idx would be different,
        # but for custom datasets we use identity mapping as a safe default.
        flip_idx = list(range(max_keypoints))

        logger.info(f"Detected keypoint annotations: kpt_shape={kpt_shape}")
        return kpt_shape, flip_idx

    @staticmethod
    def _build_yaml_for_existing_layout(
        images_path: Path,
        labels_path: Path,
        classes: Optional[list[str]],
        output_yaml: Optional[str],
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
    ) -> str:
        """Build data.yaml for an already YOLO-standard directory tree.

        If train/val subdirectories exist, reference them directly.
        If all images are in a flat images/ dir, split them into train/val.
        """
        # Determine the dataset root (parent of "images")
        if images_path.name == "images":
            dataset_root = images_path.parent
            image_subdirs = sorted([
                d.name for d in images_path.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            ])
        else:
            # We're inside images/train or similar
            dataset_root = images_path.parent.parent
            image_subdirs = [images_path.name]

        # Auto-detect classes if not provided
        if classes is None:
            classes = DatasetManager._detect_classes_from_labels(
                images_path, labels_path
            )

        # Determine train/val/test paths
        data_yaml_dict: dict = {
            "path": str(dataset_root.absolute()),
            "names": {i: name for i, name in enumerate(classes)},
            "nc": len(classes),
        }

        # Add keypoint config if detected (required for pose training)
        kpt = DatasetManager._detected_kpt_shape
        fidx = DatasetManager._detected_flip_idx
        if kpt is not None:
            data_yaml_dict["kpt_shape"] = kpt
        if fidx is not None:
            data_yaml_dict["flip_idx"] = fidx

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

        if "train" in image_subdirs and "val" in image_subdirs:
            # Already split — just reference
            data_yaml_dict["train"] = "images/train"
            data_yaml_dict["val"] = "images/val"
            if "test" in image_subdirs:
                data_yaml_dict["test"] = "images/test"
        else:
            # No train/val split yet — check if there are image files
            # directly under images/ or in existing subdirs
            has_images = any(
                f.is_file() and f.suffix.lower() in image_extensions
                for f in images_path.iterdir()
            ) if images_path.is_dir() else False

            if has_images or len(image_subdirs) > 0:
                # Split images into train/val subdirs
                DatasetManager._split_into_train_val(
                    dataset_root=dataset_root,
                    images_path=images_path,
                    labels_path=labels_path,
                    train_ratio=train_ratio,
                    val_ratio=val_ratio,
                    test_ratio=test_ratio,
                )
                data_yaml_dict["train"] = "images/train"
                data_yaml_dict["val"] = "images/val"
                if test_ratio > 0:
                    data_yaml_dict["test"] = "images/test"
            else:
                raise FileNotFoundError(f"No images found in {images_path}")

        # Write data.yaml
        yaml_path = Path(output_yaml) if output_yaml else dataset_root / "data.yaml"
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data_yaml_dict, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"Built data.yaml for YOLO-standard layout: {yaml_path}")
        return str(yaml_path)

    @staticmethod
    def _build_yaml_from_flat_folder(
        images_path: Path,
        labels_path: Path,
        classes: Optional[list[str]],
        output_yaml: Optional[str],
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
    ) -> str:
        """Build data.yaml from a flat folder of images (legacy layout).

        Copies images and labels into the standard YOLO directory structure:
            dataset_root/
            ├── images/train/  & images/val/
            ├── labels/train/  & labels/val/
            └── data.yaml
        """
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        image_files = sorted([
            f for f in images_path.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ])
        if not image_files:
            raise FileNotFoundError(f"No images found in {images_path}")

        # Auto-detect classes if not provided
        if classes is None:
            classes = DatasetManager._detect_classes_from_labels(
                images_path, labels_path
            )

        # Shuffle and split
        random.shuffle(image_files)
        total = len(image_files)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        if test_ratio > 0:
            splits = {
                "train": image_files[:train_end],
                "val": image_files[train_end:val_end],
                "test": image_files[val_end:],
            }
        else:
            # No test split — put all remaining images into val
            splits = {
                "train": image_files[:train_end],
                "val": image_files[train_end:],
            }

        # Create dataset directory structure
        dataset_dir = images_path.parent / (images_path.name + "_dataset")
        for split_name in splits:
            (dataset_dir / "images" / split_name).mkdir(parents=True, exist_ok=True)
            (dataset_dir / "labels" / split_name).mkdir(parents=True, exist_ok=True)

        for split_name, files in splits.items():
            img_dir = dataset_dir / "images" / split_name
            lbl_dir = dataset_dir / "labels" / split_name

            for img_file in files:
                shutil.copy2(img_file, img_dir / img_file.name)
                label_file = labels_path / (img_file.stem + ".txt")
                if label_file.exists():
                    shutil.copy2(label_file, lbl_dir / label_file.name)

        # Write data.yaml
        data_yaml_dict = {
            "path": str(dataset_dir.absolute()),
            "train": "images/train",
            "val": "images/val",
            "names": {i: name for i, name in enumerate(classes)},
            "nc": len(classes),
        }
        if test_ratio > 0:
            data_yaml_dict["test"] = "images/test"

        # Add keypoint config if detected (required for pose training)
        kpt = DatasetManager._detected_kpt_shape
        fidx = DatasetManager._detected_flip_idx
        if kpt is not None:
            data_yaml_dict["kpt_shape"] = kpt
        if fidx is not None:
            data_yaml_dict["flip_idx"] = fidx

        yaml_path = Path(output_yaml) if output_yaml else dataset_dir / "data.yaml"
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data_yaml_dict, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"Built dataset from flat folder: {total} images → {dataset_dir}")
        return str(yaml_path)

    @staticmethod
    def _detect_classes_from_labels(
        images_path: Path,
        labels_path: Path,
    ) -> list[str]:
        """Auto-detect class names from label files.

        Returns a list of class names indexed by class_id.
        """
        class_set: set[int] = set()
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

        # Collect all image files (may be in subdirectories)
        if images_path.is_dir():
            image_files = [
                f for f in images_path.rglob("*")
                if f.is_file() and f.suffix.lower() in image_extensions
            ]
        else:
            image_files = []

        for img in image_files:
            # Compute label path using YOLO convention
            rel = img.relative_to(images_path)
            label_file = labels_path / rel.with_suffix(".txt")
            if label_file.exists():
                with open(label_file, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            try:
                                class_set.add(int(parts[0]))
                            except ValueError:
                                pass

        if not class_set:
            return ["目标"]
        return [f"类别_{i}" for i in range(max(class_set) + 1)]

    @staticmethod
    def _split_into_train_val(
        dataset_root: Path,
        images_path: Path,
        labels_path: Path,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
    ) -> None:
        """Split images in a flat images/ directory into train/val/test subdirs.

        This moves files from:
            images/*.jpg → images/train/*.jpg, images/val/*.jpg, …
            images/subdir/*.jpg → images/train/*.jpg, images/val/*.jpg, …
            labels/*.txt → labels/train/*.txt, labels/val/*.txt, …
        """
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

        # Collect images (recursively from all subdirs, excluding train/val/test)
        excluded_dirs = {"train", "val", "test"}
        image_files = sorted([
            f for f in images_path.rglob("*")
            if f.is_file()
            and f.suffix.lower() in image_extensions
            and not any(part in excluded_dirs for part in f.relative_to(images_path).parts[:-1])
        ])
        if not image_files:
            logger.warning(f"No images found in {images_path} for splitting")
            return

        random.shuffle(image_files)
        total = len(image_files)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        if test_ratio > 0:
            splits = {
                "train": image_files[:train_end],
                "val": image_files[train_end:val_end],
                "test": image_files[val_end:],
            }
        else:
            # No test split — put all remaining images into val
            # to avoid losing images due to integer truncation
            splits = {
                "train": image_files[:train_end],
                "val": image_files[train_end:],
            }

        for split_name, files in splits.items():
            img_dir = images_path / split_name
            lbl_dir = labels_path / split_name
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)

            for img_file in files:
                # Move image
                dest_img = img_dir / img_file.name
                shutil.move(str(img_file), str(dest_img))
                # Move label - search recursively in labels_path
                label_candidates = list(labels_path.rglob(img_file.stem + ".txt"))
                if label_candidates:
                    label_file = label_candidates[0]  # Take first match
                    dest_lbl = lbl_dir / label_file.name
                    shutil.move(str(label_file), str(dest_lbl))

        logger.info(f"Split {total} images into train/val/test under {images_path}")

    # ------------------------------------------------------------------
    # Create new dataset
    # ------------------------------------------------------------------

    def create_yolo_dataset(
        self,
        name: str,
        classes: list[str],
        train_ratio: float = 0.8,
        val_ratio: float = 0.15,
        test_ratio: float = 0.05,
    ) -> str:
        """Create a new YOLO dataset with standard directory structure.

        Creates:
            dataset_root/name/
            ├── images/train/
            ├── images/val/
            ├── images/test/       (if test_ratio > 0)
            ├── labels/train/
            ├── labels/val/
            ├── labels/test/
            └── data.yaml
        """
        dataset_path = self.root / name
        for split in ("train", "val"):
            (dataset_path / "images" / split).mkdir(parents=True, exist_ok=True)
            (dataset_path / "labels" / split).mkdir(parents=True, exist_ok=True)
        if test_ratio > 0:
            (dataset_path / "images" / "test").mkdir(parents=True, exist_ok=True)
            (dataset_path / "labels" / "test").mkdir(parents=True, exist_ok=True)

        # Create data.yaml
        data_yaml = {
            "path": str(dataset_path.absolute()),
            "train": "images/train",
            "val": "images/val",
            "names": {i: name for i, name in enumerate(classes)},
            "nc": len(classes),
        }
        if test_ratio > 0:
            data_yaml["test"] = "images/test"

        yaml_path = dataset_path / "data.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"Created dataset at {dataset_path}")
        return str(dataset_path)

    # ------------------------------------------------------------------
    # Split dataset
    # ------------------------------------------------------------------

    def split_dataset(
        self,
        images_dir: str,
        labels_dir: str,
        output_dir: str,
        classes: list[str],
        train_ratio: float = 0.8,
        val_ratio: float = 0.15,
        test_ratio: float = 0.05,
    ) -> str:
        """Split a dataset into train/val/test sets with YOLO standard layout.

        Output structure:
            output_dir/
            ├── images/train/  & images/val/  & images/test/
            ├── labels/train/  & labels/val/  & labels/test/
            └── data.yaml
        """
        images_path = Path(images_dir)
        labels_path = Path(labels_dir)
        output_path = Path(output_dir)

        # Get all image files
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        image_files = [
            f for f in images_path.iterdir()
            if f.suffix.lower() in image_extensions
        ]
        random.shuffle(image_files)

        total = len(image_files)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        if test_ratio > 0:
            splits: dict[str, list] = {
                "train": image_files[:train_end],
                "val": image_files[train_end:val_end],
                "test": image_files[val_end:],
            }
        else:
            splits: dict[str, list] = {
                "train": image_files[:train_end],
                "val": image_files[train_end:],
            }

        for split_name, files in splits.items():
            img_dir = output_path / "images" / split_name
            lbl_dir = output_path / "labels" / split_name
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)

            for img_file in files:
                shutil.copy2(img_file, img_dir / img_file.name)
                label_file = labels_path / (img_file.stem + ".txt")
                if label_file.exists():
                    shutil.copy2(label_file, lbl_dir / label_file.name)

        # Create data.yaml
        data_yaml = {
            "path": str(output_path.absolute()),
            "train": "images/train",
            "val": "images/val",
            "names": {i: name for i, name in enumerate(classes)},
            "nc": len(classes),
        }
        if test_ratio > 0:
            data_yaml["test"] = "images/test"

        yaml_path = output_path / "data.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"Dataset split complete: {total} images -> {output_path}")
        return str(output_path)

    # ------------------------------------------------------------------
    # Dataset info & validation
    # ------------------------------------------------------------------

    def get_dataset_info(self, dataset_path: str) -> dict:
        """Get information about a dataset."""
        path = Path(dataset_path)
        yaml_path = path / "data.yaml"

        info: dict = {"path": str(path), "splits": {}}

        if yaml_path.exists():
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            info["classes"] = data.get("names", {})
            info["nc"] = data.get("nc", 0)

        for split in ["train", "val", "test"]:
            # YOLO standard layout: images/train, labels/train
            split_img_dir = path / "images" / split
            split_lbl_dir = path / "labels" / split
            # Legacy layout: train/images, train/labels
            if not split_img_dir.exists():
                split_img_dir = path / split / "images"
                split_lbl_dir = path / split / "labels"

            if split_img_dir.exists():
                images = list(split_img_dir.iterdir())
                labels = list(split_lbl_dir.iterdir()) if split_lbl_dir.exists() else []
                info["splits"][split] = {
                    "images": len(images),
                    "labels": len(labels),
                }

        return info

    def validate_dataset(self, dataset_path: str) -> list[str]:
        """Validate a YOLO dataset for common issues."""
        path = Path(dataset_path)
        issues: list[str] = []

        # Check data.yaml exists
        yaml_path = path / "data.yaml"
        if not yaml_path.exists():
            issues.append("data.yaml not found")
            return issues

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        nc = data.get("nc", 0)
        names = data.get("names", {})

        if nc != len(names):
            issues.append(f"nc ({nc}) != len(names) ({len(names)})")

        split_members: dict[str, list[str]] = {}

        for split in ["train", "val", "test"]:
            # YOLO standard layout
            img_dir = path / "images" / split
            lbl_dir = path / "labels" / split
            # Legacy layout fallback
            if not img_dir.exists():
                img_dir = path / split / "images"
                lbl_dir = path / split / "labels"

            if not img_dir.exists():
                continue

            image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
            images = [f for f in img_dir.iterdir() if f.suffix.lower() in image_extensions]
            labels = {f.stem: f for f in lbl_dir.iterdir() if f.suffix == ".txt"} if lbl_dir.exists() else {}

            # Check for missing labels
            missing_labels = 0
            invalid_labels = 0
            for img in images:
                split_members.setdefault(img.stem, []).append(split)
                if img.stem not in labels:
                    missing_labels += 1
                else:
                    # Validate label format
                    with open(labels[img.stem], "r", encoding="utf-8") as f:
                        for line_num, line in enumerate(f, 1):
                            parts = line.strip().split()
                            if len(parts) < 5:
                                invalid_labels += 1
                            elif int(parts[0]) >= nc:
                                issues.append(f"{split}/{img.stem}.txt line {line_num}: class_id >= nc")

            if missing_labels > 0:
                issues.append(f"{split}: {missing_labels} images without labels")
            if invalid_labels > 0:
                issues.append(f"{split}: {invalid_labels} invalid label lines")

        overlaps = {
            stem: splits
            for stem, splits in split_members.items()
            if len(splits) > 1
        }
        if overlaps:
            examples = ", ".join(
                f"{stem} ({'/'.join(splits)})"
                for stem, splits in list(overlaps.items())[:5]
            )
            suffix = "" if len(overlaps) <= 5 else f", ... +{len(overlaps) - 5} more"
            issues.append(f"{len(overlaps)} images appear in multiple splits: {examples}{suffix}")

        return issues
