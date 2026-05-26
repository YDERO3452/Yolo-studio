"""Workflow optimization module for batch processing and quality checks."""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import Counter
from loguru import logger

import numpy as np


@dataclass
class QualityMetrics:
    """Data quality metrics."""
    total_images: int
    total_annotations: int
    avg_annotations_per_image: float
    class_distribution: Dict[str, int]
    annotation_size_stats: Dict[str, Any]
    missing_annotations: List[str]
    duplicate_annotations: List[str]


class WorkflowBatchProcessor:
    """Batch processing for annotations."""

    def __init__(self):
        """Initialize BatchProcessor."""
        logger.info("BatchProcessor initialized")

    def batch_export(
        self,
        image_dir: str,
        annotation_dir: str,
        output_dir: str,
        output_format: str,
        class_names: List[str],
        input_format: str = "yolo",
    ) -> Dict[str, Any]:
        """
        Batch export annotations to different format.

        Args:
            image_dir: Directory containing images
            annotation_dir: Directory containing annotations
            output_dir: Output directory
            output_format: Output format (yolo, voc, coco, dota)
            class_names: List of class names
            input_format: Input format (default: yolo)

        Returns:
            Dictionary with export results
        """
        from core.format_converter import FormatConverter

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        converter = FormatConverter(class_names)

        results = {
            "total_files": 0,
            "successful": 0,
            "failed": 0,
            "errors": [],
        }

        conversion_results = converter.convert_folder(
            input_dir=annotation_dir,
            output_dir=output_dir,
            input_format=input_format,
            output_format=output_format,
            image_dir=image_dir,
            progress_callback=lambda cur, total: logger.info(
                f"Exporting {cur + 1}/{total}"
            ),
        )

        results["total_files"] = len(conversion_results)

        for cr in conversion_results:
            if cr.success:
                results["successful"] += 1
            else:
                results["failed"] += 1
                if cr.error_message:
                    results["errors"].append(f"{cr.input_file}: {cr.error_message}")

        return results

    def batch_auto_label(
        self,
        image_dir: str,
        output_dir: str,
        model_manager,
        conf: float = 0.25,
        iou: float = 0.45,
    ) -> Dict[str, Any]:
        """
        Batch auto-label images.

        Args:
            image_dir: Directory containing images
            output_dir: Output directory
            model_manager: ModelManager instance
            conf: Confidence threshold
            iou: IOU threshold

        Returns:
            Dictionary with labeling results
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        results = {
            "total_images": 0,
            "successful": 0,
            "failed": 0,
            "total_detections": 0,
            "errors": [],
        }

        # Get all images
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        image_files = [
            f for f in Path(image_dir).iterdir()
            if f.suffix.lower() in image_extensions
        ]
        results["total_images"] = len(image_files)

        for i, image_file in enumerate(image_files):
            try:
                logger.info(f"Auto-labeling {i + 1}/{len(image_files)}: {image_file.name}")

                # Run inference
                detections = model_manager.predict(str(image_file), conf, iou)

                if detections:
                    # Save annotations
                    output_file = Path(output_dir) / image_file.stem
                    output_file = output_file.with_suffix(".txt")

                    with open(output_file, "w", encoding="utf-8") as f:
                        for det in detections:
                            bbox = det.get("bbox", [0, 0, 0, 0])
                            line = f"{det['class_id']} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n"
                            f.write(line)

                    results["successful"] += 1
                    results["total_detections"] += len(detections)
                else:
                    results["failed"] += 1

            except Exception as e:
                results["failed"] += 1
                results["errors"].append(f"Error processing {image_file.name}: {e}")
                logger.error(f"Failed to process {image_file.name}: {e}")

        return results



class AnnotationValidator:
    """Validate annotations for quality and completeness."""

    def __init__(self):
        """Initialize AnnotationValidator."""
        logger.info("AnnotationValidator initialized")

    def validate_folder(
        self,
        image_dir: str,
        annotation_dir: str,
        class_names: List[str],
    ) -> Dict[str, Any]:
        """
        Validate all annotations in a folder.

        Args:
            image_dir: Directory containing images
            annotation_dir: Directory containing annotations
            class_names: List of class names

        Returns:
            Validation report
        """
        report = {
            "total_images": 0,
            "annotated_images": 0,
            "missing_annotations": [],
            "invalid_annotations": [],
            "warnings": [],
        }

        # Get all images
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        image_files = [
            f for f in Path(image_dir).iterdir()
            if f.suffix.lower() in image_extensions
        ]
        report["total_images"] = len(image_files)

        for image_file in image_files:
            ann_file = Path(annotation_dir) / image_file.stem
            ann_file = ann_file.with_suffix(".txt")

            if not ann_file.exists():
                report["missing_annotations"].append(image_file.name)
            else:
                report["annotated_images"] += 1

                # Validate annotation file
                try:
                    with open(ann_file, "r", encoding="utf-8") as f:
                        for line_num, line in enumerate(f, 1):
                            line = line.strip()
                            if not line:
                                continue

                            parts = line.split()
                            if len(parts) < 5:
                                report["invalid_annotations"].append(
                                    f"{ann_file.name}:{line_num} - Invalid format"
                                )
                                continue

                            # Check class ID
                            try:
                                class_id = int(parts[0])
                                if class_id >= len(class_names):
                                    report["warnings"].append(
                                        f"{ann_file.name}:{line_num} - Invalid class ID: {class_id}"
                                    )
                            except ValueError:
                                report["invalid_annotations"].append(
                                    f"{ann_file.name}:{line_num} - Invalid class ID"
                                )

                except Exception as e:
                    report["invalid_annotations"].append(f"{ann_file.name} - {e}")

        return report

    def validate_annotation(
        self,
        annotation_text: str,
        image_width: int,
        image_height: int,
        class_names: List[str],
    ) -> Dict[str, Any]:
        """
        Validate a single annotation.

        Args:
            annotation_text: Annotation text (YOLO format)
            image_width: Image width
            image_height: Image height
            class_names: List of class names

        Returns:
            Validation result
        """
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
        }

        for line_num, line in enumerate(annotation_text.strip().split("\n"), 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 5:
                result["valid"] = False
                result["errors"].append(f"Line {line_num}: Invalid format")
                continue

            try:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                # Check class ID
                if class_id >= len(class_names):
                    result["warnings"].append(f"Line {line_num}: Invalid class ID {class_id}")

                # Check coordinates
                if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
                    result["valid"] = False
                    result["errors"].append(f"Line {line_num}: Center coordinates out of range")

                if not (0 <= width <= 1 and 0 <= height <= 1):
                    result["valid"] = False
                    result["errors"].append(f"Line {line_num}: Width/height out of range")

            except ValueError as e:
                result["valid"] = False
                result["errors"].append(f"Line {line_num}: {e}")

        return result


class DataQualityChecker:
    """Check data quality metrics."""

    def __init__(self):
        """Initialize DataQualityChecker."""
        logger.info("DataQualityChecker initialized")

    def check_quality(
        self,
        image_dir: str,
        annotation_dir: str,
        class_names: List[str],
    ) -> QualityMetrics:
        """
        Check data quality metrics.

        Args:
            image_dir: Directory containing images
            annotation_dir: Directory containing annotations
            class_names: List of class names

        Returns:
            QualityMetrics object
        """
        metrics = QualityMetrics(
            total_images=0,
            total_annotations=0,
            avg_annotations_per_image=0,
            class_distribution={},
            annotation_size_stats={},
            missing_annotations=[],
            duplicate_annotations=[],
        )

        # Get all images
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        image_files = [
            f for f in Path(image_dir).iterdir()
            if f.suffix.lower() in image_extensions
        ]
        metrics.total_images = len(image_files)

        class_counts = Counter()
        annotation_sizes = []

        for image_file in image_files:
            ann_file = Path(annotation_dir) / image_file.stem
            ann_file = ann_file.with_suffix(".txt")

            if not ann_file.exists():
                metrics.missing_annotations.append(image_file.name)
                continue

            try:
                with open(ann_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    metrics.total_annotations += len(lines)
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

        # Calculate statistics
        if metrics.total_images > 0:
            metrics.avg_annotations_per_image = metrics.total_annotations / metrics.total_images

        metrics.class_distribution = dict(class_counts)

        if annotation_sizes:
            metrics.annotation_size_stats = {
                "min": min(annotation_sizes),
                "max": max(annotation_sizes),
                "mean": np.mean(annotation_sizes),
                "std": np.std(annotation_sizes),
            }

        logger.info(f"Quality check completed: {metrics.total_images} images, {metrics.total_annotations} annotations")

        return metrics


class PresetManager:
    """Manage annotation presets."""

    def __init__(self, presets_dir: str = "./presets"):
        """
        Initialize PresetManager.

        Args:
            presets_dir: Directory to store presets
        """
        self.presets_dir = Path(presets_dir)
        self.presets_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"PresetManager initialized: {self.presets_dir}")

    def save_preset(self, preset_name: str, preset_data: Dict[str, Any]) -> bool:
        """
        Save a preset.

        Args:
            preset_name: Name of the preset
            preset_data: Preset data

        Returns:
            True if saved successfully
        """
        try:
            import json
            preset_file = self.presets_dir / f"{preset_name}.json"
            with open(preset_file, "w", encoding="utf-8") as f:
                json.dump(preset_data, f, indent=2)
            logger.info(f"Preset saved: {preset_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to save preset: {e}")
            return False

    def load_preset(self, preset_name: str) -> Optional[Dict[str, Any]]:
        """
        Load a preset.

        Args:
            preset_name: Name of the preset

        Returns:
            Preset data or None
        """
        try:
            import json
            preset_file = self.presets_dir / f"{preset_name}.json"
            if not preset_file.exists():
                logger.warning(f"Preset not found: {preset_name}")
                return None

            with open(preset_file, "r", encoding="utf-8") as f:
                preset_data = json.load(f)
            logger.info(f"Preset loaded: {preset_name}")
            return preset_data
        except Exception as e:
            logger.error(f"Failed to load preset: {e}")
            return None

    def list_presets(self) -> List[str]:
        """
        List all available presets.

        Returns:
            List of preset names
        """
        presets = [f.stem for f in self.presets_dir.glob("*.json")]
        return sorted(presets)

    def delete_preset(self, preset_name: str) -> bool:
        """
        Delete a preset.

        Args:
            preset_name: Name of the preset

        Returns:
            True if deleted successfully
        """
        try:
            preset_file = self.presets_dir / f"{preset_name}.json"
            if preset_file.exists():
                preset_file.unlink()
                logger.info(f"Preset deleted: {preset_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete preset: {e}")
            return False
