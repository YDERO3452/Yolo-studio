"""Advanced features module for annotation statistics and reporting."""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import Counter
from datetime import datetime
from loguru import logger

import numpy as np


@dataclass
class AnnotationStatistics:
    """Annotation statistics."""
    total_images: int
    total_annotations: int
    avg_annotations_per_image: float
    class_distribution: Dict[str, int]
    annotation_size_stats: Dict[str, float]
    image_coverage: float
    timestamp: str


class AnnotationStatisticsCollector:
    """Collect and analyze annotation statistics."""

    def __init__(self):
        """Initialize AnnotationStatisticsCollector."""
        logger.info("AnnotationStatisticsCollector initialized")

    def collect_statistics(
        self,
        image_dir: str,
        annotation_dir: str,
        class_names: List[str],
    ) -> AnnotationStatistics:
        """
        Collect annotation statistics.

        Args:
            image_dir: Directory containing images
            annotation_dir: Directory containing annotations
            class_names: List of class names

        Returns:
            AnnotationStatistics object
        """
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        image_files = [
            f for f in Path(image_dir).iterdir()
            if f.suffix.lower() in image_extensions
        ]

        total_images = len(image_files)
        total_annotations = 0
        class_counts = Counter()
        annotation_sizes = []
        annotated_images = 0

        for image_file in image_files:
            ann_file = Path(annotation_dir) / image_file.stem
            ann_file = ann_file.with_suffix(".txt")

            if not ann_file.exists():
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

        # Calculate statistics
        avg_annotations = total_annotations / total_images if total_images > 0 else 0
        image_coverage = annotated_images / total_images if total_images > 0 else 0

        size_stats = {}
        if annotation_sizes:
            size_stats = {
                "min": min(annotation_sizes),
                "max": max(annotation_sizes),
                "mean": float(np.mean(annotation_sizes)),
                "std": float(np.std(annotation_sizes)),
                "median": float(np.median(annotation_sizes)),
            }

        stats = AnnotationStatistics(
            total_images=total_images,
            total_annotations=total_annotations,
            avg_annotations_per_image=avg_annotations,
            class_distribution=dict(class_counts),
            annotation_size_stats=size_stats,
            image_coverage=image_coverage,
            timestamp=datetime.now().isoformat(),
        )

        logger.info(f"Statistics collected: {total_images} images, {total_annotations} annotations")
        return stats


class ReportGenerator:
    """Generate annotation reports."""

    def __init__(self):
        """Initialize ReportGenerator."""
        logger.info("ReportGenerator initialized")

    def generate_html_report(
        self,
        statistics: AnnotationStatistics,
        output_file: str,
        title: str = "Annotation Report",
    ) -> bool:
        """
        Generate HTML report.

        Args:
            statistics: AnnotationStatistics object
            output_file: Output file path
            title: Report title

        Returns:
            True if generated successfully
        """
        try:
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .section {{ margin: 20px 0; padding: 10px; border: 1px solid #ddd; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        .stat-box {{ display: inline-block; margin: 10px; padding: 10px; background-color: #f0f0f0; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <p>Generated: {statistics.timestamp}</p>

    <div class="section">
        <h2>Summary Statistics</h2>
        <div class="stat-box">
            <strong>Total Images:</strong> {statistics.total_images}
        </div>
        <div class="stat-box">
            <strong>Total Annotations:</strong> {statistics.total_annotations}
        </div>
        <div class="stat-box">
            <strong>Avg Annotations/Image:</strong> {statistics.avg_annotations_per_image:.2f}
        </div>
        <div class="stat-box">
            <strong>Image Coverage:</strong> {statistics.image_coverage*100:.1f}%
        </div>
    </div>

    <div class="section">
        <h2>Class Distribution</h2>
        <table>
            <tr>
                <th>Class Name</th>
                <th>Count</th>
                <th>Percentage</th>
            </tr>
"""

            total_count = sum(statistics.class_distribution.values())
            for class_name, count in sorted(
                statistics.class_distribution.items(),
                key=lambda x: x[1],
                reverse=True
            ):
                percentage = (count / total_count * 100) if total_count > 0 else 0
                html_content += f"""
            <tr>
                <td>{class_name}</td>
                <td>{count}</td>
                <td>{percentage:.1f}%</td>
            </tr>
"""

            html_content += """
        </table>
    </div>

    <div class="section">
        <h2>Annotation Size Statistics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
"""

            for key, value in statistics.annotation_size_stats.items():
                html_content += f"""
            <tr>
                <td>{key.capitalize()}</td>
                <td>{value:.2f}</td>
            </tr>
"""

            html_content += """
        </table>
    </div>

</body>
</html>
"""

            with open(output_file, "w", encoding="utf-8") as f:
                f.write(html_content)

            logger.info(f"HTML report generated: {output_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to generate HTML report: {e}")
            return False

    def generate_json_report(
        self,
        statistics: AnnotationStatistics,
        output_file: str,
    ) -> bool:
        """
        Generate JSON report.

        Args:
            statistics: AnnotationStatistics object
            output_file: Output file path

        Returns:
            True if generated successfully
        """
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(asdict(statistics), f, indent=2)

            logger.info(f"JSON report generated: {output_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to generate JSON report: {e}")
            return False

    def generate_text_report(
        self,
        statistics: AnnotationStatistics,
        output_file: str,
    ) -> bool:
        """
        Generate text report.

        Args:
            statistics: AnnotationStatistics object
            output_file: Output file path

        Returns:
            True if generated successfully
        """
        try:
            report_text = f"""
ANNOTATION REPORT
{'='*60}
Generated: {statistics.timestamp}

SUMMARY STATISTICS
{'-'*60}
Total Images: {statistics.total_images}
Total Annotations: {statistics.total_annotations}
Avg Annotations/Image: {statistics.avg_annotations_per_image:.2f}
Image Coverage: {statistics.image_coverage*100:.1f}%

CLASS DISTRIBUTION
{'-'*60}
"""

            total_count = sum(statistics.class_distribution.values())
            for class_name, count in sorted(
                statistics.class_distribution.items(),
                key=lambda x: x[1],
                reverse=True
            ):
                percentage = (count / total_count * 100) if total_count > 0 else 0
                report_text += f"{class_name:20s} {count:6d} ({percentage:5.1f}%)\n"

            report_text += f"""
ANNOTATION SIZE STATISTICS
{'-'*60}
"""

            for key, value in statistics.annotation_size_stats.items():
                report_text += f"{key.capitalize():20s} {value:8.2f}\n"

            with open(output_file, "w", encoding="utf-8") as f:
                f.write(report_text)

            logger.info(f"Text report generated: {output_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to generate text report: {e}")
            return False


class DataAugmentationHelper:
    """Helper for data augmentation."""

    def __init__(self):
        """Initialize DataAugmentationHelper."""
        logger.info("DataAugmentationHelper initialized")

    def suggest_augmentation(
        self,
        statistics: AnnotationStatistics,
    ) -> Dict[str, Any]:
        """
        Suggest data augmentation strategies.

        Args:
            statistics: AnnotationStatistics object

        Returns:
            Dictionary with augmentation suggestions
        """
        suggestions = {
            "strategies": [],
            "reasons": [],
        }

        # Check class imbalance
        if statistics.class_distribution:
            counts = list(statistics.class_distribution.values())
            if len(counts) > 1:
                max_count = max(counts)
                min_count = min(counts)
                imbalance_ratio = max_count / min_count if min_count > 0 else 0

                if imbalance_ratio > 2:
                    suggestions["strategies"].append("class_balancing")
                    suggestions["reasons"].append(
                        f"High class imbalance (ratio: {imbalance_ratio:.2f})"
                    )

        # Check data volume
        if statistics.total_images < 100:
            suggestions["strategies"].append("geometric_transforms")
            suggestions["reasons"].append("Limited dataset size")

        # Check annotation density
        if statistics.avg_annotations_per_image < 2:
            suggestions["strategies"].append("mixup")
            suggestions["reasons"].append("Low annotation density")

        return suggestions

    def get_augmentation_config(self) -> Dict[str, Any]:
        """
        Get recommended augmentation configuration.

        Returns:
            Dictionary with augmentation configuration
        """
        return {
            "geometric_transforms": {
                "rotation": {"min": -15, "max": 15},
                "scale": {"min": 0.8, "max": 1.2},
                "flip": {"horizontal": True, "vertical": False},
            },
            "color_transforms": {
                "brightness": {"min": 0.8, "max": 1.2},
                "contrast": {"min": 0.8, "max": 1.2},
                "saturation": {"min": 0.8, "max": 1.2},
            },
            "noise": {
                "gaussian": {"std": 0.01},
                "salt_pepper": {"ratio": 0.001},
            },
        }


class ModelFineTuningHelper:
    """Helper for model fine-tuning."""

    def __init__(self):
        """Initialize ModelFineTuningHelper."""
        logger.info("ModelFineTuningHelper initialized")

    def suggest_training_config(
        self,
        statistics: AnnotationStatistics,
    ) -> Dict[str, Any]:
        """
        Suggest training configuration based on statistics.

        Args:
            statistics: AnnotationStatistics object

        Returns:
            Dictionary with training configuration
        """
        config = {
            "batch": 16,
            "epochs": 100,
            "lr0": 0.001,
            "optimizer": "adam",
        }

        # Adjust batch size based on dataset size
        if statistics.total_images < 100:
            config["batch"] = 8
            config["epochs"] = 200
        elif statistics.total_images < 500:
            config["batch"] = 16
            config["epochs"] = 150
        else:
            config["batch"] = 32
            config["epochs"] = 100

        # Adjust learning rate based on class imbalance
        if statistics.class_distribution:
            counts = list(statistics.class_distribution.values())
            if len(counts) > 1:
                max_count = max(counts)
                min_count = min(counts)
                imbalance_ratio = max_count / min_count if min_count > 0 else 0

                if imbalance_ratio > 3:
                    config["lr0"] = 0.0005

        return config

    def get_training_tips(self) -> List[str]:
        """Get training tips."""
        return [
            "Use data augmentation to increase dataset diversity",
            "Monitor validation metrics to prevent overfitting",
            "Use learning rate scheduling for better convergence",
            "Consider using class weights for imbalanced datasets",
            "Save best model checkpoint during training",
            "Use early stopping to prevent overfitting",
        ]
