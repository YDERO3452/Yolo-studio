"""Enhanced auto-labeling module with advanced features.

Architecture overview:
- Configuration-driven labeling
- Multiple output modes (rectangle, polygon, point)
- Confidence and IOU threshold management
- Result caching and optimization
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from loguru import logger

from core.model_manager import ModelManager


@dataclass
class AutoLabelingConfig:
    """Configuration for auto-labeling."""
    model_name: str
    conf_threshold: float = 0.25
    iou_threshold: float = 0.7          # Ultralytics NMS default=0.7
    max_detections: int = 300
    output_mode: str = "rectangle"  # rectangle, polygon, point
    preserve_existing: bool = False
    auto_save: bool = True
    device: str = "0"
    custom_params: Dict[str, Any] = field(default_factory=dict)


class AutoLabelingEngine:
    """Enhanced auto-labeling engine with advanced features.

    Architecture pattern: configuration-driven auto-labeling workflow.
    - Configuration management
    - Model switching and caching
    - Output mode support
    - Result filtering and post-processing
    """

    def __init__(self, model_manager: ModelManager, class_names: List[str]):
        """Initialize AutoLabelingEngine.

        Args:
            model_manager: ModelManager instance
            class_names: List of class names
        """
        self.model_manager = model_manager
        self.class_names = class_names
        self.config: Optional[AutoLabelingConfig] = None
        self.last_results: Optional[List[Dict[str, Any]]] = None
        logger.info("AutoLabelingEngine initialized")

    def set_config(self, config: AutoLabelingConfig) -> None:
        """Set auto-labeling configuration.

        Args:
            config: AutoLabelingConfig instance
        """
        self.config = config
        logger.info(f"Configuration updated: model={config.model_name}, "
                   f"conf={config.conf_threshold}, iou={config.iou_threshold}")

    def predict(self, image_path: str) -> Optional[List[Dict[str, Any]]]:
        """Run auto-labeling on an image.

        Args:
            image_path: Path to image file

        Returns:
            List of detections or None if failed
        """
        if self.config is None:
            logger.error("No configuration set")
            return None

        # Load model if not already loaded
        if self.model_manager.get_current_model_name() != self.config.model_name:
            if not self.model_manager.load_model(
                self.config.model_name,
                device=self.config.device
            ):
                logger.error(f"Failed to load model: {self.config.model_name}")
                return None

        # Run inference
        detections = self.model_manager.predict(
            image_path,
            conf=self.config.conf_threshold,
            iou=self.config.iou_threshold,
            max_det=self.config.max_detections,
        )

        if detections is None:
            logger.warning(f"No detections for {image_path}")
            return None

        # Post-process results
        detections = self._post_process_detections(detections)

        # Cache results
        self.last_results = detections

        logger.debug(f"Auto-labeling completed: {len(detections)} detections")
        return detections

    def _post_process_detections(
        self,
        detections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Post-process detections based on configuration.

        Args:
            detections: Raw detections from model

        Returns:
            Post-processed detections
        """
        processed = []

        for det in detections:
            # Filter by confidence
            if det.get("confidence", 0) < self.config.conf_threshold:
                continue

            # Convert to output mode
            if self.config.output_mode == "rectangle":
                processed.append(det)
            elif self.config.output_mode == "polygon":
                det = self._convert_to_polygon(det)
                if det:
                    processed.append(det)
            elif self.config.output_mode == "point":
                det = self._convert_to_point(det)
                if det:
                    processed.append(det)

        return processed

    def _convert_to_polygon(self, detection: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert rectangle detection to polygon.

        Args:
            detection: Detection dictionary

        Returns:
            Modified detection or None
        """
        bbox = detection.get("bbox", [])
        if len(bbox) < 4:
            return None

        x1, y1, x2, y2 = bbox[:4]

        # Create polygon from rectangle corners
        polygon = [
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2],
        ]

        detection["polygon"] = polygon
        return detection

    def _convert_to_point(self, detection: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert rectangle detection to point (center).

        Args:
            detection: Detection dictionary

        Returns:
            Modified detection or None
        """
        bbox = detection.get("bbox", [])
        if len(bbox) < 4:
            return None

        x1, y1, x2, y2 = bbox[:4]

        # Calculate center point
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        detection["point"] = [center_x, center_y]
        return detection

    def filter_detections(
        self,
        detections: List[Dict[str, Any]],
        min_confidence: float = 0.0,
        class_ids: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """Filter detections by confidence and class.

        Args:
            detections: List of detections
            min_confidence: Minimum confidence threshold
            class_ids: List of class IDs to keep (None = all)

        Returns:
            Filtered detections
        """
        filtered = []

        for det in detections:
            # Check confidence
            if det.get("confidence", 0) < min_confidence:
                continue

            # Check class ID
            if class_ids is not None and det.get("class_id") not in class_ids:
                continue

            filtered.append(det)

        return filtered

    def merge_detections(
        self,
        detections1: List[Dict[str, Any]],
        detections2: List[Dict[str, Any]],
        iou_threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Merge two sets of detections, removing duplicates.

        Args:
            detections1: First set of detections
            detections2: Second set of detections
            iou_threshold: IOU threshold for duplicate detection

        Returns:
            Merged detections
        """
        merged = list(detections1)

        for det2 in detections2:
            is_duplicate = False

            for det1 in merged:
                iou = self._calculate_iou(det1["bbox"], det2["bbox"])
                if iou > iou_threshold:
                    is_duplicate = True
                    # Keep the one with higher confidence
                    if det2.get("confidence", 0) > det1.get("confidence", 0):
                        merged.remove(det1)
                        merged.append(det2)
                    break

            if not is_duplicate:
                merged.append(det2)

        return merged

    @staticmethod
    def _calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate Intersection over Union (IOU) between two bboxes.

        Args:
            bbox1: First bbox [x1, y1, x2, y2]
            bbox2: Second bbox [x1, y1, x2, y2]

        Returns:
            IOU value between 0 and 1
        """
        if len(bbox1) < 4 or len(bbox2) < 4:
            return 0.0
        x1_min, y1_min, x1_max, y1_max = bbox1[:4]
        x2_min, y2_min, x2_max, y2_max = bbox2[:4]

        # Calculate intersection
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)

        if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
            return 0.0

        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)

        # Calculate union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area

        if union_area == 0:
            return 0.0

        return inter_area / union_area

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics from last auto-labeling run.

        Returns:
            Dictionary with statistics
        """
        if self.last_results is None:
            return {}

        stats = {
            "total_detections": len(self.last_results),
            "by_class": {},
            "confidence_stats": {
                "min": 0.0,
                "max": 0.0,
                "avg": 0.0,
            }
        }

        # Count by class
        for det in self.last_results:
            class_id = det.get("class_id", -1)
            class_name = self.class_names[class_id] if 0 <= class_id < len(self.class_names) else "unknown"

            if class_name not in stats["by_class"]:
                stats["by_class"][class_name] = 0
            stats["by_class"][class_name] += 1

        # Confidence statistics
        confidences = [det.get("confidence", 0) for det in self.last_results]
        if confidences:
            stats["confidence_stats"]["min"] = min(confidences)
            stats["confidence_stats"]["max"] = max(confidences)
            stats["confidence_stats"]["avg"] = sum(confidences) / len(confidences)

        return stats
