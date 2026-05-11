"""Model management module — handles YOLO model loading and inference."""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from loguru import logger

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    logger.warning("ultralytics not installed, YOLO models will not be available")


class ModelManager:
    """Manages YOLO model loading, caching, and inference."""

    def __init__(self, models_dir: Optional[str] = None):
        """Initialize ModelManager.

        Args:
            models_dir: Directory to store/load models. If None, uses ./models
        """
        self.models_dir = Path(models_dir) if models_dir else Path("./models")
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.loaded_models: Dict[str, Any] = {}  # Cache for loaded models
        self.current_model: Optional[Any] = None
        self.current_model_name: Optional[str] = None
        self._device: str = ""  # Device to use for predict calls

        logger.info(f"ModelManager initialized with models_dir: {self.models_dir}")

    def list_available_models(self) -> List[str]:
        """List available YOLO models.

        Returns:
            List of model names (e.g., ['yolov8n.pt', 'yolov8s.pt', ...])
        """
        if not ULTRALYTICS_AVAILABLE:
            logger.warning("ultralytics not available")
            return []

        models = [
            # YOLO26 (newest)
            "yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt",
            "yolo26n-seg.pt", "yolo26s-seg.pt", "yolo26m-seg.pt",
            "yolo26n-pose.pt", "yolo26s-pose.pt",
            "yolo26n-obb.pt", "yolo26s-obb.pt",
            # YOLO11
            "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
            "yolo11n-seg.pt", "yolo11s-seg.pt",
            "yolo11n-pose.pt",
            "yolo11n-obb.pt",
            # YOLOv8
            "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
            "yolov8n-seg.pt", "yolov8s-seg.pt",
            "yolov8n-pose.pt",
            "yolov8n-obb.pt",
            # YOLOv5
            "yolov5nu.pt", "yolov5su.pt", "yolov5mu.pt",
            # RT-DETR
            "rtdetr-l.pt", "rtdetr-x.pt",
        ]
        return models

    def list_local_models(self) -> List[str]:
        """List locally available model files.

        Returns:
            List of model file paths
        """
        model_files = []
        for ext in ["*.pt", "*.onnx", "*.pth"]:
            model_files.extend(self.models_dir.glob(ext))
        return [str(f) for f in model_files]

    def load_model(self, model_name: str, device: str = "") -> bool:
        """Load a YOLO model.

        Args:
            model_name: Model name (e.g., 'yolov8n.pt') or path to model file (.pt/.onnx)
            device: Device to use ('0' for GPU, 'cpu' for CPU, '' for auto)

        Returns:
            True if loaded successfully, False otherwise
        """
        if not ULTRALYTICS_AVAILABLE:
            logger.error("ultralytics not available")
            return False

        try:
            # Check if model is already loaded (cache hit)
            cache_key = f"{model_name}@{device}"
            if cache_key in self.loaded_models:
                self.current_model = self.loaded_models[cache_key]
                self.current_model_name = model_name
                logger.info(f"Using cached model: {cache_key}")
                return True

            # Load model (factory pattern)
            logger.info(f"Loading model: {model_name}")
            model = YOLO(model_name)

            # Auto-detect device if not specified
            if not device:
                try:
                    import torch
                    device = "0" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    device = "cpu"

            # Store device for use in predict() calls.
            # NOTE: Ultralytics YOLO does NOT honor model.to(device).
            # The device must be passed explicitly via predict(device=...).
            self._device = device
            logger.info(f"Model will use device: {self._device}")

            # Cache model
            self.loaded_models[cache_key] = model
            self.current_model = model
            self.current_model_name = model_name

            logger.info(f"Model loaded successfully: {model_name} on {device}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return False

    def unload_model(self, model_name: Optional[str] = None) -> None:
        """Unload a model from cache.

        Args:
            model_name: Model name to unload. If None, unload current model.
        """
        if model_name is None:
            model_name = self.current_model_name

        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            if self.current_model_name == model_name:
                self.current_model = None
                self.current_model_name = None
            logger.info(f"Model unloaded: {model_name}")

    def predict(
        self,
        image_path: str,
        conf: float = 0.25,
        iou: float = 0.45,
        max_det: int = 300,
    ) -> Optional[List[Dict[str, Any]]]:
        """Run inference on an image.

        Args:
            image_path: Path to image file
            conf: Confidence threshold
            iou: IOU threshold for NMS
            max_det: Maximum number of detections

        Returns:
            List of detections or None if inference failed
        """
        if self.current_model is None:
            logger.error("No model loaded")
            return None

        try:
            logger.debug(f"Running inference on: {image_path}")
            results = self.current_model.predict(
                image_path,
                conf=conf,
                iou=iou,
                max_det=max_det,
                verbose=False,
                device=self._device or None,  # Pass device explicitly for GPU acceleration
            )

            if not results:
                logger.warning(f"No results from inference on: {image_path}")
                return None

            # Parse results — supports detect, OBB, and pose tasks
            detections = []
            for result in results:
                names = getattr(result, 'names', {}) or {}

                # --- OBB (Oriented Bounding Box) ---
                obb = getattr(result, "obb", None)
                if obb is not None and len(obb) > 0:
                    for i in range(len(obb)):
                        cls_id = int(obb.cls[i])
                        det = {
                            "class_id": cls_id,
                            "class_name": names.get(cls_id),
                            "confidence": float(obb.conf[i]),
                            "type": "obb",
                            "bbox": obb.xyxy[i].cpu().numpy().tolist() if obb.xyxy is not None else [],
                        }
                        # Convert xywhr to 4 corner points
                        if obb.xywhr is not None and len(obb.xywhr[i]) >= 5:
                            import math
                            cx, cy, w, h, r = obb.xywhr[i].cpu().numpy().tolist()[:5]
                            cos_a = math.cos(r)
                            sin_a = math.sin(r)
                            corners = []
                            for dx, dy in [(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)]:
                                px = cx + dx * cos_a - dy * sin_a
                                py = cy + dx * sin_a + dy * cos_a
                                corners.append((px, py))
                            det["corners"] = corners
                        detections.append(det)
                    continue

                # --- Keypoint / Pose ---
                keypoints = getattr(result, "keypoints", None)
                if keypoints is not None and len(keypoints) > 0 and result.boxes is not None:
                    for i in range(len(result.boxes)):
                        box = result.boxes[i]
                        cls_id = int(box.cls[0])
                        det = {
                            "class_id": cls_id,
                            "class_name": names.get(cls_id),
                            "confidence": float(box.conf[0]),
                            "type": "keypoint",
                            "bbox": box.xyxy[0].cpu().numpy().tolist(),
                        }
                        if hasattr(keypoints, "xy") and keypoints.xy is not None:
                            kps = keypoints.xy[i].cpu().numpy()
                            vis = None
                            if hasattr(keypoints, "visible") and keypoints.visible is not None:
                                vis = keypoints.visible[i].cpu().numpy()
                            det["keypoints"] = []
                            for ki in range(len(kps)):
                                kx, ky = float(kps[ki][0]), float(kps[ki][1])
                                v = int(vis[ki]) if vis is not None and ki < len(vis) else 2
                                det["keypoints"].append((kx, ky, v))
                        detections.append(det)
                    continue

                # --- Standard Detect (bbox) ---
                if result.boxes is not None:
                    for box in result.boxes:
                        cls_id = int(box.cls[0])
                        detection = {
                            "bbox": box.xyxy[0].cpu().numpy().tolist(),
                            "confidence": float(box.conf[0]),
                            "class_id": cls_id,
                            "class_name": names.get(cls_id),
                            "type": "bbox",
                        }
                        detections.append(detection)

            logger.debug(f"Found {len(detections)} detections")
            return detections

        except Exception as e:
            logger.error(f"Inference failed on {image_path}: {e}")
            return None

    def get_current_model(self) -> Optional[Any]:
        """Get the currently loaded model."""
        return self.current_model

    def get_current_model_name(self) -> Optional[str]:
        """Get the name of the currently loaded model."""
        return self.current_model_name

    def is_model_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self.current_model is not None

    def get_loaded_models_count(self) -> int:
        """Get number of loaded models in cache."""
        return len(self.loaded_models)

    def clear_cache(self) -> None:
        """Clear all cached models."""
        self.loaded_models.clear()
        self.current_model = None
        self.current_model_name = None
        logger.info("Model cache cleared")
