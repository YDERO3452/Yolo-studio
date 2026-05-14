"""SAM-based interactive annotation handler."""

import json
import gc
import os
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from loguru import logger
from PyQt6.QtCore import QObject, QThread, pyqtSignal

SAM_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "sam_config.json"

DEFAULT_SAM_CONFIG = {
    "sam_type": "SAM2",
    "model_file": "sam2.1_b.pt",
    "device": "",
    "imgsz": 1024,
    "conf": 0.25,
    "iou": 0.9,
    "retina_masks": True,
    "usage_mode": "normal",
    "output_shape": "auto",
}


def load_sam_config() -> dict:
    if SAM_CONFIG_PATH.exists():
        try:
            return json.loads(SAM_CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(f"Failed to load SAM config: {exc}")
    return dict(DEFAULT_SAM_CONFIG)


def save_sam_config(config: dict):
    SAM_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    SAM_CONFIG_PATH.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")


class SAMModelManager(QObject):
    """Singleton caching manager for SAM/FastSAM models."""

    _instance = None
    _model = None
    _current_config = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_model(self, config: dict) -> bool:
        config_key = (config.get("sam_type"), config.get("model_file"), config.get("device"))
        if self._model is not None and self._current_config == config_key:
            return True

        self.release_model()
        try:
            sam_type = config.get("sam_type", "SAM2")
            model_file = self._resolve_model_file(config.get("model_file", "sam2.1_b.pt"))
            device = config.get("device", "") or None

            if sam_type in ("SAM2", "SAM3", "SAM"):
                from ultralytics import SAM as UltralyticsSAM
                self._model = UltralyticsSAM(model_file)
            elif sam_type == "MobileSAM":
                from ultralytics import SAM as UltralyticsSAM
                self._model = UltralyticsSAM(model_file)
            elif sam_type == "FastSAM":
                from ultralytics import FastSAM
                self._model = FastSAM(model_file)
            else:
                logger.error(f"Unknown SAM type: {sam_type}")
                return False

            if device:
                self._model.to(device)
            self._current_config = config_key
            logger.info(f"SAM model loaded: {sam_type}/{model_file} on {device or 'auto'}")
            return True
        except Exception as exc:
            logger.error(f"Failed to load SAM model: {exc}")
            return False

    @staticmethod
    def _resolve_model_file(model_file: str) -> str:
        """Resolve a SAM model from absolute path, ./models, or Ultralytics name."""
        if not model_file:
            return model_file

        candidate = Path(model_file).expanduser()
        if candidate.is_file():
            return str(candidate)

        project_root = Path(__file__).resolve().parent.parent
        for root in (project_root / "models", project_root):
            local = root / model_file
            if local.is_file():
                return str(local)

        return model_file

    def predict(
        self,
        image: np.ndarray,
        points=None,
        labels=None,
        boxes=None,
        conf=None,
        iou=None,
        imgsz=None,
        retina_masks=True,
    ) -> Optional:
        if self._model is None:
            return None
        try:
            kwargs = {}
            if points is not None:
                kwargs["points"] = points
            if labels is not None:
                kwargs["labels"] = labels
            if boxes is not None:
                kwargs["bboxes"] = boxes
            if conf is not None:
                kwargs["conf"] = conf
            if iou is not None:
                kwargs["iou"] = iou
            if imgsz is not None:
                kwargs["imgsz"] = imgsz
            if retina_masks is not None:
                kwargs["retina_masks"] = retina_masks
            results = self._model(image, **kwargs)
            return results
        except Exception as exc:
            logger.error(f"SAM predict error: {exc}")
            return None

    def release_model(self):
        if self._model is not None:
            try:
                import gc
                import torch
                del self._model
                torch.cuda.empty_cache()
                gc.collect()
            except Exception:
                pass
            self._model = None
            self._current_config = None

    def is_loaded(self) -> bool:
        return self._model is not None


class SAMInferenceWorker(QThread):
    """Background SAM inference thread."""
    finished = pyqtSignal(object)  # results
    error = pyqtSignal(str)

    def __init__(self, image: np.ndarray, sam_manager: SAMModelManager, config: dict,
                 points=None, labels=None, boxes=None, prompt_type="point", parent=None):
        super().__init__(parent)
        self.image = image
        self.sam_manager = sam_manager
        self.config = config
        self.points = points
        self.labels = labels
        self.boxes = boxes
        self.prompt_type = prompt_type

    def run(self):
        try:
            results = self.sam_manager.predict(
                self.image,
                points=self.points,
                labels=self.labels,
                boxes=self.boxes,
                conf=self.config.get("conf", 0.25),
                iou=self.config.get("iou", 0.9),
                imgsz=self.config.get("imgsz", 1024),
                retina_masks=self.config.get("retina_masks", True),
            )
            self.finished.emit(results)
        except Exception as exc:
            self.error.emit(str(exc))


class SAMMemoryPredictorManager:
    """Singleton manager for SAM2/SAM3 dynamic memory prediction."""

    _instance = None
    _instance_lock = threading.Lock()

    def __init__(self):
        self._lock = threading.Lock()
        self._predictor = None
        self._predictor_key = None

    @classmethod
    def instance(cls):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def get_predictor(self, config: dict):
        sam_type = config.get("sam_type", "SAM2")
        if sam_type not in ("SAM2", "SAM3"):
            raise ValueError("SAM memory mode requires SAM2 or SAM3")

        model_file = SAMModelManager._resolve_model_file(config.get("model_file", "sam2.1_b.pt"))
        device = config.get("device", "") or None
        imgsz = int(config.get("imgsz", 1024))
        predictor_key = (sam_type, model_file, str(device), imgsz)

        with self._lock:
            if self._predictor is not None and self._predictor_key == predictor_key:
                return self._predictor

            self._release_locked()
            from ultralytics.models.sam import SAM2DynamicInteractivePredictor

            overrides = {
                "conf": 0.01,
                "task": "segment",
                "mode": "predict",
                "imgsz": imgsz,
                "model": model_file,
                "save": False,
                "verbose": False,
            }
            if device:
                overrides["device"] = device
            self._predictor = SAM2DynamicInteractivePredictor(overrides=overrides, max_obj_num=20)
            self._predictor_key = predictor_key
            logger.info(f"SAM memory predictor loaded: {sam_type}/{model_file} on {device or 'auto'}")
            return self._predictor

    def predict(self, config: dict, image_path: str, update_objects: Optional[list[dict]] = None):
        predictor = self.get_predictor(config)
        last_update_results = None
        for obj in update_objects or []:
            kwargs = {
                "source": image_path,
                "obj_ids": [int(obj.get("obj_id", 1))],
                "update_memory": True,
            }
            bboxes = obj.get("bboxes") or []
            points = obj.get("points") or []
            if bboxes:
                kwargs["bboxes"] = [list(map(float, bboxes[-1]))]
            if points:
                kwargs["points"] = [[float(p[0]), float(p[1])] for p in points]
                kwargs["labels"] = [1] * len(points)
            update_results = predictor(**kwargs)
            if update_results:
                last_update_results = update_results

        infer_results = predictor(source=image_path)
        if memory_results_empty(infer_results) and not memory_results_empty(last_update_results):
            return last_update_results
        return infer_results

    def clear(self):
        with self._lock:
            self._release_locked()

    def _release_locked(self):
        if self._predictor is not None:
            try:
                del self._predictor
            except Exception:
                pass
            self._predictor = None
            self._predictor_key = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass


def memory_results_empty(results) -> bool:
    if not results or len(results) == 0:
        return True
    result = results[0]
    masks = getattr(result, "masks", None)
    data = getattr(masks, "data", None) if masks is not None else None
    if data is None:
        return True
    try:
        return len(data) == 0
    except Exception:
        return False


def mask_to_polygon(mask: np.ndarray, min_area: int = 50, epsilon: float = 2.0) -> Optional[list]:
    """Convert a binary mask to a polygon approximation."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return None
    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    return [(int(pt[0][0]), int(pt[0][1])) for pt in approx]


def mask_to_bbox(mask: np.ndarray) -> Optional[tuple]:
    """Convert a binary mask to a bounding box (x1, y1, x2, y2)."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
