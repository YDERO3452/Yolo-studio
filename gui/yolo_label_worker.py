"""YOLO auto-label background worker."""

from loguru import logger
from PyQt6.QtCore import QObject, pyqtSignal

from core.model_manager import ModelManager


class YOLOAutoLabelWorker(QObject):
    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(
        self,
        model_manager: ModelManager,
        image_paths: list[str],
        conf: float,
        iou: float,
        max_det: int,
    ):
        super().__init__()
        self.model_manager = model_manager
        self.image_paths = image_paths
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self._running = True

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        try:
            results = {}
            total = len(self.image_paths)
            for index, image_path in enumerate(self.image_paths, start=1):
                if not self._running:
                    break
                detections = self.model_manager.predict(
                    image_path,
                    conf=self.conf,
                    iou=self.iou,
                    max_det=self.max_det,
                )
                results[image_path] = detections or []
                self.progress.emit(index, total, image_path)
            self.finished.emit(results)
        except Exception as exc:
            logger.error(f"YOLO auto-label error: {exc}")
            self.error.emit(str(exc))
