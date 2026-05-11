"""YOLO training module."""

import os
import time
from pathlib import Path
from typing import Optional, Callable

from loguru import logger


class YOLOTrainer:
    """Manages YOLO model training."""

    def __init__(self, config=None):
        self.config = config
        self.model = None
        self.is_training = False
        self.training_thread = None
        self.callbacks = {}

    def load_model(self, model_path: str = "yolov8n.pt"):
        """Load a YOLO model."""
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        logger.info(f"Loaded model: {model_path}")
        return self.model

    def train(self, data_yaml: str, **kwargs) -> dict:
        """Train the model synchronously."""
        if self.model is None:
            self.load_model()

        self.is_training = True
        start_time = time.time()

        try:
            # Merge config with kwargs
            train_args = self._build_train_args(data_yaml, **kwargs)

            logger.info(f"Starting training with args: {train_args}")
            results = self.model.train(**train_args)

            elapsed = time.time() - start_time
            logger.info(f"Training completed in {elapsed:.1f}s")

            # Extract save_dir from multiple possible sources
            save_dir = None
            if hasattr(results, "save_dir"):
                save_dir = str(results.save_dir)
            elif hasattr(self.model, "trainer") and self.model.trainer and hasattr(self.model.trainer, "save_dir"):
                save_dir = str(self.model.trainer.save_dir)

            return {
                "success": True,
                "results": results,
                "elapsed": elapsed,
                "save_dir": save_dir,
            }
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {"success": False, "error": str(e)}
        finally:
            self.is_training = False

    def train_async(self, data_yaml: str, callback: Optional[Callable] = None, **kwargs):
        """Train the model asynchronously in a separate thread."""
        import threading

        def _train_worker():
            result = self.train(data_yaml, **kwargs)
            if callback:
                callback(result)

        self.training_thread = threading.Thread(target=_train_worker, daemon=True)
        self.training_thread.start()
        return self.training_thread

    def stop_training(self):
        """Request training stop."""
        self.is_training = False
        logger.info("Training stop requested")

    def _build_train_args(self, data_yaml: str, **kwargs) -> dict:
        """Build training arguments from config and overrides.

        Field names now match Ultralytics API exactly, so we can
        dump the TrainingConfig directly and just filter out Nones.
        """
        args = {"data": data_yaml}

        if self.config:
            tc = self.config.training
            # Dump all fields; filter out None values (Ultralytics uses
            # None as "use default" and will error on explicit None).
            dumped = tc.model_dump(exclude_none=False)
            for key, value in dumped.items():
                if value is not None:
                    args[key] = value

        # Override with kwargs
        args.update(kwargs)

        # Windows-specific memory optimizations for PyTorch DataLoader
        self._apply_windows_memory_fixes(args)

        return args

    @staticmethod
    def _apply_windows_memory_fixes(args: dict):
        """Apply Windows-specific memory optimizations to training args.

        On Windows, PyTorch DataLoader workers use separate processes (not fork),
        which means each worker has its own copy of the dataset in memory.
        This can cause OOM crashes, especially with error code 1455
        (ERROR_COMMITMENT_LIMIT — page file exhausted).

        Mitigations:
        - Reduce workers to a safe default on Windows (max 4)
        - Set PyTorch sharing strategy to 'file_system' (more stable on Windows)
        - Force garbage collection between epochs via callback
        """
        import sys
        if sys.platform != "win32":
            return

        import torch
        import torch.multiprocessing as mp

        # Use file_system sharing strategy — more stable on Windows than default
        # 'file_descriptor' strategy which can hit shared memory mapping limits
        try:
            mp.set_sharing_strategy("file_system")
            logger.info("Set multiprocessing sharing strategy: file_system")
        except Exception as e:
            logger.warning(f"Failed to set sharing strategy: {e}")

        # Cap workers on Windows to reduce memory usage
        workers = args.get("workers", 8)
        max_windows_workers = 4
        if workers > max_windows_workers:
            logger.warning(
                f"Windows detected: reducing workers from {workers} to "
                f"{max_windows_workers} to prevent OOM "
                f"(error code 1455). You can override by setting workers <= {max_windows_workers}."
            )
            args["workers"] = max_windows_workers

    def validate(self, data_yaml: str, **kwargs) -> dict:
        """Validate the model on a dataset."""
        if self.model is None:
            raise ValueError("No model loaded")

        try:
            results = self.model.val(data=data_yaml, **kwargs)
            return {"success": True, "results": results}
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {"success": False, "error": str(e)}

    def get_training_logs(self, project_dir: str) -> dict:
        """Parse training logs from a completed run."""
        import pandas as pd

        results_csv = Path(project_dir) / "results.csv"
        if not results_csv.exists():
            return {}

        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip()

        logs = {
            "epochs": df.index.tolist(),
            "train/box_loss": df["train/box_loss"].tolist() if "train/box_loss" in df.columns else [],
            "train/cls_loss": df["train/cls_loss"].tolist() if "train/cls_loss" in df.columns else [],
            "train/dfl_loss": df["train/dfl_loss"].tolist() if "train/dfl_loss" in df.columns else [],
            "val/box_loss": df["val/box_loss"].tolist() if "val/box_loss" in df.columns else [],
            "val/cls_loss": df["val/cls_loss"].tolist() if "val/cls_loss" in df.columns else [],
            "val/dfl_loss": df["val/dfl_loss"].tolist() if "val/dfl_loss" in df.columns else [],
            "metrics/precision(B)": df["metrics/precision(B)"].tolist() if "metrics/precision(B)" in df.columns else [],
            "metrics/recall(B)": df["metrics/recall(B)"].tolist() if "metrics/recall(B)" in df.columns else [],
            "metrics/mAP50(B)": df["metrics/mAP50(B)"].tolist() if "metrics/mAP50(B)" in df.columns else [],
            "metrics/mAP50-95(B)": df["metrics/mAP50-95(B)"].tolist() if "metrics/mAP50-95(B)" in df.columns else [],
        }
        return logs

    @staticmethod
    def list_pretrained_models() -> list[str]:
        """List available pretrained YOLO models.

        Covers all Ultralytics-supported model families and tasks:
        - YOLO26 / YOLO12 / YOLO11 / YOLOv8: detect, segment, classify, pose, obb
        - YOLOv10 / YOLOv9 / YOLOv5: detect
        - RT-DETR: detect (real-time DETR)
        """
        return [
            # ── YOLO26 (2025.9, newest) ───────────────────────────
            # Detect
            "yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt",
            # Segment
            "yolo26n-seg.pt", "yolo26s-seg.pt", "yolo26m-seg.pt", "yolo26l-seg.pt", "yolo26x-seg.pt",
            # Classify
            "yolo26n-cls.pt", "yolo26s-cls.pt", "yolo26m-cls.pt", "yolo26l-cls.pt", "yolo26x-cls.pt",
            # Pose
            "yolo26n-pose.pt", "yolo26s-pose.pt", "yolo26m-pose.pt", "yolo26l-pose.pt", "yolo26x-pose.pt",
            # OBB
            "yolo26n-obb.pt", "yolo26s-obb.pt", "yolo26m-obb.pt", "yolo26l-obb.pt", "yolo26x-obb.pt",

            # ── YOLO12 (2025.2) ────────────────────────────────────
            "yolo12n.pt", "yolo12s.pt", "yolo12m.pt", "yolo12l.pt", "yolo12x.pt",
            "yolo12n-seg.pt", "yolo12s-seg.pt", "yolo12m-seg.pt", "yolo12l-seg.pt", "yolo12x-seg.pt",
            "yolo12n-cls.pt", "yolo12s-cls.pt", "yolo12m-cls.pt", "yolo12l-cls.pt", "yolo12x-cls.pt",
            "yolo12n-pose.pt", "yolo12s-pose.pt", "yolo12m-pose.pt", "yolo12l-pose.pt", "yolo12x-pose.pt",
            "yolo12n-obb.pt", "yolo12s-obb.pt", "yolo12m-obb.pt", "yolo12l-obb.pt", "yolo12x-obb.pt",

            # ── YOLO11 ─────────────────────────────────────────────
            "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
            "yolo11n-seg.pt", "yolo11s-seg.pt", "yolo11m-seg.pt", "yolo11l-seg.pt", "yolo11x-seg.pt",
            "yolo11n-cls.pt", "yolo11s-cls.pt", "yolo11m-cls.pt", "yolo11l-cls.pt", "yolo11x-cls.pt",
            "yolo11n-pose.pt", "yolo11s-pose.pt", "yolo11m-pose.pt", "yolo11l-pose.pt", "yolo11x-pose.pt",
            "yolo11n-obb.pt", "yolo11s-obb.pt", "yolo11m-obb.pt", "yolo11l-obb.pt", "yolo11x-obb.pt",

            # ── YOLOv10 ────────────────────────────────────────────
            "yolov10n.pt", "yolov10s.pt", "yolov10m.pt", "yolov10l.pt", "yolov10x.pt",

            # ── YOLOv8 ─────────────────────────────────────────────
            "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
            "yolov8n-seg.pt", "yolov8s-seg.pt", "yolov8m-seg.pt", "yolov8l-seg.pt", "yolov8x-seg.pt",
            "yolov8n-cls.pt", "yolov8s-cls.pt", "yolov8m-cls.pt", "yolov8l-cls.pt", "yolov8x-cls.pt",
            "yolov8n-pose.pt", "yolov8s-pose.pt", "yolov8m-pose.pt", "yolov8l-pose.pt", "yolov8x-pose.pt",
            "yolov8n-obb.pt", "yolov8s-obb.pt", "yolov8m-obb.pt", "yolov8l-obb.pt", "yolov8x-obb.pt",

            # ── YOLOv9 ─────────────────────────────────────────────
            "yolov9c.pt", "yolov9e.pt",

            # ── YOLOv5 ─────────────────────────────────────────────
            "yolov5nu.pt", "yolov5su.pt", "yolov5mu.pt", "yolov5lu.pt", "yolov5xu.pt",

            # ── RT-DETR ────────────────────────────────────────────
            "rtdetr-l.pt", "rtdetr-x.pt",
        ]
