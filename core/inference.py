"""YOLO inference module with performance optimisations.

Key optimisations vs. plain Ultralytics predict():
  - **FP16 half-precision**: 2-3x faster on GPU (auto-disabled on CPU).
  - **Controlled inference resolution (imgsz)**: avoids sending huge frames
    (e.g. 4K) into the model; 640 is the YOLO sweet-spot.
  - **Model warmup**: runs a dummy forward pass after loading so CUDA
    kernels are compiled *before* the first real frame.
  - **Explicit device passing**: ensures GPU is used on every predict call.
"""

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from loguru import logger

from core.detection_parser import parse_results


class YOLOInference:
    """Manages YOLO model inference for images, videos, and webcam."""

    def __init__(self, config=None):
        self.config = config
        self.model = None
        self.is_running = False
        self._device: str = ""       # '' = auto, '0' = GPU, 'cpu'
        self._half: bool = True      # FP16 on GPU
        self._imgsz: int = 640       # inference resolution

    # ------------------------------------------------------------------
    # Model loading + warmup
    # ------------------------------------------------------------------

    def load_model(self, model_path: str, device: str = ""):
        """Load a YOLO model for inference.

        Args:
            model_path: Path to the model file.
            device: Device string ('0' for GPU, 'cpu' for CPU, '' for auto).
                    NOTE: Ultralytics YOLO does NOT honor model.to(device).
                    The device must be passed via predict(device=...) each call.
        """
        from ultralytics import YOLO
        self.model = YOLO(model_path)

        # Auto-detect device if not specified
        if not device:
            from core.gpu import get_device
            device = get_device()
        self._device = device

        # Read performance settings from config
        if self.config:
            ic = getattr(self.config, "inference", None)
            if ic:
                self._half = getattr(ic, "half", True)
                self._imgsz = getattr(ic, "imgsz", 640)

        # FP16 only works on CUDA — force-disable on CPU
        if self._device == "cpu":
            self._half = False
            logger.info("CPU device detected — FP16 disabled (not supported on CPU)")

        logger.info(
            f"Loaded model: {model_path}  device={self._device}  "
            f"half={self._half}  imgsz={self._imgsz}"
        )

        # ---- Warmup: run a dummy forward pass ----
        self._warmup()

        return self.model

    def _warmup(self):
        """Run a single dummy inference to pre-compile CUDA kernels.

        Without warmup the very first frame can take 3-10 seconds on GPU
        because PyTorch/CUDA lazily initialises kernels.
        """
        if self.model is None:
            return
        try:
            dummy = np.zeros((self._imgsz, self._imgsz, 3), dtype=np.uint8)
            t0 = time.perf_counter()
            self.model.predict(
                source=dummy,
                device=self._device or None,
                half=self._half,
                imgsz=self._imgsz,
                verbose=False,
            )
            elapsed = time.perf_counter() - t0
            logger.info(f"Model warmup done in {elapsed:.2f}s")
        except Exception as e:
            logger.warning(f"Model warmup failed (non-fatal): {e}")

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------

    def _inject_perf_args(self, args: dict) -> dict:
        """Inject performance-critical args if caller didn't override them."""
        if "device" not in args and self._device:
            args["device"] = self._device
        if "half" not in args:
            args["half"] = self._half
        if "imgsz" not in args:
            args["imgsz"] = self._imgsz
        return args

    def predict_image(self, image_path: str, **kwargs) -> dict:
        """Run inference on a single image."""
        if self.model is None:
            raise ValueError("No model loaded")

        args = self._build_predict_args(**kwargs)
        args = self._inject_perf_args(args)

        start_time = time.time()
        results = self.model.predict(source=image_path, **args)
        elapsed = time.time() - start_time

        return {
            "success": True,
            "results": results,
            "elapsed": elapsed,
            "image_path": image_path,
        }

    def predict_frame(self, frame: np.ndarray, **kwargs) -> dict:
        """Run inference on a single frame (numpy array).

        Performance notes:
        - The frame is sent as-is; Ultralytics handles resizing to ``imgsz``
          internally, which is faster than a manual cv2.resize round-trip.
        - ``half=True`` (FP16) gives ~2-3x speedup on GPU.
        - ``verbose=False`` avoids per-frame logging overhead.
        """
        if self.model is None:
            raise ValueError("No model loaded")

        args = self._build_predict_args(**kwargs)
        args = self._inject_perf_args(args)

        results = self.model.predict(source=frame, **args)

        return {
            "success": True,
            "results": results,
        }

    def predict_video(self, video_path: str, output_path: Optional[str] = None, **kwargs) -> dict:
        """Run inference on a video file."""
        if self.model is None:
            raise ValueError("No model loaded")

        args = self._build_predict_args(**kwargs)
        args = self._inject_perf_args(args)

        if output_path:
            args["project"] = str(Path(output_path).parent)
            args["name"] = Path(output_path).stem

        results = self.model.predict(source=video_path, **args)
        return {"success": True, "results": results}

    def predict_webcam(self, camera_id: int = 0, callback=None, **kwargs):
        """Run inference on webcam feed."""
        if self.model is None:
            raise ValueError("No model loaded")

        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"Cannot open camera {camera_id}")

        self.is_running = True
        args = self._build_predict_args(**kwargs)
        args = self._inject_perf_args(args)

        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    break

                results = self.model.predict(source=frame, **args)

                if callback:
                    annotated = results[0].plot().copy()
                    callback(annotated, results[0])
                else:
                    logger.warning(
                        "predict_webcam called without callback; "
                        "skipping display to avoid blocking the PyQt event loop. "
                        "Use a callback to receive frames, or call "
                        "gui.inference_panel for a built-in webcam UI."
                    )
        finally:
            cap.release()
            self.is_running = False

    def stop(self):
        """Stop webcam inference."""
        self.is_running = False

    def _build_predict_args(self, **kwargs) -> dict:
        """Build prediction arguments from config and overrides.

        Field names now match Ultralytics model.predict() API exactly.
        """
        args = {}

        if self.config:
            ic = self.config.inference
            # Dump all fields; filter out None values (Ultralytics errors on
            # explicit None — it uses None as "use default internally").
            dumped = ic.model_dump(exclude_none=False)
            for key, value in dumped.items():
                if value is not None:
                    args[key] = value

        args.update(kwargs)
        return args

    # ------------------------------------------------------------------
    # Detection parsing (supports detect / OBB / pose)
    # ------------------------------------------------------------------

    def get_detections(self, results) -> list[dict]:
        """Extract detection results into a structured format.

        Supports all YOLO task types:
        - Detect:  result.boxes  → bbox
        - OBB:     result.obb   → obb (rotated box with 4 corners)
        - Pose:    result.boxes + result.keypoints → keypoint
        """
        return parse_results(results)

    def annotate_frame(self, frame: np.ndarray, results, show_labels: bool = True, show_conf: bool = True) -> np.ndarray:
        """Annotate a frame with detection results."""
        annotated = results[0].plot(
            labels=show_labels,
            conf=show_conf,
        ).copy()
        return annotated

    # ------------------------------------------------------------------
    # Device / performance info
    # ------------------------------------------------------------------

    def get_device_info(self) -> dict:
        """Return current device and performance info for UI display."""
        from core.gpu import detect_cuda
        detection = detect_cuda()
        info = {
            "device": self._device or "auto",
            "half": self._half,
            "imgsz": self._imgsz,
        }
        if detection.cuda_available and detection.gpus:
            gpu = detection.gpus[0]
            info["gpu_name"] = gpu.name
            info["gpu_memory"] = f"{gpu.vram_total_mb / 1024:.1f} GB" if gpu.vram_total_mb else "N/A"
            info["cuda_version"] = detection.cuda_version or "N/A"
        elif detection.torch_version:
            info["gpu_name"] = "N/A (CPU only)"
        else:
            info["gpu_name"] = "N/A (torch not installed)"
        return info
