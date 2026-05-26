"""Batch processing module for automatic annotation.

Architecture overview:
- Batch processing with progress tracking
- Model-based inference on multiple images
- Result persistence and format conversion
- Error handling and logging
"""

from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass
from loguru import logger
import json

from core.model_manager import ModelManager
from core.format_converter import FormatConverter


@dataclass
class BatchProcessingConfig:
    """Configuration for batch processing."""
    model_name: str
    input_dir: str
    output_dir: str
    conf_threshold: float = 0.25
    iou_threshold: float = 0.7          # Ultralytics NMS default=0.7
    max_detections: int = 300
    output_format: str = "yolo"  # yolo, coco, voc
    save_images: bool = False
    device: str = "0"


@dataclass
class ProcessingResult:
    """Result of processing a single image."""
    image_path: str
    success: bool
    detections: Optional[List[Dict[str, Any]]] = None
    error_message: Optional[str] = None
    processing_time: float = 0.0


class BatchProcessor:
    """Processes multiple images with automatic annotation.

    Architecture pattern: model-driven batch processing workflow.
    - Iterates through images in a directory
    - Uses ModelManager for inference
    - Converts results to specified format
    - Tracks progress and handles errors
    """

    def __init__(self, model_manager: ModelManager, class_names: List[str]):
        """Initialize BatchProcessor.

        Args:
            model_manager: ModelManager instance for inference
            class_names: List of class names
        """
        self.model_manager = model_manager
        self.class_names = class_names
        self.format_converter = FormatConverter(class_names)
        self.results: List[ProcessingResult] = []
        logger.info("BatchProcessor initialized")

    def process_directory(
        self,
        config: BatchProcessingConfig,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[ProcessingResult]:
        """Process all images in a directory.

        Args:
            config: Batch processing configuration
            progress_callback: Optional callback for progress updates (current, total)

        Returns:
            List of processing results
        """
        self.results = []

        # Load model
        if not self.model_manager.load_model(config.model_name, device=config.device):
            logger.error(f"Failed to load model: {config.model_name}")
            return self.results

        # Create output directory
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get list of images
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        input_dir = Path(config.input_dir)
        image_files = [
            f for f in input_dir.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]

        logger.info(f"Found {len(image_files)} images to process")

        # Process each image
        for idx, image_file in enumerate(image_files):
            if progress_callback:
                progress_callback(idx, len(image_files))

            result = self._process_single_image(image_file, config, output_dir)
            self.results.append(result)

            if result.success:
                logger.info(f"Processed {image_file.name}: {len(result.detections)} detections")
            else:
                logger.error(f"Failed to process {image_file.name}: {result.error_message}")

        if progress_callback:
            progress_callback(len(image_files), len(image_files))

        logger.info(f"Batch processing completed: {len(self.results)} images processed")
        return self.results

    def _process_single_image(
        self,
        image_path: Path,
        config: BatchProcessingConfig,
        output_dir: Path,
    ) -> ProcessingResult:
        """Process a single image.

        Args:
            image_path: Path to image file
            config: Batch processing configuration
            output_dir: Output directory for results

        Returns:
            ProcessingResult object
        """
        import time
        start_time = time.time()

        try:
            # Run inference
            detections = self.model_manager.predict(
                str(image_path),
                conf=config.conf_threshold,
                iou=config.iou_threshold,
                max_det=config.max_detections,
            )

            if detections is None:
                return ProcessingResult(
                    image_path=str(image_path),
                    success=False,
                    error_message="Inference returned None",
                    processing_time=time.time() - start_time,
                )

            # Save results in specified format
            self._save_results(image_path, detections, config, output_dir)

            return ProcessingResult(
                image_path=str(image_path),
                success=True,
                detections=detections,
                processing_time=time.time() - start_time,
            )

        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return ProcessingResult(
                image_path=str(image_path),
                success=False,
                error_message=str(e),
                processing_time=time.time() - start_time,
            )

    def _save_results(
        self,
        image_path: Path,
        detections: List[Dict[str, Any]],
        config: BatchProcessingConfig,
        output_dir: Path,
    ) -> None:
        """Save detection results in specified format.

        Args:
            image_path: Path to image file
            detections: List of detections
            config: Batch processing configuration
            output_dir: Output directory
        """
        output_file = output_dir / f"{image_path.stem}.txt"

        if config.output_format == "yolo":
            self._save_yolo_format(image_path, detections, output_file)
        elif config.output_format == "coco":
            self._save_coco_format(image_path, detections, output_file)
        elif config.output_format == "voc":
            self._save_voc_format(image_path, detections, output_file)
        else:
            logger.warning(f"Unknown output format: {config.output_format}")

    def _save_yolo_format(
        self,
        image_path: Path,
        detections: List[Dict[str, Any]],
        output_file: Path,
    ) -> None:
        """Save detections in YOLO format."""
        from PIL import Image
        img = Image.open(image_path)
        img_width, img_height = img.size

        with open(output_file, "w", encoding="utf-8") as f:
            for det in detections:
                bbox = det["bbox"]  # [x1, y1, x2, y2]
                x1, y1, x2, y2 = bbox

                # Convert to YOLO format (normalized center coordinates)
                x_center = ((x1 + x2) / 2) / img_width
                y_center = ((y1 + y2) / 2) / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height

                class_id = det["class_id"]
                confidence = det.get("confidence", 1.0)

                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    def _save_coco_format(
        self,
        image_path: Path,
        detections: List[Dict[str, Any]],
        output_file: Path,
    ) -> None:
        """Save detections in COCO format."""
        coco_data = {
            "image_id": hash(image_path.name) % (10 ** 8),
            "annotations": []
        }

        for det in detections:
            bbox = det["bbox"]  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1

            annotation = {
                "id": len(coco_data["annotations"]),
                "category_id": det["class_id"],
                "bbox": [x1, y1, width, height],
                "area": width * height,
                "iscrowd": 0,
                "confidence": det.get("confidence", 1.0),
            }
            coco_data["annotations"].append(annotation)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(coco_data, f, indent=2)

    def _save_voc_format(
        self,
        image_path: Path,
        detections: List[Dict[str, Any]],
        output_file: Path,
    ) -> None:
        """Save detections in VOC XML format."""
        from PIL import Image
        img = Image.open(image_path)
        img_width, img_height = img.size

        root_str = f"""<?xml version="1.0" encoding="UTF-8"?>
<annotation>
    <filename>{image_path.name}</filename>
    <size>
        <width>{img_width}</width>
        <height>{img_height}</height>
        <depth>3</depth>
    </size>
"""

        for det in detections:
            bbox = det["bbox"]  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = bbox
            class_name = self.class_names[det["class_id"]]

            root_str += f"""    <object>
        <name>{class_name}</name>
        <bndbox>
            <xmin>{int(x1)}</xmin>
            <ymin>{int(y1)}</ymin>
            <xmax>{int(x2)}</xmax>
            <ymax>{int(y2)}</ymax>
        </bndbox>
        <confidence>{det.get("confidence", 1.0):.4f}</confidence>
    </object>
"""

        root_str += "</annotation>"

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(root_str)
