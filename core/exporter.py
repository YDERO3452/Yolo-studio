"""Model export module."""

from typing import Optional

from loguru import logger


class ModelExporter:
    """Exports YOLO models to various formats."""

    SUPPORTED_FORMATS = {
        "torchscript": {"suffix": ".torchscript", "description": "TorchScript"},
        "onnx": {"suffix": ".onnx", "description": "ONNX"},
        "openvino": {"suffix": "_openvino_model/", "description": "OpenVINO"},
        "engine": {"suffix": ".engine", "description": "TensorRT"},
        "coreml": {"suffix": ".mlpackage", "description": "CoreML"},
        "saved_model": {"suffix": "_saved_model/", "description": "TensorFlow SavedModel"},
        "pb": {"suffix": ".pb", "description": "TensorFlow GraphDef"},
        "tflite": {"suffix": ".tflite", "description": "TensorFlow Lite"},
        "edgetpu": {"suffix": "_edgetpu.tflite", "description": "TF Lite Edge TPU"},
        "tfjs": {"suffix": "_web_model/", "description": "TensorFlow.js"},
        "paddle": {"suffix": "_paddle_model/", "description": "PaddlePaddle"},
        "ncnn": {"suffix": "_ncnn_model/", "description": "NCNN"},
    }

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None

    def load_model(self):
        """Load the model for export."""
        from ultralytics import YOLO
        self.model = YOLO(self.model_path)
        logger.info(f"Loaded model for export: {self.model_path}")

    def export(self, format: str = "onnx", **kwargs) -> dict:
        """Export model to specified format."""
        if self.model is None:
            self.load_model()

        if format not in self.SUPPORTED_FORMATS:
            return {"success": False, "error": f"Unsupported format: {format}"}

        try:
            export_path = self.model.export(format=format, **kwargs)
            logger.info(f"Model exported to {format}: {export_path}")
            return {
                "success": True,
                "format": format,
                "path": str(export_path),
            }
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return {"success": False, "error": str(e)}

    def export_onnx(self, imgsz: int = 640, simplify: bool = True, opset: Optional[int] = None) -> dict:
        """Export model to ONNX format."""
        kwargs = {"format": "onnx", "imgsz": imgsz, "simplify": simplify}
        if opset is not None:
            kwargs["opset"] = opset
        return self.export(**kwargs)

    @staticmethod
    def get_supported_formats() -> dict:
        """Get list of supported export formats."""
        return ModelExporter.SUPPORTED_FORMATS

    @staticmethod
    def check_format_requirements(format: str) -> dict:
        """Check if requirements for a format are met."""
        requirements = {
            "onnx": ["onnx", "onnxruntime"],
            "torchscript": [],
            "engine": ["tensorrt"],
            "coreml": ["coremltools"],
            "openvino": ["openvino"],
            "tflite": ["tensorflow"],
            "paddle": ["paddlepaddle"],
            "ncnn": ["ncnn"],
        }

        reqs = requirements.get(format, [])
        missing = []
        for req in reqs:
            try:
                __import__(req)
            except ImportError:
                missing.append(req)

        return {
            "format": format,
            "requirements": reqs,
            "missing": missing,
            "available": len(missing) == 0,
        }
