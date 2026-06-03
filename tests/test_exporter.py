"""Tests for core/exporter.py — ModelExporter."""

import pytest

from core.exporter import ModelExporter


class TestModelExporterInit:
    """Tests for ModelExporter initialization."""

    def test_init_stores_path(self):
        exporter = ModelExporter("model.pt")
        assert exporter.model_path == "model.pt"
        assert exporter.model is None


class TestSupportedFormats:
    """Tests for supported format methods."""

    def test_get_supported_formats(self):
        formats = ModelExporter.get_supported_formats()
        assert "onnx" in formats
        assert "torchscript" in formats
        assert "engine" in formats
        assert "coreml" in formats
        assert "openvino" in formats
        assert "tflite" in formats
        assert "paddle" in formats
        assert "ncnn" in formats

    def test_format_has_suffix(self):
        formats = ModelExporter.get_supported_formats()
        for fmt, info in formats.items():
            assert "suffix" in info, f"Format {fmt} missing suffix"
            assert "description" in info, f"Format {fmt} missing description"

    def test_onnx_suffix(self):
        formats = ModelExporter.get_supported_formats()
        assert formats["onnx"]["suffix"] == ".onnx"

    def test_torchscript_suffix(self):
        formats = ModelExporter.get_supported_formats()
        assert formats["torchscript"]["suffix"] == ".torchscript"


class TestExportValidation:
    """Tests for export format validation."""

    def test_unsupported_format_returns_error(self):
        """Exporting to unsupported format returns error dict."""
        exporter = ModelExporter("model.pt")
        # Mock the model to avoid actual loading
        exporter.model = type("MockModel", (), {"export": lambda self, **kw: "path"})()
        result = exporter.export(format="invalid_format_xyz")
        assert result["success"] is False
        assert "Unsupported format" in result["error"]

    def test_export_onnx_delegates(self):
        """export_onnx calls export with correct kwargs."""
        exporter = ModelExporter("model.pt")
        called_with = {}

        # Mock self.export to capture calls
        original_export = exporter.export

        def mock_export(**kwargs):
            called_with.update(kwargs)
            return {"success": True, "path": "output.onnx"}

        exporter.export = mock_export
        result = exporter.export_onnx(imgsz=640, simplify=True)
        assert called_with["format"] == "onnx"
        assert called_with["imgsz"] == 640
        assert called_with["simplify"] is True

    def test_export_onnx_with_opset(self):
        """export_onnx passes opset when specified."""
        exporter = ModelExporter("model.pt")
        called_with = {}

        def mock_export(**kwargs):
            called_with.update(kwargs)
            return {"success": True, "path": "output.onnx"}

        exporter.export = mock_export
        exporter.export_onnx(opset=13)
        assert called_with["opset"] == 13


class TestCheckFormatRequirements:
    """Tests for check_format_requirements static method."""

    def test_onnx_requirements(self):
        result = ModelExporter.check_format_requirements("onnx")
        assert result["format"] == "onnx"
        assert "onnx" in result["requirements"]
        assert "onnxruntime" in result["requirements"]

    def test_torchscript_no_requirements(self):
        result = ModelExporter.check_format_requirements("torchscript")
        assert result["requirements"] == []
        assert result["available"] is True

    def test_unknown_format(self):
        result = ModelExporter.check_format_requirements("unknown_fmt")
        assert result["requirements"] == []
        assert result["available"] is True
