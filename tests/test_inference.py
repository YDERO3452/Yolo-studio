"""Tests for core/inference.py — YOLOInference."""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from core.inference import YOLOInference


class TestYOLOInferenceInit:
    """Tests for YOLOInference initialization."""

    def test_default_init(self):
        yi = YOLOInference()
        assert yi.model is None
        assert yi.is_running is False
        assert yi._device == ""
        assert yi._half is True
        assert yi._imgsz == 640

    def test_init_with_config(self):
        config = MagicMock()
        config.inference.half = False
        config.inference.imgsz = 1280
        yi = YOLOInference(config)
        assert yi.config is config


class TestBuildPredictArgs:
    """Tests for _build_predict_args method."""

    def test_no_config(self):
        yi = YOLOInference()
        args = yi._build_predict_args()
        assert isinstance(args, dict)

    def test_kwargs_override(self):
        yi = YOLOInference()
        args = yi._build_predict_args(conf=0.5, iou=0.7)
        assert args["conf"] == 0.5
        assert args["iou"] == 0.7


class TestInjectPerfArgs:
    """Tests for _inject_perf_args method."""

    def test_injects_device(self):
        yi = YOLOInference()
        yi._device = "0"
        args = yi._inject_perf_args({})
        assert args["device"] == "0"

    def test_injects_half(self):
        yi = YOLOInference()
        yi._half = True
        args = yi._inject_perf_args({})
        assert args["half"] is True

    def test_injects_imgsz(self):
        yi = YOLOInference()
        yi._imgsz = 640
        args = yi._inject_perf_args({})
        assert args["imgsz"] == 640

    def test_does_not_override_existing(self):
        yi = YOLOInference()
        yi._device = "0"
        yi._half = True
        args = yi._inject_perf_args({"device": "cpu", "half": False})
        assert args["device"] == "cpu"
        assert args["half"] is False

    def test_empty_device_not_injected(self):
        yi = YOLOInference()
        yi._device = ""
        args = yi._inject_perf_args({})
        assert "device" not in args


class TestGetDetections:
    """Tests for get_detections method."""

    def test_delegates_to_parse_results(self):
        yi = YOLOInference()
        mock_results = [MagicMock()]
        with patch("core.inference.parse_results", return_value=[{"type": "bbox"}]) as mock_parse:
            result = yi.get_detections(mock_results)
            mock_parse.assert_called_once_with(mock_results)
            assert result == [{"type": "bbox"}]


class TestAnnotateFrame:
    """Tests for annotate_frame method."""

    def test_annotates_frame(self):
        yi = YOLOInference()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_result = MagicMock()
        mock_result.plot.return_value = np.ones((480, 640, 3), dtype=np.uint8) * 255
        results = [mock_result]

        annotated = yi.annotate_frame(frame, results)
        assert annotated.shape == (480, 640, 3)
        mock_result.plot.assert_called_once()


class TestGetDeviceInfo:
    """Tests for get_device_info method."""

    def test_returns_basic_info(self):
        yi = YOLOInference()
        yi._device = "0"
        yi._half = True
        yi._imgsz = 640
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": mock_torch}):
            info = yi.get_device_info()
        assert info["device"] == "0"
        assert info["half"] is True
        assert info["imgsz"] == 640

    def test_auto_device_shows_auto(self):
        yi = YOLOInference()
        yi._device = ""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": mock_torch}):
            info = yi.get_device_info()
        assert info["device"] == "auto"


class TestStop:
    """Tests for stop method."""

    def test_stop_sets_running_false(self):
        yi = YOLOInference()
        yi.is_running = True
        yi.stop()
        assert yi.is_running is False


class TestWarmup:
    """Tests for _warmup method."""

    def test_warmup_no_model(self):
        """Warmup does nothing when no model is loaded."""
        yi = YOLOInference()
        yi._warmup()  # Should not raise

    def test_warmup_with_model(self):
        """Warmup calls model.predict with a dummy frame."""
        yi = YOLOInference()
        yi._imgsz = 640
        yi._device = "cpu"
        yi._half = False
        mock_model = MagicMock()
        yi.model = mock_model
        yi._warmup()
        mock_model.predict.assert_called_once()


class TestLoadModel:
    """Tests for load_model method."""

    def test_load_model_sets_device_cpu(self):
        yi = YOLOInference()
        mock_model = MagicMock()
        with patch("ultralytics.YOLO", return_value=mock_model):
            yi.load_model("model.pt", device="cpu")
        assert yi._device == "cpu"
        assert yi._half is False  # FP16 disabled on CPU
        assert yi.model is mock_model

    def test_load_model_sets_device_gpu(self):
        yi = YOLOInference()
        mock_model = MagicMock()
        with patch("ultralytics.YOLO", return_value=mock_model):
            yi.load_model("model.pt", device="0")
        assert yi._device == "0"

    def test_load_model_auto_device_cpu(self):
        yi = YOLOInference()
        mock_model = MagicMock()
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch("ultralytics.YOLO", return_value=mock_model):
            with patch.dict("sys.modules", {"torch": mock_torch}):
                yi.load_model("model.pt")
        assert yi._device == "cpu"


class TestPredictImage:
    """Tests for predict_image method."""

    def test_no_model_raises(self):
        yi = YOLOInference()
        with pytest.raises(ValueError, match="No model loaded"):
            yi.predict_image("test.jpg")

    def test_predict_image_success(self):
        yi = YOLOInference()
        yi._device = "cpu"
        yi._half = False
        yi._imgsz = 640
        mock_result = MagicMock()
        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result]
        yi.model = mock_model

        with patch("core.inference.parse_results", return_value=[]):
            result = yi.predict_image("test.jpg")
        assert result["success"] is True
        assert "elapsed" in result


class TestPredictFrame:
    """Tests for predict_frame method."""

    def test_no_model_raises(self):
        yi = YOLOInference()
        with pytest.raises(ValueError, match="No model loaded"):
            yi.predict_frame(np.zeros((100, 100, 3), dtype=np.uint8))

    def test_predict_frame_success(self):
        yi = YOLOInference()
        yi._device = "cpu"
        yi._half = False
        yi._imgsz = 640
        mock_result = MagicMock()
        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result]
        yi.model = mock_model

        with patch("core.inference.parse_results", return_value=[]):
            result = yi.predict_frame(np.zeros((100, 100, 3), dtype=np.uint8))
        assert result["success"] is True
