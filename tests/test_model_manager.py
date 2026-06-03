"""Tests for core/model_manager.py — ModelManager."""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from core.model_manager import ModelManager


class TestModelManagerInit:
    """Tests for ModelManager initialization."""

    def test_default_models_dir(self, tmp_path):
        import os
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            mm = ModelManager()
            assert mm.models_dir == Path("models")
        finally:
            os.chdir(old_cwd)

    def test_custom_models_dir(self, tmp_path):
        mm = ModelManager(str(tmp_path / "my_models"))
        assert mm.models_dir == tmp_path / "my_models"
        assert mm.models_dir.exists()

    def test_initial_state(self, tmp_path):
        mm = ModelManager(str(tmp_path))
        assert mm.current_model is None
        assert mm.current_model_name is None
        assert mm.loaded_models == {}


class TestListAvailableModels:
    """Tests for list_available_models method."""

    def test_returns_model_list(self, tmp_path):
        mm = ModelManager(str(tmp_path))
        models = mm.list_available_models()
        assert len(models) > 0
        assert any("yolov8" in m for m in models)
        assert any("yolo11" in m for m in models)

    def test_includes_seg_pose_obb(self, tmp_path):
        mm = ModelManager(str(tmp_path))
        models = mm.list_available_models()
        assert any("-seg" in m for m in models)
        assert any("-pose" in m for m in models)
        assert any("-obb" in m for m in models)


class TestListLocalModels:
    """Tests for list_local_models method."""

    def test_finds_pt_files(self, tmp_path):
        (tmp_path / "model1.pt").touch()
        (tmp_path / "model2.onnx").touch()
        (tmp_path / "not_a_model.txt").touch()
        mm = ModelManager(str(tmp_path))
        local = mm.list_local_models()
        assert len(local) == 2

    def test_empty_dir(self, tmp_path):
        mm = ModelManager(str(tmp_path))
        local = mm.list_local_models()
        assert local == []


class TestLoadModel:
    """Tests for load_model method."""

    def test_load_success(self, tmp_path):
        mm = ModelManager(str(tmp_path))
        mock_model = MagicMock()
        with patch("core.model_manager.YOLO", return_value=mock_model):
            result = mm.load_model("model.pt", device="cpu")
        assert result is True
        assert mm.current_model is mock_model
        assert mm.current_model_name == "model.pt"

    def test_load_caches_model(self, tmp_path):
        mm = ModelManager(str(tmp_path))
        mock_model = MagicMock()
        with patch("core.model_manager.YOLO", return_value=mock_model):
            mm.load_model("model.pt", device="cpu")
            mm.load_model("model.pt", device="cpu")  # Second load should use cache
        assert "model.pt@cpu" in mm.loaded_models

    @patch("core.model_manager.ULTRALYTICS_AVAILABLE", False)
    def test_load_fails_without_ultralytics(self, tmp_path):
        mm = ModelManager(str(tmp_path))
        result = mm.load_model("model.pt")
        assert result is False


class TestUnloadModel:
    """Tests for unload_model method."""

    def test_unload_current(self, tmp_path):
        mm = ModelManager(str(tmp_path))
        mm.current_model = MagicMock()
        mm.current_model_name = "model.pt"
        mm.loaded_models["model.pt@cpu"] = mm.current_model

        mm.unload_model()
        assert mm.current_model is None
        assert mm.current_model_name is None
        assert len(mm.loaded_models) == 0


class TestPredict:
    """Tests for predict method."""

    def test_no_model_returns_none(self, tmp_path):
        mm = ModelManager(str(tmp_path))
        result = mm.predict("image.jpg")
        assert result is None

    def test_predict_returns_detections(self, tmp_path):
        mm = ModelManager(str(tmp_path))
        mock_result = MagicMock()
        mock_result.boxes = []
        mock_result.obb = None
        mock_result.keypoints = None
        mock_result.names = {0: "person"}

        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result]
        mm.current_model = mock_model
        mm._device = "cpu"

        with patch("core.model_manager.parse_results", return_value=[{"type": "bbox"}]):
            result = mm.predict("image.jpg")
        assert result is not None
        assert len(result) == 1


class TestUtilityMethods:
    """Tests for utility methods."""

    def test_is_model_loaded(self, tmp_path):
        mm = ModelManager(str(tmp_path))
        assert mm.is_model_loaded() is False
        mm.current_model = MagicMock()
        assert mm.is_model_loaded() is True

    def test_get_current_model(self, tmp_path):
        mm = ModelManager(str(tmp_path))
        assert mm.get_current_model() is None
        mock = MagicMock()
        mm.current_model = mock
        assert mm.get_current_model() is mock

    def test_get_current_model_name(self, tmp_path):
        mm = ModelManager(str(tmp_path))
        assert mm.get_current_model_name() is None
        mm.current_model_name = "model.pt"
        assert mm.get_current_model_name() == "model.pt"

    def test_get_loaded_models_count(self, tmp_path):
        mm = ModelManager(str(tmp_path))
        assert mm.get_loaded_models_count() == 0
        mm.loaded_models["key"] = MagicMock()
        assert mm.get_loaded_models_count() == 1

    def test_clear_cache(self, tmp_path):
        mm = ModelManager(str(tmp_path))
        mm.current_model = MagicMock()
        mm.current_model_name = "model.pt"
        mm.loaded_models["key"] = MagicMock()
        mm.clear_cache()
        assert mm.current_model is None
        assert len(mm.loaded_models) == 0
