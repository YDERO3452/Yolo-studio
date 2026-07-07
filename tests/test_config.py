"""Tests for core/config.py."""

import os

import pytest
import yaml

from core.config import (
    AnnotationConfig,
    AppConfig,
    AppGeneralConfig,
    ConfigManager,
    ExportConfig,
    InferenceConfig,
    TrainingConfig,
)


class TestTrainingConfig:
    """Tests for TrainingConfig model."""

    def test_defaults(self):
        cfg = TrainingConfig()
        assert cfg.model == "yolov8n.pt"
        assert cfg.epochs == 100
        assert cfg.batch == 16
        assert cfg.imgsz == 640
        assert cfg.device is None  # None = auto-detect (was "0", changed to prevent crash on no-GPU machines)
        assert cfg.workers == 8
        assert cfg.time is None

    def test_custom_values(self):
        cfg = TrainingConfig(model="yolov8s.pt", epochs=50, batch=8)
        assert cfg.model == "yolov8s.pt"
        assert cfg.epochs == 50
        assert cfg.batch == 8

    def test_model_dump_includes_non_none_only(self):
        cfg = TrainingConfig(model="yolov8n.pt", epochs=100)
        data = cfg.model_dump()
        assert "time" in data
        assert data["time"] is None

    def test_model_dump_exclude_none_filters_none(self):
        cfg = TrainingConfig(model="yolov8n.pt")
        data = cfg.model_dump(exclude_none=True)
        assert "time" not in data


class TestInferenceConfig:
    """Tests for InferenceConfig model."""

    def test_defaults(self):
        cfg = InferenceConfig()
        assert cfg.conf == 0.25
        assert cfg.iou == 0.7
        assert cfg.max_det == 300
        assert cfg.classes is None
        assert cfg.half is True
        assert cfg.imgsz == 640

    def test_custom_values(self):
        cfg = InferenceConfig(conf=0.5, imgsz=320, half=False)
        assert cfg.conf == 0.5
        assert cfg.imgsz == 320
        assert cfg.half is False


class TestAnnotationConfig:
    """Tests for AnnotationConfig model."""

    def test_default_classes(self):
        cfg = AnnotationConfig()
        assert cfg.default_classes == ["目标"]

    def test_custom_classes(self):
        cfg = AnnotationConfig(default_classes=["person", "car"])
        assert cfg.default_classes == ["person", "car"]

    def test_default_shape_type(self):
        cfg = AnnotationConfig()
        assert cfg.default_shape_type == "bbox"

    def test_num_keypoints(self):
        cfg = AnnotationConfig()
        assert cfg.num_keypoints == 17


class TestExportConfig:
    """Tests for ExportConfig model."""

    def test_defaults(self):
        cfg = ExportConfig()
        assert cfg.format == "onnx"
        assert cfg.imgsz == 640
        assert cfg.half is False
        assert cfg.simplify is True


class TestAppGeneralConfig:
    """Tests for AppGeneralConfig model."""

    def test_defaults(self):
        cfg = AppGeneralConfig()
        assert cfg.name == "YOLO Studio"
        assert cfg.theme == "dark"
        assert cfg.language == "zh"


class TestAppConfig:
    """Tests for AppConfig top-level model."""

    def test_defaults(self):
        cfg = AppConfig()
        assert isinstance(cfg.app, AppGeneralConfig)
        assert isinstance(cfg.training, TrainingConfig)
        assert isinstance(cfg.inference, InferenceConfig)
        assert isinstance(cfg.annotation, AnnotationConfig)
        assert isinstance(cfg.export, ExportConfig)

    def test_custom_sub_configs(self):
        cfg = AppConfig(
            app=AppGeneralConfig(name="Custom"),
            training=TrainingConfig(epochs=200),
        )
        assert cfg.app.name == "Custom"
        assert cfg.training.epochs == 200


class TestConfigManager:
    """Tests for ConfigManager."""

    def test_init_with_no_file(self, tmp_path):
        mgr = ConfigManager(config_path=str(tmp_path / "nonexistent.yaml"))
        assert isinstance(mgr.config, AppConfig)
        assert mgr.config.training.epochs == 100

    def test_init_with_existing_file(self, tmp_path, sample_config_yaml):
        mgr = ConfigManager(config_path=str(tmp_path / sample_config_yaml))
        assert mgr.config.training.epochs == 50

    def test_load(self, tmp_path, sample_config_yaml):
        mgr = ConfigManager(config_path=str(tmp_path / "does-not-exist.yaml"))
        mgr.load(str(tmp_path / sample_config_yaml))
        assert mgr.config.training.epochs == 50
        assert mgr.config.app.theme == "light"

    def test_save_and_reload_roundtrip(self, tmp_path):
        mgr = ConfigManager(config_path=str(tmp_path / "config.yaml"))
        mgr.config.training.epochs = 75
        mgr.save()

        mgr2 = ConfigManager(config_path=str(tmp_path / "config.yaml"))
        assert mgr2.config.training.epochs == 75

    def test_update_valid_section(self):
        mgr = ConfigManager()
        mgr.update("training", epochs=99, batch=32)
        assert mgr.config.training.epochs == 99
        assert mgr.config.training.batch == 32

    def test_update_unknown_key_is_ignored(self):
        mgr = ConfigManager()
        mgr.update("training", nonexistent_field=999)
        assert not hasattr(mgr.config.training, "nonexistent_field")

    def test_update_unknown_section_is_noop(self):
        mgr = ConfigManager()
        # Should not raise
        mgr.update("nonexistent_section", key="value")

    def test_get_existing_key(self):
        mgr = ConfigManager()
        assert mgr.get("training", "epochs") == 100

    def test_get_missing_key_returns_default(self):
        mgr = ConfigManager()
        assert mgr.get("training", "nonexistent", 42) == 42

    def test_get_missing_section_returns_default(self):
        mgr = ConfigManager()
        assert mgr.get("nonexistent_section", "key", "fallback") == "fallback"

    def test_save_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "config.yaml"
        mgr = ConfigManager(config_path=str(path))
        mgr.save()
        assert path.exists()

    def test_save_nonexistent_dir_does_not_crash(self, tmp_path):
        """save() handles the corner case where dirname is empty (e.g. 'config.yaml' in cwd)."""
        cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            mgr = ConfigManager(config_path="flat_config.yaml")
            mgr.save()
            assert (tmp_path / "flat_config.yaml").exists()
        finally:
            os.chdir(cwd)


# --- Fixtures ---

@pytest.fixture
def sample_config_yaml(tmp_path):
    """Create a minimal YAML config file."""
    data = {
        "app": {"name": "Test", "theme": "light"},
        "training": {"epochs": 50, "batch": 8},
    }
    path = tmp_path / "test_config.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f)
    return path.name
