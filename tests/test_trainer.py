"""Tests for core/trainer.py — pure logic only, no GPU/hardware required."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from core.config import TrainingConfig
from core.trainer import YOLOTrainer

pandas_available = False
try:
    import pandas as pd
    pandas_available = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Fixtures: inject mock torch so _apply_windows_memory_fixes works without torch
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_torch():
    """Inject mock torch into sys.modules so import torch works in function body."""
    mock_t = MagicMock()
    mock_mp = MagicMock()
    mock_t.multiprocessing = mock_mp

    saved_t = sys.modules.get("torch")
    saved_mp = sys.modules.get("torch.multiprocessing")
    sys.modules["torch"] = mock_t
    sys.modules["torch.multiprocessing"] = mock_mp
    yield mock_t
    for name, saved in [("torch", saved_t), ("torch.multiprocessing", saved_mp)]:
        if saved is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = saved


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_state(self):
        trainer = YOLOTrainer()
        assert trainer.config is None
        assert trainer.model is None
        assert trainer.is_training is False
        assert trainer.training_thread is None
        assert trainer.callbacks == {}

    def test_with_config(self):
        cfg = MagicMock()
        trainer = YOLOTrainer(config=cfg)
        assert trainer.config is cfg

    def test_is_training_mutable(self):
        trainer = YOLOTrainer()
        trainer.is_training = True
        assert trainer.is_training is True


# ---------------------------------------------------------------------------
# stop_training
# ---------------------------------------------------------------------------

class TestStopTraining:
    def test_sets_flag_false(self):
        trainer = YOLOTrainer()
        trainer.is_training = True
        trainer.stop_training()
        assert trainer.is_training is False

    def test_idempotent(self):
        trainer = YOLOTrainer()
        trainer.stop_training()
        trainer.stop_training()
        assert trainer.is_training is False


# ---------------------------------------------------------------------------
# _build_train_args
# ---------------------------------------------------------------------------

class TestBuildTrainArgs:
    def test_no_config_kwargs_only(self):
        trainer = YOLOTrainer()
        args = trainer._build_train_args(
            "data.yaml", epochs=50, batch=8, imgsz=320, device="cpu"
        )
        assert args["data"] == "data.yaml"
        assert args["epochs"] == 50
        assert args["batch"] == 8
        assert args["imgsz"] == 320
        assert args["device"] == "cpu"

    def test_config_merged_kwargs_override(self):
        cfg = MagicMock()
        tc = TrainingConfig(epochs=100, batch=16, imgsz=640)
        cfg.training = tc
        trainer = YOLOTrainer(config=cfg)
        args = trainer._build_train_args("data.yaml", epochs=200, imgsz=512)
        assert args["epochs"] == 200
        assert args["imgsz"] == 512
        assert args["batch"] == 16
        assert args["data"] == "data.yaml"

    def test_none_values_filtered(self):
        cfg = MagicMock()
        tc = TrainingConfig(name=None, freeze=None, epochs=100)
        cfg.training = tc
        trainer = YOLOTrainer(config=cfg)
        args = trainer._build_train_args("data.yaml")
        assert "name" not in args
        assert "freeze" not in args
        assert "epochs" in args

    def test_string_freeze_passed_through(self):
        cfg = MagicMock()
        tc = TrainingConfig(freeze="10", epochs=1)
        cfg.training = tc
        trainer = YOLOTrainer(config=cfg)
        args = trainer._build_train_args("data.yaml")
        assert args["freeze"] == "10"

    def test_zero_values_not_filtered(self):
        cfg = MagicMock()
        tc = TrainingConfig(seed=0, epochs=100)
        cfg.training = tc
        trainer = YOLOTrainer(config=cfg)
        args = trainer._build_train_args("data.yaml")
        assert args["seed"] == 0

    def test_false_values_not_filtered(self):
        cfg = MagicMock()
        tc = TrainingConfig(amp=False, epochs=100)
        cfg.training = tc
        trainer = YOLOTrainer(config=cfg)
        args = trainer._build_train_args("data.yaml")
        assert args["amp"] is False

    def test_float_values(self):
        cfg = MagicMock()
        tc = TrainingConfig(lr0=0.001, momentum=0.9, epochs=100)
        cfg.training = tc
        trainer = YOLOTrainer(config=cfg)
        args = trainer._build_train_args("data.yaml")
        assert args["lr0"] == 0.001
        assert args["momentum"] == 0.9

    def test_default_config_values(self):
        cfg = MagicMock()
        tc = TrainingConfig()
        cfg.training = tc
        trainer = YOLOTrainer(config=cfg)
        args = trainer._build_train_args("data.yaml")
        assert args["model"] == "yolov8n.pt"
        assert args["epochs"] == 100
        assert args["batch"] == 16
        assert args["imgsz"] == 640
        assert "device" not in args or args["device"] is None  # None = auto-detect (excluded from args)
        assert args["workers"] == 8 or (sys.platform == "win32" and args["workers"] == 4)  # Windows caps workers at 4


# ---------------------------------------------------------------------------
# _apply_windows_memory_fixes
# ---------------------------------------------------------------------------

class TestApplyWindowsMemoryFixes:
    def test_linux_does_nothing(self, mock_torch):
        args = {"workers": 16, "batch": 32}
        with patch.object(sys, "platform", "linux"):
            YOLOTrainer._apply_windows_memory_fixes(args)
        assert args["workers"] == 16

    def test_darwin_does_nothing(self, mock_torch):
        args = {"workers": 16, "batch": 32}
        with patch.object(sys, "platform", "darwin"):
            YOLOTrainer._apply_windows_memory_fixes(args)
        assert args["workers"] == 16

    def test_windows_caps_workers_at_4(self, mock_torch):
        args = {"workers": 8, "batch": 32}
        with patch.object(sys, "platform", "win32"):
            YOLOTrainer._apply_windows_memory_fixes(args)
        assert args["workers"] == 4

    def test_windows_leaves_low_workers_alone(self, mock_torch):
        args = {"workers": 2, "batch": 32}
        with patch.object(sys, "platform", "win32"):
            YOLOTrainer._apply_windows_memory_fixes(args)
        assert args["workers"] == 2

    def test_windows_exactly_at_cap(self, mock_torch):
        args = {"workers": 4, "batch": 32}
        with patch.object(sys, "platform", "win32"):
            YOLOTrainer._apply_windows_memory_fixes(args)
        assert args["workers"] == 4

    def test_windows_default_workers_8(self, mock_torch):
        args = {"workers": 8}
        with patch.object(sys, "platform", "win32"):
            YOLOTrainer._apply_windows_memory_fixes(args)
        assert args["workers"] == 4

    def test_windows_no_workers_key_defaults_to_cap(self, mock_torch):
        """When workers not in args, default=8 exceeds cap, so workers=4 is set."""
        args = {"batch": 32}
        with patch.object(sys, "platform", "win32"):
            YOLOTrainer._apply_windows_memory_fixes(args)
        assert args["workers"] == 4

    def test_windows_calls_set_sharing_strategy(self, mock_torch):
        args = {"workers": 8}
        with patch.object(sys, "platform", "win32"):
            YOLOTrainer._apply_windows_memory_fixes(args)
        mock_torch.multiprocessing.set_sharing_strategy.assert_called_once_with("file_system")

    def test_windows_sharing_strategy_failure_graceful(self, mock_torch):
        mock_torch.multiprocessing.set_sharing_strategy.side_effect = RuntimeError("boom")
        args = {"workers": 8}
        with patch.object(sys, "platform", "win32"):
            YOLOTrainer._apply_windows_memory_fixes(args)
        assert args["workers"] == 4


# ---------------------------------------------------------------------------
# list_pretrained_models
# ---------------------------------------------------------------------------

class TestListPretrainedModels:
    def test_returns_list(self):
        models = YOLOTrainer.list_pretrained_models()
        assert isinstance(models, list)
        assert len(models) > 0

    def test_all_end_with_pt(self):
        models = YOLOTrainer.list_pretrained_models()
        for m in models:
            assert m.endswith(".pt"), f"{m} should end with .pt"

    def test_no_duplicates(self):
        models = YOLOTrainer.list_pretrained_models()
        assert len(models) == len(set(models))

    def test_includes_yolov8(self):
        models = YOLOTrainer.list_pretrained_models()
        assert "yolov8n.pt" in models
        assert "yolov8x.pt" in models

    def test_includes_yolo11(self):
        models = YOLOTrainer.list_pretrained_models()
        assert "yolo11n.pt" in models
        assert "yolo11x.pt" in models

    def test_includes_yolo12(self):
        models = YOLOTrainer.list_pretrained_models()
        assert "yolo12n.pt" in models
        assert "yolo12x.pt" in models

    def test_includes_yolo26(self):
        models = YOLOTrainer.list_pretrained_models()
        assert "yolo26n.pt" in models
        assert "yolo26x.pt" in models

    def test_includes_yolov10(self):
        models = YOLOTrainer.list_pretrained_models()
        assert "yolov10n.pt" in models
        assert "yolov10x.pt" in models

    def test_includes_yolov9(self):
        models = YOLOTrainer.list_pretrained_models()
        assert "yolov9c.pt" in models
        assert "yolov9e.pt" in models

    def test_includes_yolov5(self):
        models = YOLOTrainer.list_pretrained_models()
        assert "yolov5nu.pt" in models
        assert "yolov5xu.pt" in models

    def test_includes_rtdetr(self):
        models = YOLOTrainer.list_pretrained_models()
        assert "rtdetr-l.pt" in models
        assert "rtdetr-x.pt" in models

    def test_includes_seg_variants(self):
        models = YOLOTrainer.list_pretrained_models()
        assert "yolov8n-seg.pt" in models
        assert "yolo11n-seg.pt" in models

    def test_includes_cls_variants(self):
        models = YOLOTrainer.list_pretrained_models()
        assert "yolov8n-cls.pt" in models
        assert "yolo11n-cls.pt" in models

    def test_includes_pose_variants(self):
        models = YOLOTrainer.list_pretrained_models()
        assert "yolov8n-pose.pt" in models
        assert "yolo11n-pose.pt" in models

    def test_includes_obb_variants(self):
        models = YOLOTrainer.list_pretrained_models()
        assert "yolov8n-obb.pt" in models
        assert "yolo11n-obb.pt" in models

    def test_all_have_family_prefix(self):
        models = YOLOTrainer.list_pretrained_models()
        valid_prefixes = ("yolo", "rtdetr")
        for m in models:
            assert m.startswith(valid_prefixes), f"{m} has unexpected prefix"


# ---------------------------------------------------------------------------
# get_training_logs
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not pandas_available, reason="pandas not installed")
class TestGetTrainingLogs:
    def test_missing_csv_returns_empty(self, tmp_path):
        trainer = YOLOTrainer()
        logs = trainer.get_training_logs(str(tmp_path))
        assert logs == {}

    def test_csv_with_standard_columns(self, tmp_path):
        df = pd.DataFrame({
            "epoch": [1, 2, 3],
            "train/box_loss": [0.5, 0.4, 0.3],
            "train/cls_loss": [0.2, 0.15, 0.1],
            "train/dfl_loss": [0.8, 0.7, 0.6],
            "val/box_loss": [0.55, 0.45, 0.35],
            "val/cls_loss": [0.25, 0.2, 0.15],
            "val/dfl_loss": [0.85, 0.75, 0.65],
            "metrics/precision(B)": [0.5, 0.6, 0.7],
            "metrics/recall(B)": [0.4, 0.5, 0.6],
            "metrics/mAP50(B)": [0.45, 0.55, 0.65],
            "metrics/mAP50-95(B)": [0.3, 0.35, 0.4],
        })
        csv_path = tmp_path / "results.csv"
        df.to_csv(csv_path, index=False)

        trainer = YOLOTrainer()
        logs = trainer.get_training_logs(str(tmp_path))

        assert logs["epochs"] == [0, 1, 2]
        assert logs["train/box_loss"] == [0.5, 0.4, 0.3]
        assert logs["metrics/mAP50(B)"] == [0.45, 0.55, 0.65]
        assert logs["metrics/mAP50-95(B)"] == [0.3, 0.35, 0.4]

    def test_csv_with_whitespace_columns(self, tmp_path):
        df = pd.DataFrame({
            "  train/box_loss  ": [0.5, 0.4],
        })
        csv_path = tmp_path / "results.csv"
        df.to_csv(csv_path, index=False)

        trainer = YOLOTrainer()
        logs = trainer.get_training_logs(str(tmp_path))
        assert "train/box_loss" in logs
        assert logs["train/box_loss"] == [0.5, 0.4]

    def test_csv_missing_optional_columns(self, tmp_path):
        df = pd.DataFrame({
            "epoch": [1],
            "train/box_loss": [0.5],
            "train/cls_loss": [0.1],
        })
        csv_path = tmp_path / "results.csv"
        df.to_csv(csv_path, index=False)

        trainer = YOLOTrainer()
        logs = trainer.get_training_logs(str(tmp_path))
        assert logs["train/dfl_loss"] == []
        assert logs["val/box_loss"] == []
        assert logs["metrics/precision(B)"] == []


# ---------------------------------------------------------------------------
# train_async
# ---------------------------------------------------------------------------

class TestTrainAsync:
    def test_returns_thread(self):
        trainer = YOLOTrainer()
        with patch.object(trainer, "train", return_value={"success": True}):
            thread = trainer.train_async("data.yaml", epochs=1)
            thread.join(timeout=2)
            assert not thread.is_alive()

    def test_callback_called(self):
        results = {}
        trainer = YOLOTrainer()

        def cb(result):
            results["data"] = result

        with patch.object(trainer, "train", return_value={"success": True}):
            thread = trainer.train_async("data.yaml", callback=cb, epochs=1)
            thread.join(timeout=2)

        assert results["data"] == {"success": True}
