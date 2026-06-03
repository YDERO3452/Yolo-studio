"""Tests for TrainingPanel dataset validation and helpers."""

from unittest.mock import MagicMock

import pytest

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _make_data_yaml(tmp_path, **kwargs):
    """Write a data.yaml to tmp_path and return its path."""
    import yaml

    content = dict(kwargs)
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text(yaml.dump(content), encoding="utf-8")
    return str(yaml_path)


def _make_label_file(lbl_dir, img_name, lines):
    """Create a label .txt file inside lbl_dir."""
    lbl_dir.mkdir(parents=True, exist_ok=True)
    (lbl_dir / f"{img_name}.txt").write_text("\n".join(lines), encoding="utf-8")


def _make_image(img_dir, name):
    """Create an empty image file."""
    img_dir.mkdir(parents=True, exist_ok=True)
    (img_dir / name).touch()


# -----------------------------------------------------------------------
# Test _count_images (static method)
# -----------------------------------------------------------------------

class TestCountImages:
    """Tests for TrainingPanel._count_images static method."""

    def test_empty_dir_returns_zero(self, tmp_path):
        from gui.training_panel import TrainingPanel

        empty = tmp_path / "empty"
        empty.mkdir()
        assert TrainingPanel._count_images(str(empty)) == 0

    def test_mixed_extensions_counts_only_images(self, tmp_path):
        from gui.training_panel import TrainingPanel

        d = tmp_path / "mixed"
        d.mkdir()
        (d / "a.jpg").touch()
        (d / "b.JPEG").touch()
        (d / "c.png").touch()
        (d / "d.txt").touch()
        (d / "e.bmp").touch()
        (d / "f.xml").touch()
        (d / "g.tiff").touch()
        (d / "h.webp").touch()
        assert TrainingPanel._count_images(str(d)) == 6

    def test_subdirs_counted(self, tmp_path):
        from gui.training_panel import TrainingPanel

        d = tmp_path / "root"
        sub = d / "sub"
        sub.mkdir(parents=True)
        (d / "a.jpg").touch()
        (sub / "b.jpg").touch()
        (sub / "c.png").touch()
        assert TrainingPanel._count_images(str(d)) == 3

    def test_nonexistent_path_returns_zero(self):
        from gui.training_panel import TrainingPanel

        assert TrainingPanel._count_images("/nonexistent/path_12345") == 0

    def test_none_path_returns_zero(self):
        from gui.training_panel import TrainingPanel

        assert TrainingPanel._count_images("") == 0

    def test_file_not_dir_returns_zero(self, tmp_path):
        from gui.training_panel import TrainingPanel

        f = tmp_path / "file.txt"
        f.touch()
        assert TrainingPanel._count_images(str(f)) == 0


# -----------------------------------------------------------------------
# Test _validate_dataset_before_training
# -----------------------------------------------------------------------

class TestValidateDataset:
    """Tests for TrainingPanel._validate_dataset_before_training."""

    @pytest.fixture
    def panel(self):
        """Create a mock panel that can call _validate_dataset_before_training."""
        from gui.training_panel import TrainingPanel

        mock = MagicMock(spec=TrainingPanel)
        # Bind the real static method
        mock._count_images = TrainingPanel._count_images
        # Bind the real validation method
        mock._validate_dataset_before_training = TrainingPanel._validate_dataset_before_training.__get__(
            mock, TrainingPanel
        )
        return mock

    # ── Fatal error cases ──────────────────────────────────────────

    def test_yaml_missing_val_path(self, panel, tmp_path):
        """val path missing from data.yaml → fatal error."""
        yaml_path = _make_data_yaml(tmp_path, train="images/train", nc=3)
        is_fatal, warnings, errors = panel._validate_dataset_before_training(yaml_path)
        assert is_fatal is True
        assert any("val" in e.lower() for e in errors)

    def test_val_dir_nonexistent(self, panel, tmp_path):
        """val path points nowhere → fatal error."""
        yaml_path = _make_data_yaml(tmp_path, train="images/train", val="images/val", nc=3)
        _make_image(tmp_path / "images/train", "a.jpg")
        _make_label_file(tmp_path / "labels/train", "a", ["0 0.5 0.5 0.2 0.3"])
        is_fatal, warnings, errors = panel._validate_dataset_before_training(yaml_path)
        assert is_fatal is True
        assert any("不存在" in e for e in errors)

    def test_val_no_images(self, panel, tmp_path):
        """val dir exists but empty → fatal error."""
        (tmp_path / "images/val").mkdir(parents=True)
        _make_image(tmp_path / "images/train", "a.jpg")
        _make_label_file(tmp_path / "labels/train", "a", ["0 0.5 0.5 0.2 0.3"])
        yaml_path = _make_data_yaml(tmp_path, train="images/train", val="images/val", nc=3)
        is_fatal, warnings, errors = panel._validate_dataset_before_training(yaml_path)
        assert is_fatal is True
        assert any("没有" in e for e in errors)

    def test_val_all_images_no_labels(self, panel, tmp_path):
        """All val images lack labels → fatal error."""
        _make_image(tmp_path / "images/train", "a.jpg")
        _make_label_file(tmp_path / "labels/train", "a", ["0 0.5 0.5 0.2 0.3"])
        (tmp_path / "labels/val").mkdir(parents=True)  # empty labels dir
        _make_image(tmp_path / "images/val", "img1.jpg")
        _make_image(tmp_path / "images/val", "img2.jpg")
        yaml_path = _make_data_yaml(tmp_path, train="images/train", val="images/val", nc=3)
        is_fatal, warnings, errors = panel._validate_dataset_before_training(yaml_path)
        assert is_fatal is True
        assert any("全部没有标签" in e for e in errors)

    def test_val_labels_dir_missing(self, panel, tmp_path):
        """No labels/val directory at all → fatal error."""
        _make_image(tmp_path / "images/train", "a.jpg")
        _make_label_file(tmp_path / "labels/train", "a", ["0 0.5 0.5 0.2 0.3"])
        _make_image(tmp_path / "images/val", "img1.jpg")
        yaml_path = _make_data_yaml(tmp_path, train="images/train", val="images/val", nc=3)
        is_fatal, warnings, errors = panel._validate_dataset_before_training(yaml_path)
        assert is_fatal is True
        assert any("标签目录不存在" in e for e in errors)

    def test_val_invalid_label_format(self, panel, tmp_path):
        """Labels with < 5 columns → error."""
        _make_image(tmp_path / "images/train", "a.jpg")
        _make_label_file(tmp_path / "labels/train", "a", ["0 0.5 0.5 0.2 0.3"])
        _make_image(tmp_path / "images/val", "img1.jpg")
        _make_label_file(tmp_path / "labels/val", "img1", ["0 0.5 0.5"])  # only 3 cols
        yaml_path = _make_data_yaml(tmp_path, train="images/train", val="images/val", nc=3)
        is_fatal, warnings, errors = panel._validate_dataset_before_training(yaml_path)
        assert is_fatal is True
        assert any("格式无效" in e for e in errors)

    def test_val_class_id_exceeds_nc(self, panel, tmp_path):
        """Class ID >= nc in label → error."""
        _make_image(tmp_path / "images/train", "a.jpg")
        _make_label_file(tmp_path / "labels/train", "a", ["0 0.5 0.5 0.2 0.3"])
        _make_image(tmp_path / "images/val", "img1.jpg")
        _make_label_file(tmp_path / "labels/val", "img1", ["5 0.5 0.5 0.2 0.3"])  # class_id=5, nc=3
        yaml_path = _make_data_yaml(tmp_path, train="images/train", val="images/val", nc=3)
        is_fatal, warnings, errors = panel._validate_dataset_before_training(yaml_path)
        assert is_fatal is True
        assert any("类别 ID" in e for e in errors)

    # ── Warning cases ──────────────────────────────────────────────

    def test_val_some_missing_labels_warning(self, panel, tmp_path):
        """Some val images without labels → warning, not fatal."""
        _make_image(tmp_path / "images/train", "a.jpg")
        _make_label_file(tmp_path / "labels/train", "a", ["0 0.5 0.5 0.2 0.3"])
        _make_image(tmp_path / "images/val", "img1.jpg")
        _make_image(tmp_path / "images/val", "img2.jpg")
        _make_image(tmp_path / "images/val", "img3.jpg")
        _make_label_file(tmp_path / "labels/val", "img1", ["0 0.5 0.5 0.2 0.3"])
        # img2, img3 have no labels
        yaml_path = _make_data_yaml(tmp_path, train="images/train", val="images/val", nc=3)
        is_fatal, warnings, errors = panel._validate_dataset_before_training(yaml_path)
        assert is_fatal is False
        assert errors == []
        assert any("缺少标签" in w for w in warnings)

    def test_train_small_dataset_warning(self, panel, tmp_path):
        """Train split < 10 images → warning."""
        _make_image(tmp_path / "images/val", "v1.jpg")
        _make_label_file(tmp_path / "labels/val", "v1", ["0 0.5 0.5 0.2 0.3"])
        for i in range(5):
            _make_image(tmp_path / "images/train", f"t{i}.jpg")
            _make_label_file(tmp_path / "labels/train", f"t{i}", ["0 0.5 0.5 0.2 0.3"])
        yaml_path = _make_data_yaml(tmp_path, train="images/train", val="images/val", nc=3)
        is_fatal, warnings, errors = panel._validate_dataset_before_training(yaml_path)
        assert is_fatal is False
        assert any("仅有" in w for w in warnings)

    # ── Happy path ──────────────────────────────────────────────────

    def test_fully_valid_dataset_no_issues(self, panel, tmp_path):
        """Both splits have labels → no errors, no warnings."""
        for i in range(20):
            _make_image(tmp_path / "images/train", f"t{i}.jpg")
            _make_label_file(tmp_path / "labels/train", f"t{i}", ["0 0.5 0.5 0.2 0.3"])
        for i in range(5):
            _make_image(tmp_path / "images/val", f"v{i}.jpg")
            _make_label_file(tmp_path / "labels/val", f"v{i}", ["0 0.5 0.5 0.2 0.3"])
        yaml_path = _make_data_yaml(tmp_path, train="images/train", val="images/val", nc=3)
        is_fatal, warnings, errors = panel._validate_dataset_before_training(yaml_path)
        assert is_fatal is False
        assert errors == []
        assert warnings == []

    # ── data.yaml path key ──────────────────────────────────────────

    def test_yaml_with_path_key_resolves_correctly(self, panel, tmp_path):
        """data.yaml has 'path' key → resolve relative to that."""
        dataset_root = tmp_path / "dataset"
        (dataset_root / "images/train").mkdir(parents=True)
        (dataset_root / "labels/train").mkdir(parents=True)
        (dataset_root / "images/val").mkdir(parents=True)
        (dataset_root / "labels/val").mkdir(parents=True)
        for i in range(10):
            (dataset_root / f"images/train/t{i}.jpg").touch()
            (dataset_root / f"labels/train/t{i}.txt").write_text("0 0.5 0.5 0.2 0.3")
        for i in range(3):
            (dataset_root / f"images/val/v{i}.jpg").touch()
            (dataset_root / f"labels/val/v{i}.txt").write_text("0 0.5 0.5 0.2 0.3")

        yaml_path = tmp_path / "data.yaml"
        import yaml
        yaml_path.write_text(yaml.dump({
            "path": str(dataset_root),
            "train": "images/train",
            "val": "images/val",
            "nc": 3,
        }), encoding="utf-8")

        is_fatal, warnings, errors = panel._validate_dataset_before_training(str(yaml_path))
        assert is_fatal is False
        assert errors == []

    # ── Edge cases ──────────────────────────────────────────────────

    def test_unparseable_yaml_returns_fatal(self, panel, tmp_path):
        """Corrupt yaml → fatal error."""
        yaml_path = tmp_path / "data.yaml"
        yaml_path.write_text(":: not yaml :: {{bad", encoding="utf-8")
        is_fatal, warnings, errors = panel._validate_dataset_before_training(str(yaml_path))
        assert is_fatal is True

    def test_yaml_without_train_key(self, panel, tmp_path):
        """data.yaml without 'train' → warning, val still checked."""
        _make_image(tmp_path / "images/val", "v1.jpg")
        _make_label_file(tmp_path / "labels/val", "v1", ["0 0.5 0.5 0.2 0.3"])
        yaml_path = _make_data_yaml(tmp_path, val="images/val", nc=3)
        # no 'train' key
        is_fatal, warnings, errors = panel._validate_dataset_before_training(yaml_path)
        assert is_fatal is False
        assert errors == []

    def test_segmentation_labels_pass_validation(self, panel, tmp_path):
        """Labels with > 5 columns (seg/poly) should pass."""
        _make_image(tmp_path / "images/train", "a.jpg")
        _make_label_file(tmp_path / "labels/train", "a", ["0 0.5 0.5 0.2 0.3"])
        _make_image(tmp_path / "images/val", "v1.jpg")
        # Segmentation: 11 columns (class + 5 point pairs)
        _make_label_file(tmp_path / "labels/val", "v1",
                         ["0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.1"])
        yaml_path = _make_data_yaml(tmp_path, train="images/train", val="images/val", nc=3)
        is_fatal, warnings, errors = panel._validate_dataset_before_training(yaml_path)
        assert is_fatal is False
        assert errors == []

    def test_absolute_paths_in_yaml(self, panel, tmp_path):
        """Absolute paths in data.yaml should work."""
        _make_image(tmp_path / "images/val", "v1.jpg")
        _make_label_file(tmp_path / "labels/val", "v1", ["0 0.5 0.5 0.2 0.3"])
        for i in range(20):
            _make_image(tmp_path / "images/train", f"t{i}.jpg")
            _make_label_file(tmp_path / "labels/train", f"t{i}", ["0 0.5 0.5 0.2 0.3"])

        yaml_path = _make_data_yaml(
            tmp_path,
            train=str(tmp_path / "images/train"),
            val=str(tmp_path / "images/val"),
            nc=3,
        )
        is_fatal, warnings, errors = panel._validate_dataset_before_training(yaml_path)
        assert is_fatal is False
        assert errors == []
