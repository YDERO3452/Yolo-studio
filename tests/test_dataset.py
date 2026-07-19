"""Tests for core/dataset.py — DatasetManager."""

from pathlib import Path

import yaml

from core.dataset import DatasetManager


class TestDatasetManagerInit:
    """Tests for DatasetManager initialization."""

    def test_init_sets_paths(self, tmp_path):
        dm = DatasetManager(str(tmp_path))
        assert dm.root == tmp_path
        assert dm.images_dir == tmp_path / "images"
        assert dm.labels_dir == tmp_path / "labels"


class TestAutoLabelsPath:
    """Tests for _auto_labels_path static method."""

    def test_images_subdir_convention(self, tmp_path):
        """When images/ exists alongside labels/, detect labels/."""
        images_dir = tmp_path / "images" / "train"
        images_dir.mkdir(parents=True)
        labels_dir = tmp_path / "labels" / "train"
        labels_dir.mkdir(parents=True)
        result = DatasetManager._auto_labels_path(images_dir)
        assert result == labels_dir

    def test_sibling_labels_dir(self, tmp_path):
        """When images_dir is named 'images', replace with 'labels'."""
        images_dir = tmp_path / "images"
        images_dir.mkdir(parents=True)
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir(parents=True)
        result = DatasetManager._auto_labels_path(images_dir)
        assert result == labels_dir

    def test_fallback_to_self(self, tmp_path):
        """When no labels dir exists and no 'images' segment, return images_path."""
        images_dir = tmp_path / "my_images"
        images_dir.mkdir(parents=True)
        result = DatasetManager._auto_labels_path(images_dir)
        assert result == images_dir


class TestIsYoloStandardLayout:
    """Tests for _is_yolo_standard_layout static method."""

    def test_images_dir_is_standard(self, tmp_path):
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        assert DatasetManager._is_yolo_standard_layout(images_dir) is True

    def test_inside_images_subdir(self, tmp_path):
        images_dir = tmp_path / "images" / "train"
        images_dir.mkdir(parents=True)
        assert DatasetManager._is_yolo_standard_layout(images_dir) is True

    def test_flat_folder_not_standard(self, tmp_path):
        images_dir = tmp_path / "my_photos"
        images_dir.mkdir()
        assert DatasetManager._is_yolo_standard_layout(images_dir) is False


class TestBuildDataYaml:
    """Tests for build_data_yaml static method."""

    def test_basic_yaml_generation(self, tmp_path):
        """Generate data.yaml with train/val split."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()

        # Create test images
        for i in range(10):
            (images_dir / f"img{i}.jpg").touch()
            (labels_dir / f"img{i}.txt").write_text("0 0.5 0.5 0.2 0.3\n", encoding="utf-8")

        output_yaml = str(tmp_path / "data.yaml")
        result = DatasetManager.build_data_yaml(
            images_dir=str(images_dir),
            labels_dir=str(labels_dir),
            classes=["person"],
            output_yaml=output_yaml,
        )

        assert Path(result).exists()
        with open(result, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["nc"] == 1
        assert data["names"] == {0: "person"}
        assert "train" in data
        assert "val" in data

    def test_single_image_keeps_nonempty_train_and_val(self, tmp_path):
        """One image must not produce an empty train split after truncation."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        (images_dir / "only.jpg").write_bytes(b"x")
        (labels_dir / "only.txt").write_text("0 0.5 0.5 0.2 0.3\n", encoding="utf-8")

        result = DatasetManager.build_data_yaml(
            images_dir=str(images_dir),
            labels_dir=str(labels_dir),
            classes=["obj"],
            output_yaml=str(tmp_path / "data.yaml"),
            train_ratio=0.8,
            val_ratio=0.2,
            test_ratio=0.0,
        )
        assert Path(result).exists()
        train_imgs = list((images_dir / "train").glob("*"))
        val_imgs = list((images_dir / "val").glob("*"))
        assert len(train_imgs) == 1
        assert len(val_imgs) == 1
        assert (labels_dir / "train" / "only.txt").exists()
        assert (labels_dir / "val" / "only.txt").exists()

    def test_nested_video_frames_labels_follow_images(self, tmp_path):
        """Labels under labels/video_frames/ move with images/video_frames/."""
        images_dir = tmp_path / "images"
        frames = images_dir / "video_frames"
        frames.mkdir(parents=True)
        labels_dir = tmp_path / "labels" / "video_frames"
        labels_dir.mkdir(parents=True)
        (frames / "f0.jpg").write_bytes(b"x")
        (frames / "f1.jpg").write_bytes(b"x")
        (labels_dir / "f0.txt").write_text("0 0.5 0.5 0.2 0.3\n", encoding="utf-8")
        (labels_dir / "f1.txt").write_text("0 0.4 0.4 0.1 0.1\n", encoding="utf-8")

        DatasetManager.build_data_yaml(
            images_dir=str(images_dir),
            labels_dir=str(tmp_path / "labels"),
            classes=["obj"],
            output_yaml=str(tmp_path / "data.yaml"),
            train_ratio=0.5,
            val_ratio=0.5,
            test_ratio=0.0,
        )
        moved_labels = list((tmp_path / "labels").rglob("*.txt"))
        assert len(moved_labels) == 2
        assert all(p.parent.name in {"train", "val"} for p in moved_labels)


class TestCreateYoloDataset:
    """Tests for create_yolo_dataset instance method."""

    def test_creates_directory_structure(self, tmp_path):
        dm = DatasetManager(str(tmp_path))
        path = dm.create_yolo_dataset("test_ds", ["person", "car"])
        ds_path = Path(path)
        assert (ds_path / "images" / "train").exists()
        assert (ds_path / "images" / "val").exists()
        assert (ds_path / "labels" / "train").exists()
        assert (ds_path / "labels" / "val").exists()
        assert (ds_path / "data.yaml").exists()

    def test_data_yaml_content(self, tmp_path):
        dm = DatasetManager(str(tmp_path))
        path = dm.create_yolo_dataset("test_ds", ["person", "car"])
        with open(Path(path) / "data.yaml", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["nc"] == 2
        assert data["names"] == {0: "person", 1: "car"}

    def test_with_test_split(self, tmp_path):
        dm = DatasetManager(str(tmp_path))
        path = dm.create_yolo_dataset("test_ds", ["obj"], test_ratio=0.1)
        assert (Path(path) / "images" / "test").exists()

    def test_without_test_split(self, tmp_path):
        dm = DatasetManager(str(tmp_path))
        path = dm.create_yolo_dataset("test_ds", ["obj"], test_ratio=0)
        assert not (Path(path) / "images" / "test").exists()


class TestValidateDataset:
    """Tests for validate_dataset instance method."""

    def test_valid_dataset(self, tmp_path):
        """A properly structured dataset passes validation."""
        dm = DatasetManager(str(tmp_path))
        ds_path = Path(dm.create_yolo_dataset("test_ds", ["person", "car"]))

        # Add images and labels
        img_dir = ds_path / "images" / "train"
        lbl_dir = ds_path / "labels" / "train"
        for i in range(3):
            (img_dir / f"img{i}.jpg").touch()
            (lbl_dir / f"img{i}.txt").write_text("0 0.5 0.5 0.2 0.3\n", encoding="utf-8")

        issues = dm.validate_dataset(str(ds_path))
        # Should have no critical issues
        assert not any("data.yaml not found" in i for i in issues)

    def test_missing_data_yaml(self, tmp_path):
        """Missing data.yaml is flagged."""
        dm = DatasetManager(str(tmp_path))
        issues = dm.validate_dataset(str(tmp_path))
        assert any("data.yaml not found" in i for i in issues)

    def test_nc_mismatch(self, tmp_path):
        """nc != len(names) is flagged."""
        dm = DatasetManager(str(tmp_path))
        ds_path = Path(dm.create_yolo_dataset("test_ds", ["person"]))
        # Corrupt data.yaml
        with open(ds_path / "data.yaml", "w") as f:
            yaml.dump({"nc": 3, "names": {0: "person"}, "train": "images/train", "val": "images/val"}, f)
        issues = dm.validate_dataset(str(ds_path))
        assert any("nc" in i for i in issues)


class TestGetDatasetInfo:
    """Tests for get_dataset_info instance method."""

    def test_returns_info(self, tmp_path):
        dm = DatasetManager(str(tmp_path))
        ds_path = Path(dm.create_yolo_dataset("test_ds", ["person", "car"]))
        info = dm.get_dataset_info(str(ds_path))
        assert info["nc"] == 2
        assert "splits" in info

    def test_info_with_train_images(self, tmp_path):
        dm = DatasetManager(str(tmp_path))
        ds_path = Path(dm.create_yolo_dataset("test_ds", ["obj"]))
        img_dir = ds_path / "images" / "train"
        (img_dir / "a.jpg").touch()
        (img_dir / "b.jpg").touch()
        info = dm.get_dataset_info(str(ds_path))
        assert info["splits"]["train"]["images"] == 2


class TestDetectClassesFromLabels:
    """Tests for _detect_classes_from_labels static method."""

    def test_detects_class_ids(self, tmp_path):
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()

        (images_dir / "a.jpg").touch()
        (images_dir / "b.jpg").touch()
        (labels_dir / "a.txt").write_text("0 0.5 0.5 0.2 0.3\n1 0.1 0.1 0.3 0.3\n", encoding="utf-8")
        (labels_dir / "b.txt").write_text("0 0.3 0.4 0.4 0.5\n", encoding="utf-8")

        result = DatasetManager._detect_classes_from_labels(images_dir, labels_dir)
        assert len(result) == 2  # class 0 and class 1

    def test_no_labels_returns_default(self, tmp_path):
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        result = DatasetManager._detect_classes_from_labels(images_dir, tmp_path / "labels")
        assert result == ["目标"]
