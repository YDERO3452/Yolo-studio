"""Tests for annotation utility functions."""

from core.annotation_utils import collect_annotation_stats


class TestCollectAnnotationStats:
    def test_empty_directory(self, tmp_path):
        """Empty image directory returns zero counts."""
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        ann_dir = tmp_path / "labels"
        ann_dir.mkdir()

        total_img, total_ann, annotated, class_dist, ann_sizes, missing = \
            collect_annotation_stats(str(img_dir), str(ann_dir), ["class0"])

        assert total_img == 0
        assert total_ann == 0
        assert annotated == 0
        assert class_dist == {}
        assert ann_sizes == []
        assert missing == []

    def test_images_without_labels(self, tmp_path):
        """Images without corresponding labels are counted but not annotated."""
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        for name in ["a.jpg", "b.png"]:
            (img_dir / name).touch()
        ann_dir = tmp_path / "labels"
        ann_dir.mkdir()

        total_img, total_ann, annotated, class_dist, ann_sizes, missing = \
            collect_annotation_stats(str(img_dir), str(ann_dir), ["class0"])

        assert total_img == 2
        assert total_ann == 0
        assert annotated == 0
        assert len(missing) == 2

    def test_images_with_labels(self, tmp_path):
        """Valid labels produce correct class distribution."""
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        (img_dir / "a.jpg").touch()
        (img_dir / "b.jpg").touch()

        ann_dir = tmp_path / "labels"
        ann_dir.mkdir()
        (ann_dir / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n")
        (ann_dir / "b.txt").write_text("0 0.7 0.7 0.1 0.1\n")

        total_img, total_ann, annotated, class_dist, ann_sizes, missing = \
            collect_annotation_stats(str(img_dir), str(ann_dir), ["cat", "dog"])

        assert total_img == 2
        assert total_ann == 3
        assert annotated == 2
        assert class_dist == {"cat": 2, "dog": 1}
        assert sorted(ann_sizes) == [1, 2]

    def test_skips_invalid_class_ids(self, tmp_path):
        """Labels with out-of-range class IDs are silently skipped."""
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        (img_dir / "a.jpg").touch()

        ann_dir = tmp_path / "labels"
        ann_dir.mkdir()
        # class_id 99 exceeds len(class_names)=2
        (ann_dir / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n99 0.3 0.3 0.1 0.1\n")

        total_img, total_ann, annotated, class_dist, _, _ = \
            collect_annotation_stats(str(img_dir), str(ann_dir), ["cat", "dog"])

        assert total_ann == 2  # both lines counted (raw total)
        assert class_dist == {"cat": 1}  # but only valid class in distribution

    def test_subdirectory_images_found(self, tmp_path):
        """Images in subdirectories are found via rglob."""
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        sub = img_dir / "sub"
        sub.mkdir()
        (sub / "deep.jpg").touch()

        ann_dir = tmp_path / "labels"
        ann_dir.mkdir()

        total_img, _, _, _, _, _ = \
            collect_annotation_stats(str(img_dir), str(ann_dir), ["class0"])

        assert total_img == 1

    def test_train_val_labels_are_matched(self, tmp_path):
        """After split, labels live under labels/train|val and must be found."""
        images = tmp_path / "images"
        (images / "train").mkdir(parents=True)
        (images / "val").mkdir(parents=True)
        (images / "train" / "a.jpg").touch()
        (images / "val" / "b.jpg").touch()

        labels = tmp_path / "labels"
        (labels / "train").mkdir(parents=True)
        (labels / "val").mkdir(parents=True)
        (labels / "train" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        (labels / "val" / "b.txt").write_text("1 0.4 0.4 0.1 0.1\n", encoding="utf-8")

        total_img, total_ann, annotated, class_dist, _, missing = collect_annotation_stats(
            str(images), str(labels), ["cat", "dog"]
        )
        assert total_img == 2
        assert annotated == 2
        assert total_ann == 2
        assert missing == []
        assert class_dist == {"cat": 1, "dog": 1}
