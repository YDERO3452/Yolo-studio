"""Tests for core/annotation.py."""

import os

import pytest

from core.annotation import (
    Annotation,
    AnnotationManager,
    BBoxAnnotation,
    KeypointAnnotation,
    OBBoxAnnotation,
    PolygonAnnotation,
    ShapeType,
)


class TestAnnotation:
    """Tests for the Annotation base class and from_yolo_line parser."""

    def test_from_yolo_bbox_line(self):
        """Parses standard bbox format: cls x_c y_c w h."""
        ann = Annotation.from_yolo_line("0 0.5 0.5 0.2 0.3")
        assert isinstance(ann, BBoxAnnotation)
        assert ann.class_id == 0
        assert ann.x_center == 0.5
        assert ann.y_center == 0.5
        assert ann.width == 0.2
        assert ann.height == 0.3

    def test_from_yolo_polygon_line(self):
        """Parses polygon line with 6 coordinates (3 points)."""
        ann = Annotation.from_yolo_line("0 0.1 0.1 0.3 0.1 0.2 0.4")
        assert isinstance(ann, PolygonAnnotation)
        assert len(ann.points) == 3
        assert ann.points[0] == (0.1, 0.1)

    def test_from_yolo_obb_line(self):
        """Parses 8-value OBB line as 4 corner points."""
        ann = Annotation.from_yolo_line("0 0.1 0.1 0.3 0.1 0.3 0.3 0.1 0.3")
        assert isinstance(ann, OBBoxAnnotation)
        assert len(ann.corners) == 4

    def test_from_yolo_keypoint_line(self):
        """Parses keypoint line with bbox + 1 keypoint."""
        ann = Annotation.from_yolo_line("0 0.5 0.5 0.2 0.3 0.5 0.5 2")
        assert isinstance(ann, KeypointAnnotation)
        assert ann.class_id == 0
        assert len(ann.keypoints) == 1
        assert ann.keypoints[0] == (0.5, 0.5, 2)

    def test_from_yolo_keypoint_line_multiple_kps(self):
        """Parses keypoint line with bbox + 3 keypoints."""
        ann = Annotation.from_yolo_line("0 0.5 0.5 0.2 0.3 0.4 0.4 1 0.5 0.5 2 0.6 0.6 2")
        assert isinstance(ann, KeypointAnnotation)
        assert len(ann.keypoints) == 3

    def test_from_yolo_short_line_raises(self):
        """Raises ValueError for line with only class_id and no coordinates."""
        with pytest.raises(ValueError):
            Annotation.from_yolo_line("0")

    def test_from_yolo_empty_line_raises(self):
        """Raises ValueError for empty line."""
        with pytest.raises(ValueError):
            Annotation.from_yolo_line("")

    def test_to_yolo_not_implemented_in_base(self):
        """Base Annotation.to_yolo raises NotImplementedError."""
        ann = Annotation(0, ShapeType.BBOX)
        with pytest.raises(NotImplementedError):
            ann.to_yolo()

    def test_to_canvas_shape_not_implemented_in_base(self):
        ann = Annotation(0, ShapeType.BBOX)
        with pytest.raises(NotImplementedError):
            ann.to_canvas_shape(640, 480)


class TestBBoxAnnotation:
    """Tests for BBoxAnnotation."""

    def test_roundtrip_from_xyxy(self):
        """Pixel → normalized → pixel roundtrip."""
        bbox = BBoxAnnotation.from_xyxy(0, 100, 100, 200, 200, 500, 500)
        assert bbox.class_id == 0
        assert pytest.approx(bbox.x_center) == 0.3
        assert bbox.width == 0.2

    def test_from_xyxy_zero_dim_raises(self):
        """Raises ValueError for zero image dimensions."""
        with pytest.raises(ValueError):
            BBoxAnnotation.from_xyxy(0, 0, 0, 10, 10, 0, 100)

    def test_to_xyxy(self):
        """Normalized → pixel conversion."""
        bbox = BBoxAnnotation(0, 0.5, 0.5, 0.2, 0.3)
        x1, y1, x2, y2 = bbox.to_xyxy(1000, 500)
        assert x1 == 400
        assert y1 == 175
        assert x2 == 600
        assert y2 == 325

    def test_to_yolo_format(self):
        """Serializes to correct YOLO string."""
        bbox = BBoxAnnotation(1, 0.5, 0.6, 0.25, 0.30)
        line = bbox.to_yolo()
        assert line == "1 0.500000 0.600000 0.250000 0.300000"

    def test_to_canvas_shape(self):
        """Returns correct dict structure."""
        bbox = BBoxAnnotation(2, 0.5, 0.5, 0.2, 0.2)
        shape = bbox.to_canvas_shape(500, 500)
        assert shape["type"] == ShapeType.BBOX
        assert shape["class_id"] == 2
        assert "data" in shape

    def test_from_yolo_classmethod(self):
        """Class method parses YOLO line directly."""
        bbox = BBoxAnnotation.from_yolo("0 0.3 0.4 0.5 0.6")
        assert bbox.class_id == 0
        assert bbox.x_center == 0.3

    def test_from_yolo_invalid_line_raises(self):
        """Raises ValueError for short line."""
        with pytest.raises(ValueError):
            BBoxAnnotation.from_yolo("0 0.3")


class TestPolygonAnnotation:
    """Tests for PolygonAnnotation."""

    def test_roundtrip_from_pixel_points(self):
        """Pixel points → norm → YOLO string roundtrip."""
        poly = PolygonAnnotation.from_pixel_points(0, [(100, 100), (200, 100), (150, 200)], 400, 300)
        assert len(poly.points) == 3
        assert pytest.approx(poly.points[0][0]) == 0.25

    def test_from_pixel_points_zero_dim_raises(self):
        """Raises ValueError for zero dims."""
        with pytest.raises(ValueError):
            PolygonAnnotation.from_pixel_points(0, [(10, 10)], 0, 100)

    def test_to_yolo(self):
        """Correct YOLO polygon format."""
        poly = PolygonAnnotation(0, [(0.1, 0.2), (0.3, 0.4), (0.5, 0.6)])
        line = poly.to_yolo()
        assert line.startswith("0 ")
        assert len(line.split()) == 7  # cls + 3*2 coords

    def test_to_canvas_shape(self):
        poly = PolygonAnnotation(1, [(0.5, 0.5), (0.6, 0.6)])
        shape = poly.to_canvas_shape(200, 200)
        assert shape["type"] == ShapeType.POLYGON
        assert len(shape["data"]["points"]) == 2


class TestOBBoxAnnotation:
    """Tests for OBBoxAnnotation."""

    def test_from_pixel_corners(self):
        obb = OBBoxAnnotation.from_pixel_corners(0, [(0, 0), (100, 0), (100, 50), (0, 50)], 200, 100)
        assert len(obb.corners) == 4
        assert pytest.approx(obb.corners[0][0]) == 0.0

    def test_from_pixel_corners_zero_dim_raises(self):
        with pytest.raises(ValueError):
            OBBoxAnnotation.from_pixel_corners(0, [(0, 0), (10, 0), (10, 10), (0, 10)], 0, 100)

    def test_to_yolo_format(self):
        obb = OBBoxAnnotation(0, [(0.1, 0.1), (0.3, 0.1), (0.3, 0.3), (0.1, 0.3)])
        line = obb.to_yolo()
        assert line.startswith("0 ")

    def test_to_canvas_shape(self):
        obb = OBBoxAnnotation(2, [(0.1, 0.1), (0.3, 0.1), (0.3, 0.3), (0.1, 0.3)])
        shape = obb.to_canvas_shape(300, 300)
        assert shape["type"] == ShapeType.OBB
        assert len(shape["data"]["corners"]) == 4


class TestKeypointAnnotation:
    """Tests for KeypointAnnotation."""

    def test_to_yolo_with_keypoints(self):
        kp = KeypointAnnotation(0, 0.5, 0.5, 0.2, 0.3, [(0.4, 0.4, 1), (0.6, 0.6, 2)])
        line = kp.to_yolo()
        parts = line.strip().split()
        assert int(parts[0]) == 0
        assert len(parts) == 11  # 1 cls + 4 bbox + 2*3=6 kp

    def test_to_yolo_no_keypoints(self):
        kp = KeypointAnnotation(0, 0.5, 0.5, 0.2, 0.3)
        line = kp.to_yolo()
        assert len(line.strip().split()) == 5

    def test_from_pixel_data(self):
        kp = KeypointAnnotation.from_pixel_data(0, 100, 100, 200, 200, 400, 300)
        assert kp.x_center > 0

    def test_from_pixel_data_zero_dim_raises(self):
        with pytest.raises(ValueError):
            KeypointAnnotation.from_pixel_data(0, 0, 0, 10, 10, 0, 100)

    def test_to_canvas_shape(self):
        kp = KeypointAnnotation(0, 0.5, 0.5, 0.2, 0.3, [(0.5, 0.4, 2)])
        shape = kp.to_canvas_shape(400, 300)
        assert shape["type"] == ShapeType.KEYPOINT
        assert len(shape["data"]["keypoints"]) == 1


class TestAnnotationManager:
    """Tests for AnnotationManager."""

    def test_default_classes(self):
        mgr = AnnotationManager()
        assert mgr.classes == ["目标"]

    def test_set_classes(self):
        mgr = AnnotationManager()
        mgr.set_classes(["person", "car"])
        assert mgr.classes == ["person", "car"]

    def test_load_annotation_valid_file(self, tmp_path):
        (tmp_path / "images").mkdir()
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        (labels_dir / "img1.txt").write_text("0 0.5 0.5 0.2 0.3\n1 0.1 0.1 0.3 0.3\n", encoding="utf-8")
        mgr = AnnotationManager(["person", "car", "bicycle"])
        anns = mgr.load_annotation(str(tmp_path / "images" / "img1.jpg"))
        assert len(anns) == 2
        assert isinstance(anns[0], BBoxAnnotation)
        assert anns[0].class_id == 0

    def test_load_annotation_empty_file(self, tmp_path):
        (tmp_path / "images").mkdir()
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        (labels_dir / "img3.txt").write_text("", encoding="utf-8")
        mgr = AnnotationManager()
        anns = mgr.load_annotation(str(tmp_path / "images" / "img3.jpg"))
        assert anns == []

    def test_load_annotation_missing_file(self):
        mgr = AnnotationManager()
        anns = mgr.load_annotation("/nonexistent/path/image.jpg")
        assert anns == []

    def test_load_annotation_skips_bad_lines(self, tmp_path):
        (tmp_path / "images").mkdir()
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        (labels_dir / "img4.txt").write_text("0 0.5 0.5 0.2 0.3\ninvalid line\n", encoding="utf-8")
        mgr = AnnotationManager(["person", "car", "bicycle"])
        anns = mgr.load_annotation(str(tmp_path / "images" / "img4.jpg"))
        # Should load the valid bbox line, skip the "invalid line"
        assert len(anns) == 1

    def test_save_and_reload_roundtrip(self, tmp_dir):
        images_dir = tmp_dir / "images"
        images_dir.mkdir(parents=True)
        img_path = str(images_dir / "test.jpg")

        mgr = AnnotationManager(["person", "car"])
        mgr.add_bbox(0, 100, 100, 200, 200, 640, 480)
        mgr.save_annotation(img_path)

        # Reload
        mgr2 = AnnotationManager(["person", "car"])
        loaded = mgr2.load_annotation(img_path)
        assert len(loaded) == 1
        assert isinstance(loaded[0], BBoxAnnotation)
        assert loaded[0].class_id == 0

    def test_add_bbox(self):
        mgr = AnnotationManager()
        mgr.add_bbox(0, 100, 100, 200, 200, 640, 480)
        assert mgr.is_modified
        assert len(mgr.current_annotations) == 1

    def test_add_polygon(self):
        mgr = AnnotationManager()
        mgr.add_polygon(0, [(100, 100), (200, 100), (150, 200)], 640, 480)
        assert len(mgr.current_annotations) == 1
        assert any(isinstance(a, PolygonAnnotation) for a in mgr.current_annotations)

    def test_add_obb(self):
        mgr = AnnotationManager()
        mgr.add_obb(0, [(100, 100), (300, 100), (300, 300), (100, 300)], 640, 480)
        assert len(mgr.current_annotations) == 1
        assert isinstance(mgr.current_annotations[0], OBBoxAnnotation)

    def test_add_keypoint(self):
        mgr = AnnotationManager()
        mgr.add_keypoint(0, 100, 100, 200, 200, 640, 480, [(150, 150, 2)])
        assert isinstance(mgr.current_annotations[0], KeypointAnnotation)

    def test_remove_by_valid_index(self):
        mgr = AnnotationManager()
        mgr.add_bbox(0, 100, 100, 200, 200, 640, 480)
        mgr.add_bbox(1, 50, 50, 150, 150, 640, 480)
        mgr.remove_annotation(0)
        assert len(mgr.current_annotations) == 1

    def test_remove_by_invalid_index(self):
        mgr = AnnotationManager()
        mgr.remove_annotation(99)  # Should not raise
        assert len(mgr.current_annotations) == 0

    def test_clear_annotations(self):
        mgr = AnnotationManager()
        mgr.add_bbox(0, 100, 100, 200, 200, 640, 480)
        mgr.clear_annotations()
        assert len(mgr.current_annotations) == 0
        assert mgr.is_modified

    def test_get_annotation_count(self):
        mgr = AnnotationManager()
        assert mgr.get_annotation_count() == 0
        mgr.add_bbox(0, 100, 100, 200, 200, 640, 480)
        assert mgr.get_annotation_count() == 1

    def test_from_yolo_polygon_not_keypoint_3points(self):
        """3-point polygon (6 coords) must NOT be misidentified as keypoint.

        With class_id=0 + 6 coords = 7 values. Old code checked keypoint
        (n>=7 and (n-4)%3==0) before polygon, so 7 values matched keypoint
        with 1 kp. After the fix polygon is checked first (n>=6 and n%2==0).
        """
        ann = Annotation.from_yolo_line("0 0.1 0.1 0.3 0.1 0.2 0.4")
        assert isinstance(ann, PolygonAnnotation), (
            f"Expected PolygonAnnotation, got {type(ann).__name__}"
        )
        assert len(ann.points) == 3

    def test_from_yolo_polygon_not_keypoint_5points(self):
        """5-point polygon (10 coords) must NOT be misidentified as keypoint.

        11 values: (11-4)%3 = 1 so old code wouldn't match keypoint either,
        but verifying the fix is robust.
        """
        ann = Annotation.from_yolo_line(
            "0 0.1 0.1 0.2 0.1 0.3 0.1 0.4 0.1 0.5 0.1"
        )
        assert isinstance(ann, PolygonAnnotation), (
            f"Expected PolygonAnnotation, got {type(ann).__name__}"
        )
        assert len(ann.points) == 5

    def test_from_yolo_keypoint_not_polygon(self):
        """Keypoint with 1 kp (bbox+3 = 7 values) must not be polygon.

        With class_id=0+bbox(4)+kp(3) = 8 values. Old code would check
        polygon first if it saw n>=6, but keypoint has exactly 8 values
        which is n%2==0. After fix: polygon checked first triggers on
        any n>=6 with n%2==0, so 8 values would be polygon. But actually
        8 values WITH (n-4)%3!=0 means it could be either 4-point polygon
        or 1-kp keypoint. Let's check actual behavior.
        """
        ann = Annotation.from_yolo_line("0 0.5 0.5 0.2 0.3 0.5 0.5 2")
        # 8 values: class(1) + bbox(4) + kp(3) = keypoint with 1 kp
        # or class(1) + 7 coords — but 7 is odd, not a polygon.
        # Since 7 coords is odd, polygon check (n>=6 and n%2==0) fails,
        # so it falls through to keypoint check.
        assert isinstance(ann, KeypointAnnotation), (
            f"Expected KeypointAnnotation, got {type(ann).__name__}"
        )

    def test_get_label_path_yolo_standard_structure(self):
        """labels dir mirrors images dir: images/train/x.jpg → labels/train/x.txt."""
        mgr = AnnotationManager()
        path = mgr._get_label_path("/data/images/train/img001.jpg")
        assert path.replace("\\", "/").endswith("labels/train/img001.txt"), (
            f"Unexpected label path: {path}"
        )

    def test_get_label_path_no_images(self):
        mgr = AnnotationManager()
        path = mgr._get_label_path("/data/pics/img001.jpg")
        assert "labels" in path
        assert path.endswith(".txt")

    def test_clear_boxes_alias(self):
        mgr = AnnotationManager()
        mgr.add_bbox(0, 100, 100, 200, 200, 640, 480)
        mgr.clear_boxes()
        assert len(mgr.current_annotations) == 0

    def test_get_annotation_stats(self, tmp_dir):
        # Set up a sample dataset structure
        images_dir = tmp_dir / "images"
        images_dir.mkdir(parents=True)
        labels_dir = tmp_dir / "labels"
        labels_dir.mkdir(parents=True)
        (images_dir / "img1.jpg").touch()
        (images_dir / "img2.jpg").touch()
        (images_dir / "img3.jpg").touch()
        (labels_dir / "img1.txt").write_text("0 0.5 0.5 0.2 0.3\n", encoding="utf-8")
        (labels_dir / "img2.txt").write_text("0 0.1 0.1 0.1 0.1\n1 0.2 0.2 0.2 0.2\n", encoding="utf-8")

        mgr = AnnotationManager(["person", "car"])
        stats = mgr.get_annotation_stats(str(tmp_dir))
        assert stats["total_images"] == 3
        assert stats["annotated_images"] == 2
        assert stats["total_annotations"] == 3
