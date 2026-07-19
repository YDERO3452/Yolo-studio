"""Tests for core/format_converter.py — pure data transformation logic."""

import xml.etree.ElementTree as ET

import pytest

from core.format_converter import Detection, FormatConverter


@pytest.fixture
def converter():
    return FormatConverter(["person", "car", "bicycle"])


@pytest.fixture
def sample_detections():
    return [
        Detection(class_id=0, class_name="person", x1=100, y1=100, x2=200, y2=300),
        Detection(class_id=1, class_name="car", x1=400, y1=200, x2=600, y2=400),
    ]


# ---------------------------------------------------------------------------
# YOLO format
# ---------------------------------------------------------------------------

class TestYOLOFormat:
    def test_detections_to_yolo_zero_size_returns_empty(self, converter, sample_detections):
        assert converter.detections_to_yolo(sample_detections, 0, 600) == ""
        assert converter.detections_to_yolo(sample_detections, 800, 0) == ""

    def test_convert_folder_nested_train_val(self, converter, tmp_path, monkeypatch):
        """Labels under labels/train|val must convert with matching images."""
        images = tmp_path / "images"
        labels = tmp_path / "labels"
        (images / "train").mkdir(parents=True)
        (images / "val").mkdir(parents=True)
        (labels / "train").mkdir(parents=True)
        (labels / "val").mkdir(parents=True)
        (images / "train" / "a.jpg").write_bytes(b"fake")
        (images / "val" / "b.jpg").write_bytes(b"fake")
        (labels / "train" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        (labels / "val" / "b.txt").write_text("1 0.4 0.4 0.1 0.1\n", encoding="utf-8")

        class _FakeImage:
            width = 64
            height = 48

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        import sys
        from types import SimpleNamespace

        fake_pil = SimpleNamespace(Image=SimpleNamespace(open=lambda *a, **k: _FakeImage()))
        monkeypatch.setitem(sys.modules, "PIL", fake_pil)
        monkeypatch.setitem(sys.modules, "PIL.Image", fake_pil.Image)

        out = tmp_path / "voc_out"
        results = converter.convert_folder(
            str(labels), str(out), "yolo", "voc", image_dir=str(images)
        )
        assert len(results) == 2
        assert all(r.success for r in results)
        assert (out / "train" / "a.xml").is_file()
        assert (out / "val" / "b.xml").is_file()

    def test_detections_to_yolo(self, converter, sample_detections):
        out = converter.detections_to_yolo(sample_detections, 800, 600)
        lines = out.strip().split("\n")
        assert len(lines) == 2
        assert lines[0].startswith("0 ")
        assert lines[1].startswith("1 ")

    def test_yolo_roundtrip(self, converter, tmp_path):
        """detections → YOLO string → write file → read → detections."""
        dets = [Detection(class_id=0, class_name="person", x1=200, y1=150, x2=400, y2=300)]
        yolo_str = converter.detections_to_yolo(dets, 800, 600)

        yolo_file = tmp_path / "test.txt"
        yolo_file.write_text(yolo_str, encoding="utf-8")

        loaded = converter.yolo_to_detections(str(yolo_file), 800, 600)
        assert len(loaded) == 1
        assert loaded[0].class_id == 0
        assert loaded[0].class_name == "person"
        assert abs(loaded[0].x1 - 200) < 1
        assert abs(loaded[0].y1 - 150) < 1

    def test_yolo_to_detections_missing_file(self, converter):
        dets = converter.yolo_to_detections("/nonexistent/yolo.txt", 640, 480)
        assert dets == []

    def test_detections_to_yolo_empty(self, converter):
        assert converter.detections_to_yolo([], 800, 600) == ""


# ---------------------------------------------------------------------------
# VOC format
# ---------------------------------------------------------------------------

class TestVOCFormat:
    def test_detections_to_voc(self, converter, sample_detections):
        xml_str = converter.detections_to_voc(sample_detections, "images/img.jpg", 800, 600)
        root = ET.fromstring(xml_str)
        assert root.find("filename").text == "img.jpg"
        assert root.find("size/width").text == "800"
        objects = root.findall("object")
        assert len(objects) == 2
        assert objects[0].find("name").text == "person"

    def test_voc_roundtrip(self, converter, tmp_path):
        dets = [Detection(class_id=1, class_name="car", x1=50, y1=30, x2=150, y2=130)]
        xml_str = converter.detections_to_voc(dets, "photo.jpg", 640, 480)

        voc_file = tmp_path / "test.xml"
        voc_file.write_text(xml_str, encoding="utf-8")

        # VOC → detections uses int() so x/y will be truncated
        loaded = converter.voc_to_detections(str(voc_file))
        assert len(loaded) == 1
        assert loaded[0].class_name == "car"

    def test_voc_to_detections_missing_file(self, converter):
        dets = converter.voc_to_detections("/nonexistent/voc.xml")
        assert dets == []


# ---------------------------------------------------------------------------
# DOTA format
# ---------------------------------------------------------------------------

class TestDOTAFormat:
    def test_detections_to_dota(self, converter, sample_detections):
        out = converter.detections_to_dota(sample_detections)
        lines = out.strip().split("\n")
        assert len(lines) == 2
        # DOTA line: x1 y1 x2 y1 x2 y2 x1 y2 class_name difficulty
        assert lines[0].endswith(" 0")
        assert "person" in lines[0]

    def test_dota_roundtrip(self, converter, tmp_path):
        dota_str = "100 100 200 100 200 200 100 200 car 0"
        dota_file = tmp_path / "test.txt"
        dota_file.write_text(dota_str, encoding="utf-8")

        dets = converter.dota_to_detections(str(dota_file))
        assert len(dets) == 1
        assert dets[0].class_name == "car"
        # Bounding box should enclose all 4 corners
        assert abs(dets[0].x1 - 100) < 1
        assert abs(dets[0].x2 - 200) < 1

    def test_dota_to_detections_missing_file(self, converter):
        dets = converter.dota_to_detections("/nonexistent/dota.txt")
        assert dets == []


# ---------------------------------------------------------------------------
# Detection dataclass
# ---------------------------------------------------------------------------

class TestDetection:
    def test_default_confidence(self):
        det = Detection(class_id=0, class_name="person", x1=10, y1=20, x2=30, y2=40)
        assert det.confidence == 1.0

    def test_explicit_confidence(self):
        det = Detection(class_id=0, class_name="car", x1=0, y1=0, x2=100, y2=100, confidence=0.85)
        assert det.confidence == 0.85
