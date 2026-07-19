"""Tests for batch_processor bbox normalization."""

from pathlib import Path

from core.batch_processor import BatchProcessor


def test_bbox_xyxy_accepts_dict_and_list():
    assert BatchProcessor._bbox_xyxy({"bbox": {"x1": 1, "y1": 2, "x2": 3, "y2": 4}}) == (1.0, 2.0, 3.0, 4.0)
    assert BatchProcessor._bbox_xyxy({"bbox": [10, 20, 30, 40]}) == (10.0, 20.0, 30.0, 40.0)
    assert BatchProcessor._bbox_xyxy({"bbox": None}) is None
    assert BatchProcessor._bbox_xyxy({}) is None


def test_save_yolo_format_with_dict_bbox(tmp_path, monkeypatch):
    img_path = tmp_path / "a.jpg"
    img_path.write_bytes(b"fake")
    out = tmp_path / "a.txt"

    monkeypatch.setattr(
        "core.image_utils.read_image_size",
        lambda _path: (100, 80),
    )

    processor = BatchProcessor(model_manager=None, class_names=["person"])
    processor._save_yolo_format(
        img_path,
        [{"class_id": 0, "bbox": {"x1": 10, "y1": 10, "x2": 50, "y2": 50}}],
        out,
    )
    text = out.read_text(encoding="utf-8").strip()
    assert text.startswith("0 ")
    parts = text.split()
    assert len(parts) == 5
