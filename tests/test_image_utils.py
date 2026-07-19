"""Tests for core/image_utils.py."""

import numpy as np

import core.image_utils as image_utils


def test_write_image_unicode_path(tmp_path, monkeypatch):
    payload = np.frombuffer(b"\xff\xd8\xff\xd9", dtype=np.uint8)
    monkeypatch.setattr(image_utils.cv2, "imencode", lambda *args, **kwargs: (True, payload))

    path = tmp_path / "中文目录" / "sample.jpg"
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    assert image_utils.write_image(path, image) is True
    assert path.is_file()
    assert path.read_bytes().startswith(b"\xff\xd8")


def test_read_missing_returns_none(tmp_path):
    assert image_utils.read_image_bgr(tmp_path / "missing.jpg") is None
