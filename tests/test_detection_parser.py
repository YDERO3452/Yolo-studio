"""Tests for core/detection_parser.py — parse_results."""

from unittest.mock import MagicMock

import numpy as np

from core.detection_parser import parse_results


def _make_mock_box(cls_id, conf, x1, y1, x2, y2):
    """Create a mock box object."""
    box = MagicMock()
    box.cls = [cls_id]
    box.conf = [conf]
    xyxy_mock = MagicMock()
    xyxy_mock.__getitem__ = lambda self, i: [x1, y1, x2, y2][i]
    box.xyxy = [xyxy_mock]
    return box


def _make_tensor(values):
    """Create a mock tensor that supports .cpu().numpy().tolist()."""
    arr = np.array(values)
    mock_tensor = MagicMock()
    mock_tensor.cpu.return_value = mock_tensor
    mock_tensor.numpy.return_value = arr
    mock_tensor.tolist.return_value = arr.tolist()
    mock_tensor.__len__ = lambda self: len(arr)
    mock_tensor.__getitem__ = lambda self, i: arr[i]
    return mock_tensor


class TestParseResultsBBox:
    """Tests for standard bbox detection parsing."""

    def test_single_detection(self):
        box = _make_mock_box(0, 0.95, 100, 100, 200, 200)
        result = MagicMock()
        result.names = {0: "person"}
        result.boxes = [box]
        result.obb = None
        result.keypoints = None

        detections = parse_results([result])
        assert len(detections) == 1
        assert detections[0]["type"] == "bbox"
        assert detections[0]["class_id"] == 0
        assert detections[0]["class_name"] == "person"
        assert detections[0]["confidence"] == 0.95
        assert detections[0]["bbox"]["x1"] == 100
        assert detections[0]["bbox"]["y2"] == 200

    def test_multiple_detections(self):
        box1 = _make_mock_box(0, 0.9, 10, 10, 50, 50)
        box2 = _make_mock_box(1, 0.8, 60, 60, 100, 100)
        result = MagicMock()
        result.names = {0: "person", 1: "car"}
        result.boxes = [box1, box2]
        result.obb = None
        result.keypoints = None

        detections = parse_results([result])
        assert len(detections) == 2
        assert detections[0]["class_id"] == 0
        assert detections[1]["class_id"] == 1

    def test_empty_boxes(self):
        result = MagicMock()
        result.names = {0: "obj"}
        result.boxes = []
        result.obb = None
        result.keypoints = None
        detections = parse_results([result])
        assert detections == []

    def test_unknown_class_name(self):
        box = _make_mock_box(99, 0.5, 0, 0, 10, 10)
        result = MagicMock()
        result.names = {}
        result.boxes = [box]
        result.obb = None
        result.keypoints = None
        detections = parse_results([result])
        assert detections[0]["class_name"] == "99"


class TestParseResultsOBB:
    """Tests for OBB detection parsing."""

    def test_obb_detection(self):
        mock_obb = MagicMock()
        mock_obb.__len__ = MagicMock(return_value=1)
        mock_obb.cls = [0]
        mock_obb.conf = [0.92]
        mock_obb.xywhr = [_make_tensor([0.5, 0.5, 0.2, 0.1, 0.3])]

        xyxy_mock = MagicMock()
        xyxy_mock.__getitem__ = lambda self, i: [100, 100, 200, 200][i]
        mock_obb.xyxy = [xyxy_mock]

        result = MagicMock()
        result.names = {0: "ship"}
        result.obb = mock_obb
        result.boxes = None
        result.keypoints = None

        detections = parse_results([result])
        assert len(detections) == 1
        assert detections[0]["type"] == "obb"
        assert detections[0]["class_name"] == "ship"
        assert "corners" in detections[0]
        assert "bbox" in detections[0]


class TestParseResultsKeypoint:
    """Tests for keypoint/pose detection parsing."""

    def test_keypoint_detection(self):
        box = _make_mock_box(0, 0.88, 100, 100, 300, 300)

        kps_data = np.array([[150, 150], [200, 200], [250, 250]])
        kps_tensor = _make_tensor(kps_data)
        kps_xy = MagicMock()
        kps_xy.__getitem__ = lambda self, i: kps_tensor if i == 0 else None

        vis_data = np.array([2, 2, 0])
        vis_tensor = _make_tensor(vis_data)
        kps_vis = MagicMock()
        kps_vis.__getitem__ = lambda self, i: vis_tensor if i == 0 else None

        mock_kps = MagicMock()
        mock_kps.xy = kps_xy
        mock_kps.visible = kps_vis
        mock_kps.__len__ = MagicMock(return_value=1)

        result = MagicMock()
        result.names = {0: "person"}
        result.boxes = [box]
        result.keypoints = mock_kps
        result.obb = None

        detections = parse_results([result])
        assert len(detections) == 1
        assert detections[0]["type"] == "keypoint"
        assert len(detections[0]["keypoints"]) == 3
        assert detections[0]["keypoints"][0] == (150.0, 150.0, 2)
        assert detections[0]["keypoints"][2] == (250.0, 250.0, 0)

    def test_keypoint_default_visibility(self):
        """When visible attribute is None, default visibility is 2."""
        box = _make_mock_box(0, 0.8, 0, 0, 100, 100)

        kps_data = np.array([[50, 50]])
        kps_tensor = _make_tensor(kps_data)
        kps_xy = MagicMock()
        kps_xy.__getitem__ = lambda self, i: kps_tensor if i == 0 else None

        mock_kps = MagicMock()
        mock_kps.xy = kps_xy
        mock_kps.visible = None
        mock_kps.__len__ = MagicMock(return_value=1)

        result = MagicMock()
        result.names = {0: "person"}
        result.boxes = [box]
        result.keypoints = mock_kps
        result.obb = None

        detections = parse_results([result])
        assert detections[0]["keypoints"][0] == (50.0, 50.0, 2)


class TestParseResultsMultipleResults:
    """Tests for parsing multiple result objects."""

    def test_multiple_results(self):
        box1 = _make_mock_box(0, 0.9, 10, 10, 50, 50)
        box2 = _make_mock_box(1, 0.8, 60, 60, 100, 100)

        r1 = MagicMock()
        r1.names = {0: "person", 1: "car"}
        r1.boxes = [box1]
        r1.obb = None
        r1.keypoints = None

        r2 = MagicMock()
        r2.names = {0: "person", 1: "car"}
        r2.boxes = [box2]
        r2.obb = None
        r2.keypoints = None

        detections = parse_results([r1, r2])
        assert len(detections) == 2

    def test_empty_results_list(self):
        detections = parse_results([])
        assert detections == []
