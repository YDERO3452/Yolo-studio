"""Tests for core/video_extractor.py — VideoFrameExtractor."""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from core.video_extractor import VideoFrameExtractor


class TestVideoFrameExtractorInit:
    """Tests for VideoFrameExtractor initialization."""

    def test_init(self):
        vfe = VideoFrameExtractor()
        assert vfe.cap is None
        assert vfe.video_path is None
        assert vfe.total_frames == 0
        assert vfe.fps == 0.0
        assert vfe.width == 0
        assert vfe.height == 0
        assert vfe.duration_sec == 0.0
        assert vfe._prev_gray is None
        assert vfe._extracted_hashes == []


class TestOpen:
    """Tests for open method.

    Note: cv2 is mocked via conftest._LazyMockModule, so VideoCapture
    tests require patching at the sys.modules level. These tests verify
    the open/close logic using direct cap attribute manipulation.
    """

    def test_close_releases_cap(self):
        vfe = VideoFrameExtractor()
        mock_cap = MagicMock()
        vfe.cap = mock_cap
        vfe.video_path = "test.mp4"
        vfe.close()
        mock_cap.release.assert_called_once()
        assert vfe.cap is None

    def test_is_opened_no_cap(self):
        vfe = VideoFrameExtractor()
        assert vfe.is_opened() is False

    def test_is_opened_with_cap(self):
        vfe = VideoFrameExtractor()
        vfe.cap = MagicMock()
        vfe.cap.isOpened.return_value = True
        assert vfe.is_opened() is True


class TestClose:
    """Tests for close method."""

    def test_close_releases_cap(self):
        vfe = VideoFrameExtractor()
        mock_cap = MagicMock()
        vfe.cap = mock_cap
        vfe.video_path = "test.mp4"

        vfe.close()
        mock_cap.release.assert_called_once()
        assert vfe.cap is None
        assert vfe.video_path is None

    def test_close_when_no_cap(self):
        """Closing when no video is open doesn't raise."""
        vfe = VideoFrameExtractor()
        vfe.close()


class TestIsOpened:
    """Tests for is_opened method."""

    def test_no_cap(self):
        vfe = VideoFrameExtractor()
        assert vfe.is_opened() is False

    def test_with_cap(self):
        vfe = VideoFrameExtractor()
        vfe.cap = MagicMock()
        vfe.cap.isOpened.return_value = True
        assert vfe.is_opened() is True


class TestHistCorrelation:
    """Tests for _hist_correlation static method."""

    def test_identical_hashes(self):
        h = np.ones(34, dtype=np.float64).tobytes()
        corr = VideoFrameExtractor._hist_correlation(h, h)
        assert corr == pytest.approx(1.0)

    def test_orthogonal_hashes(self):
        a = np.array([1, 0, 0], dtype=np.float64).tobytes()
        b = np.array([0, 1, 0], dtype=np.float64).tobytes()
        corr = VideoFrameExtractor._hist_correlation(a, b)
        assert corr == pytest.approx(0.0)

    def test_different_lengths(self):
        a = np.ones(5, dtype=np.float64).tobytes()
        b = np.ones(10, dtype=np.float64).tobytes()
        corr = VideoFrameExtractor._hist_correlation(a, b)
        assert corr == 0.0

    def test_zero_norm(self):
        a = np.zeros(5, dtype=np.float64).tobytes()
        b = np.zeros(5, dtype=np.float64).tobytes()
        corr = VideoFrameExtractor._hist_correlation(a, b)
        assert corr == 1.0  # both zero → equal


class TestFormatDuration:
    """Tests for _format_duration static method."""

    def test_seconds_only(self):
        assert VideoFrameExtractor._format_duration(45) == "00:45"

    def test_minutes_and_seconds(self):
        assert VideoFrameExtractor._format_duration(125) == "02:05"

    def test_hours(self):
        assert VideoFrameExtractor._format_duration(3661) == "1:01:01"

    def test_zero(self):
        assert VideoFrameExtractor._format_duration(0) == "00:00"


class TestDetectPlatform:
    """Tests for detect_platform class method."""

    def test_douyin(self):
        assert VideoFrameExtractor.detect_platform("https://v.douyin.com/abc123/") == "抖音"

    def test_bilibili(self):
        assert VideoFrameExtractor.detect_platform("https://www.bilibili.com/video/BV123") == "B站"

    def test_youtube(self):
        assert VideoFrameExtractor.detect_platform("https://www.youtube.com/watch?v=abc") == "YouTube"

    def test_unknown(self):
        assert VideoFrameExtractor.detect_platform("https://example.com/video") == "其他"

    def test_sharing_text(self):
        """Extracts URL from sharing text."""
        text = "8.28 W@M.Wm https://v.douyin.com/abc123/ 复制此链接..."
        assert VideoFrameExtractor.detect_platform(text) == "抖音"


class TestExtractUrl:
    """Tests for extract_url static method."""

    def test_plain_url(self):
        url = "https://v.douyin.com/abc123/"
        assert VideoFrameExtractor.extract_url(url) == url

    def test_url_in_text(self):
        text = "8.28 https://v.douyin.com/abc123/ 复制此链接..."
        result = VideoFrameExtractor.extract_url(text)
        assert result.startswith("https://v.douyin.com/")

    def test_no_url(self):
        text = "just some text"
        assert VideoFrameExtractor.extract_url(text) == text


class TestGetSupportedPlatforms:
    """Tests for get_supported_platforms class method."""

    def test_returns_list(self):
        platforms = VideoFrameExtractor.get_supported_platforms()
        assert isinstance(platforms, list)
        assert "抖音" in platforms
        assert "YouTube" in platforms


class TestGetInfo:
    """Tests for get_info method."""

    def test_returns_dict(self):
        vfe = VideoFrameExtractor()
        vfe.video_path = "test.mp4"
        vfe.width = 1920
        vfe.height = 1080
        vfe.fps = 30.0
        vfe.total_frames = 900
        vfe.duration_sec = 30.0
        info = vfe.get_info()
        assert info["path"] == "test.mp4"
        assert info["width"] == 1920
        assert info["fps"] == 30.0
        assert "duration_str" in info
