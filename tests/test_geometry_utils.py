"""Tests for geometry utility functions."""

import math

from core.geometry_utils import obb_xywhr_to_corners


class TestOBBConversion:
    def test_axis_aligned_box(self):
        """OBB at 0 degrees should match axis-aligned corners."""
        corners = obb_xywhr_to_corners(0.5, 0.5, 0.2, 0.4, 0.0)
        # (cx±w/2, cy±h/2) — expect exactly 4 corners
        assert len(corners) == 4
        # Top-left: (0.4, 0.3)
        assert math.isclose(corners[0][0], 0.4, abs_tol=1e-9)
        assert math.isclose(corners[0][1], 0.3, abs_tol=1e-9)

    def test_90_degree_rotation(self):
        """OBB rotated 90 degrees swaps width and height."""
        corners = obb_xywhr_to_corners(0.5, 0.5, 0.2, 0.4, math.pi / 2)
        assert len(corners) == 4
        # After 90 deg rotation, the "top-left" is different
        # All corners should still be within [0,1] range
        for x, y in corners:
            assert 0.0 <= x <= 1.0
            assert 0.0 <= y <= 1.0

    def test_45_degree_rotation(self):
        """OBB at 45 degrees."""
        corners = obb_xywhr_to_corners(0.5, 0.5, 0.2, 0.4, math.pi / 4)
        assert len(corners) == 4
        # Four unique corners
        unique = set((round(x, 6), round(y, 6)) for x, y in corners)
        assert len(unique) == 4

    def test_zero_size_box(self):
        """Degenerate OBB with zero dimensions."""
        corners = obb_xywhr_to_corners(0.5, 0.5, 0.0, 0.0, 0.3)
        assert len(corners) == 4
        # All corners should collapse to center
        for x, y in corners:
            assert math.isclose(x, 0.5, abs_tol=1e-9)
            assert math.isclose(y, 0.5, abs_tol=1e-9)

    def test_off_center_position(self):
        """OBB at non-center position."""
        corners = obb_xywhr_to_corners(0.1, 0.9, 0.2, 0.2, 0.0)
        assert len(corners) == 4
        # Bounds: x in [0, 0.2], y in [0.8, 1.0]
        for x, y in corners:
            assert 0.0 <= x <= 0.2
            assert 0.8 <= y <= 1.0
