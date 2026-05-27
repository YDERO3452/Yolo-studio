"""Geometry utility functions shared across the codebase."""

import math
from typing import List, Tuple


def obb_xywhr_to_corners(cx: float, cy: float, w: float, h: float,
                         r: float) -> List[Tuple[float, float]]:
    """Convert an oriented bounding box from (cx, cy, w, h, rotation) to
    four corner points.

    Args:
        cx, cy: Centre coordinates.
        w, h: Width and height of the box.
        r: Rotation angle in radians.

    Returns:
        List of four (x, y) corner tuples in order:
        top-left, top-right, bottom-right, bottom-left.
    """
    cos_a = math.cos(r)
    sin_a = math.sin(r)
    corners = []
    for dx, dy in [(-w / 2, -h / 2), (w / 2, -h / 2),
                   (w / 2, h / 2), (-w / 2, h / 2)]:
        px = cx + dx * cos_a - dy * sin_a
        py = cy + dx * sin_a + dy * cos_a
        corners.append((px, py))
    return corners
