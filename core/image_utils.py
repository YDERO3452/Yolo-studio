"""Image utility functions — pure helpers with no UI dependencies."""

from PyQt6.QtGui import QImageReader


def read_image_size(image_path: str) -> tuple[int, int]:
    """Read image dimensions without loading the full image into memory."""
    reader = QImageReader(image_path)
    size = reader.size()
    if size.isValid():
        return size.width(), size.height()
    return 0, 0
