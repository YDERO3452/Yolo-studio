"""Image utility functions — pure helpers with no UI dependencies."""


from PIL import Image
from loguru import logger


def read_image_size(image_path: str) -> tuple[int, int]:
  """Read image dimensions without loading the full image into memory.

  Uses PIL (Pillow) to read only the image header, avoiding the need
  for PyQt6 or loading pixel data.
  """
  try:
    with Image.open(image_path) as img:
      return img.size
  except Exception:
    logger.debug(f"Failed to read image size for: {image_path}")
    return 0, 0
