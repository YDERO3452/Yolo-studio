"""Image utility functions — pure helpers with no UI dependencies."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
from loguru import logger
from PIL import Image


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


def read_image_bgr(image_path: Union[str, Path]) -> Optional[np.ndarray]:
  """Read an image as BGR, supporting non-ASCII paths on Windows."""
  path = Path(image_path)
  try:
    data = path.read_bytes()
  except OSError:
    return None
  if not data:
    return None
  image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
  return image


def write_image(image_path: Union[str, Path], image: np.ndarray, *, jpeg_quality: int = 95) -> bool:
  """Write an image, supporting non-ASCII paths on Windows.

  Prefer OpenCV imencode when available; fall back to Pillow so Unicode paths
  still work when cv2.imwrite would fail.
  """
  path = Path(image_path)
  try:
    path.parent.mkdir(parents=True, exist_ok=True)
  except OSError as exc:
    logger.warning(f"Failed to create parent for {path}: {exc}")
    return False

  suffix = path.suffix.lower() or ".jpg"
  ext = suffix if suffix.startswith(".") else f".{suffix}"
  params: list[int] = []
  if ext in {".jpg", ".jpeg"}:
    params = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
  elif ext == ".png":
    params = [cv2.IMWRITE_PNG_COMPRESSION, 3]

  try:
    encoded = cv2.imencode(ext, image, params)
    if isinstance(encoded, tuple) and len(encoded) >= 2:
      ok, buf = encoded[0], encoded[1]
      if ok:
        path.write_bytes(np.asarray(buf).tobytes())
        return True
  except Exception:
    pass

  try:
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[2] >= 3:
      # OpenCV callers pass BGR; Pillow expects RGB.
      rgb = arr[:, :, :3][:, :, ::-1]
    else:
      rgb = arr
    pil_image = Image.fromarray(rgb)
    if not hasattr(pil_image, "save"):
      raise TypeError("PIL Image backend unavailable")
    save_kwargs = {}
    if ext in {".jpg", ".jpeg"}:
      save_kwargs["quality"] = int(jpeg_quality)
    pil_image.save(path, **save_kwargs)
    return path.is_file()
  except Exception as exc:
    logger.warning(f"Failed to write image {path}: {exc}")
    return False
