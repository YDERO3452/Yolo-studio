"""Shared detection result parser for Ultralytics YOLO results.

Extracted from ModelManager and YOLOInference to eliminate code duplication.
Supports detect, segment (masks→polygon), OBB, and pose.
"""

from core.geometry_utils import obb_xywhr_to_corners


def _mask_points(masks, index: int) -> list[tuple[float, float]]:
  """Extract polygon points from Ultralytics masks.xy for one instance."""
  xy = getattr(masks, "xy", None)
  if xy is None:
    return []
  try:
    poly = xy[index]
  except (IndexError, TypeError, KeyError):
    return []
  if poly is None:
    return []
  try:
    arr = poly.cpu().numpy() if hasattr(poly, "cpu") else poly
  except Exception:
    arr = poly
  points: list[tuple[float, float]] = []
  try:
    for pt in arr:
      if len(pt) >= 2:
        points.append((float(pt[0]), float(pt[1])))
  except TypeError:
    return []
  return points if len(points) >= 3 else []


def parse_results(results) -> list[dict]:
  """Extract detection results from Ultralytics YOLO results into a structured format.

  Supports:
  - Detect:  result.boxes  → bbox
  - Segment: result.masks + boxes → polygon
  - OBB:     result.obb   → obb (rotated box with 4 corners)
  - Pose:    result.boxes + result.keypoints → keypoint
  """
  detections = []
  for result in results:
    names = getattr(result, "names", {}) or {}

    # --- OBB (Oriented Bounding Box) ---
    obb = getattr(result, "obb", None)
    if obb is not None and len(obb) > 0:
      for i in range(len(obb)):
        cls_id = int(obb.cls[i])
        det = {
          "class_id": cls_id,
          "class_name": names.get(cls_id, str(cls_id)),
          "confidence": float(obb.conf[i]),
          "type": "obb",
        }
        # xywhr format: [cx, cy, w, h, rotation]
        if obb.xywhr is not None and len(obb.xywhr[i]) >= 5:
          cx, cy, w, h, r = obb.xywhr[i].cpu().numpy().tolist()[:5]
          det["corners"] = obb_xywhr_to_corners(cx, cy, w, h, r)
          # Also provide xyxy bbox for compatibility
          if obb.xyxy is not None:
            det["bbox"] = {
              "x1": float(obb.xyxy[i][0]),
              "y1": float(obb.xyxy[i][1]),
              "x2": float(obb.xyxy[i][2]),
              "y2": float(obb.xyxy[i][3]),
            }
        detections.append(det)
      continue  # OBB results are handled, skip boxes

    # --- Keypoint / Pose ---
    keypoints = getattr(result, "keypoints", None)
    boxes = result.boxes
    if keypoints is not None and len(keypoints) > 0 and boxes is not None:
      for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(box.cls[0])
        det = {
          "class_id": cls_id,
          "class_name": names.get(cls_id, str(cls_id)),
          "confidence": float(box.conf[0]),
          "type": "keypoint",
          "bbox": {
            "x1": float(box.xyxy[0][0]),
            "y1": float(box.xyxy[0][1]),
            "x2": float(box.xyxy[0][2]),
            "y2": float(box.xyxy[0][3]),
          },
        }
        # Extract keypoint data
        if hasattr(keypoints, "xy") and keypoints.xy is not None:
          kps = keypoints.xy[i].cpu().numpy()  # shape: (num_keypoints, 2)
          vis = None
          if hasattr(keypoints, "visible") and keypoints.visible is not None:
            vis = keypoints.visible[i].cpu().numpy()  # shape: (num_keypoints,)
          det["keypoints"] = []
          for ki in range(len(kps)):
            kx, ky = float(kps[ki][0]), float(kps[ki][1])
            v = int(vis[ki]) if vis is not None and ki < len(vis) else 2
            det["keypoints"].append((kx, ky, v))
        detections.append(det)
      continue  # Keypoint results are handled

    # --- Segment (masks → polygon) ---
    masks = getattr(result, "masks", None)
    if masks is not None and len(masks) > 0 and boxes is not None and len(boxes) > 0:
      for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(box.cls[0])
        points = _mask_points(masks, i)
        det = {
          "class_id": cls_id,
          "class_name": names.get(cls_id, str(cls_id)),
          "confidence": float(box.conf[0]),
          "bbox": {
            "x1": float(box.xyxy[0][0]),
            "y1": float(box.xyxy[0][1]),
            "x2": float(box.xyxy[0][2]),
            "y2": float(box.xyxy[0][3]),
          },
        }
        if points:
          det["type"] = "polygon"
          det["points"] = points
        else:
          det["type"] = "bbox"
        detections.append(det)
      continue

    # --- Standard Detect (bbox) ---
    if boxes is not None:
      for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(box.cls[0])
        detections.append({
          "class_id": cls_id,
          "class_name": names.get(cls_id, str(cls_id)),
          "confidence": float(box.conf[0]),
          "type": "bbox",
          "bbox": {
            "x1": float(box.xyxy[0][0]),
            "y1": float(box.xyxy[0][1]),
            "x2": float(box.xyxy[0][2]),
            "y2": float(box.xyxy[0][3]),
          },
        })
  return detections
