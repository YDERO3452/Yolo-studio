"""Shared detection result parser for Ultralytics YOLO results.

Extracted from ModelManager and YOLOInference to eliminate code duplication.
Supports all YOLO task types: detect, OBB, pose/keypoint.
"""

from core.geometry_utils import obb_xywhr_to_corners


def parse_results(results) -> list[dict]:
  """Extract detection results from Ultralytics YOLO results into a structured format.

  Supports all YOLO task types:
  - Detect:  result.boxes  → bbox
  - OBB:     result.obb   → obb (rotated box with 4 corners)
  - Pose:    result.boxes + result.keypoints → keypoint

  Args:
      results: Ultralytics YOLO results (list of Result objects).

  Returns:
      List of detection dicts, each containing:
      - class_id, class_name, confidence, type
      - bbox (for detect/keypoint), corners (for obb), keypoints (for pose)
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
