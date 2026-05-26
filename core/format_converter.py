"""Format converter for annotation data — supports VOC, COCO, DOTA, YOLO formats.

Architecture overview:
- Bidirectional format conversion
- Batch processing with progress tracking
- Format validation and error handling
- Support for multiple annotation formats
"""

import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from loguru import logger
import time

import numpy as np


@dataclass
class Detection:
    """Standard detection object."""
    class_id: int
    class_name: str
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0


@dataclass
class ConversionResult:
    """Result of format conversion."""
    input_file: str
    output_file: str
    success: bool
    detections_count: int = 0
    error_message: Optional[str] = None
    conversion_time: float = 0.0


class FormatConverter:
    """Convert between different annotation formats.

    Architecture pattern: unified multi-format conversion workflow.
    - Supports multiple formats (YOLO, VOC, COCO, DOTA)
    - Bidirectional conversion
    - Batch processing with progress tracking
    - Format validation
    """

    SUPPORTED_FORMATS = ['yolo', 'voc', 'coco', 'dota']
    FORMAT_EXTENSIONS = {
        'yolo': 'txt',
        'voc': 'xml',
        'coco': 'json',
        'dota': 'txt',
    }

    def __init__(self, class_names: List[str]):
        """Initialize FormatConverter.

        Args:
            class_names: List of class names
        """
        self.class_names = class_names
        self.class_id_map = {name: idx for idx, name in enumerate(class_names)}
        logger.info(f"FormatConverter initialized with {len(class_names)} classes")

    # =========================================================================
    # YOLO Format (txt)
    # =========================================================================

    def yolo_to_detections(self, yolo_file: str, image_width: int, image_height: int) -> List[Detection]:
        """Convert YOLO format to detections.

        Args:
            yolo_file: Path to YOLO annotation file
            image_width: Image width
            image_height: Image height

        Returns:
            List of Detection objects
        """
        detections = []

        if not os.path.exists(yolo_file):
            logger.warning(f"YOLO file not found: {yolo_file}")
            return detections

        try:
            with open(yolo_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    if len(parts) < 5:
                        continue

                    class_id = int(parts[0])
                    x_center = float(parts[1]) * image_width
                    y_center = float(parts[2]) * image_height
                    width = float(parts[3]) * image_width
                    height = float(parts[4]) * image_height

                    x1 = x_center - width / 2
                    y1 = y_center - height / 2
                    x2 = x_center + width / 2
                    y2 = y_center + height / 2

                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else str(class_id)

                    detections.append(Detection(
                        class_id=class_id,
                        class_name=class_name,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                    ))

            logger.info(f"Loaded {len(detections)} detections from YOLO format")
            return detections

        except Exception as e:
            logger.error(f"Failed to load YOLO format: {e}")
            return detections

    def detections_to_yolo(self, detections: List[Detection], image_width: int, image_height: int) -> str:
        """Convert detections to YOLO format.

        Args:
            detections: List of Detection objects
            image_width: Image width
            image_height: Image height

        Returns:
            YOLO format string
        """
        lines = []

        for det in detections:
            x_center = (det.x1 + det.x2) / 2 / image_width
            y_center = (det.y1 + det.y2) / 2 / image_height
            width = (det.x2 - det.x1) / image_width
            height = (det.y2 - det.y1) / image_height

            line = f"{det.class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            lines.append(line)

        return "\n".join(lines)

    # =========================================================================
    # VOC Format (XML)
    # =========================================================================

    def voc_to_detections(self, voc_file: str) -> List[Detection]:
        """Convert VOC format to detections.

        Args:
            voc_file: Path to VOC annotation file

        Returns:
            List of Detection objects
        """
        detections = []

        if not os.path.exists(voc_file):
            logger.warning(f"VOC file not found: {voc_file}")
            return detections

        try:
            tree = ET.parse(voc_file)
            root = tree.getroot()

            for obj in root.findall('object'):
                class_name = obj.find('name').text
                bbox = obj.find('bndbox')

                x1 = float(bbox.find('xmin').text)
                y1 = float(bbox.find('ymin').text)
                x2 = float(bbox.find('xmax').text)
                y2 = float(bbox.find('ymax').text)

                class_id = self.class_id_map.get(class_name, 0)

                detections.append(Detection(
                    class_id=class_id,
                    class_name=class_name,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                ))

            logger.info(f"Loaded {len(detections)} detections from VOC format")
            return detections

        except Exception as e:
            logger.error(f"Failed to load VOC format: {e}")
            return detections

    def detections_to_voc(self, detections: List[Detection], image_path: str, image_width: int, image_height: int) -> str:
        """Convert detections to VOC format (XML).

        Args:
            detections: List of Detection objects
            image_path: Path to image file
            image_width: Image width
            image_height: Image height

        Returns:
            VOC format XML string
        """
        root = ET.Element('annotation')

        # Add image info
        filename = ET.SubElement(root, 'filename')
        filename.text = os.path.basename(image_path)

        size = ET.SubElement(root, 'size')
        width = ET.SubElement(size, 'width')
        width.text = str(image_width)
        height = ET.SubElement(size, 'height')
        height.text = str(image_height)
        depth = ET.SubElement(size, 'depth')
        depth.text = '3'

        # Add objects
        for det in detections:
            obj = ET.SubElement(root, 'object')

            name = ET.SubElement(obj, 'name')
            name.text = det.class_name

            bndbox = ET.SubElement(obj, 'bndbox')
            xmin = ET.SubElement(bndbox, 'xmin')
            xmin.text = str(int(det.x1))
            ymin = ET.SubElement(bndbox, 'ymin')
            ymin.text = str(int(det.y1))
            xmax = ET.SubElement(bndbox, 'xmax')
            xmax.text = str(int(det.x2))
            ymax = ET.SubElement(bndbox, 'ymax')
            ymax.text = str(int(det.y2))

        return ET.tostring(root, encoding='unicode')

    # =========================================================================
    # COCO Format (JSON)
    # =========================================================================

    def coco_to_detections(self, coco_file: str, image_id: int) -> List[Detection]:
        """Convert COCO format to detections.

        Args:
            coco_file: Path to COCO annotation file
            image_id: Image ID in COCO format

        Returns:
            List of Detection objects
        """
        detections = []

        if not os.path.exists(coco_file):
            logger.warning(f"COCO file not found: {coco_file}")
            return detections

        try:
            with open(coco_file, 'r', encoding='utf-8') as f:
                coco_data = json.load(f)

            # Find annotations for this image
            for ann in coco_data.get('annotations', []):
                if ann['image_id'] != image_id:
                    continue

                category_id = ann['category_id']
                bbox = ann['bbox']  # [x, y, width, height]

                x1 = bbox[0]
                y1 = bbox[1]
                x2 = bbox[0] + bbox[2]
                y2 = bbox[1] + bbox[3]

                # Find class name
                class_name = str(category_id)
                for cat in coco_data.get('categories', []):
                    if cat['id'] == category_id:
                        class_name = cat['name']
                        break

                detections.append(Detection(
                    class_id=category_id,
                    class_name=class_name,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                ))

            logger.info(f"Loaded {len(detections)} detections from COCO format")
            return detections

        except Exception as e:
            logger.error(f"Failed to load COCO format: {e}")
            return detections

    def detections_to_coco(self, detections: List[Detection], image_id: int, image_width: int, image_height: int) -> Dict[str, Any]:
        """Convert detections to COCO format.

        Args:
            detections: List of Detection objects
            image_id: Image ID
            image_width: Image width
            image_height: Image height

        Returns:
            COCO format dictionary
        """
        coco_data = {
            'images': [{
                'id': image_id,
                'file_name': f'image_{image_id}.jpg',
                'width': image_width,
                'height': image_height,
            }],
            'annotations': [],
            'categories': [
                {'id': idx, 'name': name} for idx, name in enumerate(self.class_names)
            ]
        }

        for i, det in enumerate(detections):
            ann = {
                'id': i,
                'image_id': image_id,
                'category_id': det.class_id,
                'bbox': [det.x1, det.y1, det.x2 - det.x1, det.y2 - det.y1],
                'area': (det.x2 - det.x1) * (det.y2 - det.y1),
                'iscrowd': 0,
            }
            coco_data['annotations'].append(ann)

        return coco_data

    # =========================================================================
    # DOTA Format (TXT)
    # =========================================================================

    def dota_to_detections(self, dota_file: str) -> List[Detection]:
        """Convert DOTA format to detections (approximate as axis-aligned boxes).

        Args:
            dota_file: Path to DOTA annotation file

        Returns:
            List of Detection objects
        """
        detections = []

        if not os.path.exists(dota_file):
            logger.warning(f"DOTA file not found: {dota_file}")
            return detections

        try:
            with open(dota_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    if len(parts) < 9:
                        continue

                    # DOTA format: x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty
                    coords = [float(p) for p in parts[:8]]
                    class_name = parts[8] if len(parts) > 8 else 'unknown'

                    # Convert to axis-aligned box
                    xs = [coords[i] for i in range(0, 8, 2)]
                    ys = [coords[i] for i in range(1, 8, 2)]

                    x1 = min(xs)
                    y1 = min(ys)
                    x2 = max(xs)
                    y2 = max(ys)

                    class_id = self.class_id_map.get(class_name, 0)

                    detections.append(Detection(
                        class_id=class_id,
                        class_name=class_name,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                    ))

            logger.info(f"Loaded {len(detections)} detections from DOTA format")
            return detections

        except Exception as e:
            logger.error(f"Failed to load DOTA format: {e}")
            return detections

    def detections_to_dota(self, detections: List[Detection]) -> str:
        """Convert detections to DOTA format.

        Args:
            detections: List of Detection objects

        Returns:
            DOTA format string
        """
        lines = []

        for det in detections:
            # Convert axis-aligned box to rotated box (0 rotation)
            x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2

            # Corners: top-left, top-right, bottom-right, bottom-left
            line = f"{x1} {y1} {x2} {y1} {x2} {y2} {x1} {y2} {det.class_name} 0"
            lines.append(line)

        return "\n".join(lines)

    # =========================================================================
    # Batch conversion
    # =========================================================================

    def convert_folder(
        self,
        input_dir: str,
        output_dir: str,
        input_format: str,
        output_format: str,
        image_dir: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[ConversionResult]:
        """Convert all annotations in a folder.

        Args:
            input_dir: Input directory
            output_dir: Output directory
            input_format: Input format (yolo, voc, coco, dota)
            output_format: Output format (yolo, voc, coco, dota)
            image_dir: Image directory (for getting image dimensions)
            progress_callback: Optional callback for progress updates

        Returns:
            List of ConversionResult objects
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # COCO is a single-file multi-image format, handle separately
        if input_format == 'coco':
            return self._convert_coco_input(
                input_dir, output_dir, output_format, progress_callback
            )

        if output_format == 'coco':
            return self._convert_coco_output(
                input_dir, output_dir, input_format, image_dir, progress_callback
            )

        input_files = list(Path(input_dir).glob(f"*.{self.FORMAT_EXTENSIONS.get(input_format, 'txt')}"))
        logger.info(f"Found {len(input_files)} files to convert")

        results = []

        for i, input_file in enumerate(input_files):
            if progress_callback:
                progress_callback(i, len(input_files))

            start_time = time.time()
            result = ConversionResult(
                input_file=str(input_file),
                output_file="",
                success=False,
            )

            try:
                logger.info(f"Converting {i + 1}/{len(input_files)}: {input_file.name}")

                # Resolve image file for dimensions
                img_width, img_height = 0, 0
                if input_format in ('yolo', 'dota') or output_format in ('yolo', 'voc'):
                    search_dir = Path(image_dir) if image_dir else input_dir
                    image_file = self._find_image_file_in_dir(input_file, search_dir)
                    if image_file:
                        from PIL import Image
                        img = Image.open(image_file)
                        img_width, img_height = img.width, img.height
                    elif input_format == 'yolo':
                        logger.warning(f"Image not found for {input_file}")
                        result.error_message = "Image not found"
                        results.append(result)
                        continue

                # Load detections
                if input_format == 'yolo':
                    detections = self.yolo_to_detections(str(input_file), img_width, img_height)
                elif input_format == 'voc':
                    detections = self.voc_to_detections(str(input_file))
                    if not img_width and image_file:
                        from PIL import Image as _Img
                        _im = _Img.open(image_file)
                        img_width, img_height = _im.width, _im.height
                elif input_format == 'dota':
                    detections = self.dota_to_detections(str(input_file))
                else:
                    logger.warning(f"Unsupported input format: {input_format}")
                    result.error_message = f"Unsupported format: {input_format}"
                    results.append(result)
                    continue

                # Save in output format
                output_file = Path(output_dir) / input_file.stem
                if output_format == 'yolo':
                    output_file = output_file.with_suffix('.txt')
                    content = self.detections_to_yolo(detections, img_width, img_height)
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                elif output_format == 'voc':
                    output_file = output_file.with_suffix('.xml')
                    content = self.detections_to_voc(detections, str(input_file), img_width, img_height)
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                elif output_format == 'dota':
                    output_file = output_file.with_suffix('.txt')
                    content = self.detections_to_dota(detections)
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                else:
                    logger.warning(f"Unsupported output format: {output_format}")
                    result.error_message = f"Unsupported format: {output_format}"
                    results.append(result)
                    continue

                result.output_file = str(output_file)
                result.success = True
                result.detections_count = len(detections)
                logger.info(f"Saved: {output_file}")

            except Exception as e:
                logger.error(f"Failed to convert {input_file}: {e}")
                result.error_message = str(e)

            result.conversion_time = time.time() - start_time
            results.append(result)

        if progress_callback:
            progress_callback(len(input_files), len(input_files))

        return results

    def _convert_coco_input(
        self,
        input_dir: str,
        output_dir: str,
        output_format: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[ConversionResult]:
        """Convert COCO input to other formats.

        COCO is a single JSON file containing all annotations,
        so we split it into per-image files in the target format.
        """
        results = []
        coco_files = list(Path(input_dir).glob("*.json"))
        if not coco_files:
            logger.warning("No COCO JSON files found in input directory")
            return results

        for coco_file in coco_files:
            start_time = time.time()
            result = ConversionResult(
                input_file=str(coco_file),
                output_file="",
                success=False,
            )

            try:
                with open(coco_file, 'r', encoding='utf-8') as f:
                    coco_data = json.load(f)

                images = coco_data.get('images', [])
                annotations = coco_data.get('categories', [])
                cat_map = {cat['id']: cat['name'] for cat in annotations}

                if progress_callback:
                    progress_callback(0, len(images))

                for idx, img_info in enumerate(images):
                    img_id = img_info['id']
                    img_w = img_info.get('width', 0)
                    img_h = img_info.get('height', 0)
                    file_stem = Path(img_info.get('file_name', f'image_{img_id}')).stem

                    # Gather detections for this image
                    dets = []
                    for ann in coco_data.get('annotations', []):
                        if ann['image_id'] != img_id:
                            continue
                        bbox = ann['bbox']
                        cat_id = ann['category_id']
                        cat_name = cat_map.get(cat_id, str(cat_id))
                        dets.append(Detection(
                            class_id=cat_id,
                            class_name=cat_name,
                            x1=bbox[0],
                            y1=bbox[1],
                            x2=bbox[0] + bbox[2],
                            y2=bbox[1] + bbox[3],
                        ))

                    output_file = Path(output_dir) / file_stem
                    if output_format == 'yolo':
                        output_file = output_file.with_suffix('.txt')
                        content = self.detections_to_yolo(dets, img_w, img_h)
                    elif output_format == 'voc':
                        output_file = output_file.with_suffix('.xml')
                        content = self.detections_to_voc(dets, str(output_file), img_w, img_h)
                    elif output_format == 'dota':
                        output_file = output_file.with_suffix('.txt')
                        content = self.detections_to_dota(dets)
                    else:
                        continue

                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(content)

                    if progress_callback:
                        progress_callback(idx + 1, len(images))

                result.output_file = str(output_dir)
                result.success = True
                result.detections_count = len(coco_data.get('annotations', []))
                logger.info(f"COCO conversion complete: {coco_file.name}")

            except Exception as e:
                logger.error(f"Failed to convert COCO file {coco_file}: {e}")
                result.error_message = str(e)

            result.conversion_time = time.time() - start_time
            results.append(result)

        return results

    def _convert_coco_output(
        self,
        input_dir: str,
        output_dir: str,
        input_format: str,
        image_dir: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[ConversionResult]:
        """Convert other formats to COCO output.

        Collects all per-image annotation files and merges them
        into a single COCO JSON file.
        """
        input_files = list(Path(input_dir).glob(f"*.{self.FORMAT_EXTENSIONS.get(input_format, 'txt')}"))
        logger.info(f"Found {len(input_files)} files to convert to COCO")

        results = []
        all_annotations = []
        all_images = []
        ann_id = 0

        for i, input_file in enumerate(input_files):
            if progress_callback:
                progress_callback(i, len(input_files))

            img_width, img_height = 0, 0
            image_file = None

            search_dir = Path(image_dir) if image_dir else input_dir
            image_file = self._find_image_file_in_dir(input_file, search_dir)
            if image_file:
                from PIL import Image
                img = Image.open(image_file)
                img_width, img_height = img.width, img.height

            img_id = i + 1
            all_images.append({
                'id': img_id,
                'file_name': input_file.stem + (image_file.suffix if image_file else '.jpg'),
                'width': img_width,
                'height': img_height,
            })

            try:
                if input_format == 'yolo':
                    detections = self.yolo_to_detections(str(input_file), img_width, img_height)
                elif input_format == 'voc':
                    detections = self.voc_to_detections(str(input_file))
                elif input_format == 'dota':
                    detections = self.dota_to_detections(str(input_file))
                else:
                    continue

                for det in detections:
                    all_annotations.append({
                        'id': ann_id,
                        'image_id': img_id,
                        'category_id': det.class_id,
                        'bbox': [det.x1, det.y1, det.x2 - det.x1, det.y2 - det.y1],
                        'area': (det.x2 - det.x1) * (det.y2 - det.y1),
                        'iscrowd': 0,
                    })
                    ann_id += 1

            except Exception as e:
                logger.error(f"Failed to read {input_file}: {e}")

        # Write single COCO JSON
        coco_data = {
            'images': all_images,
            'annotations': all_annotations,
            'categories': [
                {'id': idx, 'name': name} for idx, name in enumerate(self.class_names)
            ],
        }

        output_file = Path(output_dir) / 'annotations.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, indent=2, ensure_ascii=False)

        result = ConversionResult(
            input_file=str(input_dir),
            output_file=str(output_file),
            success=True,
            detections_count=ann_id,
            conversion_time=0,
        )
        results.append(result)

        if progress_callback:
            progress_callback(len(input_files), len(input_files))

        logger.info(f"COCO output written: {output_file} ({ann_id} annotations)")
        return results

    def _find_image_file_in_dir(self, annotation_file: Path, search_dir: Path) -> Optional[Path]:
        """Find corresponding image file in a specific directory."""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        for ext in image_extensions:
            image_file = search_dir / f"{annotation_file.stem}{ext}"
            if image_file.exists():
                return image_file
        return None

    def _get_extension(self, format_name: str) -> str:
        """Get file extension for format."""
        return self.FORMAT_EXTENSIONS.get(format_name, 'txt')

    def validate_format(self, file_path: str, format_name: str) -> Tuple[bool, List[str]]:
        """Validate annotation file format.

        Args:
            file_path: Path to annotation file
            format_name: Format name (yolo, voc, coco, dota)

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        if not os.path.exists(file_path):
            errors.append(f"File not found: {file_path}")
            return False, errors

        try:
            if format_name == 'yolo':
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) < 5:
                            errors.append(f"Line {line_num}: Invalid YOLO format (expected 5+ fields)")
                        try:
                            int(parts[0])  # class_id
                            for i in range(1, 5):
                                float(parts[i])  # coordinates
                        except ValueError:
                            errors.append(f"Line {line_num}: Invalid numeric values")

            elif format_name == 'voc':
                tree = ET.parse(file_path)
                root = tree.getroot()
                for obj in root.findall('object'):
                    if obj.find('name') is None:
                        errors.append("Object missing 'name' field")
                    if obj.find('bndbox') is None:
                        errors.append("Object missing 'bndbox' field")

            elif format_name == 'coco':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'annotations' not in data:
                        errors.append("Missing 'annotations' field")
                    if 'categories' not in data:
                        errors.append("Missing 'categories' field")

        except Exception as e:
            errors.append(f"Validation error: {str(e)}")

        return len(errors) == 0, errors
