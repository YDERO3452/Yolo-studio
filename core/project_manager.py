"""Lightweight project management for YOLO Studio.

Projects are stored as normal YOLO-compatible folders instead of a SQLite
database.  Each project keeps images, labels, models, runs and project.json
under one root directory, so it remains easy to inspect and train with
Ultralytics directly.
"""

from __future__ import annotations

import json
import re
import shutil
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np
import yaml
from loguru import logger

from core.class_manager import ClassManager
from core.dataset import DatasetManager
from core.video_extractor import VideoFrameExtractor

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}


class ProjectManager:
    """Manage project folders and import workflows."""

    def __init__(self, workspace_root: str | Path = "projects"):
        self.workspace_root = Path(workspace_root)
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.workspace_root / "projects.json"

    # ------------------------------------------------------------------
    # Project registry
    # ------------------------------------------------------------------

    def list_projects(self) -> list[dict[str, Any]]:
        projects: list[dict[str, Any]] = []
        if self.registry_path.exists():
            try:
                raw = json.loads(self.registry_path.read_text(encoding="utf-8"))
                if isinstance(raw, list):
                    projects.extend(p for p in raw if isinstance(p, dict))
            except Exception as exc:
                logger.warning(f"Failed to read project registry: {exc}")

        # Also discover project.json files so manually copied projects show up.
        known_roots = {str(Path(p.get("root", "")).resolve()) for p in projects if p.get("root")}
        for project_json in self.workspace_root.glob("*/project.json"):
            try:
                project = json.loads(project_json.read_text(encoding="utf-8"))
                root = str(project_json.parent.resolve())
                if root not in known_roots:
                    project["root"] = root
                    projects.append(project)
                    known_roots.add(root)
            except Exception:
                # harmless: malformed project entry, skip
                continue

        projects = [p for p in projects if p.get("root") and Path(p["root"]).exists()]
        projects.sort(key=lambda p: p.get("updated_at", p.get("created_at", "")), reverse=True)
        self._save_registry(projects)
        return projects

    def create_project(
        self,
        name: str,
        task: str = "detect",
        classes: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        slug = self._slugify(name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = self.workspace_root / f"{slug}_{timestamp}"
        for subdir in ("images", "labels", "models", "runs"):
            (root / subdir).mkdir(parents=True, exist_ok=True)

        project = {
            "name": name.strip() or slug,
            "task": task,
            "root": str(root.resolve()),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        self._write_project(project)

        class_manager = ClassManager(str(root))
        class_manager.classes = []
        for class_name in classes or ["目标"]:
            cleaned = str(class_name).strip()
            if cleaned:
                class_manager.get_or_create_class(cleaned)
        class_manager.save()

        projects = [p for p in self.list_projects() if Path(p["root"]).resolve() != root.resolve()]
        projects.insert(0, project)
        self._save_registry(projects)
        logger.info(f"Created project: {project['name']} at {root}")
        return project

    def open_project(self, root: str | Path) -> dict[str, Any]:
        root_path = Path(root)
        project_json = root_path / "project.json"
        if project_json.exists():
            project = json.loads(project_json.read_text(encoding="utf-8"))
        else:
            project = {
                "name": root_path.name,
                "task": "detect",
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
        project["root"] = str(root_path.resolve())
        project["updated_at"] = datetime.now().isoformat(timespec="seconds")
        for subdir in ("images", "labels", "models", "runs"):
            (root_path / subdir).mkdir(parents=True, exist_ok=True)
        self._write_project(project)
        self._upsert_registry(project)
        return project

    def delete_project(self, project: dict[str, Any], delete_files: bool = False) -> None:
        root = Path(project["root"]).resolve()
        projects = [
            p for p in self.list_projects()
            if Path(p.get("root", "")).resolve() != root
        ]
        self._save_registry(projects)
        if delete_files and root.exists() and self._is_inside_workspace(root):
            shutil.rmtree(root)

    def touch_project(self, project: dict[str, Any]) -> dict[str, Any]:
        project = dict(project)
        project["updated_at"] = datetime.now().isoformat(timespec="seconds")
        self._write_project(project)
        self._upsert_registry(project)
        return project

    # ------------------------------------------------------------------
    # Imports
    # ------------------------------------------------------------------

    def import_folder(
        self,
        project: dict[str, Any],
        folder: str | Path,
        progress: Optional[Callable[[int, int, str], None]] = None,
    ) -> tuple[int, int]:
        files = [
            path for path in Path(folder).rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        return self.import_images(project, files, progress=progress)

    def import_dataset_as_project(
        self,
        folder: str | Path,
        name: Optional[str] = None,
        task: str = "detect",
        progress: Optional[Callable[[int, int, str], None]] = None,
    ) -> tuple[dict[str, Any], int, int, int, int]:
        """Copy an existing image folder or YOLO dataset into a managed project."""
        source = Path(folder)
        if not source.is_dir():
            raise ValueError(f"项目目录不存在: {source}")

        classes = self._classes_from_dataset(source) or ["目标"]
        project = self.create_project(name or source.name, task=task, classes=classes)
        root = Path(project["root"])
        images_root = source / "images" if (source / "images").is_dir() else source
        image_files = [
            path for path in images_root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]

        imported = 0
        skipped = 0
        total = len(image_files)
        for index, src in enumerate(image_files, start=1):
            if progress:
                progress(index, total, src.name)
            try:
                rel = src.relative_to(images_root)
                dest = root / "images" / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists():
                    dest = self._unique_path(dest)
                shutil.copy2(src, dest)
                imported += 1
            except Exception as exc:
                logger.warning(f"Failed to import image {src}: {exc}")
                skipped += 1

        label_imported, label_skipped = self._import_dataset_labels(source, images_root, root)
        project["import_source"] = str(source.resolve())
        project = self.touch_project(project)
        return project, imported, skipped, label_imported, label_skipped

    def import_images(
        self,
        project: dict[str, Any],
        files: list[str | Path],
        progress: Optional[Callable[[int, int, str], None]] = None,
    ) -> tuple[int, int]:
        images_dir = Path(project["root"]) / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        imported = 0
        skipped = 0
        total = len(files)
        for index, src in enumerate(files, start=1):
            src_path = Path(src)
            if progress:
                progress(index, total, src_path.name)
            if not src_path.is_file() or src_path.suffix.lower() not in IMAGE_EXTENSIONS:
                skipped += 1
                continue
            dest = self._unique_path(images_dir / src_path.name)
            try:
                shutil.copy2(src_path, dest)
                imported += 1
            except Exception as exc:
                logger.warning(f"Failed to import image {src_path}: {exc}")
                skipped += 1
        self.touch_project(project)
        return imported, skipped

    def import_video(
        self,
        project: dict[str, Any],
        video_path: str | Path,
        interval_frames: int = 30,
        progress: Optional[Callable[[int, int, str], None]] = None,
    ) -> int:
        video = Path(video_path)
        if not video.is_file() or video.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"无效的视频文件: {video}")

        output_dir = Path(project["root"]) / "images"
        extractor = VideoFrameExtractor()
        if not extractor.open(str(video)):
            raise RuntimeError(f"无法打开视频: {video}")
        try:
            paths = extractor.extract_auto(
                str(output_dir),
                mode="interval",
                interval_frames=max(1, interval_frames),
                dedup=True,
                progress_callback=lambda current, total: progress(current, total, video.name) if progress else None,
            )
        finally:
            extractor.close()
        self.touch_project(project)
        return len(paths)

    def import_yolo_labels(
        self,
        project: dict[str, Any],
        labels_dir: str | Path,
        overwrite: bool = False,
    ) -> tuple[int, int]:
        labels_path = Path(labels_dir)
        if not labels_path.is_dir():
            raise ValueError(f"标注目录不存在: {labels_path}")

        dest_dir = Path(project["root"]) / "labels"
        dest_dir.mkdir(parents=True, exist_ok=True)
        image_map = self._image_map(project)
        imported = 0
        skipped = 0
        for label_file in labels_path.rglob("*.txt"):
            if label_file.name.lower() == "classes.txt":
                continue
            image_path = image_map.get(label_file.stem.lower())
            dest = self._label_path_for_image(image_path) if image_path else dest_dir / label_file.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() and not overwrite:
                skipped += 1
                continue
            shutil.copy2(label_file, dest)
            imported += 1

        self._merge_classes_from_candidates(project, [labels_path / "classes.txt", labels_path.parent / "classes.txt"])
        self.touch_project(project)
        return imported, skipped

    def import_voc_labels(
        self,
        project: dict[str, Any],
        voc_dir: str | Path,
        overwrite: bool = False,
    ) -> tuple[int, int]:
        voc_path = Path(voc_dir)
        if not voc_path.is_dir():
            raise ValueError(f"VOC 目录不存在: {voc_path}")
        class_manager = ClassManager(project["root"])
        dest_dir = Path(project["root"]) / "labels"
        dest_dir.mkdir(parents=True, exist_ok=True)
        image_map = self._image_map(project)

        imported = 0
        skipped = 0
        for xml_file in voc_path.rglob("*.xml"):
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                filename = root.findtext("filename") or f"{xml_file.stem}.jpg"
                image_path = image_map.get(Path(filename).stem.lower())
                if not image_path:
                    skipped += 1
                    continue
                width, height = self._read_size_from_voc(root)
                if width <= 0 or height <= 0:
                    width, height = self._read_image_size(image_path)
                if width <= 0 or height <= 0:
                    skipped += 1
                    continue
                lines = []
                for obj in root.findall("object"):
                    class_name = (obj.findtext("name") or "object").strip()
                    class_id = class_manager.get_or_create_class(class_name)
                    bbox = obj.find("bndbox")
                    if bbox is None:
                        continue
                    x1 = float(bbox.findtext("xmin", "0"))
                    y1 = float(bbox.findtext("ymin", "0"))
                    x2 = float(bbox.findtext("xmax", "0"))
                    y2 = float(bbox.findtext("ymax", "0"))
                    lines.append(self._bbox_to_yolo_line(class_id, x1, y1, x2, y2, width, height))
                dest = self._label_path_for_image(image_path)
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists() and not overwrite:
                    skipped += 1
                    continue
                dest.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                imported += 1
            except Exception as exc:
                logger.warning(f"Failed to import VOC {xml_file}: {exc}")
                skipped += 1
        class_manager.save()
        self.touch_project(project)
        return imported, skipped

    def import_coco_labels(
        self,
        project: dict[str, Any],
        coco_file: str | Path,
        overwrite: bool = False,
    ) -> tuple[int, int]:
        coco_path = Path(coco_file)
        if not coco_path.is_file():
            raise ValueError(f"COCO 文件不存在: {coco_path}")
        data = json.loads(coco_path.read_text(encoding="utf-8"))
        categories = {cat["id"]: cat.get("name", str(cat["id"])) for cat in data.get("categories", [])}
        images = {img["id"]: img for img in data.get("images", [])}
        anns_by_image: dict[int, list[dict[str, Any]]] = {}
        for ann in data.get("annotations", []):
            anns_by_image.setdefault(ann.get("image_id"), []).append(ann)

        class_manager = ClassManager(project["root"])
        category_to_class = {
            cat_id: class_manager.get_or_create_class(name)
            for cat_id, name in categories.items()
        }
        image_map = self._image_map(project)
        dest_dir = Path(project["root"]) / "labels"
        dest_dir.mkdir(parents=True, exist_ok=True)

        imported = 0
        skipped = 0
        for image_id, image in images.items():
            filename = image.get("file_name", "")
            image_path = image_map.get(Path(filename).stem.lower())
            if not image_path:
                skipped += 1
                continue
            width = int(image.get("width") or 0)
            height = int(image.get("height") or 0)
            if width <= 0 or height <= 0:
                width, height = self._read_image_size(image_path)
            if width <= 0 or height <= 0:
                skipped += 1
                continue
            lines = []
            for ann in anns_by_image.get(image_id, []):
                bbox = ann.get("bbox") or []
                if len(bbox) < 4:
                    continue
                x, y, w, h = map(float, bbox[:4])
                class_id = category_to_class.get(ann.get("category_id"), 0)
                lines.append(self._bbox_to_yolo_line(class_id, x, y, x + w, y + h, width, height))
            dest = self._label_path_for_image(image_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() and not overwrite:
                skipped += 1
                continue
            dest.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            imported += 1

        class_manager.save()
        self.touch_project(project)
        return imported, skipped

    def build_data_yaml(
        self,
        project: dict[str, Any],
        train_ratio: float = 0.8,
        val_ratio: float = 0.15,
        test_ratio: float = 0.05,
    ) -> str:
        root = Path(project["root"])
        class_manager = ClassManager(str(root))
        yaml_path = DatasetManager.build_data_yaml(
            images_dir=str(root / "images"),
            labels_dir=str(root / "labels"),
            classes=class_manager.get_all_classes(),
            output_yaml=str(root / "data.yaml"),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
        self.touch_project(project)
        return yaml_path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def list_images(project: dict[str, Any]) -> list[str]:
        images_root = Path(project["root"]) / "images"
        if not images_root.exists():
            return []
        return sorted(
            str(path)
            for path in images_root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    @staticmethod
    def _slugify(name: str) -> str:
        slug = re.sub(r"[^\w\u4e00-\u9fff.-]+", "_", name.strip(), flags=re.UNICODE).strip("._")
        return slug or "project"

    def _write_project(self, project: dict[str, Any]) -> None:
        root = Path(project["root"])
        root.mkdir(parents=True, exist_ok=True)
        (root / "project.json").write_text(json.dumps(project, indent=2, ensure_ascii=False), encoding="utf-8")

    def _save_registry(self, projects: list[dict[str, Any]]) -> None:
        self.registry_path.write_text(json.dumps(projects, indent=2, ensure_ascii=False), encoding="utf-8")

    def _upsert_registry(self, project: dict[str, Any]) -> None:
        root = Path(project["root"]).resolve()
        projects = [p for p in self.list_projects() if Path(p.get("root", "")).resolve() != root]
        projects.insert(0, project)
        self._save_registry(projects)

    def _is_inside_workspace(self, path: Path) -> bool:
        try:
            path.resolve().relative_to(self.workspace_root.resolve())
            return True
        except ValueError:
            return False

    @staticmethod
    def _unique_path(path: Path) -> Path:
        if not path.exists():
            return path
        stem = path.stem
        suffix = path.suffix
        parent = path.parent
        counter = 1
        while True:
            candidate = parent / f"{stem}_{counter:03d}{suffix}"
            if not candidate.exists():
                return candidate
            counter += 1

    def _merge_classes_from_candidates(self, project: dict[str, Any], candidates: list[Path]) -> None:
        class_manager = ClassManager(project["root"])
        changed = False
        for candidate in candidates:
            if not candidate.is_file():
                continue
            for line in candidate.read_text(encoding="utf-8").splitlines():
                name = line.strip()
                if name and class_manager.get_class_id(name) is None:
                    class_manager.get_or_create_class(name)
                    changed = True
        if changed:
            class_manager.save()

    @staticmethod
    def _classes_from_dataset(source: Path) -> list[str]:
        candidates = [source / "classes.txt", source / "labels" / "classes.txt"]
        for candidate in candidates:
            if candidate.is_file():
                classes = [line.strip() for line in candidate.read_text(encoding="utf-8").splitlines() if line.strip()]
                if classes:
                    return classes

        data_yaml = source / "data.yaml"
        if data_yaml.is_file():
            try:
                data = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
                names = data.get("names")
                if isinstance(names, dict):
                    def _sort_key(item):
                        try:
                            return (0, int(item))
                        except (ValueError, TypeError):
                            return (1, item)
                    return [str(names[key]).strip() for key in sorted(names, key=_sort_key) if str(names[key]).strip()]
                if isinstance(names, list):
                    return [str(name).strip() for name in names if str(name).strip()]
            except Exception as exc:
                logger.warning(f"Failed to read classes from {data_yaml}: {exc}")
        return []

    @staticmethod
    def _import_dataset_labels(source: Path, images_root: Path, project_root: Path) -> tuple[int, int]:
        labels_root = source / "labels"
        imported = 0
        skipped = 0
        if labels_root.is_dir():
            label_files = [path for path in labels_root.rglob("*.txt") if path.name.lower() != "classes.txt"]
            for src in label_files:
                try:
                    rel = src.relative_to(labels_root)
                    dest = project_root / "labels" / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dest)
                    imported += 1
                except Exception as exc:
                    logger.warning(f"Failed to import label {src}: {exc}")
                    skipped += 1
            return imported, skipped

        image_files = [
            path for path in images_root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        for image_path in image_files:
            src = image_path.with_suffix(".txt")
            if not src.is_file() or src.name.lower() == "classes.txt":
                continue
            try:
                rel = image_path.relative_to(images_root).with_suffix(".txt")
                dest = project_root / "labels" / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
                imported += 1
            except Exception as exc:
                logger.warning(f"Failed to import sidecar label {src}: {exc}")
                skipped += 1
        return imported, skipped

    @staticmethod
    def _image_map(project: dict[str, Any]) -> dict[str, str]:
        return {
            Path(path).stem.lower(): path
            for path in ProjectManager.list_images(project)
        }

    @staticmethod
    def _label_path_for_image(image_path: str | Path) -> Path:
        img = Path(image_path)
        parts = list(img.parts)
        for i in range(len(parts) - 1, -1, -1):
            if parts[i] == "images":
                return Path(*parts[:i], "labels", *parts[i + 1:]).with_suffix(".txt")
        return img.with_suffix(".txt")

    @staticmethod
    def _read_image_size(image_path: str) -> tuple[int, int]:
        try:
            data = Path(image_path).read_bytes()
            image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is not None:
                h, w = image.shape[:2]
                return int(w), int(h)
        except Exception:
            # harmless: image unreadable or corrupt, return zero size
            pass
        return 0, 0

    @staticmethod
    def _read_size_from_voc(root: ET.Element) -> tuple[int, int]:
        size = root.find("size")
        if size is None:
            return 0, 0
        try:
            return int(size.findtext("width", "0")), int(size.findtext("height", "0"))
        except Exception:
            return 0, 0

    @staticmethod
    def _bbox_to_yolo_line(class_id: int, x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> str:
        x1, x2 = sorted((max(0.0, min(float(x1), width)), max(0.0, min(float(x2), width))))
        y1, y2 = sorted((max(0.0, min(float(y1), height)), max(0.0, min(float(y2), height))))
        xc = ((x1 + x2) / 2.0) / width
        yc = ((y1 + y2) / 2.0) / height
        bw = (x2 - x1) / width
        bh = (y2 - y1) / height
        return f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
