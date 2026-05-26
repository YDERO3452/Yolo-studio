"""Shared test fixtures for Yolo Studio."""

import os
import sys
from pathlib import Path

import pytest

# Ensure project root is on sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def tmp_dir(tmp_path):
    """A temporary directory for file I/O tests."""
    return tmp_path


@pytest.fixture
def sample_yolo_labels(tmp_path):
    """Create a labels directory with sample YOLO label files."""
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir(parents=True)

    label_content = {
        "img1.txt": "0 0.5 0.5 0.2 0.3\n1 0.1 0.1 0.3 0.3\n",
        "img2.txt": "0 0.3 0.4 0.4 0.5\n2 0.7 0.7 0.1 0.1 0.05 0.05 0.95 0.95\n",
        "img3.txt": "",
        "img4.txt": "0 0.5 0.5 0.2 0.3\ninvalid line\n",
    }
    for name, content in label_content.items():
        (labels_dir / name).write_text(content, encoding="utf-8")

    return labels_dir


@pytest.fixture
def sample_image_dir(tmp_path):
    """Create an images directory with dummy files."""
    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True)
    for name in ["img1.jpg", "img2.jpg", "img3.jpg"]:
        (images_dir / name).touch()
    return images_dir


@pytest.fixture
def sample_classes_file(tmp_path):
    """Create a classes.txt file."""
    classes_file = tmp_path / "classes.txt"
    classes_file.write_text("person\ncar\nbicycle\n", encoding="utf-8")
    return classes_file
