"""Shared test fixtures for Yolo Studio."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure project root is on sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# -----------------------------------------------------------------------
# Mock heavy imports at module level — MUST run before pytest collects
# tests, because importing test files triggers project imports which try
# to load torch / ultralytics / PyQt6 / etc.
# -----------------------------------------------------------------------

class _LazyMockModule:
    """Lazy-creating mock for module-level symbols.

    On attribute access it returns MagicMock (the class, not an instance),
    so the result can be used as a base class, an import target, or
    instantiated.  Known sub-packages (pre-registered in sys.modules) are
    returned as _LazyMockModule instances for further traversal.
    """

    def __init__(self, name):
        object.__setattr__(self, "_name", name)

    def __getattr__(self, attr):
        if attr.startswith("_"):
            raise AttributeError(attr)
        child_name = f"{self._name}.{attr}"
        # If it's a known submodule, return the module mock
        if child_name in sys.modules:
            return sys.modules[child_name]
        # Otherwise return MagicMock CLASS — valid as base class / callable
        return MagicMock

    def __call__(self, *args, **kwargs):
        return MagicMock()(*args, **kwargs)


def _mock_module(name, attrs=None):
    """Inject a lazy-mock module into sys.modules if not already present."""
    if name not in sys.modules:
        m = _LazyMockModule(name)
        if attrs:
            for k, v in attrs.items():
                object.__setattr__(m, k, v)
        sys.modules[name] = m


# Pre-populate with heavy modules that may not be installed
_mock_module("torch")
_mock_module("torch.multiprocessing")
_mock_module("torch.multiprocessing.spawn")
_mock_module("torchvision")
_mock_module("torchvision.ops")
_mock_module("ultralytics")
_mock_module("ultralytics.nn")
_mock_module("ultralytics.utils")
_mock_module("ultralytics.data")
_mock_module("ultralytics.engine")
_mock_module("cv2", {"__version__": "4.8.0"})
_mock_module("albumentations")
_mock_module("transformers")
_mock_module("accelerate")
_mock_module("onnxruntime")
_mock_module("onnx")
_mock_module("PyQt6")
_mock_module("PyQt6.QtWidgets")
_mock_module("PyQt6.QtCore")
_mock_module("PyQt6.QtGui")
_mock_module("PyQt6.QtSvg")
_mock_module("PyQt6.QtSvgWidgets")
_mock_module("PyQt6.sip")
_mock_module("matplotlib")
_mock_module("matplotlib.backends")
_mock_module("matplotlib.backends.backend_qtagg")
_mock_module("matplotlib.backends.backend_agg")
_mock_module("matplotlib.figure")
_mock_module("sklearn")
_mock_module("sklearn.metrics")


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

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
