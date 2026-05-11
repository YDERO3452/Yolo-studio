"""Class management module — handles class definitions, colors, and persistence.

Architecture patterns:
- Follows LabelConverter design for file I/O
- Uses similar configuration management approach
- Implements color management with HSV color space
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from loguru import logger
import colorsys

# Built-in COCO 80-class English → Chinese mapping
COCO_EN_ZH_MAP: Dict[str, str] = {
    "person": "人", "bicycle": "自行车", "car": "汽车", "motorcycle": "摩托车",
    "airplane": "飞机", "bus": "公交车", "train": "火车", "truck": "卡车",
    "boat": "船", "traffic light": "红绿灯", "fire hydrant": "消防栓",
    "stop sign": "停止标志", "parking meter": "停车收费器", "bench": "长椅",
    "bird": "鸟", "cat": "猫", "dog": "狗", "horse": "马",
    "sheep": "羊", "cow": "牛", "elephant": "大象", "bear": "熊",
    "zebra": "斑马", "giraffe": "长颈鹿", "backpack": "背包",
    "umbrella": "雨伞", "handbag": "手提包", "tie": "领带",
    "suitcase": "行李箱", "frisbee": "飞盘", "skis": "滑雪板",
    "snowboard": "单板滑雪", "sports ball": "运动球", "kite": "风筝",
    "baseball bat": "棒球棒", "baseball glove": "棒球手套",
    "skateboard": "滑板", "surfboard": "冲浪板", "tennis racket": "网球拍",
    "bottle": "瓶子", "wine glass": "酒杯", "cup": "杯子",
    "fork": "叉子", "knife": "刀", "spoon": "勺子", "bowl": "碗",
    "banana": "香蕉", "apple": "苹果", "sandwich": "三明治",
    "orange": "橙子", "broccoli": "西兰花", "carrot": "胡萝卜",
    "hot dog": "热狗", "pizza": "披萨", "donut": "甜甜圈",
    "cake": "蛋糕", "chair": "椅子", "couch": "沙发",
    "potted plant": "盆栽", "bed": "床", "dining table": "餐桌",
    "toilet": "马桶", "tv": "电视", "laptop": "笔记本电脑",
    "mouse": "鼠标", "remote": "遥控器", "keyboard": "键盘",
    "cell phone": "手机", "microwave": "微波炉", "oven": "烤箱",
    "toaster": "烤面包机", "sink": "水槽", "refrigerator": "冰箱",
    "book": "书", "clock": "时钟", "vase": "花瓶",
    "scissors": "剪刀", "teddy bear": "泰迪熊", "hair drier": "吹风机",
    "toothbrush": "牙刷",
}


class ClassManager:
    """Manages annotation classes with persistence to classes.txt and classes.colors files.

    Architecture:
    - Separates class definitions (classes.txt) from color mappings (classes.colors)
    - Uses JSON for color persistence
    - Generates colors using HSV color space for better distribution
    """

    def __init__(self, project_dir: Optional[str] = None):
        """Initialize ClassManager.

        Args:
            project_dir: Project directory path. If None, uses current directory.
        """
        self.project_dir = Path(project_dir) if project_dir else Path.cwd()
        self.project_dir.mkdir(parents=True, exist_ok=True)

        self.classes_file = self.project_dir / "classes.txt"
        self.colors_file = self.project_dir / "classes.colors"
        self.name_map_file = self.project_dir / "class_name_map.json"

        self.classes: List[str] = []  # List of class names
        self.colors: Dict[str, Tuple[int, int, int]] = {}  # class_name -> (R, G, B)
        self.name_map: Dict[str, str] = {}  # model_class_name → project_class_name

        self.load()
        logger.info(f"ClassManager initialized with project_dir: {self.project_dir}")

    def load(self) -> None:
        """Load classes, colors, and name map from files."""
        self._load_classes()
        self._load_colors()
        self._load_name_map()
        # Generate missing colors
        self._generate_missing_colors()

    def _load_classes(self) -> None:
        """Load classes from classes.txt file."""
        if self.classes_file.exists():
            try:
                self.classes = self.read_lines(str(self.classes_file))
                logger.info(f"Loaded {len(self.classes)} classes from {self.classes_file}")
            except Exception as e:
                logger.error(f"Failed to load classes from {self.classes_file}: {e}")
                self.classes = []
        else:
            logger.info(f"Classes file not found: {self.classes_file}")
            self.classes = []

    def _load_colors(self) -> None:
        """Load colors from classes.colors file."""
        if self.colors_file.exists():
            try:
                colors_data = self.read_json(str(self.colors_file))
                self.colors = {
                    class_name: tuple(color)
                    for class_name, color in colors_data.items()
                }
                logger.info(f"Loaded colors for {len(self.colors)} classes")
            except Exception as e:
                logger.error(f"Failed to load colors from {self.colors_file}: {e}")
                self.colors = {}
        else:
            logger.info(f"Colors file not found: {self.colors_file}")
            self.colors = {}

    def _generate_missing_colors(self) -> None:
        """Generate colors for classes that don't have one."""
        for class_name in self.classes:
            if class_name not in self.colors:
                color = self._generate_color(len(self.colors))
                self.colors[class_name] = color
                logger.debug(f"Generated color for class '{class_name}': {color}")

    def _generate_color(self, index: int) -> Tuple[int, int, int]:
        """Generate a color based on index using HSV color space.

        Args:
            index: Index for color generation

        Returns:
            RGB color tuple (R, G, B)
        """
        # Use HSV color space for better color distribution
        hue = (index * 0.618033988749895) % 1.0  # Golden ratio for hue
        saturation = 0.7 + (index % 3) * 0.1  # Vary saturation
        value = 0.9  # Keep brightness high

        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        return tuple(int(c * 255) for c in rgb)

    def save(self) -> bool:
        """Save classes, colors, and name map to files.

        Returns:
            True if successful, False otherwise
        """
        try:
            self._save_classes()
            self._save_colors()
            self._save_name_map()
            logger.info("Classes, colors, and name map saved successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to save classes, colors, and name map: {e}")
            return False

    def _save_classes(self) -> None:
        """Save classes to classes.txt file."""
        with open(self.classes_file, "w", encoding="utf-8") as f:
            for class_name in self.classes:
                f.write(f"{class_name}\n")
        logger.debug(f"Saved {len(self.classes)} classes to {self.classes_file}")

    def _save_colors(self) -> None:
        """Save colors to classes.colors file."""
        colors_data = {
            class_name: list(color) for class_name, color in self.colors.items()
        }
        self.save_json(colors_data, str(self.colors_file))
        logger.debug(f"Saved colors for {len(self.colors)} classes to {self.colors_file}")

    def add_class(self, class_name: str) -> bool:
        """Add a new class.

        Args:
            class_name: Name of the class to add

        Returns:
            True if successful, False if class already exists
        """
        if class_name in self.classes:
            logger.warning(f"Class '{class_name}' already exists")
            return False

        self.classes.append(class_name)
        # Generate color for new class
        color = self._generate_color(len(self.colors))
        self.colors[class_name] = color
        logger.info(f"Added class '{class_name}' with color {color}")
        return True

    def remove_class(self, class_name: str) -> bool:
        """Remove a class.

        Args:
            class_name: Name of the class to remove

        Returns:
            True if successful, False if class doesn't exist
        """
        if class_name not in self.classes:
            logger.warning(f"Class '{class_name}' not found")
            return False

        self.classes.remove(class_name)
        if class_name in self.colors:
            del self.colors[class_name]
        logger.info(f"Removed class '{class_name}'")
        return True

    def rename_class(self, old_name: str, new_name: str) -> bool:
        """Rename a class.

        Args:
            old_name: Current name of the class
            new_name: New name for the class

        Returns:
            True if successful, False otherwise
        """
        if old_name not in self.classes:
            logger.warning(f"Class '{old_name}' not found")
            return False

        if new_name in self.classes:
            logger.warning(f"Class '{new_name}' already exists")
            return False

        idx = self.classes.index(old_name)
        self.classes[idx] = new_name

        # Update color mapping
        if old_name in self.colors:
            self.colors[new_name] = self.colors.pop(old_name)

        logger.info(f"Renamed class '{old_name}' to '{new_name}'")
        return True

    def get_class_color(self, class_name: str) -> Optional[Tuple[int, int, int]]:
        """Get color for a class.

        Args:
            class_name: Name of the class

        Returns:
            RGB color tuple or None if class not found
        """
        return self.colors.get(class_name)

    def set_class_color(self, class_name: str, color: Tuple[int, int, int]) -> bool:
        """Set color for a class.

        Args:
            class_name: Name of the class
            color: RGB color tuple

        Returns:
            True if successful, False if class not found
        """
        if class_name not in self.classes:
            logger.warning(f"Class '{class_name}' not found")
            return False

        self.colors[class_name] = color
        logger.debug(f"Set color for class '{class_name}' to {color}")
        return True

    def get_all_classes(self) -> List[str]:
        """Get all class names.

        Returns:
            List of class names
        """
        return self.classes.copy()

    def get_class_count(self) -> int:
        """Get number of classes.

        Returns:
            Number of classes
        """
        return len(self.classes)

    def get_class_index(self, class_name: str) -> Optional[int]:
        """Get index of a class.

        Args:
            class_name: Name of the class

        Returns:
            Index of the class or None if not found
        """
        try:
            return self.classes.index(class_name)
        except ValueError:
            return None

    # Aliases used by gui modules
    def get_color(self, class_name: str) -> Optional[Tuple[int, int, int]]:
        """Alias for get_class_color."""
        return self.get_class_color(class_name)

    def set_color(self, class_name: str, color: Tuple[int, int, int]) -> bool:
        """Alias for set_class_color."""
        return self.set_class_color(class_name, color)

    def get_class_id(self, class_name: str) -> Optional[int]:
        """Alias for get_class_index."""
        return self.get_class_index(class_name)

    def get_class_name(self, class_id: int) -> Optional[str]:
        """Get class name by index (alias for get_class_by_index)."""
        return self.get_class_by_index(class_id)

    def get_or_create_class(self, class_name: str) -> int:
        """Get class ID, creating the class if it doesn't exist.

        Returns:
            Integer class ID.
        """
        idx = self.get_class_id(class_name)
        if idx is not None:
            return idx
        self.add_class(class_name)
        return self.get_class_id(class_name) or 0

    # ------------------------------------------------------------------
    # Name mapping (model class name → project class name)
    # ------------------------------------------------------------------

    def _load_name_map(self) -> None:
        """Load name mapping from class_name_map.json."""
        if self.name_map_file.exists():
            try:
                data = self.read_json(str(self.name_map_file))
                self.name_map = {str(k): str(v) for k, v in data.items()}
                logger.info(f"Loaded {len(self.name_map)} name mappings from {self.name_map_file}")
            except Exception as e:
                logger.error(f"Failed to load name map from {self.name_map_file}: {e}")
                self.name_map = {}
        else:
            # Initialize with built-in COCO mapping
            self.name_map = dict(COCO_EN_ZH_MAP)
            self._save_name_map()
            logger.info(f"Created default name map with {len(self.name_map)} COCO entries")

    def _save_name_map(self) -> None:
        """Save name mapping to class_name_map.json."""
        self.save_json(self.name_map, str(self.name_map_file))
        logger.debug(f"Saved {len(self.name_map)} name mappings to {self.name_map_file}")

    def map_class_name(self, model_class_name: str) -> str:
        """Map a model class name to the project class name.

        If a mapping exists in name_map, use it.
        Otherwise, return the original name unchanged.

        Args:
            model_class_name: Class name from the model (e.g. "person")

        Returns:
            Mapped class name (e.g. "人") or original if no mapping exists
        """
        return self.name_map.get(model_class_name, model_class_name)

    def set_name_mapping(self, model_name: str, project_name: str) -> None:
        """Add or update a name mapping entry.

        Args:
            model_name: Original model class name (e.g. "person")
            project_name: Project class name (e.g. "人")
        """
        self.name_map[model_name] = project_name
        logger.debug(f"Name mapping: '{model_name}' -> '{project_name}'")

    def remove_name_mapping(self, model_name: str) -> None:
        """Remove a name mapping entry."""
        if model_name in self.name_map:
            del self.name_map[model_name]

    def get_name_map(self) -> Dict[str, str]:
        """Get the full name mapping dict."""
        return dict(self.name_map)

    def import_model_names(self, model_names: Dict[int, str], translate: bool = True) -> int:
        """Import class names from a model and create mappings.

        Args:
            model_names: Dict of {class_id: class_name} from the model
            translate: If True, apply built-in COCO translation for known names

        Returns:
            Number of new mappings added
        """
        added = 0
        for cls_id, cls_name in model_names.items():
            cls_name = str(cls_name)
            if cls_name not in self.name_map:
                if translate and cls_name in COCO_EN_ZH_MAP:
                    self.name_map[cls_name] = COCO_EN_ZH_MAP[cls_name]
                else:
                    self.name_map[cls_name] = cls_name
                added += 1
        if added > 0:
            self._save_name_map()
        return added

    def import_from_list(self, class_list: List[str]) -> None:
        """Import classes from a list, generating colors for new ones."""
        for class_name in class_list:
            if class_name not in self.classes:
                self.classes.append(class_name)
        self._generate_missing_colors()

    def __len__(self) -> int:
        return len(self.classes)

    def get_class_by_index(self, index: int) -> Optional[str]:
        """Get class name by index.

        Args:
            index: Index of the class

        Returns:
            Class name or None if index out of range
        """
        if 0 <= index < len(self.classes):
            return self.classes[index]
        return None

    @staticmethod
    def read_lines(file_path: str, encoding: str = "utf-8") -> List[str]:
        """Read lines from a file.

        Args:
            file_path: Path to the file
            encoding: File encoding

        Returns:
            List of lines (stripped)
        """
        with open(file_path, "r", encoding=encoding) as f:
            return f.read().splitlines()

    @staticmethod
    def read_json(file_path: str, encoding: str = "utf-8") -> Dict:
        """Read JSON from a file.

        Args:
            file_path: Path to the file
            encoding: File encoding

        Returns:
            Parsed JSON data
        """
        with open(file_path, "r", encoding=encoding) as f:
            return json.load(f)

    @staticmethod
    def save_json(
        data: Dict,
        file_path: str,
        indent: int = 2,
        ensure_ascii: bool = False,
        encoding: str = "utf-8",
    ) -> None:
        """Save data to JSON file.

        Args:
            data: Data to save
            file_path: Path to the file
            indent: JSON indentation
            ensure_ascii: Whether to escape non-ASCII characters
            encoding: File encoding
        """
        with open(file_path, "w", encoding=encoding) as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
