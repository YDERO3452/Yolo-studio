"""Configuration management module."""

import os
from typing import Any, Optional

import yaml
from loguru import logger
from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    """Training configuration — field names match Ultralytics YOLO.train() API.

    NOTE: Keys like ``batch`` and ``imgsz`` deliberately follow the
    Ultralytics naming convention so that ``TrainingConfig.model_dump()``
    can be passed directly to ``model.train(**kwargs)``.
    """

    # --- Core training settings ---
    model: str = "yolov8n.pt"
    epochs: int = 100
    time: Optional[float] = None          # max training hours (overrides epochs)
    batch: int = 16                       # was batch_size
    imgsz: int = 640                      # was img_size
    device: str = "0"
    workers: int = 8
    patience: int = 100
    save_period: int = -1                 # -1 = disabled
    save: bool = True
    cache: bool = False
    project: str = "runs/train"
    name: Optional[str] = None
    exist_ok: bool = False
    pretrained: bool = True
    optimizer: str = "auto"
    verbose: bool = True
    seed: int = 0
    deterministic: bool = True
    single_cls: bool = False
    rect: bool = False
    cos_lr: bool = False
    close_mosaic: int = 10
    resume: bool = False
    amp: bool = True
    fraction: float = 1.0
    profile: bool = False
    freeze: Optional[str] = None          # e.g. "10" or "[0,1,2]"
    multi_scale: float = 0.0
    # compile: bool = False               # torch.compile – advanced, skip for now

    # --- Segmentation ---
    overlap_mask: bool = True
    mask_ratio: int = 4

    # --- Classification ---
    dropout: float = 0.0

    # --- Loss gains ---
    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: float = 3.0
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1
    box: float = 7.5
    cls: float = 0.5
    dfl: float = 1.5
    pose: float = 12.0
    kobj: float = 1.0
    nbs: int = 64

    # --- Augmentation ---
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    bgr: float = 0.0
    mosaic: float = 1.0
    mixup: float = 0.0
    copy_paste: float = 0.0
    copy_paste_mode: str = "flip"
    auto_augment: str = "randaugment"
    erasing: float = 0.4


class InferenceConfig(BaseModel):
    """Inference / prediction configuration.

    Field names follow Ultralytics ``model.predict()`` API exactly.
    """
    conf: float = 0.25                    # was conf_threshold
    iou: float = 0.7                      # was iou_threshold (Ultralytics default=0.7 for NMS)
    max_det: int = 300
    classes: Optional[list] = None
    agnostic_nms: bool = False
    augment: bool = False                 # test-time augmentation
    show_labels: bool = True
    show_conf: bool = True
    show_boxes: bool = True
    line_width: Optional[int] = None      # None = auto-scale with image size
    vid_stride: int = 1
    retina_masks: bool = False

    # --- Performance tuning ---
    half: bool = True                     # FP16 semi-precision (GPU only, ~2-3x faster)
    imgsz: int = 640                      # inference resolution (lower = faster)


class AnnotationConfig(BaseModel):
    default_classes: list = Field(default_factory=lambda: ["目标"])
    auto_save: bool = True
    label_format: str = "yolo"
    min_box_size: int = 10
    default_shape_type: str = "bbox"
    num_keypoints: int = 17
    keypoint_names: list = Field(default_factory=list)


class ExportConfig(BaseModel):
    """Export configuration.

    Field names follow Ultralytics ``model.export()`` API.
    """
    format: str = "onnx"
    imgsz: int = 640                      # was img_size
    half: bool = False
    dynamic: bool = False
    simplify: bool = True
    opset: Optional[int] = None          # None = use Ultralytics tested default
    int8: bool = False
    workspace: Optional[float] = None     # TensorRT workspace in GiB
    nms: bool = False                     # fuse NMS into exported model


class AppGeneralConfig(BaseModel):
    name: str = "YOLO Studio"
    version: str = "1.0.0"
    theme: str = "dark"
    language: str = "zh"


class AppConfig(BaseModel):
    app: AppGeneralConfig = Field(default_factory=AppGeneralConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    annotation: AnnotationConfig = Field(default_factory=AnnotationConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)


class ConfigManager:
    """Manages application configuration."""

    def __init__(self, config_path: Optional[str] = None):
        from freeze import get_resource_path
        self.config_path = config_path or str(get_resource_path("configs/default.yaml"))
        self.config = AppConfig()
        if os.path.exists(self.config_path):
            self.load()

    def load(self, path: Optional[str] = None):
        """Load configuration from YAML file."""
        path = path or self.config_path
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found: {path}, using defaults")
            return
        if data:
            if "app" in data:
                self.config.app = AppGeneralConfig(**data["app"])
            if "training" in data:
                self.config.training = TrainingConfig(**data["training"])
            if "inference" in data:
                self.config.inference = InferenceConfig(**data["inference"])
            if "annotation" in data:
                self.config.annotation = AnnotationConfig(**data["annotation"])
            if "export" in data:
                self.config.export = ExportConfig(**data["export"])

    def save(self, path: Optional[str] = None):
        """Save configuration to YAML file."""
        path = path or self.config_path
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        data = {
            "app": self.config.app.model_dump(),
            "training": self.config.training.model_dump(),
            "inference": self.config.inference.model_dump(),
            "annotation": self.config.annotation.model_dump(),
            "export": self.config.export.model_dump(),
        }
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    def update(self, section: str, **kwargs):
        """Update a configuration section."""
        section_config = getattr(self.config, section, None)
        if section_config:
            for key, value in kwargs.items():
                if hasattr(section_config, key):
                    setattr(section_config, key, value)

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        section_config = getattr(self.config, section, None)
        if section_config:
            return getattr(section_config, key, default)
        return default
