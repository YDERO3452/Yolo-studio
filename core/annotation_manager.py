"""Annotation Manager - 整合所有核心模块的集成层.

采用模块化架构设计：
- 单例模式管理全局状态
- 配置文件管理（JSON 格式）
- 项目元数据管理
- 模块间通信协调
"""

import json
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from loguru import logger

from core.class_manager import ClassManager
from core.model_manager import ModelManager
from core.batch_processor import BatchProcessor, BatchProcessingConfig, ProcessingResult
from core.auto_labeling_enhanced import AutoLabelingEngine, AutoLabelingConfig
from core.format_converter import FormatConverter, ConversionResult


@dataclass
class ProjectConfig:
    """项目配置."""
    name: str
    path: str
    classes: List[str]
    model_name: str = "yolov8n"
    device: str = "0"
    conf_threshold: float = 0.25
    iou_threshold: float = 0.7          # Ultralytics NMS default=0.7


class AnnotationManager:
    """注解管理器 - 整合所有核心模块.

    架构模式：模块化集成设计
    - 单例模式管理全局状态
    - 提供统一的 API 接口
    - 管理项目配置和状态
    - 协调模块间的通信
    """

    _instance: Optional['AnnotationManager'] = None

    def __new__(cls):
        """单例模式实现."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """初始化注解管理器."""
        if self._initialized:
            return

        self.project_dir: Optional[Path] = None
        self.project_config: Optional[ProjectConfig] = None

        # 核心模块
        self.class_manager: Optional[ClassManager] = None
        self.model_manager: Optional[ModelManager] = None
        self.batch_processor: Optional[BatchProcessor] = None
        self.auto_labeling_engine: Optional[AutoLabelingEngine] = None
        self.format_converter: Optional[FormatConverter] = None

        logger.info("AnnotationManager initialized")
        self._initialized = True

    def create_project(
        self,
        project_dir: str,
        project_name: str,
        classes: List[str],
        model_name: str = "yolov8n",
    ) -> bool:
        """创建新项目.

        Args:
            project_dir: 项目目录
            project_name: 项目名称
            classes: 类别列表
            model_name: 默认模型名称

        Returns:
            True 如果成功，False 否则
        """
        try:
            self.project_dir = Path(project_dir)
            self.project_dir.mkdir(parents=True, exist_ok=True)

            # 创建项目配置
            self.project_config = ProjectConfig(
                name=project_name,
                path=str(self.project_dir),
                classes=classes,
                model_name=model_name,
            )

            # 初始化核心模块
            self.class_manager = ClassManager(str(self.project_dir))
            for cls in classes:
                self.class_manager.add_class(cls)
            self.class_manager.save()

            self.model_manager = ModelManager(str(self.project_dir / "models"))
            self.batch_processor = BatchProcessor(
                self.model_manager,
                self.class_manager.get_all_classes(),
            )
            self.auto_labeling_engine = AutoLabelingEngine(
                self.model_manager,
                self.class_manager.get_all_classes(),
            )
            self.format_converter = FormatConverter(
                self.class_manager.get_all_classes()
            )

            # 保存项目配置
            self._save_project_config()

            logger.info(f"Project created: {project_name} at {self.project_dir}")
            return True

        except Exception as e:
            logger.error(f"Failed to create project: {e}")
            return False

    def load_project(self, project_dir: str) -> bool:
        """加载现有项目.

        Args:
            project_dir: 项目目录

        Returns:
            True 如果成功，False 否则
        """
        try:
            self.project_dir = Path(project_dir)

            if not self.project_dir.exists():
                logger.error(f"Project directory not found: {project_dir}")
                return False

            # 加载项目配置
            if not self._load_project_config():
                logger.error("Failed to load project config")
                return False

            # 初始化核心模块
            self.class_manager = ClassManager(str(self.project_dir))
            self.model_manager = ModelManager(str(self.project_dir / "models"))
            self.batch_processor = BatchProcessor(
                self.model_manager,
                self.class_manager.get_all_classes(),
            )
            self.auto_labeling_engine = AutoLabelingEngine(
                self.model_manager,
                self.class_manager.get_all_classes(),
            )
            self.format_converter = FormatConverter(
                self.class_manager.get_all_classes()
            )

            logger.info(f"Project loaded: {self.project_config.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to load project: {e}")
            return False

    def save_project(self) -> bool:
        """保存项目.

        Returns:
            True 如果成功，False 否则
        """
        try:
            if self.class_manager:
                self.class_manager.save()

            self._save_project_config()

            logger.info("Project saved")
            return True

        except Exception as e:
            logger.error(f"Failed to save project: {e}")
            return False

    def auto_label_image(
        self,
        image_path: str,
        conf_threshold: Optional[float] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """自动标注单张图像.

        Args:
            image_path: 图像路径
            conf_threshold: 置信度阈值（可选）

        Returns:
            检测结果列表或 None
        """
        if not self.auto_labeling_engine:
            logger.error("AutoLabelingEngine not initialized")
            return None

        try:
            config = AutoLabelingConfig(
                model_name=self.project_config.model_name,
                conf_threshold=conf_threshold or self.project_config.conf_threshold,
            )

            result = self.auto_labeling_engine.auto_label(image_path, config)

            if result.validation_passed:
                logger.info(f"Auto-labeled {image_path}: {len(result.detections)} detections")
                return result.detections
            else:
                logger.warning(f"Validation failed: {result.validation_errors}")
                return None

        except Exception as e:
            logger.error(f"Failed to auto-label image: {e}")
            return None

    def batch_process(
        self,
        input_dir: str,
        output_dir: str,
        output_format: str = "yolo",
        progress_callback=None,
    ) -> Optional[List[ProcessingResult]]:
        """批量处理图像.

        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            output_format: 输出格式
            progress_callback: 进度回调函数

        Returns:
            处理结果列表或 None
        """
        if not self.batch_processor:
            logger.error("BatchProcessor not initialized")
            return None

        try:
            config = BatchProcessingConfig(
                model_name=self.project_config.model_name,
                input_dir=input_dir,
                output_dir=output_dir,
                conf_threshold=self.project_config.conf_threshold,
                output_format=output_format,
                device=self.project_config.device,
            )

            results = self.batch_processor.process_directory(config, progress_callback)

            logger.info(f"Batch processing completed: {len(results)} images processed")
            return results

        except Exception as e:
            logger.error(f"Failed to batch process: {e}")
            return None

    def convert_format(
        self,
        input_dir: str,
        output_dir: str,
        input_format: str,
        output_format: str,
        progress_callback=None,
    ) -> Optional[List[ConversionResult]]:
        """转换标注格式.

        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            input_format: 输入格式
            output_format: 输出格式
            progress_callback: 进度回调函数

        Returns:
            转换结果列表或 None
        """
        if not self.format_converter:
            logger.error("FormatConverter not initialized")
            return None

        try:
            results = self.format_converter.convert_folder(
                input_dir=input_dir,
                output_dir=output_dir,
                input_format=input_format,
                output_format=output_format,
                progress_callback=progress_callback,
            )

            logger.info(f"Format conversion completed: {len(results)} files converted")
            return results

        except Exception as e:
            logger.error(f"Failed to convert format: {e}")
            return None

    def get_statistics(self) -> Dict[str, Any]:
        """获取项目统计信息.

        Returns:
            统计信息字典
        """
        stats = {
            "project_name": self.project_config.name if self.project_config else None,
            "classes_count": self.class_manager.get_class_count() if self.class_manager else 0,
            "model_name": self.project_config.model_name if self.project_config else None,
        }

        if self.batch_processor:
            batch_stats = self.batch_processor.get_statistics()
            stats["batch_processing"] = batch_stats

        return stats

    def get_class_manager(self) -> Optional[ClassManager]:
        """获取类别管理器."""
        return self.class_manager

    def get_model_manager(self) -> Optional[ModelManager]:
        """获取模型管理器."""
        return self.model_manager

    def get_batch_processor(self) -> Optional[BatchProcessor]:
        """获取批量处理器."""
        return self.batch_processor

    def get_auto_labeling_engine(self) -> Optional[AutoLabelingEngine]:
        """获取自动标注引擎."""
        return self.auto_labeling_engine

    def get_format_converter(self) -> Optional[FormatConverter]:
        """获取格式转换器."""
        return self.format_converter

    def get_project_config(self) -> Optional[ProjectConfig]:
        """获取项目配置."""
        return self.project_config

    def set_project_config(self, config: ProjectConfig) -> None:
        """设置项目配置."""
        self.project_config = config
        self._save_project_config()

    def _save_project_config(self) -> None:
        """保存项目配置到文件."""
        if not self.project_dir or not self.project_config:
            return

        config_file = self.project_dir / "project.json"

        try:
            config_dict = asdict(self.project_config)
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=2)

            logger.debug(f"Project config saved to {config_file}")

        except Exception as e:
            logger.error(f"Failed to save project config: {e}")

    def _load_project_config(self) -> bool:
        """从文件加载项目配置."""
        if not self.project_dir:
            return False

        config_file = self.project_dir / "project.json"

        try:
            if not config_file.exists():
                logger.warning(f"Project config file not found: {config_file}")
                return False

            with open(config_file, "r", encoding="utf-8") as f:
                config_dict = json.load(f)

            self.project_config = ProjectConfig(**config_dict)
            logger.debug(f"Project config loaded from {config_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to load project config: {e}")
            return False
