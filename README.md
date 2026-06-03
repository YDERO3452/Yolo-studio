# YOLO Studio

<div align="center">

[![License](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)
[![PyQt6](https://img.shields.io/badge/GUI-PyQt6-orange.svg)](https://www.riverbankcomputing.com/static/Docs/PyQt6/)
[![Platform](https://img.shields.io/badge/Platform-Windows_|_Linux-lightgrey.svg)](https://github.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-purple.svg)](https://opencv.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-8.2+-yellow.svg)](https://ultralytics.com/)

**专业的 YOLO 系列模型标注与训练桌面工作台**

[![Release](https://img.shields.io/github/v/release/YDERO3452/Yolo-studio?label=Latest)](https://github.com/YDERO3452/Yolo-studio/releases/latest)
[![Downloads](https://img.shields.io/github/downloads/YDERO3452/Yolo-studio/total)](https://github.com/YDERO3452/Yolo-studio/releases/latest)

</div>

---

## 下载安装

前往 [Releases](https://github.com/YDERO3452/Yolo-studio/releases/latest) 下载最新版本：

| 平台 | 文件 | 说明 |
|------|------|------|
| Windows | `YoloStudio.exe` | 双击运行，无需安装 |
| Linux | `YoloStudio` | `chmod +x YoloStudio && ./YoloStudio` |

> 首次运行会自动下载 YOLO 模型，如需 GPU 加速请安装 [CUDA + PyTorch](https://pytorch.org/get-started/locally/)

---

## 界面预览

<!-- 截图放这里 -->
<!-- ![标注工作台](docs/screenshots/annotation.png) -->
<!-- ![训练面板](docs/screenshots/training.png) -->
<!-- ![推理面板](docs/screenshots/inference.png) -->

---

## 项目简介

YOLO Studio 是一个基于 PyQt6 和 Ultralytics YOLO 框架开发的桌面端应用程序，提供完整的目标检测工作流支持，包括数据标注、数据集管理、模型训练、推理测试和格式转换等功能。

### 核心特性

- **本地化部署**：完全离线可用，数据隐私安全
- **模块化架构**：清晰的代码结构，易于扩展和维护
- **高效工作流**：针对 YOLO 任务优化的操作流程
- **多格式支持**：支持 YOLO、VOC、COCO、DOTA 等主流格式

---

## 功能模块

### 1. 标注工作台

- 支持多种标注类型：矩形框、多边形、旋转框、关键点
- 智能类别管理与颜色配置
- 文件队列导航与快速切换
- 自动保存与批量自动标注功能

### 2. 数据集管理

- YOLO 标准目录结构自动生成
- data.yaml 配置文件构建
- 训练集/验证集/测试集自动划分
- 数据集完整性检查

### 3. 训练面板

- 基于 Ultralytics 训练引擎
- 可视化参数配置界面
- 实时训练日志与指标监控
- 训练状态实时反馈

### 4. 推理面板

- 支持单张图片、目录批量、视频文件和摄像头实时推理
- 多种 YOLO 模型版本支持（YOLOv8、YOLO26）
- 检测结果可视化与导出

### 5. 智能标注 (LLM)

- 集成 Qwen-VL 视觉语言模型
- 基于自然语言提示的自动目标检测
- 辅助人工标注，提升效率

### 6. 导出与格式转换

- 支持 YOLO、Pascal VOC、COCO、DOTA 格式互转
- 灵活的导出路径配置
- 批量转换处理

---

## 环境要求

- **操作系统**：Windows 10/11、Linux (Ubuntu 20.04+)
- **Python 版本**：3.10 或更高版本
- **GPU 支持**：CUDA 11.x+（可选，CPU 模式也可运行）
- **内存建议**：8GB RAM 以上

---

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 启动应用

```bash
python main.py
```

---

## 项目结构

```text
Yolo Studio/
├─ core/          # 训练、推理、数据集、格式转换、配置等核心逻辑
├─ gui/           # 主窗口、画布、面板、对话框、主题
├─ tests/         # pytest 测试
├─ configs/       # 配置文件
├─ scripts/       # 工具脚本
└─ main.py        # 应用入口
```

---

## 技术参考

本项目在开发过程中参考了以下优秀开源项目和技术文档：

### X-AnyLabeling

- **项目地址**：https://github.com/CVHub520/X-AnyLabeling
- **参考内容**：标注交互设计、自动标注流程、格式转换架构

### Ultralytics YOLO

- **官方文档**：https://docs.ultralytics.com/
- **参考内容**：训练/推理接口、数据集规范、模型导出

### Large Language Models (LLM)

- **参考实现**：OpenAI API / 阿里云通义千问
- **参考内容**：视觉语言模型目标检测、智能标注生成
- **已集成**：基于 Qwen-VL 的 LLM 自动标注功能，支持通过文本提示快速生成标注框。

### PyQt6

- **官方文档**：https://www.riverbankcomputing.com/static/Docs/PyQt6/
- **参考内容**：GUI 框架、信号槽机制、自定义控件

### OpenCV

- **官方文档**：https://docs.opencv.org/
- **参考内容**：图像处理、绘制函数、格式转换

### 数据格式标准

- **COCO**：https://cocodataset.org/
- **Pascal VOC**：http://host.robots.ox.ac.uk/pascal/VOC/
- **DOTA**：https://captain-whu.github.io/DOTA/

---

## 许可证

本项目采用 GNU General Public License v3.0 开源许可证。

详细信息请参阅 [LICENSE](LICENSE) 文件。

---

<div align="center">

**如有问题或建议，欢迎提交 Issue 或 Pull Request**

</div>

---

## 模型下载

模型文件（`.pt`）不包含在代码仓库中。首次运行时，Ultralytics 会自动下载所需模型。如需手动下载，可访问 [Ultralytics Models](https://docs.ultralytics.com/models/) 获取预训练权重。

### 支持的模型

| 系列 | 模型 | 用途 |
|------|------|------|
| YOLO26 | yolo26n/s/m/l/x.pt | 目标检测 |
| YOLO26 | yolo26n/s-seg.pt | 实例分割 |
| YOLO26 | yolo26n/s-pose.pt | 姿态估计 |
| YOLO26 | yolo26n/s-obb.pt | 旋转框检测 |
| YOLO11 | yolo11n/s/m/l/x.pt | 目标检测 |
| YOLOv8 | yolov8n/s/m/l/x.pt | 目标检测 |
| RT-DETR | rtdetr-l/x.pt | 实时检测 |

---

## CUDA 安装指南

如需 GPU 加速训练，请确保安装：
1. NVIDIA 驱动程序（推荐最新版本）
2. CUDA Toolkit 11.8+ 或 12.x
3. PyTorch with CUDA support

应用内置了环境检测功能（设置 → 环境检测），可自动诊断 GPU/CUDA/PyTorch 兼容性并给出安装建议。
