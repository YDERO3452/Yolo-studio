# YOLO Studio

[![License](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)
[![PyQt6](https://img.shields.io/badge/GUI-PyQt6-orange.svg)](https://www.riverbankcomputing.com/static/Docs/PyQt6/)
[![Platform](https://img.shields.io/badge/Platform-Windows_|_Linux-lightgrey.svg)](https://github.com)

YOLO 标注 / 训练 / 推理桌面工具。首页是节点画布：双击进模块，HUD 里可以整条链路跑起来。

[Releases](https://github.com/YDERO3452/Yolo-studio/releases/latest) · [Issues](https://github.com/YDERO3452/Yolo-studio/issues)

---

## 下载

| 平台 | 文件 |
|------|------|
| Windows | `YoloStudio.exe`（免安装） |
| Linux | `YoloStudio` → `chmod +x` 后运行 |

第一次跑会拉 YOLO 权重。要用 GPU，装好对应的 [CUDA 版 PyTorch](https://pytorch.org/get-started/locally/) 即可。

---

## 能干什么

- **画布工作流**：数据 → 标注 → 训练 → 结果 → 推理 → 导出；旁边有质检。子节点挂项目、抽帧、环境、格式转换等。
- **多任务**：detect / segment / pose / obb / classify（打开项目会跟着切标注模式和默认权重）。
- **标注**：框、多边形、旋转框、多点关键点；YOLO / LLM 自动标；`-seg` 能把 mask 落成多边形。
- **训练 / 推理 / 导出**：Ultralytics 训练与监控；单图、批量、视频、摄像头；导出 ONNX 等。
- **质检**：统计、质量检查；可跑数据增强，也可把推荐训练参数写回训练页。
- **格式**：YOLO ↔ VOC / COCO / DOTA。

本地跑，数据不用上传。

---

## 要求

- Windows 10/11 或 Linux（Ubuntu 20.04+）
- Python 3.10+
- 可选：NVIDIA + CUDA（没有也能 CPU 跑）
- 建议 8GB+ 内存

---

## 安装

**脚本（推荐）**

```bat
scripts\install.bat
```

```bash
bash scripts/install.sh
```

**手动**

国内镜像（可选）：

```bash
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple
pip config set global.trusted-host mirrors.aliyun.com
```

CPU：

```bash
pip install -r requirements.txt
```

GPU：先看 `nvidia-smi` 的 CUDA 版本，再装对应 wheel，例如：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

| wheel | 驱动大致要求 | CUDA |
|-------|-------------|------|
| cu118 | ≥ 450.36 | 11.8 |
| cu121 | ≥ 525.60 | 12.1 |
| cu124 | ≥ 545.84 | 12.4 |

启动：

```bash
python main.py
```

上手：建项目 → 导入图片 → 画布上双击节点。菜单在左上角 **⋯**。要串跑点 **运行**，中途可 **停止**。

---

## 目录

```text
Yolo-studio/
├─ core/       # 训练、推理、数据集、格式转换
├─ gui/        # 窗口、画布、节点、各面板
├─ tests/
├─ configs/
├─ scripts/
├─ freeze.py
└─ main.py
```

---

## 参考过

- [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling) — 标注交互、自动标注、格式转换
- [Ultralytics](https://docs.ultralytics.com/) — 训练 / 推理 / 数据集约定
- Qwen-VL 一类视觉模型 — LLM 辅助标注
- [PyQt6](https://www.riverbankcomputing.com/static/Docs/PyQt6/) / [OpenCV](https://docs.opencv.org/)
- 数据格式：[COCO](https://cocodataset.org/) · [VOC](http://host.robots.ox.ac.uk/pascal/VOC/) · [DOTA](https://captain-whu.github.io/DOTA/)

---

## 模型

仓库不含 `.pt`。首次用时 Ultralytics 会下；也可自己去[模型页](https://docs.ultralytics.com/models/)拿。常用名：`yolo11n.pt`、`-seg` / `-pose` / `-obb` / `-cls`，以及 YOLOv8、YOLO26、RT-DETR 等。

---

## GPU（打包版）

默认 CPU。要加速：

1. **⋯ → 工具 → 环境**（或双击「系统 / 环境」节点）看检测结果  
2. 点安装 CUDA 版 PyTorch，或自己执行：

```bash
pip install torch torchvision --force-reinstall --index-url https://download.pytorch.org/whl/cu121
```

驱动一般要 ≥ 525.60（CUDA 12.x）。`nvidia-smi` 能看到驱动支持的 CUDA 版本。pip 的 torch wheel 自带 CUDA 运行时，不必再装 Toolkit。

---

## 许可证

Copyright (C) 2024–2026 YDERO3452

GNU GPL v3.0，见 [LICENSE](LICENSE)。
