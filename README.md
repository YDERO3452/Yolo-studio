# YOLO Studio

![License](https://img.shields.io/badge/License-GPL--3.0-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10+-green.svg)
![PyQt6](https://img.shields.io/badge/GUI-PyQt6-orange.svg)

YOLO Studio 是一个基于 `PyQt6 + Ultralytics` 的桌面端标注与训练工作台，覆盖数据标注、数据集组织、训练、推理与导出流程。

本项目定位为工程化本地工具，强调：
- 单机可用
- 可扩展模块化结构
- 面向 YOLO 工作流的高频操作效率

## 功能概览

1. 标注工作台  
支持矩形框、多边形、旋转框、关键点；支持类别管理、文件队列、自动保存、批量自动标注。

2. 数据集管理  
支持 YOLO 目录结构管理、`data.yaml` 构建、数据切分与基础检查。

3. 训练面板  
基于 Ultralytics 训练接口，支持常见参数配置、训练日志与状态回传。

4. 推理面板  
支持图片、目录、视频与摄像头推理流程。

5. 导出与格式转换  
支持常见导出路径与 YOLO / VOC / COCO / DOTA 格式转换。

## 环境要求

- Python 3.10+
- Windows（当前项目主要在 Windows 环境下开发与测试）
- CUDA 环境（可选，CPU 也可运行）

## 安装

```bash
pip install -r requirements.txt
```

## 运行

```bash
python main.py
```

## 项目结构

```text
Yolo Studio/
├─ core/        # 训练、推理、数据集、格式转换、配置等核心逻辑
├─ gui/         # 主窗口、画布、面板、对话框、主题
├─ resources/   # 图标等静态资源
├─ models/      # 本地模型文件
├─ configs/     # 配置文件
└─ main.py      # 应用入口
```

## 参考与致谢

本项目在设计和实现过程中参考了以下开源项目与文档。

### 1) X-AnyLabeling（重点参考）

项目地址：  
`https://github.com/CVHub520/X-AnyLabeling`

参考点（主要是交互与工程组织思路）：

- 标注交互细节与操作习惯  
  参考位置：`gui/class_panel.py`, `gui/canvas.py`
  - 类别列表交互
  - 双击编辑标签
  - 十字辅助线交互

- 自动标注与批处理组织方式  
  参考位置：`core/auto_labeling_enhanced.py`, `core/batch_processor.py`
  - 配置驱动的自动标注流程
  - 批量任务进度与结果处理

- 格式转换模块组织方式  
  参考位置：`core/format_converter.py`
  - 多格式转换入口组织
  - 转换结果统一封装

- 类别与映射管理思路  
  参考位置：`core/class_manager.py`
  - 类别持久化
  - 名称映射管理

说明：本项目为独立实现，以上为设计思路与交互层面的参考来源说明。

### 2) Ultralytics YOLO

文档地址：  
`https://docs.ultralytics.com/`

参考点：
- 训练、推理、导出接口参数命名与使用方式
- YOLO 数据集目录约定与标注格式约定

相关实现位置：  
`core/config.py`, `core/trainer.py`, `core/inference.py`, `gui/annotation_io.py`, `core/dataset.py`

### 3) PyQt6 / Qt

参考文档：  
- `https://www.riverbankcomputing.com/static/Docs/PyQt6/`  
- `https://doc.qt.io/qt-6/`

参考点：
- 信号槽机制
- 自定义绘制与控件布局
- 主题样式组织

### 4) OpenCV

文档地址：  
`https://docs.opencv.org/`

参考点：
- 图像读写与处理
- 绘制与转换辅助

相关实现位置：  
`gui/canvas.py`, `core/annotation.py`

### 5) 数据格式标准

- COCO: `https://cocodataset.org/`  
- Pascal VOC: `http://host.robots.ox.ac.uk/pascal/VOC/`  
- DOTA: `https://captain-whu.github.io/DOTA/`

参考点：
- 标注字段定义
- 不同格式之间的转换约束

## 许可证

本项目采用 GNU General Public License v3.0 许可证 - 详见 [LICENSE](LICENSE) 文件

