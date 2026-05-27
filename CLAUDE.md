# Yolo Studio — Claude 工作指南

## 项目概述

基于 PyQt6 的 YOLO 目标检测标注与训练桌面应用。LabelImg 风格的极简标注体验 + 完整的训练/推理/导出流水线。

## 技术栈

- **UI**: PyQt6 (QMainWindow + QStackedWidget workspace)
- **训练**: ultralytics (YOLO)
- **推理**: ONNX Runtime / ultralytics
- **LLM**: OpenAI-compatible API (Qwen-VL-Max 等)
- **测试**: pytest (268 tests)
- **打包**: PyInstaller (onefile)

## 项目结构

```
Yolo-studio/
├── main.py              # 入口
├── core/                # 纯逻辑层 (无 Qt 依赖，可独立测试)
│   ├── annotation.py    # YOLO 标注解析 (BBox/OBBox/Polygon/Keypoint)
│   ├── annotation_utils.py  # 标注统计工具
│   ├── geometry_utils.py    # OBB 坐标转换
│   ├── image_utils.py       # 图片尺寸读取
│   ├── config.py        # YAML 配置管理 (Pydantic)
│   ├── project_manager.py   # 项目管理
│   ├── dataset.py       # 数据集解析
│   ├── augmentor.py     # 数据增强
│   ├── trainer.py       # 训练封装
│   ├── inference.py     # 推理封装
│   ├── exporter.py      # 模型导出
│   └── ...
├── gui/                 # 界面层
│   ├── main_window.py   # 主窗口 (~1300 lines, 待拆分)
│   ├── canvas.py        # 标注画布
│   ├── training_panel.py    # 训练面板
│   ├── inference_panel.py   # 推理面板
│   ├── export_panel.py      # 导出面板
│   ├── llm_handler.py       # LLM 自动标注
│   └── ...
└── tests/               # pytest 测试
```

## 编码约定

- Python 3.10+, 行宽 120, 2 空格缩进
- core 层不引入 PyQt6（保持纯逻辑可测试）
- 路径操作统一用 pathlib.Path
- 日志用 loguru (from loguru import logger)
- 注释用中文，面向国内用户

## Worker 线程管理

所有 QThread worker 必须用统一的 cleanup 模式：
```python
def _cleanup_worker(self):
    if self.worker is not None:
        if self.worker.isRunning():
            self.worker.quit()
            if not self.worker.wait(3000):
                self.worker.terminate()
                self.worker.wait(1000)
        self.worker.deleteLater()
        self.worker = None
```

原因：自定义 run() 的 worker 不响应 quit()，必须有 terminate() 兜底。
deleteLater() 在 quit()+wait() 后无效（事件循环已停），所以放在 terminate 之后。

## 异常处理

- 不允许裸 `except: pass` 或 `except Exception: pass`
- 每个 except 至少要有 `# harmless:` 注释说明为什么可以忽略，或用 logger 记录
- 资源打开优先用 `with` 语句（Image.open 等）

## 测试

```bash
python3 -m pytest tests/ -v
```

- 不 mock 文件系统：用 tmp_path fixture 写真实临时文件
- 不 mock 数据库/网络：用真实调用或 skip
- test_training_panel 中的 UI 测试标记为 SKIP（需 pytest-qt + X display）

## 内存注意事项

- canvas 切换图片时先置 None 释放旧缓冲区，再加载新图
- 批量操作注意 fd 耗尽：Image.open 必须 with，文件迭代器用完即关

## Git

- 远程: https://github.dpik.top/https://github.com/YDERO3452/Yolo-studio.git
- 禁止 amend 已发布的 commit
- 一个 commit 只做一件事（不要混打包配置和 bug 修复）
