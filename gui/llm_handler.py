"""LLM-based auto-labeling via OpenAI-compatible API."""

import base64
import json
import re
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from loguru import logger
from PyQt6.QtCore import QThread, pyqtSignal

from freeze import get_resource_path, get_writable_dir

# Writable config path (user's saved settings)
LLM_CONFIG_PATH = get_writable_dir() / "config" / "llm_config.json"
# Bundled config path (defaults shipped with the app)
LLM_CONFIG_BUNDLE_PATH = get_resource_path("config/llm_config.json")

OLD_SYSTEM_PROMPT = "You are a precise object detection assistant. Output ONLY bounding boxes in the format: label,[xmin,ymin,xmax,ymax] with one box per line. Each coordinate value is a number between 0 and 1 (relative coordinates). Do NOT include any other text, explanation, or markdown formatting."
OLD_USER_PROMPT = "Detect all '{target}' objects in this image. Output only bounding boxes in the format: label,[xmin,ymin,xmax,ymax] per line."

EZYOLO_SYSTEM_PROMPT = """你是面向计算机视觉数据集的目标检测标注专家，仅输出指定目标的边界框坐标。
核心要求：
1. 目标类别：仅处理用户指定的类别，需找出图片中所有该类别实例；
2. 坐标格式：每个边界框以 [xmin, ymin, xmax, ymax] 格式输出；
3. 坐标范围：使用归一化坐标，每个值在 0 到 1 之间（相对于图片宽高），左上角为原点 (0,0)；
4. 输出格式：每行一个目标，格式为 "标签,[xmin,ymin,xmax,ymax]"；
5. 无目标时输出空内容；
6. 仅返回坐标数据，无任何说明文字。"""

EZYOLO_USER_PROMPT = """请检测图片中的所有 {target}，返回归一化坐标（0-1之间），格式每行一个：
{target},[xmin,ymin,xmax,ymax]
{target},[xmin,ymin,xmax,ymax]
..."""

# Old prompt versions — used for auto-migration
OLD_EZYOLO_SYSTEM_PROMPT = """你是面向计算机视觉数据集的目标检测标注专家，仅输出指定目标的边界框坐标。
核心要求：
1. 目标类别：仅处理用户指定的类别，需找出图片中所有该类别实例；
2. 坐标格式：每个边界框以 [xmin, ymin, xmax, ymax] 格式输出；
3. 输出格式：每行一个目标，格式为 "标签,[xmin,ymin,xmax,ymax]"；
4. 无目标时输出空内容；
5. 仅返回坐标数据，无任何说明文字。"""

OLD_EZYOLO_USER_PROMPT = """请检测图片中的所有 {target}，按以下格式返回每行一个：
{target},[xmin,ymin,xmax,ymax]
{target},[xmin,ymin,xmax,ymax]
..."""

# Free detection prompts — model discovers all objects and names them
FREE_DETECT_SYSTEM_PROMPT = """你是面向计算机视觉数据集的目标检测标注专家。请检测图片中所有显著物体并标注。

核心要求：
1. 识别图片中所有显著的前景物体（人、动物、车辆、家具、食物、电子产品等）；
2. 为每个物体起一个简短的中文名称（如 "人"、"汽车"、"杯子"、"狗"）；
3. 坐标使用归一化坐标，每个值在 0 到 1 之间（相对于图片宽高），左上角为原点 (0,0)；
4. 输出格式：每行一个目标，格式为 "名称,[xmin,ymin,xmax,ymax]"；
5. 无明显物体时输出空内容；
6. 仅返回坐标数据，无任何说明文字。"""

FREE_DETECT_USER_PROMPT = """请检测这张图片中的所有显著物体，返回归一化坐标（0-1之间），格式每行一个：
物体名称,[xmin,ymin,xmax,ymax]
物体名称,[xmin,ymin,xmax,ymax]
..."""

DEFAULT_LLM_CONFIG = {
    "api_key": "",
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "model_name": "qwen-vl-max",
    "system_prompt": EZYOLO_SYSTEM_PROMPT,
    "user_prompt": EZYOLO_USER_PROMPT,
}


def load_llm_config() -> dict:
    config = dict(DEFAULT_LLM_CONFIG)
    # Try writable location first (user's saved config)
    if LLM_CONFIG_PATH.exists():
        try:
            saved = json.loads(LLM_CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(saved, dict):
                config.update(saved)
        except Exception as exc:
            logger.warning(f"Failed to load LLM config: {exc}")
    # Fall back to bundled config (shipped with app)
    elif LLM_CONFIG_BUNDLE_PATH.exists():
        try:
            saved = json.loads(LLM_CONFIG_BUNDLE_PATH.read_text(encoding="utf-8"))
            if isinstance(saved, dict):
                config.update(saved)
        except Exception:
            # harmless: config file missing or malformed, use defaults
            pass
    # Migrate old prompts to current version (covers all known legacy prompts)
    system_prompt = config.get("system_prompt", "")
    if system_prompt in ("", OLD_SYSTEM_PROMPT, OLD_EZYOLO_SYSTEM_PROMPT) or "像素绝对坐标" in system_prompt:
        config["system_prompt"] = EZYOLO_SYSTEM_PROMPT
    user_prompt = config.get("user_prompt", "")
    if user_prompt in ("", OLD_USER_PROMPT, OLD_EZYOLO_USER_PROMPT) or "像素坐标" in user_prompt:
        config["user_prompt"] = EZYOLO_USER_PROMPT
    return config


def save_llm_config(config: dict):
    LLM_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LLM_CONFIG_PATH.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")


class LLMInferenceWorker(QThread):
    """Background worker for LLM-based auto-labeling."""
    finished = pyqtSignal(list)  # list of (class_name, x1, y1, x2, y2)
    error = pyqtSignal(str)

    def __init__(self, image_path: str, target_class: str, config: dict, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.target_class = target_class
        self.config = config

    def _encode_image(self) -> str:
        with open(self.image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def run(self):
        try:
            detections = self.infer_image(self.image_path, self.target_class, self.config)
            self.finished.emit(detections)
        except ImportError:
            self.error.emit("请安装 openai 库: pip install openai")
        except Exception as exc:
            self.error.emit(str(exc))

    @classmethod
    def infer_image(cls, image_path: str, target_class: str, config: dict) -> list:
        import openai

        # For local LLM servers (like LM Studio), api_key can be empty or any string
        api_key = config.get("api_key", "")
        if not api_key:
            api_key = "not-needed"  # LM Studio and other local servers accept any non-empty key

        client = openai.OpenAI(
            api_key=api_key,
            base_url=cls._normalize_base_url(config.get("base_url", "")),
        )

        image_b64 = cls._encode_image_file(image_path)
        mime_type = cls._image_mime_type(image_path)

        # Free detect mode: model discovers all objects, no pre-defined class
        if target_class == "__free_detect__":
            system_prompt = FREE_DETECT_SYSTEM_PROMPT
            user_prompt = FREE_DETECT_USER_PROMPT
        else:
            system_prompt = config.get("system_prompt", DEFAULT_LLM_CONFIG["system_prompt"])
            user_prompt = cls._format_user_prompt(
                config.get("user_prompt", DEFAULT_LLM_CONFIG["user_prompt"]), target_class,
            )

        try:
            response = client.chat.completions.create(
                model=config.get("model_name", "qwen-vl-max"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime_type};base64,{image_b64}"},
                            },
                        ],
                    },
                ],
            )
        except Exception as exc:
            if "Unsupported content type" in str(exc):
                raise RuntimeError(
                    "LLM 网关返回 Unsupported content type。当前 Base URL 的 /v1/models 可用，"
                    "但 /v1/chat/completions 不接受标准 OpenAI Chat 请求；"
                    "请换成支持 Chat Completions 多模态的 API 地址/模型，或检查该网关的调用格式。"
                ) from exc
            raise

        content = cls._extract_text_content(response)
        cls._raise_if_gateway_html(content)
        logger.debug(f"LLM response for {Path(image_path).name}:\n{content[:2000]}")
        return cls._parse_response(content)

    @staticmethod
    def _normalize_base_url(base_url: str) -> str:
        base_url = (base_url or "").strip().rstrip("/")
        if not base_url:
            return base_url
        parsed = urlparse(base_url)
        if parsed.scheme and parsed.netloc and parsed.path in ("", "/"):
            return f"{base_url}/v1"
        return base_url

    @staticmethod
    def _raise_if_gateway_html(content: str) -> None:
        head = (content or "").lstrip()[:500].lower()
        if head.startswith("<!doctype html") or "<html" in head:
            raise RuntimeError(
                "LLM API 返回了网页 HTML，不是 OpenAI-compatible API 响应。"
                "请检查 Base URL，应填写 API 地址，例如 https://.../v1。"
            )

    @staticmethod
    def _encode_image_file(image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    @staticmethod
    def _read_image_size(image_path: str) -> tuple[int, int]:
        """Read image dimensions from file header (fast, no full decode)."""
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                return img.size
        except Exception:
            return 0, 0

    @staticmethod
    def _image_mime_type(image_path: str) -> str:
        suffix = Path(image_path).suffix.lower()
        return {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".bmp": "image/bmp",
            ".webp": "image/webp",
        }.get(suffix, "image/jpeg")

    @staticmethod
    def _format_user_prompt(template: str, target_class: str) -> str:
        try:
            return template.format(target=target_class)
        except Exception:
            # harmless: str.format may fail on malformed template, fallback to replace
            return template.replace("{target}", target_class)

    @classmethod
    def _extract_text_content(cls, response: Any) -> str:
        """Extract assistant text from OpenAI-compatible responses.

        Some proxy endpoints return the SDK object, some return a dict-like
        payload, and a few return the assistant text directly as a string.
        """
        if response is None:
            return ""

        if isinstance(response, str):
            text = response.strip()
            if not text:
                return ""
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return text
            extracted = cls._extract_text_content(parsed)
            return extracted or text

        output_text = cls._get_value(response, "output_text")
        if output_text:
            return cls._content_to_text(output_text)

        choices = cls._get_value(response, "choices")
        if choices:
            choice = choices[0] if isinstance(choices, (list, tuple)) else choices
            message = cls._get_value(choice, "message")
            if message is not None:
                content = cls._get_value(message, "content")
                return cls._content_to_text(content)
            content = cls._get_value(choice, "content") or cls._get_value(choice, "text")
            return cls._content_to_text(content)

        content = cls._get_value(response, "content")
        if content is not None:
            return cls._content_to_text(content)

        return str(response)

    @staticmethod
    def _get_value(source: Any, key: str) -> Any:
        if isinstance(source, dict):
            return source.get(key)
        return getattr(source, key, None)

    @classmethod
    def _content_to_text(cls, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue
                text = cls._get_value(item, "text")
                if isinstance(text, str):
                    parts.append(text)
                    continue
                if isinstance(text, dict):
                    value = text.get("value")
                    if isinstance(value, str):
                        parts.append(value)
            return "\n".join(parts)
        return str(content)

    @staticmethod
    def _parse_response(content: str) -> list:
        """Parse LLM response into (class_name, x1, y1, x2, y2) tuples."""
        detections = []
        content = content.strip()
        if not content:
            return detections

        # Some multimodal APIs return JSON despite prompt constraints.
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                parsed = parsed.get("detections") or parsed.get("boxes") or parsed.get("objects") or []
            if isinstance(parsed, list):
                for item in parsed:
                    det = LLMInferenceWorker._coerce_json_detection(item)
                    if det is not None:
                        detections.append(det)
                if detections:
                    return detections
        except Exception as e:
            logger.debug(f"JSON detection parsing failed, falling back to regex: {e}")

        pattern = r'([^,\[\]\n]+?)\s*,\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]'
        for match in re.finditer(pattern, content):
            label = match.group(1).strip()
            x1 = float(match.group(2))
            y1 = float(match.group(3))
            x2 = float(match.group(4))
            y2 = float(match.group(5))
            detections.append((label, x1, y1, x2, y2))
        return detections

    @staticmethod
    def _coerce_json_detection(item) -> Optional[tuple]:
        if not isinstance(item, dict):
            return None
        label = item.get("label") or item.get("class") or item.get("name")
        box = item.get("bbox") or item.get("box") or item.get("xyxy")
        if isinstance(box, dict):
            box = [box.get("xmin"), box.get("ymin"), box.get("xmax"), box.get("ymax")]
        if not label or not isinstance(box, (list, tuple)) or len(box) < 4:
            return None
        try:
            return (str(label).strip(), float(box[0]), float(box[1]), float(box[2]), float(box[3]))
        except (TypeError, ValueError):
            return None


class LLMBatchInferenceWorker(QThread):
    """Background worker for EzYOLO-style LLM batch auto-labeling."""
    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(dict)  # image_path -> detections
    error = pyqtSignal(str)

    def __init__(self, image_paths: list[str], target_class: str, config: dict, parent=None):
        super().__init__(parent)
        self.image_paths = image_paths
        self.target_class = target_class
        self.config = config
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        try:
            results = {}
            total = len(self.image_paths)
            for index, image_path in enumerate(self.image_paths, start=1):
                if self._stop:
                    break
                self.progress.emit(index, total, image_path)
                results[image_path] = LLMInferenceWorker.infer_image(image_path, self.target_class, self.config)
            self.finished.emit(results)
        except ImportError:
            self.error.emit("请安装 openai 库: pip install openai")
        except Exception as exc:
            self.error.emit(str(exc))
