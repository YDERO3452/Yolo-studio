"""GPU / CUDA detection and monitoring module."""

import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

# 模块级缓存：避免重复调用 nvidia-smi / torch
_cached_detection: Optional["CUDADetection"] = None
_cache_timestamp: float = 0.0
_CACHE_TTL_SECONDS: int = 300  # 5 minutes — auto-refresh stale cache


@dataclass
class GPUInfo:
    """Information about a single GPU device."""
    index: int
    name: str
    vram_total_mb: int = 0
    vram_used_mb: int = 0
    vram_free_mb: int = 0
    driver_version: str = ""
    cuda_version: str = ""
    compute_capability: str = ""
    temperature: int = 0
    utilization: int = 0  # percent


@dataclass
class CUDADetection:
    """Result of CUDA/GPU detection."""
    cuda_available: bool = False
    cuda_version: str = ""
    cudnn_version: str = ""
    driver_version: str = ""
    gpu_count: int = 0
    gpus: list[GPUInfo] = field(default_factory=list)
    torch_cuda_available: bool = False
    torch_version: str = ""
    recommended_device: str = "cpu"
    error: str = ""


def _find_nvidia_smi() -> str:
    """Find nvidia-smi executable across platforms.

    Delegates to env_setup._find_nvidia_smi which searches:
    - System PATH
    - Windows Registry (any install drive)
    - Environment variables (CUDA_PATH, etc.)
    - `where` / `which` command
    - Common paths across all drives
    - CUDA Toolkit directories across all drives
    """
    try:
        from core.env_setup import _find_nvidia_smi as _env_find
        return _env_find()
    except ImportError:
        pass

    # Fallback: simple search if env_setup is unavailable
    try:
        result = subprocess.run(
            ["nvidia-smi", "--help"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0:
            return "nvidia-smi"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return ""


def detect_cuda(*, force_refresh: bool = False) -> CUDADetection:
    """Detect CUDA availability and GPU information.

    cuda_available 和 recommended_device 严格以 torch.cuda.is_available() 为准。
    nvidia-smi 只用于补充 GPU 详情（VRAM、温度、利用率、驱动版本）。

    Results are cached for 5 minutes. Use force_refresh=True to bypass cache.
    """
    global _cached_detection, _cache_timestamp

    # Check cache: return if fresh enough and not force_refresh
    if _cached_detection is not None and not force_refresh:
        if time.time() - _cache_timestamp < _CACHE_TTL_SECONDS:
            return _cached_detection

    result = CUDADetection()

    # 1. Check PyTorch CUDA — 这是唯一的 cuda_available 判定来源
    try:
        import torch
        result.torch_version = torch.__version__
        result.torch_cuda_available = torch.cuda.is_available()
        if result.torch_cuda_available:
            result.cuda_version = torch.version.cuda or ""
            result.gpu_count = torch.cuda.device_count()
            for i in range(result.gpu_count):
                props = torch.cuda.get_device_properties(i)
                vram = getattr(props, "total_mem", 0)
                if vram:
                    vram = vram // (1024 * 1024)
                gpu = GPUInfo(
                    index=i,
                    name=props.name,
                    vram_total_mb=vram,
                    compute_capability=f"{props.major}.{props.minor}",
                )
                result.gpus.append(gpu)
    except ImportError:
        result.error = "PyTorch not installed"
    except Exception as e:
        result.error = f"PyTorch CUDA check failed: {e}"

    # 2. Try nvidia-smi for more details（仅补充信息，不影响 cuda_available）
    nvidia_smi_gpus: list[GPUInfo] = []
    try:
        nvidia_info = _query_nvidia_smi()
        if nvidia_info:
            result.driver_version = nvidia_info.get("driver_version", "")
            # 如果 torch 没有提供 CUDA 版本，用 nvidia-smi 的
            if not result.cuda_version:
                result.cuda_version = nvidia_info.get("cuda_version", "")
            for gpu_data in nvidia_info.get("gpus", []):
                idx = gpu_data.get("index", -1)
                nvidia_gpu = GPUInfo(
                    index=idx,
                    name=gpu_data.get("name", "Unknown"),
                    vram_total_mb=gpu_data.get("vram_total_mb", 0),
                    vram_used_mb=gpu_data.get("vram_used_mb", 0),
                    vram_free_mb=gpu_data.get("vram_free_mb", 0),
                    temperature=gpu_data.get("temperature", 0),
                    utilization=gpu_data.get("utilization", 0),
                    driver_version=gpu_data.get("driver_version", ""),
                )
                nvidia_smi_gpus.append(nvidia_gpu)
                # 合并详情到 torch 已发现的 GPU
                for gpu in result.gpus:
                    if gpu.index == idx:
                        gpu.vram_total_mb = nvidia_gpu.vram_total_mb or gpu.vram_total_mb
                        gpu.vram_used_mb = nvidia_gpu.vram_used_mb
                        gpu.vram_free_mb = nvidia_gpu.vram_free_mb
                        gpu.temperature = nvidia_gpu.temperature
                        gpu.utilization = nvidia_gpu.utilization
                        gpu.driver_version = nvidia_gpu.driver_version
                        break
    except Exception as e:
        logger.debug(f"nvidia-smi query failed: {e}")

    # 3. 判定 cuda_available 和 recommended_device — 严格以 torch 为准
    result.cuda_available = result.torch_cuda_available
    result.recommended_device = "0" if result.torch_cuda_available else "cpu"

    # 4. 如果 torch 没检测到 CUDA 但 nvidia-smi 有 GPU，给出诊断信息
    if not result.torch_cuda_available and nvidia_smi_gpus:
        # 保留 nvidia-smi 发现的 GPU 信息供 UI 显示
        if not result.gpus:
            result.gpus = nvidia_smi_gpus
            result.gpu_count = len(nvidia_smi_gpus)
        try:
            import torch  # noqa: F811
            if not result.torch_version:
                result.torch_version = torch.__version__
            torch_cuda = getattr(torch.version, "cuda", None)
            if torch_cuda:
                result.error = (
                    f"PyTorch 编译了 CUDA {torch_cuda} 但 torch.cuda.is_available() 返回 False。"
                    f"可能是驱动版本过低或 PyTorch CUDA 版本与驱动不兼容。"
                )
            else:
                result.error = "安装了 CPU 版本的 PyTorch，需重装 CUDA 版。"
        except ImportError:
            result.error = "PyTorch 未安装"
        except Exception:
            pass

    _cached_detection = result
    _cache_timestamp = time.time()
    return result


def _safe_int(value: str, default: int = 0) -> int:
    """Safely parse an int from nvidia-smi output, handling '[N/A]' and other edge cases.

    nvidia-smi can return '[N/A]' for temperature, utilization, etc. on
    virtual GPUs (vGPU), cloud instances (AWS/GCP), or certain driver versions.
    """
    if not value or value.strip().upper() in ("[N/A]", "N/A", "NA", "[NOT AVAIL]", ""):
        return default
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default


def _query_nvidia_smi() -> Optional[dict]:
    """Query nvidia-smi for GPU information.

    Returns driver_version, cuda_version (from nvidia-smi header), and
    per-GPU details (VRAM, temperature, utilization).
    """
    try:
        nvidia_smi = _find_nvidia_smi()
        if not nvidia_smi:
            return None

        # Query GPU info
        cmd = [
            nvidia_smi,
            "--query-gpu=index,name,memory.total,memory.used,memory.free,"
            "driver_version,temperature.gpu,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if proc.returncode != 0:
            return None

        gpus = []
        for line in proc.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 8:
                gpus.append({
                    "index": _safe_int(parts[0]),
                    "name": parts[1],
                    "vram_total_mb": _safe_int(parts[2]),
                    "vram_used_mb": _safe_int(parts[3]),
                    "vram_free_mb": _safe_int(parts[4]),
                    "driver_version": parts[5],
                    "temperature": _safe_int(parts[6]),
                    "utilization": _safe_int(parts[7]),
                })

        # Driver version already in each GPU's data from the first query
        driver_version = gpus[0]["driver_version"] if gpus else ""

        # Read CUDA version from nvidia-smi header output.
        # nvidia-smi prints: "NVIDIA-SMI 550.54  Driver Version: 550.54  CUDA Version: 12.4"
        # This is the max CUDA version the driver supports.
        cuda_version = _read_cuda_version_from_smi(nvidia_smi)

        return {
            "driver_version": driver_version,
            "cuda_version": cuda_version,
            "gpus": gpus,
        }

    except FileNotFoundError:
        return None
    except subprocess.TimeoutExpired:
        return None
    except Exception as e:
        logger.debug(f"nvidia-smi error: {e}")
        return None


def _read_cuda_version_from_smi(nvidia_smi: str) -> str:
    """Read the CUDA version directly from nvidia-smi header output.

    The nvidia-smi command always prints a header line like:
        NVIDIA-SMI 550.54.15  Driver Version: 550.54.15  CUDA Version: 12.4
    This CUDA Version represents the maximum CUDA version the installed
    driver supports.
    """
    try:
        proc = subprocess.run(
            [nvidia_smi],
            capture_output=True, text=True, timeout=5,
        )
        if proc.returncode == 0:
            # Match "CUDA Version: 12.4" or "CUDA Version: 13.0"
            match = re.search(r'CUDA Version:\s*(\d+\.\d+)', proc.stdout)
            if match:
                return match.group(1)
    except Exception as e:
        logger.debug(f"Failed to read CUDA version from nvidia-smi: {e}")
    return ""



def format_gpu_summary(detection: CUDADetection) -> str:
    """Format a human-readable GPU summary string."""
    lines = []

    if detection.error:
        lines.append(f"Error: {detection.error}")

    if detection.torch_version:
        lines.append(f"PyTorch: {detection.torch_version}")

    if detection.cuda_available:
        lines.append(f"CUDA: {detection.cuda_version}")
        if detection.driver_version:
            lines.append(f"Driver: {detection.driver_version}")
        lines.append(f"GPUs: {detection.gpu_count}")
        for gpu in detection.gpus:
            vr = f"{gpu.vram_total_mb}MB"
            if gpu.vram_free_mb:
                vr += f" (free: {gpu.vram_free_mb}MB)"
            temp = f" | {gpu.temperature}C" if gpu.temperature else ""
            util = f" | {gpu.utilization}%" if gpu.utilization else ""
            lines.append(f"  [{gpu.index}] {gpu.name} | {vr}{temp}{util}")
            if gpu.compute_capability:
                lines.append(f"       Compute Capability: {gpu.compute_capability}")
    else:
        lines.append("CUDA: Not available")
        lines.append("Device: CPU only")

    return "\n".join(lines)


def get_device() -> str:
    """返回推荐的推理/训练 device："0"（GPU）或 "cpu"。

    统一入口，所有需要选 device 的模块都应调用此函数。
    """
    return detect_cuda().recommended_device


def clear_cache():
    """清除检测缓存，下次 detect_cuda() 将重新检测。"""
    global _cached_detection, _cache_timestamp
    _cached_detection = None
    _cache_timestamp = 0.0
