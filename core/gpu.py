"""GPU / CUDA detection and monitoring module."""

import os
import subprocess
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from loguru import logger


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


def detect_cuda() -> CUDADetection:
    """Detect CUDA availability and GPU information."""
    result = CUDADetection()

    # 1. Check PyTorch CUDA
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
            result.recommended_device = "0"
        else:
            result.recommended_device = "cpu"
    except ImportError:
        result.error = "PyTorch not installed"
        result.recommended_device = "cpu"
    except Exception as e:
        result.error = f"PyTorch CUDA check failed: {e}"

    # 2. Try nvidia-smi for more details
    try:
        nvidia_info = _query_nvidia_smi()
        if nvidia_info:
            result.driver_version = nvidia_info.get("driver_version", "")
            for gpu_data in nvidia_info.get("gpus", []):
                idx = gpu_data.get("index", -1)
                found = False
                for gpu in result.gpus:
                    if gpu.index == idx:
                        gpu.vram_total_mb = gpu_data.get("vram_total_mb", gpu.vram_total_mb)
                        gpu.vram_used_mb = gpu_data.get("vram_used_mb", 0)
                        gpu.vram_free_mb = gpu_data.get("vram_free_mb", 0)
                        gpu.temperature = gpu_data.get("temperature", 0)
                        gpu.utilization = gpu_data.get("utilization", 0)
                        gpu.driver_version = gpu_data.get("driver_version", "")
                        found = True
                        break
                if not found:
                    result.gpus.append(GPUInfo(
                        index=idx,
                        name=gpu_data.get("name", "Unknown"),
                        vram_total_mb=gpu_data.get("vram_total_mb", 0),
                        vram_used_mb=gpu_data.get("vram_used_mb", 0),
                        vram_free_mb=gpu_data.get("vram_free_mb", 0),
                        temperature=gpu_data.get("temperature", 0),
                        utilization=gpu_data.get("utilization", 0),
                        driver_version=gpu_data.get("driver_version", ""),
                    ))
            if not result.cuda_version:
                result.cuda_version = nvidia_info.get("cuda_version", "")
            result.cuda_available = bool(result.gpus)
            result.gpu_count = len(result.gpus)
    except Exception as e:
        logger.debug(f"nvidia-smi query failed: {e}")

    # If torch says CUDA is available, trust that
    if result.torch_cuda_available:
        result.cuda_available = True

    # If nvidia-smi found GPUs but torch didn't, still set available
    if result.gpus and not result.cuda_available:
        result.cuda_available = True

    # Update recommended device if we have GPUs
    if result.cuda_available and result.gpus and result.recommended_device == "cpu":
        result.recommended_device = "0"

    return result


def _query_nvidia_smi() -> Optional[dict]:
    """Query nvidia-smi for GPU information."""
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
                    "index": int(parts[0]),
                    "name": parts[1],
                    "vram_total_mb": int(float(parts[2])),
                    "vram_used_mb": int(float(parts[3])),
                    "vram_free_mb": int(float(parts[4])),
                    "driver_version": parts[5],
                    "temperature": int(float(parts[6])),
                    "utilization": int(float(parts[7])),
                })

        # Get driver version
        driver_version = ""
        cmd2 = [nvidia_smi, "--query-gpu=driver_version", "--format=csv,noheader"]
        proc2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=5)
        if proc2.returncode == 0:
            driver_version = proc2.stdout.strip().split("\n")[0].strip()

        return {
            "driver_version": driver_version,
            "cuda_version": "",
            "gpus": gpus,
        }

    except FileNotFoundError:
        return None
    except subprocess.TimeoutExpired:
        return None
    except Exception as e:
        logger.debug(f"nvidia-smi error: {e}")
        return None



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
