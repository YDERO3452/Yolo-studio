"""Environment detection and auto-configuration module.

Detects GPU (NVIDIA / AMD / Intel), driver, CUDA, and PyTorch compatibility,
then generates appropriate installation commands for the user.

CUDA version detection strategy (in priority order):
  1. nvidia-smi output  (directly prints "CUDA Version: X.Y", always up-to-date)
  2. Driver version -> CUDA mapping table  (fallback when nvidia-smi is unavailable)
  3. CUDA Toolkit nvcc  (only tells what toolkit is installed, not driver limit)

GPU detection strategy:
  1. nvidia-smi  (most reliable for NVIDIA, gives driver + GPU + VRAM)
  2. torch.cuda  (only works if PyTorch with CUDA is installed)
  3. Windows WMI / wmic  (fallback, detects all GPU vendors)
  4. Linux /proc/driver/nvidia  (Linux fallback)

Typical workflow:
  1. detect_environment() -> EnvInfo
  2. diagnose_environment(env) -> list[DiagnosisItem]
  3. Generate pip install commands or show guidance
"""

import json
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.request import urlopen, Request
from urllib.error import URLError

from loguru import logger


# -----------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------
_APP_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DATA_DIR = _APP_DIR / "data"
_CACHE_DIR = _DATA_DIR / "cache"
_DRIVER_MAP_LOCAL = _DATA_DIR / "driver_cuda_map.json"  # Shipped with the app, manually updateable
_DRIVER_MAP_CACHE = _CACHE_DIR / "driver_cuda_map_remote.json"  # Downloaded from remote
_CACHE_TTL_SECONDS = 7 * 24 * 3600  # 7 days

# Remote JSON URL for driver-CUDA mapping.
# Set this to your own repo's raw file URL to enable remote updates.
# If the URL is unreachable, the local file or built-in table is used instead.
DRIVER_MAP_REMOTE_URL = ""
# Example: "https://raw.githubusercontent.com/YOUR_USER/yolo-studio/main/data/driver_cuda_map.json"


# -----------------------------------------------------------------------
# Built-in NVIDIA driver -> CUDA version mapping (fallback)
# Source: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/
# This table is ONLY used when nvidia-smi is unavailable.
# When nvidia-smi works, CUDA version is read directly from its output.
# Updated: 2026-05
# -----------------------------------------------------------------------
_BUILTIN_DRIVER_CUDA_MAP: list[tuple[str, str]] = [
    # Sorted from newest to oldest — CUDA 13.x
    ("590.00", "13.0"),
    ("580.00", "13.0"),
    ("570.00", "12.8"),
    # CUDA 12.x
    ("560.35", "12.6"),
    ("555.85", "12.5"),
    ("550.54", "12.4"),
    ("545.84", "12.3"),
    ("535.54", "12.2"),
    ("530.30", "12.1"),
    ("525.60", "12.0"),
    # CUDA 11.x
    ("520.61", "11.8"),
    ("515.43", "11.7"),
    ("510.39", "11.6"),
    ("495.29", "11.5"),
    ("470.42", "11.4"),
    ("465.19", "11.3"),
    ("460.27", "11.2"),
    ("455.23", "11.1"),
    ("450.36", "11.0"),
    # CUDA 10.x and older
    ("440.33", "10.2"),
    ("418.96", "10.1"),
    ("410.48", "10.0"),
    ("396.26",  "9.2"),
    ("384.81",  "9.0"),
    ("375.26",  "8.0"),
]


# -----------------------------------------------------------------------
# PyTorch wheel channels.
#
# Important:
# - We do NOT require the user's system CUDA Toolkit to match this.
#   PyTorch pip wheels bundle the needed CUDA runtime libraries.
# - We choose the highest PyTorch wheel CUDA channel that is <= the
#   NVIDIA driver's max supported CUDA version.
# - If the currently installed PyTorch CUDA build already works, we do
#   not recommend reinstalling just because the driver supports a newer
#   CUDA channel.
# -----------------------------------------------------------------------
_PYTORCH_CUDA_BUILDS: list[dict[str, str]] = [
    {"cuda": "13.0", "tag": "cu130"},
    {"cuda": "12.8", "tag": "cu128"},
    {"cuda": "12.6", "tag": "cu126"},
    {"cuda": "12.4", "tag": "cu124"},
    {"cuda": "12.1", "tag": "cu121"},
    {"cuda": "11.8", "tag": "cu118"},
]
_PYTORCH_PACKAGES = ("torch", "torchvision")


def _pip_install_prefix(*, force_reinstall: bool = False) -> str:
    prefix = f'"{sys.executable}" -m pip install '
    if force_reinstall:
        prefix += "--force-reinstall "
    return prefix


def _build_official_pytorch_cmd(wheel_tag: str, *, force_reinstall: bool = False) -> str:
    return (
        f"{_pip_install_prefix(force_reinstall=force_reinstall)}"
        f"{' '.join(_PYTORCH_PACKAGES)} --index-url https://download.pytorch.org/whl/{wheel_tag}"
    )


PYTORCH_INSTALL_COMMANDS: dict[str, str] = {
    build["cuda"]: _build_official_pytorch_cmd(build["tag"])
    for build in _PYTORCH_CUDA_BUILDS
}
PYTORCH_INSTALL_COMMANDS["cpu"] = _build_official_pytorch_cmd("cpu")

# Force-reinstall variants — used when a CPU-only torch is already present.
# Without --force-reinstall, pip sees the CPU wheel as "already satisfied"
# and silently skips the CUDA installation.
PYTORCH_FORCE_REINSTALL_COMMANDS: dict[str, str] = {
    build["cuda"]: _build_official_pytorch_cmd(build["tag"], force_reinstall=True)
    for build in _PYTORCH_CUDA_BUILDS
}
PYTORCH_FORCE_REINSTALL_COMMANDS["cpu"] = _build_official_pytorch_cmd("cpu", force_reinstall=True)

# Preferred CUDA versions for PyTorch install (in priority order)
# We try to match the highest available PyTorch build that the driver supports
_PREFERRED_CUDA_ORDER = [build["cuda"] for build in _PYTORCH_CUDA_BUILDS]


def _cuda_to_wheel_tag(cuda_version: str) -> str:
    normalized = str(cuda_version).strip()
    for build in _PYTORCH_CUDA_BUILDS:
        if build["cuda"] == normalized:
            return build["tag"]
    return "cu" + normalized.replace(".", "")


def _parse_version_float(version: str) -> Optional[float]:
    match = re.search(r"(\d+)(?:\.(\d+))?", str(version or ""))
    if not match:
        return None
    major = match.group(1)
    minor = match.group(2) or "0"
    try:
        return float(f"{major}.{minor}")
    except ValueError:
        return None


def get_python_wheel_tags() -> dict[str, str]:
    """Return wheel tags the user should download for this Python/platform."""
    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    if platform.system() == "Windows":
        platform_tag = "win_amd64" if platform.machine().endswith("64") else "win32"
    elif platform.system() == "Linux":
        platform_tag = "linux_x86_64" if platform.machine() in {"x86_64", "AMD64"} else platform.machine()
    elif platform.system() == "Darwin":
        platform_tag = "macosx"
    else:
        platform_tag = platform.machine() or "unknown"
    return {"python": py_tag, "abi": py_tag, "platform": platform_tag}


def _pytorch_index_url(wheel_tag: str) -> str:
    return f"https://download.pytorch.org/whl/{wheel_tag}"


def _plan_wheel_tag(plan: dict[str, str | bool]) -> str:
    wheel_tag = str(plan.get("wheel_tag") or "cpu")
    if wheel_tag in {"current", "cuda"}:
        wheel_tag = _cuda_to_wheel_tag(str(plan.get("cuda") or "cpu"))
    return wheel_tag


def _default_pytorch_download_dir(wheel_tag: str) -> Path:
    return Path.home() / "Downloads" / f"pytorch-{wheel_tag}"


def get_pytorch_download_url(env: "EnvInfo") -> str:
    """Return the official PyTorch wheel directory for this machine."""
    plan = get_pytorch_install_plan(env)
    return f"{_pytorch_index_url(_plan_wheel_tag(plan))}/"


def get_pytorch_install_commands(
    env: "EnvInfo",
    download_dir: str | Path | None = None,
) -> dict[str, str]:
    """Return matched online, download, and local install commands.

    The ``pip download`` command intentionally downloads package names instead
    of hard-coded wheel filenames. Pip then selects the correct torch and
    torchvision wheels for this Python ABI, platform, and PyTorch wheel channel,
    and also downloads any required dependency wheels.
    """
    plan = get_pytorch_install_plan(env)
    wheel_tag = _plan_wheel_tag(plan)
    index_url = _pytorch_index_url(wheel_tag)
    if download_dir is None:
        download_dir = _default_pytorch_download_dir(wheel_tag)

    download_dir_text = str(download_dir)
    package_args = " ".join(_PYTORCH_PACKAGES)
    force_reinstall = bool(plan.get("force_reinstall"))
    force_flag = "--force-reinstall " if force_reinstall else ""
    pip_prefix = f'"{sys.executable}" -m pip'
    wheel_tags = get_python_wheel_tags()
    wheel_marker = f"+{wheel_tag}"
    wheel_suffix = f"{wheel_tags['python']}-{wheel_tags['abi']}-{wheel_tags['platform']}"

    return {
        "wheel_tag": wheel_tag,
        "index_url": index_url,
        "download_url": f"{index_url}/",
        "torch_page": f"{index_url}/torch/",
        "torchvision_page": f"{index_url}/torchvision/",
        "packages": package_args,
        "download_dir": download_dir_text,
        "online_install": _build_official_pytorch_cmd(wheel_tag, force_reinstall=force_reinstall),
        "download_wheels": (
            f'{pip_prefix} download {package_args} --only-binary=:all: '
            f'--dest "{download_dir_text}" --index-url {index_url}'
        ),
        "install_local_dir": (
            f'{pip_prefix} install {force_flag}--no-index '
            f'--find-links "{download_dir_text}" {package_args}'
        ),
        "torch_pattern": f"torch-*{wheel_marker}-{wheel_suffix}.whl",
        "torchvision_pattern": f"torchvision-*{wheel_marker}-{wheel_suffix}.whl",
    }


def _select_pytorch_cuda_build(env: "EnvInfo") -> Optional[dict[str, str]]:
    """Select the best PyTorch CUDA wheel for this machine.

    This is based on the NVIDIA driver capability, not the locally installed
    CUDA Toolkit. PyTorch wheels carry their own CUDA runtime.

    Matching strategy:
    - Pick the highest PyTorch wheel whose CUDA version is <= the driver's
      max supported CUDA version.
    - If the driver supports a "gap" CUDA version (e.g. 12.3, 12.5) that has
      no corresponding PyTorch wheel, we automatically step down to the next
      available wheel and record this as a "downgrade" so the UI can explain
      the reason to the user.
    - For legacy GPU architectures (compute capability < 7.5), cap at cu126.
    """
    max_cuda = _parse_version_float(env.driver_max_cuda)
    if max_cuda is None:
        return None

    max_build_cuda = 12.6 if _has_legacy_nvidia_arch(env) else None
    downgrade = False

    for build in _PYTORCH_CUDA_BUILDS:
        build_cuda = _parse_version_float(build["cuda"])
        if max_build_cuda is not None and build_cuda is not None and build_cuda > max_build_cuda:
            continue
        if build_cuda is not None and max_cuda >= build_cuda:
            # Check if the driver's exact CUDA version has a matching wheel
            exact_match = str(env.driver_max_cuda).strip()
            if build["cuda"] != exact_match:
                downgrade = True
            # Attach downgrade info for the caller to use in reason text
            build_copy = dict(build)
            build_copy["downgrade"] = downgrade
            build_copy["driver_max_cuda"] = str(env.driver_max_cuda)
            return build_copy
    return None


def _has_legacy_nvidia_arch(env: "EnvInfo") -> bool:
    """Return True for NVIDIA architectures that should stay on CUDA 12.x wheels.

    Maxwell/Pascal/Volta class GPUs are better served by the legacy CUDA 12.6
    PyTorch wheels even if a newer driver is installed.
    """
    for gpu in env.gpus:
        if gpu.vendor != GPUVendor.NVIDIA or not gpu.compute_capability:
            continue
        try:
            major, minor = gpu.compute_capability.split(".", 1)
            cc = float(f"{int(major)}.{int(minor)}")
        except Exception:
            continue
        if cc < 7.5:
            return True
    return False


def get_pytorch_install_plan(env: "EnvInfo") -> dict[str, str | bool]:
    """Return a machine-specific PyTorch install recommendation."""
    need_force = env.pytorch_installed and not env.pytorch_cuda_available
    plan = {
        "backend": "cpu",
        "cuda": "",
        "wheel_tag": "cpu",
        "official_cmd": _build_official_pytorch_cmd("cpu", force_reinstall=need_force),
        "reason": "未检测到可用的 NVIDIA CUDA 环境，使用 CPU 版 PyTorch。",
        "already_ok": False,
        "force_reinstall": need_force,
    }

    if env.pytorch_installed and env.pytorch_cuda_available:
        driver_max = _parse_version_float(env.driver_max_cuda)
        torch_cuda = _parse_version_float(env.pytorch_cuda_version)
        if driver_max is None or torch_cuda is None or torch_cuda <= driver_max:
            plan.update({
                "backend": "current",
                "cuda": env.pytorch_cuda_version or "",
                "wheel_tag": _cuda_to_wheel_tag(env.pytorch_cuda_version) if env.pytorch_cuda_version else "cuda",
                "official_cmd": "",
                "reason": (
                    f"当前 PyTorch {env.pytorch_version} 的 CUDA "
                    f"{env.pytorch_cuda_version or ''} 已可用，无需重装。"
                ),
                "already_ok": True,
                "force_reinstall": False,
            })
            return plan
        # This is unusual because torch.cuda.is_available() normally implies
        # compatibility, but keep a repair path for inconsistent environments.
        need_force = True

    if env.has_nvidia_gpu:
        build = _select_pytorch_cuda_build(env)
        if build is not None:
            legacy_note = (
                " 当前 GPU 架构较旧，已自动选择兼容性更稳的 CUDA 12.x wheel。"
                if _has_legacy_nvidia_arch(env) else ""
            )
            # Explain downgrade: driver supports a CUDA version that has no
            # matching PyTorch wheel, so we picked the next lower one
            downgrade_note = ""
            if build.get("downgrade"):
                downgrade_note = (
                    f" 驱动支持 CUDA {build.get('driver_max_cuda', env.driver_max_cuda)} "
                    f"没有对应 PyTorch wheel，已自动匹配 CUDA {build['cuda']} ({build['tag']})。"
                )
            plan.update({
                "backend": "cuda",
                "cuda": build["cuda"],
                "wheel_tag": build["tag"],
                "official_cmd": _build_official_pytorch_cmd(build["tag"], force_reinstall=need_force),
                "reason": (
                    f"NVIDIA 驱动最高支持 CUDA {env.driver_max_cuda}，"
                    f"推荐安装 PyTorch {build['tag']} wheel。"
                    f"{downgrade_note}{legacy_note}"
                ),
                "already_ok": False,
                "force_reinstall": need_force,
            })
        elif env.driver_max_cuda:
            plan["reason"] = (
                f"NVIDIA 驱动最高只支持 CUDA {env.driver_max_cuda}，"
                "低于当前 PyTorch GPU wheel 的最低推荐版本，请先更新 NVIDIA 驱动。"
            )
        else:
            plan["reason"] = "检测到 NVIDIA GPU，但无法确定驱动支持的 CUDA 版本，请先安装或更新 NVIDIA 驱动。"
        return plan

    if env.has_amd_gpu and not env.is_windows:
        rocm_cmd = f"{_pip_install_prefix(force_reinstall=need_force)}torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2"
        plan.update({
            "backend": "rocm",
            "cuda": "ROCm 6.2",
            "wheel_tag": "rocm6.2",
            "official_cmd": rocm_cmd,
            "reason": "检测到 AMD GPU。Windows 不支持 ROCm；Linux 可尝试 ROCm 版 PyTorch。",
            "already_ok": False,
            "force_reinstall": need_force,
        })
    return plan


# -----------------------------------------------------------------------
# GPU vendor keywords
# -----------------------------------------------------------------------
NVIDIA_KEYWORDS = (
    "nvidia", "geforce", "rtx", "gtx", "quadro", "tesla",
    "titan", "mx1", "mx2", "mx3", "mx4", "mx5",
)
AMD_KEYWORDS = (
    "amd", "radeon", "rx ", "vega", "navi", "firepro",
    "rx5", "rx6", "rx7", "rx9", "instinct", "pro w",
)
INTEL_KEYWORDS = (
    "intel", "arc", "iris", "uhd", "hd graphics", "xe graphics",
)


# -----------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------

class GPUVendor:
    NVIDIA = "NVIDIA"
    AMD = "AMD"
    Intel = "Intel"
    UNKNOWN = "Unknown"


@dataclass
class GPUDevice:
    name: str
    vendor: str = GPUVendor.UNKNOWN
    driver_version: str = ""
    vram_mb: int = 0
    compute_capability: str = ""


@dataclass
class EnvInfo:
    os_name: str = ""
    os_version: str = ""
    os_arch: str = ""
    python_version: str = ""
    is_windows: bool = False

    gpus: list[GPUDevice] = field(default_factory=list)
    has_nvidia_gpu: bool = False
    has_amd_gpu: bool = False
    has_intel_gpu: bool = False

    nvidia_driver_installed: bool = False
    nvidia_driver_version: str = ""
    driver_max_cuda: str = ""  # Max CUDA version supported by this driver

    cuda_toolkit_version: str = ""

    pytorch_installed: bool = False
    pytorch_version: str = ""
    pytorch_cuda_version: str = ""
    pytorch_cuda_available: bool = False

    ultralytics_installed: bool = False
    ultralytics_version: str = ""


@dataclass
class DiagnosisItem:
    level: str  # "ok", "warning", "error", "info"
    title: str
    detail: str = ""
    action: str = ""


# -----------------------------------------------------------------------
# Remote cache for driver-CUDA mapping
# -----------------------------------------------------------------------

def _load_driver_cuda_map() -> list[tuple[str, str]]:
    """Load the driver->CUDA mapping table.

    Priority:
    1. Local static JSON file (data/driver_cuda_map.json, shipped with app)
    2. Remote cache (downloaded from DRIVER_MAP_REMOTE_URL, if configured)
    3. Built-in fallback table (hardcoded in this file)

    The local static JSON file can be manually updated by the user or
    by a future app update. The remote URL allows automatic updates
    without an app release (if configured).
    """
    # 1. Try local static JSON file (shipped with the app)
    local_data = _load_local_map()
    if local_data is not None:
        # Also try remote update in background for next time
        _try_update_cache_async()
        return local_data

    # 2. Try remote cache (previously downloaded)
    cache_data = _load_cached_map()
    if cache_data is not None:
        _try_update_cache_async()
        return cache_data

    # 3. Try remote fetch (if URL is configured)
    if DRIVER_MAP_REMOTE_URL:
        remote_data = _try_fetch_remote_map()
        if remote_data is not None:
            _save_cached_map(remote_data)
            return remote_data

    # 4. Built-in fallback table
    return _BUILTIN_DRIVER_CUDA_MAP


def _load_local_map() -> Optional[list[tuple[str, str]]]:
    """Load the local static driver map (shipped with the app)."""
    if not _DRIVER_MAP_LOCAL.exists():
        return None
    try:
        data = json.loads(_DRIVER_MAP_LOCAL.read_text(encoding="utf-8"))
        entries = data.get("entries", [])
        if not entries:
            return None
        return [(e["min_driver"], e["cuda_version"]) for e in entries]
    except Exception as e:
        logger.debug(f"Failed to load local driver map: {e}")
        return None


def _load_cached_map() -> Optional[list[tuple[str, str]]]:
    """Load remote-cached map if it exists and is within TTL."""
    if not _DRIVER_MAP_CACHE.exists():
        return None

    try:
        data = json.loads(_DRIVER_MAP_CACHE.read_text(encoding="utf-8"))
        ts = data.get("timestamp", 0)
        if time.time() - ts > _CACHE_TTL_SECONDS:
            return None  # expired

        entries = data.get("entries", [])
        if not entries:
            return None

        return [(e["min_driver"], e["cuda_version"]) for e in entries]
    except Exception as e:
        logger.debug(f"Failed to load driver map cache: {e}")
        return None


def _save_cached_map(entries: list[tuple[str, str]]):
    """Save mapping entries to local cache."""
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        data = {
            "timestamp": time.time(),
            "entries": [
                {"min_driver": drv, "cuda_version": cuda}
                for drv, cuda in entries
            ],
        }
        _DRIVER_MAP_CACHE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as e:
        logger.debug(f"Failed to save driver map cache: {e}")


def _try_fetch_remote_map() -> Optional[list[tuple[str, str]]]:
    """Attempt to fetch the driver-CUDA map from the remote URL."""
    if not DRIVER_MAP_REMOTE_URL:
        return None
    try:
        req = Request(DRIVER_MAP_REMOTE_URL, headers={"User-Agent": "YoloStudio/1.0"})
        with urlopen(req, timeout=5) as resp:
            raw = resp.read().decode("utf-8")
        data = json.loads(raw)
        entries = data.get("entries", [])
        if not entries:
            return None
        return [(e["min_driver"], e["cuda_version"]) for e in entries]
    except Exception as e:
        logger.debug(f"Failed to fetch remote driver map: {e}")
        return None


def _try_update_cache_async():
    """Fire-and-forget: try to update the cache from remote.

    We don't await the result — this just pre-populates for next launch.
    """
    try:
        import threading
        def _update():
            remote = _try_fetch_remote_map()
            if remote:
                _save_cached_map(remote)
        t = threading.Thread(target=_update, daemon=True)
        t.start()
    except Exception:
        pass


# Module-level: load map once
_DRIVER_CUDA_MAP: list[tuple[str, str]] = _load_driver_cuda_map()


# -----------------------------------------------------------------------
# Detection
# -----------------------------------------------------------------------

def detect_environment() -> EnvInfo:
    env = EnvInfo()
    env.os_name = platform.system()
    env.os_version = platform.version()
    env.os_arch = platform.machine()
    env.python_version = platform.python_version()
    env.is_windows = platform.system() == "Windows"

    # Strategy 1: nvidia-smi (most reliable for NVIDIA)
    _detect_nvidia_smi(env)

    # Strategy 2: PyTorch CUDA (if already installed)
    _detect_pytorch(env)

    # Strategy 3: Windows WMI / wmic (detects ALL GPU vendors)
    if env.is_windows:
        _detect_wmi_gpu(env)

    # Strategy 4: Linux /proc/driver/nvidia
    if not env.is_windows and not env.has_nvidia_gpu:
        _detect_linux_nvidia(env)

    # CUDA toolkit (nvcc)
    _detect_cuda_toolkit(env)

    # Ultralytics
    _detect_ultralytics(env)

    # Determine max CUDA version
    # Priority: nvidia-smi direct read > driver map lookup
    if env.nvidia_driver_version and not env.driver_max_cuda:
        env.driver_max_cuda = _driver_to_max_cuda(env.nvidia_driver_version)

    return env


def _detect_nvidia_smi(env: EnvInfo):
    """Detect GPU, driver, and CUDA info via nvidia-smi.

    Key improvement: nvidia-smi directly outputs the CUDA version
    that the driver supports, so we don't need a lookup table.
    """
    try:
        nvidia_smi = _find_nvidia_smi()
        if not nvidia_smi:
            return

        # Query GPU details
        cmd = [
            nvidia_smi,
            "--query-gpu=name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if proc.returncode != 0:
            return

        env.nvidia_driver_installed = True

        for line in proc.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                gpu = GPUDevice(
                    name=parts[0],
                    vendor=GPUVendor.NVIDIA,
                    driver_version=parts[1],
                    vram_mb=int(float(parts[2])) if parts[2] else 0,
                    compute_capability=parts[3],
                )
                env.gpus.append(gpu)
                if not env.nvidia_driver_version and parts[1]:
                    env.nvidia_driver_version = parts[1]

        env.has_nvidia_gpu = any(g.vendor == GPUVendor.NVIDIA for g in env.gpus)

        # **Directly read CUDA version from nvidia-smi header**
        # nvidia-smi outputs something like:
        #   NVIDIA-SMI 550.54.15  Driver Version: 550.54.15  CUDA Version: 12.4
        # This is the most reliable way — always up-to-date, no lookup needed
        if not env.driver_max_cuda:
            env.driver_max_cuda = _read_cuda_version_from_smi(nvidia_smi)

    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    except Exception as e:
        logger.debug(f"nvidia-smi detection failed: {e}")


def _read_cuda_version_from_smi(nvidia_smi: str) -> str:
    """Read the CUDA version directly from nvidia-smi header output.

    The nvidia-smi command always prints a header line like:
        NVIDIA-SMI 550.54.15  Driver Version: 550.54.15  CUDA Version: 12.4
    This CUDA Version represents the maximum CUDA version the installed
    driver supports, which is exactly what we need.
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


def _find_nvidia_smi() -> str:
    """Find nvidia-smi executable across platforms.

    Search strategy:
    1. System PATH (most common)
    2. Windows Registry (always written by NVIDIA installer, even on D: drive)
    3. Environment variables (CUDA_PATH, etc.)
    4. `where` / `which` command (broader PATH search)
    5. Common hard-coded paths (C:, D:, etc.)
    6. CUDA Toolkit directories (scan all drives)
    """
    # --- 1. Direct PATH check ---
    try:
        result = subprocess.run(
            ["nvidia-smi", "--help"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0:
            return "nvidia-smi"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # --- 2. Windows-specific searches ---
    if platform.system() == "Windows":
        found = _find_nvidia_smi_windows()
        if found:
            return found

    # --- 3. Linux-specific searches ---
    elif platform.system() == "Linux":
        found = _find_nvidia_smi_linux()
        if found:
            return found

    return ""


def _find_nvidia_smi_windows() -> str:
    """Find nvidia-smi on Windows using registry, env vars, and disk scan."""
    candidates: list[str] = []

    # --- Registry: NVIDIA installer always writes here ---
    reg_paths = _read_nvidia_registry_paths()
    candidates.extend(reg_paths)

    # --- Environment variables ---
    env_paths = _read_nvidia_env_paths()
    candidates.extend(env_paths)

    # --- `where` command (searches entire PATH, more thorough than direct run) ---
    try:
        proc = subprocess.run(
            ["where", "nvidia-smi.exe"],
            capture_output=True, text=True, timeout=5,
        )
        if proc.returncode == 0:
            for line in proc.stdout.strip().split("\n"):
                p = line.strip().strip('"')
                if p and os.path.isfile(p):
                    candidates.insert(0, p)  # highest priority
    except Exception:
        pass

    # --- Common hard-coded paths across all drives ---
    common_paths = _get_windows_common_paths()
    candidates.extend(common_paths)

    # --- CUDA Toolkit directories across all drives ---
    cuda_paths = _scan_cuda_toolkit_paths()
    candidates.extend(cuda_paths)

    # Return first existing file
    for path in candidates:
        if path and os.path.isfile(path):
            return path

    return ""


def _read_nvidia_registry_paths() -> list[str]:
    """Read NVIDIA installation paths from Windows Registry.

    NVIDIA installer always writes to these registry keys:
    - HKLM/SOFTWARE/NVIDIA Corporation/Install* (various sub-keys)
    - HKLM/SYSTEM/CurrentControlSet/Services/nvlddmkm (driver service)
    - HKLM/SOFTWARE/WOW6432Node/NVIDIA Corporation (32-bit compat)
    """
    paths = []
    try:
        import winreg  # type: ignore

        # Key 1: Driver service path — contains the actual install directory
        try:
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SYSTEM\CurrentControlSet\Services\nvlddmkm",
                0, winreg.KEY_READ,
            )
            try:
                image_path, _ = winreg.QueryValueEx(key, "ImagePath")
                # ImagePath is like "SystemRoot/System32/DriverStore/FileRepository/nv_dispi.inf_amd64_xxx/nvlddmkm.sys"
                # The NVSMI folder is typically in the same parent directory
                if image_path:
                    parent = Path(image_path).parent.parent
                    smi_candidate = parent / "NVSMI" / "nvidia-smi.exe"
                    if smi_candidate.exists():
                        paths.append(str(smi_candidate))
            except FileNotFoundError:
                pass
            winreg.CloseKey(key)
        except FileNotFoundError:
            pass

        # Key 2: NVIDIA Corporation install paths
        for reg_key_path in [
            r"SOFTWARE\NVIDIA Corporation",
            r"SOFTWARE\WOW6432Node\NVIDIA Corporation",
        ]:
            try:
                key = winreg.OpenKey(
                    winreg.HKEY_LOCAL_MACHINE, reg_key_path, 0, winreg.KEY_READ,
                )
                # Enumerate sub-keys looking for install directories
                idx = 0
                while True:
                    try:
                        subkey_name = winreg.EnumKey(key, idx)
                        idx += 1
                        if "install" in subkey_name.lower():
                            try:
                                subkey = winreg.OpenKey(key, subkey_name)
                                try:
                                    install_dir, _ = winreg.QueryValueEx(subkey, "InstallDir")
                                    if install_dir:
                                        p = Path(install_dir)
                                        # Check various relative locations
                                        for rel in [
                                            "NVSMI\\nvidia-smi.exe",
                                            "nvidia-smi.exe",
                                            "bin\\nvidia-smi.exe",
                                        ]:
                                            candidate = p / rel
                                            if candidate.exists():
                                                paths.append(str(candidate))
                                except FileNotFoundError:
                                    pass
                                # Also try Comonent
                                try:
                                    comp_dir, _ = winreg.QueryValueEx(subkey, "Component")
                                    if comp_dir:
                                        candidate = Path(comp_dir) / "nvidia-smi.exe"
                                        if candidate.exists():
                                            paths.append(str(candidate))
                                except FileNotFoundError:
                                    pass
                                winreg.CloseKey(subkey)
                            except FileNotFoundError:
                                pass
                    except OSError:
                        break
                winreg.CloseKey(key)
            except FileNotFoundError:
                pass

        # Key 3: Global NVSMI path from registry
        try:
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\NVIDIA Corporation\Global\NVSMI",
                0, winreg.KEY_READ,
            )
            try:
                nvsmi_path, _ = winreg.QueryValueEx(key, "Path")
                if nvsmi_path:
                    candidate = Path(nvsmi_path) / "nvidia-smi.exe"
                    if candidate.exists():
                        paths.append(str(candidate))
            except FileNotFoundError:
                pass
            winreg.CloseKey(key)
        except FileNotFoundError:
            pass

    except ImportError:
        # winreg not available (shouldn't happen on Windows)
        pass
    except Exception as e:
        logger.debug(f"Registry scan failed: {e}")

    return paths


def _read_nvidia_env_paths() -> list[str]:
    """Read NVIDIA-related paths from environment variables."""
    paths = []

    # CUDA_PATH — set by CUDA Toolkit installer
    cuda_path = os.environ.get("CUDA_PATH", "")
    if cuda_path:
        candidate = Path(cuda_path) / "bin" / "nvidia-smi.exe"
        if candidate.exists():
            paths.append(str(candidate))

    # CUDA_PATH_V12_4, CUDA_PATH_V11_8, etc.
    for key, value in os.environ.items():
        if key.startswith("CUDA_PATH_V") and value:
            candidate = Path(value) / "bin" / "nvidia-smi.exe"
            if candidate.exists():
                paths.append(str(candidate))

    # NVSMI_PATH — sometimes set by custom installs
    nvsmi_path = os.environ.get("NVSMI_PATH", "")
    if nvsmi_path:
        candidate = Path(nvsmi_path) / "nvidia-smi.exe"
        if candidate.exists():
            paths.append(str(candidate))

    # PATH — scan for nvidia-smi in all PATH entries
    path_env = os.environ.get("PATH", "")
    for entry in path_env.split(os.pathsep):
        if not entry:
            continue
        entry_lower = entry.lower()
        # Only check paths that look NVIDIA-related to avoid slow disk scan
        if any(k in entry_lower for k in ("nvidia", "cuda", "nvsmi")):
            candidate = Path(entry) / "nvidia-smi.exe"
            if candidate.exists():
                paths.append(str(candidate))

    return paths


def _get_windows_common_paths() -> list[str]:
    """Get common nvidia-smi paths across all drives."""
    paths = []

    # Get available drive letters
    drives = _get_windows_drives()

    for drive in drives:
        d = f"{drive}:"
        paths.extend([
            f"{d}\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe",
            f"{d}\\Program Files (x86)\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe",
            f"{d}\\NVIDIA\\NVSMI\\nvidia-smi.exe",
            f"{d}\\Windows\\System32\\nvidia-smi.exe",
        ])

    # Always include C: explicitly (most common)
    paths.insert(0, r"C:\Windows\System32\nvidia-smi.exe")
    paths.insert(0, r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe")

    return paths


def _get_windows_drives() -> list[str]:
    """Get available Windows drive letters. Returns empty list on non-Windows."""
    if platform.system() != "Windows":
        return []
    drives = []
    try:
        import string
        for letter in string.ascii_uppercase:
            drive = f"{letter}:\\"
            if os.path.exists(drive):
                drives.append(letter)
    except Exception:
        # Fallback: at least C and D
        drives = ["C", "D"]
    return drives


def _scan_cuda_toolkit_paths() -> list[str]:
    """Scan for CUDA Toolkit installations across all Windows drives."""
    if platform.system() != "Windows":
        return []
    paths = []
    drives = _get_windows_drives()

    for drive in drives:
        cuda_base = Path(f"{drive}:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA")
        if cuda_base.exists():
            for v_dir in sorted(cuda_base.iterdir(), reverse=True):
                candidate = v_dir / "bin" / "nvidia-smi.exe"
                if candidate.exists():
                    paths.append(str(candidate))

    return paths


def _find_nvidia_smi_linux() -> str:
    """Find nvidia-smi on Linux."""
    # --- `which` command ---
    try:
        proc = subprocess.run(
            ["which", "nvidia-smi"],
            capture_output=True, text=True, timeout=5,
        )
        if proc.returncode == 0:
            p = proc.stdout.strip()
            if p and os.path.isfile(p):
                return p
    except Exception:
        pass

    # --- Common paths ---
    common_paths = [
        "/usr/bin/nvidia-smi",
        "/usr/local/bin/nvidia-smi",
        "/usr/local/cuda/bin/nvidia-smi",
        "/opt/nvidia/bin/nvidia-smi",
        "/snap/bin/nvidia-smi",
    ]
    for path in common_paths:
        if os.path.isfile(path):
            return path

    # --- Environment variable CUDA_PATH ---
    cuda_path = os.environ.get("CUDA_PATH", "")
    if cuda_path:
        candidate = Path(cuda_path) / "bin" / "nvidia-smi"
        if candidate.exists():
            return str(candidate)

    # --- Scan /opt and /usr/local for CUDA installations ---
    for base in [Path("/opt"), Path("/usr/local")]:
        if not base.exists():
            continue
        try:
            for item in base.iterdir():
                if "cuda" in item.name.lower() or "nvidia" in item.name.lower():
                    candidate = item / "bin" / "nvidia-smi"
                    if candidate.exists():
                        return str(candidate)
        except Exception:
            pass

    return ""


def _find_column_index(header_parts: list[str], column_name: str) -> int:
    """Find the index of a column in wmic CSV header.

    wmic /format:csv reorders columns alphabetically,
    so we must parse the header to find correct indices.
    """
    for i, part in enumerate(header_parts):
        if part == column_name:
            return i
    return -1


def _detect_wmi_gpu(env: EnvInfo):
    """Detect ALL GPUs via Windows WMI."""
    # Try the wmi Python package first
    try:
        import wmi  # type: ignore
        c = wmi.WMI()
        for gpu in c.Win32_VideoController():
            _add_gpu_from_wmi(
                env,
                gpu.Name or "",
                gpu.DriverVersion or "",
                gpu.AdapterRAM or 0,
            )
        return
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"WMI Python package detection failed: {e}")

    # Fallback: wmic subprocess
    # NOTE: wmic /format:csv reorders columns ALPHABETICALLY!
    # "Name,DriverVersion,AdapterRAM" becomes "Node,AdapterRAM,DriverVersion,Name"
    try:
        cmd = [
            "wmic", "path", "win32_VideoController",
            "get", "Name,DriverVersion,AdapterRAM", "/format:csv",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if proc.returncode != 0:
            cmd_simple = [
                "wmic", "path", "win32_VideoController",
                "get", "Name,DriverVersion", "/format:csv",
            ]
            proc = subprocess.run(cmd_simple, capture_output=True, text=True, timeout=10)
            if proc.returncode != 0:
                return

        lines = proc.stdout.strip().split("\n")
        # Parse header to determine column order (wmic reorders alphabetically)
        header_parts = [p.strip().lower() for p in lines[0].split(",")] if lines else []
        name_idx = _find_column_index(header_parts, "name")
        drv_idx = _find_column_index(header_parts, "driverversion")
        ram_idx = _find_column_index(header_parts, "adapterram")

        for line in lines[1:]:  # Skip header row
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue

            name = parts[name_idx] if name_idx >= 0 and name_idx < len(parts) else ""
            drv = parts[drv_idx] if drv_idx >= 0 and drv_idx < len(parts) else ""
            ram_str = parts[ram_idx] if ram_idx >= 0 and ram_idx < len(parts) else "0"
            ram = int(ram_str) if ram_str.isdigit() else 0

            if not name:
                continue
            _add_gpu_from_wmi(env, name, drv, ram)

    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    except Exception as e:
        logger.debug(f"wmic subprocess GPU detection failed: {e}")


def _add_gpu_from_wmi(env: EnvInfo, name: str, driver_version: str, adapter_ram: int):
    """Add a GPU device from WMI data, deduplicating with existing results."""
    if not name or name.lower() == "unknown":
        return

    name_lower = name.lower()
    vendor = _classify_gpu_vendor(name)

    # Skip if already detected by nvidia-smi (more accurate)
    for existing in env.gpus:
        if existing.name.lower() == name_lower:
            return

    vram_mb = adapter_ram // (1024 * 1024) if adapter_ram > 0 else 0

    gpu = GPUDevice(
        name=name,
        vendor=vendor,
        driver_version=driver_version,
        vram_mb=vram_mb,
    )
    env.gpus.append(gpu)

    if vendor == GPUVendor.NVIDIA:
        env.has_nvidia_gpu = True
        if not env.nvidia_driver_installed:
            env.nvidia_driver_installed = True
            if not env.nvidia_driver_version:
                env.nvidia_driver_version = driver_version
    elif vendor == GPUVendor.AMD:
        env.has_amd_gpu = True
    elif vendor == GPUVendor.Intel:
        env.has_intel_gpu = True


def _classify_gpu_vendor(name: str) -> str:
    """Classify a GPU name into its vendor."""
    name_lower = name.lower()

    if any(k in name_lower for k in NVIDIA_KEYWORDS):
        return GPUVendor.NVIDIA
    if re.search(r'\b(rtx|gtx|gt)\s*\d{3,4}\b', name_lower):
        return GPUVendor.NVIDIA
    if re.search(r'\bnvidia\b', name_lower):
        return GPUVendor.NVIDIA

    if any(k in name_lower for k in AMD_KEYWORDS):
        return GPUVendor.AMD

    if any(k in name_lower for k in INTEL_KEYWORDS):
        return GPUVendor.Intel

    return GPUVendor.UNKNOWN


def _detect_linux_nvidia(env: EnvInfo):
    """Detect NVIDIA GPUs on Linux via /proc/driver/nvidia."""
    gpus_dir = Path("/proc/driver/nvidia/gpus")
    if not gpus_dir.exists():
        return

    try:
        for gpu_dir in gpus_dir.iterdir():
            if not gpu_dir.is_dir():
                continue
            info_file = gpu_dir / "information"
            if not info_file.exists():
                continue
            try:
                content = info_file.read_text(encoding="utf-8")
                model = ""
                for line in content.split("\n"):
                    if line.startswith("Model:"):
                        model = line.split(":", 1)[1].strip().strip("\t")
                if model:
                    gpu = GPUDevice(name=model, vendor=GPUVendor.NVIDIA)
                    env.gpus.append(gpu)
                    env.has_nvidia_gpu = True
            except Exception:
                pass

        version_file = Path("/proc/driver/nvidia/version")
        if version_file.exists():
            try:
                content = version_file.read_text(encoding="utf-8")
                match = re.search(r'Kernel Module\s+(\d+\.\d+\.\d+)', content)
                if match:
                    env.nvidia_driver_version = match.group(1)
                    env.nvidia_driver_installed = True
            except Exception:
                pass
    except Exception as e:
        logger.debug(f"Linux NVIDIA detection failed: {e}")


def _detect_cuda_toolkit(env: EnvInfo):
    nvcc_paths = ["nvcc"]
    if env.is_windows:
        cuda_base = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
        if cuda_base.exists():
            for v_dir in sorted(cuda_base.iterdir(), reverse=True):
                nvcc_candidate = v_dir / "bin" / "nvcc.exe"
                if nvcc_candidate.exists():
                    nvcc_paths.insert(0, str(nvcc_candidate))

    for nvcc in nvcc_paths:
        try:
            proc = subprocess.run(
                [nvcc, "--version"],
                capture_output=True, text=True, timeout=5,
            )
            if proc.returncode == 0:
                match = re.search(r"release (\d+\.\d+)", proc.stdout)
                if match:
                    env.cuda_toolkit_version = match.group(1)
                    return
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        except Exception:
            continue


def _detect_pytorch(env: EnvInfo):
    try:
        import torch
        env.pytorch_installed = True
        env.pytorch_version = torch.__version__
        env.pytorch_cuda_version = torch.version.cuda or ""
        env.pytorch_cuda_available = torch.cuda.is_available()

        if env.pytorch_cuda_available:
            existing_names = {g.name for g in env.gpus}
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                if props.name not in existing_names:
                    vram = getattr(props, "total_mem", 0)
                    if vram:
                        vram = vram // (1024 * 1024)
                    gpu = GPUDevice(
                        name=props.name,
                        vendor=GPUVendor.NVIDIA,
                        vram_mb=vram,
                        compute_capability=f"{props.major}.{props.minor}",
                    )
                    env.gpus.append(gpu)
                    existing_names.add(props.name)
                env.has_nvidia_gpu = True
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"PyTorch detection failed: {e}")


def _detect_ultralytics(env: EnvInfo):
    try:
        import ultralytics
        env.ultralytics_installed = True
        env.ultralytics_version = ultralytics.__version__
    except ImportError:
        pass
    except Exception:
        pass


# -----------------------------------------------------------------------
# Driver -> CUDA mapping (fallback only)
# -----------------------------------------------------------------------

def _parse_driver_version(version: str) -> tuple[int, ...]:
    """Parse a driver version string into a comparable tuple.

    Handles:
    1. nvidia-smi / Linux: "550.54.15" -> (550, 54)
    2. Windows WMI 4-part: "32.0.15.7697" -> (576, 97)
       Conversion: parts[2][1:] + parts[3][:2] = major, parts[3][2:] = minor
    3. Windows WMI 2-part: "560.94" -> (560, 94)
    """
    if not version:
        return (0,)

    version = re.sub(r'^[^0-9]*', '', version.strip())
    parts = version.split(".")

    if not parts or not parts[0]:
        return (0,)

    # Case 1: Standard nvidia-smi format (first part >= 100)
    try:
        major = int(parts[0])
        if major >= 100:
            minor = int(parts[1]) if len(parts) > 1 else 0
            return (major, minor)
    except ValueError:
        pass

    # Case 2: Windows WMI 4-part format: "32.0.15.7697"
    if len(parts) >= 4:
        try:
            p2 = parts[2]  # e.g. "15"
            p3 = parts[3]  # e.g. "7697"
            if len(p2) >= 2 and len(p3) >= 4:
                real_major = int(p2[1:] + p3[:2])  # "5" + "76" = 576
                real_minor = int(p3[2:])             # "97"
                return (real_major, real_minor)
            elif len(p3) >= 2:
                real_major = int(p3[:len(p3) - 2] or "0")
                real_minor = int(p3[-2:])
                return (real_major, real_minor)
        except (ValueError, IndexError):
            pass

    # Case 3: 2-part format
    if len(parts) >= 2:
        try:
            major = int(parts[0])
            minor = int(parts[1])
            if major >= 10:
                return (major, minor)
        except ValueError:
            pass

    # Fallback
    try:
        nums = [int(p) for p in parts if p.isdigit()]
        if nums:
            return tuple(nums[:2])
    except ValueError:
        pass

    return (0,)


def _driver_to_max_cuda(driver_version: str) -> str:
    """Map a NVIDIA driver version to the maximum supported CUDA version.

    NOTE: This is a FALLBACK. When nvidia-smi is available, the CUDA
    version is read directly from its output, which is always accurate
    and up-to-date. This function is only used when nvidia-smi is not
    accessible (e.g. driver installed but PATH not configured).

    The mapping table is loaded from:
    1. Local JSON cache (auto-updated from remote)
    2. Built-in fallback table
    """
    parsed = _parse_driver_version(driver_version)
    if not parsed or parsed[0] == 0:
        return ""

    major = parsed[0]
    minor = parsed[1] if len(parsed) > 1 else 0
    driver_val = f"{major}.{minor}"

    # Use the module-level map (which may have been updated from remote)
    for min_driver, cuda_ver in _DRIVER_CUDA_MAP:
        try:
            if float(driver_val) >= float(min_driver):
                return cuda_ver
        except ValueError:
            continue

    return "<8.0"


# -----------------------------------------------------------------------
# Diagnosis
# -----------------------------------------------------------------------

def diagnose_environment(env: EnvInfo) -> list[DiagnosisItem]:
    items: list[DiagnosisItem] = []

    # 1. No GPUs at all
    if not env.gpus:
        items.append(DiagnosisItem(
            level="warning",
            title="未检测到独立显卡",
            detail="未找到任何 GPU，将使用 CPU 模式（训练速度非常慢）。",
            action="如果您的电脑有 NVIDIA 显卡，请确认已安装驱动程序。",
        ))
    else:
        for gpu in env.gpus:
            vram = f" ({gpu.vram_mb}MB)" if gpu.vram_mb else ""
            items.append(DiagnosisItem(
                level="ok" if gpu.vendor == GPUVendor.NVIDIA else "info",
                title=f"检测到 GPU: {gpu.name}{vram}",
                detail=f"厂商: {gpu.vendor}",
            ))

    # 2. AMD GPU (no CUDA)
    if env.has_amd_gpu and not env.has_nvidia_gpu:
        items.append(DiagnosisItem(
            level="warning",
            title="检测到 AMD GPU，不支持 CUDA",
            detail=(
                "AMD GPU 不支持 NVIDIA CUDA，无法用于 PyTorch GPU 加速训练。\n"
                "ROCm (AMD 的 CUDA 替代方案) 目前仅支持 Linux。"
            ),
            action=(
                "方案1: 使用 CPU 训练 (较慢)\n"
                "方案2: Linux 用户可尝试 ROCm 版 PyTorch:\n"
                "  pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2\n"
                "方案3: 使用云 GPU 服务 (Google Colab, AutoDL 等)"
            ),
        ))

    # 3. Intel GPU
    if env.has_intel_gpu and not env.has_nvidia_gpu:
        items.append(DiagnosisItem(
            level="info",
            title="检测到 Intel GPU",
            detail="Intel GPU 可通过 Intel Extension for PyTorch 加速，但兼容性有限。",
            action=(
                "Intel Arc GPU 可尝试:\n"
                "  pip install intel-extension-for-pytorch\n"
                "详情: https://github.com/intel/intel-extension-for-pytorch"
            ),
        ))

    # 4. NVIDIA driver
    if env.has_nvidia_gpu and not env.nvidia_driver_installed:
        items.append(DiagnosisItem(
            level="error",
            title="NVIDIA 驱动未安装",
            detail="检测到 NVIDIA 显卡但未找到驱动程序，GPU 无法使用。",
            action="请前往 https://www.nvidia.com/Download/index.aspx 下载并安装最新驱动。",
        ))
    elif env.nvidia_driver_version:
        items.append(DiagnosisItem(
            level="ok",
            title=f"NVIDIA 驱动已安装: {env.nvidia_driver_version}",
            detail=f"该驱动最高支持 CUDA {env.driver_max_cuda}" if env.driver_max_cuda else "",
        ))

    # 5. Driver too old
    if env.nvidia_driver_version and env.driver_max_cuda:
        try:
            max_cuda = float(env.driver_max_cuda)
            if max_cuda < 11.0:
                items.append(DiagnosisItem(
                    level="error",
                    title="NVIDIA 驱动版本过旧",
                    detail=f"当前驱动仅支持 CUDA {env.driver_max_cuda}，需要 CUDA >= 11.8。",
                    action="请更新 NVIDIA 驱动: https://www.nvidia.com/Download/index.aspx",
                ))
            elif max_cuda < 11.8:
                items.append(DiagnosisItem(
                    level="warning",
                    title="NVIDIA 驱动版本偏低",
                    detail=f"当前驱动支持 CUDA 最高 {env.driver_max_cuda}，推荐 CUDA >= 11.8。",
                    action="建议更新 NVIDIA 驱动以支持 CUDA 12.x。",
                ))
        except ValueError:
            pass

    # 6. PyTorch
    torch_plan = get_pytorch_install_plan(env)
    if not env.pytorch_installed:
        items.append(DiagnosisItem(
            level="error",
            title="PyTorch 未安装",
            detail="YOLO Studio 依赖 PyTorch 进行模型训练和推理。",
            action=_get_pytorch_install_cmd(env),
        ))
    else:
        items.append(DiagnosisItem(
            level="ok",
            title=f"PyTorch 已安装: {env.pytorch_version}",
            detail=f"CUDA 版本: {env.pytorch_cuda_version or 'CPU only'}",
        ))

    if env.has_nvidia_gpu and not torch_plan.get("already_ok") and torch_plan.get("backend") == "cuda":
        items.append(DiagnosisItem(
            level="info",
            title=f"推荐 PyTorch 安装通道: {torch_plan.get('wheel_tag')}",
            detail=str(torch_plan.get("reason", "")),
            action=str(torch_plan.get("official_cmd", "")),
        ))
    elif torch_plan.get("already_ok"):
        items.append(DiagnosisItem(
            level="ok",
            title="PyTorch CUDA 匹配当前机器",
            detail=str(torch_plan.get("reason", "")),
        ))

    # 7. PyTorch CUDA mismatch
    if env.pytorch_installed and env.has_nvidia_gpu:
        if not env.pytorch_cuda_available:
            if env.pytorch_cuda_version:
                items.append(DiagnosisItem(
                    level="error",
                    title="PyTorch CUDA 不可用",
                    detail=(
                        f"PyTorch 编译了 CUDA {env.pytorch_cuda_version}，"
                        f"但 torch.cuda.is_available() 返回 False。\n"
                        f"可能原因: 驱动版本过低，或 PyTorch CUDA 版本与驱动不兼容。"
                    ),
                    action=_get_pytorch_install_cmd(env),
                ))
            else:
                items.append(DiagnosisItem(
                    level="error",
                    title="安装了 CPU 版本的 PyTorch",
                    detail=(
                        "当前 PyTorch 是 CPU 版本，无法使用 GPU 加速。\n"
                        "注意: 直接 pip install 会因 'already satisfied' 被跳过，"
                        "必须使用 --force-reinstall 强制替换。"
                    ),
                    action=_get_pytorch_install_cmd(env),
                ))
        elif env.driver_max_cuda and env.pytorch_cuda_version:
            try:
                pytorch_cuda = float(env.pytorch_cuda_version)
                driver_max = float(env.driver_max_cuda)
                if pytorch_cuda > driver_max:
                    items.append(DiagnosisItem(
                        level="error",
                        title="PyTorch CUDA 版本与驱动不兼容",
                        detail=(
                            f"PyTorch 使用 CUDA {env.pytorch_cuda_version}，"
                            f"但驱动最高仅支持 CUDA {env.driver_max_cuda}。"
                        ),
                        action=(
                            f"方案1: 更新 NVIDIA 驱动以支持 CUDA {env.pytorch_cuda_version}\n"
                            f"方案2: 重新安装匹配驱动的 PyTorch:\n"
                            f"  {_get_pytorch_install_cmd(env)}"
                        ),
                    ))
            except ValueError:
                pass

    # 8. GPU acceleration OK
    if env.pytorch_cuda_available:
        items.append(DiagnosisItem(
            level="ok",
            title="GPU 加速可用",
            detail=f"PyTorch CUDA {env.pytorch_cuda_version} 正常工作。",
        ))

    # 9. Ultralytics
    if not env.ultralytics_installed:
        items.append(DiagnosisItem(
            level="error",
            title="Ultralytics 未安装",
            detail="YOLO Studio 依赖 Ultralytics 提供 YOLO 模型支持。",
            action="pip install ultralytics",
        ))
    else:
        items.append(DiagnosisItem(
            level="ok",
            title=f"Ultralytics 已安装: {env.ultralytics_version}",
        ))

    # 10. Python version
    py_ver = tuple(int(x) for x in env.python_version.split(".")[:2])
    if py_ver < (3, 9):
        items.append(DiagnosisItem(
            level="error",
            title=f"Python 版本过低: {env.python_version}",
            detail="需要 Python >= 3.9。",
            action="请安装 Python 3.10 或更高版本。",
        ))
    elif py_ver < (3, 10):
        items.append(DiagnosisItem(
            level="warning",
            title=f"Python 版本: {env.python_version}",
            detail="推荐 Python 3.10+。",
        ))
    else:
        items.append(DiagnosisItem(
            level="ok",
            title=f"Python 版本: {env.python_version}",
        ))

    # 11. VRAM check
    for gpu in env.gpus:
        if gpu.vendor == GPUVendor.NVIDIA and gpu.vram_mb > 0:
            if gpu.vram_mb < 4000:
                items.append(DiagnosisItem(
                    level="warning",
                    title=f"显存较小: {gpu.name} ({gpu.vram_mb}MB)",
                    detail="建议: 减小 batch_size、使用 YOLOv8n、降低 imgsz。",
                ))
            elif gpu.vram_mb < 8000:
                items.append(DiagnosisItem(
                    level="info",
                    title=f"显存: {gpu.name} ({gpu.vram_mb}MB)",
                    detail="适合训练中小型模型 (YOLOv8n/s/m)。",
                ))

    # 12. Dual GPU (laptop)
    nvidia_gpus = [g for g in env.gpus if g.vendor == GPUVendor.NVIDIA]
    intel_gpus = [g for g in env.gpus if g.vendor == GPUVendor.Intel]
    if nvidia_gpus and intel_gpus:
        items.append(DiagnosisItem(
            level="info",
            title="检测到双显卡 (笔记本模式)",
            detail=(
                f"核显: {', '.join(g.name for g in intel_gpus)}\n"
                f"独显: {', '.join(g.name for g in nvidia_gpus)}"
            ),
            action="如需确保使用独显，请在 NVIDIA 控制面板中设置 YOLO Studio 使用高性能 GPU。",
        ))

    return items


def _get_pytorch_install_cmd(env: EnvInfo) -> str:
    """Return the appropriate pip install command for PyTorch.

    When a CPU-only PyTorch is already installed, automatically uses
    ``--force-reinstall`` so that pip actually replaces the CPU wheel
    with the CUDA one.  Without this flag, pip treats the CPU and CUDA
    wheels as the same package and silently skips installation.
    """
    plan = get_pytorch_install_plan(env)
    cmd = str(plan.get("official_cmd", ""))
    if cmd:
        return cmd
    return "# 当前 PyTorch CUDA 已可用，无需重装"


def _get_mirror_install_cmds(env: EnvInfo) -> list[str]:
    """Deprecated compatibility hook.

    Do not recommend domestic PyTorch mirrors: CUDA wheel availability is
    inconsistent. The UI now points users to the official wheel directory and
    supports installing local .whl files selected by the user.
    """
    return []


# -----------------------------------------------------------------------
# Formatting
# -----------------------------------------------------------------------

def format_diagnosis_summary(items: list[DiagnosisItem]) -> str:
    lines = []
    for item in items:
        icon = {"ok": "[OK]", "warning": "[!!]", "error": "[XX]", "info": "[--]"}.get(item.level, "  ")
        lines.append(f"{icon} {item.title}")
        if item.detail:
            for dl in item.detail.split("\n"):
                lines.append(f"    {dl}")
        if item.action:
            lines.append(f"    -> {item.action}")
        lines.append("")
    return "\n".join(lines)


def format_diagnosis_html(items: list[DiagnosisItem]) -> str:
    html_parts = []
    for item in items:
        color = {
            "ok": "#34C759", "warning": "#FFD60A",
            "error": "#FF453A", "info": "#5AC8FA",
        }.get(item.level, "#aaa")
        icon = {"ok": "&#10003;", "warning": "&#9888;", "error": "&#10007;", "info": "&#8505;"}.get(item.level, "")

        html = f'<p style="color:{color}; margin:4px 0;">{icon} <b>{item.title}</b></p>'
        if item.detail:
            detail_escaped = item.detail.replace("\n", "<br>")
            html += f'<p style="color:#aaa; margin:2px 0 2px 24px; font-size:12px;">{detail_escaped}</p>'
        if item.action:
            action_escaped = item.action.replace("\n", "<br>")
            html += f'<p style="color:#5AC8FA; margin:2px 0 2px 24px; font-size:12px;">-> {action_escaped}</p>'
        html_parts.append(html)
    return "".join(html_parts)


# -----------------------------------------------------------------------
# Quick API
# -----------------------------------------------------------------------

def check_environment() -> tuple[EnvInfo, list[DiagnosisItem]]:
    env = detect_environment()
    diag = diagnose_environment(env)
    return env, diag


def get_install_guide() -> str:
    env = detect_environment()
    diag = diagnose_environment(env)
    lines = ["=" * 50, "YOLO Studio Environment Check", "=" * 50, ""]
    lines.append(format_diagnosis_summary(diag))
    if env.has_nvidia_gpu:
        lines.append("=" * 50)
        lines.append("Recommended install:")
        lines.append("=" * 50)
        lines.append(f"  {_get_pytorch_install_cmd(env)}")
        lines.append("  pip install ultralytics")
    return "\n".join(lines)
