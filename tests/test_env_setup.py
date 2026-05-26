"""Tests for core/env_setup.py — pure logic only, no hardware required."""

import platform
import sys

import pytest

from core.env_setup import (
    EnvInfo,
    GPUDevice,
    GPUVendor,
    _classify_gpu_vendor,
    _driver_to_max_cuda,
    _has_legacy_nvidia_arch,
    _lookup_compute_capability,
    _parse_driver_version,
    _parse_version_float,
    get_pytorch_install_plan,
    get_python_wheel_tags,
)


# ---------------------------------------------------------------------------
# _parse_version_float
# ---------------------------------------------------------------------------

class TestParseVersionFloat:
    def test_simple(self):
        assert _parse_version_float("12.4") == 12.4

    def test_major_only(self):
        assert _parse_version_float("13") == 13.0

    def test_none(self):
        assert _parse_version_float(None) is None

    def test_empty(self):
        assert _parse_version_float("") is None

    def test_garbage(self):
        assert _parse_version_float("abc") is None

    def test_embedded_number(self):
        assert _parse_version_float("CUDA Version: 11.8") == 11.8


# ---------------------------------------------------------------------------
# _parse_driver_version
# ---------------------------------------------------------------------------

class TestParseDriverVersion:
    def test_nvidia_smi_format(self):
        # "550.54.15" -> (550, 54)
        assert _parse_driver_version("550.54.15") == (550, 54)

    def test_windows_wmi_4part(self):
        # "32.0.15.7697" -> parts[2][1:] + parts[3][:2] = "5"+"76"=576, "97"
        result = _parse_driver_version("32.0.15.7697")
        assert result[0] >= 500

    def test_2part_format(self):
        assert _parse_driver_version("560.94") == (560, 94)

    def test_empty(self):
        assert _parse_driver_version("") == (0,)

    def test_leading_text(self):
        # Should strip non-numeric prefix
        assert _parse_driver_version("Driver: 535.54")[0] == 535


# ---------------------------------------------------------------------------
# _classify_gpu_vendor
# ---------------------------------------------------------------------------

class TestClassifyGPUVendor:
    def test_nvidia_geforce(self):
        assert _classify_gpu_vendor("NVIDIA GeForce RTX 3060") == GPUVendor.NVIDIA

    def test_nvidia_quadro(self):
        assert _classify_gpu_vendor("Quadro P4000") == GPUVendor.NVIDIA

    def test_nvidia_tesla(self):
        assert _classify_gpu_vendor("Tesla T4") == GPUVendor.NVIDIA

    def test_nvidia_rtx_pattern(self):
        assert _classify_gpu_vendor("RTX 4090") == GPUVendor.NVIDIA

    def test_amd_radeon(self):
        assert _classify_gpu_vendor("AMD Radeon RX 7900 XTX") == GPUVendor.AMD

    def test_intel_arc(self):
        assert _classify_gpu_vendor("Intel Arc A770") == GPUVendor.Intel

    def test_intel_uhd(self):
        assert _classify_gpu_vendor("Intel(R) UHD Graphics 630") == GPUVendor.Intel

    def test_unknown(self):
        assert _classify_gpu_vendor("Some Random VGA Controller") == GPUVendor.UNKNOWN


# ---------------------------------------------------------------------------
# _lookup_compute_capability
# ---------------------------------------------------------------------------

class TestLookupComputeCapability:
    def test_rtx_5090(self):
        assert _lookup_compute_capability("NVIDIA GeForce RTX 5090") == "10.0"

    def test_rtx_4090(self):
        assert _lookup_compute_capability("NVIDIA GeForce RTX 4090") == "8.9"

    def test_rtx_a4000(self):
        assert _lookup_compute_capability("NVIDIA RTX A4000") == "8.0"

    def test_rtx_2080(self):
        assert _lookup_compute_capability("NVIDIA GeForce RTX 2080 Ti") == "7.5"

    def test_gtx_1660(self):
        assert _lookup_compute_capability("NVIDIA GeForce GTX 1660 SUPER") == "7.5"

    def test_tesla_v100(self):
        assert _lookup_compute_capability("Tesla V100-SXM2-32GB") == "7.0"

    def test_gtx_1080(self):
        assert _lookup_compute_capability("NVIDIA GeForce GTX 1080 Ti") == "6.1"

    def test_gt_1030(self):
        assert _lookup_compute_capability("NVIDIA GeForce GT 1030") == "6.1"

    def test_gtx_980(self):
        assert _lookup_compute_capability("NVIDIA GeForce GTX 980 Ti") == "5.2"

    def test_gtx_750(self):
        assert _lookup_compute_capability("NVIDIA GeForce GTX 750 Ti") == "5.2"

    def test_gtx_680(self):
        assert _lookup_compute_capability("NVIDIA GeForce GTX 680") == "3.0"

    def test_unknown(self):
        assert _lookup_compute_capability("Unknown GPU Device") == ""

    def test_non_nvidia(self):
        assert _lookup_compute_capability("Intel(R) UHD Graphics 630") == ""


# ---------------------------------------------------------------------------
# _has_legacy_nvidia_arch
# ---------------------------------------------------------------------------

class TestHasLegacyNvidiaArch:
    def _env(self, gpus):
        e = EnvInfo()
        e.gpus = gpus
        return e

    def test_pascal_is_legacy(self):
        gpu = GPUDevice(name="GTX 1080", vendor=GPUVendor.NVIDIA, compute_capability="6.1")
        assert _has_legacy_nvidia_arch(self._env([gpu])) is True

    def test_turing_is_not_legacy(self):
        gpu = GPUDevice(name="RTX 2060", vendor=GPUVendor.NVIDIA, compute_capability="7.5")
        assert _has_legacy_nvidia_arch(self._env([gpu])) is False

    def test_ampere_is_not_legacy(self):
        gpu = GPUDevice(name="RTX 3080", vendor=GPUVendor.NVIDIA, compute_capability="8.6")
        assert _has_legacy_nvidia_arch(self._env([gpu])) is False

    def test_unknown_cap_is_legacy_safety_net(self):
        """When ALL NVIDIA GPUs have unknown compute capability, default to legacy."""
        gpu = GPUDevice(name="Unknown NVIDIA", vendor=GPUVendor.NVIDIA, compute_capability="")
        assert _has_legacy_nvidia_arch(self._env([gpu])) is True

    def test_mixed_known_unknown(self):
        """One known modern GPU + one unknown → not legacy (known GPU wins)."""
        gpu1 = GPUDevice(vendor=GPUVendor.NVIDIA, compute_capability="8.6", name="RTX 3060")
        gpu2 = GPUDevice(vendor=GPUVendor.NVIDIA, compute_capability="", name="Unknown NVIDIA")
        assert _has_legacy_nvidia_arch(self._env([gpu1, gpu2])) is False

    def test_amd_is_skipped(self):
        gpu = GPUDevice(name="Radeon RX 6800", vendor=GPUVendor.AMD, compute_capability="")
        assert _has_legacy_nvidia_arch(self._env([gpu])) is False

    def test_no_gpus(self):
        assert _has_legacy_nvidia_arch(self._env([])) is False


# ---------------------------------------------------------------------------
# _driver_to_max_cuda
# ---------------------------------------------------------------------------

class TestDriverToMaxCuda:
    def test_recent_driver(self):
        # Driver 550.54 → should map to >= CUDA 12.4
        cuda = _driver_to_max_cuda("550.54.15")
        assert cuda != ""
        assert float(cuda) >= 12.4

    def test_old_driver(self):
        cuda = _driver_to_max_cuda("440.33")
        assert cuda == "10.2"

    def test_very_old_driver(self):
        cuda = _driver_to_max_cuda("375.26")
        assert cuda == "8.0"

    def test_empty_driver(self):
        assert _driver_to_max_cuda("") == ""


# ---------------------------------------------------------------------------
# get_pytorch_install_plan
# ---------------------------------------------------------------------------

class TestGetPytorchInstallPlan:
    def _env(self, **kwargs):
        e = EnvInfo()
        for k, v in kwargs.items():
            setattr(e, k, v)
        return e

    def test_no_gpu_no_pytorch(self):
        plan = get_pytorch_install_plan(self._env())
        assert plan["backend"] == "cpu"
        assert "CPU" in plan["reason"]

    def test_pytorch_cuda_already_ok(self):
        plan = get_pytorch_install_plan(self._env(
            pytorch_installed=True,
            pytorch_cuda_available=True,
            pytorch_cuda_version="12.4",
            pytorch_version="2.5.0",
            driver_max_cuda="12.4",
        ))
        assert plan["already_ok"] is True
        assert "无需重装" in plan["reason"]

    def test_nvidia_driver_too_old(self):
        plan = get_pytorch_install_plan(self._env(
            has_nvidia_gpu=True,
            gpus=[GPUDevice(compute_capability="8.6", vendor=GPUVendor.NVIDIA, name="RTX 3060")],
            driver_max_cuda="9.0",
            nvidia_driver_version="390.00",
        ))
        assert "更新 NVIDIA 驱动" in plan["reason"]
        assert plan["backend"] == "cpu"

    def test_cpu_pytorch_installed_force_reinstall(self):
        plan = get_pytorch_install_plan(self._env(
            pytorch_installed=True,
            pytorch_cuda_available=False,
            pytorch_version="2.5.0",
            has_nvidia_gpu=True,
            gpus=[GPUDevice(compute_capability="8.9", vendor=GPUVendor.NVIDIA, name="RTX 4090")],
            driver_max_cuda="12.8",
        ))
        assert plan["force_reinstall"] is True
        assert plan["backend"] == "cuda"
        assert "force-reinstall" in str(plan.get("official_cmd", ""))

    def test_nvidia_with_recent_driver(self):
        plan = get_pytorch_install_plan(self._env(
            has_nvidia_gpu=True,
            gpus=[GPUDevice(compute_capability="8.9", vendor=GPUVendor.NVIDIA, name="RTX 4090")],
            driver_max_cuda="12.8",
        ))
        assert plan["backend"] == "cuda"
        assert plan["cuda"] == "12.8"

    def test_legacy_gpu_capped_at_cu126(self):
        """Legacy GPU (GTX 1080 = 6.1) should be capped at cu126 regardless of driver."""
        plan = get_pytorch_install_plan(self._env(
            has_nvidia_gpu=True,
            gpus=[GPUDevice(compute_capability="6.1", vendor=GPUVendor.NVIDIA, name="GTX 1080")],
            driver_max_cuda="12.8",
        ))
        assert plan["backend"] == "cuda"
        assert plan["cuda"] == "12.6"

    def test_amd_linux_rocm(self):
        plan = get_pytorch_install_plan(self._env(
            has_amd_gpu=True,
            is_windows=False,
        ))
        assert plan["backend"] == "rocm"
        assert "rocm" in str(plan.get("official_cmd", ""))

    def test_amd_windows_no_rocm(self):
        plan = get_pytorch_install_plan(self._env(
            has_amd_gpu=True,
            is_windows=True,
        ))
        assert plan["backend"] == "cpu"


# ---------------------------------------------------------------------------
# get_python_wheel_tags
# ---------------------------------------------------------------------------

class TestGetPythonWheelTags:
    def test_returns_python_tag(self):
        tags = get_python_wheel_tags()
        ver = sys.version_info
        expected = f"cp{ver.major}{ver.minor}"
        assert tags["python"] == expected
        assert tags["abi"] == expected

    def test_returns_platform_tag(self):
        tags = get_python_wheel_tags()
        assert "platform" in tags
        assert len(tags["platform"]) > 0
